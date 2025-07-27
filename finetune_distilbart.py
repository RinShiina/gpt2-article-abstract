import os
import time
import math
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import shutil

# -------------------- 环境与 DDP 初始化 --------------------
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "models")
os.environ["DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")
for d in [os.environ["HF_HOME"], os.environ["TRANSFORMERS_CACHE"], os.environ["DATASETS_CACHE"]]:
    os.makedirs(d, exist_ok=True)

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    master = (rank == 0)
else:
    rank = local_rank = 0
    world_size = 1
    master = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = "cuda" if device.startswith("cuda") else "cpu"
torch.set_float32_matmul_precision('high')
torch.manual_seed(1337)
if device_type == "cuda":
    torch.cuda.manual_seed(1337)

# -------------------- 模型和 Tokenizer --------------------
model_name = "sshleifer/distilbart-cnn-6-6"
if master:
    print("Loading model and tokenizer...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)
if ddp:
    model = DDP(model, device_ids=[local_rank])
raw_model = model.module if ddp else model

# -------------------- 数据集预处理 --------------------
if master:
    print("Loading and preprocessing dataset...")
shutil.rmtree(os.path.join(os.environ["DATASETS_CACHE"], "datasets--scientific_papers"), ignore_errors=True)
dataset = load_dataset("scientific_papers", "pubmed", split="train[:10%]")

def preprocess_function(examples):
    inputs, targets = examples["article"], examples["abstract"]
    # 这里复用原逻辑截断文章
    processed_inputs = []
    for article in inputs:
        paras = article.split("\n")
        sel, cnt = "", 0
        for para in paras:
            ids = tokenizer(para, add_special_tokens=False)["input_ids"]
            if cnt + len(ids) <= 600:
                sel += para + " "
                cnt += len(ids)
            else:
                break
        for para in paras[len(sel.split("\n"))-1:]:
            sents = para.split(". ")
            if sents and cnt < 1024:
                first = sents[0] + (". " if not sents[0].endswith(".") else " ")
                ids = tokenizer(first, add_special_tokens=False)["input_ids"]
                if cnt + len(ids) <= 1024:
                    sel += first
                    cnt += len(ids)
                else:
                    break
        processed_inputs.append(sel.strip())
    mi = tokenizer(processed_inputs, max_length=1024, truncation=True, padding="max_length")
    lb = tokenizer(targets, max_length=200, truncation=True, padding="max_length")
    mi["labels"] = lb["input_ids"]
    return mi

dataset = dataset.map(preprocess_function, batched=True, num_proc=8)
dataset.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

# -------------------- 训练设置 --------------------
total_batch_size = 256
per_device_bs = 16
grad_accum_steps = max(1, total_batch_size // (per_device_bs * world_size))
num_epochs = 3
# 计算每个 epoch 的 step 数 (向上取整)
steps_per_epoch = math.ceil(len(dataset) / (per_device_bs * world_size))
update_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
total_update_steps = update_steps_per_epoch * num_epochs
total_steps = steps_per_epoch * num_epochs
warmup_steps = 10
max_lr = 3e-5
min_lr = max_lr * 0.1

def get_lr(gs):
    if gs < warmup_steps:
        return max_lr * ((gs + 1) / warmup_steps)
    if gs > total_update_steps:
        return min_lr
    decay = (gs - warmup_steps) / (total_update_steps - warmup_steps)
    return min_lr + 0.5*(1+math.cos(math.pi*decay))*(max_lr-min_lr)

# DataLoader + DistributedSampler
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
train_loader = DataLoader(dataset, batch_size=per_device_bs, sampler=sampler, num_workers=8)

# 优化器 & 全局步数
optimizer = torch.optim.AdamW(raw_model.parameters(), lr=max_lr, betas=(0.9,0.95), eps=1e-8)
global_step = 0
current_lr = max_lr
rouge = evaluate.load("rouge")

# 日志文件
log_dir = "/root/autodl-tmp/log"; os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
if master:
    open(log_file, "w").close()

# -------------------- 训练循环 --------------------
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    raw_model.train()
    for step, batch in enumerate(train_loader):
        # lr = get_lr(global_step)
        t0 = time.time()
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            outputs = raw_model(**batch)
            loss = outputs.loss / grad_accum_steps
        loss.backward()

        if (step+1) % grad_accum_steps == 0 or step == len(train_loader)-1:
            if ddp:
                all_reduce(loss, op=ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)

            # for pg in optimizer.param_groups:
            #     pg['lr'] = lr

            current_lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
            
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # 同步并计时
        if device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        processed = per_device_bs * grad_accum_steps * world_size
        tps = processed / dt

        if master and (step % 10 == 0 or step == len(train_loader)-1):
            preds = outputs.logits.argmax(dim=-1)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            rouge_res = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            r1 = rouge_res["rouge1"]
            r2 = rouge_res["rouge2"]
            rL = rouge_res["rougeL"]
            msg = (f"E{epoch+1}/{num_epochs} S{step+1}/{steps_per_epoch} | "
                   f"loss {loss.item():.6f} | lr {current_lr:.4e} | "
                   f"t {dt*1000:.1f}ms | samples/s {tps:.1f} | "
                   f"rouge1 {r1:.4f} rouge2 {r2:.4f} rougeL {rL:.4f}")
            #msg = f"E{epoch+1}/{num_epochs} S{step+1}/{steps_per_epoch} | loss {loss.item():.6f} | lr {lr:.4e} | t {dt*1000:.1f}ms | samples/s {tps:.1f}"
            #msg = f"E{epoch+1}/{num_epochs} S{step+1}/{steps_per_epoch} | loss {loss.item():.6f} | lr {current_lr:.4e} | t {dt*1000:.1f}ms | samples/s {tps:.1f}"
            print(msg)
            with open(log_file,"a") as f:
                f.write(msg+"\n")

    # epoch 结束保存 checkpoint
    if master:
        ckpt = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'epoch': epoch,
            'global_step': global_step
        }
        fn = os.path.join(log_dir, f"model_epoch{epoch+1:02d}_step{global_step:05d}.pt")
        torch.save(ckpt, fn)
        print(f"Checkpoint saved: {fn}")

# -------------------- 最终模型 --------------------
if master:
    final_path = os.path.join(log_dir, "model_final.pt")
    torch.save(raw_model.state_dict(), final_path)
    tokenizer.save_pretrained("./distilbart_pubmed_finetuned")
    print(f"Final model saved: {final_path}")

# 结束 DDP
if ddp:
    destroy_process_group()
if master:
    print("Training complete!")