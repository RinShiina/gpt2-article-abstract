import os
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
import tiktoken
import shutil
from train_gpt2 import GPT2, GPT2Config

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
if master:
    print("Loading model and tokenizer...")
# 使用 tiktoken 作为分词器
tokenizer = tiktoken.get_encoding("gpt2")
# 从头开始初始化或从预训练的 gpt2-medium 加载
model = GPT2(GPT2Config(vocab_size=tokenizer.n_vocab))
model.to(device)
if ddp:
    model = DDP(model, device_ids=[local_rank])
raw_model = model.module if ddp else model

# -------------------- 数据集预处理 --------------------
if master:
    print("Loading and preprocessing dataset...")
shutil.rmtree(os.path.join(os.environ["DATASETS_CACHE"], "datasets--scientific_papers"), ignore_errors=True)
dataset = load_dataset("scientific_papers", "pubmed", split="train[:10%]")

# GPT-2 的最大序列长度
MAX_LEN = 1024

def preprocess_function(examples):
    eot_token = tokenizer.eot_token_id
    processed = {"input_ids": [], "attention_mask": []}
    
    for article, abstract in zip(examples["article"], examples["abstract"]):
        # 编码文章和摘要
        article_tokens = tokenizer.encode(article)
        abstract_tokens = tokenizer.encode(abstract)
        
        # 截断文章以确保总长度不超过 MAX_LEN - 2 (为 EOT token 留出空间)
        max_article_len = MAX_LEN - len(abstract_tokens) - 2
        if len(article_tokens) > max_article_len:
            article_tokens = article_tokens[:max_article_len]
            
        # 构造输入序列: [文章] <eot> [摘要] <eot>
        input_ids = article_tokens + [eot_token] + abstract_tokens + [eot_token]
        
        # 构造目标序列 (与输入相同，但在计算损失时会进行移位和掩码)
        # 同时创建 attention_mask
        padding_len = MAX_LEN - len(input_ids)
        
        # 创建一个 target，其中文章部分被-100替换，不计算损失
        targets = ([-100] * (len(article_tokens) + 1)) + abstract_tokens + [eot_token]
        
        # 对 input_ids, targets 和 attention_mask 进行填充
        input_ids += [tokenizer.pad_token_id] * padding_len
        targets += [-100] * padding_len
        attention_mask = [1] * (len(input_ids) - padding_len) + [0] * padding_len
        
        processed["input_ids"].append(input_ids)
        # 将 targets 重命名为 labels 以匹配常用约定
        processed["labels"] = targets
        processed["attention_mask"].append(attention_mask)

    return processed

dataset = dataset.map(preprocess_function, batched=True, num_proc=8, remove_columns=dataset.column_names)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# -------------------- 训练设置 --------------------
total_batch_size = 256
per_device_bs = 8 # GPT2 模型较大，减小批次大小
grad_accum_steps = max(1, total_batch_size // (per_device_bs * world_size))
num_epochs = 3
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
    return min_lr + 0.5 * (1 + math.cos(math.pi * decay)) * (max_lr - min_lr)

sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
train_loader = DataLoader(dataset, batch_size=per_device_bs, sampler=sampler, num_workers=8, pin_memory=True)

optimizer = torch.optim.AdamW(raw_model.parameters(), lr=max_lr, betas=(0.9, 0.95), eps=1e-8)
global_step = 0
current_lr = max_lr

log_dir = "/root/autodl-tmp/log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log_gpt2.txt")
if master:
    open(log_file, "w").close()

# -------------------- 训练循环 --------------------
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    raw_model.train()
    for step, batch in enumerate(train_loader):
        t0 = time.time()
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 前向传播
        logits, loss = model(batch["input_ids"], batch["labels"])
        
        # 缩放损失以进行梯度累积
        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or step == len(train_loader) - 1:
            if ddp:
                all_reduce(loss, op=ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)

            current_lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
            
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        if device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        processed = per_device_bs * world_size
        tps = processed / dt if dt > 0 else float('inf')

        if master and (step % 10 == 0 or step == len(train_loader) - 1):
            msg = (f"E{epoch+1}/{num_epochs} S{step+1}/{steps_per_epoch} | "
                   f"loss {loss.item() * grad_accum_steps:.4f} | lr {current_lr:.4e} | "
                   f"t {dt*1000:.1f}ms | samples/s {tps:.1f}")
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")

    if master:
        ckpt = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': raw_model.config,
            'epoch': epoch,
            'global_step': global_step
        }
        fn = os.path.join(log_dir, f"model_epoch{epoch+1:02d}_step{global_step:05d}.pt")
        torch.save(ckpt, fn)
        print(f"Checkpoint saved: {fn}")

# -------------------- 最终模型 --------------------
if master:
    final_path = os.path.join(log_dir, "model_final.pt")
    final_ckpt = {
        'model': raw_model.state_dict(),
        'config': raw_model.config,
    }
    torch.save(final_ckpt, final_path)
    print(f"Final model saved: {final_path}")

if ddp:
    destroy_process_group()
if master:
    print("Training complete!")