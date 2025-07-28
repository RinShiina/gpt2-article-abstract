from train_gpt2 import GPT2, GPT2Config
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import tiktoken

ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    assert torch.cuda.is_available(), "Distributed training requires CUDA to be available."
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

model  = GPT2.from_pretrained(None, checkpoint_path="./log/model_final.pt")
model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")

num_return = 5
max_length = 150
input_file_path = "article.txt"
max_context_tokens = 1024 # GPT-2 的典型上下文窗口大小

# --- 1. 读取并处理文章 ---
try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        article_text = f.read()
except FileNotFoundError:
    print(f"Error: Input file not found at '{input_file_path}'")
    exit()

# 将文章编码为 tokens
tokens = enc.encode(article_text)
# 截断以适应模型的上下文窗口
if len(tokens) > max_context_tokens:
    tokens = tokens[:max_context_tokens]

# 检查 max_length 是否大于输入长度
if max_length <= len(tokens):
    print(f"Warning: max_length ({max_length}) is not greater than the input text token count ({len(tokens)}).")
    print(f"Increasing max_length to {len(tokens) + max_length} to allow for generation.")
    max_length += len(tokens)

context_tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
context_len = context_tokens.size(1)

torch.manual_seed(114)
torch.cuda.manual_seed(114)

print(f"Starting generation with {context_len} context tokens, generating up to {max_length} total tokens.\n")

# --- 2. 批量生成并计时 ---
generated_sequences = []
total_start_time = time.time()

for i in range(num_return):
    x = context_tokens.clone() # 每次都从完整的文章上下文开始

    # 生成循环，直到达到目标总长度
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
        ix = torch.multinomial(topk_probs, num_samples=1)
        xcol = torch.gather(topk_indices, dim=-1, index=ix)
        x = torch.cat((x, xcol), dim=1)
    
    generated_sequences.append(x)

# 停止计时
total_end_time = time.time()

# --- 3. 报告计时结果 ---
total_time = total_end_time - total_start_time
avg_time = total_time / num_return
print(f"Total generation time for {num_return} samples: {total_time:.2f} seconds.")
print(f"Average time per sample: {avg_time:.2f} seconds.\n")

# --- 4. 解码并打印所有结果 ---
for i, seq in enumerate(generated_sequences):
    # 仅提取并解码新生成的部分
    newly_generated_tokens = seq[0, context_len:].tolist()
    text = enc.decode(newly_generated_tokens)
    print(f"--- Generated text {i+1} ---\n{text}\n")