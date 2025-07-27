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
max_length = 40

tokens = enc.encode("Hello, I'm an article summary generator. So what is an article summary generator?")
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
tokens = tokens.repeat(num_return, 1)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:

    with torch.no_grad():
        logits, _ = model(x)

    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)

    topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)

    ix = torch.multinomial(topk_probs, num_samples=1)

    xcol = torch.gather(topk_indices, dim=-1, index=ix)

    x = torch.cat((x, xcol), dim=1)

for i in range(num_return):
    tokens = x[i, :max_length].tolist()
    text = enc.decode(tokens)
    print(f"Generated text {i+1}> {text}")