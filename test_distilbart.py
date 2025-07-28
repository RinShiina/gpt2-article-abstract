import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import time

torch.use_deterministic_algorithms(False)

# --- 1. 环境设置 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "models")
os.environ["DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")
for d in [os.environ["HF_HOME"], os.environ["TRANSFORMERS_CACHE"], os.environ["DATASETS_CACHE"]]:
    os.makedirs(d, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- 2. 定义路径和模型 ---
base_model_name = "sshleifer/distilbart-cnn-6-6"
input_file_path = "article.txt"

# --- 3. 加载原始模型和 Tokenizer ---
print(f"Loading original pre-trained model and tokenizer: {base_model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model.to(device)
    model.eval()
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

# --- 4. 定义与 finetune 脚本一致的文章处理函数 ---
def process_article_like_finetune(article_text, tokenizer):
    """
    完全复现 finetune_distilbart_rouge.py 中的文章截断逻辑。
    首先按段落累加，直到 token 数接近 600；
    然后按句子累加，直到 token 数达到 1024 的上限。
    """
    paras = article_text.split("\n")
    sel, cnt = "", 0
    
    # 记录处理到哪个段落
    para_idx_to_process_sents = 0

    # 第一步：按段落累加，直到 token 数超过 600
    for i, para in enumerate(paras):
        ids = tokenizer(para, add_special_tokens=False)["input_ids"]
        if cnt + len(ids) <= 600:
            sel += para + " "
            cnt += len(ids)
        else:
            para_idx_to_process_sents = i
            break
    else: # 如果所有段落加起来都不足600，则从头开始处理句子
        para_idx_to_process_sents = 0

    # 第二步：从上一步结束的段落开始，按句子累加，直到 token 数达到 1024
    for para in paras[para_idx_to_process_sents:]:
        sents = para.split(". ")
        if not sents: continue

        for i, sent in enumerate(sents):
            # 确保句子以句号结尾，以便正确分割
            first = sent + (". " if not sent.endswith(".") and i < len(sents) - 1 else " ")
            ids = tokenizer(first, add_special_tokens=False)["input_ids"]
            if cnt + len(ids) <= 1024:
                sel += first
                cnt += len(ids)
            else:
                # 达到1024上限，停止处理
                return sel.strip()
    
    return sel.strip()

# --- 5. 读取并使用指定方法预处理输入文章 ---
try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        original_article_text = f.read()
except FileNotFoundError:
    print(f"Error: Input file not found at '{input_file_path}'")
    print("Please create the file and add the article text you want to summarize.")
    exit()

print("\nProcessing article with '600+400' logic...")
start_time = time.time()
processed_article_text = process_article_like_finetune(original_article_text, tokenizer)

# 使用 tokenizer 对处理后的文章进行编码
inputs = tokenizer(
    processed_article_text, 
    return_tensors="pt", 
    max_length=1024, 
    truncation=True # 加上以防万一
)
inputs = {k: v.to(device) for k, v in inputs.items()}

# --- 6. 生成摘要 ---
print("\nGenerating summary...")
torch.manual_seed(int(time.time()))
torch.cuda.manual_seed(int(time.time()))
with torch.no_grad():
    summary_ids = model.generate(
        inputs['input_ids'],
        #num_beams=4,
        max_length=200,
        #early_stopping=True
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.2
    )
end_time = time.time()
print(f"Article processed in {end_time - start_time:.2f} seconds.")


# --- 7. 解码并打印结果 ---
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# print("\n" + "="*20 + " ORIGINAL ARTICLE " + "="*20)
# print(original_article_text)
# print("\n" + "="*20 + " PROCESSED ARTICLE (Input to Model) " + "="*20)
# print(processed_article_text)
print("\n" + "="*20 + " GENERATED SUMMARY " + "="*20)
print(summary)