import os
from openai import OpenAI

def generate_summary_with_deepseek(article_content: str, api_key: str) -> str:
    """
    调用 DeepSeek API 为给定文章生成摘要。

    Args:
        article_content: 从文件中读取的文章内容。
        api_key: 你的 DeepSeek API 密钥。

    Returns:
        API 返回的摘要文本，如果出错则返回错误信息。
    """
    try:
        # 初始化客户端，关键在于设置 base_url 指向 DeepSeek 的 API 地址
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

        # 组合文章内容和提示语
        prompt_instruction = "请根据以上文章内容，生成一份简洁、流畅、专业的英文摘要总结。"
        full_prompt = f"{article_content}\n\n{prompt_instruction}"

        # 调用 API
        response = client.chat.completions.create(
            model="deepseek-chat",  # 使用 DeepSeek V2 的聊天模型
            messages=[
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=500,       # 您可以根据需要调整摘要的最大长度
            temperature=0.7,      # 较低的温度使输出更具确定性和事实性
            stream=False          # 设置为 False 以接收完整响应
        )

        # 提取并返回结果
        summary = response.choices[0].message.content
        return summary.strip()

    except Exception as e:
        return f"调用 API 时发生错误: {e}"

if __name__ == "__main__":
    # 1. 从环境变量中获取 API 密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：未找到 DEEPSEEK_API_KEY 环境变量。")
        print("请先设置环境变量: export DEEPSEEK_API_KEY='your_api_key'")
        exit()

    # 2. 读取 article.txt 文件
    input_file = "article.txt"
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            article = f.read()
        print(f"成功读取文件 '{input_file}'。")
    except FileNotFoundError:
        print(f"错误：输入文件 '{input_file}' 未找到。")
        print("请在当前目录下创建该文件并填入文章内容。")
        exit()

    # 3. 调用函数生成摘要
    print("\n正在向 DeepSeek API 发送请求并生成摘要...")
    summary_result = generate_summary_with_deepseek(article, api_key)

    # 4. 打印结果
    print("\n" + "="*20 + " 生成的摘要 " + "="*20)
    print(summary_result)
    print("="*55)
