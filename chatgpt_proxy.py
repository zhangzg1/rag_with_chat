import requests
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('chatgpt_api_key')


class ChatGPTProxy():
    def __init__(self, model="gpt-3.5-turbo", temperature=0.1):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def get_response(self, prompt):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        try:
            response = requests.post(url, json=payload, headers=headers)
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"请求失败: {str(e)}"

    def infer(self, prompts):
        # 使用线程池并行处理
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.get_response, prompts))
        return results


if __name__ == "__main__":
    prompts = [
        "你好",
        "你会干什么",
        "推荐5本人工智能入门书籍"
    ]
    llm = ChatGPTProxy()
    results = llm.infer(prompts)
    print(results)
