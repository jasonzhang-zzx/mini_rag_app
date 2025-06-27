from openai import OpenAI
import os
import torch
import requests

# # load environment variables from .env file
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv(), verbose=True) # read local .env file

# client = OpenAI()

def get_completion(prompt, model="deepseek-r1:latest"):
    """encapsulate the llm api"""
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False  # non-streaming output
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"请求失败: {response.text}")

prompt_template = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户的问题。

已知信息：
{context}

用户问：
{question}

如果已知信息不包含用户问题的答案，或已知信息不足以回答用户的问题，请直接回复“抱歉，我无法回答这个问题”。
请不要输出已知信息中不包含的信息或答案。
请用中文回答用户的问题。
"""


def build_prompt(prompt_template, **kwargs):
    """construct the prompt using the template and the parameters"""
    inputs = {}
    for k, v in kwargs.items():
        if isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = "\n\n".join(v) 
        else:
            val = v
        inputs[k] = val
    return prompt_template.format(**inputs)
