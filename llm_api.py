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
You are a question-answering chatbot.
Your task is to answer the user's question based on the following provided information.

Known information:
{context}

User's question:
{question}

If the known information does not contain the answer to the user's question, or if the information is insufficient to answer it, please reply directly with "Sorry, I cannot answer this question."
Do not provide any information or answers that are not included in the known information.
Please answer the user's question in English.
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
