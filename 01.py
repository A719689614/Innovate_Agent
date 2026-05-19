# 大模型的调用和传输方式 ==================================
from langchain_ollama import OllamaLLM
model = OllamaLLM(model="deepseek-r1:1.5b")

# res = model.invoke("你好呀")
# print(res)

res = model.stream("你好呀")

for chunk in res:
    print(chunk, end="", flush=True)