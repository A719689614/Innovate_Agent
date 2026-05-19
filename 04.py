from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

model = OllamaLLM(model="deepseek-r1:1.5b")

prompt_template = PromptTemplate.from_template("我的邻居姓{last_name},刚生了{gender}孩,帮我起名字，请简略回答")
print(prompt_template, type(prompt_template))
# prompt_text = prompt_template.format(last_name="王", gender="女")

# res = model.stream(prompt_text)
# for chunk in res:
#     print(chunk, end="", flush=True)


"""
    基于链的写法
"""

chain = prompt_template | model

res = chain.stream(input={"last_name": "王", "gender": "女"})
for chunk in res:
    print(chunk, end="", flush=True)