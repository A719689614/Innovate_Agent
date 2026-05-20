from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = OllamaLLM(model="deepseek-r1:1.5b")

prompt_1 = PromptTemplate.from_template(
    '我邻居姓：{lastname}，刚生了{gender}，帮孩子取个名字，简洁回答'
)

prompt_2 = PromptTemplate.from_template(
    '帮我{result}解释含义'
)

def show_any(result):
    print("临时打印结果",result)
    return result

chain = prompt_1 | model | StrOutputParser() | show_any | prompt_2 | model

res = chain.stream(input={"lastname": "林", "gender": "女孩"})
for chunk in res:
    print(chunk, end="", flush=True)