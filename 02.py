from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="deepseek-r1:1.5b")

# messages = [
#     SystemMessage("你是一名来自边塞的诗人"),
#     HumanMessage("请写一首关于唐诗"),
#     AIMessage("锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。"),
#     HumanMessage("给予你上一首的格式，再来一首"),
# ]

messages = [
    ("system", "你作为一名来自边塞的诗人"),
    ("human", "请写一首关于唐诗"),
    ("ai", "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。"),
    ("human", "给予你上一首的格式，再来一首")

]

res = model.stream(input=messages)

for chunk in res:
    print(chunk, end="", flush=True)