from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个著名的诗人"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

history = [
    ("human", "你来写一首唐诗"),
    ("ai", "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。"),
    ("human", "请继续"),
    ("ai", "床前明月光，疑似地上霜。举头望明月，低头思故乡。"),
]

model = OllamaLLM(model="deepseek-r1:1.5b")

chain = chat_prompt_template | model

for chunk in chain.stream(input={"history": history, "input": "请继续"}):
    print(chunk, end="", flush=True)