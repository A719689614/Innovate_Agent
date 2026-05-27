from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

model = OllamaLLM(model="deepseek-r1:1.5b")

# 聊天历史模板 =======================================================================
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "你是一个诗人。"),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "按照格式, 写关于{topic}的一首诗?"),
#     ])

# history = [
#     ("human", "写关于春天的一首诗?"),
#     ("assistant", "春天来了，花儿开了，鸟儿唱了，大地复苏了，万物生长了。"),
#     ("human", "写关于夏天的一首诗?"),
#     ("assistant", "夏天来了，天气晴朗，风和日丽，人来人往，香风吹着。"),
# ]

# chain = prompt | model

# for chunk in chain.stream(input={"history": history, "topic": "明天"}):
#     print(chunk, end="", flush=True)
