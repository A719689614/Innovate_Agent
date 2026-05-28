from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, ChatMessagePromptTemplate, PromptTemplate

model = OllamaLLM(model="deepseek-r1:1.5b")

# prompt = ChatPromptTemplate.from_messages(
#   [("system", "你是一个诗词高手，很会帮用户写诗"),
#   MessagesPlaceholder(variable_name="hsitory"),
#   ("human", "{user_input}")]
# )
# history = [
#   ("human", "请写一首唐诗"),
#   ("ai", "白日依山尽，黄河入海流。欲穷千里目， 更上一层楼。"),
# ]

# chain = prompt | model

# for chunk in chain.stream(input={"user_input": "参考上面的格式，写一首内容含北京的唐诗", "hsitory": history}):
#   print(chunk, end="", flush=True)

prompt = PromptTemplate.from_template("请写一首关于{topic}的唐诗")

chain = prompt | model

for chunk in chain.stream(input={"topic": "北京"}):
  print(chunk, end="", flush=True)