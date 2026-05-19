from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "给出每个单词的反义词."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

model = OllamaLLM(model="deepseek-r1:1.5b")

history = [
    HumanMessage(content="大"),
    AIMessage(content="小"),
    HumanMessage(content="前"),
    AIMessage(content="后"),
]

chain = prompt | model

for chunk in chain.stream(input = {"question": "左", "history": history}):
    print(chunk, end="", flush=True)

