from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

model = OllamaLLM(model="deepseek-r1:1.5b")
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题，参考资料{context}"),
        ("human", "用户提问: {input}")
    ]
)


vector_store = InMemoryVectorStore(embedding=embeddings)

vector_store.add_texts(texts=["减肥就是要少吃多练", "在减脂期间吃东西很重要，清淡少油控制卡路里摄入并运动起来", "跑步是很好的运动哦"])

input = "怎么减肥？"

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def format_func(docs) -> list[Document]:
    if not docs:
        return "无参考资料"
    formatted_str = "["
    for doc in docs:
        formatted_str += doc.page_content
    formatted_str += "]"
    return formatted_str

def prompt_content(prompt):
    print(prompt.to_string())
    return prompt


chain = (
    {"input": RunnablePassthrough(), "context": retriever | format_func} | prompt | prompt_content | model | StrOutputParser()
)

result = chain.stream(input = input)
for chunk in result:
    print(chunk, end="", flush=True)