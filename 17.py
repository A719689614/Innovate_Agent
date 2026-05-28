from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

res = vector_store.similarity_search(input, k=2)

reference = [doc.page_content for doc in res]
def print_content(prompt):
    print(prompt.to_string())
    return prompt

chain = prompt | print_content | model | StrOutputParser()

if __name__ == "__main__":
    res = chain.stream({"input": input, "context": reference})
    for chunk in res:
        print(chunk, end="", flush=True)