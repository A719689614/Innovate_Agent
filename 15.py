from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import CSVLoader

vector_store = InMemoryVectorStore(embedding=OllamaEmbeddings(model="qwen3-embedding:0.6b"))


loader = CSVLoader(
    file_path="./data/info.csv",
    encoding="utf-8",
    source_column="source",
)
documents = loader.load()

vector_store.add_documents(documents=[doc for doc in documents], ids=["id"+str(i) for i in range(1, len(documents) + 1)])

vector_store.delete(ids=["id1", "id2"])

result = vector_store.similarity_search(
    query="Python是不是简单易学呀?",
    k=3,
)

print(result)