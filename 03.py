from langchain_ollama.embeddings import OllamaEmbeddings

model = OllamaEmbeddings(model="qwen3-embedding:0.6b")


res1 = model.embed_query("What is langchain?")
res2 = model.embed_documents(["What is langchain?", "What is langchain?"])
# print(res1)
print(res2)