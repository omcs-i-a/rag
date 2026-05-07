from openai import OpenAI
import numpy as np
import faiss

from settings import OPENAI_API_KEY, EMBEDDING_MODEL_NAME, COMPLETION_MODEL_NAME, BASE_URL

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL
)

docs = [
    "RAGは検索拡張生成。",
    "Embeddingは意味ベクトル。",
    "FAISSはベクトル検索ライブラリ。"
]

embeddings = []

for doc in docs:
    r = client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=doc
    )
    embeddings.append(r.data[0].embedding)

vectors = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

query = "RAGとは？"

q = client.embeddings.create(
    model=EMBEDDING_MODEL_NAME,
    input=query
)

qv = np.array([q.data[0].embedding]).astype("float32")

D, I = index.search(qv, k=2)

print("index: ")
print(I)

retrieved_docs = [docs[i] for i in I[0]]

print("Retrieved documents:")
print(retrieved_docs)

context = "\n".join(retrieved_docs)
prompt = f"""
以下の情報を参考に回答してください。

[Context]
{context}

[Question]
RAGとは？
"""

response = client.chat.completions.create(
    model=COMPLETION_MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "与えられたContextのみを使って回答してください。"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response.choices[0].message.content)