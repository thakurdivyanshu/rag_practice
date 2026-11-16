from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vector_store = FAISS.load_local(
    "faiss_index_clean",
    embedding_model,
    allow_dangerous_deserialization=True
)

query = "What is multi headed attention?"
print(repr(query))
results = vector_store.similarity_search_with_score(query,k=3)

# for doc,score in result:
#     print(f"Score :{score:.4f} | Page: {doc.metadata.get('page')}")
#     print(doc.page_content[:200])
#     print("-"*50)

import numpy as np

query_vec = np.array(embedding_model.embed_query(query))
print("Query vector length:", np.linalg.norm(query_vec))

for doc, score in results:
    doc_vec = np.array(embedding_model.embed_documents([doc.page_content])[0])
    cosine = float(np.dot(query_vec, doc_vec))
    print(f"FAISS score: {score:.4f} | cosine: {cosine:.4f} | 2 - 2*cosine: {2 - 2*cosine:.4f}")



for k in [1, 3, 5, 10]:
    results = vector_store.similarity_search_with_score(query, k=k)
    scores = [round(float(score), 4) for _, score in results]
    pages = [doc.metadata.get("page") for doc, _ in results]
    print(f"k={k} -> scores: {scores}")
    print(f"      pages: {pages}")


results = vector_store.similarity_search_with_score(query, k=10)

# for rank in [4, 7, 10]:
#     doc, score = results[rank - 1]
#     print(f"Rank {rank} | score {score:.4f} | page {doc.metadata.get('page')} | chars {len(doc.page_content)}")
#     print(doc.page_content)
#     print("=" * 60)    

docs = list(vector_store.docstore._dict.values())

def short_line_ratio(text):
    lines = text.split("\n")
    short = sum(1 for line in lines if len(line.strip()) <= 3)
    return short / len(lines)

flagged = [d for d in docs if short_line_ratio(d.page_content) > 0.5 or "<pad>" in d.page_content]

print("Total chunks:", len(docs))
print("Flagged as junk:", len(flagged))
print("Pages of flagged chunks:", sorted({d.metadata.get("page") for d in flagged}))