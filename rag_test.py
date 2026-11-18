import huggingface_hub
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

#load embedding model 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

#load exsisting faiss index
vector_store=FAISS.load_local(
    'faiss_index_clean',
    embeddings,
    allow_dangerous_deserialization=True
)

def retrieve_chunks(question,k=4):
    return vector_store.similarity_search_with_score(question,k=k)



question = 'What is multiheaded attention?'
results=retrieve_chunks(question)

REFUSAL_THRESHOLD=1.5
def should_refuse(results):
    top_score=results[0][1]
    if(top_score> REFUSAL_THRESHOLD):
        return True
    return False

print("Should question be refused?")

if should_refuse(results):
    print("Yes")
else:
    print("No")

for i,(doc,score) in enumerate(results):
    print(f"\nChunk {i+1}")
    print(f"Score {score:.4f}")
    print(f"Page {doc.metadata.get('page',-1)+1}")
    print(f"Content {doc.page_content[:300]}")

