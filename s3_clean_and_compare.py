"""
S3 wrap-up for Chat Scholar. Run from the folder that contains faiss_index/.

Does three things:
  1. Builds a clean index (junk figure chunks removed) -> faiss_index_clean/
  2. Compares top-3 scores for in-scope vs out-of-scope questions
  3. Counts MiniLM tokens per chunk to check the 256-token truncation limit
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256  # MiniLM silently drops everything past this

embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ---------- 1. Build the clean index ----------
old_store = FAISS.load_local(
    "faiss_index", embedding_model, allow_dangerous_deserialization=True
)
# docstore._dict is LangChain's internal chunk store: fine for inspection, not for production code
all_docs = list(old_store.docstore._dict.values())


def is_junk(doc):
    lines = doc.page_content.split("\n")
    short = sum(1 for line in lines if len(line.strip()) <= 3)
    return short / len(lines) > 0.5 or "<pad>" in doc.page_content


clean_docs = [d for d in all_docs if not is_junk(d)]
print(f"[1] Kept {len(clean_docs)} of {len(all_docs)} chunks")

clean_store = FAISS.from_documents(clean_docs, embedding_model)
clean_store.save_local("faiss_index_clean")
 