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

# ---------- 2. In-scope vs out-of-scope scores ----------
questions = [
    ("IN ", "What is multi-head attention?"),
    ("IN ", "How does positional encoding work?"),
    ("OUT", "What is the capital of France?"),
    ("OUT", "How do I bake sourdough bread?"),
]

print("\n[2] Top-3 scores per question (lower = more similar)")
for label, q in questions:
    results = clean_store.similarity_search_with_score(q, k=3)
    scores = [round(float(s), 4) for _, s in results]
    pages = [d.metadata.get("page") for d, _ in results]
    print(f"{label} | {q}")
    print(f"      scores: {scores} | pages: {pages}")

# ---------- 3. Token counts per chunk ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL)
counts = [(len(tokenizer.encode(d.page_content)), d) for d in clean_docs]
over = [(n, d) for n, d in counts if n > MAX_TOKENS]

print(f"\n[3] Chunks over {MAX_TOKENS} tokens: {len(over)} of {len(counts)}")
print("    Longest 5 chunks (tokens | chars | page):")
for n, d in sorted(counts, key=lambda x: x[0], reverse=True)[:5]:
    print(f"    {n} | {len(d.page_content)} | {d.metadata.get('page')}")
print("    Median tokens:", sorted(n for n, _ in counts)[len(counts) // 2])



import numpy as np

old_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
new_store = FAISS.load_local("faiss_index_clean", embedding_model, allow_dangerous_deserialization=True)
q = "What is multi headed attention?"

for name, store in [("old", old_store), ("new", new_store)]:
    doc, score = store.similarity_search_with_score(q, k=1)[0]
    pos = next(i for i, id_ in store.index_to_docstore_id.items()
               if store.docstore._dict[id_].page_content == doc.page_content)
    stored = store.index.reconstruct(pos)
    fresh = np.array(embedding_model.embed_documents([doc.page_content])[0])
    print(name, "| score", round(float(score), 4), "| page", doc.metadata.get("page"),
          "| cos(stored, fresh):", round(float(np.dot(stored, fresh)), 6),
          "| stored norm:", round(float(np.linalg.norm(stored)), 4))
    print("   ", doc.page_content[:80].replace("\n", " "))



import os, time
for f in ["faiss_index/index.faiss", "faiss_index/index.pkl", "faiss_index_clean/index.faiss"]:
    print(f, time.ctime(os.path.getmtime(f)))