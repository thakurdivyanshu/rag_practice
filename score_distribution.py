import json
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_DIR = "faiss_index_clean"
K = 4

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# Labels are my guesses about the Transformer paper. Check each IN against the PDF.
questions = [
    # IN: three phrasings of the same idea (tests wording sensitivity)
    ("IN", "What is multi-head attention?"),
    ("IN", "Explain how multi-head attention works"),
    ("IN", "why does the transformer use several attention heads"),
    # IN: other topics
    ("IN", "How does positional encoding work?"),
    ("IN", "Why is the dot product scaled by the square root of d_k?"),
    ("IN", "How many layers does the encoder have?"),
    ("IN", "What optimizer was used for training?"),
    ("IN", "What BLEU score did the model get on English to German translation?"),
    # OUT: plainly unrelated
    ("OUT", "What is the capital of France?"),
    ("OUT", "How do I bake sourdough bread?"),
    ("OUT", "Who won the football world cup in 2018?"),
    ("OUT", "What is the weather like in Delhi today?"),
    # TRAP: near the topic, but the paper does not answer it
    ("TRAP", "How many parameters does GPT-4 have?"),
    ("TRAP", "How does BERT pretraining work?"),
    ("TRAP", "What is ChatGPT?"),
    ("TRAP", "How do I fine-tune a transformer on my own data?"),
]

rows = []
for label, q in questions:
    results = db.similarity_search_with_score(q, k=K)
    top1 = float(results[0][1])
    page = results[0][0].metadata.get("page", -1) + 1  # E7: page + 1
    rows.append((label, q, top1, page))

print(f"{'label':<5} {'top1':>5} {'cos':>5} {'page':>4}  question")
for label, q, top1, page in rows:
    print(f"{label:<5} {top1:>5.2f} {1 - top1 / 2:>5.2f} {page:>4}  {q}")

print()
for label in ["IN", "TRAP", "OUT"]:
    s = [r[2] for r in rows if r[0] == label]
    print(f"{label:<5} min={min(s):.2f}  max={max(s):.2f}  n={len(s)}")

# E8: log what is needed to reproduce this run
with open("score_runs.jsonl", "a") as f:
    f.write(json.dumps({
        "time": datetime.now().isoformat(),
        "index": INDEX_DIR,
        "chunks": db.index.ntotal,
        "k": K,
        "rows": [{"label": l, "query": q, "top1": s, "page": p} for l, q, s, p in rows],
    }) + "\n")