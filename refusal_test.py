import json
from datetime import datetime

from langchain_community.vectorstores import FAISS

# ASSUMPTION: copy the embeddings import and setup EXACTLY from your search_test.py.
# It must be the same model with normalize_embeddings=True, or the scores will not match.
from langchain_huggingface import HuggingFaceEmbeddings

import llm

INDEX_DIR = "faiss_index_clean"
K = 4
REFUSE_ABOVE = 1.5  # provisional line (option A). Re-check with the eval set in S4b.
REFUSAL = "I cannot find this in the document."

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

PROMPT = """You are a research assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: "I cannot find this in the document."

Context:
{context}

Question: {question}
Answer:"""

questions = [
    ("IN  ", "What is multi-head attention?"),
    ("IN  ", "What optimizer was used for training?"),  # worst real question so far (1.17)
    ("OUT ", "What is the capital of France?"),
    ("OUT ", "How do I bake sourdough bread?"),
    ("TRAP", "How many parameters does GPT-4 have?"),
    ("TRAP", "What is ChatGPT?"),
]

for label, q in questions:
    results = db.similarity_search_with_score(q, k=K)  # (Document, squared L2 score)

    context = "\n\n".join(
        f"[page {doc.metadata.get('page', '?') + 1}] {doc.page_content}"
        for doc, _ in results
    )
    top1 = float(results[0][1])
    if top1 > REFUSE_ABOVE:  # the guard: best chunk is too far, so skip the LLM
        answer = REFUSAL
        llm_called = False
    else:
        answer = llm.generate(PROMPT.format(context=context, question=q))
        llm_called = True

    print("=" * 70)
    print(f"{label} | {q}")
    print("scores:", [round(float(s), 2) for _, s in results])
    print("LLM called:", llm_called)
    print("ANSWER:", answer)

    # Rule E8: log everything needed to reproduce this run
    with open("rag_runs.jsonl", "a") as f:
        f.write(json.dumps({
            "time": datetime.now().isoformat(),
            "model": llm.model_name(),
            "index": INDEX_DIR,
            "chunks": db.index.ntotal,
            "k": K,
            "refuse_above": REFUSE_ABOVE,
            "llm_called": llm_called,
            "query": q,
            "scores": [float(s) for _, s in results],
            "answer": answer,
        }) + "\n")
