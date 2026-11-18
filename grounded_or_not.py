from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import llm

INDEX_DIR = "faiss_index_clean"
K = 4

# Same embedding setup as the ingestion script
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

GROUNDED_PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: "I cannot find this in the document."

Context:
{context}

Question: {question}
Answer:"""

questions = [
    "What is multi-head attention?",
    "What BLEU score did the base model get on English to German translation?",
]

print(f"model={llm.model_name()} index={INDEX_DIR} chunks={db.index.ntotal} k={K}")

for q in questions:
    print("=" * 70)
    print("QUESTION:", q)

    # Way 1: no pages at all. The AI can only use its memory.
    print("\n--- WITHOUT pages (memory only) ---")
    print(llm.generate(q))

    # Way 2: retrieved pages plus the "answer only from these" rule.
    results = db.similarity_search_with_score(q, k=K)
    context = "\n\n".join(
        f"[page {doc.metadata.get('page', 0) + 1}] {doc.page_content}"
        for doc, _ in results
    )
    print("\n--- WITH pages (grounded) ---")
    print("pages used:", [doc.metadata.get("page", 0) + 1 for doc, _ in results])
    print(llm.generate(GROUNDED_PROMPT.format(context=context, question=q)))