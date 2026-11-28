from pydoc import text

from langchain_community.document_loaders import PyPDFLoader

pdf_path='attention_is_all_you_need.pdf'

loader = PyPDFLoader(pdf_path)
documents=loader.load()

# create text spliiter
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

chunks=text_splitter.split_documents(documents)

def is_junk(chunk):
    lines=chunk.page_content.split("\n")
    short = sum(1 for line in lines if len(line.strip()) <=3)
    return short/len(lines) > 0.5 or "<pad>" in chunk.page_content

cleaned_chunks=[c for c in chunks if(is_junk(c) == False)]
print(f"Kept {len(cleaned_chunks)} of {len(chunks)} chunks")

from langchain_huggingface import HuggingFaceEmbeddings
# Load a free embedding model locally
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectors=embedding_model.embed_documents([chunk.page_content for chunk in cleaned_chunks])

from langchain_community.vectorstores import FAISS

text_embeddings=[(chunk.page_content,vector) for chunk,vector in zip(cleaned_chunks,vectors)]

vector_store=FAISS.from_embeddings(
    metadatas=[chunk.metadata for chunk in cleaned_chunks],
    embedding=embedding_model,
    text_embeddings=text_embeddings
)

vector_store.save_local("faiss_index_clean")


question=input("ask a question \n")
results = vector_store.similarity_search(question,k=3)

for i,doc in enumerate(results):
    print(f"result {i+1}")
    print(f"content {doc.page_content[:500]}")
    print(f"metadata {doc.metadata}")
    print("*************************** \n")


