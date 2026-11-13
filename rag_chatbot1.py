from pydoc import doc, text
from langchain.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import chatgpt_oauth
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

#load the pdf 
pdf_path='/Users/divyanshuthakur/Downloads/RAGPaper.pdf'
loader = PyPDFLoader(pdf_path)
documents=loader.load()

#create the splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)


#split the loaded documents 
chunks = text_splitter.split_documents(documents)

#convert PDF chunks into vector embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load a free embedding model locally
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vectors = embedding_model.embed_documents([chunk.page_content for chunk in chunks])


from langchain_community.vectorstores import FAISS

text_embeddings = [(chunk.page_content,vector) for chunk,vector in zip(chunks,vectors)]
vector_store = FAISS.from_embeddings(
    metadatas=[chunk.metadata for chunk in chunks],
    embedding=embedding_model,
    text_embeddings=text_embeddings
)

vector_store.save_local("faiss_index")
print("Total vectors stored:", vector_store.index.ntotal)

#retiever
retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})


from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


system_prompt = """
You are a helpful assistant answering questions from a research paper . 

Answer using ONLY the context given . If the answer is not available in the context
say: "I dont know based on the provided document"

After generating the answer, identify the document IDs that directly support your answer.

Return your response in this format:

Answer: <your answer>

Supporting document IDs: [list of IDs that support the answer]

Context: 
{context}
"""

#creating prompt with chathistory 
chat_history=[]
prompt = ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{question}")
])

while True:
        
    question = input("\n User: ")
    if(question.lower()=="exit"): 
        print("Chatbot: Goodbye!!")
        break

    retrieved_docs=retriever.invoke(question)

    #assign unqiue id to each retieved doc
    for i,doc in enumerate(retrieved_docs):
        doc.metadata["doc_id"]=i

    source_pages = sorted({doc.metadata["page"] + 1 
                           for doc in retrieved_docs 
                           if "page" in doc.metadata})

    #now LLM comes into picture 
    from langchain_groq import ChatGroq

    #enrich content with page number also to give to LLm(will be useful later) 
    context="\n\n".join(f"Document id: { doc.metadata.get('doc_id')+1} \n\n"
                        f"Context: {doc.page_content}"
                        for doc in retrieved_docs
                        )
    formatted_prompt = prompt.invoke({"context":context,
                                      "question":question,
                                      "chat_history":chat_history
                                      })

    # Initialize Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.3
    )

    response=llm.invoke(formatted_prompt)
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))
    print("\n Chatbot:",response.content)
    print("\n Retrieved Source Pages: ", source_pages)

