# from .chroma import deta_input_in_vector_db
import chromadb
from chromadb import Client
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
client = chromadb.PersistentClient(path="./chroma_db")

EMPLOYEE_COLLECTION = "transcriber_rag"
def retrieval_of_documents(collection,question):
    # collection=client.get_or_create_collection(name="transcriber_rag")
    result = collection.query(
                query_texts=[question],
                n_results=5,
                include=[
                    "documents",
                    "metadatas",
                    "distances"
                ]
            )
    print("RESULT OF RETRIEVAL",result)
    # retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return result

# def get_documents_by_key(key):
#     #here the documents will be list of chunks rather than original document 
#     collection = client.get_or_create_collection(
#         name="transcriber_rag"
#     )
#     results = collection.get(
#         include=["documents"]
#     )
#     chunks = results["documents"]
#     # print("key----DOCUMENT",chunks)
#     get_data_by_key=BM25Retriever.from_documents(chunks)
#     limited_content=get_data_by_key.k=2
#     print("key----DOCUMENT",limited_content)
#     get_data=get_data_by_key.get_relevant_documents(key)
#     print("KEY_DATA",get_data)
#     return get_data

def get_documents_by_key(key):
    collection = client.get_or_create_collection(
        name="transcriber_rag"
    )
    results = collection.get(
        include=["documents"]
    )
    chunks = results["documents"]
    # Convert strings into LangChain Documents
    documents = [
        Document(page_content=chunk)
        for chunk in chunks
    ]
    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(
        documents
    )
    bm25_retriever.k = 2
    # Retrieve documents
    get_data = bm25_retriever.invoke(key)
    print("\nKEY_DATA")
    for doc in get_data:
        print(doc.page_content[:300])
    print("\nKEY_DATA",get_data)
    return get_data