from langchain_chroma import Chroma
from .retrieval import retrieval_of_documents,get_documents_by_key
import chromadb
from chromadb import Client
import uuid
import os
from openai import OpenAI

client = chromadb.PersistentClient(path="./chroma_db")

EMPLOYEE_COLLECTION = "transcriber_rag"
# vector_store = Chroma(
#     collection_name="foo",
#     embedding_function=OpenAIEmbeddings(),
#     # other params...
# )
def create_db_collection(chunks,embeddings):
    collection = client.get_or_create_collection(name="transcriber_rag")
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    documents = [
        chunk.page_content
        for chunk in chunks
    ]
    vectors = embeddings.embed_documents(
        documents
    )
    collection.add(ids=ids,embeddings=vectors,documents=documents)
    return "Data Added To Database"
    # Or for temporary memory storage (erased when program stops):
# client = chromadb.Client()

##########################################################################################################################
# 3rd iteration
import json
from typing import Dict, List, Any

# history_queue=ConversationHistoryQueue(max_size=10)
from collections import deque
from typing import Dict, List, Optional


class ConversationHistoryQueue:
    def __init__(self, max_size: int = 10):
        """Fixed-size FIFO queue for storing conversation turns."""
        self.queue = deque(maxlen=max_size)

    def add_interaction(self, question: str, answer: str) -> None:
        """Add a (question, answer) pair. Automatically removes the oldest entry if size > 10."""
        self.queue.append({
            "question": question,
            "answer": answer
        })

    def get_most_recent(self) -> Optional[Dict[str, str]]:
        """Return the most recently inserted question-answer pair."""
        if not self.queue:
            return None
        return self.queue[-1]

    def get_all_as_messages(self) -> List[Dict[str, str]]:
        """Format the entire queue into OpenAI/Groq messages format."""
        messages = []
        for turn in self.queue:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        return messages

    def get_formatted_context_string(self) -> str:
        """Format the history as a string to inject into a prompt template."""
        if not self.queue:
            return "No previous history."
        
        history_lines = []
        for i, turn in enumerate(self.queue, 1):
            history_lines.append(f"Turn {i}:")
            history_lines.append(f"User: {turn['question']}")
            history_lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(history_lines)

    def __len__(self) -> int:
        return len(self.queue)

history_queue=ConversationHistoryQueue(max_size=10)

# print("INSIDE QUEUE",history_queue)

def deta_retrieval_from_vector_db(question: str) -> Dict[str, Any]:
    """
    Perform hybrid retrieval (Vector + BM25) and return structured JSON response
    """
    # Fetch previous interaction(context)
    history_str=history_queue.get_formatted_context_string()
    collection = client.get_or_create_collection(
        name="transcriber_rag"
    )

    print("\nQUESTION:", question)

    # Initialize response structure
    response_data = {
        "status": "success",
        "question": question,
        "answer": "",
        "sources": {
            "vector_search": [],
            "bm25_search": [],
            "combined": []
        },
        "metadata": {
            "vector_count": 0,
            "bm25_count": 0,
            "total_retrieved": 0,
            "unique_count": 0,
            "final_count": 0
        },
        "error": None
    }

    try:
        # ==================================================
        # 1. VECTOR SEARCH
        # ==================================================
        vector_results = retrieval_of_documents(
            collection,
            question
        )

        # Extract actual documents from Chroma
        vector_documents = vector_results["documents"][0]
        
        # Store vector results with metadata
        vector_source = []
        for i, doc in enumerate(vector_documents):
            vector_source.append({
                "index": i + 1,
                "content": doc,
                "preview": doc[:300] + "..." if len(doc) > 300 else doc,
                "length": len(doc)
            })

        print("\n===== VECTOR DOCUMENTS =====")
        for i, doc in enumerate(vector_documents):
            print(f"\nVector Document {i + 1}:")
            print(doc[:300])

        # ==================================================
        # 2. BM25 SEARCH
        # ==================================================
        bm25_results = get_documents_by_key(
            question
        )

        # Store BM25 results with metadata
        bm25_source = []
        for i, doc in enumerate(bm25_results):
            bm25_source.append({
                "index": i + 1,
                "content": doc.page_content,
                "preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "length": len(doc.page_content),
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            })

        print("\n===== BM25 DOCUMENTS =====")
        for i, doc in enumerate(bm25_results):
            print(f"\nBM25 Document {i + 1}:")
            print(doc.page_content[:300])

        # ==================================================
        # 3. CONVERT BM25 DOCUMENTS TO STRINGS
        # ==================================================
        bm25_documents = [
            doc.page_content
            for doc in bm25_results
        ]

        # ==================================================
        # 4. COMBINE VECTOR + BM25
        # ==================================================
        combined_documents = (
            vector_documents +
            bm25_documents
        )

        # ==================================================
        # 5. REMOVE DUPLICATES (preserve order)
        # ==================================================
        unique_documents = list(
            dict.fromkeys(
                combined_documents
            )
        )

        # ==================================================
        # 6. LIMIT NUMBER OF DOCUMENTS
        # ==================================================
        final_documents = unique_documents[:6]

        # Update metadata
        response_data["metadata"]["vector_count"] = len(vector_documents)
        response_data["metadata"]["bm25_count"] = len(bm25_documents)
        response_data["metadata"]["total_retrieved"] = len(combined_documents)
        response_data["metadata"]["unique_count"] = len(unique_documents)
        response_data["metadata"]["final_count"] = len(final_documents)

        # Store sources
        response_data["sources"]["vector_search"] = vector_source
        response_data["sources"]["bm25_search"] = bm25_source
        response_data["sources"]["combined"] = [
            {
                "index": i + 1,
                "content": doc,
                "preview": doc[:300] + "..." if len(doc) > 300 else doc,
                "length": len(doc)
            }
            for i, doc in enumerate(final_documents)
        ]

        print(
            "\nFINAL DOCUMENT COUNT:",
            len(final_documents)
        )

        # ==================================================
        # 7. CREATE CONTEXT
        # ==================================================
        context = "\n\n".join(
            final_documents
        )

        print("\n===== FINAL CONTEXT =====")
        print(context)

        # ==================================================
        # 8. CREATE RAG PROMPT
        # ==================================================
#         prompt = f"""
# You are an AI assistant who will answer the question asked based on the retrieved context/document not from any other sources.
# Donot make up any answer and answer the question directly in limited words.
# Answer strictly in JSON format only
# Retrieved Context:
# -----------------------------
# {context}
# -----------------------------

# User Question:
# -----------------------------
# {question}
# -----------------------------

# Final Answer:
# """
        prompt = f"""You are an AI assistant. Answer the question based ONLY on the provided context.
            Do NOT make up any answer.

            You MUST respond with a valid JSON object. The JSON object MUST have this exact structure:
            {{
                "answer": "your concise answer here",
                "confidence": "high" or "medium" or "low",
                "sources_found": true or false,
                "source_count": number of sources used
            }}

            If the answer is NOT found in the context, return:
            {{
                "answer": "I don't have enough information to answer this question.",
                "confidence": "low",
                "sources_found": false,
                "source_count": 0
            }}

            RETURN ONLY THE JSON OBJECT. NO OTHER TEXT.
            If any context of previous context exists take it from here else create a new workflow:
            {history_str}

            Retrieved Context:
            {context}

            User Question:
            {question}"""

        # ==================================================
        # 9. GROQ CLIENT
        # ==================================================
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

        ai_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

        # ==================================================
        # 10. LLM
        # ==================================================
        completion = ai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a direct assistant that gives concise answers based only on context You must respond strictly in JSON format. Answer directly."},
                {"role": "user", "content": prompt}
            ],
            # response_format = {"type": "json_object"},
            model="qwen/qwen3.6-27b",
            max_tokens=250,
            reasoning_effort="none"
        )

        # Extract answer
        answer = completion.choices[0].message.content

        print("\n===== FINAL ANSWER =====")
        print("ANSWER:", answer)

        # # Update response with answer
        response_data["answer"] = answer
        history_queue.add_interaction(question=question, answer=response_data["answer"])
        response_data["metadata"]["tokens_used"] = completion.usage.total_tokens if hasattr(completion, 'usage') else 0
        # raw_content = completion.choices[0].message.content
        # parsed = json.loads(raw_content)
        # answer = parsed.get("answer", raw_content)
        # To maintain context we push the question and answer to the queue
        print("INSIDE_QUEUE:", len(history_queue))
        return response_data

    except Exception as e:
        print("ERROR:", str(e))
        
        response_data["status"] = "error"
        response_data["error"] = str(e)
        response_data["answer"] = f"Error: {str(e)}"
        
        return response_data


# ============================================================
# HELPER FUNCTION TO DISPLAY JSON
# ============================================================

def display_rag_response(response_data: Dict[str, Any]):
    """Pretty print the RAG response in a readable format"""
    
    print("\n" + "="*60)
    print("📋 RAG RESPONSE SUMMARY")
    print("="*60)
    
    print(f"\n✅ Status: {response_data['status']}")
    print(f"❓ Question: {response_data['question']}")
    
    if response_data['status'] == 'error':
        print(f"❌ Error: {response_data['error']}")
        return
    
    print("\n" + "="*60)
    print("📄 ANSWER")
    print("="*60)
    print(response_data['answer'])
    
    print("\n" + "="*60)
    print("📊 RETRIEVAL STATISTICS")
    print("="*60)
    metadata = response_data['metadata']
    print(f"   Vector Documents: {metadata['vector_count']}")
    print(f"   BM25 Documents: {metadata['bm25_count']}")
    print(f"   Total Retrieved: {metadata['total_retrieved']}")
    print(f"   Unique Documents: {metadata['unique_count']}")
    print(f"   Final Documents Used: {metadata['final_count']}")
    
    print("\n" + "="*60)
    print("📚 SOURCES")
    print("="*60)
    
    for i, source in enumerate(response_data['sources']['combined'], 1):
        print(f"\n  [Source {i}] (Length: {source['length']} chars)")
        print(f"  {source['preview']}")
        print("  " + "-"*40)

    return response_data['answer']
# ============================================================
# USAGE EXAMPLE
# ============================================================

# Call the function
# result = deta_retrieval_from_vector_db("What is nuclear fusion?")