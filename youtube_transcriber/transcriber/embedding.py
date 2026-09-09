# from transcriber.utils import get_the_video_transcripts
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_openai import OpenAIEmbeddings
# from langchain_faiss import FAISS
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from .chroma import display_rag_response,create_db_collection,deta_retrieval_from_vector_db
from .retrieval import get_documents_by_key
from langchain_core.prompts import PromptTemplate
import os
from openai import OpenAI
import json
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "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")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
# GEMINI_API_KEY="AQ.Ab8RN6K9BDLFyOlaWylDA21jJ5VXYB4Xl4CEUugWc9pcbJv6Iw"
# client = OpenAI(
#     base_url="https://router.huggingface.co/v1",
#     api_key=HF_API_TOKEN,
# )
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
def embed_the_chunks(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    print(len(chunks))
    # print(chunks[0].page_content)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    create_db_collection(chunks,embeddings)   #Populating the DB
    print("Data embedded and added to DB")
    # Embedding and storing the embeddings done
    return "Data embedded and added to DB"

def retrieve_based_on_question(question):
    embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    question_embedding = embeddings.embed_query(question)
    # Call the Vector DB
    retrieved_docs=deta_retrieval_from_vector_db(question)
    # 1. Display as pretty JSON
    json_response=display_rag_response(retrieved_docs)
    # 2. Get raw JSON string (for API response)
    json_response = json.dumps(retrieved_docs, indent=2, ensure_ascii=False)
    # raw_answer = json_response.get("answer", "")
    print("\n\nRAW JSON RESPONSE:")
    print("-------------------------------------------------------FINALLLL-----------------------------------------------------------------")
    print(type(json_response))
    parsed = json.loads(json_response)
    answer = parsed.get("answer", "")
    print("FINAL",type(parsed),parsed)
    print("FINAL",type(answer),answer)
    parsed_data = json.loads(answer)
    print("PARSED DATA COUNT",type(parsed_data),parsed_data.get("source_count"))
    searched_data={}
    if parsed_data.get("source_count") == 0:
        # Start flow for web search
        web_search = DuckDuckGoSearchAPIWrapper(
            max_results=5,
            region="wt-wt",
            safesearch="moderate"
        )
        web_results = web_search.results(
            query=question,
            max_results=5
        )
        print("Search results-----" ,web_results,type(web_results))
        searched_data["searched_info"]=web_results
    # Extract the answer
    # extracted_answer = parsed_data["answer"]
    return parsed_data,searched_data

# 3. Save to file (optional)
    with open("rag_response.json", "w", encoding="utf-8") as f:
        json.dump(retrieved_docs, f, indent=2, ensure_ascii=False)
        # retrieved_key_based_search_result=get_documents_by_key(question)
        print(type(retrieved_docs),retrieved_docs)
        return retrieved_docs

def creating_video_summary(chunks):
    # This is the code for generating the sumamryof the whole video

    chunk_summaries = []
    combined_summaries=""
    prompt = PromptTemplate(
        template="""
        Summarize this section of a video transcript.

            Capture:
            - important concepts
            - explanations
            - examples
            - important facts

        Do not add information that isn't present.

        Transcript section:
        {context}
        """,
        input_variables=["context"]
    )
    for chunk in chunks:

        final_prompt = prompt.invoke({
            "context": chunk
        })
        # completion = client.responses.create(
        #     input=final_prompt.to_string(),
        #     # model="openai/gpt-oss-20b",
        #     model="qwen/qwen3.6-27b",  #NEW MODEL 26/8 2313 HRS
        #     # model="openai/gpt-oss-120b", ALTERNATIVE
        #     max_output_tokens=1500,
        #     reasoning_effort="none"
        # )

        completion = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {
                    "role": "user",
                    "content": final_prompt.to_string()
                }
            ],
            max_tokens=250,
            temperature=0,
            reasoning_effort="none"
        )

        summary = answer=completion.choices[0].message.content
        chunk_summaries.append(summary)
        print(summary)
        combined_summaries = "\n\n".join(chunk_summaries)
# Another LLM call for summarising the summaeised texts

    # prompt = PromptTemplate(
    #     template=f"""
    # Create ONE comprehensive summary of the entire video
    # using the section summaries below.

    # The final summary should:

    # 1. Explain what the video is about
    # 2. Cover all major topics
    # 3. Explain important concepts
    # 4. Include important examples
    # 5. Highlight key takeaways
    # 6. Remove repetition
    # 7. Maintain the logical flow of the video

    # Do not introduce information that isn't present
    # in the section summaries.

    # Section summaries:

    # {combined_summaries}
    # """
    # )
    total_summary = f"""
        Create ONE comprehensive summary of the entire video
        using the section summaries below.

        The final summary should:

        1. Explain what the video is about
        2. Cover all major topics
        3. Explain important concepts
        4. Include important examples
        5. Highlight key takeaways
        6. Remove repetition
        7. Maintain the logical flow of the video

        Do not introduce information that isn't present
        in the section summaries.

        Section summaries:

        {combined_summaries}
        """
    # total_summary = prompt.invoke({"context":combined_summaries})
    # completion = client.responses.create(
    #             input=total_summary,
    #             # model="openai/gpt-oss-20b",
    #             model="qwen/qwen3.6-27b",  #NEW MODEL 26/8 2313 HRS
    #             # model="openai/gpt-oss-120b", ALTERNATIVE
    #             max_output_tokens=1500,
    #             reasoning_effort="none"
    #         )
    completion = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {
                "role": "user",
                "content": total_summary
            }
        ],
        max_tokens=800,
        temperature=0,
        reasoning_effort="none"
    )
    print(completion.choices[0].message.content)
    answer=completion.choices[0].message.content
    response_data = {
            "status": "success",
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
          
    response_data["answer"] = answer
    return response_data["answer"]
    # print(final_response.content)


