from django.shortcuts import render
from .utils import qa_queue, get_youtube_video_id,get_the_video_transcripts,get_the_video_summary
from .embedding import retrieve_based_on_question,embed_the_chunks
from .models import VideoTranscriptData
# Create your views here.
def landing_page(request):
    video_obj=VideoTranscriptData.objects.all().order_by("-created_at")[0:5]
    return render(request,"index.html",context={"video_obj":video_obj})

def get_transcripts_by_id(request,video_id):
    video_obj=VideoTranscriptData.objects.filter(video_id=video_id).first()
    return render(request,"summary.html",context={"video_obj":video_obj})

def question_me(request):
    if request.method == "POST":
        question = request.POST.get("question", "").strip()

        if not question:
            return render(
                request,
                "ask.html",
                {
                    "error": "Please enter a question.",
                    "chat_history": [],
                    "searched_data": []
                }
            )

        # `answer` should be your parsed answer dictionary.
        answer, searched_data = retrieve_based_on_question(question)

        # If parsed_data is {"answer": "...", "source_count": 0}
        # final_answer = answer.get("answer", "")

        # Store one complete interaction and return latest 10 entries.
        chat_history = qa_queue(
            question=question,
            answer=answer
        )

        # `searched_data` may not contain searched_info if vector sources exist.
        get_searched_data = searched_data.get(
            "searched_info",
            []
        )

        print("CHAT HISTORY:", chat_history)
        print("SEARCHED DATA:", get_searched_data)
        print("HISTORY",chat_history)
        context = {
            "chat_history": chat_history,
            "searched_data": get_searched_data
        }
        return render(request, "ask.html", context)
    return render(
        request,
        "ask.html",
        {
            "chat_history": [],
            "searched_data": []
        }
    )
# def video_summary(request):
#     get_video_url=request.post("video_url")
#     # From here we start the pipeline for the LLM to generate the summary
#     video_id=get_youtube_video_id(get_video_url)
#     get_transcripts=get_the_video_transcripts(request,video_id)
#     get_the_video_summary(request,get_transcripts)
#     return render(request, "summary.html")

def video_summary(request):

    if request.method == "POST":

        # Get YouTube URL from form
        get_video_url = request.POST.get("video_url")

        if not get_video_url:
            return render(
                request,
                "summary.html",
                {
                    "error": "Please provide a YouTube URL."
                }
            )

        # Get YouTube video ID
        video_id = get_youtube_video_id(get_video_url)

        if not video_id:
            return render(
                request,
                "summary.html",
                {
                    "error": "Invalid YouTube URL."
                }
            )

        # Get transcript
        transcripts = get_the_video_transcripts(
            request,
            video_id
        )
        print("HERE---------->",type(transcripts),transcripts)
        # Generate summary
        summary = get_the_video_summary(
            request,
            transcripts
        )
        print("Generating the Embeddings for VIDEO")
        generate_video_embeddings=embed_the_chunks(transcripts)
        print("SUMMARY OF THE VIDEO------------------------------------------------------------------------------------------------",summary)
        return render(
            request,
            "summary.html",
            {
                "summary": summary,
                "video_id": video_id
            }
        )
    # GET request
    return render(request, "summary.html")

 