from urllib.parse import urlparse, parse_qs
from collections import deque
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from .embedding import creating_video_summary
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from youtube_transcript_api._errors import (
    NotTranslatable,
    TranslationLanguageNotAvailable,
)
import yt_dlp
from .models import VideoTranscriptData

# Create a context of 

# To retrieve the video Id from the url
def get_youtube_video_id(url: str) -> str | None:
    parsed_url = urlparse(url)

    # youtu.be/<video_id>
    if parsed_url.netloc in {"youtu.be", "www.youtu.be"}:
        return parsed_url.path.strip("/").split("/")[0]

    # youtube.com/watch?v=<video_id>
    if parsed_url.netloc in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com"
    }:
        query_params = parse_qs(parsed_url.query)

        if "v" in query_params:
            return query_params["v"][0]

        # youtube.com/embed/<video_id>
        if parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/embed/")[1].split("/")[0]

        # youtube.com/shorts/<video_id>
        if parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/shorts/")[1].split("/")[0]

    return None

# def get_the_video_transcripts(request,video_id):
#     api = YouTubeTranscriptApi()
#     # request.session['video_id'] = video_id
# ## If you don’t care which language, this returns the “best” one
#     transcript_list = api.fetch(video_id=video_id, languages=["en","hi"])

# ## Flatten it to plain text
#     transcript = " ".join(
#         snippet.text
#         for snippet in transcript_list
#     )

#     print(transcript)
#     try:
#         # api = YouTubeTranscriptApi()
#         transcript_list = api.list(video_id)

#         hindi_transcript = transcript_list.find_transcript(
#             ["hi"]
#         )
#         try:
#             english_transcript = (
#                 hindi_transcript
#                 .translate("en")
#                 .fetch()
#             )

#             english_text = " ".join(
#                 snippet.text
#                 for snippet in english_transcript
#             )

#             print(english_text)
#             # We will be sending the transscript to be embedded
#             # If the we want summary then a different response and for question a different one
#             # deta_retrieval_from_vector_db()
#             return english_text #Here we returnthe wholetranscript
#         except NotTranslatable:
#             print(
#                 "This Hindi transcript cannot be translated "
#                 "by YouTube."
#             )
#     except TranscriptsDisabled:
#         print("Transcripts are disabled.")

#     except NoTranscriptFound:
#         print("No Hindi transcript was found.")

#     except VideoUnavailable:
#         print("The video is unavailable.")
# except TranscriptsDisabled:
#     print("No captions available for this video.")

def get_the_video_transcripts(request, video_id):
    
    api = YouTubeTranscriptApi()
    ydl_opts = {
        'extract_flat': True,  # True skips downloading and just pulls metadata
        'skip_download': True,
    }
    url = f'https://www.youtube.com/watch?v={video_id}'
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        print(f"Title: {info_dict.get('title')}")
        print(f"Duration: {info_dict.get('duration')} seconds")
        print(f"View Count: {info_dict.get('view_count')}")
        print(f"Channel: {info_dict.get('uploader')}")
    try:
        transcript_list = api.fetch(
            video_id,
            languages=["en", "hi"]
        )

        transcript = " ".join(
            snippet.text
            for snippet in transcript_list
        )
        create_video_object=VideoTranscriptData.objects.create(
                    video_id=video_id,
                    video_name=info_dict.get('title'),
                    video_duration=info_dict.get('duration'),
                    video_count=info_dict.get('view_count'),
                    channel_name=info_dict.get('uploader'),
                    transcript=transcript
                )
        create_video_object.save()
        print(transcript)

        if transcript.strip():
            return transcript

        return None

    except TranscriptsDisabled:
        print("Transcripts are disabled")
        return None

    except NoTranscriptFound:
        print("No English/Hindi transcript found")
        return None

    except VideoUnavailable:
        print("Video is unavailable")
        return None

    except Exception as e:
        print(f"Transcript error: {e}")
        return None

def get_the_video_summary(request,transcript):
    print(type(transcript))
    # video_id=request.session["video_id"]
    # get_the_video_transcripts(request,video_id)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    len(chunks)
    data=creating_video_summary(chunks)
    return data

# def get_retrieved_document(question):
#     get_the_video_transcripts(request,video_id)
#     embed_the_chunks(transcript=english_text)  #For vector retrieval pipeline
    
history_queue = deque(maxlen=10)

def qa_queue(question: str, answer: str) -> list[dict]:
    """
    Store one question-answer pair.

    Keeps only the latest 10 conversations.
    Returns a normal list for Django template rendering.
    """
    history_queue.append({
        "question": question,
        "answer": answer
    })

    return list(history_queue)

# History queue


