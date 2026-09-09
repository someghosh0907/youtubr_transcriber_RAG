from django.contrib import admin
from django.urls import path
from .views import landing_page,question_me,video_summary,get_transcripts_by_id

urlpatterns = [
    path('', landing_page, name="landing"),
    path('ask', question_me, name="question_me"),
    path("summary", video_summary, name="video_summary"),
    path("summary/<str:video_id>", get_transcripts_by_id, name="get_transcripts_by_id")
]
