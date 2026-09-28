from datetime import timedelta

from django.db import models

# Create your models here.
class VideoTranscriptData(models.Model):
    video_id = models.CharField(max_length=255, unique=True)
    video_name = models.CharField(max_length=255)
    video_duration = models.IntegerField() 
    video_count = models.IntegerField()
    channel_name = models.CharField(max_length=50)
    transcript = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"VideoTranscript(video_id={self.video_id})"

    def seconds_to_timedelta(self):
        """Converts seconds to standard duration string using timedelta."""
        return str(timedelta(seconds=(self.video_duration)))
    