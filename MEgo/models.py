from django.conf import settings
from django.db import models
from django.utils import timezone


class Event(models.Model):
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    text = models.TextField()
    created_date = models.DateTimeField(
            default=timezone.now)
    published_date = models.DateTimeField(
            blank=True, null=True)
    # Health Data
    walks = models.IntegerField()
    sleep = models.FloatField()
    deep_sleep = models.FloatField()
    heartbeat = models.IntegerField()
    calorie = models.IntegerField()
    distance = models.FloatField()
    # Mental Data
    event = models.TextField()
    thoughts = models.TextField()
    emotion = models.TextField()
    emotion_intensity = models.IntegerField()
    importance = models.IntegerField()
    future = models.IntegerField()
    #Social Data
    related_people = models.TextField()
    related_place = models.TextField()

    def publish(self):
        self.published_date = timezone.now()
        self.save()

    def __str__(self):
        return self.title