from django.conf import settings
from django.db import models
from django.utils import timezone


class Event(models.Model):
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, default='admin')
    input_date = models.DateTimeField(
            default=timezone.now)
    exp_date = models.DateTimeField(
            blank=True, null=True)
    # Health Data
    walks = models.IntegerField(blank=True, null=True)
    sleep = models.FloatField(blank=True, null=True)
    deep_sleep = models.FloatField(blank=True, null=True)
    heartbeat = models.IntegerField(blank=True, null=True)
    calorie = models.IntegerField(blank=True, null=True)
    distance = models.FloatField(blank=True, null=True)
    # Mental Data
    event = models.TextField()
    thoughts = models.TextField(blank=True, null=True)
    emotion = models.TextField()
    emotion_intensity = models.IntegerField()
    importance = models.IntegerField()
    future = models.IntegerField(default=0)
    #Social Data
    related_people = models.TextField(blank=True, null=True)
    related_place = models.TextField(blank=True, null=True)

    def publish(self):
        self.exp_date = timezone.now()
        self.save()