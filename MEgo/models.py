from django.conf import settings
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import AbstractUser

from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models


class UserManager(BaseUserManager):
    use_in_migrations = True

    def create_user(self, email, nickname, password=None):
        if not email:
            raise ValueError('must have user email')
        user = self.model(
            email=self.normalize_email(email),
            nickname=nickname
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, nickname, password):
        user = self.create_user(
            email=self.normalize_email(email),
            nickname=nickname,
            password=password
        )
        user.is_admin = True
        user.is_superuser = True
        user.is_staff = True
        user.save(using=self._db)
        return user


class User(AbstractBaseUser, PermissionsMixin):
    objects = UserManager()

    email = models.EmailField(
        max_length=255,
        unique=True,
    )
    nickname = models.CharField(
        max_length=20,
        null=False,
        unique=True
    )
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)
    date_joined = models.DateTimeField(auto_now_add=True)
    USERNAME_FIELD = 'nickname'
    REQUIRED_FIELDS = ['email']

class Experience(models.Model):
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
    Experience = models.TextField()
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

    def __str__(self):
        return self.event