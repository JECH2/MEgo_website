from django.conf import settings
from django.utils import timezone
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models


# when user account is created via command, user manager function is called
class UserManager(BaseUserManager):
    use_in_migrations = True

    def create_user(self, email, age, gender, nickname, password=None):
        if not (email):
            raise ValueError('must have user email')
        user = self.model(
            email=self.normalize_email(email),
            nickname=nickname,
            age=age,
            gender=gender
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, nickname, password):
        user = self.create_user(
            email=self.normalize_email(email),
            nickname=nickname,
            password=password,
            age=22,
            gender="Female"
        )
        user.is_admin = True
        user.is_superuser = True
        user.is_staff = True
        user.save(using=self._db)
        return user

# we use AbstractBaseUser to define our custom user
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
    # additional field : age and gender
    age = models.IntegerField(null = False)
    gender = models.CharField(max_length=200, null = False)

    USERNAME_FIELD = 'nickname'
    REQUIRED_FIELDS = ['email']

    def __str__(self):
        return self.nickname


# example : if question is "What makes you happy?",
# data is stored as (What makes you happy?, emotion, angry)
class Questions(models.Model):
    content = models.CharField(max_length=200) # content of question
    question_area = models.CharField(max_length=200) # event, thoughts, emotion
    related_tags = models.CharField(max_length=200)


# experience data structure
class Experience(models.Model):
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, default='admin')
    input_date = models.DateTimeField(
            default=timezone.now)
    exp_date = models.DateTimeField(
            default=timezone.now)
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
    # emotion_intensity = models.IntegerField()
    importance = models.IntegerField()
    future = models.IntegerField(default=0)
    #Social Data
    related_people = models.CharField(max_length=200, blank=True, null=True)
    related_place = models.CharField(max_length=200, blank=True, null=True)

    # picture & video & youtube --> link(str) : youtube API 찾아보기
    media_links = models.TextField(blank=True, null=True)

    def publish(self):
        self.save()

    def __str__(self):
        return self.event