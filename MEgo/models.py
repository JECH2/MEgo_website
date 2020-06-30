# our data frames are defined in here

from django.conf import settings
from django.utils import timezone
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models


# when user account is created via command, user manager function is called
class UserManager(BaseUserManager):
    use_in_migrations = True

    def create_user(self, user_id, email, age, gender, nickname, password):
        if not (email):
            raise ValueError('must have user email')
        user = self.model(
            email=self.normalize_email(email),
            nickname=nickname,
            user_id=user_id,
            age=age,
            gender=gender
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, user_id, nickname, password):
        user = self.create_user(
            user_id=user_id,
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
    user_id = models.CharField(
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

    USERNAME_FIELD = 'user_id'
    REQUIRED_FIELDS = ['email','nickname']

    def __str__(self):
        return self.user_id

# color of emotion
class EmotionColor(models.Model):
    color_name = models.CharField(max_length=200) # content of question
    emotion = models.CharField(max_length=200) # event, thoughts, emotion
    r = models.IntegerField()
    g = models.IntegerField()
    b = models.IntegerField()
    a = models.FloatField()

# questions of daily life
# example : if question is "What makes you happy?",
# data is stored as (What makes you happy?, emotion, angry)
class ExpQuestions(models.Model):
    content = models.CharField(max_length=200) # content of question
    question_area = models.CharField(max_length=200) # event, thoughts, emotion
    related_tags = models.CharField(max_length=200)
    answer_area = models.CharField(max_length=200, blank=True, null=True)

# questions of life
class LifeQuestions(models.Model):
    content = models.CharField(max_length=200) # content of question
    question_area = models.CharField(max_length=200) # one of life I wish
    related_tags = models.CharField(max_length=200, blank=True, null=True)
    answer_area = models.CharField(max_length=200, blank=True, null=True)

# data structure for recording life I wish
class LifeIWish(models.Model):
    input_date = models.DateTimeField(default=timezone.now)
    life_values_high = models.TextField(blank=True, null=True)
    life_values_low = models.TextField(blank=True, null=True)
    ideal_person = models.TextField(blank=True, null=True)
    life_goals = models.TextField(blank=True, null=True)
    goal_of_the_year_2020 = models.TextField(blank=True, null=True) # goals of this year
    goal_of_the_year_2030 = models.TextField(blank=True, null=True)  # goals of this year
    goal_of_the_year_2040 = models.TextField(blank=True, null=True) # goals of this year
    goal_of_the_year_2050 = models.TextField(blank=True, null=True) # goals of this year


# This function is needed for uploading user's data
def user_path(instance, filename): #param instance is meaning for model, filename is the name of uploaded file
    from random import choice
    import string # string.ascii_letters : ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
    arr = [choice(string.ascii_letters) for _ in range(8)]
    pid = ''.join(arr) # 8 length random string is file name
    extension = filename.split('.')[-1] # extract the file extension such as .png .jpg. and etc.
    # file will be uploaded to MEDIA_ROOT/user_<id>/<random>
    return '%s/%s.%s' % (instance.author.id, pid, extension) # ex : wayhome/abcdefgs.png

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
    thoughts = models.TextField(null=True)
    emotion = models.TextField()
    # emotion_intensity = models.IntegerField()
    importance = models.IntegerField()
    future = models.IntegerField(default=0)
    #Social Data
    related_people = models.CharField(max_length=200, blank=True, null=True)
    related_place = models.CharField(max_length=200, blank=True, null=True)

    # picture & video & youtube --> link(str) : youtube API 찾아보기
    photo = models.ImageField(blank=True,upload_to=user_path)  # path based on the function and settings
    thumbnail_photo = models.ImageField(blank=True, upload_to=user_path) # it is not required field
    media_links = models.TextField(blank=True, null=True)

    emotion_color = models.CharField(max_length=100, default="", blank=True, null=True)

    def publish(self):
        self.save()

    def __str__(self):
        return self.event