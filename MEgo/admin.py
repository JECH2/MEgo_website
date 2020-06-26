from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import Experience, User, ExpQuestions, LifeQuestions, LifeIWish, EmotionColor

admin.site.register(Experience)
admin.site.register(User)
admin.site.register(ExpQuestions)
admin.site.register(LifeQuestions)
admin.site.register(LifeIWish)
admin.site.register(EmotionColor)
