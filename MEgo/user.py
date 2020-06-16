# coningguapp/models/user.py
from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    tel = models.CharField("연락처", max_length=20, null=True, blank=True)
    created_on = models.DateTimeField("등록일자", auto_now_add=True)
    updated_on = models.DateTimeField("수정일자", auto_now=True)

    class Meta:
        db_table = 'user'