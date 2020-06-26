import requests
import json
import csv
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MEgo_website.settings")
django.setup()

# 어떤 모델에서 가져올 것인지
from MEgo.models import Experience, User

with open('exp.csv', 'w', newline='') as csvfile:
    fieldnames = ['emotion', 'event'] # 가져올 field name
    data_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    data_writer.writeheader()

    usr = User.objects.get(user_id='jech') # 누구의 데이터를 가져올 것인가

    for exp in Experience.objects.filter(author__exact=usr.id): # 필터링해서 가져올 수도 있다.
        data_writer.writerow({'emotion':exp.emotion, 'event':exp.event})