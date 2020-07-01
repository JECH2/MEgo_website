import requests
import json
import csv
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MEgo_website.settings")
django.setup()

# 어떤 모델에서 가져올 것인지
#from MEgo.models import Experience, User

#with open('exp.csv', 'w', newline='') as csvfile:
#    fieldnames = ['emotion', 'event'] # 가져올 field name
#    data_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#    data_writer.writeheader()

#    usr = User.objects.get(user_id='jech') # 누구의 데이터를 가져올 것인가

#    for exp in Experience.objects.filter(author__exact=usr.id): # 필터링해서 가져올 수도 있다.
#        data_writer.writerow({'emotion':exp.emotion, 'event':exp.event})

from MEgo.models import *
#CSV_PATH = 'MEgo/report/jiseong.csv'
# CSV PATH is MEgo/report/username.csv
# export experience data into csv format
def export_exp_data(CSV_PATH)
    username = os.path.basename(CSV_PATH)[:-4]
    usr = User.objects.get(user_id=username)
    with open(CSV_PATH, 'w', newline='') as csvfile:
        fieldnames = ['author','emotion_label', 'event', 'thoughts','importance','related_people','related_place'] # 가져올 field name
        data_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        data_writer.writeheader()
        for exp in Experience.objects.filter(author__exact=usr.id): # 필터링해서 가져올 수도 있다.
            data_writer.writerow({'author':exp.author,'emotion_label':exp.emotion, 'event':exp.event,'thoughts':exp.thoughts, 'related_people':exp.related_people, 'related_place':exp.related_place})
