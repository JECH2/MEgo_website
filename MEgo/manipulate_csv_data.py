import csv
import os
import django

from MEgo.models import ExpQuestions, LifeQuestions, EmotionColor

from MEgo.models import Experience, User
from MEgo.color import emo_to_hex

from MEgo.models import *

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MEgo_website.settings")
django.setup()

#CSV_PATH = 'MEgo/report/jiseong.csv'
# CSV PATH is MEgo/report/username.csv
# export experience data into csv format
def export_exp_data(csv_path):
    username = os.path.basename(csv_path)[:-4]
    usr = User.objects.get(user_id=username)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['author','emotion_label', 'event', 'thoughts','importance','related_people','related_place'] # 가져올 field name
        data_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        data_writer.writeheader()
        for exp in Experience.objects.filter(author__exact=usr.id): # 필터링해서 가져올 수도 있다.
            data_writer.writerow({'author':exp.author,'emotion_label':exp.emotion, 'event':exp.event,'thoughts':exp.thoughts, 'related_people':exp.related_people, 'related_place':exp.related_place})

#CSV_PATH = 'MEgo/report/jiseong.csv'

# writes experience data as csv
def import_exp_data(csv_path):
    username = os.path.basename(csv_path)[:-4]
    usr = User.objects.get(user_id=username)

    with open(csv_path, newline='') as csvfile:
        data_reader = csv.DictReader(csvfile)

        for row in data_reader:
            print(row)
            parsed_emotion = row['emotion_label'].strip().split(',')
            exp = Experience.objects.create(
                author = usr,
                event = row['event'],
               thoughts = row['thoughts'],
              emotion = row['emotion_label'],
                importance = row['importance'],
                related_people=row['related_people'],
                related_place=row['related_place'],
                emotion_color=emo_to_hex(parsed_emotion),
            )
