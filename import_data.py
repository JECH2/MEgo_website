import csv
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MEgo_website.settings")
django.setup()

from MEgo.models import ExpQuestions, LifeQuestions, EmotionColor

# CSV_PATH = 'Question Data - Daily Questions.csv'
#CSV_PATH = 'Question Data - Life Questions.csv'
CSV_PATH = 'Question Data - emotion color.csv'

# with open(CSV_PATH, newline='') as csvfile:
#     data_reader = csv.DictReader(csvfile)
#     for row in data_reader:
#         print(row)
#         ExpQuestions.objects.create(
#             content = row['content'],
#             question_area = row['question area'],
#             related_tags = row['related tags']
#         )

#with open(CSV_PATH, newline='') as csvfile:
#    data_reader = csv.DictReader(csvfile)
#    for row in data_reader:
#        print(row)
#        LifeQuestions.objects.create(
#            content = row['content'],
#            question_area = row['question area'],
#            related_tags = row['related tags'],
#            answer_area = row['answer area']
#        )

with open(CSV_PATH, newline='') as csvfile:
    data_reader = csv.DictReader(csvfile)
    for row in data_reader:
        print(row)
        EmotionColor.objects.create(
            color_name = row['color name'],
            emotion = row['emotion'],
            r = row['r'],
            g = row['g'],
            b = row['b'],
            a = row['a']
        )

