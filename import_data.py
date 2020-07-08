# Import data for initialization

import csv
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MEgo_website.settings")
django.setup()

from MEgo.models import ExpQuestions, LifeQuestions, EmotionColor, LifeIWish, User, Experience

CSV_PATH = 'Question Data - Life I Wish.csv'
with open(CSV_PATH, newline='') as csvfile:
    username = 'jiseong'
    usr = User.objects.get(user_id=username)

    data_reader = csv.DictReader(csvfile)
    for row in data_reader:
        print(row)
        exp = LifeIWish.objects.create(
                             author = usr,
                             life_values_high = row['life values high'],
                             life_values_low = row['life values low'],
                             ideal_person = row['ideal person'],
                             life_goals = row['life goals'],
                             goal_of_the_year_2020 =row['goal of the year 2020'],
                            goal_of_the_year_2030=row['goal of the year 2030'],
                            goal_of_the_year_2040=row['goal of the year 2040'],
                            goal_of_the_year_2050=row['goal of the year 2050'],
            )
#CSV_PATH = 'Question Data - Daily Questions.csv'

#with open(CSV_PATH, newline='') as csvfile:
#   data_reader = csv.DictReader(csvfile)
#   for row in data_reader:
#       print(row)
#       ExpQuestions.objects.create(
#           content = row['content'],
#           question_area = row['question area'],
#           related_tags = row['related tags'],
#           answer_area = row['answer area']
#       )

#
# CSV_PATH = 'Question Data - Life Questions.csv'
# with open(CSV_PATH, newline='') as csvfile:
#    data_reader = csv.DictReader(csvfile)
#    for row in data_reader:
#        print(row)
#        LifeQuestions.objects.create(
#            content = row['content'],
#           question_area = row['question area'],
#          related_tags = row['related tags'],
#            answer_area = row['answer area']
#        )
#
# CSV_PATH = 'Question Data - emotion color.csv'
# with open(CSV_PATH, newline='') as csvfile:
#    data_reader = csv.DictReader(csvfile)
#    for row in data_reader:
#        print(row)
#        EmotionColor.objects.create(
#            color_name = row['color name'],
#            emotion = row['emotion'],
#            r = row['r'],
#            g = row['g'],
#            b = row['b'],
#            a = row['a']
#        )
#
# from MEgo.models import Experience, User
# from MEgo.color import emo_to_hex
# #CSV_PATH = 'MEgo/report/jiseong.csv'
#
# # writes experience data as csv
# def import_exp_data(CSV_PATH)
#     username = os.path.basename(CSV_PATH)[:-4]
#     usr = User.objects.get(user_id=username)
#
#     with open(CSV_PATH, newline='') as csvfile:
#         data_reader = csv.DictReader(csvfile)
#
#         for row in data_reader:
#             print(row)
#             parsed_emotion = row['emotion_label'].strip().split(',')
#             exp = Experience.objects.create(
#                 author = usr,
#                 event = row['event'],
#                thoughts = row['thoughts'],
#               emotion = row['emotion_label'],
#                 importance = row['importance'],
#                 related_people=row['related_people'],
#                 related_place=row['related_place'],
#                 emotion_color=emo_to_hex(parsed_emotion),
#             )
