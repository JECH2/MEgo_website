from .models import *
from django.core.files import File

# example code for uploading .png by using python code
def make_report():

    # username.csv
    username = export_user_experience()
    analysis(username)

    report = Report()
    report.event_wordcloud.save('e_w.jpg',File(open('report/'+username+'_wordcloud_event.png', 'rb')))
    report.thought_wordcloud.save('t_w.jpg',File(open('report/'+username+'_wordcloud_thought.jpg', 'rb')))

