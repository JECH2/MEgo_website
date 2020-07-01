from .models import *
from django.core.files import File


def make_report():

    # username.csv
    username = export_user_experience()
    analysis(username)

    report = Report()
    report.event_wordcloud.save('e_w.jpg',File(open('reports/'+username+'_wordcloud_event.png', 'rb')))
    report.thought_wordcloud.save('t_w.jpg',File(open('reports/'+username+'_wordcloud_thought.jpg', 'rb')))

