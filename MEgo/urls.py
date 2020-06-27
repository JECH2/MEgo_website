# all urls are managed in here (and check also ../MEgo_website/urls.py)
# if get a request, it is connected to views.py

from django.urls import path
from . import views
import django.contrib.auth.views as views2
from .forms import *

urlpatterns = [
    path('', views.experience_list, name='experience_list'),
    path('MEgo/<int:pk>/', views.experience_detail, name='experience_detail'),
    path('MEgo/new/', views.experience_new, name='experience_new'),
    path('MEgo/<int:pk>/edit/', views.experience_edit, name='experience_edit'),
    #path('MEgo/<int:pk>/edit/', views.NewExpbyQView.as_view(), name='experience_edit'),
    path('MEgo/new_m/', views. ExpFormWizardView.as_view([ExpFormStepOne, ExpFormStepTwo, ExpFormStepThree]), name='experience_new_m'),
    path('MEgo/report/', views.analysis_report, name='analysis_report'),
    path('MEgo/new_q/', views.new_question, name='new_question'),
    #path('MEgo/new_q/<int:pk>/', views.NewExpbyQView.as_view(), name='experience_new_by_question'),
    path('MEgo/new_q/<int:pk>/', views.experience_new_by_question, name='experience_new_by_question'),
    path('support/', views.support, name='support'),
    path('accounts/signup/', views.signup, name='signup'),
    path('accounts/login/', views2.LoginView.as_view(
            template_name="registration/login.html",
            authentication_form=CustomUserLoginForm
            ),
         name='login'),
]