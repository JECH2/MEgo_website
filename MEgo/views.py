from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from .models import Experience, User
from .forms import ExpForm, UserForm
from django.urls import reverse_lazy
from django.views.generic.edit import CreateView

@login_required
def experience_list(request):
    if request.user.is_authenticated and request.user.user_id == 'admin':
        exps = Experience.objects.order_by('exp_date')
    else:
        exps = Experience.objects.filter(author__exact=request.user.id).order_by('exp_date')
    return render(request, 'MEgo/experience_list.html', {'exps':exps})

@login_required
def experience_detail(request, pk):
    exp = get_object_or_404(Experience, pk=pk)
    return render(request, 'MEgo/experience_detail.html', {'exp':exp})

@login_required
def experience_new(request):
    # 폼에 데이터를 받은 상황
    if request.method == "POST":
        form = ExpForm(request.POST, request.FILES)
        if form.is_valid():
            exp = form.save(commit=False)
            exp.author = request.user
            exp.save()
            return redirect('experience_detail', pk=exp.pk)
    else:
        form = ExpForm() # 폼 저장 전
    return render(request, 'MEgo/experience_edit.html', {'form': form})

@login_required
def experience_edit(request, pk):
    exp = get_object_or_404(Experience, pk=pk)
    if request.method == "POST":
        form = ExpForm(request.POST, request.FILES, instance=exp)
        if form.is_valid():
            exp = form.save(commit=False)
            exp.author = request.user
            exp.save()
            return redirect('experience_detail', pk=exp.pk)
    else:
        form = ExpForm(instance=exp)
    return render(request, 'MEgo/experience_edit.html', {'form': form})

def analysis_report(request):
    return render(request, 'MEgo/analysis_report.html')

def new_question(request):
    return render(request, 'MEgo/new_question.html')

def support(request):
    return render(request, 'support.html')

def signup(request):
    if request.method == 'POST':
        form = UserForm(request.POST)
    # 모델폼의 유효성 검증이 valid할 경우, DB에 저장
        if form.is_valid():
            user_instance = form.save()
            login(request, user_instance)
        return render(request, 'registration/signup_complete.html', {'id' : id })

# HTTP Method가 GET 인 경우
    else:
        form = UserForm()

    return render(request, 'registration/signup.html', {'form': form})

