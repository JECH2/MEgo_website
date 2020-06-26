from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from .models import Experience, User, ExpQuestions, EmotionColor
from .forms import ExpForm, DynamicExpForm, UserForm
from django.urls import reverse_lazy
from django.views.generic.edit import CreateView
from random import randint

def rgb_to_hex(r, g, b):
    r, g, b = int(r), int(g), int(b)
    return '#' + hex(r)[2:].zfill(2) + hex(g)[2:].zfill(2) + hex(b)[2:].zfill(2)

# from emotion list to overall hexcode
def emo_to_hex(parsed_emotion):
    n = len(parsed_emotion)
    r = 0
    g = 0
    b = 0
    for emo in parsed_emotion:
        color = EmotionColor.objects.get(emotion__exact=emo)
        r = r + color.r / n
        g = g + color.g / n
        b = b + color.b / n
    return rgb_to_hex(r, g, b)

@login_required
def experience_list(request):
    if request.user.is_authenticated and request.user.user_id == 'admin':
        exps = Experience.objects.order_by('exp_date')
    else:
        exps = Experience.objects.filter(author__exact=request.user.id).order_by('exp_date')
    emotion_color = EmotionColor.objects.all()
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
            parsed_emotion = exp.emotion.strip().split(',')
            exp.emotion_color = emo_to_hex(parsed_emotion)
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
            parsed_emotion = exp.emotion.strip().split(',')
            exp.emotion_color = emo_to_hex(parsed_emotion)
            exp.save()
            return redirect('experience_detail', pk=exp.pk)
    else:
        form = ExpForm(instance=exp)
    return render(request, 'MEgo/experience_edit.html', {'form': form})


class NewExpbyQView(CreateView):
    model = Experience
    template_name = 'MEgo/experience_edit.html'
    form_class = DynamicExpForm
    def get_form_kwargs(self):
        pk = self.kwargs['pk']
        q = None
        skipped_category = None

        if pk > 0:
            q = get_object_or_404(ExpQuestions, pk=pk)
        if q is not None:
            skipped_category = q.question_area

        kwargs = super(NewExpbyQView, self).get_form_kwargs()
        kwargs.update({'skipped_category': skipped_category})

        return kwargs

@login_required
def experience_new_by_question(request, pk):
    q = None
    if pk > 0 :
        q = get_object_or_404(ExpQuestions, pk=pk)
    if request.method == "POST":
        form = ExpForm(request.POST, request.FILES)
        if form.is_valid():
            exp = form.save(commit=False)
            exp.author = request.user
            parsed_emotion = exp.emotion.strip().split(',')
            exp.emotion_color = emo_to_hex(parsed_emotion)
            exp.save()
            return redirect('experience_detail', pk=exp.pk)
    else:
        if pk == 0:
            form = ExpForm() #normal case
        else:
            # fill form based on pre-defined tags
            form = DynamicExpForm(initial={q.question_area: q.related_tags})

    return render(request, 'MEgo/experience_edit.html', {'form': form})

@login_required
def analysis_report(request):
    return render(request, 'MEgo/analysis_report.html')

@login_required
def new_question(request):
    random_number = randint(1, ExpQuestions.objects.count() + 1)  # 전체 question 중 하나를 랜덤으로 숫자를 고름
    print(random_number)
    q = get_object_or_404(ExpQuestions, pk=random_number)
    print(q)
    return render(request, 'MEgo/new_question.html', {'q':q})


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

