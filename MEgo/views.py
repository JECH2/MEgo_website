# views.py manages page rendering : dynamic!

from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from .models import *
from .forms import *
from django.urls import reverse_lazy
from django.views.generic.edit import CreateView

from formtools.wizard.views import SessionWizardView

from random import randint
from .color import emo_to_hex

# for rendering page of experience circle
@login_required
def experience_circle(request):
    if request.user.is_authenticated and request.user.user_id == 'admin':
        return experience_as_list(request)
    else:
        exps = Experience.objects.filter(author__exact=request.user.id).order_by('exp_date')
    return render(request, 'MEgo/experience_circle.html', {'exps':exps})


# for rendering page of experience as list
@login_required
def experience_as_list(request):
    if request.user.is_authenticated and request.user.user_id == 'admin':
        exps = Experience.objects.order_by('exp_date')
    else:
        exps = Experience.objects.filter(author__exact=request.user.id).order_by('exp_date').reverse()
    return render(request, 'MEgo/experience_as_list.html', {'exps':exps})

# for rendering page of seeing detail of experience
@login_required
def experience_detail(request, pk):
    exp = get_object_or_404(Experience, pk=pk)
    return render(request, 'MEgo/experience_detail.html', {'exp':exp})

# for rendering page of recording new experience
@login_required
def experience_new(request):
    # when user clicked submit button
    if request.method == "POST":
        form = ExpForm(request.POST, request.FILES)
        if form.is_valid():
            exp = form.save(commit=False)
            exp.author = request.user
            # emotion_color is added directly from emotion
            parsed_emotion = exp.emotion.strip().split(',')
            exp.emotion_color = emo_to_hex(parsed_emotion)

            exp.save()
            return redirect('experience_detail', pk=exp.pk)
    else:
        # for the first time
        form = ExpForm()
    return render(request, 'MEgo/experience_edit.html', {'form': form})


class ExpFormWizardView(SessionWizardView):
    form_list = [ExpFormStepOne, ExpFormStepTwo, ExpFormStepThree]

    def get_template_names(self):
        return ['Diary/step_{0}_template.html'.format(self.steps.current)]

    def done(self, form_list, **kwargs):
        exp = Experience()
        for form in form_list:
            for field, value in form.cleaned_data.items():
                #print(field, value)
                setattr(exp, field, value)
        exp.author = self.request.user
        # emotion_color is added directly from emotion
        parsed_emotion = exp.emotion.strip().split(',')
        exp.emotion_color = emo_to_hex(parsed_emotion)
        exp.save()
        return redirect('/')

# for rendering page of editing existing experience
@login_required
def experience_edit(request, pk):
    exp = get_object_or_404(Experience, pk=pk)
    if request.method == "POST":
        # when user clicked submit button
        form = ExpForm(request.POST, request.FILES, instance=exp)
        if form.is_valid():
            exp = form.save(commit=False)
            exp.author = request.user
            # emotion_color is added directly from emotion
            parsed_emotion = exp.emotion.strip().split(',')
            exp.emotion_color = emo_to_hex(parsed_emotion)
            exp.save()
            return redirect('experience_detail', pk=exp.pk)
    else:
        # for the first time
        form = ExpForm(instance=exp)
    return render(request, 'MEgo/experience_edit.html', {'form': form})

# view of experience editing from question
# another type of view : not function view, class view
# for dynamic field change
@login_required
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

# for rendering another view of experience by question
# when user choose question,
# related field with the question is automatically filled in the form
@login_required
def experience_new_by_question(request, qt, pk):
    q = None
    if qt:
        if pk:
            q = get_object_or_404(LifeQuestions, pk=pk)
            if request.method == "POST":
                form = LifeIWishForm(request.POST, request.FILES)
                if form.is_valid():
                    lfv = form.save(commit=False)
                    lfv.author = request.user
                    lfv.save()
                    return redirect('lfv_detail', pk=lfv.pk)
            else:
                # fill form based on pre-defined tags
                form = LifeIWishForm(initial={q.question_area: q.related_tags})
    else:
        if pk:
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
                form = ExpForm(initial={q.question_area: q.related_tags})

    return render(request, 'MEgo/experience_edit.html', {'form': form})

@login_required
def life_value_detail(request, pk):
    lfv = get_object_or_404(LifeIWish, pk=pk)
    return render(request, 'MEgo/life_I_wish_detail.html', {'lfv': lfv})

# for rendering page of delete a experience
@login_required
def experience_delete(request, pk):
    exp = get_object_or_404(Experience, pk=pk)
    exp.delete()
    return redirect('experience_circle')

# for rendering page of social map page
@login_required
def social_map(request):
    return render(request, 'MEgo/social_map.html')

# for rendering page of analysis report page
@login_required
def analysis_report(request):
    return render(request, 'MEgo/analysis_report.html')

# for rendering page of getting new question when skip button or page refresh is clicked
@login_required
def new_question(request):
    question_type = randint(0, 1)  # 0 = daily 1 = life questions
    if question_type: # 1 = life questions
        first = LifeQuestions.objects.order_by('pk')[0].pk
        end = first + LifeQuestions.objects.count() - 1
        random_number = randint(first, end)  # 전체 question 중 하나를 랜덤으로 숫자를 고름
        q = get_object_or_404(LifeQuestions, pk=random_number)
    else:
        first = ExpQuestions.objects.order_by('pk')[0].pk
        end = first + ExpQuestions.objects.count() - 1
        random_number = randint(first, end)  # 전체 question 중 하나를 랜덤으로 숫자를 고름
        q = get_object_or_404(ExpQuestions, pk=random_number)
    return render(request, 'MEgo/new_question.html', {'q':q, 'qt':question_type})

# for rendering page of support page
def support(request):
    return render(request, 'support.html')

# for rendering page of sign up
def signup(request):
    if request.method == 'POST':
        form = UserForm(request.POST)
        if form.is_valid():
            print('is valid form')
            print(form)
            user_instance = form.save()
            login(request, user_instance)
        return render(request, 'registration/signup_complete.html', {'id' : id })
    else:
        form = UserForm()

    return render(request, 'registration/signup.html', {'form': form})

