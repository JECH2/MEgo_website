from django.shortcuts import render, get_object_or_404
from django.utils import timezone
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from .models import Experience
from .forms import ExpForm
from django.urls import reverse_lazy
from django.views.generic.edit import CreateView

# Create your views here.
def experience_list(request):
    exps = Experience.objects.filter(exp_date__lte=timezone.now()).order_by('exp_date')
    return render(request, 'MEgo/experience_list.html', {'exps':exps})

@login_required
def experience_detail(request, pk):
    exp = get_object_or_404(Experience, pk=pk)
    return render(request, 'MEgo/experience_detail.html', {'exp':exp})

@login_required
def experience_new(request):
    # 폼에 데이터를 받은 상황
    if request.method == "POST":
        form = ExpForm(request.POST)
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
        form = ExpForm(request.POST, instance=exp)
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
