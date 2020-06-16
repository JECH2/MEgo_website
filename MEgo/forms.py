from django import forms
from .models import User
from .models import Experience

class ExpForm(forms.ModelForm):

    class Meta:
        model = Experience
        fields = ('exp_date','experience', 'thoughts','emotion','emotion_intensity','importance')
