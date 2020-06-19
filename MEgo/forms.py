from django import forms
from .models import User
from .models import Experience

class ExpForm(forms.ModelForm):

    class Meta:
        model = Experience
        fields = ('exp_date', 'media_links', 'event', 'thoughts','emotion','importance')