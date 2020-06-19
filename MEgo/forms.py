from django import forms
from .models import User, UserManager
from .models import Experience

class ExpForm(forms.ModelForm):

    class Meta:
        IMPORTANCE_CHOICES = [('1','not important'), ('2','important'), ('3', 'very important')]
        model = Experience
        fields = ['exp_date', 'photo', 'thumbnail_photo', 'media_links', 'event', 'thoughts','emotion','importance']
        widgets = {
            #'exp_date' : forms.SplitDateTimeWidget({'class':'form-control'}),
            'media_links' : forms.TextInput(attrs={'class': 'form-control', 'placeholder':'insert some medium(picture, video) as link : https://mego.pythonanywhere.com'}),
            'event' : forms.TextInput(attrs={'class':'form-control','placeholder':'what happened to you'}),
            'thoughts' : forms.TextInput(attrs={'class':'form-control','placeholder':'what did you think'}),
            'emotion' : forms.TextInput(attrs={'class':'form-control','placeholder':'how did you feel'}),
            'importance' : forms.RadioSelect(attrs=None, choices=IMPORTANCE_CHOICES),
        }

class UserForm(forms.ModelForm):
    # # widget을 오버라이드하여 입력될때 *표시를 찍어준다
    # password = forms.CharField(label='Password', widget=forms.PasswordInput)
    # password2 = forms.CharField(label='Password2', widget=forms.PasswordInput)

    class Meta:
        AGE_CHOICES = [(age, age) for age in range(20, 66)]  # 20 ~ 65
        GENDER_CHOICES = [('M', 'Male'),('W', 'Female')]

        model = User
        fields = ['nickname', 'email', 'age', 'gender', 'password']
        widgets = {
            'nickname': forms.TextInput(attrs={'class': 'form-control', 'placeholder':'input within 15 characters'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder':'example : mego.kaist@gmail.com'}),
            'password' : forms.PasswordInput(attrs={'class': 'form-control','placeholder':'within 15 characters'}),
            'age' : forms.Select(attrs={'class': 'form-control'}, choices=AGE_CHOICES),
            'gender' : forms.RadioSelect(attrs=None, choices=GENDER_CHOICES)
        }
    # 글자수 제한
    def __init__(self, *args, **kwargs):
        super(UserForm, self).__init__( *args, **kwargs)
        self.fields['nickname'].widget.attrs['maxlength'] = 150

    def save(self, commit=True):
        # Save the provided password in hashed format
        user = super(UserForm, self).save(commit=False)
        user.email = UserManager.normalize_email(self.cleaned_data['email'])
        user.set_password(self.cleaned_data["password"])
        if commit:
            user.save()
        return user

