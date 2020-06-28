# forms to get user's input

from django import forms
from .models import User, UserManager
from .models import Experience, EmotionColor
from django.contrib.auth.forms import AuthenticationForm, UsernameField
from .color import emo_to_hex

class ExpFormStepOne(forms.ModelForm):
    class Meta:
        model = Experience
        fields = ['exp_date', 'event', 'related_people', 'related_place', 'media_links']
        widgets = {
            'exp_date': forms.TextInput(attrs={'class':'custom-form','placeholder':'When did it happen?'}),
            'event' : forms.TextInput(attrs={'class':'custom-form','placeholder':'What happened?'}),
            'related_people': forms.TextInput(attrs={'class': 'custom-form', 'placeholder': 'Who were you with?'}),
            'related_place': forms.TextInput(attrs={'class': 'custom-form', 'placeholder': 'Where did it happen?'}),
            #'photo' : forms.ClearableFileInput(attrs={'class':'custom-fileInput-form'}),
            'media_links' : forms.TextInput(
                attrs={'class': 'custom-form',
                        'placeholder':'insert some medium(picture, video) as link : https://mego.pythonanywhere.com'}),
        }
        labels = {
            'exp_date':'Date',
            'event':'Event',
            'related_people': 'People',
            'related_place': 'Location',
            'media_links': 'Videos',
        }
    def __init__(self, *args, **kwargs):
        super(ExpFormStepOne, self).__init__(*args, **kwargs)
        self.label_suffix = ''

class ExpFormStepTwo(forms.ModelForm):
    class Meta:
        model = Experience
        fields = ['thoughts','emotion']
        widgets = {
            'thoughts' : forms.TextInput(attrs={'class':'custom-form','placeholder':'What did you think'}),
            'emotion' : forms.CheckboxSelectMultiple(attrs={'class': 'custom-checkbox-form', 'placeholder':'How did you feel?'}, choices=[(item.emotion,emo_to_hex([item.emotion])) for item in EmotionColor.objects.all()]),
        }
        labels = {
            'thoughts':'Thoughts',
            'emotion':'Emotions',
        }
    def __init__(self, *args, **kwargs):
        super(ExpFormStepTwo, self).__init__(*args, **kwargs)
        self.label_suffix = ''

class ExpFormStepThree(forms.ModelForm):
    class Meta:
        IMPORTANCE_CHOICES = [('1','20px'), ('2','40px'), ('3', '60px')]
        model = Experience
        fields = ['importance']
        widgets = {
            'importance' : forms.RadioSelect(attrs={'class':'custom-radio-form'}, choices=IMPORTANCE_CHOICES),
        }
        labels = {
            'importance':'How important is this experience?',
        }
    def __init__(self, *args, **kwargs):
        super(ExpFormStepThree, self).__init__(*args, **kwargs)
        self.label_suffix = ''


class ExpForm(forms.ModelForm):
    class Meta:
        IMPORTANCE_CHOICES = [('1','not important'), ('2','important'), ('3', 'very important')]
        model = Experience
        fields = ['exp_date', 'related_people','related_place','thumbnail_photo', 'media_links', 'event', 'thoughts','emotion','importance']
        widgets = {
            'exp_date': forms.TextInput(attrs={'class': 'custom-form', 'placeholder': 'When did it happen?'}),
            'related_people': forms.TextInput(attrs={'class': 'custom-form', 'placeholder': 'Who were you with?'}),
            'related_place': forms.TextInput(attrs={'class': 'custom-form', 'placeholder': 'Where did it happen?'}),
            'thumbnail_photo' : forms.ClearableFileInput(attrs={'class':'custom-fileInput-form'}),
            'media_links' : forms.TextInput(
                attrs={'class': 'custom-form',
                        'placeholder':'insert some medium(picture, video) as link : https://mego.pythonanywhere.com'}),
            'event' : forms.TextInput(attrs={'class':'custom-form','placeholder':'what happened to you'}),
            'thoughts' : forms.TextInput(attrs={'class':'custom-form','placeholder':'what did you think'}),
            'emotion' : forms.TextInput(attrs={'class':'custom-form','placeholder':'how did you feel'}),
            'importance' : forms.RadioSelect(attrs={'class':'custom-radio-form-default'}, choices=IMPORTANCE_CHOICES),
        }
        labels = {
            'exp_date':'Date',
            'event':'Event',
            'related_people': 'People',
            'related_place': 'Location',
            'thumbnail_photo':'Photo',
            'media_links': 'Videos',
            'thoughts': 'Thoughts',
            'emotion': 'Emotions',
            'importance': 'Importance',
        }
    def __init__(self, *args, **kwargs):
        #self.skipped_category = kwargs.pop('skipped_category', None)
        super(ExpForm, self).__init__(*args, **kwargs)
        #self.fields[self.skipped_category].widget = forms.HiddenInput()
        self.label_suffix = ''


class DynamicExpForm(forms.ModelForm):
    class Meta:
        IMPORTANCE_CHOICES = [('1', 'not important'), ('2', 'important'), ('3', 'very important')]
        model = Experience
        fields = ['photo', 'thumbnail_photo', 'media_links', 'event', 'thoughts','emotion','importance']
        widgets = {
            'event': forms.TextInput(attrs={'class': 'custom-form', 'placeholder': 'what happened to you?'}),
            'thoughts': forms.TextInput(attrs={'class': 'custom-form', 'placeholder': 'what did you think?'}),
            'emotion': forms.TextInput(attrs={'class': 'custom-form', 'placeholder': 'how did you feel?'}),
            'importance': forms.RadioSelect(attrs={'class': 'custom-radio-form-default'}, choices=IMPORTANCE_CHOICES),
            'photo' : forms.ClearableFileInput(attrs={'class':'custom-fileInput-form'}),
            'thumbnail_photo' : forms.ClearableFileInput(attrs={'class':'custom-fileInput-form'}),
            'media_links' : forms.TextInput(
                attrs={'class': 'custom-form',
                        'placeholder':'insert some medium(picture, video) as link : https://mego.pythonanywhere.com'}),
        }

    def __init__(self, *args, **kwargs):
        #self.skipped_category = kwargs.pop('skipped_category', None)
        super(DynamicExpForm, self).__init__(*args, **kwargs)
        #self.fields[self.skipped_category].widget = forms.HiddenInput()
        self.label_suffix = ''

# user signup form
class UserForm(forms.ModelForm):
    # # widget을 오버라이드하여 입력될때 *표시를 찍어준다
    # password = forms.CharField(label='Password', widget=forms.PasswordInput)
    # password2 = forms.CharField(label='Password2', widget=forms.PasswordInput)

    class Meta:
        AGE_CHOICES = [(age, age) for age in range(20, 66)]  # 20 ~ 65
        GENDER_CHOICES = [('M', 'Male'),('W', 'Female')]

        model = User
        fields = ['nickname', 'email', 'age', 'gender', 'user_id','password']
        widgets = {
            'nickname': forms.TextInput(attrs={'class': 'custom-form', 'placeholder':'input within 15 characters', 'autofocus': True}),
            'age': forms.Select(attrs={'class': 'custom-form'}, choices=AGE_CHOICES),
            'email': forms.EmailInput(attrs={'class': 'custom-form', 'placeholder':'example : mego.kaist@gmail.com'}),
            'gender': forms.RadioSelect(attrs={'class': 'custom-radio-form'}, choices=GENDER_CHOICES),
            'user_id': forms.TextInput(attrs={'class': 'custom-form', 'placeholder': 'input within 15 characters'}),
            'password' : forms.PasswordInput(attrs={'class': 'custom-form','placeholder':'within 15 characters'}),
        }
        labels = {
            'nickname': 'Name',
            'age':'Age',
            'email': 'Email',
            'gender': 'Gender',
            'user_id': 'ID',
            'password': 'PW',
        }

    # 글자수 제한
    def __init__(self, *args, **kwargs):
        super(UserForm, self).__init__( *args, **kwargs)
        self.fields['nickname'].widget.attrs['maxlength'] = 20
        self.fields['user_id'].widget.attrs['maxlength'] = 20
        self.label_suffix = ''

    def save(self, commit=True):
        # Save the provided password in hashed format
        user = super(UserForm, self).save(commit=False)
        user.email = UserManager.normalize_email(self.cleaned_data['email'])
        user.set_password(self.cleaned_data["password"])
        if commit:
            user.save()
        return user

# custom user login form
class CustomUserLoginForm(AuthenticationForm):
    username = UsernameField(
        label='ID',
        label_suffix='',
        widget=forms.TextInput(attrs={'autofocus': True}),
    )
    password = forms.CharField(
        label='PW',
        label_suffix = '',
        widget=forms.PasswordInput(),
    )