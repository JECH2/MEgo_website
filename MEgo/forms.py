# forms to get user's input

from django import forms
from .models import *
from django.contrib.auth.forms import AuthenticationForm, UsernameField
from .color import emo_to_hex
from django_select2 import forms as s2forms
from django.utils.encoding import force_str

# For multi-page form using FormWizard
class ExpFormStepOne(forms.ModelForm):
    class Meta:
        model = Experience
        fields = ['exp_date', 'event', 'related_people', 'related_place', 'media_links']
        widgets = {
            'exp_date': forms.TextInput(attrs={'class':'custom-form','placeholder':'When did it happen?'}),
            'event' : forms.Textarea(attrs={'class':'custom-form','placeholder':'What happened?'}),
            'related_people': forms.TextInput(attrs={'class': 'custom-form', 'placeholder': 'Who were you with?(e.g. EunjinChoi, JiwonYoon, JiseongYang)'}),
            'related_place': forms.TextInput(attrs={'class': 'custom-form', 'placeholder': 'Where did it happen?'}),
            #'photo' : forms.ClearableFileInput(attrs={'class':'custom-fileInput-form'}),
            'media_links' : forms.TextInput(
                attrs={'class': 'custom-form',
                        'placeholder':'insert some medium(picture, video) as link : https://mego.pythonanywhere.com'}),
        }
        labels = {
            'exp_date':'Date',
            'event':'Event*',
            'related_people': 'People',
            'related_place': 'Location',
            'media_links': 'Videos',
        }
    def __init__(self, *args, **kwargs):
        super(ExpFormStepOne, self).__init__(*args, **kwargs)
        self.label_suffix = ''

# For multi-page form using FormWizard
class ExpFormStepTwo(forms.ModelForm):
    class Meta:
        model = Experience
        fields = ['thoughts','emotion']
        widgets = {
            'thoughts' : forms.Textarea(attrs={'class':'custom-form','placeholder':'What did you think'}),
            'emotion' : forms.CheckboxSelectMultiple(attrs={'class': 'custom-checkbox-form', 'placeholder':'How did you feel?'}, choices=[(item.emotion,emo_to_hex([item.emotion])) for item in EmotionColor.objects.all()]),
        }
        labels = {
            'thoughts':'Thoughts*',
            'emotion':'Emotions*',
        }
    def __init__(self, *args, **kwargs):
        super(ExpFormStepTwo, self).__init__(*args, **kwargs)
        self.label_suffix = ''

# For multi-page form using FormWizard
class ExpFormStepThree(forms.ModelForm):
    class Meta:
        IMPORTANCE_CHOICES = [('1','20px'), ('2','40px'), ('3', '60px')]
        model = Experience
        fields = ['importance']
        widgets = {
            'importance' : forms.RadioSelect(attrs={'class':'custom-radio-form'}, choices=IMPORTANCE_CHOICES),
        }
        labels = {
            'importance':'How important is this experience?*',
        }
    def __init__(self, *args, **kwargs):
        super(ExpFormStepThree, self).__init__(*args, **kwargs)
        self.label_suffix = ''

# This is tag selection widget, due to small bug, we will fix this and apply this in the future
class TestWidget(s2forms.ModelSelect2MultipleWidget):
    #model = EmotionColor
    queryset = EmotionColor.objects.only('emotion')
    #query_set = [item.emotion for item in EmotionColor.objects.all()]
    search_fields = [
        "emotion__icontains",
    ]

    def label_from_instance(self, obj):
        return force_str(obj.emotion)

# Form for LifeIWish : default
class LifeIWishForm(forms.ModelForm):
    class Meta:
        model = LifeIWish
        fields = ['life_values_high','life_values_low', 'ideal_person', 'life_goals',
                  'goal_of_the_year_2020', 'goal_of_the_year_2030', 'goal_of_the_year_2040', 'goal_of_the_year_2050']
        widgets = {
            'life_values_high': forms.Textarea(attrs={'class': 'custom-form', 'placeholder': 'What are the important things for you?'}),
            'life_values_low': forms.Textarea(attrs={'class': 'custom-form', 'placeholder': 'What things are not that important for you?'}),
            'ideal_person' : forms.Textarea(attrs={'class':'custom-form','placeholder':'what kind of person do you want to be?'}),
            'life_goals': forms.Textarea(attrs={'class':'custom-form','placeholder':'What kind of life do you want to live?\nWhat do you want to achieve?'}),
            'goal_of_the_year_2020' : forms.Textarea(attrs={'class':'custom-form','placeholder':'What do you want to achieve by 2020?'}),
            'goal_of_the_year_2030' : forms.Textarea(attrs={'class':'custom-form','placeholder':'How do you imagine yourself in 2030?'}),
            'goal_of_the_year_2040': forms.Textarea(attrs={'class': 'custom-form', 'placeholder': 'How do you imagine yourself in 2040?'}),
            'goal_of_the_year_2050': forms.Textarea(attrs={'class': 'custom-form', 'placeholder': 'How do you imagine yourself in 2050?'}),
        }
        labels = {
            'life_values_high':'High Priorities',
            'life_values_low': 'Low Priorities',
            'ideal_person': 'Your Ideal Person',
            'life_goals':'Whole Life',
            'goal_of_the_year_2020': 'Year 2020',
            'goal_of_the_year_2030': 'Year 2030',
            'goal_of_the_year_2040': 'Year 2040',
            'goal_of_the_year_2050': 'Year 2050',
        }
    def __init__(self, *args, **kwargs):
        super(LifeIWishForm, self).__init__(*args, **kwargs)
        self.label_suffix = ''

# Form for Experience : default
class ExpForm(forms.ModelForm):
    class Meta:
        IMPORTANCE_CHOICES = [('1','not important'), ('2','important'), ('3', 'very important')]
        model = Experience
        fields = ['exp_date', 'related_people','related_place','thumbnail_photo', 'media_links', 'event', 'thoughts','emotion','importance']
        widgets = {
            'exp_date': forms.TextInput(attrs={'class': 'custom-form', 'placeholder': 'When did it happen?'}),
            'related_people': forms.TextInput(attrs={'class': 'custom-form', 'placeholder': 'Who were you with?(e.g. EunjinChoi, JiwonYoon, JiseongYang)'}),
            'related_place': forms.TextInput(attrs={'class': 'custom-form', 'placeholder': 'Where did it happen?'}),
            'thumbnail_photo' : forms.ClearableFileInput(attrs={'class':'custom-fileInput-form'}),
            'media_links' : forms.TextInput(
                attrs={'class': 'custom-form',
                        'placeholder':'insert some medium(picture, video) as link : https://mego.pythonanywhere.com'}),
            'event' : forms.Textarea(attrs={'class':'custom-form','placeholder':'what happened to you'}),
            'thoughts' : forms.Textarea(attrs={'class':'custom-form','placeholder':'what did you think'}),
            #'emotion' : TestWidget,
            'emotion' : forms.TextInput(attrs={'class':'custom-form','placeholder':'how did you feel'}),
            'importance' : forms.RadioSelect(attrs={'class':'custom-radio-form-default'}, choices=IMPORTANCE_CHOICES),
        }
        labels = {
            'exp_date':'Date',
            'event':'Event*',
            'related_people': 'People',
            'related_place': 'Location',
            'thumbnail_photo':'Photo',
            'media_links': 'Videos',
            'thoughts': 'Thoughts*',
            'emotion': 'Emotions*',
            'importance': 'Importance*',
        }
    def __init__(self, *args, **kwargs):
        #self.skipped_category = kwargs.pop('skipped_category', None)
        super(ExpForm, self).__init__(*args, **kwargs)
        #self.fields[self.skipped_category].widget = forms.HiddenInput()
        self.label_suffix = ''

# Form for LifeIWish : dynamic
# only selecte field will be shown to user
class DynamicLifeIWishForm(forms.ModelForm):
    class Meta:
        model = LifeIWish
        fields = ['author','life_values_high','life_values_low', 'ideal_person', 'life_goals',
                  'goal_of_the_year_2020', 'goal_of_the_year_2030', 'goal_of_the_year_2040', 'goal_of_the_year_2050']
        widgets = {
            'life_values_high': forms.Textarea(attrs={'class': 'custom-form', 'placeholder': 'What are the important things for you?'}),
            'life_values_low': forms.Textarea(attrs={'class': 'custom-form', 'placeholder': 'What things are not that important for you?'}),
            'ideal_person' : forms.Textarea(attrs={'class':'custom-form','placeholder':'What kind of person do you want to be?'}),
            'life_goals': forms.Textarea(attrs={'class':'custom-form','placeholder':'What kind of life do you want to live?\nWhat do you want to achieve?'}),
            'goal_of_the_year_2020' : forms.Textarea(attrs={'class':'custom-form','placeholder':'What do you want to achieve by 2020?'}),
            'goal_of_the_year_2030' : forms.Textarea(attrs={'class':'custom-form','placeholder':'How do you imagine yourself in 2030?'}),
            'goal_of_the_year_2040': forms.Textarea(attrs={'class': 'custom-form', 'placeholder': 'How do you imagine yourself in 2040?'}),
            'goal_of_the_year_2050': forms.Textarea(attrs={'class': 'custom-form', 'placeholder': 'How do you imagine yourself in 2050?'}),
        }
        labels = {
            'life_values_high':'High Priorities',
            'life_values_low': 'Low Priorities',
            'ideal_person': 'Your Ideal Person',
            'life_goals':'Whole Life',
            'goal_of_the_year_2020': 'Year 2020',
            'goal_of_the_year_2030': 'Year 2030',
            'goal_of_the_year_2040': 'Year 2040',
            'goal_of_the_year_2050': 'Year 2050',
        }
    def __init__(self, *args, **kwargs):
        self.only_see_category = kwargs.pop('only_see_category', None)
        super(DynamicLifeIWishForm, self).__init__(*args, **kwargs)
        for field in self.fields:
            if field != self.only_see_category:
                print(field, self.only_see_category)
                self.fields[field].widget = forms.HiddenInput()
        self.label_suffix = ''

# Form for Experience : dynamic
# selecte field will not be shown to user, but we don't use this right now,
# to enable user modify the pre-input words
class DynamicExpForm(forms.ModelForm):
    class Meta:
        IMPORTANCE_CHOICES = [('1', 'not important'), ('2', 'important'), ('3', 'very important')]
        model = Experience
        fields = ['author','photo', 'thumbnail_photo', 'media_links', 'event', 'thoughts','emotion','importance']
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
    class Meta:
        AGE_CHOICES = [(age, age) for age in range(20, 66)]  # 20 ~ 65
        GENDER_CHOICES = [('M', 'Male'),('W', 'Female')]

        model = User
        fields = ['nickname', 'email', 'age', 'gender', 'user_id','password']
        widgets = {
            'nickname': forms.TextInput(attrs={'class': 'custom-form', 'placeholder':'input within 15 characters', 'autofocus': True}),
            'age': forms.Select(attrs={'class': 'custom-form'}, choices=AGE_CHOICES),
            'email': forms.EmailInput(attrs={'class': 'custom-form', 'placeholder':'example : mego.kaist@gmail.com'}),
            'gender': forms.RadioSelect(attrs={'class': 'custom-radio-form-default'}, choices=GENDER_CHOICES),
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