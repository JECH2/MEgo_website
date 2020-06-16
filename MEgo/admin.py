from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import Experience
from .models import User

admin.site.register(Experience)

@admin.register(User)
class UserAdmin(admin.ModelAdmin):

    list_display = (
        'nickname',
        'email',
        'date_joined',
    )

    list_display_links = (
        'nickname',
        'email',
    )
