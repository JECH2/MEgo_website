# Generated by Django 3.0.7 on 2020-06-16 15:44

import MEgo.models
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('auth', '0011_update_proxy_permissions'),
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('password', models.CharField(max_length=128, verbose_name='password')),
                ('last_login', models.DateTimeField(blank=True, null=True, verbose_name='last login')),
                ('email', models.EmailField(max_length=255, unique=True)),
                ('nickname', models.CharField(max_length=20, unique=True)),
                ('is_active', models.BooleanField(default=True)),
                ('is_admin', models.BooleanField(default=False)),
                ('is_superuser', models.BooleanField(default=False)),
                ('is_staff', models.BooleanField(default=False)),
                ('date_joined', models.DateTimeField(auto_now_add=True)),
                ('groups', models.ManyToManyField(blank=True, help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.', related_name='user_set', related_query_name='user', to='auth.Group', verbose_name='groups')),
                ('user_permissions', models.ManyToManyField(blank=True, help_text='Specific permissions for this user.', related_name='user_set', related_query_name='user', to='auth.Permission', verbose_name='user permissions')),
            ],
            options={
                'abstract': False,
            },
            managers=[
                ('objects', MEgo.models.UserManager()),
            ],
        ),
        migrations.CreateModel(
            name='Experience',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('input_date', models.DateTimeField(default=django.utils.timezone.now)),
                ('exp_date', models.DateTimeField(blank=True, null=True)),
                ('walks', models.IntegerField(blank=True, null=True)),
                ('sleep', models.FloatField(blank=True, null=True)),
                ('deep_sleep', models.FloatField(blank=True, null=True)),
                ('heartbeat', models.IntegerField(blank=True, null=True)),
                ('calorie', models.IntegerField(blank=True, null=True)),
                ('distance', models.FloatField(blank=True, null=True)),
                ('Experience', models.TextField()),
                ('thoughts', models.TextField(blank=True, null=True)),
                ('emotion', models.TextField()),
                ('emotion_intensity', models.IntegerField()),
                ('importance', models.IntegerField()),
                ('future', models.IntegerField(default=0)),
                ('related_people', models.TextField(blank=True, null=True)),
                ('related_place', models.TextField(blank=True, null=True)),
                ('author', models.ForeignKey(default='admin', on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
