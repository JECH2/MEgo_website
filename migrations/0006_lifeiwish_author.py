# Generated by Django 3.0.7 on 2020-07-01 11:10

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('MEgo', '0005_auto_20200701_1941'),
    ]

    operations = [
        migrations.AddField(
            model_name='lifeiwish',
            name='author',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]