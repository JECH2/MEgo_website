# Generated by Django 3.0.7 on 2020-06-16 12:24

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Event',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('input_date', models.DateTimeField(default=django.utils.timezone.now)),
                ('exp_date', models.DateTimeField(blank=True, null=True)),
                ('walks', models.IntegerField()),
                ('sleep', models.FloatField()),
                ('deep_sleep', models.FloatField()),
                ('heartbeat', models.IntegerField()),
                ('calorie', models.IntegerField()),
                ('distance', models.FloatField()),
                ('event', models.TextField()),
                ('thoughts', models.TextField()),
                ('emotion', models.TextField()),
                ('emotion_intensity', models.IntegerField()),
                ('importance', models.IntegerField()),
                ('future', models.IntegerField()),
                ('related_people', models.TextField()),
                ('related_place', models.TextField()),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
