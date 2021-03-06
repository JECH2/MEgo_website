# Generated by Django 3.0.7 on 2020-06-26 17:06

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('MEgo', '0002_auto_20200627_0148'),
    ]

    operations = [
        migrations.CreateModel(
            name='EmotionColor',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('color_name', models.CharField(max_length=200)),
                ('emotion', models.CharField(max_length=200)),
                ('r', models.IntegerField()),
                ('g', models.IntegerField()),
                ('b', models.IntegerField()),
                ('a', models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name='LifeIWish',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('input_date', models.DateTimeField(default=django.utils.timezone.now)),
                ('life_values', models.TextField(blank=True, max_length=200, null=True)),
                ('priority', models.TextField(blank=True, max_length=200, null=True)),
                ('ideal_person', models.TextField(blank=True, max_length=200, null=True)),
                ('life_goals', models.TextField(blank=True, max_length=200, null=True)),
                ('year_of_the_goal', models.CharField(blank=True, max_length=200, null=True)),
                ('goal_of_the_year', models.TextField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='LifeQuestions',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.CharField(max_length=200)),
                ('question_area', models.CharField(max_length=200)),
                ('related_tags', models.CharField(max_length=200)),
                ('answer_area', models.CharField(max_length=200)),
            ],
        ),
    ]
