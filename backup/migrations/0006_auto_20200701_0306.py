# Generated by Django 3.0.7 on 2020-06-30 18:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MEgo', '0005_auto_20200701_0108'),
    ]

    operations = [
        migrations.RenameField(
            model_name='lifeiwish',
            old_name='goal_of_the_year',
            new_name='goal_of_the_year_2020',
        ),
        migrations.RenameField(
            model_name='lifeiwish',
            old_name='life_values',
            new_name='goal_of_the_year_2030',
        ),
        migrations.RenameField(
            model_name='lifeiwish',
            old_name='priority',
            new_name='goal_of_the_year_2040',
        ),
        migrations.RemoveField(
            model_name='lifeiwish',
            name='year_of_the_goal',
        ),
        migrations.AddField(
            model_name='lifeiwish',
            name='goal_of_the_year_2050',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='lifeiwish',
            name='life_values_high',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='lifeiwish',
            name='life_values_low',
            field=models.TextField(blank=True, null=True),
        ),
    ]