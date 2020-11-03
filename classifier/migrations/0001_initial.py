# Generated by Django 2.2.7 on 2020-10-12 07:13

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Camera',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ip_adress', models.CharField(max_length=124)),
                ('city', models.PositiveSmallIntegerField(choices=[(0, 'Москва'), (1, 'Питер'), (2, 'Нижний')], default=0)),
                ('open_link', models.CharField(max_length=124)),
            ],
        ),
    ]
