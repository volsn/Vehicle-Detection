# Generated by Django 2.2.7 on 2020-10-12 13:36

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0013_delete_options'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Image',
            new_name='Shot',
        ),
        migrations.AlterModelOptions(
            name='shot',
            options={'verbose_name': 'Скорая', 'verbose_name_plural': 'Скорые'},
        ),
    ]
