# Generated by Django 2.2.7 on 2020-10-12 08:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0006_image_timestamp'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='image',
            field=models.ImageField(upload_to='', verbose_name='Изображение'),
        ),
    ]
