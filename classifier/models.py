import pytz
from datetime import datetime

from django.db import models

# Create your models here.

CITY_TIMEZONE = {
    'Москва': pytz.timezone('Europe/Moscow'),
    'Питер': pytz.timezone('Europe/Moscow'),
    'Нижний': pytz.timezone('Europe/Moscow')
}

class Camera(models.Model):
    CITY = (
        (0, 'Москва'),
        (1, 'Питер'),
        (2, 'Нижний'),
    )

    ip_adress = models.CharField(blank=False, max_length=124, \
            verbose_name='IP Адрес')
    mask = models.ImageField(upload_to='masks/', verbose_name='Маска')
    city = models.PositiveSmallIntegerField(choices=CITY, default=0, \
            verbose_name='Город')
    adress = models.CharField(default='', max_length=256, verbose_name='Адрес')
    open_link = models.CharField(blank=False, max_length=124, \
            verbose_name='Ссылка на открытие')
    active = models.BooleanField(default=False, verbose_name='Активна',
        help_text='Считываются ли данные при работе программы')
    seconds = models.PositiveSmallIntegerField(default=1, \
            verbose_name='Частота считывания (секунд)', \
            help_text='При установке параметра учитывать \
            частоту траффика на конкртеном шлагбауме')

    min_width = models.IntegerField(default=150, \
                    verbose_name='Минимальная ширина автомобиля')
    min_height = models.IntegerField(default=150, \
                    verbose_name='Минимальная высота автомобиля')

    class Meta:
        verbose_name = 'Камера'
        verbose_name_plural = 'Камеры'

    def __str__(self):
        return str(self.get_city_display()) + '. ' + str(self.adress)

class Shot(models.Model):
    CLASSES = (
        (0, 'Скорая'),
        (1, 'Гражданское'),
        (2, 'Полицейская'),
        (3, 'Аварийная'),
        (4, 'Пожарная'),
        (5, 'Трактор'),
        (6, 'Мусоровоз'),
    )

    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, \
            verbose_name='Камера')
    image = models.ImageField(upload_to='', verbose_name='Полное изображение')
    car = models.ImageField(upload_to='', verbose_name='Само авто')
    timestamp = models.CharField(max_length=256, editable=False, \
            verbose_name='Информация')
    type = models.PositiveSmallIntegerField(choices=CLASSES, default=0, \
            verbose_name='Класс')
    wrong_label = models.BooleanField(default=False, \
            verbose_name='Ошибка в классификации', \
            help_text='При дообучении модели поставить \
            флаг и изменить класс на верный')

    def save(self, *args, **kwargs):
        self.timestamp = self.generate_name()
        super(Shot, self).save(*args, **kwargs)

    def generate_name(self):
        timezone = CITY_TIMEZONE[self.camera.get_city_display()]
        time_ = datetime.strftime(datetime.now(timezone), \
                    '%d.%m.20%y %H:%M')
        return str(self.camera) + '. ' + time_

    class Meta:
        verbose_name = 'Машина'
        verbose_name_plural = 'Машины'

    def __str__(self):
        return self.timestamp
