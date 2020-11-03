FROM python:3.8
ENV PYTHONUNBUFFERED=1
RUN mkdir /code
WORKDIR /code
RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\ 
	'libsm6'\ 
    'libxext6'  -y
RUN apt-get install libgl1-mesa-glx -y

COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/
EXPOSE 8000
RUN python manage.py makemigrations
RUN python manage.py migrate
RUN python manage.py loaddata data.json
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
