from django.shortcuts import render
from django.http.response import StreamingHttpResponse, HttpResponse
from classifier.models import Camera, Shot

from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files.base import ContentFile
from django.conf import settings
from django.contrib.auth.decorators import login_required

import tensorflow as tf
from tensorflow import keras
import numpy as np
import threading
import cv2
import os
import requests
import time
from PIL import Image
import pickle
import random
import string
import imagehash


class VehicleDetector:
    def __init__(self, config):
        self.min_confidence = config['min_confidence']
        self.threshold = config['threshold']
        self.classes = config['labels']
        self.min_sizes = config['min_sizes']
        self.add_margin = True

    def fit(self, yolo_path):
        labelsPath = os.path.sep.join([yolo_path, "coco.names"])
        self.LABELS = open(labelsPath).read().strip().split("\n")

        weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
        configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])

        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in \
            self.net.getUnconnectedOutLayers()]

    def predict(self, image):
        return self._get_objects(image)

    def predict_proba(self, image):
        return self._get_objects(image, add_proba=True)

    def _get_objects(self, image, add_proba=False):

        image = image.copy()
        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.min_confidence:

                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, \
                self.min_confidence, self.threshold)

        output = []

        if len(idxs) > 0:
            for i in idxs.flatten():
                if self.LABELS[classIDs[i]] in self.classes:
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    if self.add_margin:
                        y -= 50
                        x -= 50
                        w += 50
                        h += 50

                    if self.min_sizes is not None:
                        if w < self.min_sizes['width'] \
                                or h < self.min_sizes['height']:
                            continue

                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    if x+w > W:
                        w = W - x
                    if y+h > H:
                        h = H - y

                    if add_proba:
                        output.append(((y, h, x, w), self.LABELS[classIDs[i]], \
                                confidences[i]))
                    else:
                        output.append(((y, h, x, w), self.LABELS[classIDs[i]]))

        return output


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def save_image(camera, image, roi, label):
    filename = 'temp/{}.jpg'.format(get_random_string(10))
    filename_roi = 'temp/{}.jpg'.format(get_random_string(10))
    cv2.imwrite(filename, image)
    cv2.imwrite(filename_roi, roi)
    shot = Shot(camera=camera, type=label)
    shot.image.save(
        shot.generate_name() + '.jpg',
        open(filename, 'rb')
    )
    shot.car.save(
        shot.generate_name() + '.jpg',
        open(filename_roi, 'rb')
    )
    shot.save()

    os.remove(filename)
    os.remove(filename_roi)

# Create your views here.

def is_car_unique(img):
    n_last_cars = settings.N_LAST_CARS
    cutoff = settings.CUTOFF

    last_shots = Shot.objects.filter().order_by('-id')[:n_last_cars]
    hash_img = imagehash.average_hash(
        Image.fromarray(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )
    )
    for shot in last_shots:
        if shot.car:
            hash_car = imagehash.average_hash(Image.open(shot.car))
            print('[HASH]', hash_img - hash_car)
            if hash_img - hash_car < cutoff:
                return False

    return True

def test_similar(request, pk):
    shot = Shot.objects.get(pk=pk)
    img = Image.open(shot.car)
    return HttpResponse(is_car_unique(img))

class Logger:
    def __call__(self, error_message):
        print(error_message)

logger = Logger()
class RTSPReader:
    def __init__(self, link, frequency_seconds, reload_on_errors=True, reload_after_n_frames=None):
        self.link = link
        self.frequency_seconds = frequency_seconds
        self.reload_on_errors = reload_on_errors
        self.reload_after_n_frames = reload_after_n_frames
        self.counter = 0
        self.cap = None

    def __iter__(self):
        self.connect_to_camera()
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frequency_frames = self.fps * self.frequency_seconds
        return self

    def __next__(self):
        self.counter += 1

        if self.reload_after_n_frames:
            if self.counter % self.reload_after_n_frames == 0:
                self.connect_to_camera()

        if self.counter % self.frequency_frames == 0:
            rate, frame = self.cap.read()
            if not rate:
                if self.reload_on_errors:
                    self.connect_to_camera()
                    logger('[Error] Ignored')
                else:
                    logger('[Error] Can\'t read from {}'.format(self.link))
                return None
            return frame

    def connect_to_camera(self):
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.link)

        if not self.cap.isOpened():
            logger('[Error] Unable to connect to {}'.format(self.link))

def define_model():
    base_model = tf.keras.applications.MobileNetV2(weights=None, include_top=False)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(1024, activation='elu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dropout(rate=0.25)(x)
    x = tf.keras.layers.Dense(1024, activation='elu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dropout(rate=0.25)(x)
    x = tf.keras.layers.Dense(512, activation='elu', kernel_initializer='he_normal')(x)
    output = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=[base_model.input,], outputs=[output,])
    return model

def load_model(file):
    if file.endswith('.pickle'):
        with open(file, 'rb') as file:
            model = pickle.load(file)

        return model, 'sklearn'
    elif file.endswith('weights.h5'):
        print(tf.__version__)
        print(keras.__version__)
        print('test loading model')
        model = define_model()
        print(model)
        model.load_weights(file)
        return model, 'tf'
    elif file.endswith('.h5'):
        print(os.listdir())
        model = tf.keras.models.load_model(file)
        return model, 'tf'
    else:
        logger('[Error] Unsuported model type')
        return None

def sklearn_pipeline(image):
    image = cv2.resize(image, (64, 64))
    W, H, C = image.shape
    image = image.reshape((W * H * C))
    image = np.array([image]).astype(np.float32)
    return image

def tensorflow_pipeline(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    return np.array([image])

def predict(model, image, type_):
    if type_ == 'sklearn':
        image = sklearn_pipeline(image)
        return model.predict(image)
    elif type_ == 'tf':
        image = tensorflow_pipeline(image)
        return np.argmax(model.predict(image), axis=1)

def read_camera(camera):

    model, type_ = load_model('model.pickle')
    detector = \
        VehicleDetector({
            "labels": ["car", "truck", "bus"],
            "min_confidence": 0.25,
            "threshold": 0.3,
            "min_sizes": {
                "width": camera.min_width,
                "height": camera.min_height
        }})
    detector.fit('yolo-coco')

    mask = cv2.imread(os.path.join(settings.MEDIA_ROOT, str(camera.mask)))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thread = threading.currentThread()

    for frame in RTSPReader(camera.ip_adress, 1, reload_after_n_frames=50):
        if frame is not None:
            if not getattr(thread, "do_run", True):
                return
            logger('[Read] {}'.format(camera.pk))

            orig = frame.copy()
            image = cv2.bitwise_and(frame, frame, mask=mask)
            boxes = detector.predict(image)
            for box, _ in boxes:
                (y, h, x, w) = box
                roi = orig[y: y+h, x: x+w]

                if is_car_unique(roi):
                    logger('Found car {}'.format(camera.pk))
                    label = predict(model, roi, type_)

                    if label != 1:
                        requests.get(camera.open_link)

                    save_image(camera, orig, orig[y: y+h, x: x+w], label)


threads = {}

@login_required
def start(request, pk):
    camera = Camera.objects.get(pk=pk)

    if camera.active:
        return HttpResponse('Ошибка! Камера уже запущена')

    t = threading.Thread(target=read_camera, args=(camera,))
    threads[camera.pk] = t
    t.start()

    camera.active = True
    camera.save()

    return HttpResponse('Была запущена камера: {}'.format(camera.adress))

def start_all(request):
    #print('Hiii')
    cameras = Camera.objects.all()
    for camera in cameras:
        if not camera.active:
            t = threading.Thread(target=read_camera, args=(camera,))
            threads[camera.pk] = t
            t.start()
            time.sleep(5)
            camera.active = True
            camera.save()

    return HttpResponse('Запущенны все камеры')

@login_required
def stop(request, pk):
    camera = Camera.objects.get(pk=pk)
    if not camera.active:
        return HttpResponse('Ошибка! Камера уже выключена')

    threads[pk].do_run = False
    threads[pk].join()

    camera.active=False
    camera.save()

    return HttpResponse('Камера {} была отключена'\
                .format(Camera.objects.get(pk=pk).adress))

@login_required
def stop_all(request):
    for pk, thread in threads.items():
        camera = Camera.objects.get(pk=pk)
        if camera.active:

            thread.do_run = False
            thread.join()

            camera.active = False
            camera.save()

    return HttpResponse('Все камеры были выключены')


@login_required
def partial_train(request):

    # Loading model
    with open('model.pickle', 'rb') as file:
        model = pickle.load(file)

    # Loading training instances
    X = []
    y = []
    shots = Shot.objects.filter(wrong_label=True)
    for shot in shots:
        image = cv2.imread(os.path.join(settings.MEDIA_ROOT, str(shot.car)))
        image = cv2.resize(image, (64, 64))
        W, H, C = image.shape
        image = image.reshape((W * H * C))

        X.append(image)
        y.append(shot.type)

        shot.wrong_label = False
        shot.save()

    X = np.array(X).astype(np.float32)
    y = np.array(y)

    if len(X) == 0:
        return HttpResponse('Ошибка! Не было выбрано ни одного фото')

    model.partial_fit(X, y)

    with open('model.pickle', 'wb') as file:
        pickle.dump(model, file)

    return HttpResponse('Модель дообучена!')
