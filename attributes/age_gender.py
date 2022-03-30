"""
Minimal implementation of https://github.com/yu4u/age-gender-estimation
Uses the pre-built UTK and IMDB weights from that repo
"""
from pathlib import Path

import cv2
import numpy as np
from .wide_resnet import WideResNet

from utils import log_complete, load_image_from_tar


DEPTH = 16
WIDTH = 8
AGES = np.arange(0, 101).reshape(101, 1)

# base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
# prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, activation="softmax", name="pred_age")(base_model.output)
# model = Model(inputs=base_model.input, outputs=prediction)

imdb_model = WideResNet(image_size=64, depth=DEPTH, k=WIDTH)()
utk_model = WideResNet(image_size=64, depth=DEPTH, k=WIDTH)()
imdb_model.load_weights("models/weights.28-3.73.hdf5")
utk_model.load_weights("models/weights.29-3.76_utk.hdf5")


def age_gender_iterator(image_batch, classifier):
    """
    """
    imgs = []
    for image in image_batch:
        img = cv2.resize(image, (64, 64))
        imgs.append(img)
    if not imgs:
        return
    imgs = np.stack(imgs)
    if classifier == 'imdb':
        results = imdb_model.predict(imgs)
    elif classifier == 'utk':
        results = utk_model.predict(imgs)
    for i, (genders, ages) in enumerate(zip(*results)):
        age = ages.dot(AGES).flatten()[0]
        f, m = genders
        yield (classifier, age, f)


def age_gender_batch(image_paths: list, output_file: Path, classifier):
    """
    TODO: Verify
    """
    assert classifier in ('imdb', 'utk')
    imgs = []
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        img = cv2.resize(img, (64, 64))
        imgs.append(img)
    if not imgs:
        return
    imgs = np.stack(imgs)
    if classifier == 'imdb':
        results = imdb_model.predict(imgs)
    elif classifier == 'utk':
        results = utk_model.predict(imgs)
    for i, (genders, ages) in enumerate(zip(*results)):
        age = ages.dot(AGES).flatten()[0]
        f, m = genders
        age_gender_base = f'{image_paths[i].name}\t{classifier}'
        with open(output_file, 'a') as af:
            af.write(f'{age_gender_base}\tage\t{age}\n')
            af.write(f'{age_gender_base}\tfemale\t{f}\n')
            af.write(f'{age_gender_base}\tmale\t{m}\n')


@log_complete
def age_gender_batch_imdb(image_paths: list, output_file: Path):
    age_gender_batch(image_paths, output_file, 'imdb')


@log_complete
def age_gender_batch_utk(image_paths: list, output_file: Path):
    age_gender_batch(image_paths, output_file, 'utk')


@log_complete
def age_gender_imdb(image_path: Path, image_name, output_file: Path):
    img = load_image_from_tar(image_path)
    # img = cv2.imread(str(image_path))
    img = cv2.resize(img, (64, 64))
    imgs = np.expand_dims(img, axis=0)
    result = imdb_model.predict(imgs)
    ages = np.arange(0, 101).reshape(101, 1)
    age = result[1].dot(ages).flatten()[0]
    f, m = result[0][0]
    age_gender_base = f'{image_name}\timdb'
    with open(output_file, 'a') as af:
        af.write(f'{age_gender_base}\tage\t{age}\n')
        af.write(f'{age_gender_base}\tfemale\t{f}\n')
        af.write(f'{age_gender_base}\tmale\t{m}\n')


@log_complete
def age_gender_utk(image_path: Path, image_name, output_file: Path):
    img = load_image_from_tar(image_path)
    # img = cv2.imread(str(image_path))
    img = cv2.resize(img, (64, 64))
    imgs = np.expand_dims(img, axis=0)
    result = utk_model.predict(imgs)
    ages = np.arange(0, 101).reshape(101, 1)
    age = result[1].dot(ages).flatten()[0]
    f, m = result[0][0]
    age_gender_base = f'{image_name}\tutk'
    with open(output_file, 'a') as af:
        af.write(f'{age_gender_base}\tage\t{age}\n')
        af.write(f'{age_gender_base}\tfemale\t{f}\n')
        af.write(f'{age_gender_base}\tmale\t{m}\n')
