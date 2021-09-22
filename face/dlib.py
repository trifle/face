import argparse
from pathlib import Path
from platform import system
import os
import typing
import shutil
import random

import tqdm
import numpy
import dlib
import dlib.cuda as cuda
import numpy as np
from wide_resnet import WideResNet
from tqdm.contrib.concurrent import thread_map


# Prevent TF from allocating all memory on GPU 0
if system() == 'Linux':
    try:
        # Make TF less verbose
#        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#        import tensorflow as tf
#        gpus = tf.config.experimental.list_physical_devices('GPU')
#        tf.config.experimental.set_memory_growth(gpus[0], True)
        cuda.set_device(0)
        print(f"dlib CUDA: {dlib.DLIB_USE_CUDA}")
    except ImportError:
        print(f'Warning: Running on linux but could not load tensorflow. Intentional?')

# Const
CHIP_SIZE = 150  # Edge length of extracted face chips. dlib works with 150*150
NET_DEPTH = 16
NET_WIDTH = 8


def get_args():
    parser = argparse.ArgumentParser(description="Automatic face detection machinery",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--img_dir", type=Path, default="images",
                        help="target image directory; if set, images in img_dir are used instead of webcam")
    parser.add_argument("--face_dir", type=Path, default="faces",
                        help="save extracted faces to the face_dir")
    args = parser.parse_args()
    return args


# dlib detectors
#detector = dlib.get_frontal_face_detector()
#detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
shape_predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1(
    "dlib_face_recognition_resnet_model_v1.dat")


class Face(typing.NamedTuple):
    face_id: str
    image_path: Path
    detector: str
    extractor: str
    bbox: dlib.rectangle  # typing.Tuple[int, int, int, int]
    chip: numpy.ndarray

    def __str__(self, include_chip=False) -> str:
        line = [f'FACE-{self.face_id}', self.image_path, self.detector, self.extractor,
                self.bbox.left(), self.bbox.top(), self.bbox.right(), self.bbox.bottom(),
                ]
        if include_chip:  # Toggle inclusion of full chip image in line
            line.append(self.chip.dumps())
        return '\t'.join((str(e) for e in line))


def extract_faces_dlib(img_path: Path) -> typing.Generator[Face, None, None]:
    img = dlib.load_rgb_image(str(img_path))
    # Segment image - grayscale, find identical horizontal lines,
    # omit those, slice rest
    imgg = dlib.as_grayscale(img)
    vars = np.var(imgg, axis=1, dtype=int)
    varsm = np.ma.asarray(vars)
    mask = (varsm == 0)
    varsm[mask] = np.ma.masked
    slices = np.ma.flatnotmasked_contiguous(varsm)
    candidate_chips = [img[s.start:s.stop] for s in slices if (s.stop-s.start) > 150]
    if not candidate_chips:
        return
    max_height = max((i.shape[0] for i in candidate_chips))
    # Padding would be necessary to batch process.
    # Not so useful since batches might be too large?
    # Consider smarter approach
#    chips = []
#    for c in candidate_chips:
#        height, width, dim = c.shape
#        chip = np.concatenate([c, np.zeros([max_height-height, width, dim], dtype=np.uint8)])
#        chips.append(chip)
    detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    faces = []
    for chip in candidate_chips:
        detected = detector(chip, upsample_num_times=0)  # number parameter is upsampling
        for bbox in detected:
            bbox = bbox.rect
            face_shape = shape_predictor(img, bbox)
            chip = dlib.get_face_chip(img, face_shape, CHIP_SIZE)
            face_id = f'{img_path.stem}-{bbox.left()}.{bbox.top()}.{bbox.right()}.{bbox.bottom()}'
            yield Face(face_id, img_path, 'dlib_ff', 'dlib_chip', bbox, chip)
#    return faces


def extract_embeddings_dlib(chip: numpy.ndarray) -> dlib.vector:
    return facerec.compute_face_descriptor(chip)


#
# Main
#


def yield_new(known, new):
    for element in new:
        if element not in known:
            yield element


def read_known_images(datafile_path):
    known_files = set()
    if not Path(datafile_path).exists():
        return known_files
    for line in open(datafile_path, 'r'):
        image_id, image_path, *_ = line.split('\t')
        known_files.add(Path(image_path))
    return known_files


def main():
    args = get_args()
    args.face_dir.mkdir(exist_ok=True, parents=True)
    known = read_known_images('dlib_ff.dlib_chip.log')
    target_files = list(args.img_dir.glob('*.*'))
    target_files = list(yield_new(known, target_files))
    random.shuffle(target_files)
    with open('dlib_ff.dlib_chip.log', 'a') as of:
        for image in tqdm.tqdm(target_files):
            for face in extract_faces_dlib(image):
                of.write(str(face) + "\n")
                face_filename = args.face_dir / f'{face.face_id}.jpg'
                dlib.save_image(face.chip, str(face_filename))


if __name__ == '__main__':
    main()

