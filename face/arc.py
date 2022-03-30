from pathlib import Path

import cv2
import insightface

from utils import log_complete

GPU_ID = 0
FACE_DETECTION_THRESHOLD = 0.8

arc_model = insightface.model_zoo.get_model('arcface_r100_v1')
arc_model.prepare(ctx_id=GPU_ID)


@log_complete
def extract_embeddings(image_path: Path, output_file: Path):
    """
    Extract embeddings from a single image file
    """
    img = cv2.imread(str(image_path))
    embeddings = arc_model.get_embedding(img).flatten()
    embedding_string = "\t".join((str(e) for e in embeddings))
    embedding_line = f'{image_path.name}\tarcface\tembeddings\t{embedding_string}\n'
    with open(output_file, 'a') as ef:
        ef.write(embedding_line)


def single_embedding(image):
    """
    Extract embeddings from a single image
    """
    embeddings = arc_model.get_embedding(image).flatten()
    embedding_string = "\t".join((str(e) for e in embeddings))
    yield('ARC', embedding_string)


def embedding_iterator(image_batch):
    for image in image_batch:
        embeddings = arc_model.get_embedding(image).flatten()
        embedding_string = "\t".join((str(e) for e in embeddings))
        yield('ARC', embedding_string)
