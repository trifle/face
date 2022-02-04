"""
Our own wrapper to retinaface face extraction methods
"""

from pathlib import Path
import tarfile

import cv2
import numpy as np
from .retinaface import RetinaFace
from .retinaface_align import norm_crop
from decord import VideoReader, cpu, gpu


from utils import log_complete, write_tar_image, image_scale

GPU_ID = 0
FACE_DETECTION_THRESHOLD = 0.8

face_detector = RetinaFace('./models/R50', 0, GPU_ID, 'net3')


def get_bboxes(img, faces):
    """
    Helper function to extract bounding boxes from an image
    """
    for i in range(faces.shape[0]):
        f = faces[i].astype(np.int)
        bbimg = img[f[1]:f[3], f[0]:f[2], :]
        yield bbimg


def sort_faces(faces, landmarks, min_size=None):
    """
    Make bboxes sortable (int -> tuple), then
    combine with landmarks and sort both
    """
    bboxes = []
    sizes = []
    for f in faces:
        confidence = f[4]
        f = f.astype(np.int)
        face_dim = (f[3] - f[1], f[2] - f[0])
        if min_size and (min(face_dim) < min_size):
            continue
        bbox = (f[0], f[1], f[2], f[3], confidence)
        bboxes.append(bbox)
        sizes.append(face_dim)

    if not bboxes:
        return [], [], []
    sorted_faces = sorted(list(zip(bboxes, landmarks)))
    faces, landmarks = zip(*sorted_faces)
    return faces, landmarks, sizes


@log_complete
def extract_faces_video(video_path: Path, output_dir: Path, min_size=None, skip=1, output_frames=0):
    """
    Extract and align faces from a single video using RetinaFace

    skip: factor for skipping frames, i.e. 2=every 2nd second
    output_frames: if anything other than 0, write every nth frame to disk - for debugging
    """
    metadata_filename = output_dir / f'{video_path.stem}_face_metadata.tsv'
    if not metadata_filename.exists():
        with open(metadata_filename, 'w') as mf:
            mf.write(f'image_basename\twidth\theight\ttop\tleft\tbottom\tright\timg_width\timg_height\tconfidence\n')
    chip_tar_name = output_dir / f'{video_path.stem}_chips.tar'
    # Append fails if the file does not exist yet!
    try:
        chip_tar_archive = tarfile.open(chip_tar_name, "a")
    except tarfile.ReadError:
        chip_tar_archive = tarfile.open(chip_tar_name, "w")

    vr = VideoReader(str(video_path), ctx=cpu())
    video_basename = video_path.stem
    fps = int(vr.get_avg_fps())

    # Determine frame offsets if file is overlapping / X.40 minutes slot
    first_frame = 0
    last_frame = len(vr)
    if '40.ts.mp4' in str(video_path):
        first_frame = fps * 60 * 10  # skip first 10 minutes
        last_frame = first_frame + (fps * 60 * 20)  # use 20 minutes

    seq = range(first_frame, last_frame, fps * skip)
    # Memory hungry since it reads all frames into one tensor
    # frames = vr.get_batch(seq).asnumpy()    
    # for idx, img in zip(seq, frames):
    for idx in seq:
        img = vr[idx].asnumpy()
        img_height, img_width, _ = img.shape
        # Rescale
        scales = image_scale(img)
        # Detect faces and landmarks
        faces, landmarks = face_detector.detect(
            img, FACE_DETECTION_THRESHOLD, scales=scales)
        faces, landmarks, sizes = sort_faces(faces, landmarks, min_size)
        if output_frames and (output_frames % 100 == 0):
            annotated_img = np.copy(img)
        # Write faces to tar file
        for i in range(len(faces)):
            lmk = landmarks[i].astype(np.int)
            chip = norm_crop(img, lmk, 224,)
            width, height = sizes[i]
            top, left, bottom, right, confidence = faces[i]
            # Draw bounding box
            if output_frames and (output_frames % 100 == 0):
                cv2.rectangle(annotated_img, (top, left), (bottom, right), (0, 0, 255), 2)
            image_basename = f'{video_basename}_frame_{idx}_second_{int(idx/fps)}_face{i}.jpg'
            chip = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB)
            write_tar_image(chip, image_basename, chip_tar_archive)
            with open(metadata_filename, 'a') as mf:
                metadata = f'{image_basename}\t{width}\t{height}\t{top}\t{left}\t{bottom}\t{right}\t{img_width}\t{img_height}\t{confidence}\n'
                mf.write(metadata)
        # Write frame to disk if index is a match
        if output_frames and (output_frames % 100 == 0):
            frame_filepath = output_dir / f'{video_basename}_frame_{idx}_second_{int(idx/fps)}.jpg'
            cv2.imwrite(str(frame_filepath), cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    chip_tar_archive.close()


@log_complete
def extract_faces(image_path: Path, output_dir: Path, min_size=None):
    """
    Extract and align faces from a single image using RetinaFace
    """
    metadata_filename = output_dir / f'{output_dir.stem}_face_metadata.tsv'
    if not metadata_filename.exists():
        with open(metadata_filename, 'w') as mf:
            mf.write(f'image_basename\twidth\theight\ttop\tleft\tbottom\tright\timg_width\timg_height\tconfidence\n')
    chip_tar_name = output_dir / f'{output_dir.stem}_chips.tar'
    chip_tar_archive = tarfile.open(chip_tar_name, "a")

    img = cv2.imread(str(image_path))
    img_height, img_width, _ = img.shape
    scales = image_scale(img)
    faces, landmarks = face_detector.detect(
        img, FACE_DETECTION_THRESHOLD, scales=scales)
    faces, landmarks, sizes = sort_faces(faces, landmarks, min_size)
    for i in range(len(faces)):
        lmk = landmarks[i].astype(np.int)
        chip = norm_crop(img, lmk, 224,)
        width, height = sizes[i]
        top, left, bottom, right, confidence = faces[i]
        image_basename = f'{image_path.stem}_face{i}.jpg'
        write_tar_image(chip, image_basename, chip_tar_archive)
        with open(metadata_filename, 'a') as mf:
            metadata = f'{image_basename}\t{width}\t{height}\t{top}\t{left}\t{bottom}\t{right}\t{img_width}\t{img_height}\t{confidence}\n'
            mf.write(metadata)
    chip_tar_archive.close()
