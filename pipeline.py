from pathlib import Path
import os
import argparse

from utils import *

# Make tensorflow less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
Master pipeline for face analysist
"""

parser = argparse.ArgumentParser(description='Analyze video files')
parser.add_argument('--frames', type=int, help='output every nth annotated frame to show extraction')
parser.add_argument('--input', type=str, default='input', help='Input directory with video files')
parser.add_argument('--output', type=str, default='output', help='Output directory for faces and data')
args = parser.parse_args()

SOURCE_DIR = Path(args.input)
OUTPUT_DIR = Path(args.output)


@log_complete
def analyze_tar(tarpath, output_dir):
    """
    Analyze faces stored in a tar file
    """
    from attributes.age_gender import age_gender_iterator
    outfile = output_dir / f'{tarpath.stem}_age_gender.tsv'

    def flush(chunk, names, outfile):
        """
        Analyze a batch of faces and write results to disk
        """
        results = list(age_gender_iterator(chunk, "imdb"))
        with open(outfile, 'a') as of:
            for name, (classifier, age, f) in zip(names, results):
                of.write(f'{name}\t{classifier}\t{age}\t{f}\n')
        results = list(age_gender_iterator(chunk, "utk"))
        with open(outfile, 'a') as of:
            for name, (classifier, age, f) in zip(names, results):
                of.write(f'{name}\t{classifier}\t{age}\t{f}\n')

    chunk = []
    names = []
    for img, name in iter_tar(tarpath):
        chunk.append(img)
        names.append(name)
        if len(chunk) == 128:
            flush(chunk, names, outfile)
            chunk = []
            names = []
    # Flush the final batch
    flush(chunk, names, outfile)


if __name__ == '__main__':
    """
    Main function
    """
    print(f'>>- Analyzing faces from video files in {args.input}')
    print(f'>>- Startup - loading libraries and models ...')
    from face.retina import extract_faces_video
    for video in Path(SOURCE_DIR).glob("**/*.mp4"):
        print(f'>>- Extracting faces from {video}')
        extract_faces_video(video, OUTPUT_DIR, output_frames=args.frames)

    for tarfile in Path(OUTPUT_DIR).glob("**/*.tar"):
        print(f'>>- Analyzing faces from {tarfile}')
        analyze_tar(tarfile, OUTPUT_DIR)
