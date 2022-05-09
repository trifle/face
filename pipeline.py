from pathlib import Path
import os
import argparse
import multiprocessing

from utils import *

import tqdm

# Make tensorflow less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
Master pipeline for face analysist
"""

parser = argparse.ArgumentParser(description='Analyze video files')
parser.add_argument('--frames', type=int,
                    help='output every nth annotated frame to show extraction')
parser.add_argument('--input', type=str, default='input',
                    help='Input directory with video files')
parser.add_argument('--output', type=str, default='output',
                    help='Output directory for faces and data')
parser.add_argument('--chips', action='store_true',
                    default=False,
                    help='Extract face chips')
parser.add_argument('--agegender', action='store_true',
                    default=False,
                    help='Enable age-gender classifier')
parser.add_argument('--fair', action='store_true',
                    default=False,
                    help='Enable fairface classifier')
parser.add_argument('--embeddings', action='store_true',
                    default=False,
                    help='Enable embedding extraction')
parser.add_argument('--create-identities', type=str, default=None,
                    help='directory with one folder of face images per person')
parser.add_argument('--identity', type=str, default=None,
                    help='directory with target identity embeddings as tsv')
args = parser.parse_args()

SOURCE_DIR = Path(args.input)
OUTPUT_DIR = Path(args.output)


@log_complete
def analyze_tar_age_gender(tarpath: Path, output_dir: Path):
    """
    Analyze faces stored in a tar file
    """
    from attributes.age_gender import age_gender_iterator
    outfile = output_dir / f'{tarpath.stem}_age_gender.tsv'
    # Write header on news files
    if not outfile.exists():
        with open(outfile, 'w') as of:
            of.write(f'filename\tclassifier\tage\tgender\trace\n')

    def flush(chunk, names, outfile):
        """
        Analyze a batch of faces and write results to disk
        """
        if not chunk:
            return
        results = list(age_gender_iterator(chunk, "imdb"))
        with open(outfile, 'a') as of:
            for name, (classifier, age, f) in zip(names, results):
                of.write(f'{name}\t{classifier}\t{age}\t{f}\t\n')
        results = list(age_gender_iterator(chunk, "utk"))
        with open(outfile, 'a') as of:
            for name, (classifier, age, f) in zip(names, results):
                of.write(f'{name}\t{classifier}\t{age}\t{f}\t\n')

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


@log_complete
def analyze_tar_fair(tarpath: Path, output_dir: Path):
    """
    Analyze faces stored in a tar file
    """
    from attributes.fair import fair_iterator
    outfile = output_dir / f'{tarpath.stem}_fair.tsv'
    # Write header on news files
    if not outfile.exists():
        with open(outfile, 'w') as of:
            of.write(f'filename\tclassifier\tage\tgender\trace\n')

    def flush(chunk, names, outfile):
        """
        Analyze a batch of faces and write results to disk
        """
        if not chunk:
            return
        results = list(fair_iterator(chunk))
        with open(outfile, 'a') as of:
            for name, (classifier, age, f, race) in zip(names, results):
                of.write(f'{name}\t{classifier}\t{age}\t{f}\t{race}\n')

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


@log_complete
def analyze_tar_embeddings(tarpath: Path, output_dir: Path):
    """
    Analyze faces stored in a tar file to extract
    embeddings with ARC face
    """
    from face.arc import single_embedding
    outfile = output_dir / f'{tarpath.stem}_embeddings.tsv'
    # Write header on news files
    if not outfile.exists():
        with open(outfile, 'w') as of:
            of.write(f'filename\tclassifier\tembeddings\n')

    with open(outfile, 'a') as of:
        for img, name in iter_tar(tarpath):
            classifier, embeddings = single_embedding(img)
            of.write(f'{name}\t{classifier}\t{embeddings}\n')


if __name__ == '__main__':
    """ 
    Main function
    """
    print(f'>>- Analyzing faces from video files in {args.input}')
    print(f'>>- Startup - loading libraries and models ...')
    from face.retina import extract_faces_video, extract_faces

    # If identites should be processed, we need input faces for each identity
    # to be recognized. Those should live in per-person folders. Since the
    # folder structure is nested, let's process those items first and
    # make sure that we have one tsv with mean embeddings for all of those
    # people
    if args.create_identities:
        print(f'>>- Creating identities from {args.create_identities}')
        for child in Path(args.create_identities).iterdir():
            if child.is_dir:
                identities_output = OUTPUT_DIR / f'identities/{child.stem}'
                identities_output.mkdir(
                    exist_ok=True,
                    parents=True)
                for extension in IMAGE_PATTERNS:
                    for image in Path(child).glob(f"**/{extension}"):
                        extract_faces(image, identities_output)
        identity_dir = Path(OUTPUT_DIR) / f'identities'
        for tarfile in identity_dir.glob("**/*.tar"):
            print(f'>>- Extracting embeddings from {tarfile}')
            analyze_tar_embeddings(tarfile, tarfile.parent)

            # Create mean embeddings
            from face.identity import mean_identity
            for i_emb in identity_dir.glob('**/*_embeddings.tsv'):
                mean_identity(i_emb, identity_dir)

    if args.chips:
        for extension in IMAGE_PATTERNS:
            for image in Path(SOURCE_DIR).glob(f"**/{extension}"):
                print(f'>>- Extracting faces from {image.name}')
                extract_faces(image, OUTPUT_DIR)
        for pattern in VIDEO_PATTERNS:
            for video in Path(SOURCE_DIR).glob(f"**/{pattern}"):
                print(f'>>- Extracting faces from {video}')
                extract_faces_video(video, OUTPUT_DIR,
                                    output_frames=args.frames)

    for tarfile in Path(OUTPUT_DIR).glob("**/*.tar"):
        if args.fair:
            print(f'>>- Performing fairface classification on {tarfile}')
            analyze_tar_fair(tarfile, OUTPUT_DIR)
        if args.agegender:
            print(f'>>- Performing age-gender classification on {tarfile}')
            analyze_tar_age_gender(tarfile, OUTPUT_DIR)
        if args.embeddings:
            print(f'>>- Extracting embeddings from {tarfile}')
            analyze_tar_embeddings(tarfile, OUTPUT_DIR)
    if args.identity:
        # Use pre-built mean embedding files to find matching faces
        # and annotate extracted data with them.
        identity_dir = Path(OUTPUT_DIR) / f'identities'
        identity_file = Path(OUTPUT_DIR) / f'identities/identities.tsv'

        from face.identity import augment_with_identities
        target_embedding_files = list(Path(OUTPUT_DIR).glob("**/*_embeddings.tsv"))
        with multiprocessing.Pool() as P:
            for _ in tqdm.tqdm(P.imap_unordered(augment_with_identities, target_embedding_files)):
                pass
        # for emb_file in Path(OUTPUT_DIR).glob("**/*_embeddings.tsv"):
        #     print(f'>>- Building mean embedding from {emb_file}')
        #     augment_with_identities(emb_file, identity_file)
