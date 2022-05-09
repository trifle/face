"""
Tooling to build mean embeddings from known people.
Prerequisites:

- One folder per person with images inside, should not contain
  any other person
- Run face extraction pipeline to generate tars
- Run this tool against the produced tars to generate mean embeddings
- Re-run if folder structure is updated
"""

import argparse
from pathlib import Path
import tarfile

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Aggregate face embeddings",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path,
                        help="embedding directories")
    args = parser.parse_args()
    return args


def parse_embeddings(emb: str):
    features = [float(d) for d in emb.split('\t')]
    return np.array(features)


def load_embeddings(embeddings_file: Path):
    with open(embeddings_file, 'r') as ef:
        embeddings = {}
        for i, line in enumerate(ef):
            # Skip header
            if line.startswith("filename"):
                continue
            # Pattern match head, *rest; erlang style
            image_path, embedder, *dimensions = line.strip("\n").split("\t")
            features = [float(dim) for dim in dimensions]
            embeddings[image_path] = np.array(features)
    return embeddings


def scan_dir(directory: Path):
    emb_file = directory.glob('*_embeddings.tsv')
    if emb_file:
        embeddings = load_embeddings(emb_file[0])
        embeddings = embeddings.values()
        mean_embeddings = np.mean(embeddings, axis=0)
        basename = str(emb_file.name)[:-len('*_embeddings.tsv')]
        return (basename, mean_embeddings)


def mean_identity(emb_file: Path, output_dir: Path):
    embeddings = load_embeddings(emb_file)
    embeddings = [np.array(e) for e in embeddings.values()]
    # distances = [cos_distance(embeddings[i], embeddings[i + 1])
    #              for i in range(len(embeddings)-1)]
    # print(f'average internal distance: {sum(distances)/len(distances)}')
    mean_embeddings = np.mean(embeddings, axis=0)
    basename = str(emb_file.name)[:-len('_embeddings.tsv')]
    identity_file = output_dir / f'identities.tsv'
    emb_string = "\t".join((str(e) for e in mean_embeddings.flatten()))
    with open(identity_file, 'a') as idf:
        idf.write(f'{basename}\tARC\t{emb_string}\n')


def norm(emb):
    """
    Normalize embeddings to make comparisons stable
    """
    emb_norm = np.linalg.norm(emb)
    return emb / emb_norm


def cos_distance(e1, e2, normalize=True):
    # Compare two embeddings by cosine distance.
    # This is the distance metric of choice for ARCface embeddings,
    # which are designed to maximize angular distance between faces
    # Normalization should always be applied.
    if normalize:
        e1 = norm(e1).flatten()
        e2 = norm(e2).flatten()
    return np.dot(e1, e2.T)


def augment_with_identities(emb_file: Path):
    identity_file = Path("/output/identities/identities.tsv")
    identities = {}
    for line in open(identity_file):
        identity, embedder, *dimensions = line.strip("\n").split("\t")
        features = [float(dim) for dim in dimensions]
        identities[identity] = np.array(features)

    annotated_filename = emb_file.parent / f'{emb_file.stem}_annotated.tsv'
    with open(annotated_filename, 'w') as af:
        header = ['filename', 'ARC']
        for identity, _ in identities.items():
            header.append(identity)
        af.write('\t'.join(header) + '\n')

        for line in open(emb_file):
            if line.startswith('filename'):
                continue
            image_path, embedder, *dimensions = line.strip("\n").split("\t")
            outline = [image_path, embedder]
            features = [float(dim) for dim in dimensions]
            embedding = np.array(features)
            for identity, i_emb in identities.items():
                cos = cos_distance(i_emb, embedding)
                outline.append(str(bool(cos > 0.8)))
                if cos > 0.8:
                    id_list_file = identity_file.parent / f'{identity}'
                    with open(id_list_file, 'a') as ilf:
                        ilf.write(image_path + "\n")
            af.write('\t'.join(outline) + '\n')


def main():
    args = get_args()

    for child in args.input.iterdir():
        if child.is_dir():
            name, embeddings = scan_dir(child)


if __name__ == '__main__':
    main()
