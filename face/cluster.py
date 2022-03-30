from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import hdbscan
from sklearn.preprocessing import normalize


from utils import log_complete, image_scale, segment_vertical, segment_horizontal, write_collage

"""
Cluster face embeddings produced by ArcFace

Note: ArcFace implements an angular loss, meaning it optimizes for angular
(arccos) distances between face embeddings.
Clustering should therefore ideally occur using a distance or similarity measure
that is angular as well. Some of those exist, for example hdbscan with
(metric="arccos", algorithm="generic") - that implementation is very memory
inefficient, however.

As a stopgap, we are normalizing (L2) embeddings so that euclidean distances
relate to cosine distances.
"""


def cluster_once(data, paths):
    """
    Minimal clustering that is meant to be nestable
    """
    data = normalize(np.array(data))
    labels = hdbscan.HDBSCAN().fit_predict(data)
    grouped = defaultdict(list)
    grouped_ids = defaultdict(list)
    for i, label in enumerate(labels):
        grouped[label].append(data[i])
        grouped_ids[label].extend(paths[i])
    means = [np.mean(np.array(e), axis=0) for e in grouped.values()]
    mean_ids = list(grouped_ids.values())
    return labels, means, mean_ids


def load_embeddings(filepath, ignore):
    """
    Load and deduplicate embeddings from a tsv file
    """
    embeddings = defaultdict(list)
    for line in open(filepath, 'r'):
        if line.startswith('filename'):
            continue
        image_path, classifier, embedder, * \
            dimensions = line.strip("\n").split("\t")
        if Path(image_path).name in ignore:
            continue
        embedding_string = '\t'.join(dimensions)
        embeddings[embedding_string].append(image_path)
    data = [[float(d) for d in dim.split('\t')] for dim in embeddings.keys()]
    paths = list(embeddings.values())
    return data, paths


def partition(paths, pattern):
    """
    Partition a list of image paths into groups designated
    by matches from a regex pattern
    """
    if not pattern:
        return {'*': range(len(paths))}
    partitions = defaultdict(list)
    for i, p in enumerate(paths):
        match = pattern.findall(p)
        partitions[match].append(i)
    return partitions


def cluster_embeddings(embeddings_files, output_file,
                       grouping_pattern=None,
                       metadata_files=None,
                       min_size=None,
                       min_confidence=None):
    """
    """
    ignore = set()
    full_paths = {}
    print(f'>- Loading metadata')
    for m_file in metadata_files:
        for line in open(m_file):
            path, width, height, top, left, bottom, right, confidence = line.strip(
                "\n").split("\t")
            if min_confidence and (float(confidence) < min_confidence):
                ignore.add(Path(path).name)
            if min_size and min((int(width), int(height))) < min_size:
                ignore.add(Path(path).name)
            full_paths[Path(path).name] = path
    data, paths = [], []
    print(f'>- Loading embeddings')
    for e_file in embeddings_files:
        d, p = load_embeddings(e_file, ignore=ignore)
        data.extend(d)
        paths.extend(p)
    for match, indices in partition(paths, grouping_pattern).items():
        data_slice = [data[i] for i in indices]
        paths_slice = [paths[i] for i in indices]
        print(f'>>- Clustering partition {match} >-> {output_file}')
        labels, means, mean_ids = cluster_once(data_slice, paths_slice)
        # Write these level one cluster IDs
        with open(output_file, 'a') as cf:
            for i, ci in enumerate(labels):
                matching_images = paths[i]
                for image_path in matching_images:
                    cf.write(
                        f'{image_path}\thdbscan_normalized\t{match}\t{ci}\n')
        cluster_dict = defaultdict(list)
        for i, ci in enumerate(labels):
            cluster_dict[ci].extend([full_paths[p] for p in paths[i]])
        for ci, matching_images in cluster_dict.items():
            print(f'>>- Writing collage for {ci}')
            collage_file = output_file.parent / f'{output_file.stem}_{ci}_collage.jpg'
            sample = False
            if len(matching_images) > 1000:
                sample = 1000
            write_collage(matching_images,
                          collage_file,
                          width=16,
                          image_dim=(64, 64),
                          sample=sample)


# Todo: Level 2 clustering
