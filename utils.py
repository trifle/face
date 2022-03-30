from pathlib import Path
import tqdm
import datetime
from contextlib import contextmanager
from timeit import default_timer
import random
import tarfile
import io

import cv2
import numpy as np


IMAGE_PATTERNS = ['*.jpg', '*.png', '*.gif']
VIDEO_PATTERNS = ['*.mp4', '*.mov', '*.mpg', '*.mp2', '*.mv4']


# This is sort of a LRU cache
# But instead of growing it one by one, re-reading the
# entire log file on each pass,
# we populate it on the first read of a logfile.
KNOWN_DONE = set()

ALM_NAMES = {
    'ard': 'ARD Das Erste',
    'zdf': 'ZDV',
    'pro7': 'ProSieben',
    'sat.1': 'Sat.1',
    'vox': 'VOX',
    'rtl': 'RTL',
}


def parse_tv_filename(filename):
    """
    Parse filenames from TV archive recordings
    Example: rtl-201701190940.ts.mp4

    Returns:
    station
    station-alm
    datetime
    """
    base = filename[:-7]
    station, datestring = base.split('-')
    alm = ALM_NAMES[station]
    dt = datetime.datetime.strptime(datestring, "%Y%m%d%H%M")
    return (station, alm, dt, base)


def chunks(lst, n):
    """
    Yield successive n-sized chunks - evergreen snippet from
    https://stackoverflow.com/a/312464
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@contextmanager
def elapsed_timer():
    """
    Helper to benchmark runtimes.
    from https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python/30024601#30024601
    """
    start = default_timer()

    def elapser():
        return default_timer() - start
    yield lambda: elapser()
    end = default_timer()

    def elapser():
        return end - start


def is_done(param, log_name):
    # Early return - is in cache
    if param in KNOWN_DONE:
        return True
    # Not in cache - read this logfile
    if Path(log_name).exists():
        for line in open(log_name):
            KNOWN_DONE.add(line.strip())
        KNOWN_DONE.add(log_name)
    # return known status after populating cache
    return (param in KNOWN_DONE)


def log_complete(func, pos=0, use_output=True):
    """
    Decorator function

    Use with pizza-syntax:
    @log_complete
    def function(…):

    Arguments:
    pos: Index of arguments that is used for memoization
    use_output: Search arguments for an output directory to
    write the memoization files to

    This decorator inspects the name of the enclosed function
    and its arguments. On successful completion of
    the target function, it writes a signature of the call
    (by default the first unnamed argument) to a logfile
    named after the target function.

    This is later used to filter items that have already
    been processed.
    In essence it's a persistent memoization pattern.
    """
    log_name = f'.{func.__module__}.{func.__name__}.complete'

    def find_output_dir(args):
        """
        Identify first argument that is a directory
        """
        for arg in args[1:]:
        # for arg in args:
            if isinstance(arg, Path):
                return arg

    def wrapper(*args, **kwargs):
        target_dir = find_output_dir(args)
        if is_done(args[0].name, target_dir / log_name):
            # already done
            print(f'>>- Skipping cached {func.__name__} {args} {kwargs}')
            return
        if not target_dir.is_dir():
            target_dir = target_dir.parent
        with open(target_dir / log_name, 'a') as logfile:
            func(*args, **kwargs)
            key_arg = args[pos]
            if not isinstance(key_arg, list):
                key_arg = [key_arg]
            for param in key_arg:
                if isinstance(param, Path):
                    logline = f'{param.name}\n'
                else:
                    logline = f'{param}\n'
                logfile.write(logline)
                # print(f'>- {logline.strip()}')
    return wrapper


def make_iterable(item):
    if isinstance(item, list) or isinstance(item, tuple):
        return item
    return [item]


def filter_files(input_dir, patterns, lognames, batch_size=None):
    """
    Given an input directory, a glob pattern
    and a logfile name, this filter yields
    only files that are not present in the
    logfile.
    """
    complete = set()
    pending = set()
    for logname in make_iterable(lognames):
        Path(logname).touch()
        for line in open(logname, 'r'):
            complete.add(line.strip())
    for pattern in make_iterable(patterns):
        for path in input_dir.glob(f'**/{pattern}'):
            if str(path.name) not in complete:
                pending.add(path)
    print(f'>>- Skipping {len(complete)} complete items')
    description = f">-> {input_dir}:{logname.split('/')[-1][:-9]}"
    if batch_size:
        pending = list(pending)
        for batch in tqdm.tqdm(list(chunks(pending, batch_size)), desc=description):
            yield batch
    else:
        for path in tqdm.tqdm(pending, desc=description):
            yield path


def filter_tar(input_dir, lognames):
    """
    Given an input tar file, a glob pattern
    and a logfile name, this filter yields
    only files that are not present in the
    logfile.
    """
    complete = set()
    for logname in make_iterable(lognames):
        Path(logname).touch()
        for line in open(logname, 'r'):
            complete.add(line.strip())
    for archive in input_dir.glob(f'*.tar'):
        description = f">-> {archive.name}"
        with tarfile.open(str(archive), 'r') as tf:
            for member in tqdm.tqdm(tf, desc=description, total=len(tf.getnames())):
                if member.name not in complete:
                    yield (tf.extractfile(member), member.name)


def iter_tar(input_file: Path):
    """
    Yield images from a tar file
    """
    try:
        with tarfile.open(str(input_file), 'r') as tf:
            for member in tqdm.tqdm(tf, desc=f">-> {input_file.name}", total=len(tf.getnames())):
                img = load_image_from_tar(tf.extractfile(member))
                yield (img, member.name)
    except tarfile.ReadError as exc:
        print(f'Error reading tarfile: {input_file}\n{exc}')


def iter_projects(SOURCE_DIR, OUTPUT_DIR):
    """
    Set up per-project output directory structure
    """
    PROJECT_DIRS = [p for p in SOURCE_DIR.iterdir() if p.is_dir()]
    for project_in in PROJECT_DIRS:
        # Set up output directories for each project-subdir in the input folder
        project_out = OUTPUT_DIR / project_in.name
        project_frames = project_out / 'frames'
        project_chips = project_out / 'chips'

        # Create if necessary. Just like mkdir -p
        project_frames.mkdir(exist_ok=True, parents=True)
        project_chips.mkdir(exist_ok=True, parents=True)
        yield (project_in, project_out, project_frames, project_chips)


def image_scale(img):
    """
    Calculate shrink factor required to fit image in given
    target boundary.
    """
    h, w, colors = img.shape
    r = 640 / max(w, h)
    return [r]


def chunks_fill(lst, n, blank):
    """
    Yield successive n-sized chunks from lst.
    Fill blanks with the np array supplied in blank so that
    vstack/hstack don't complain about ragged shapes.
    """
    for i in range(0, len(lst), n):
        cand = lst[i:i + n]
        missing = n - len(cand)
        [cand.append(blank) for i in range(missing)]
        yield cand


def write_collage(image_paths, target_path, width=16, image_dim=(64, 64), sample=False):
    """
    Create a collage rows of width=width images.
    As tall as necessary.
    Pick a target number of images to include with sample=K.
    """
    if sample:
        image_paths = random.choices(image_paths, k=sample)
    blank = np.zeros((image_dim[0], image_dim[1], 3), np.uint8)
    imgs = []
    for ip in image_paths:
        imgs.append(cv2.resize(cv2.imread(str(ip)), image_dim))
    image_lines = [np.hstack(c) for c in chunks_fill(imgs, width, blank)]
    image = np.vstack(image_lines)
    cv2.imwrite(str(target_path), image)


def write_tar_image(image, filename, tar_archive):
    """
    Prepare write image data to a tar file
    input: image - a numpy array as produced by cv2.imread
    tarfile: handle to the tar file instance
    """
    _, encoded = cv2.imencode('.jpg', image)
    encoded = encoded.tobytes()
    record = tarfile.TarInfo()
    record.name = filename
    record.size = len(encoded)
    image_filelike = io.BytesIO(encoded)
    tar_archive.addfile(record, image_filelike)
    return True


def load_image_from_tar(filehandle):
    """
    Files loaded from tar files don't have a true path.
    Instead, they are returned as file-like objects (i.e. with
    open(), close(), seek() mechanics etc.).
    We therefore need to read them into a memory buffer and decode
    from there.
    """
    image_array = np.asarray(bytearray(filehandle.read()), dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
