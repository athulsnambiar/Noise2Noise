import numpy as np
import pathlib
import shutil
import os

FOLDER_NAMES = ["imgs", "nparrays"]

def clamp_img(img: np.ndarray):
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)
    return img

def clean_dir(p: pathlib.Path):
    for i in p.iterdir():
        if (i.is_dir() and
            not i.is_symlink() and
            i.name in FOLDER_NAMES):
        shutil.rmtree(p.absolute())

def check_file_exists(f: str) -> bool:
    """Check Whether file exist or not

    Arguments:
        f {str} -- File Name

    Returns:
        bool -- Returns True file file exists, else returns false
    """
    p = pathlib.Path(f)
    if p.exists() and p.is_file():
        return True
    return False
