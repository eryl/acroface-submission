#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
#import multiprocessing.dummy as multiprocessing
import argparse

import importlib
import sys
from pathlib import Path
import io
import os
from typing import Tuple

from PIL import Image, UnidentifiedImageError, ImageOps

from tqdm import tqdm, trange
import numpy as np


def alpha_composite(front, back):
    """Alpha composite two RGBA images.

    Source: http://stackoverflow.com/a/9166671/284318

    Keyword Arguments:
    front -- PIL RGBA Image object
    back -- PIL RGBA Image object

    """
    front = np.asarray(front)
    back = np.asarray(back)
#    print("shape of front", front.shape)
#    print("shape of back", back.shape)

    result = np.empty(front.shape, dtype='float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    falpha = front[alpha] / 255.0
    balpha = back[alpha] / 255.0
    result[alpha] = falpha + balpha * (1 - falpha)
    old_setting = np.seterr(invalid='ignore')
    result[rgb] = (front[rgb] * falpha + back[rgb] * balpha * (1 - falpha)) / result[alpha]
    np.seterr(**old_setting)
    result[alpha] *= 255
    np.clip(result, 0, 255)
    # astype('uint8') maps np.nan and np.inf to 0
    result = result.astype('uint8')
    #result = Image.fromarray(result, 'RGBA')
    result = Image.fromarray(result) # mode is apperantly deprecated
    return result


def alpha_composite_with_color(image, color=(255, 255, 255)):
    """Alpha composite an RGBA image with a single color image of the
    specified color and the same size as the original image.

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    back = Image.new('RGBA', size=image.size, color=color + (255,))
    return alpha_composite(image, back)


def remove_alpha(work_package):
    image_path, output_file, = work_package
    #print(f"Encrypting {path}")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    pil_image = Image.open(str(image_path))
    if 'A' in pil_image.mode:
        pil_image = alpha_composite_with_color(pil_image).convert('RGB')
    pil_image.save(str(output_file))
    return output_file

    
def main():
    parser = argparse.ArgumentParser(description="Script to remove the alpha channel from images")
    parser.add_argument('source_dir', help="Directory with image files to remove alpha channel from", type=Path)
    parser.add_argument('--output-directory', help="directory to store flattened_images_to", type=Path)
    args = parser.parse_args()
    
    images = args.source_dir.glob('**/*.png')
    
    work_packages = []
    for image_path in images:
        if args.output_directory is not None:
            relative_image_path = image_path.relative_to(args.source_dir)
            output_image_path = args.output_directory / relative_image_path
        else: 
            # save to the same great grand-parent directory, but with new grand-parent
            
            image_grand_parent = image_path.parent.parent
            image_great_grand_parent = image_path.parent.parent.parent
            output_image_path = image_great_grand_parent / f"{image_grand_parent.name}_no-alpha" / image_path.relative_to(image_grand_parent)

        if not output_image_path.exists():
            work_packages.append((image_path, output_image_path))

    with multiprocessing.Pool() as pool:
        for name in tqdm(pool.imap_unordered(remove_alpha, work_packages), desc="Removing alpha", total=len(work_packages)):
            pass


if __name__ == '__main__':
    main()
