#!/usr/bin/env python3

"""
find_faces.py - find faces in a given picture, and output them to new files
"""

from pathlib import Path

import click
import cv2


class GiveUp(Exception):
    pass


def load_algorithm():
    """Load the haar case algorithm"""
    algorithm = "haarcascade_frontalface_default.xml"
    # passing the algorithm to OpenCV
    haar_cascade = cv2.CascadeClassifier(algorithm)
    if haar_cascade.empty():
        raise GiveUp(f'Error reading algorithm file {algorithm} - no algorithm found')
    return haar_cascade


def find_faces(gray_image, haar_cascade):
    """Find faces in the image, and return them"""
    return haar_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.05,
        minNeighbors=2,
        minSize=(250, 250),
        #minSize=(100, 100),
    )


def write_faces(faces, orig_image, picture_file):
    """Write the faces out as individual files"""
    face_base = Path(picture_file).stem
    for x, y, w, h in faces:
        # crop the image to select just that face
        cropped_image = orig_image[y : y + h, x : x + w]
        target_file_name = f'{face_base}_{x}-{y}-{w}-{h}.png'
        cv2.imwrite(target_file_name, cropped_image)
    print(f'Found {len(faces)} faces')


@click.command(no_args_is_help=True)
@click.argument('picture_file', required=True)
def main(picture_file: str):
    haar_cascade = load_algorithm()

    # Read the image in, as greyscale
    orig_image = cv2.imread(picture_file, cv2.IMREAD_GRAYSCALE)

    faces = find_faces(orig_image, haar_cascade)
    write_faces(faces, orig_image, picture_file)


if __name__ == '__main__':
    try:
        main()
    except GiveUp as e:
        print(f'{e}')
        exit(1)
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        exit(1)
