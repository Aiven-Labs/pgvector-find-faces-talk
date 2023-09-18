#!/usr/bin/env python3

"""
Given one or mmore image files, find the faces in them and store the embeddings
for those faces in PostgreSQL
"""

import os

from pathlib import Path

import click
import cv2
import psycopg2

from imgbeddings import imgbeddings
from PIL import Image


PG_SERVICE_URI = os.environ.get('PG_SERVICE_URI', None)


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


def write_to_pg(conn, face_key, file_name, embedding):
    with conn.cursor() as cur:
        cur.execute('INSERT INTO pictures (face_key, filename, embedding)'
                    ' VALUES (%s,%s,%s)'
                    ' ON CONFLICT (face_key) DO UPDATE'
                    '   SET filename = EXCLUDED.filename,'
                    '       embedding = EXCLUDED.embedding'
                    ';',
                    (face_key, file_name, embedding.tolist())
                    )


def write_faces_to_pg(faces, orig_image, picture_file, conn, ibed):
    """Write the faces out to the database"""
    file_path = Path(picture_file)
    file_base = file_path.stem
    file_posix = file_path.as_posix()
    for x, y, w, h in faces:
        # crop the image to select just that face
        # convert to a Pillow image since that's what imgbeddings wants
        cropped_image = Image.fromarray(orig_image[y: y + h, x: x + w])
        embedding = ibed.to_embeddings(cropped_image)[0]
        face_key = f'{file_base}-{x}-{y}-{w}-{h}'
        write_to_pg(conn, face_key, file_posix, embedding)


@click.command(no_args_is_help=True)
@click.argument('image_files', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('-p', '--pg-uri', default=PG_SERVICE_URI,
              help='the URI for the PostgreSQL service, defaulting to $PG_SERVICE_URI if that is set')
def main(image_files: tuple[str], pg_uri: str):
    haar_cascade = load_algorithm()

    print('Loading imgbeddings')
    ibed = imgbeddings()

    total_faces = 0
    total_files = 0

    for image_file in image_files:
        # If I try to use one connection for ALL the files, the connection tends to
        # get terminated, so let's try with short connections
        with psycopg2.connect(pg_uri) as conn:
            try:
                print(f'Processing {picture_file}')

                orig_image = cv2.imread(picture_file, 0)
                gray_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
                faces = find_faces(gray_image, haar_cascade)

                write_faces_to_pg(faces, orig_image, picture_file, conn, ibed)
            except Exception as e:
                print(f'Error {e.__class__.__name__} {e}')
                break

            num_faces = len(faces)
            print(f'Found {num_faces} "faces" in {image_file}')
            total_faces += num_faces
            total_files += 1

    print(f'Found {total_faces} "faces" in {total_files} files')

if __name__ == '__main__':
    try:
        main()
    except GiveUp as e:
        print(f'{e}')
        exit(1)
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        exit(1)
