#!/usr/bin/env python3

"""
Given a face (as an image file) calculate its embedding, and then look for
nearby embeddings in the PostgreSQL database.
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


def calc_reference_embedding(face_file, haar_cascade, ibed):
    orig_image = cv2.imread(face_file, cv2.IMREAD_GRAYSCALE)
    faces = find_faces(orig_image, haar_cascade)

    if len(faces) == 0:
        raise GiveUp(f"Didn't find any faces in {face_file}")
    elif len(faces) > 1:
        raise GiveUp(f"Found more than one face in {face_file}")

    cropped_images = []
    for x, y, w, h in faces:
        # crop the image to select only the face
        cropped_images.append(orig_image[y : y + h, x : x + w])

    # convert to a Pillow image since that's what imgbeddings wants
    face = Image.fromarray(cropped_images[0])
    # calculate the embeddings for that face and return them
    return ibed.to_embeddings(face)[0]


def ask_pg_and_report(pg_uri, vector_str, number_matches):
    """Ask PostgreSQL for the requested number of matches and report the filenames"""

    with psycopg2.connect(pg_uri) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT filename FROM pictures ORDER BY embedding <-> %s LIMIT %s;",
                (vector_str, number_matches)
            )
            rows = cur.fetchall()
        print(f'Number of results: {len(rows)}')
        for index, row in enumerate(rows):
            print(f'  {index}: {row[0]}')



@click.command(no_args_is_help=True)
@click.argument('face_file', nargs=1, type=click.Path(exists=True), required=True)
@click.option('-n', '--number-matches', default=5)
@click.option('-p', '--pg-uri', default=PG_SERVICE_URI,
              help='the URI for the PostgreSQL service, defaulting to $PG_SERVICE_URI if that is set')
def main(face_file: tuple[str], number_matches: int, pg_uri: str):
    haar_cascade = load_algorithm()

    print('Loading imgbeddings')
    ibed = imgbeddings()

    # Calculate the embedding for the face file - we assume only one face
    embedding = calc_reference_embedding(face_file, haar_cascade, ibed)

    # Convert to something that will work in SQL
    vector_str = ", ".join(str(x) for x in embedding.tolist())
    vector_str = f'[{vector_str}]'

    ask_pg_and_report(pg_uri, vector_str, number_matches)


if __name__ == '__main__':
    try:
        main()
    except GiveUp as e:
        print(f'{e}')
        exit(1)
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        exit(1)
