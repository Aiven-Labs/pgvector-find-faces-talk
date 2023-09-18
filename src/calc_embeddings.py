#!/usr/bin/env python3

"""
calc_embeddings.py - calculate the embeddings of a picture (of a face)
"""

import os

import click
import psycopg2


from imgbeddings import imgbeddings
from PIL import Image


PG_SERVICE_URI = os.environ.get('PG_SERVICE_URI', None)


def calculate_embeddings(face_file):
    """Return the embeddings for the (first face) in the given file"""
    img = Image.open(face_file)
    print('Loading imgbeddings')
    ibed = imgbeddings()
    # calculating the embeddings
    print(f'Calculating embeddings for {face_file}')
    return ibed.to_embeddings(img)[0]

def write_to_pg(pg_uri, face_file, embedding):
    print(f'Inserting embeddings for {face_file} into PG')
    conn = psycopg2.connect(pg_uri)
    cur = conn.cursor()
    cur.execute('INSERT INTO pictures values (%s,%s)', (face_file, embedding.tolist()))
    conn.commit()
    conn.close()


@click.command(no_args_is_help=True)
@click.argument('face_file', required=True)
@click.option('-p', '--pg-uri', default=PG_SERVICE_URI,
              help='the URI for the PostgreSQL service, defaulting to $PG_SERVICE_URI if that is set')
def main(face_file: str, pg_uri: str):
    embedding = calculate_embeddings(face_file)
    write_to_pg(pg_uri, face_file, embedding)


if __name__ == '__main__':
    main()
