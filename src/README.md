# Source code used when preparing the talk

## The programs

* [find_faces.py](./find_faces.py) is code for the first part of the tutorial.
  It finds faces in an image, and writes out those found faces as separate
  image files.
* [calc_embeddings.py](./calc_embeddings.py) is code for the second part of
  the tutorial. Given an image, it will calculate the embeddings for that
  image, and write them to PostgreSQL. Note that the schema it uses is the
  same as described in the tutorial itself.
  
The next two programs are from following the ideas at the end of the tutorial.

* [find_faces_store_embeddings.py](./find_faces_store_embeddings.py) finds
  faces in an image, and stores their embeddings in PostgreSQL. Note that the
  schema it uses is *not* the same as in the tutorial - it adds a column for
  the original filename.
* [find_nearby_faces.py](./find_nearby_faces.py) takes a reference face image
  file, finds the face in, calculates its embedding, and looks for the N
  nearest faces storedi in the PostgreSQL database (where N defaults to 5).
  
To run the programs, follow the start of the tutorial in order to set up a
virtual environment (Python requirements are in
[requirements.txt](./requirements.txt)). Don't forget to download the HAAR
cascade XML file, again as described in the tutorial.


## License

The code is licensed under the Apache license, version 2.0, for compatibility
with the [tutorial source code repository](https://github.com/Aiven-Labs/pgvector-image-recognition/). The full license text is available in the [LICENSE][./LICENSE] file.

Please note that the project explicitly does not require a CLA (Contributor License Agreement) from its contributors.

