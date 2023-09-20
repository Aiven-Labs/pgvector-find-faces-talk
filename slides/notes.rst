How I used PostgreSQL\ :sup:`®` to find pictures of me at a party
==================================================================


.. class:: title-slide-info

    By Tibs (they / he)

    Slides and source code at
    https://github.com/aiven-labs/pgvector-find-faces-talk

.. footer::

   *tony.ibbs@aiven.io* / *https://aiven.io/tibs*  / *@much_of_a*


Broad structure
---------------

Introduction and vague background

Not an explanation of ML

Finding pictures of me

Why PostgreSQL\ :sup:`®`.

Part the first: Introduction and vague background
-------------------------------------------------

I'm Tibs_, and I'm a Developer Educator at Aiven_:

  Aiven: The trusted open source data platform for everyone

.. _Aiven: https://aiven.io/
.. _Tibs: https://aiven.io/tibs

Recently, our engineering colleagues enabled the ``pgvector`` plugin for
PostgreSQL\ :sup:`®` on our platform, and my other colleagues started playing
with it. In particular, Francesco Tisiot wrote a tutorial on how to use it.


I'm an AI skeptic
~~~~~~~~~~~~~~~~~

I lived through the last `AI boom`_ in the 1980s, and the subsequent `AI winter`_ of the 1990s

But we did get expert systems, knowledge based systems, etc. - they just dropped the name "AI"

.. _`AI boom`: https://en.wikipedia.org/wiki/History_of_artificial_intelligence#Boom_(1980%E2%80%931987)
.. _`AI winter`: https://en.wikipedia.org/wiki/AI_winter


My colleagues have been convincing me
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interesting used of ChatGPT and friends:

* Quick prototypes of boring code
* Rewrite this paragraph a different way

But I think the tutorial by my colleague Francesco Tisiot actually sparked my
interest to look at this all a bit more:

* And now, finding my face

Also, recently, a demo I saw on multi-modal comparisons - comparing text, audio,
image and video.

The talk title says "how I used", but it should probably be "how I'm learning
to use" - this is still very much something I'm learning about, and this talk
is only partway through my journey

As medical doctors say:

* See one
* Do one
* Teach one

(which is appropriately scary!)

But yes, one good way of figuring out just that bit more about a process is to
get to the stage where you can give a talk on it.

Part the second: not an explanation of ML
-----------------------------------------

.. figure:: images/markus-winkler-f57lx37DCM4-unsplash.jpg
    :width: 43%

    Photo by `Markus Winkler`_ on Unsplash_

.. _`Markus Winkler`: https://unsplash.com/@markuswinkler?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText

.. _Unsplash: `ML Typewriter`_
.. _`ML Typewriter`:
   https://unsplash.com/photos/f57lx37DCM4?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText

Vector embeddings
-----------------

ML people talk about vectors and embeddings and vector embeddings.

"Embedding" means representing something in a computer.

So a "vector embedding" is

* a vector that represents something,
* stored in a computer.



Not an introduction to vectors and embeddings
---------------------------------------------

That's (at least) a whole other talk!

Broadly, we can describe the characteristics of things with numbers.

For instance, we can describe colours with RGB values (and some colours will
have the same representation), but also (thinking of
an XKCD post) with whether men tend to use that colour name or not.

Or we can describe words with their category of meaning, their part of speech,
and how likely they are to have another word following them.

Or we might describe a picture by it's dominant colours, whether it has
"feathery" or "scaly" parts, whether there are large blocks of particular
colours.

Once we've made such descriptions, we can treat the resultant array of numbers
as a **vector**, and we know how to do lots of interesting mathematics on
vectors.

For instance, considering the RGB case, we have an our RGB value is
essentially an X, Y, Z value, and we can treat the vector for our colour
being::

  (0, 0, 0) -> (R, G, B)

In different words, we can draw an arrow starting at ``(0, 0, 0)`` that goes to
``(R, G, B)``. That gives us a *direction* (in 3d space, because we have 3 values
that can change) and also a length (the length of the arrow).

For instance, we could represent ``(5, 8, 3)`` as follows (``origin`` is ``(0, 0, 0)``)

    **A 3d graph showing a vector**

    .. image:: images/3d-vector.png
        :width: 50%

    Image from http://jccc-mpg.wikidot.com/the-vector-class, `CC BY-SA 3.0`_

    .. _`CC BY-SA 3.0`: https://creativecommons.org/licenses/by-sa/3.0/

It may seem a bit odd to turn our RGB "points" into vectors (arrows) in this
way, but that's because we can then use vector maths to compare two different
colours - for instance, are they going in more-or-less the same direction, are
they of similar lengths, and so on.

    **We can do mathematics with vectors**

    We can compare their

    * length
    * direction

    and we can do maths between vectors - but look elsewhere for that

.. Comment to split the text
..

   **Note** On the "do maths between vectors" - you can ask questions like:

   * "is the vector between colour 1 and colour 2 *similar to* the vector
     between colour 3 and colour 4",

   * "what is the colour that relates to colour 3 in the same way that colour
     2 relates to colour 1".

   but that's way beyond the scope of this talk.

This gets harder to think about when there are more values in our vector!

So once we've got vectors, we can compare them.

    **Calculating the vectors**

    Possible to do by hand for relatively simple cases

        (for instance, in early text analysis)

    but with ML, we can

    * *train* a machine learning system
    * to *"recognise"* that a thing belongs to particular categories.

    This is wonderful - and sometimes leads to surprising results


**Categorising** things to get those arrays is possible to do by hand for
relatively simple cases (people have done this in text analysis, for
instance), but what ML has given us is the ability to *train* a machine
learning system to "recognise" that a thing belongs to particular categories.

This is wonderful - it allows us to categorise things like images, allowing us
to find faces and all sorts of things.

We do, however, need to remember that the system will only work out categories
as it has been taught - and with ML, not all of those categories are one's we
can tell are there. This is why we get problems like a system being good at
recognising faces, but only if they belong to white men. It's also why we can
be surprised when a picture is recognised as a turtle when there are no
turtles in it - something we didn't expect is "similar" to something in the
training pictures, and it's not the thing we hoped for.

See the `References for future reading`_ at the end.

And note the quotation marks around *recognise* - it's very tempting to
anthropomorphise ML software, but it's not actually recognising anything, it's
just performing calculations.


Part the third: Finding pictures of me
--------------------------------------

.. image:: images/slack-picture.jpg


Disclaimer: not as many pictures as you expect
----------------------------------------------

Since many of the photographs would contain other people

(and I'd need their consent to show them)

Based on a tutorial
-------------------

See

* https://aiven.io/developer/find-faces-with-pgvector
* https://github.com/Aiven-Labs/pgvector-image-recognition

`find_faces.py`_ is my version of the initial piece of code, from
`Retrieve the faces from the photos`_, and with my own
adjustments to the ``haar_cascade.detectMultiScale`` settings.

`calc_embeddings.py`_ is my version of the second piece of code, from
`Calculate the embeddings`_.

`find_faces_store_embeddings.py`_ and `find_nearby_faces.py`_ are then the
convenience scripts I wrote to manage the final part of the tutorial, actually
scanning a directory of images to find the faces therein (the first script)
and to find (for instance) my face (the second script).

.. _`find_faces.py`: ../src/find_faces.py
.. _`calc_embeddings.py`: ../src/calc_embeddings.py
.. _`Retrieve the faces from the photos`:
   https://aiven.io/developer/find-faces-with-pgvector#retrieve-the-faces-from-photos
.. _`Calculate the embeddings`:
   https://aiven.io/developer/find-faces-with-pgvector#calculate-the-embeddings
.. _`find_faces_store_embeddings.py`: ../src/find_faces_store_embeddings.py
.. _`find_nearby_faces.py`: ../src/find_nearby_faces.py

Our aim
-------

Find which files contain my face, using SQL like the following:

.. code:: sql

    SELECT filename FROM pictures
      ORDER BY embedding <-> [0.38162553310394287, ..., 0.20030969381332397]
      LIMIT 10;

1. Finding faces and store their embeddings
-------------------------------------------

The first process we want is one that finds the faces in an image, calculates
the embedding for each face, and stores that (along with other information) in
our PostgreSQL database.

.. image:: images/faces-to-pg.png
           :width: 100%

But it's not perfect!
---------------------

In the photo in the previous slide, it only found two faces.

And when analysing another group photo, it also found these two *not* faces:

.. raw:: pdf

    Spacer 0 50


.. |not-a-face| image:: images/not-a-face.png
                        :width: 256

.. |not-a-face2| image:: images/not-a-face2.png
                        :width: 256

|not-a-face| |not-a-face2|

768 floating point numbers
--------------------------

The embedding is an array of 768 floating point numbers.

  ``0.38162553310394287, ..., 0.20030969381332397``

We *could* print that array out, or save it to a text file, and then copy if
when we want to do something with it. But we have a database, so let's use it.

   ** Note** Why 768? Some searching gave me `this answer on stackoverflow`_,
   which says:

     768 comes from the embedding of ViT used by CLIP. In ViT, it transform
     the input image of 224 * 224 pixels, to patches of size 16 * 16 pixels.
     Therefore, when you embed (flatten and use an MLP) the patches with size
     16 * 16 * 3 (RGB) = 768.

.. _`this answer on stackoverflow`: https://stackoverflow.com/questions/75693493/why-the-text-embedding-or-image-embedding-generated-by-clip-model-is-768-%C3%97-n#:~:text=768%20comes%20from%20the%20embedding,3%20(RGB)%20%3D%20768.



2. Looking for photos with my face in them
-------------------------------------------

The second process we want is one that, given a face, calculates its embedding
and then finds the most "similar" faces in the PostgreSQL database.

.. image:: images/find-nearby-faces.png
           :width: 100%

The example here is using my slack image as the reference face (note: those
numbers are not from its actual embedding!)

Set up the environment
----------------------

We're going to be using

* `opencv-python`_ to find faces
* imgbeddings_ to calculate embeddings from an image

We also need to download the ::

  haarcascade_frontalface_default.xml

file from the `OpenCV GitHub repository`_ - this is the pre-trained Haar
Cascade model that we will use to recognise faces.

 **Note** The article `Face Detection with Haar Cascade`_ describes how a Haar
 Cascade model recognises faces.

.. _`Face Detection with Haar Cascade`: https://towardsdatascience.com/face-detection-with-haar-cascade-727f68dafd08


My example programs also use click_ and `psycopg2-binary`_

.. _`opencv-python`: https://pypi.org/project/opencv-python/
.. _imgbeddings: https://github.com/minimaxir/imgbeddings
.. _click: https://click.palletsprojects.com/
.. _`psycopg2-binary`: https://pypi.org/project/psycopg2-binary/
.. _`OpenCV GitHub repository`: https://github.com/opencv/opencv/tree/master/data/haarcascades

Enable pgvector
---------------

Enable the pgvector extension:

.. code:: sql

   CREATE EXTENSION vector;

This only works if the ``pgvector`` extension is installed.

It may already be available, as is the case with Aiven for
PostgreSQL\ :sup:`®`.

See the `Installation instructions`_ on the `pgvector GitHub page`_, which
give some indication of whether it's likely to be available, and how to
install it if not.

.. _`Installation instructions`: https://github.com/pgvector/pgvector#installation
.. _`pgvector GitHub page`: https://github.com/pgvector/pgvector

Create our database table
-------------------------

.. code:: sql

   CREATE TABLE pictures (face text PRIMARY KEY, filename text, embedding vector(768));


* ``face`` is the string we use to identify this particular face:

    ``2023-04-26_170836174_104-1075-260-260``

    It's the base (stem) of the filename, plus the location and dimensions of
    the face in the original file. We use this as our primary key.

* ``filename`` is the name of the file we found the face in:

    ``2023-04-26_170836174.png``

    We want this so we can report the file without needing to work it out from
    the ``face``.

* ``embedding`` is the vector itself, the vector of dimension 768.

Find faces and store their embeddings
-------------------------------------

``find_faces_store_embeddings.py``

::

    Usage: find_faces_store_embeddings.py [OPTIONS] IMAGE_FILES...

    Options:
    -p, --pg-uri TEXT  the URI for the PostgreSQL service, defaulting to
                        $PG_SERVICE_URI if that is set
    --help             Show this message and exit.

Reminder: we're doing this sequence

.. image:: images/faces-to-pg.png
           :width: 100%

Find faces and store their embeddings (1)
-----------------------------------------

.. code:: python

    def main(image_files: tuple[str], pg_uri: str):
        haar_cascade = load_algorithm()
        ibed = imgbeddings()

        for image_file in image_files:
            with psycopg2.connect(pg_uri) as conn:
                orig_image = cv2.imread(picture_file, 0)
                gray_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
                faces = find_faces(gray_image, haar_cascade)

                write_faces_to_pg(faces, orig_image, picture_file, conn, ibed)


``cv2`` is the OpenCV package

Find faces and store their embeddings (2)
-----------------------------------------

.. code:: python

    def load_algorithm():
        algorithm = "haarcascade_frontalface_default.xml"
        haar_cascade = cv2.CascadeClassifier(algorithm)
        if haar_cascade.empty():
            raise GiveUp(f'Error reading algorithm file {algorithm} - no algorithm found')
        return haar_cascade

Find faces and store their embeddings (3)
-----------------------------------------

.. code:: python

        # Read the image in, and convert it to greyscale
        orig_image = cv2.imread(picture_file, 0)
        gray_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)

Find faces and store their embeddings (4)
-----------------------------------------

.. code:: python

    def find_faces(gray_image, haar_cascade):
        return haar_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(250, 250),
            #minSize=(100, 100),
        )

Find faces and store their embeddings (5)
-----------------------------------------

.. code:: python

    def write_faces_to_pg(faces, orig_image, picture_file, conn, ibed):
        file_path = Path(picture_file)
        file_base = file_path.stem
        file_posix = file_path.as_posix()

        for x, y, w, h in faces:
            # Convert to a Pillow image since that's what imgbeddings wants
            cropped_image = Image.fromarray(orig_image[y: y + h, x: x + w])
            embedding = ibed.to_embeddings(cropped_image)[0]
            face_key = f'{file_base}-{x}-{y}-{w}-{h}'

            write_to_pg(conn, face_key, file_posix, embedding)

We wrap the cropped image up as a Pillow ``Image``, since that's
what ``ibed.to_embeddings`` wants.

We *could* look for an image embedding library that doesn't expect an
``Image``, but it's not worth it for this tutorial (and it's not a big
issue).

Find faces and store their embeddings (6)
-----------------------------------------

And here's where we actually write to PostgreSQL

.. code:: python

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

I *could* do better connection management, but I found that if I tried to use
one connection for ALL the files, the connection tends to get terminated, and
I couldn't be bothered to sort that out properly for a test.

  **Note** Running this over 570-ish decent sized photos took me 8-9 minutes,
  which is acceptable for just playing.

Find faces and store their embeddings (7)
-----------------------------------------

``ON CONFLICT`` is interesting:

.. code:: sql

     ON CONFLICT (face_key) DO UPDATE
        SET filename = EXCLUDED.filename,
            embedding = EXCLUDED.embedding;

The ``ON CONFLICT`` clause allows us to overwrite a record if it already
exists - this is useful when I might want to run the same script in testing
more than once, without wanting to ``DELETE FROM pictures;`` each time, to
delete the table content.


Find "nearby" faces
-------------------

``find_nearby_faces.py``

::

    Usage: find_nearby_faces.py [OPTIONS] FACE_FILE

    Options:
    -n, --number-matches INTEGER
    -p, --pg-uri TEXT             the URI for the PostgreSQL service, defaulting
                                    to $PG_SERVICE_URI if that is set
    --help                        Show this message and exit.

Reminder: we're doing this sequence

.. image:: images/find-nearby-faces.png
           :width: 100%

The original tutorial suggested calculating the embedding for the reference
face, and then passing it to the "find" script by hand. That's a pain (768
floating point numbers!) so it's easier to program it.

For laziness, my current script calculates the embedding for the reference
face each time it's run. That's really pretty awful <smile>

- I could store it in the database, and look its embedding up
- but I'd either have to make sure to ignore it when searching
- or I'd have to store it in a different table (perhaps the ideal)
- and I couldn't be bothered for this talk <sad-face>

Find "nearby" faces (1)
-----------------------

.. code:: python

    def main(face_file: tuple[str], number_matches: int, pg_uri: str):
        haar_cascade = load_algorithm()
        ibed = imgbeddings()

        # Calculate the embedding for the face file - we assume only one face
        embedding = calc_reference_embedding(face_file, haar_cascade, ibed)

        # Convert to something that will work in SQL
        vector_str = ", ".join(str(x) for x in embedding.tolist())
        vector_str = f'[{vector_str}]'

        ask_pg_and_report(pg_uri, vector_str, number_matches)

The start, loading the algorithm and ``imgbeddings``, should be familiar from
the previous program.

Find "nearby" faces (2)
-----------------------

.. code:: python

    def calc_reference_embedding(face_file, haar_cascade, ibed):
        orig_image = cv2.imread(face_file, 0)
        gray_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
        faces = find_faces(gray_image, haar_cascade)

        # We hope there's only one face!
        cropped_images = []
        for x, y, w, h in faces:
            cropped_images.append(orig_image[y : y + h, x : x + w])

        face = Image.fromarray(cropped_images[0])
        return ibed.to_embeddings(face)[0]

This is very similar to what we had before, except we're only expecting one
face, the reference face.


Find "nearby" faces (3)
-----------------------

In fact, in the real code it doesn't say:

.. code:: python

        # We hope there's only one face!

I couldn't resist an actual check:

.. code:: python

        if len(faces) == 0:
            raise GiveUp(f"Didn't find any faces in {face_file}")
        elif len(faces) > 1:
            raise GiveUp(f"Found more than one face in {face_file}")


Find "nearby" faces (4)
-----------------------

Our embedding needs turning into something that SQL will understand:

.. code:: python

    vector_str = ", ".join(str(x) for x in embedding.tolist())
    vector_str = f'[{vector_str}]'

Find "nearby" faces (5)
-----------------------

.. code:: python

    def ask_pg_and_report(pg_uri, vector_str, number_matches):
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


This is why I stored the original filename in the database table - so I could
use it in the report at the end.


But how good is it?
-------------------

Well, the search is quick, which is satisfying.

(Something like 3 seconds to compare the embeddings for 5000 faces from 750+ photos)

Wednesday at Crab Week
----------------------

779 files, 5006 faces

* 21 minutes to calculate and store the embeddings

* 3 seconds to find the 10 nearest faces

Which I think is perfectly acceptable for demo software.

Going through the files manually, I found 25 that had my face visible,
but some were in a crowd or obscured, three were of my back (!) and two were
with a false moustache

Results the program found
-------------------------

And here are the first 10 matches from the program

::

    AIVEN2752.jpg -- just me
    AIVEN2839.jpg -- just me
    AIVEN2838.jpg -- just me
    AIVEN2806.jpg -- me in front of audience
    AIVEN2808.jpg -- just me, from side
    AIVEN2750.jpg -- me plus another
    AIVEN2751.jpg -- me plus others
    AIVEN2748.jpg -- me plus others
    AIVEN2681.jpg -- me in group sitting
    AIVEN3104.jpg -- not me, beard and glasses

The first of those, AIVEN2752 (some redaction on the whiteboard):

.. image:: images/AIVEN2752.jpeg
           :width: 30%

Me in a group, AIVEN2751 (again, some redaction of the whiteboard, and also
people's faces hidden):

.. image:: images/AIVEN2751.png
           :width: 60%

The "find faces" program actually found 5 "faces" in that photo - the extra
two were both parts of the text on the whiteboard.

If I asked the program for the first 20 matches, I'd get::

  AIVEN2752.jpg -- as above, just me
  AIVEN2839.jpg -- as above, just me
  AIVEN2838.jpg -- as above, just me
  AIVEN2806.jpg -- as above, me in front of audience
  AIVEN2808.jpg -- as above, just me, from side
  AIVEN2750.jpg -- as above, me plus another
  AIVEN2751.jpg -- as above, me plus others
  AIVEN2748.jpg -- as above, me plus others
  AIVEN2681.jpg -- as above, me in group sitting
  AIVEN3104.jpg -- as above, not me, beard and glasses
  AIVEN2944.jpg -- me, dark glasses & hat, with others
  AIVEN2806.jpg -- as above, me in front of audience
  AIVEN3088.jpg -- not me
  AIVEN3298.jpg -- not me, but a confusing picture! (fake moustaches)
  AIVEN2945.jpg -- me, dark glasses & hat, with others
  AIVEN2796.jpg -- not me, beard and glasses (Claude)
  AIVEN2995.jpg -- not me, beard and glasses
  AIVEN2795.jpg -- not me, beard and glasses (Claude)
  AIVEN3452.jpg -- not me, beard and glasses, dark conditions
  AIVEN3333.jpg -- not me, beard and glasses (twice)

and *my* list from that day was::

  AIVEN2678.jpg -- side view of me in audience
  AIVEN2679.jpg -- view of me, small, in audience
  AIVEN2681.jpg -- FOUND
  AIVEN2748.jpg -- FOUND
  AIVEN2750.jpg -- FOUND
  AIVEN2751.jpg -- FOUND
  AIVEN2752.jpg -- FOUND
  AIVEN2805.jpg -- distance view of me before audience
  AIVEN2806.jpg -- FOUND
  AIVEN2808.jpg -- FOUND
  AIVEN2809.jpg -- side view of me
  AIVEN2838.jpg -- FOUND
  AIVEN2839.jpg -- FOUND
  AIVEN2944.jpg -- FOUND
  AIVEN2945.jpg -- me, dark glasses & hat, with others
  AIVEN2945.png -- FOUND
  AIVEN2952.jpg -- my back!
  AIVEN3031.jpg -- my back in a crowd
  AIVEN3034.jpg -- me in a group
  AIVEN3037.jpg -- me in a group
  AIVEN3040.jpg -- side view of me in a group
  AIVEN3054.jpg -- me, obscured
  AIVEN3311.jpg -- me with hat and false moustache
  AIVEN3313.jpg -- me with hat and false moustache and others
  AIVEN3385.jpg -- my back

Thursday at Crab Week
---------------------

There were 574 photos.

Going through them manually, I found 7 that had my face visible, although in 4
of them I had dark glasses

::

    AIVEN3697.jpg  -- clearly me
    AIVEN3739.jpg  -- me
    AIVEN3797.jpg  -- me looking down with a hat and dark glasses
    AIVEN3798.jpg  -- me with a hat and dark glasses
    AIVEN3933.jpg  -- me looking down, slightly sideways
    AIVEN4277.jpg  -- crowd photo, me with hair down and dark glasses
    AIVEN4281.jpg  -- crowd photo, me with hair down and dark glasses

(I also found group photos and other views that I knew from context had me in
them - that's not something in scope here!)

And here are the first 10 matches from the program

::

   AIVEN3933.jpg  -- me
   AIVEN3697.jpg  -- me
   AIVEN3670.jpg  -- not me, but sort of understandable - beard & glasses
   AIVEN3760.jpg  -- not me, but sort of understandable - beard & glasses
   AIVEN3671.jpg  -- not me, but sort of understandable - beard & glasses
   AIVEN3739.jpg  -- me
   AIVEN3673.jpg  -- not me, but sort of understandable - beard & glasses
   AIVEN3999.jpg  -- not me, but sort of understandable - beard & glasses
   AIVEN4316.jpg  -- not me, but sort of understandable - beard & (dark) glasses
   AIVEN3679.jpg  -- not me, but sort of understandable - beard & glasses

So it's found 3 of the pictures I'd hope it might, with the first being
``AIVEN3933.jpg``, which has me looking down - slightly surprising.

Here's a version of that with other faces hidden:

.. image:: images/AIVEN3933.png
           :width: 53%

The "find faces" program found 24 "faces" in that picture, some of which were
faces or parts of faces (sometimes the same person) and some of which were
difficult to say. It didn't find all of the faces I might have hoped it would.

The "not me, but sort of understandable" results have someone with a beard in
them <smile>. Of course, it's an *assumption* that the "nearness" is being
done for reasons that "make sense" to us <smile>

And luckily for my piece of mind, it did find the photo found in the tutorial.
Here's part of it, cropped just to show the people who gave permission to show
their faces:

.. image:: images/AIVEN3739-cropped.jpg

Ideally, I'd go on and try different tuning factors in the embedding
calculations, and see how that affects things.

   **Note** If I intended to do that a lot, I'd probably want to

   1. give the scripts a switch to allow setting the parameters
   2. either have a database table for each set of parameters, so I could
      search only in the correct table, or put the parameters (perhaps just as
      a single string column) into the table, so I could make the SQL only
      look at appropriate records.

   If I don't do 1, then I'll need to edit the program each time, and if I
   don't do 2, then I'll need to ``DELETE FROM pictures`` each time, and
   moreover I'll need to rerun the "calculate and store embeddings" step if I
   want to re-investigate older embeddings.

So was this a success, so far?
------------------------------

Definitely yes.

I learnt a lot.

I got not awful (!) results with really very low effort.

I know what to do for the next set of investigations, and the data I collect
will be persistent, too.

And my colleague, meanwhile, has been getting great success in using the same
technique to distinguish chihuahuas and muffins:

  https://twitter.com/FTisiot/status/1697589937317564635

  Trying out #PostgreSQL #pgvector on the famous “`Chihuahua vs muffin dataset`_”, results are very good!

.. _`Chihuahua vs muffin dataset`:
   https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification

What I'd do next
----------------

Improve ``find_faces_store_embeddings.py``:

* Add a switch to allow setting the "face detecting" parameters

  This would save me haing to edit the code when I want to explore different
  values.

* Make a different table for each set of parameters

  This would save me havng to ``DELETE FROM pictures`` each time I changed the
  settings, and also save me from having to re-run the "calculate and store
  embeddings" step if I wanted to re-investigate older embeddings.

* Add a switch for "generate reference face"

  Storing the reference face embedding seems sensible. One table for all the
  references might work, and then access them by the filename + settings key.

Some other thoughts:

* Name the output table using the parameter values
* Automatically create it if it doesn't exist
* Probably also allow giving a "base name" if the user wants - this might
  make storing the reference face easier
* The "generate reference face" switch would assume a base name (e.g.
  reference) and would also (by default?) check that there's only one face

Improve ``find_nearby_faces.py``

* Add a switch to specify which face (from the db) to look for
* Add a switch to specify which table to search

Some other thoughts:

* As implied, don't recalculate the reference face each time
  - so need to work out a way of specifying the face that is wanted,
  from the database
* Since tables now have the settings in their names, probably allow

  1. Specify the table name explicitly
  2. Give the same "settings" switches as from find_faces_store_embeddings,
     and recalculate the table name - this may be easier to use!

I'd also expect to share common code, and might look at using classes
to save passing around so many variables


Part the fourth: Why PostgreSQL?
--------------------------------

.. image:: images/PostgreSQL_logo.3colors.120x120.png
           :width: 30%

Why is PostgreSQL a surprising choice?
--------------------------------------

Python is a good fit for data pipelines like this, as it has good bindings to
machine learning packages, and excellent support for talking to PostgreSQL.

So why is PostgreSQL a surprising choice?

Because people assume you need a specialised DB to store embeddings.

So why PostgreSQL?
------------------

.. |hammer| image:: images/hammer-159639_1280.webp
                    :align: middle
                    :width: 256

.. |swiss-army-knife| image:: images/swiss-army-knife-154314_1280.png
                    :align: middle
                    :width: 500

.. raw:: pdf

   Spacer 0 60

|swiss-army-knife| and/or |hammer|

.. raw:: pdf

   Spacer 0 60

* Penknife_ and Hammer_ images from https://pixabay.com/, by `OpenClipart-Vectors`_

.. _Penknife: https://pixabay.com/vectors/swiss-army-knife-pocket-knife-blade-154314/
.. _Hammer: https://pixabay.com/vectors/hammer-tool-craftsman-nail-159639/
.. _`OpenClipart-Vectors`: https://pixabay.com/users/openclipart-vectors-30363/

So why PostgreSQL?
------------------

With caveats, because:

* It's significantly better than nothing
* We already have it
* It can SQL all the things
* Indexing

It's significantly better than nothing
--------------------------------------

There comes a point when you need to store your embeddings in some sort of database, just to keep experimenting

PostgreSQL is a *good* place to start

We already have it
------------------

Quite often, we're already running PostgreSQL

You can SQL all the things together
-----------------------------------

This can be *really useful*:

* Find me things like this order, that are in stock

  It's traditional to look for an e-commerce application!

* Find the pictures of me that were taken in Portugal, between these dates

  If we're working with photographs, it seems natural to store the image
  metadata in the database as well, and then one can make queries based on the
  image and its metadata

* Find all the things that match <these qualities> and choose the one most
  like <this other thing>

  Here, we frame the relational query first, and then qualify it by the vector
  search


PostgreSQL optimisation techniques work
---------------------------------------

You can use all the techniques you normally use in PG to optimise the query

...partition the table, etc...

and can do ANALYZE on the query, too

Indexing
--------

Indexing speeds up the *use* of embeddings.

There are currently two types of index available in pgvector:

* IVFFlat - exact nearest neighbours, slower
* HNSW - approximate nearest neighbours, faster

HNSW was just added in `pgvector 0.5.0`_

.. _`pgvector 0.5.0`: https://jkatz05.com/post/postgres/pgvector-overview-0.5.0/

Quoting https://github.com/pgvector/pgvector#indexing:

    By default, pgvector performs exact nearest neighbor search, which
    provides perfect recall.

    You can add an index to use approximate nearest neighbor search, which
    trades some recall for speed. Unlike typical indexes, you will see
    different results for queries after adding an approximate index.

Since 0.5.0, pgvector supports two types of index:

   * IVFFlat (Inverted File with Flat Compression)

     Quoting https://github.com/pgvector/pgvector#ivfflat:

     "An IVFFlat index divides vectors into lists, and then searches a subset
     of those lists that are closest to the query vector. It has faster build
     times and uses less memory than HNSW, but has lower query performance (in
     terms of speed-recall tradeoff)."

   * HSNW (Hierarchical Navigable Small Worlds)

     Quoting https://github.com/pgvector/pgvector#hnsw:

     "An HNSW index creates a multilayer graph. It has slower build times and
     uses more memory than IVFFlat, but has better query performance (in terms
     of speed-recall tradeoff). There’s no training step like IVFFlat, so the
     index can be created without any data in the table."

So different index types optimise for different things:

* ease of creating the index
* speed of searching in particular ways
* accuracy of search

A recurring pattern
-------------------

As Python programmers, we should recognise this pattern:

  Work in PostgreSQL until it's not suitable for some reason, and *then* move to
  something else

It's like doing an initial implementation in Python, and then re-implementing
in another programming language if necessary.

As pgvector itself says
-----------------------

On the `pgvector GitHub page`_:

    Open-source vector similarity search for Postgres

    Store your vectors with the rest of your data. Supports:

    * exact and approximate nearest neighbor search
    * L2 distance, inner product, and cosine distance
    * any language with a Postgres client

    Plus ACID compliance, point-in-time recovery, JOINs, and all of the other great features of Postgres


When not to use PG?
-------------------

When it can't cope

When it doesn't actually do what you want

When vectors are too big
~~~~~~~~~~~~~~~~~~~~~~~~

The `pgvector Reference`_ section says:

  Each vector takes ``4 * dimensions + 8`` bytes of storage. Each element is a
  single precision floating-point number (like the ``real`` type in Postgres),
  and all elements must be finite (no ``NaN``, ``Infinity`` or ``-Infinity``).

  Vectors can have up to 16,000 dimensions.

.. _`pgvector Reference`: https://github.com/pgvector/pgvector#reference


When vectors are too big to index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

According to the `pgvector FAQ`_

  You can't currently **index** a vector if it has more than 2,000 dimensions

.. _`pgvector FAQ`: https://github.com/pgvector/pgvector#frequently-asked-questions

When there are too many vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

According to the `pgvector FAQ`_

  A non-partitioned table has a limit of 32 TB by default in Postgres. A
  partitioned table can have thousands of partitions of that size.

When you need more speed
~~~~~~~~~~~~~~~~~~~~~~~~

pgvector is ultimately limited by being based on a relational database that is
not, itself, optimised for this task.

But always remember to profile!

When you need a missing distance function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although this can change as new capabilities are added.

When the queries aren't SQL
---------------------------

Relational databases and SQL aren't always the best solution.

For instance, OpenSearch also has vector support.


Other tools
-----------

Is pgvector the only PostgreSQL solution?

Neon_ provides pg_embedding_, which uses an HNSW index

There's `an article by them`_ comparing its performance with the pgvector 0.5.0
HNSW support.

.. _Neon: https://neon.tech/
.. _pg_embedding: https://github.com/neondatabase/pg_embedding
.. _`an article by them`: https://neon.tech/blog/pgvector-meets-hnsw-index


A quick and not very rigorous search gives a variety of open source solutions:

* Weaviate https://weaviate.io/
* Milvus https://milvus.io/
* Qdrant https://qdrant.tech/
* Vespa https://vespa.ai/
* Chroma https://www.trychroma.com/

And some more

* OpenSearch_  has vector database functionality
* SingleStore vector db https://www.singlestore.com/built-in-vector-database/
* Relevance AI vector db https://relevanceai.com/vector-db
* The FAISS library https://faiss.ai/

And see lists like https://byby.dev/vector-databases

.. _OpenSearch: https://opensearch.org/



The future is bright (judging from history)
-------------------------------------------


`Vectors are the new JSON in PostgreSQL`_ by `Jonathan Katz`_ points out that
embeddings in PG are at the point JSON support was some years back.

.. _`Vectors are the new JSON in PostgreSQL`: https://jkatz05.com/post/postgres/vectors-json-postgresql/
.. _`Jonathan Katz`: https://jkatz05.com/

Things will get better and faster and support larger vectors over the next few years.

(I'm also minded of large blob support - TOAST is always an issue, but they
work on it)

Acknowledgements
----------------

Postgres, PostgreSQL and the Slonik Logo are trademarks or registered
trademarks of the PostgreSQL Community Association of Canada, and used with
their permission

* `ML Typewriter`_ image from https://unsplash.com/, by `Markus Winkler`_

* Penknife_ and Hammer_ images from https://pixabay.com/, by `OpenClipart-Vectors`_

* Vector graph from `JCC Math.Phys 191: The vector class`_, `CC BY-SA 3.0`_

.. _Penknife: https://pixabay.com/vectors/swiss-army-knife-pocket-knife-blade-154314/
.. _Hammer: https://pixabay.com/vectors/hammer-tool-craftsman-nail-159639/
.. _`OpenClipart-Vectors`: https://pixabay.com/users/openclipart-vectors-30363/

.. _`JCC Math.Phys 191: The vector class`: http://jccc-mpg.wikidot.com/the-vector-class
.. _`CC BY-SA 3.0`: https://creativecommons.org/licenses/by-sa/3.0/

My colleague Francesco Tisiot for the `original tutorial`_, and much good advice

.. _`original tutorial`: https://aiven.io/developer/find-faces-with-pgvector

References for future reading
-----------------------------

This set of interesting references is not in any way complete, so do your own research!

.. REMEMBER TO UPDATE IN README.rst AS WELL

* `Colorful vectors`_ by JP Hwang (`@_jphwang`), an Educator at Weaviate_,
  which illustrates vector search in RGB space with interactive images
* `Vector Embeddings for Developers: The Basics`_ at Pinecone_ is a nice
  introduction to what vector embeddings are and why they're useful
* `Vector Embeddings Explained`_, again from Weaviate_, is a nice explanation,
  and gives the now classic example of how it allows the computation of::

    king − man + woman ≈ queen

* If you're after a bit more mathematics (and Python code), the Aiven tutorial
  also references Mathias Grønne's `Introduction to Image Embedding and
  Accuracy`_, which uses information about a book as its base. This article
  covers a lot more of the ideas of *embedding*, *similarity* and
  *clustering*.

  And quoting that article:

    The process of representing something in a computer is called embedding.

  So a "vector embedding" is a vector that represents something, or a
  representation of something as an array of numbers.

* In general, `What are embeddings`_ by `Vicki Boykis`_ looks like a very
  useful resource, and the `Next`_ section of that site seems to have lots of
  very interesting references - I especially like `The Illustrated Word2vec`_
  by Jay Alammar, but there's a lot more there that I want to read.

* The article `Face Detection with Haar Cascade`_ by Girija Shankar Behera
  describes how a Haar Cascade model recognises faces.

* `Sebi's demo`_ - my colleague Sébastien Blanc tweeting a video showing his
  cool demo of using ``pgvector`` to find nearest colours by RGB.

And the "`AI and ethics`_" talk from `Write the Docs Atlantic`_ when it's
available - this talk by `Chris Meyns`_ was called "AI ethics for tech writers", but really it
is an excellent talk about the ethics around AI regardless of why you're using it.

.. _`What are embeddings`: https://vickiboykis.com/what_are_embeddings/
.. _`Vicki Boykis`: https://vickiboykis.com/about/
.. _`Next`: https://vickiboykis.com/what_are_embeddings/next.html
.. _`The Illustrated Word2vec`: https://jalammar.github.io/illustrated-word2vec/

.. _`colorful vectors`: https://huggingface.co/spaces/jphwang/colorful_vectors
.. _weaviate: https://weaviate.io/
.. _pinecone: https://www.pinecone.io/
.. _`Introduction to Image Embedding and Accuracy`: https://towardsdatascience.com/introduction-to-image-embedding-and-accuracy-53473e8965f
.. _`Vector Embeddings for Developers: The Basics`: https://www.pinecone.io/learn/vector-embeddings-for-developers/
.. _`Vector Embeddings Explained`: https://weaviate.io/blog/vector-embeddings-explained
.. _`Face Detection with Haar Cascade`: https://towardsdatascience.com/face-detection-with-haar-cascade-727f68dafd08
.. _`Sebi's demo`: https://twitter.com/sebi2706/status/1698715900231184755
.. _`Write the Docs Atlantic`: https://www.writethedocs.org/conf/atlantic/2023/
.. _`AI and ethics`: https://www.writethedocs.org/conf/atlantic/2023/speakers/#speaker-chris-meyns-ai-ethics-for-tech-writers-chris-meyns
.. _`Chris Meyns`: https://www.linkedin.com/in/meyns/

Fin
---

Get a free trial of Aiven services at https://go.aiven.io/pyconuk-signup

Also, we're hiring! See https://aiven.io/careers

Written in reStructuredText_, converted to PDF using rst2pdf_

..
    |cc-attr-sharealike| This slideshow is released under a
    `Creative Commons Attribution-ShareAlike 4.0 International License`_

Slides and accompanying material |cc-attr-sharealike| at
https://github.com/aiven-labs/pgvector-find-faces-talk

.. And that's the end of the slideshow

.. |cc-attr-sharealike| image:: images/cc-attribution-sharealike-88x31.png
   :alt: CC-Attribution-ShareAlike image
   :align: middle

.. _`Creative Commons Attribution-ShareAlike 4.0 International License`: http://creativecommons.org/licenses/by-sa/4.0/

.. _reStructuredText: http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html
.. _rst2pdf: https://rst2pdf.org/
.. _Aiven: https://aiven.io/
