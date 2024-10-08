=====================================================================
Slides for "How I used PostgreSQL® to find pictures of me at a party"
=====================================================================

These are the slides for the talk "How I used PostgreSQL® to find pictures of me
at a party", given by Tibs_ at
`PyCon UK 2023`_ and at `EuroPython 2024`_

.. _`PyCon UK 2023`: https://2023.pyconuk.org/
.. _`EuroPython 2024`: https://ep2024.europython.eu/
.. _Tibs: https://aiven.io/Tibs
.. _`The version from PyCon UK 2023`: https://www.youtube.com/watch?v=_FqKxKVJGWQ
.. _`The version from EuroPython 2024`: https://www.youtube.com/watch?v=qXiprdhk87g

The videos of the talks are available on YouTube:

* `The version from EuroPython 2024`_
* `The version from PyCon UK 2023`_

The source code
~~~~~~~~~~~~~~~

The source code I wrote while preparing this talk is in the `src <../src/>`_
directory. See its `README <../src/README.md>`_ for how to run them.

The slides
~~~~~~~~~~

The slides are written using reStructuredText_, and thus intended to be
readable as plain text.

The sources for the slides are in `<slides.rst>`_, and this is the version
used at EuroPython.

The sources for the slides *as given at PyCon UK* are in `<slides-at-pyconuk.rst>`_.

Note that github will present the ``.rst`` files in rendered form as HTML,
albeit using their own styling (which is occasionally a bit odd). If you want
to see the original reStructuredText source, you have to click on the "Raw"
link at the top of the file's page.

The PDF slides at 16x9 aspect ratio (`<slides-16x9.pdf>`_ and
`<slides-at-pyconuk.pdf>`_ respectively) are stored here
for convenience.

The PDF files may not always be as up-to-date as the source files, so check
their timestamps.

The QR code on the final slide was generated using the command line program
for qrencode_, which I installed with ``brew install qrencode`` on my Mac.

.. _qrencode: https://fukuchi.org/works/qrencode/

Index slides
~~~~~~~~~~~~

I created some slides around pgvector and indexing, but they made the talk too
long. The corresponding text is still in the notes, and the
slides are in `<slides-on-indexing.rst>`_ and `<slides-on-indexing.pdf>`_.

The notes
~~~~~~~~~

The notes are written using reStructuredText_, and thus intended to be
readable as plain text.

The sources for the notes are in `<notes.rst>`_.

If you want, you can make a PDF of the notes with ``make notes`` - but if you're
just reading here, it's probably easier to read `<notes.rst>`_.

Note that github will present the ``.rst`` files in rendered form as HTML,
albeit using their own styling (which is occasionally a bit odd). If you want
to see the original reStructuredText source, you have to click on the "Raw"
link at the top of the file's page.

Making the PDF files
~~~~~~~~~~~~~~~~~~~~

Make a virtual environment in the traditional way::

  python3 -m venv venv

Activate it::

  source venv/bin/activate

Install the requirements using the ``requirements.txt`` file::

  pip install -r requirements.txt

Or alternatively, be explicit::

  pip install rst2pdf docutils pygments svglibq

You will also need an appropriate ``make`` program if you want to use the
Makefile.

After that, you should be able to use the Makefile to create the PDF files.
For instance::

  $ make pdf

to make them all.

For other things the Makefile can do, use::

  $ make help

.. _reStructuredText: http://docutils.sourceforge.net/rst.html

References for future reading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the same as the section in the notes.

It's not in any way complete, so do your own research!

.. REMEMBER TO UPDATE IN notes.rst AS WELL

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

* In general, `What are embeddings`_ by `Vicki Boykis`_ looks like a very
  useful resource, and the `Next`_ section of that site seems to have lots of
  very interesting references - I especially like `The Illustrated Word2vec`_
  by Jay Alammar, but there's a lot more there that I want to read.

* The article `Face Detection with Haar Cascade`_ by Girija Shankar Behera
  describes how a Haar Cascade model recognises faces.

* `Sebi's demo`_ - my colleague Sébastien Blanc tweeting a video showing his
  cool demo of using ``pgvector`` to find nearest colours by RGB.

* I think `Vector Indexes in Postgres using pgvector: IVFFlat vs HNSW`_ is a
  decent introduction to the two types of index in pgvector, and how to decide
  which you want.

* The talk `AI ethics for tech writers`_ by `Chris Meyns`_ from `Write the
  Docs Atlantic 2023`_ is an excellent talk about the ethics around AI
  regardless of why you're using it.

* `Glyph`_'s `A Grand Unified Theory of the AI Hype Cycle`_ is a fun
  look at the shape of the history of this field. By his non-exhaustive list
  of N, this is my fourth time round the cycle :)


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
.. _`Vector Indexes in Postgres using pgvector: IVFFlat vs HNSW`: https://github.com/pgvector/pgvector#ivfflat:
.. _`Write the Docs Atlantic 2023`: https://www.writethedocs.org/conf/atlantic/2023/
.. _`AI ethics for tech writers`: https://www.youtube.com/watch?v=SDzP6Xs9WoQ
.. _`Chris Meyns`: https://www.linkedin.com/in/meyns/
.. _Glyph: https://blog.glyph.im/pages/about.html
.. _`A Grand Unified Theory of the AI Hype Cycle`: https://blog.glyph.im/2024/05/grand-unified-ai-hype.html


Acknowledgements
~~~~~~~~~~~~~~~~

Postgres, PostgreSQL and the Slonik Logo are trademarks or registered
trademarks of the PostgreSQL Community Association of Canada, and used with
their permission

Images:

* `ML Typewriter`_ image from https://unsplash.com/, by `Markus Winkler`_

* Penknife_ and Hammer_ images from https://pixabay.com/, by `OpenClipart-Vectors`_

* Vector graph from `JCC Math.Phys 191: The vector class`_, `CC BY-SA 3.0`_

.. _Unsplash: `ML Typewriter`_
.. _`ML Typewriter`:
   https://unsplash.com/photos/f57lx37DCM4?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText
.. _`Markus Winkler`: https://unsplash.com/@markuswinkler?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText
.. _Penknife: https://pixabay.com/vectors/swiss-army-knife-pocket-knife-blade-154314/
.. _Hammer: https://pixabay.com/vectors/hammer-tool-craftsman-nail-159639/
.. _`OpenClipart-Vectors`: https://pixabay.com/users/openclipart-vectors-30363/

.. _`JCC Math.Phys 191: The vector class`: http://jccc-mpg.wikidot.com/the-vector-class
.. _`CC BY-SA 3.0`: https://creativecommons.org/licenses/by-sa/3.0/

My colleagues `Francesco Tisiot`_ for the `original tutorial`_, and `Olena Kutsenko`_ for the index diagrams

.. _`original tutorial`: https://aiven.io/developer/find-faces-with-pgvector
.. _`Olena Kutsenko`: https://aiven.io/olena

License
~~~~~~~

|cc-attr-sharealike|

This talk and its related files are released under a `Creative Commons
Attribution-ShareAlike 4.0 International License`_, except as described in the
Acknowledgements_ section above.

.. |cc-attr-sharealike| image:: images/cc-attribution-sharealike-88x31.png
   :alt: CC-Attribution-ShareAlike image

.. _`Creative Commons Attribution-ShareAlike 4.0 International License`: http://creativecommons.org/licenses/by-sa/4.0/ 
