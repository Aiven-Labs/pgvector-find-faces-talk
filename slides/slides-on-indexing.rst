Extra slides about pgvector and indexing
========================================


.. class:: title-slide-info

    By Tibs (they / he)

    .. raw:: pdf

       Spacer 0 30


    Slides and source code at
    https://github.com/aiven-labs/pgvector-find-faces-talk

    .. raw:: pdf

       Spacer 0 30

.. footer::

   *tony.ibbs@aiven.io*  / *https://aiven.io/tibs*  / *@much_of_a*

   .. Add a bit of space at the bottom of the footer, to stop the underlines
      running into the bottom of the slide
   .. raw:: pdf

      Spacer 0 1

Indexing
--------

Speeds up the *use* of embeddings, but gives *approximate* results.

* IVFFlat: Inverted File with Flat Compression

  Slower to search, quicker to build, smaller index, needs rebuilding

* HNSW: Hierarchical Navigable Small Worlds

  Faster to search, slower to build, adapts to new data

IVFFlat: Inverted File with Flat Compression
--------------------------------------------

.. raw:: pdf

   Spacer 0 10

.. image:: images/IVFFLAT.jpeg
   :width: 80%


HNSW: Hierarchical Navigable Small Worlds
-----------------------------------------

.. image:: images/HNSW.jpeg
   :width: 45%

So which to choose?
~~~~~~~~~~~~~~~~~~~

Advice from `Vector Indexes in Postgres using pgvector: IVFFlat vs HNSW`_

  * If you care more about index size, then choose IVFFlat.
  * If you care more about index build time, then select IVFFlat.
  * If you care more about speed, then choose HNSW.
  * If you expect vectors to be added or modified, then select HNSW.

.. _`Vector Indexes in Postgres using pgvector: IVFFlat vs HNSW`: https://github.com/pgvector/pgvector#ivfflat:
