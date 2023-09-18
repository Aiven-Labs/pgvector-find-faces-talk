=====================================================================
Slides for "How I used PostgreSQL® to find pictures of me at a party"
=====================================================================

These are the slides for the talk "How I used PostgreSQL® to find pictures of me
at a party", (to be) given by Tibs_ at
`PyCon UK 2023`_.

.. _`PyCon UK 2023`: https://2023.pyconuk.org/
.. _Tibs: https://aiven.io/Tibs

A link to the video recording will be provided when that is available.


The source code
~~~~~~~~~~~~~~~

The source code I wrote while preparing this talk is in the `src <../src/>`_
directory. See its `README <../src/README.md>`_ for how to run them.

The slides
~~~~~~~~~~

The slides are written using reStructuredText_, and thus intended to be
readable as plain text.

The sources for the slides are in `<slides.rst>`_.

Note that github will present the ``.rst`` files in rendered form as HTML,
albeit using their own styling (which is occasionally a bit odd). If you want
to see the original reStructuredText source, you have to click on the "Raw"
link at the top of the file's page.

The PDF slides at 16x9 aspect ratio (`<slides-16x9.pdf>`_) are stored here
for convenience.

The PDF files may not always be as up-to-date as the source files, so check
their timestamps.

The QR code on the final slide was generated using the command line program
for qrencode_, which I installed with ``brew install qrencode`` on my Mac.

.. _qrencode: https://fukuchi.org/works/qrencode/

The notes
~~~~~~~~~

The notes are written using reStructuredText_, and thus intended to be
readable as plain text.

The sources for the notes are in `<notes.rst>`_.

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

.. include:: references.rst



Acknowledgements
~~~~~~~~~~~~~~~~

Postgres, PostgreSQL and the Slonik Logo are trademarks or registered
trademarks of the PostgreSQL Community Association of Canada, and used with
their permission

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

My colleague Francesco Tisiot for the `original tutorial`_, and much good advice

.. _`original tutorial`: https://aiven.io/developer/find-faces-with-pgvector

--------

  |cc-attr-sharealike|

  This talk and its related files are released under a `Creative Commons
  Attribution-ShareAlike 4.0 International License`_. The source code for the
  demo programs is dual-licensed as CC Attribution Share Alike and MIT.

.. |cc-attr-sharealike| image:: images/cc-attribution-sharealike-88x31.png
   :alt: CC-Attribution-ShareAlike image

.. _`Creative Commons Attribution-ShareAlike 4.0 International License`: http://creativecommons.org/licenses/by-sa/4.0/
