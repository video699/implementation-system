video699 – Aligns lecture recordings with study materials
=========================================================

.. image:: https://circleci.com/gh/xbankov/implementation-system/tree/master.svg?style=shield
      :target: https://circleci.com/gh/xbankov/implementation-system/tree/master
      :alt: Continuous Integration Status
.. image:: https://api.codacy.com/project/badge/Grade/9f68a717ab764173a60a2f7b916a25f0
   :alt: Codacy Badge
   :target: https://app.codacy.com/manual/xbankov/implementation-system?utm_source=github.com&utm_medium=referral&utm_content=xbankov/implementation-system&utm_campaign=Badge_Grade_Dashboard
.. image:: https://readthedocs.org/projects/implementation-system/badge/?version=latest
   :target: https://implementation-system.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://img.shields.io/badge/python-3.7-3776AB.svg?logo=python
      :alt: Python Version
.. image:: https://img.shields.io/badge/platform-linux-%23AA4400.svg?logo=linux
      :alt: Platform
.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
      :alt: License
      
One Paragraph of project description goes here

Getting Started
---------------

These instructions will get you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on how to deploy the project on a live system.

Prerequisites
~~~~~~~~~~~~~

You need the following tools for the installation:

- Git >= 1.5.3
- Python >= 3.5

You also need at least the following Python packages:

- `Pip <https://pypi.org/project/pip/>`__ >= 1.4
- `Setuptools <https://pypi.org/project/setuptools/>`__ >= 0.8

You also need to install the libxml2, and libxslt libraries, which are
required by the `lxml <https://pypi.org/project/lxml/>`__ Python package. To
install the libxml2, and libxslt libraries, follow the instructions in the
`lxml documentation <https://lxml.de/installation.html#requirements>`__.

You also need to install the libspatial library, which is required by the
`Rtree <https://pypi.org/project/Rtree/>`__ Python package.  To install the
libspatial library, follow the instructions in the `Rtree documentation
<http://toblerity.org/rtree/install.html>`__.

If you are using a platform for which wheel binary packages are not available,
you may need to perform extra steps. Consult the documentation of the
`OpenCV <https://pypi.org/project/opencv-python/>`__, and
`Shapely <https://pypi.org/project/Shapely/>`__ packages for further details.

.. What other things you need to install the software and how to install them

.. ::

..    Give examples

Installing
~~~~~~~~~~

To install the development version of the package, first clone the Git
repository, and the Git submodules inside it:

::

   $ git clone --recurse-submodules https://github.com/video699/implementation-system.git

Next, install the package using Pip:

::

   $ pip install .

If you wish to use the GPU to accelerate tensor operations in the
``video699.event.siamese`` module, consult the documentation of the
`TensorFlow <https://www.tensorflow.org/install/gpu>`__ package. Pip
will only install the CPU version of the package by default.

If you wish to run tests, or build the documentation, use Pip to download
additional Python packages specified in the ``requirements.txt`` file:

::

   $ pip install -r requirements.txt

.. A step by step series of examples that tell you how to get a development
   env running

.. Say what the step will be

.. ::

..    Give the example

.. And repeat

.. ::

..    until finished

.. End with an example of getting some data out of the system or using it
.. for a little demo

Running the tests
-----------------

Running automated tests is a good way to check that you installed the package
correctly, or that your change to the package did not break any functionality
covered by the tests. To run automated tests, use the following command:

::

   $ python setup.py test

.. Explain how to run the automated tests for this system

.. Break down into end to end tests
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. Explain what these tests test and why

.. ::

..    Give an example

.. And coding style tests
.. ~~~~~~~~~~~~~~~~~~~~~~

.. Explain what these tests test and why

.. ::

..    Give an example

.. Deployment
.. ----------

.. Add additional notes about how to deploy this on a live system

.. Built With
.. ----------

.. -  `Dropwizard <http://www.dropwizard.io/1.0.2/docs/>`__ - The web
..    framework used
.. -  `Maven <https://maven.apache.org/>`__ - Dependency Management
.. -  `ROME <https://rometools.github.io/rome/>`__ - Used to generate RSS
..    Feeds

.. Contributing
.. ------------

.. Please read
.. `CONTRIBUTING.md <https://gist.github.com/PurpleBooth/b24679402957c63ec426>`__
.. for details on our code of conduct, and the process for submitting pull
.. requests to us.

Versioning
----------

We use `SemVer <http://semver.org/>`__ for versioning. For the versions
available, see the `tags on this
repository <https://github.com/video699/implementation-system/tags>`__.

Authors
-------

-  **Vít Novotný** - *Initial work* – `witiko <https://github.com/witiko>`__

See also the list of `contributors
<https://github.com/video699/implementation-system/contributors>`__ who
participated in this project.

License
-------

This project is licensed under the GNU GPLv3 License – see the
`LICENSE <LICENSE>`__ file for details.  Note that the project uses
the `PyMuPDF <https://pypi.org/project/PyMuPDF/>`__ library, which is
released under AGPLv3. Under clause 13 of the AGPLv3, you must provide access
to source code of PyMuPDF if you use video699 in a web service.

.. Acknowledgments
.. ---------------
.. -  Hat tip to anyone whose code was used
.. -  Inspiration
.. -  etc
