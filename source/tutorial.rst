Tutorial
========
This tutorial shows how to detect new seismic events with template matching starting from a known earthquake (using FMF, of course!). The tutorial is organized into a series of four python jupyter notebooks.

Prerequisite
------------

First, create a new folder called *FMF_tuto* and download the material required by this tutorial at:
`https://www.dropbox.com/sh/3vvwn072o0kde05/AAATww2-BkKRRGf9SyjBjNC8a?dl=0 <https://www.dropbox.com/sh/3vvwn072o0kde05/AAATww2-BkKRRGf9SyjBjNC8a?dl=0>`_
Extract the content of the archive in *FMF_tuto*.


* To avoid issues caused by package discrepancies, we provide a conda environment ready for cloning. If you do not have the anaconda distribution, please have a look at `https://www.anaconda.com/distribution/ <https://www.anaconda.com/distribution/>`_. To clone our environment, open a terminal in *FMF_tuto* and run:

.. code-block:: console

    $ conda create --name FMF_tuto --file FMF_tuto_Python_packages.txt

* Activate your new conda environment with:

.. code-block:: console

    $ source activate FMF_tuto

* You now need to install FMF. Move to the folder *fast_matched_filter* and run:

.. code-block:: console

    $ python setup.py build_ext
    $ pip install .

We also refer you to the :doc:`Introduction <introduction>` tab of this documentation.

Running the Tutorial
--------------------

Open a terminal in *FMF_tuto* and run:

.. code-block:: console

    $ source activate FMF_tuto
    $ jupyter notebook

You now are connected to a jupyter environment, and thus can have an interactive access with the jupyter notebooks. Go through the four notebooks in this order:  

#. build_template_from_catalog.ipynb
#. matched_filter_search.ipynb
#. extract_new_detections.ipynb
#. plot_detections.ipynb

