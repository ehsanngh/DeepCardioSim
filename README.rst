=====================================================================
DeepCardioSim: Advanced Deep Learning for Cardiovascular Simulations
=====================================================================

This repository contains code for deep neural networks and operator models designed for rapid cardiovascular simulations and efficient inverse modeling. The GUI for using the models is available at `DeepCardioSim-GUI <https://dcsim.egr.msu.edu/>`_.

-------------
Installation
-------------

To get started, clone the repository:

.. code-block:: bash
   
   git clone git@gitlab.msu.edu:msu-computational-biomechanics-lab/DeepCardioSim.git
   cd DeepCardioSim

We recommend using `uv` for dependency management. For more information about `uv`, please refer to the `official documentation <https://docs.astral.sh/uv/>`_. To initialize the virtual environment, run:

.. code-block:: bash

   uv sync --extra plot

Or for a full installation, run:

.. code-block:: bash

   uv sync --all-extras


There is an issue with meshio that, based on `this pull request <https://github.com/nschloe/meshio/pull/1461/commits/3f1161bf786691206e72706404853aa5d8a2cf13>`_, can be manually fixed by running:

.. code-block:: bash

   sed -i '265s|=.*|= self.filename.with_suffix(".h5")|' ./.venv/lib/python3.12/site-packages/meshio/xdmf/time_series.py

----------------------------------------
Publishing to GitHub (excluding backend)
----------------------------------------

To publish this repository to GitHub while excluding the backend code, 
remove this section of the README and follow these steps:

.. code-block:: bash

   git checkout -b public
   git rm -r --cached backend
   git rm -r --cached README.rst
   rm -rf backend
   git add README.rst
   git commit -m "commit message"
   git push github public:main --force-with-lease

You can then switch back to the main branch and remove the temporary branch:

.. code-block:: bash

   git checkout main
   git branch -D public
