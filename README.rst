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

----------------------------------------
Publishing to GitHub (excluding backend)
----------------------------------------

To publish this repository to GitHub while excluding the backend code, 
remove the backend directory, remove this section of the README, and follow these steps:

.. code-block:: bash

   git checkout -b public
   git rm -r --cached backend
   git commit -m "commit message"
   git push github public:main

You can then switch back to the main branch and remove the temporary branch:

.. code-block:: bash

   git checkout main
   git branch -D public
