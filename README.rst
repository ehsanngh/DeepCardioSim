=================================================================
DeepCardioSim: Deep Neural Models for Cardiovascular Simulations
=================================================================

This repository contains code for deep neural networks and operator models designed for rapid cardiovascular simulations and efficient inverse modeling.

------------
Installation
------------

To get started, clone the repository:

.. code-block:: bash
   
   git clone git@gitlab.msu.edu:msu-computational-biomechanics-lab/DeepCardioSim.git
   cd DeepCardioSim

We recommend using `uv` for development and execution of this codebase. For more information about `uv`, please refer to the `official documentation <https://docs.astral.sh/uv/>`_. To initialize the virtual environment, run:

.. code-block:: bash

   uv sync --all-extras

------------
Publishing to GitHub
------------

To publish this repository to GitHub while excluding the backend code, follow these steps:

.. code-block:: bash

   git checkout -b public
   git rm -r --cached backend
   git commit -m "commit message"
   git push github public:main