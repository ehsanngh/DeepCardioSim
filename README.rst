=================================================================
DeepCardioSim: Deep Neural Models for Cardiovascular Simulations
=================================================================

This repository contains code related to deep neural networks and operator models for rapid cardiovascular simulations and efficient inverse modeling.

------------
Installation
------------

Start by cloning the repository:

.. code-block:: bash
   
   git clone git@gitlab.msu.edu:msu-computational-biomechanics-lab/DeepCardioSim.git
   cd DeepCardioSim

We recommend using an Anaconda environment to develop and run this code (`see the Anaconda installation guide <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_). Once Anaconda is installed, you can set up and activate the environment for this package, named ``dcs``, by running the following commands:

.. code-block:: bash
   
   conda env create -f environment.yml
   conda activate dcs

Once the environment is set up, you can install this package in editable mode:

.. code-block:: bash

   pip install -e .


