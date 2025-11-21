=======
Backend
=======

The backend server is built on FastAPI and uses Redis for task queue management.

Setup
-----

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

To set up the environment variables, update `.env.example` and copy it to `../.env`:

.. code-block:: bash
   
   cp .env.example ../.env

Required Files
~~~~~~~~~~~~~~

The following files can be downloaded from the provided SharePoint links:

MODEL_CHECKPOINT: https://michiganstate-my.sharepoint.com/:u:/r/personal/naghavis_msu_edu/Documents/best_model_snapshot_dict.pt?csf=1&web=1&e=wrSun0

REDIS_CONTAINER: https://michiganstate-my.sharepoint.com/:u:/g/personal/naghavis_msu_edu/ERvnokXH_YJPhpmnxVzeR4kBcl7tUgU_F7Xo7kfHz143cw?e=NxDsB3

DATAPROCESSOR: https://michiganstate-my.sharepoint.com/:u:/g/personal/naghavis_msu_edu/EQm2XfK5zWdCgcf5KAkAUBQBUjg6z1yo-bTMGFg81KhZ_A?e=8ordfu

FENICS_CONTAINER: https://michiganstate-my.sharepoint.com/:u:/r/personal/naghavis_msu_edu/Documents/fenics_legacy3.sif?csf=1&web=1&e=219xbM

Please contact me if you have any issues accessing these files.

Running the Backend
-------------------

To start the backend servers:

.. code-block:: bash

   bash run_servers.sh

To stop the backend servers:

.. code-block:: bash

   bash stop_servers.sh

Notes
-----

The current implementation of GPU pooling for the forward_run API only works efficiently when uvicorn is run with 1 worker. Multiple request handling is currently delegated to uvicorn itself.