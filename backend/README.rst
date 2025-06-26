=======
Backend
=======

To set up your environment, copy the `.env.path` file to `.env`:

.. code-block:: bash
   
   cp .env.path ../.env

The backend server can be run by executing the following command:

.. code-block:: bash

   uv run uvicorn main:app --reload