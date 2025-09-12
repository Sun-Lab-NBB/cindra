 .. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (autodoc, etc.).

General Command Line Interfaces
===============================

.. click:: suite2p.cli:generate_configuration_file
   :prog: ss2p-config
   :nested: full

.. click:: suite2p.cli:run_pipeline
   :prog: ss2p-run
   :nested: full

Sun Lab Command Line Interfaces
===============================

.. click:: suite2p.cli:run_single_day_pipeline
   :prog: ss2p-run
   :nested: full

.. click:: suite2p.cli:run_multi_day_pipeline
   :prog: ss2p-run
   :nested: full


Single-Day Pipeline Configuration
=================================
.. automodule:: suite2p.configuration.single_day
   :members:
   :undoc-members:
   :show-inheritance:

Single-Day Pipeline API
=======================
.. automodule:: suite2p.single_day
   :members:
   :undoc-members:
   :show-inheritance:

Multi-Day Pipeline Configuration
=================================
.. automodule:: suite2p.configuration.multi_day
   :members:
   :undoc-members:
   :show-inheritance:

Multi-Day Pipeline API
======================
.. automodule:: suite2p.multi_day
   :members:
   :undoc-members:
   :show-inheritance:

Multi-day Pipeline Algorithms
=============================
.. automodule:: suite2p.multiday
   :members:
   :undoc-members:
   :show-inheritance:

Version API
===========
.. automodule:: suite2p.version
   :members:
   :undoc-members:
   :show-inheritance:
