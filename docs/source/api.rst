.. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (automodule, etc.).

Data Structures
===============

.. automodule:: cindra.dataclasses
   :members:
   :undoc-members:
   :show-inheritance:

Pipelines
=========

.. automodule:: cindra.pipelines
   :members:
   :undoc-members:
   :show-inheritance:

Registration
============

.. automodule:: cindra.registration
   :members:
   :undoc-members:
   :show-inheritance:

Detection
=========

.. automodule:: cindra.detection
   :members:
   :undoc-members:
   :show-inheritance:

Extraction
==========

.. automodule:: cindra.extraction
   :members:
   :undoc-members:
   :show-inheritance:

Classification
==============

.. automodule:: cindra.classification
   :members:
   :undoc-members:
   :show-inheritance:

File I/O
========

.. automodule:: cindra.io
   :members:
   :undoc-members:
   :show-inheritance:

GUI Viewers
===========

.. automodule:: cindra.gui
   :members:
   :undoc-members:
   :show-inheritance:

Main CLI
========

.. click:: cindra.interface.cli:cindra_cli
   :prog: cindra
   :nested: full

GUI CLI
=======

.. click:: cindra.interface.gui_cli:cindra_gui
   :prog: cindra-gui
   :nested: full
