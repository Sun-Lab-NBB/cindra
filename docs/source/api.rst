.. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (automodule, etc.).

Layout
======

.. automodule:: cindra.layout
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
===============

.. automodule:: cindra.dataclasses
   :members:
   :undoc-members:
   :show-inheritance:

Orchestration
=============

.. automodule:: cindra.orchestration
   :members:
   :undoc-members:
   :show-inheritance:

.. Documents the package constants explicitly, since the automodule directive above discovers module-level data through
   the source of the module it documents and therefore skips a constant this package re-exports. The directive names the
   defining module rather than the package, because autodoc reads the attribute docstring from that module's source and
   falls back to the docstring of the value's own type when it is pointed at the re-exporting package. The block covers
   exactly the constants the package's __all__ names and the package's own modules define. A constant the package holds
   for its own use stays off the page, and one this page documents under its own module's section renders there.
.. autodata:: cindra.orchestration.jobs.SINGLE_RECORDING_PHASES
.. autodata:: cindra.orchestration.jobs.MULTI_RECORDING_PHASES
.. autodata:: cindra.orchestration.allocation.BINARIZATION_WORKERS
.. autodata:: cindra.orchestration.allocation.REGISTRATION_WORKERS
.. autodata:: cindra.orchestration.allocation.PROCESSING_WORKERS
.. autodata:: cindra.orchestration.allocation.COMBINATION_WORKERS
.. autodata:: cindra.orchestration.allocation.DISCOVERY_WORKERS
.. autodata:: cindra.orchestration.allocation.EXTRACTION_WORKERS
.. autodata:: cindra.orchestration.allocation.RESOURCE_CLASS_BY_JOB_NAME
.. autodata:: cindra.orchestration.footprints.WORKER_MEMORY_MB
.. autodata:: cindra.orchestration.footprints.SPAWNED_CHILD_MEMORY_MB
.. autodata:: cindra.orchestration.footprints.MEMORY_ESTIMATE_TOLERANCE

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

.. Documents the package constants explicitly, since the automodule directive above discovers module-level data through
   the source of the module it documents and therefore skips a constant this package re-exports. The directive names the
   defining module rather than the package, because autodoc reads the attribute docstring from that module's source and
   falls back to the docstring of the value's own type when it is pointed at the re-exporting package. The block covers
   exactly the constants the package's __all__ names and the package's own modules define. A constant the package holds
   for its own use stays off the page, and one this page documents under its own module's section renders there.
.. autodata:: cindra.io.tiff.TIFF_EXTENSIONS
.. autodata:: cindra.io.tiff.TIFF_DECODE_CEILING
.. autodata:: cindra.io.context.MAXIMUM_CHANNEL_COUNT

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
