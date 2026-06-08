API Reference
=============

Task Decorator
--------------

.. autofunction:: dagreon.task

Workflow
--------

.. autoclass:: dagreon.Workflow
   :members: validate, run, profile, run_variants

Profiling
---------

.. autoclass:: dagreon.ProfilingReport
   :members:

Validation Errors
-----------------

.. autoclass:: dagreon.ValidationError

.. autoclass:: dagreon.DuplicateOutputError

.. autoclass:: dagreon.MissingProducerError

.. autoclass:: dagreon.CycleError
