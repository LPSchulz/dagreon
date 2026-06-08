Getting Started
===============

Dagreon tasks are classes decorated with :func:`dagreon.task`. A task consumes
and produces Python 3.12 type aliases. Dagreon uses those alias objects as graph
nodes.

Define Tasks
------------

.. code-block:: python

   from dagreon import task

   type Samples = list[float]
   type Mean = float


   @task
   class LoadSamples:
       values: tuple[float, ...]

       def __call__(self) -> Samples:
           return list(self.values)


   @task
   class ComputeMean:
       def __call__(self, samples: Samples) -> Mean:
           return sum(samples) / len(samples)

The decorated class becomes a frozen dataclass. Fields on the class are task
configuration values and are included in Dagreon's internal result keys.

Run A Workflow
--------------

.. code-block:: python

   from dagreon import Workflow

   workflow = Workflow([
       LoadSamples(values=(1.0, 2.0, 3.0)),
       ComputeMean(),
   ])

   result = workflow.run(Mean)

Only tasks needed to compute the requested target are executed.

Validate A Target
-----------------

Validation is target-specific:

.. code-block:: python

   workflow.validate(Mean)

Validation checks that required producers exist and that the relevant subgraph is
acyclic.

Use Overrides
-------------

``overrides`` are temporary tasks used for one run. If an override produces a
type that already exists in the workflow, it replaces that producer. If it
produces a missing type, it is added for that run.

.. code-block:: python

   @task
   class AlternateSamples:
       def __call__(self) -> Samples:
           return [10.0, 20.0, 30.0]


   result = workflow.run(Mean, overrides=(AlternateSamples(),))

Run Variants
------------

Use :meth:`dagreon.Workflow.run_variants` to evaluate many temporary task sets.
Each variant can be a single task or a tuple of tasks.

.. code-block:: python

   variants = [
       AlternateSamples(),
       LoadSamples(values=(4.0, 5.0, 6.0)),
       (),  # no per-variant override
   ]

   results = workflow.run_variants(Mean, variants)

The result order matches the variant order. ``run_variants`` shows a progress
bar by default through ``tqdm``.

Persist Results With LMDB
-------------------------

Pass ``db_dir`` to persist final requested targets in LMDB:

.. code-block:: python

   workflow = Workflow(
       [LoadSamples(values=(1.0, 2.0, 3.0)), ComputeMean()],
       db_dir="dagreon-results.lmdb",
   )

   first = workflow.run(Mean)
   second = workflow.run(Mean)

LMDB storage is final-target based. If you request ``Mean``, Dagreon stores the
``Mean`` result for that exact task graph. Intermediate values are not stored
unless you request them as targets in separate calls.

Use ``load_only=True`` to read a stored final target without computing:

.. code-block:: python

   stored = workflow.run(Mean, load_only=True)

If no stored result exists, ``load_only=True`` returns ``None``.
For ``run_variants``, missing load-only results are returned as ``None`` in the
result list.

Use Ray
-------

Ray support is optional:

.. code-block:: bash

   pip install dagreon[ray]

Enable Ray at workflow construction time:

.. code-block:: python

   workflow = Workflow(tasks, use_ray=True)
   result = workflow.run(Mean)
   results = workflow.run_variants(Mean, variants)

``run`` and ``run_variants`` return computed values in both local and Ray mode.
Use ``ray_remote_args`` to pass Ray resource options for a call.

Profile Execution
-----------------

Use :meth:`dagreon.Workflow.profile` to compute a target and receive timing and
result-size measurements:

.. code-block:: python

   report = workflow.profile(Mean)
   print(report.summary())

Profiling runs the target workflow directly; it does not load or store results
through LMDB persistence.
