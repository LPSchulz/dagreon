# Dagreon

Dagreon is a flexible DAG workflow library for researchers, with persistent
results and easy parallel execution.

It lets you define typed task classes, connect them through Python 3.12 type
aliases, and compute only the dependencies required for a requested result.

```python
from dagreon import Workflow, task

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


workflow = Workflow([
    LoadSamples(values=(1.0, 2.0, 3.0)),
    ComputeMean(),
])

result = workflow.run(Mean)
```

## Features

- Typed task graphs built from Python 3.12 type aliases.
- Target-driven execution of only the subgraph needed for a result.
- Per-run task overrides for variants and experiments.
- Optional LMDB-backed result persistence for final requested targets.
- Optional Ray execution through `pip install dagreon[ray]`.
- Progress bars for variant runs.


