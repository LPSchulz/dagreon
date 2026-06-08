import pytest

from dagreon import (
    CycleError,
    DuplicateOutputError,
    MissingProducerError,
    ValidationError,
    Workflow,
    task,
)

# Type aliases used across all tests
type A = int
type B = float
type C = str
type D = dict
type E = list


def test_validate_single_task_succeeds():
    """Test validation passes for single task workflow."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 1

    workflow = Workflow([TaskA()])
    workflow.validate(target=A)  # Should not raise

    with pytest.raises(ValidationError):
        workflow.validate(target=B)  # B not produced


def test_validate_simple_chain_succeeds():
    """Test validation passes for simple dependency chain."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 1

    @task
    class TaskB:
        def __call__(self, a: A) -> B:
            return float(a)

    workflow = Workflow([TaskA(), TaskB()])
    workflow.validate(target=B)  # Should not raise


def test_validate_complex_dag_succeeds():
    """Test validation passes for complex DAG."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 1

    @task
    class TaskB:
        def __call__(self) -> B:
            return 2.0

    @task
    class TaskC:
        def __call__(self, a: A, b: B) -> C:
            return str(a + b)

    @task
    class TaskD:
        def __call__(self, c: C) -> D:
            return {"result": c}

    workflow = Workflow([TaskA(), TaskB(), TaskC(), TaskD()])
    workflow.validate(target=D)  # Should not raise


def test_duplicate_output_error():
    """Test validation fails when multiple tasks produce same output type."""

    @task
    class TaskA1:
        def __call__(self) -> A:
            return 1

    @task
    class TaskA2:
        def __call__(self) -> A:
            return 2

    with pytest.raises(
        DuplicateOutputError, match="Multiple tasks produce output type A"
    ):
        Workflow([TaskA1(), TaskA2()])


def test_missing_producer_error():
    """Test validation fails when input type has no producer."""

    @task
    class TaskB:
        def __call__(self, a: A) -> B:  # A is not produced by any task
            return float(a)

    b = TaskB()
    wf = Workflow([b])

    with pytest.raises(
        MissingProducerError, match="Target type A is not produced by any task"
    ):
        wf.validate(target=A)

    with pytest.raises(
        MissingProducerError, match="No task produces required input type A"
    ):
        wf.validate(target=B)


def test_missing_producer_error_in_chain():
    """Test validation fails when middle task input has no producer."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 1

    @task
    class TaskC:
        def __call__(self, a: A, b: B) -> C:  # B is not produced
            return str(a) + str(b)

    wf = Workflow([TaskA(), TaskC()])

    wf.validate(target=A)  # Should not raise
    with pytest.raises(
        MissingProducerError, match="No task produces required input type B"
    ):
        wf.validate(target=C)


def test_cycle_error_simple():
    """Test validation fails when workflow contains a simple cycle."""

    @task
    class TaskA:
        def __call__(self, b: B) -> A:  # A depends on B
            return int(b)

    @task
    class TaskB:
        def __call__(self, a: A) -> B:  # B depends on A -> cycle!
            return float(a)

    workflow = Workflow([TaskA(), TaskB()])

    with pytest.raises(CycleError, match="Workflow contains a cycle"):
        workflow.validate(A)
    with pytest.raises(CycleError, match="Workflow contains a cycle"):
        workflow.validate(B)


def test_cycle_error_complex():
    """Test validation fails when workflow contains a complex cycle."""

    @task
    class TaskA:
        def __call__(self, c: C) -> A:  # A depends on C
            return len(c)

    @task
    class TaskB:
        def __call__(self, a: A) -> B:  # B depends on A
            return float(a)

    @task
    class TaskC:
        def __call__(self, b: B) -> C:  # C depends on B -> cycle A->C->B->A!
            return str(b)

    wf = Workflow([TaskA(), TaskB(), TaskC()])

    with pytest.raises(CycleError, match="Workflow contains a cycle"):
        wf.validate(A)
    with pytest.raises(CycleError, match="Workflow contains a cycle"):
        wf.validate(B)
    with pytest.raises(CycleError, match="Workflow contains a cycle"):
        wf.validate(C)


def test_validate_multiple_independent_chains():
    """Test validation of multiple independent chains."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 1

    @task
    class TaskB:
        def __call__(self, a: A) -> B:
            return float(a)

    @task
    class TaskC:
        def __call__(self) -> C:  # Independent chain
            return "hello"

    @task
    class TaskD:
        def __call__(self, c: C) -> D:
            return {"msg": c}

    workflow = Workflow([TaskA(), TaskB(), TaskC(), TaskD()])
    workflow.validate(target=A)  # Should not raise
    workflow.validate(target=B)  # Should not raise
    workflow.validate(target=C)  # Should not raise
    workflow.validate(target=D)  # Should not raise


def test_validation_preserves_basegraph():
    """Test that validation doesn't modify the graph."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 1

    @task
    class TaskB:
        def __call__(self, a: A) -> B:
            return float(a)

    wf = Workflow([TaskA(), TaskB()])

    original_graph = wf.base_graph

    wf.validate(target=A)
    wf.validate(target=B)

    # Should be unchanged
    assert wf.base_graph is original_graph


def test_validate_diamond_dependency():
    """Test validation of diamond dependency pattern."""

    @task
    class Root:
        def __call__(self) -> A:
            return 1

    @task
    class Left:
        def __call__(self, a: A) -> B:
            return float(a)

    @task
    class Right:
        def __call__(self, a: A) -> C:
            return str(a)

    @task
    class Join:
        def __call__(self, b: B, c: C) -> D:
            return {"left": b, "right": c}

    workflow = Workflow([Root(), Left(), Right(), Join()])
    workflow.validate(target=A)  # Should not raise
    workflow.validate(target=B)  # Should not raise
    workflow.validate(target=C)  # Should not raise
    workflow.validate(target=D)  # Should not raise
