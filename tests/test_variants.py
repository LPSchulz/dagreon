import pytest

from dagreon import Workflow, task
from dagreon._errors import DuplicateOutputError

# Test type aliases
type A = int
type B = float
type C = str
type D = bool


def test_run_variants_single_tasks():
    """Test run_variants with individual task replacements."""

    @task
    class TaskA1:
        def __call__(self) -> A:
            return 1

    @task
    class TaskA2:
        def __call__(self) -> A:
            return 2

    @task
    class TaskA3:
        def __call__(self) -> A:
            return 3

    @task
    class TaskB:
        def __call__(self, a: A) -> B:
            return float(a * 10)

    # Create base workflow
    workflow = Workflow([TaskA1(), TaskB()])

    # Run variants with different A tasks
    variants = [TaskA2(), TaskA3()]
    results = workflow.run_variants(B, variants)

    assert len(results) == 2
    assert results[0] == 20.0  # TaskA2 gives 2, TaskB gives 2*10=20
    assert results[1] == 30.0  # TaskA3 gives 3, TaskB gives 3*10=30


def test_run_variants_tuple_replacements():
    """Test run_variants with tuple of tasks to replace."""

    @task
    class TaskA1:
        def __call__(self) -> A:
            return 1

    @task
    class TaskA2:
        def __call__(self) -> A:
            return 2

    @task
    class TaskB1:
        def __call__(self, a: A) -> B:
            return float(a * 10)

    @task
    class TaskB2:
        def __call__(self, a: A) -> B:
            return float(a * 100)

    @task
    class TaskC:
        def __call__(self, b: B) -> C:
            return f"Result: {b}"

    # Create base workflow
    workflow = Workflow([TaskA1(), TaskB1(), TaskC()])

    # Run variants with tuples of task replacements
    variants = [
        (TaskA2(), TaskB2()),  # Replace both A and B tasks
        TaskA2(),  # Replace only A task
    ]
    results = workflow.run_variants(C, variants)

    assert len(results) == 2
    assert results[0] == "Result: 200.0"  # TaskA2(2) -> TaskB2(200) -> TaskC
    assert results[1] == "Result: 20.0"  # TaskA2(2) -> TaskB1(20) -> TaskC


def test_run_variants_empty_list():
    """Test run_variants with empty variants list."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 42

    workflow = Workflow([TaskA()])
    results = workflow.run_variants(A, [])

    assert results == []


def test_run_variants_preserves_original():
    """Test that run_variants doesn't modify the original workflow."""

    @task
    class TaskA1:
        def __call__(self) -> A:
            return 1

    @task
    class TaskA2:
        def __call__(self) -> A:
            return 2

    workflow = Workflow([TaskA1()])
    original_result = workflow.run(A)

    # Run variants
    variants = [TaskA2()]
    results = workflow.run_variants(A, variants)

    # Check original workflow is unchanged
    assert workflow.run(A) == original_result == 1
    assert results[0] == 2


def test_run_variants_complex_dag():
    """Test run_variants with a complex DAG."""

    @task
    class TaskA1:
        def __call__(self) -> A:
            return 1

    @task
    class TaskA2:
        def __call__(self) -> A:
            return 10

    @task
    class TaskB:
        def __call__(self, a: A) -> B:
            return float(a * 2)

    @task
    class TaskC:
        def __call__(self, a: A) -> C:
            return str(a * 3)

    @task
    class TaskD1:
        def __call__(self, b: B, c: C) -> D:
            return len(c) > int(b)

    @task
    class TaskD2:
        def __call__(self, b: B, c: C) -> D:
            return int(b) > len(c)

    # Create base workflow
    workflow = Workflow([TaskA1(), TaskB(), TaskC(), TaskD1()])

    # Run variants
    variants = [
        TaskA2(),  # Change input source
        TaskD2(),  # Change final computation
        (TaskA2(), TaskD2()),  # Change both
    ]
    results = workflow.run_variants(D, variants)

    # TaskA1=1, TaskB=2.0, TaskC="3", TaskD1: len("3")>2 -> 1>2 -> False
    original_result = workflow.run(D)
    assert original_result is False

    # Variant 1: TaskA2=10, TaskB=20.0, TaskC="30", TaskD1: len("30")>20 -> 2>20 -> False
    assert results[0] is False

    # Variant 2: TaskA1=1, TaskB=2.0, TaskC="3", TaskD2: 2>1 -> True
    assert results[1] is True

    # Variant 3: TaskA2=10, TaskB=20.0, TaskC="30", TaskD2: 20>2 -> True
    assert results[2] is True


def test_run_variants_mixed_single_and_tuple():
    """Test run_variants with mixed single tasks and tuples."""

    @task
    class TaskA1:
        def __call__(self) -> A:
            return 1

    @task
    class TaskA2:
        def __call__(self) -> A:
            return 2

    @task
    class TaskB1:
        def __call__(self, a: A) -> B:
            return float(a * 5)

    @task
    class TaskB2:
        def __call__(self, a: A) -> B:
            return float(a * 7)

    workflow = Workflow([TaskA1(), TaskB1()])

    variants = [
        TaskA2(),  # Single task replacement
        TaskB2(),  # Different single task replacement
        (TaskA2(), TaskB2()),  # Tuple replacement
        (),  # Empty tuple (no replacements)
    ]

    results = workflow.run_variants(B, variants)

    assert len(results) == 4
    assert results[0] == 10.0  # TaskA2(2) -> TaskB1(2*5=10)
    assert results[1] == 7.0  # TaskA1(1) -> TaskB2(1*7=7)
    assert results[2] == 14.0  # TaskA2(2) -> TaskB2(2*7=14)
    assert results[3] == 5.0  # TaskA1(1) -> TaskB1(1*5=5) (no changes)


def test_run_variants_multiple_replacements_error():
    """Test that one cannot replace one task multiple times in a single variant."""

    @task
    class TaskA1:
        def __call__(self) -> A:
            return 1

    @task
    class TaskA2:
        def __call__(self) -> A:
            return 2

    @task
    class TaskA3:
        def __call__(self) -> A:
            return 3

    @task
    class TaskB:
        def __call__(self, a: A) -> B:
            return float(a * 10)

    workflow = Workflow([TaskA1(), TaskB()])

    variants = [(TaskA2(), TaskA3())]  # Both try to replace TaskA1
    # This should fail because there each variant can only replace existing tasks once
    with pytest.raises(
        DuplicateOutputError, match="Multiple tasks produce output type A"
    ):
        workflow.run_variants(B, variants)

    # instead, the user should do this
    variants = [TaskA2(), TaskA3()]  # Only replace once, but runs twice
    results = workflow.run_variants(B, variants)
    assert results[0] == 20.0  # TaskA2 gives 2
    assert results[1] == 30.0  # TaskA3 gives 3


def test_run_variants_immutability():
    """Test that each variant creates independent workflow instances."""

    @task
    class TaskA1:
        def __call__(self) -> A:
            return 1

    @task
    class TaskA2:
        def __call__(self) -> A:
            return 2

    @task
    class TaskA3:
        def __call__(self) -> A:
            return 3

    workflow = Workflow([TaskA1()])

    # Run multiple variants
    variants = [TaskA2(), TaskA3()]
    results = workflow.run_variants(A, variants)

    # Each run should be independent
    assert results[0] == 2
    assert results[1] == 3

    # Original workflow should be unchanged
    assert workflow.run(A) == 1
