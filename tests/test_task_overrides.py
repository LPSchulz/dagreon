from dagreon import Workflow, task

# Type aliases used across all tests
type A = int
type B = float
type C = str
type D = dict
type E = list


def test_override_replaces_existing_task():
    """Test that run with overrides replaces a task that produces the same output type."""

    @task
    class OriginalTaskA:
        def __call__(self) -> A:
            return 1

    @task
    class NewTaskA:
        def __call__(self) -> A:
            return 42

    @task
    class TaskB:
        def __call__(self, a: A) -> B:
            return float(a)

    # Create original workflow
    wf = Workflow([OriginalTaskA(), TaskB()])
    result_original = wf.run(B)
    assert result_original == 1.0

    # Replace TaskA with NewTaskA
    result_modified = wf.run(B, overrides=(NewTaskA(),))
    assert result_modified == 42.0


def test_override_preserves_original_workflow():
    """Test that run with overrides doesn't modify the original workflow."""

    @task
    class OriginalTaskA:
        def __call__(self) -> A:
            return 1

    @task
    class NewTaskA:
        def __call__(self) -> A:
            return 42

    wf = Workflow([OriginalTaskA()])
    result_original = wf.run(A)
    assert result_original == 1
    result_modified = wf.run(A, overrides=(NewTaskA(),))
    assert result_modified == 42
    # Original workflow should still return original result
    assert wf.run(A) == 1


def test_replace_multiple_tasks():
    """Test that run with overrides can replace multiple tasks."""

    @task
    class TaskA1:
        def __call__(self) -> A:
            return 1

    @task
    class TaskA2:
        def __call__(self) -> A:
            return 10

    @task
    class TaskB1:
        def __call__(self) -> B:
            return 2.0

    @task
    class TaskB2:
        def __call__(self) -> B:
            return 20.0

    @task
    class TaskC:
        def __call__(self, a: A, b: B) -> C:
            return f"{a},{b}"

    # Start with original workflow
    wf = Workflow([TaskA1(), TaskB1(), TaskC()])
    assert wf.run(C) == "1,2.0"

    # Chain multiple with_task calls to replace tasks
    modified = wf.run(C, overrides=(TaskA2(), TaskB2()))
    assert modified == "10,20.0"


def test_override_replaces_in_complex_dag():
    """Test task replacement in a complex DAG."""

    @task
    class ConfigA:
        def __call__(self) -> A:
            return 5

    @task
    class ConfigB:
        def __call__(self) -> B:
            return 10.0

    @task
    class ProcessC:
        def __call__(self, a: A, b: B) -> C:
            return f"Sum: {a + b}"

    @task
    class FinalD:
        def __call__(self, c: C) -> D:
            return {"result": c}

    # Original workflow
    wf = Workflow([ConfigA(), ConfigB(), ProcessC(), FinalD()])
    original_result = wf.run(D)
    assert original_result == {"result": "Sum: 15.0"}

    # Replace ConfigA with different value
    @task
    class NewConfigA:
        def __call__(self) -> A:
            return 100

    modified_result = wf.run(D, overrides=(NewConfigA(),))
    assert modified_result == {"result": "Sum: 110.0"}


def test_override_adds_task_if_it_does_not_exist():
    """Test that run with overrides can add a new task if no existing task produces
    that output type."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 10

    @task
    class TaskB:
        def __call__(self) -> B:
            return 3.14

    # Create workflow with only TaskA
    wf = Workflow([TaskA()])

    # Try to replace non-existent TaskB
    assert wf.run(A, overrides=(TaskB(),)) == 10  # TaskA should still run
    assert wf.run(B, overrides=(TaskB(),)) == 3.14  # TaskB should run


def test_override_with_new_signature():
    """Test that replacing task with different input signature works."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 1

    @task
    class TaskB:
        def __call__(self) -> B:
            return 3.0

    @task
    class TaskBWithInput:
        def __call__(self, a: A) -> B:  # Different input signature
            return float(a)

    workflow = Workflow([TaskA(), TaskB()])
    assert workflow.run(B) == 3.0
    assert workflow.run(B, overrides=(TaskBWithInput(),)) == 1.0  # Should use TaskA


def test_override_input_order_does_not_matter():
    """Test that replacing task with different number of inputs fails."""

    @task
    class OriginalTaskA:
        def __call__(self, b: B, c: C) -> A:
            return int(b)

    @task
    class CompatibleTaskA:
        def __call__(self, c: C, b: B) -> A:  # different input order
            return int(b) + len(c)

    @task
    class TaskB:
        def __call__(self) -> B:
            return 3.0

    @task
    class TaskC:
        def __call__(self) -> C:
            return "hi"

    wf = Workflow([OriginalTaskA(), TaskB(), TaskC()])

    assert wf.run(A) == 3
    assert wf.run(A, overrides=(CompatibleTaskA(),)) == 5
