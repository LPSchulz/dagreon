import pytest

from dagreon import ValidationError, Workflow, task

# Type aliases used across all tests
type A = int
type B = float
type C = str
type D = dict
type E = list


def test_run_single_task():
    """Test running workflow with single leaf task."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 42

    workflow = Workflow([TaskA()])
    result = workflow.run(A)

    assert result == 42


def test_run_simple_chain():
    """Test running workflow with simple dependency chain."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 10

    @task
    class TaskB:
        def __call__(self, a: A) -> B:
            return float(a * 2)

    workflow = Workflow([TaskA(), TaskB()])
    result = workflow.run(B)

    assert result == 20.0


def test_run_multiple_dependencies():
    """Test running workflow where task depends on multiple inputs."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 5

    @task
    class TaskB:
        def __call__(self) -> B:
            return 3.14

    @task
    class TaskC:
        def __call__(self, a: A, b: B) -> C:
            return f"A={a}, B={b}"

    workflow = Workflow([TaskA(), TaskB(), TaskC()])
    result = workflow.run(C)

    assert result == "A=5, B=3.14"


def test_run_complex_dag():
    """Test running complex DAG with multiple levels."""

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
            return {"result": c, "length": len(c)}

    @task
    class TaskE:
        def __call__(self, a: A, d: D) -> E:
            return [a] * d["length"]

    workflow = Workflow([TaskA(), TaskB(), TaskC(), TaskD(), TaskE()])
    result = workflow.run(E)

    assert result == [1, 1, 1]  # "3.0" has length 3, so [1, 1, 1]


def test_run_diamond_dependency():
    """Test running workflow with diamond dependency pattern."""

    @task
    class Root:
        def __call__(self) -> A:
            return 10

    @task
    class Left:
        def __call__(self, a: A) -> B:
            return float(a * 2)

    @task
    class Right:
        def __call__(self, a: A) -> C:
            return f"value={a}"

    @task
    class Join:
        def __call__(self, b: B, c: C) -> D:
            return {"left": b, "right": c}

    workflow = Workflow([Root(), Left(), Right(), Join()])
    result = workflow.run(D)

    expected = {"left": 20.0, "right": "value=10"}
    assert result == expected


def test_run_partial_execution():
    """Test running workflow for intermediate results (not final node)."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 100

    @task
    class TaskB:
        def __call__(self, a: A) -> B:
            return float(a / 2)

    @task
    class TaskC:
        def __call__(self, b: B) -> C:
            return f"Final: {b}"

    workflow = Workflow([TaskA(), TaskB(), TaskC()])

    # Run for intermediate result B
    result_b = workflow.run(B)
    assert result_b == 50.0

    # Run for final result C
    result_c = workflow.run(C)
    assert result_c == "Final: 50.0"


def test_run_task_instances_with_state():
    """Test that task instances maintain their state during execution."""

    @task
    class ConfigTask:
        config_value: int  # Declare as field for frozen dataclass

        def __call__(self) -> A:
            return self.config_value

    @task
    class ProcessTask:
        multiplier: int  # Declare as field for frozen dataclass

        def __call__(self, a: A) -> B:
            return float(a * self.multiplier)

    config_task = ConfigTask(config_value=7)
    process_task = ProcessTask(multiplier=3)

    workflow = Workflow([config_task, process_task])
    result = workflow.run(B)

    assert result == 21.0


def test_run_validation_called_automatically():
    """Test that run() automatically validates the workflow."""

    @task
    class TaskA:
        def __call__(self, b: B) -> A:  # Missing producer for B
            return 1

    workflow = Workflow([TaskA()])

    with pytest.raises(ValidationError):
        workflow.run(A)


def test_run_validation_error():
    """Test that run() fails for unreachable target."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 1

    workflow = Workflow([TaskA()])

    with pytest.raises(ValidationError):
        workflow.run(B)  # B is not produced by any task


def test_run_execution_order_correct():
    """Test that tasks are executed in correct topological order."""
    execution_log = []

    @task
    class TaskA:
        def __call__(self) -> A:
            execution_log.append("A")
            return 1

    @task
    class TaskB:
        def __call__(self, a: A) -> B:
            execution_log.append("B")
            return float(a)

    @task
    class TaskC:
        def __call__(self, b: B) -> C:
            execution_log.append("C")
            return str(b)

    workflow = Workflow([TaskA(), TaskB(), TaskC()])
    result = workflow.run(C)

    # Verify execution order: A must come before B, B must come before C
    assert execution_log == ["A", "B", "C"]
    assert result == "1.0"


def test_run_independent_tasks_can_run_in_any_order():
    """Test that independent tasks can run in any valid order."""
    execution_log = []

    @task
    class TaskA:
        def __call__(self) -> A:
            execution_log.append("A")
            return 1

    @task
    class TaskB:
        def __call__(self) -> B:
            execution_log.append("B")
            return 2.0

    @task
    class TaskC:
        def __call__(self, a: A, b: B) -> C:
            execution_log.append("C")
            return str(a + b)

    workflow = Workflow([TaskA(), TaskB(), TaskC()])
    result = workflow.run(C)

    # A and B can run in any order, but both must come before C
    assert execution_log.count("A") == 1
    assert execution_log.count("B") == 1
    assert execution_log.count("C") == 1
    assert execution_log.index("C") == 2  # C must be last
    assert result == "3.0"


def test_run_empty_workflow_fails():
    """Test that running empty workflow fails gracefully."""
    workflow = Workflow([])

    with pytest.raises(ValidationError):
        workflow.run(A)
