import gc
import shutil
import time
from pathlib import Path

import pytest

from dagreon import Workflow, task

# Type aliases
type A = int
type B = float
type C = str


@pytest.fixture(autouse=True)
def _cleanup_test_database_dirs():
    """Clean up test database directories after each test."""
    yield

    # On Windows, LMDB files can be briefly locked; give GC a chance and retry.
    gc.collect()
    test_dirs = [
        Path("test_db_basic"),
        Path("test_db_readonly"),
        Path("test_db_writeable"),
        Path("test_db_persist"),
        Path("test_db_variants"),
        Path("test_db_parallel"),
        Path("test_db_fingerprint"),
        Path("test_db_complex"),
        Path("test_db_load_only"),
    ]
    for db_dir in test_dirs:
        if not db_dir.exists():
            continue
        last_error: Exception | None = None
        for _ in range(5):
            try:
                shutil.rmtree(db_dir)
                last_error = None
                break
            except Exception as e:
                last_error = e
                gc.collect()
                time.sleep(0.05)
        if last_error is not None:
            raise last_error


def test_workflow_without_database_does_not_persist():
    """Test that Workflow without database recomputes each run."""
    call_count = {"A": 0}

    @task
    class TaskA:
        def __call__(self) -> A:
            call_count["A"] += 1
            return 42

    wf = Workflow([TaskA()], db_dir=None)
    assert wf.run(A) == 42
    wf = Workflow([TaskA()], db_dir=None)
    assert wf.run(A) == 42
    wf = Workflow([TaskA()], db_dir=None)
    assert wf.run(A) == 42

    # Each run should recompute
    assert call_count["A"] == 3


def test_no_database_is_default():
    """Test that Workflow recomputes each run by default."""
    call_count = {"A": 0}

    @task
    class TaskA:
        def __call__(self) -> A:
            call_count["A"] += 1
            return 42

    wf = Workflow([TaskA()])
    assert wf.run(A) == 42
    wf = Workflow([TaskA()])
    assert wf.run(A) == 42
    wf = Workflow([TaskA()])
    assert wf.run(A) == 42

    # Each run should recompute
    assert call_count["A"] == 3


def test_database_persists_across_runs():
    """Test that results are persisted and reused across runs."""
    db_path = Path("test_db_persist")
    call_count = {"A": 0, "B": 0}

    @task
    class TaskA:
        def __call__(self) -> A:
            call_count["A"] += 1
            return 10

    @task
    class TaskB:
        def __call__(self, a: A) -> B:
            call_count["B"] += 1
            return float(a * 2)

    wf = Workflow([TaskA(), TaskB()], db_dir=db_path)

    # First run - should compute
    result1 = wf.run(B)
    assert result1 == 20.0
    assert call_count == {"A": 1, "B": 1}

    # Second run - should load from database
    result2 = wf.run(B)
    assert result2 == 20.0
    assert call_count == {"A": 1, "B": 1}  # Not incremented


def test_database_different_configs_have_different_keys():
    """Test that different task configurations result in different storage keys."""
    db_path = Path("test_db_fingerprint")
    call_count = {"A": 0}

    @task
    class ConfigTask:
        value: int

        def __call__(self) -> A:
            call_count["A"] += 1
            return self.value * 10

    wf = Workflow([ConfigTask(value=1)], db_dir=db_path)

    # First config
    result1 = wf.run(A)
    assert result1 == 10
    assert call_count["A"] == 1

    # Same config - should use database
    result2 = wf.run(A)
    assert result2 == 10
    assert call_count["A"] == 1

    # Different config - should recompute
    result3 = wf.run(A, overrides=(ConfigTask(value=2),))
    assert result3 == 20
    assert call_count["A"] == 2


def test_database_with_variants():
    """Test database persistence with variant execution."""
    db_path = Path("test_db_variants")
    call_count = {"A": 0}

    @task
    class TaskA:
        value: int

        def __call__(self) -> A:
            call_count["A"] += 1
            return self.value

    wf = Workflow([TaskA(value=0)], db_dir=db_path)

    variants = [TaskA(value=1), TaskA(value=2), TaskA(value=3)]
    results1 = wf.run_variants(A, variants)
    assert results1 == [1, 2, 3]
    assert call_count["A"] == 3

    # Running same variants again should use database
    results2 = wf.run_variants(A, variants)
    assert results2 == [1, 2, 3]
    assert call_count["A"] == 3  # Not incremented


def test_database_with_complex_dag():
    """Test database with complex DAG structure."""
    db_path = Path("test_db_complex")
    call_count = {"A": 0, "B": 0, "C": 0}

    @task
    class TaskA:
        def __call__(self) -> A:
            call_count["A"] += 1
            return 5

    @task
    class TaskB:
        def __call__(self, a: A) -> B:
            call_count["B"] += 1
            return float(a * 2)

    @task
    class TaskC:
        def __call__(self, a: A, b: B) -> C:
            call_count["C"] += 1
            return f"A={a}, B={b}"

    wf = Workflow([TaskA(), TaskB(), TaskC()], db_dir=db_path)

    result1 = wf.run(C)
    assert result1 == "A=5, B=10.0"
    assert call_count == {"A": 1, "B": 1, "C": 1}

    # Second run - C from database (but A and B are recomputed since
    # only the target result is persisted, not intermediate results)
    result2 = wf.run(C)
    assert result2 == "A=5, B=10.0"
    assert call_count == {"A": 1, "B": 1, "C": 1}  # C loaded from DB

    # Running for intermediate target B requires recomputation of A and B
    # since B was not the target of the first run
    result3 = wf.run(B)
    assert result3 == 10.0
    # B was computed as part of C's computation but persisted under C's key
    # So B needs to be computed when requested directly
    assert call_count["B"] == 2  # B recomputed for direct target


def test_run_load_only_returns_persisted_result():
    """Test that run() with load_only=True loads result from database without computing."""
    db_path = Path("test_db_load_only")
    call_count = {"A": 0}

    @task
    class TaskA:
        value: int

        def __call__(self) -> A:
            call_count["A"] += 1
            return self.value * 10

    wf = Workflow([TaskA(value=5)], db_dir=db_path)

    # First run - compute and persist
    result1 = wf.run(A)
    assert result1 == 50
    assert call_count["A"] == 1

    # Second run with load_only - should load from database without computing
    result2 = wf.run(A, load_only=True)
    assert result2 == 50
    assert call_count["A"] == 1  # Not incremented


def test_run_load_only_returns_none_for_missing_result():
    """Test that run() with load_only=True returns None when result is not in database."""
    db_path = Path("test_db_load_only")
    call_count = {"A": 0}

    @task
    class TaskA:
        value: int

        def __call__(self) -> A:
            call_count["A"] += 1
            return self.value

    wf = Workflow([TaskA(value=1)], db_dir=db_path)

    # Try to load without prior computation - should return None
    result = wf.run(A, load_only=True)
    assert result is None
    assert call_count["A"] == 0  # Never computed


def test_run_load_only_without_database_returns_none():
    """Test that run() with load_only=True returns None when no database is configured."""

    @task
    class TaskA:
        def __call__(self) -> A:
            return 42

    wf = Workflow([TaskA()], db_dir=None)

    result = wf.run(A, load_only=True)
    assert result is None


def test_run_variants_load_only_returns_persisted_results():
    """Test that run_variants() with load_only=True loads results from database."""
    db_path = Path("test_db_load_only")
    call_count = {"A": 0}

    @task
    class TaskA:
        value: int

        def __call__(self) -> A:
            call_count["A"] += 1
            return self.value

    wf = Workflow([TaskA(value=0)], db_dir=db_path)

    variants = [TaskA(value=1), TaskA(value=2), TaskA(value=3)]

    # First run - compute and persist
    results1 = wf.run_variants(A, variants)
    assert results1 == [1, 2, 3]
    assert call_count["A"] == 3

    # Second run with load_only - should load from database
    results2 = wf.run_variants(A, variants, load_only=True)
    assert results2 == [1, 2, 3]
    assert call_count["A"] == 3  # Not incremented


def test_run_variants_load_only_returns_none_for_missing_results():
    """Test that run_variants() with load_only=True returns None for missing results."""
    db_path = Path("test_db_load_only")
    call_count = {"A": 0}

    @task
    class TaskA:
        value: int

        def __call__(self) -> A:
            call_count["A"] += 1
            return self.value

    wf = Workflow([TaskA(value=0)], db_dir=db_path)

    # First, persist only some variants
    wf.run_variants(A, [TaskA(value=1), TaskA(value=3)])
    assert call_count["A"] == 2

    # Now try to load with a mix of existing and missing variants
    variants = [TaskA(value=1), TaskA(value=2), TaskA(value=3)]
    results = wf.run_variants(A, variants, load_only=True)

    # Existing results should be loaded, missing should be None
    assert results[0] == 1  # Existed
    assert results[1] is None  # Did not exist
    assert results[2] == 3  # Existed
    assert call_count["A"] == 2  # Not incremented


class TestDatabaseWithRay:
    pass
