import time

from dagreon import ProfilingReport, Workflow, task

type A = int
type B = float
type C = str


def test_profile_returns_profiling_report():
    @task
    class TaskA:
        def __call__(self) -> A:
            time.sleep(0.11)
            return 2

    @task
    class TaskB:
        def __call__(self, a: A) -> B:
            time.sleep(0.11)
            return float(a * 3)

    wf = Workflow([TaskA(), TaskB()])
    report = wf.profile(B)

    assert isinstance(report, ProfilingReport)
    assert A in report.compute_times
    assert B in report.compute_times
    assert A in report.sizes
    assert B in report.sizes
    assert report.end_to_end_time > 0.0


def test_profile_uses_overrides():
    inputs: list[int] = []

    @task
    class Source:
        value: int

        def __call__(self) -> A:
            return self.value

    @task
    class Final:
        def __call__(self, a: A) -> B:
            inputs.append(a)
            return float(a * 2)

    wf = Workflow([Source(value=1), Final()])
    report = wf.profile(B, overrides=(Source(value=5),))

    assert inputs == [5]
    assert A in report.compute_times
    assert B in report.compute_times


def test_profile_does_not_load_from_database(tmp_path):
    calls = {"A": 0}

    @task
    class TaskA:
        def __call__(self) -> A:
            calls["A"] += 1
            return calls["A"]

    wf = Workflow([TaskA()], db_dir=tmp_path / "db")

    assert wf.run(A) == 1
    report = wf.profile(A)

    assert calls["A"] == 2
    assert A in report.compute_times


def test_summary_sorts_by_compute_time_by_default_and_shows_totals():
    report = ProfilingReport()
    report.compute_times = {A: 1.0, B: 3.0, C: 2.0}
    report.sizes = {A: 100, B: 300, C: 200}
    report.end_to_end_time = 7.0

    lines = report.summary().splitlines()
    row_names = [line.split()[0] for line in lines[3:6]]

    assert row_names == ["B", "C", "A"]
    assert "3.000000s (50.0%)" in lines[3]
    assert "300 B (50.0%)" in lines[3]
    assert (
        lines[-1]
        == "TOTAL | end-to-end: 7.000000s | compute: 6.000000s (100.0%) | size: 600 B (100.0%)"
    )


def test_summary_can_sort_by_size():
    report = ProfilingReport()
    report.compute_times = {A: 3.0, B: 2.0, C: 1.0}
    report.sizes = {A: 100, B: 300, C: 200}

    lines = report.summary(sort_by="size").splitlines()
    row_names = [line.split()[0] for line in lines[3:6]]

    assert row_names == ["B", "C", "A"]


def test_summary_threshold_aggregates_remaining_rows():
    report = ProfilingReport()
    report.compute_times = {A: 0.80, B: 0.15, C: 0.05}
    report.sizes = {A: 80, B: 15, C: 5}

    lines = report.summary(threshold=0.95).splitlines()
    row_names = [line.split()[0] for line in lines[3:6]]

    assert row_names == ["A", "B", "Other"]
    assert "0.050000s (5.0%)" in lines[5]
    assert "5 B (5.0%)" in lines[5]


def test_summary_size_column_handles_twelve_digit_sizes():
    report = ProfilingReport()
    report.compute_times = {A: 1.0}
    report.sizes = {A: 123456789012}

    row = report.summary().splitlines()[3]

    assert "123,456,789,012 B (100.0%)" in row


def test_str_uses_default_summary():
    report = ProfilingReport()
    report.compute_times = {A: 1.0}
    report.sizes = {A: 10}

    assert str(report) == report.summary()
