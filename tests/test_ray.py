from dagreon import Workflow, task
from dagreon import _workflow as workflow_module

type A = int
type B = float


class FakeRemoteFunction:
    def __init__(self, fake_ray, func):
        self.fake_ray = fake_ray
        self.func = func

    def options(self, **kwargs):
        self.fake_ray.option_calls.append(kwargs)
        return self

    def remote(self, *args):
        ref = f"ref-{len(self.fake_ray.refs)}"
        self.fake_ray.remote_calls.append((self.func, args))
        self.fake_ray.refs[ref] = self.func(*args)
        return ref


class FakeRay:
    def __init__(self, initialized=True):
        self.initialized = initialized
        self.init_calls = []
        self.option_calls = []
        self.remote_calls = []
        self.refs = {}

    def is_initialized(self):
        return self.initialized

    def init(self, *args, **kwargs):
        self.init_calls.append((args, kwargs))
        self.initialized = True

    def remote(self, func):
        return FakeRemoteFunction(self, func)

    def get(self, ref):
        return self.refs[ref]

    def wait(self, refs, num_returns=1):
        return refs[:num_returns], refs[num_returns:]


def test_use_ray_initializes_ray_when_needed(monkeypatch):
    fake_ray = FakeRay(initialized=False)
    monkeypatch.setattr(workflow_module, "_require_ray", lambda: fake_ray)

    @task
    class TaskA:
        def __call__(self) -> A:
            return 1

    Workflow([TaskA()], use_ray=True)

    assert len(fake_ray.init_calls) == 1
    assert fake_ray.init_calls[0][1] == {
        "namespace": "dagreon",
        "log_to_driver": False,
    }


def test_use_ray_does_not_reinitialize_if_already_running(monkeypatch):
    fake_ray = FakeRay(initialized=True)
    monkeypatch.setattr(workflow_module, "_require_ray", lambda: fake_ray)

    @task
    class TaskA:
        def __call__(self) -> A:
            return 1

    Workflow([TaskA()], use_ray=True)

    assert fake_ray.init_calls == []


def test_run_with_ray_returns_value_and_forwards_remote_args(monkeypatch):
    fake_ray = FakeRay()
    monkeypatch.setattr(workflow_module, "_require_ray", lambda: fake_ray)

    @task
    class TaskA:
        def __call__(self) -> A:
            return 1

    workflow = Workflow([TaskA()], use_ray=True)
    result = workflow.run(A, ray_remote_args={"num_cpus": 1})

    assert result == 1
    assert fake_ray.option_calls == [{"num_cpus": 1}]
    assert len(fake_ray.remote_calls) == 1


def test_profile_with_ray_returns_populated_profiling_report(monkeypatch):
    fake_ray = FakeRay()
    monkeypatch.setattr(workflow_module, "_require_ray", lambda: fake_ray)

    @task
    class TaskA:
        def __call__(self) -> A:
            return 1

    workflow = Workflow([TaskA()], use_ray=True)
    report = workflow.profile(A, ray_remote_args={"num_cpus": 1})

    assert A in report.compute_times
    assert report.end_to_end_time >= 0.0
    assert fake_ray.option_calls == [{"num_cpus": 1}]


def test_run_variants_with_ray_returns_results_in_order(monkeypatch):
    fake_ray = FakeRay()
    monkeypatch.setattr(workflow_module, "_require_ray", lambda: fake_ray)

    @task
    class TaskA1:
        def __call__(self) -> A:
            return 1

    @task
    class TaskA2:
        def __call__(self) -> A:
            return 2

    workflow = Workflow([TaskA1()], use_ray=True)
    results = workflow.run_variants(
        A,
        [TaskA1(), TaskA2()],
        ray_remote_args={"num_cpus": 1},
        tqdm_args={"disable": True},
    )

    assert results == [1, 2]
    assert fake_ray.option_calls == [{"num_cpus": 1}, {"num_cpus": 1}]


def test_run_with_ray_and_database_uses_local_writer(tmp_path, monkeypatch):
    fake_ray = FakeRay()
    monkeypatch.setattr(workflow_module, "_require_ray", lambda: fake_ray)
    calls = {"A": 0}

    @task
    class TaskA:
        def __call__(self) -> A:
            calls["A"] += 1
            return 1

    workflow = Workflow([TaskA()], use_ray=True, db_dir=tmp_path / "db")

    assert workflow.run(A) == 1
    assert workflow.run(A) == 1
    assert calls["A"] == 1
