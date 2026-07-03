import pytest

from dagreon import Workflow, task

ray = pytest.importorskip("ray")

type A = int
type B = float
type C = str


@pytest.fixture()
def ray_runtime():
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    try:
        yield
    finally:
        ray.shutdown()


@task
class MakeA:
    value: int

    def __call__(self) -> A:
        return self.value


@task
class MakeB:
    multiplier: int

    def __call__(self, a: A) -> B:
        return float(a * self.multiplier)


@task
class MakeC:
    prefix: str

    def __call__(self, a: A, b: B) -> C:
        return f"{self.prefix}: {a} -> {b}"


def test_run_with_ray_returns_same_value_as_local(ray_runtime):
    tasks = [MakeA(7), MakeB(3), MakeC("result")]

    local_result = Workflow(tasks).run(C)
    ray_result = Workflow(tasks, use_ray=True).run(C)

    assert ray_result == local_result == "result: 7 -> 21.0"


def test_run_variants_with_ray_returns_same_values_as_local(ray_runtime):
    base_tasks = [MakeA(0), MakeB(3), MakeC("variant")]
    variants = [MakeA(1), MakeA(5), MakeA(-2)]

    local_results = Workflow(base_tasks).run_variants(
        C,
        variants,
        tqdm_args={"disable": True},
    )
    ray_results = Workflow(base_tasks, use_ray=True).run_variants(
        C,
        variants,
        tqdm_args={"disable": True},
    )

    assert ray_results == local_results == [
        "variant: 1 -> 3.0",
        "variant: 5 -> 15.0",
        "variant: -2 -> -6.0",
    ]
