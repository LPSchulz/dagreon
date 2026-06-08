from dataclasses import InitVar

import pytest

from dagreon import task

# Type aliases used across all tests
type A = int
type B = float
type C = str
type D = dict


def test_missing_return_annotation_raises():
    with pytest.raises(TypeError):

        @task
        class BadTask:
            def __call__(self, a: A):
                return 1


def test_non_typealias_return_annotation_raises():
    with pytest.raises(TypeError):

        @task
        class BadTask:
            def __call__(self, a: A) -> int:
                return 1


def test_missing_input_annotation_raises():
    with pytest.raises(TypeError):

        @task
        class BadTask:
            def __call__(self, a, b: B) -> C:
                return str(b)


def test_duplicate_input_annotation_raises():
    with pytest.raises(TypeError):

        @task
        class BadTask:
            def __call__(self, a: A, b: A) -> B:
                return b


def test_duplicate_input_output_annotation_raises():
    with pytest.raises(TypeError):

        @task
        class BadTask:
            def __call__(self, a: A, b: B) -> B:
                return b


def test_non_typealias_input_annotation_raises():
    with pytest.raises(TypeError):

        @task
        class BadTask:
            def __call__(self, a: int, b: B) -> B:
                return b


def test_class_without_call_raises():
    with pytest.raises(TypeError):

        @task
        class NotATask:
            pass


def test_task_class_is_still_usable_as_class():
    """Test that decorated task is still a proper class."""

    @task
    class MyTask:
        value: int

        def __call__(self) -> A:
            return self.value

        def helper(self) -> str:
            return f"Value is {self.value}"

    inst = MyTask(value=10)
    assert isinstance(inst, MyTask)
    assert inst.helper() == "Value is 10"


def test_task_with_classmethod_and_staticmethod():
    """Test task with class and static methods."""

    @task
    class TaskWithMethods:
        value: int

        @staticmethod
        def static_helper(x: int) -> int:
            return x * 2

        @classmethod
        def class_helper(cls, x: int) -> str:
            return f"{cls.__name__}:{x}"

        def __call__(self) -> A:
            return self.static_helper(self.value)

    inst = TaskWithMethods(value=5)
    assert inst() == 10
    assert TaskWithMethods.static_helper(3) == 6
    assert TaskWithMethods.class_helper(3) == "TaskWithMethods:3"


def test_task_decorator_returns_original_class():
    @task
    class MyTask:
        def __call__(self, a: A) -> B:
            return a

    instance = MyTask()
    assert isinstance(instance, MyTask)


def test_task_decorator_does_not_break_existing_post_init():
    @task
    class MyTask:
        x: int
        y_degree: InitVar[float]

        def __post_init__(self, y_degree: float):
            object.__setattr__(self, "y_rad", y_degree * 3.14159 / 180.0)

        def __call__(self, a: A) -> B:
            return a

    instance = MyTask(x=5, y_degree=90.0)
    assert instance.x == 5
    assert instance.y_rad == 90.0 * 3.14159 / 180.0  # type: ignore
    with pytest.raises(AttributeError):
        instance.y_degree  # type: ignore


def test_task_uses_annotation_objects_not_runtime_types():
    @task
    class MyTask:
        def __call__(self, a: A) -> B:
            return 5

    spec = MyTask.__task_spec__

    # Annotation identity must be preserved
    assert spec.input_types == [A]
    assert spec.output_type is B

    # Explicitly NOT normalized
    assert spec.input_types[0] is not int
    assert spec.output_type is not int


def test_multiple_input_annotations_preserve_order():
    @task
    class MyTask:
        def __call__(self, a: A, b: B) -> C:
            return "x"

    spec = MyTask.__task_spec__

    assert spec.input_types == [A, B]
    assert spec.output_type is C


def test_leaf_task_with_alias_output():
    @task
    class MyTask:
        def __call__(self) -> D:
            return {"x": 1}

    spec = MyTask.__task_spec__

    assert spec.input_types == []
    assert spec.output_type is D


def test_only_call_method_annotations_are_used():
    @task
    class MyTask:
        def helper(self, x: C) -> C:
            return x

        def __call__(self, a: A) -> B:
            return 1

    spec = MyTask.__task_spec__

    assert spec.input_types == [A]
    assert spec.output_type is B


def test_task_spec_is_immutable():
    @task
    class MyTask:
        def __call__(self, a: A) -> B:
            return a

    spec = MyTask.__task_spec__

    with pytest.raises(AttributeError):
        spec.output_type = A


def test_task_decorator_adds_fingerprint_for_instances():
    @task
    class MyTask:
        def __call__(self, a: A) -> B:
            return a

    instance = MyTask()
    fp = instance.__task_fingerprint__
    assert isinstance(fp, (bytes, bytearray))

    # fingerprint is only created for instance, not the class
    with pytest.raises(AttributeError):
        MyTask.__task_fingerprint__

    # changing the fingerprint is not allowed
    with pytest.raises(AttributeError):
        instance.__task_fingerprint__ = b"new_fp"


def test_task_fingerprint_represents_different_instances():
    @task
    class MyTask:
        x: int

        def __call__(self, a: A) -> B:
            return a

    instance1 = MyTask(x=1)
    instance1b = MyTask(x=1)
    instance2 = MyTask(x=2)
    assert instance1.x == 1
    assert instance1b.x == 1
    assert instance2.x == 2
    assert instance1 is not instance1b
    assert instance1.__task_fingerprint__ == instance1b.__task_fingerprint__
    assert instance1.__task_fingerprint__ != instance2.__task_fingerprint__


def test_task_with_default_values_in_dataclass():
    """Test task with default field values."""

    @task
    class TaskWithDefaults:
        value: int = 42
        name: str = "default"

        def __call__(self) -> A:
            return self.value

    inst1 = TaskWithDefaults()
    inst2 = TaskWithDefaults(value=42, name="default")
    inst3 = TaskWithDefaults(value=100)

    assert inst1.value == 42
    assert inst1.name == "default"
    assert inst1.__task_fingerprint__ == inst2.__task_fingerprint__
    assert inst1.__task_fingerprint__ != inst3.__task_fingerprint__
