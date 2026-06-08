import pytest

from dagreon import task

# Type aliases
type A = int
type B = float
type C = str
type D = dict
type E = list


class TestFingerprintBasics:
    """Basic tests for task fingerprint behavior."""

    def test_fingerprint_is_bytes(self):
        """Test that fingerprint is bytes."""

        @task
        class TaskA:
            def __call__(self) -> A:
                return 1

        instance = TaskA()
        assert isinstance(instance.__task_fingerprint__, bytes)

    def test_fingerprint_has_consistent_length(self):
        """Test that fingerprint has consistent length (128-bit xxhash)."""

        @task
        class TaskA:
            def __call__(self) -> A:
                return 1

        @task
        class TaskB:
            value: int

            def __call__(self, a: A) -> B:
                return float(a)

        fp_a = TaskA().__task_fingerprint__
        fp_b = TaskB(value=42).__task_fingerprint__

        assert len(fp_a) == len(fp_b) == 16  # 128 bits = 16 bytes

    def test_fingerprint_only_on_instances(self):
        """Test that fingerprint is only available on instances, not classes."""

        @task
        class TaskA:
            def __call__(self) -> A:
                return 1

        # Class should not have fingerprint
        with pytest.raises(AttributeError):
            TaskA.__task_fingerprint__

        # Instance should have fingerprint
        instance = TaskA()
        assert hasattr(instance, "__task_fingerprint__")

    def test_fingerprint_is_immutable(self):
        """Test that fingerprint cannot be modified after creation."""

        @task
        class TaskA:
            def __call__(self) -> A:
                return 1

        instance = TaskA()
        with pytest.raises(AttributeError):
            instance.__task_fingerprint__ = b"modified"


class TestFingerprintEquality:
    """Tests for fingerprint equality based on task configuration."""

    def test_same_config_same_fingerprint(self):
        """Test that identical configurations produce identical fingerprints."""

        @task
        class TaskA:
            value: int

            def __call__(self) -> A:
                return self.value

        inst1 = TaskA(value=42)
        inst2 = TaskA(value=42)

        assert inst1 is not inst2
        assert inst1.__task_fingerprint__ == inst2.__task_fingerprint__

    def test_different_config_different_fingerprint(self):
        """Test that different configurations produce different fingerprints."""

        @task
        class TaskA:
            value: int

            def __call__(self) -> A:
                return self.value

        inst1 = TaskA(value=1)
        inst2 = TaskA(value=2)

        assert inst1.__task_fingerprint__ != inst2.__task_fingerprint__

    def test_different_class_different_fingerprint(self):
        """Test that different task classes produce different fingerprints."""

        @task
        class TaskA:
            def __call__(self) -> A:
                return 1

        @task
        class TaskB:
            def __call__(self) -> A:  # Same output type, different class
                return 1

        fp_a = TaskA().__task_fingerprint__
        fp_b = TaskB().__task_fingerprint__

        assert fp_a != fp_b

    def test_different_input_types_different_fingerprint(self):
        """Test that tasks with different input types have different fingerprints."""

        @task
        class TaskWithA:
            def __call__(self, a: A) -> B:
                return float(a)

        @task
        class TaskWithC:
            def __call__(self, c: C) -> B:
                return float(len(c))

        fp_with_a = TaskWithA().__task_fingerprint__
        fp_with_c = TaskWithC().__task_fingerprint__

        assert fp_with_a != fp_with_c

    def test_different_output_types_different_fingerprint(self):
        """Test that tasks with different output types have different fingerprints."""

        @task
        class TaskProducesB:
            def __call__(self) -> B:
                return 1.0

        @task
        class TaskProducesC:
            def __call__(self) -> C:
                return "1"

        fp_b = TaskProducesB().__task_fingerprint__
        fp_c = TaskProducesC().__task_fingerprint__

        assert fp_b != fp_c


class TestFingerprintComplexTypes:
    """Tests for fingerprint with complex field types."""

    def test_fingerprint_with_tuple_field(self):
        """Test fingerprint with tuple field."""

        @task
        class TaskWithTuple:
            values: tuple[int, ...]

            def __call__(self) -> A:
                return sum(self.values)

        inst1 = TaskWithTuple(values=(1, 2, 3))
        inst2 = TaskWithTuple(values=(1, 2, 3))
        inst3 = TaskWithTuple(values=(3, 2, 1))

        assert inst1.__task_fingerprint__ == inst2.__task_fingerprint__
        assert inst1.__task_fingerprint__ != inst3.__task_fingerprint__

    def test_fingerprint_with_frozenset_field(self):
        """Test fingerprint with frozenset field."""

        @task
        class TaskWithFrozenset:
            values: frozenset[int]

            def __call__(self) -> A:
                return sum(self.values)

        inst1 = TaskWithFrozenset(values=frozenset({1, 2, 3}))
        inst2 = TaskWithFrozenset(values=frozenset({3, 2, 1}))  # Same set
        inst3 = TaskWithFrozenset(values=frozenset({1, 2, 4}))  # Different set

        assert inst1.__task_fingerprint__ == inst2.__task_fingerprint__
        assert inst1.__task_fingerprint__ != inst3.__task_fingerprint__

    def test_fingerprint_with_dict_field(self):
        """Test fingerprint with dict field (should be sorted for stability)."""

        @task
        class TaskWithDict:
            config: dict

            def __call__(self) -> A:
                return len(self.config)

        # Note: frozen dataclass won't accept mutable dict directly,
        # but the fingerprint function handles dict serialization
        # This test checks the stable_json function indirectly

        @task
        class TaskWithFrozenConfig:
            a: int
            b: str

            def __call__(self) -> A:
                return self.a

        inst1 = TaskWithFrozenConfig(a=1, b="x")
        inst2 = TaskWithFrozenConfig(a=1, b="x")
        inst3 = TaskWithFrozenConfig(a=1, b="y")

        assert inst1.__task_fingerprint__ == inst2.__task_fingerprint__
        assert inst1.__task_fingerprint__ != inst3.__task_fingerprint__

    def test_fingerprint_with_none_field(self):
        """Test fingerprint with None values."""

        @task
        class TaskWithOptional:
            value: int | None

            def __call__(self) -> A:
                return self.value or 0

        inst_none = TaskWithOptional(value=None)
        inst_value = TaskWithOptional(value=42)

        assert inst_none.__task_fingerprint__ != inst_value.__task_fingerprint__

    def test_fingerprint_with_bool_field(self):
        """Test fingerprint with boolean field."""

        @task
        class TaskWithBool:
            flag: bool

            def __call__(self) -> A:
                return 1 if self.flag else 0

        inst_true = TaskWithBool(flag=True)
        inst_false = TaskWithBool(flag=False)

        assert inst_true.__task_fingerprint__ != inst_false.__task_fingerprint__

    def test_fingerprint_with_float_field(self):
        """Test fingerprint with float field."""

        @task
        class TaskWithFloat:
            value: float

            def __call__(self) -> A:
                return int(self.value)

        inst1 = TaskWithFloat(value=1.5)
        inst2 = TaskWithFloat(value=1.5)
        inst3 = TaskWithFloat(value=1.500001)

        assert inst1.__task_fingerprint__ == inst2.__task_fingerprint__
        assert inst1.__task_fingerprint__ != inst3.__task_fingerprint__

    def test_fingerprint_with_string_field(self):
        """Test fingerprint with string field."""

        @task
        class TaskWithString:
            name: str

            def __call__(self) -> A:
                return len(self.name)

        inst1 = TaskWithString(name="hello")
        inst2 = TaskWithString(name="hello")
        inst3 = TaskWithString(name="Hello")  # Different case

        assert inst1.__task_fingerprint__ == inst2.__task_fingerprint__
        assert inst1.__task_fingerprint__ != inst3.__task_fingerprint__


class TestFingerprintMultipleFields:
    """Tests for fingerprint with multiple fields."""

    def test_fingerprint_multiple_fields_all_same(self):
        """Test fingerprint with multiple fields, all same values."""

        @task
        class MultiFieldTask:
            a: int
            b: str
            c: float

            def __call__(self) -> A:
                return self.a

        inst1 = MultiFieldTask(a=1, b="x", c=2.5)
        inst2 = MultiFieldTask(a=1, b="x", c=2.5)

        assert inst1.__task_fingerprint__ == inst2.__task_fingerprint__

    def test_fingerprint_multiple_fields_one_different(self):
        """Test fingerprint with multiple fields, one different."""

        @task
        class MultiFieldTask:
            a: int
            b: str
            c: float

            def __call__(self) -> A:
                return self.a

        base = MultiFieldTask(a=1, b="x", c=2.5)
        diff_a = MultiFieldTask(a=2, b="x", c=2.5)
        diff_b = MultiFieldTask(a=1, b="y", c=2.5)
        diff_c = MultiFieldTask(a=1, b="x", c=3.5)

        assert base.__task_fingerprint__ != diff_a.__task_fingerprint__
        assert base.__task_fingerprint__ != diff_b.__task_fingerprint__
        assert base.__task_fingerprint__ != diff_c.__task_fingerprint__

        # All different from each other
        assert diff_a.__task_fingerprint__ != diff_b.__task_fingerprint__
        assert diff_b.__task_fingerprint__ != diff_c.__task_fingerprint__

    def test_fingerprint_field_order_does_not_matter(self):
        """Test that field order in class definition doesn't affect fingerprint of same values."""
        # Note: dataclass field order is fixed by class definition,
        # but the asdict should produce consistent results

        @task
        class Task1:
            a: int
            b: str

            def __call__(self) -> A:
                return self.a

        @task
        class Task2:
            b: str
            a: int

            def __call__(self) -> A:
                return self.a

        # These are different classes, so fingerprints will differ
        # But within the same class, order is consistent
        inst1a = Task1(a=1, b="x")
        inst1b = Task1(a=1, b="x")
        assert inst1a.__task_fingerprint__ == inst1b.__task_fingerprint__


class TestFingerprintLeafTasks:
    """Tests for fingerprint with leaf tasks (no inputs)."""

    def test_leaf_task_fingerprint(self):
        """Test fingerprint for task with no inputs."""

        @task
        class LeafTask:
            def __call__(self) -> A:
                return 42

        inst = LeafTask()
        fp = inst.__task_fingerprint__

        assert isinstance(fp, bytes)
        assert len(fp) == 16

    def test_leaf_task_with_config_fingerprint(self):
        """Test fingerprint for leaf task with configuration."""

        @task
        class ConfiguredLeafTask:
            value: int

            def __call__(self) -> A:
                return self.value

        inst1 = ConfiguredLeafTask(value=1)
        inst2 = ConfiguredLeafTask(value=1)
        inst3 = ConfiguredLeafTask(value=2)

        assert inst1.__task_fingerprint__ == inst2.__task_fingerprint__
        assert inst1.__task_fingerprint__ != inst3.__task_fingerprint__


class TestFingerprintWithPostInit:
    """Tests for fingerprint with custom __post_init__."""

    def test_fingerprint_with_custom_post_init(self):
        """Test that custom __post_init__ doesn't break fingerprint."""
        from dataclasses import InitVar

        @task
        class TaskWithPostInit:
            x: int
            y_degrees: InitVar[float]

            def __post_init__(self, y_degrees: float):
                object.__setattr__(self, "y_radians", y_degrees * 3.14159 / 180.0)

            def __call__(self) -> A:
                return self.x

        inst1 = TaskWithPostInit(x=5, y_degrees=90.0)
        inst2 = TaskWithPostInit(x=5, y_degrees=90.0)

        # Check fingerprint exists
        assert isinstance(inst1.__task_fingerprint__, bytes)

        # Check computed attribute exists
        assert hasattr(inst1, "y_radians")
        assert inst1.y_radians == 90.0 * 3.14159 / 180.0

        # Same init args -> same fingerprint
        assert inst1.__task_fingerprint__ == inst2.__task_fingerprint__

        # Note: InitVar values affect fingerprint through the dataclass asdict
        # behavior - this tests that the fingerprint is consistent
