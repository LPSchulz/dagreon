from dagreon import Workflow, task

# Type aliases
type A = int
type B = float
type C = str
type D = dict
type E = list
type F = bool
type NoneResult = None


class TestEdgeCaseVariants:
    """Edge case tests for variant execution."""

    def test_run_variants_single_variant(self):
        """Test run_variants with a single variant."""

        @task
        class TaskA:
            value: int

            def __call__(self) -> A:
                return self.value

        wf = Workflow([TaskA(value=1)])
        results = wf.run_variants(A, [TaskA(value=2)])

        assert len(results) == 1
        assert results[0] == 2

    def test_run_variants_many_variants(self):
        """Test run_variants with many variants."""

        @task
        class TaskA:
            value: int

            def __call__(self) -> A:
                return self.value

        wf = Workflow([TaskA(value=0)])
        variants = [TaskA(value=i) for i in range(100)]
        results = wf.run_variants(A, variants)

        assert len(results) == 100
        assert results == list(range(100))

    def test_run_variants_duplicate_variants(self):
        """Test run_variants with duplicate variants."""

        @task
        class TaskA:
            value: int

            def __call__(self) -> A:
                return self.value

        wf = Workflow([TaskA(value=0)])

        # Same variant twice
        variants = [TaskA(value=42), TaskA(value=42)]
        results = wf.run_variants(A, variants)

        assert len(results) == 2
        assert results == [42, 42]


class TestEdgeCaseOverrides:
    """Edge case tests for task overrides."""

    def test_override_root_task(self):
        """Test overriding the root task of a chain."""

        @task
        class Root1:
            def __call__(self) -> A:
                return 1

        @task
        class Root2:
            def __call__(self) -> A:
                return 100

        @task
        class Consumer:
            def __call__(self, a: A) -> B:
                return float(a * 2)

        wf = Workflow([Root1(), Consumer()])

        assert wf.run(B) == 2.0
        assert wf.run(B, overrides=(Root2(),)) == 200.0

    def test_override_leaf_task(self):
        """Test overriding the final task in a chain."""

        @task
        class Source:
            def __call__(self) -> A:
                return 10

        @task
        class Consumer1:
            def __call__(self, a: A) -> B:
                return float(a)

        @task
        class Consumer2:
            def __call__(self, a: A) -> B:
                return float(a * 100)

        wf = Workflow([Source(), Consumer1()])

        assert wf.run(B) == 10.0
        assert wf.run(B, overrides=(Consumer2(),)) == 1000.0

    def test_override_middle_task(self):
        """Test overriding a middle task in a chain."""

        @task
        class Start:
            def __call__(self) -> A:
                return 1

        @task
        class Middle1:
            def __call__(self, a: A) -> B:
                return float(a)

        @task
        class Middle2:
            def __call__(self, a: A) -> B:
                return float(a * 10)

        @task
        class End:
            def __call__(self, b: B) -> C:
                return str(b)

        wf = Workflow([Start(), Middle1(), End()])

        assert wf.run(C) == "1.0"
        assert wf.run(C, overrides=(Middle2(),)) == "10.0"

    def test_empty_tuple_override(self):
        """Test workflow.run with empty tuple override."""

        @task
        class TaskA:
            def __call__(self) -> A:
                return 42

        wf = Workflow([TaskA()])

        # Empty tuple should work (no overrides)
        result = wf.run(A, overrides=())
        assert result == 42

    def test_none_override(self):
        """Test workflow.run with no override (explicit)."""

        @task
        class TaskA:
            def __call__(self) -> A:
                return 42

        wf = Workflow([TaskA()])

        result = wf.run(A, overrides=[])
        assert result == 42


class TestEdgeCaseTaskExecution:
    """Edge case tests for task execution behavior."""

    def test_task_returning_none(self):
        """Test task that returns None."""

        @task
        class TaskReturnsNone:
            def __call__(self) -> NoneResult:
                return None

        wf = Workflow([TaskReturnsNone()])
        result = wf.run(NoneResult)
        assert result is None

    def test_task_returning_empty_collection(self):
        """Test task returning empty collections."""

        @task
        class EmptyList:
            def __call__(self) -> E:
                return []

        @task
        class EmptyDict:
            def __call__(self) -> D:
                return {}

        wf_list = Workflow([EmptyList()])
        wf_dict = Workflow([EmptyDict()])

        assert wf_list.run(E) == []
        assert wf_dict.run(D) == {}

    def test_task_with_complex_return_value(self):
        """Test task with complex nested return value."""

        @task
        class ComplexReturn:
            def __call__(self) -> D:
                return {
                    "nested": {"a": [1, 2, 3], "b": {"deep": True}},
                    "list": [{"x": 1}, {"y": 2}],
                }

        wf = Workflow([ComplexReturn()])
        result = wf.run(D)

        assert result["nested"]["a"] == [1, 2, 3]
        assert result["nested"]["b"]["deep"] is True
        assert result["list"][0]["x"] == 1
