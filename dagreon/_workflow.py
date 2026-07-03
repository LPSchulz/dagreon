import pickle
import sys
import time
from pathlib import Path
from typing import Any, Iterable, TypeAliasType

import rustworkx as rx
from tqdm import tqdm
from xxhash import xxh3_128_digest

from ._errors import CycleError, DuplicateOutputError, MissingProducerError
from ._persistence import BufferedLMDBWriter, initialize_lmdb_directory
from ._report import ProfilingReport
from ._task import TaskClass, is_task


def _require_ray() -> Any:
    try:
        import ray
    except ImportError as exc:
        raise ImportError(
            "Ray support requires the optional dependency: "
            "install dagreon with the 'ray' extra."
        ) from exc
    return ray


# We build a rustworkx PyDiGraph from the tasks
# A Workflow object has a base graph which gets computed once when the Workflow is
# initialized. That way the graph is already built when calling run().
# When calling run() with overrides, only the overridden tasks and everything downstream
# of them gets recomputed, instead of the full Graph.
class Graph:
    def __init__(self, new_tasks: Iterable[Any], base_graph: "Graph | None"):
        # validate new tasks
        out_types: list[TypeAliasType] = []
        for task in new_tasks:
            if not is_task(task):
                raise ValueError(
                    f"{task} is not a Task. Did you forget to decorate it with @task?"
                )
            if task.__task_spec__.output_type in out_types:
                raise DuplicateOutputError(
                    f"Multiple tasks produce output type "
                    f"{task.__task_spec__.output_type.__name__}"
                )
            out_types.append(task.__task_spec__.output_type)

        if base_graph is None:
            self.rx_graph: rx.PyDiGraph[TaskClass, None] = rx.PyDiGraph()
            self.type_to_index: dict[TypeAliasType, int] = {}
            self.index_to_type: dict[int, TypeAliasType] = {}
        else:
            self.rx_graph = base_graph.rx_graph.copy()
            self.type_to_index = base_graph.type_to_index.copy()
            self.index_to_type = base_graph.index_to_type.copy()
        # make nodes
        for task in new_tasks:
            # if node for this task exists in the base graph, override it and delete
            # incoming edges
            if task.__task_spec__.output_type in self.type_to_index:
                node_id = self.type_to_index[task.__task_spec__.output_type]
                self.rx_graph[node_id] = task
                for edge_id in self.rx_graph.in_edge_indices(node_id):
                    self.rx_graph.remove_edge_from_index(edge_id)
            # else create a node for this task
            else:
                node_id = self.rx_graph.add_node(task)
                self.type_to_index[task.__task_spec__.output_type] = node_id
                self.index_to_type[node_id] = task.__task_spec__.output_type
        # make edges (that are possible)
        # need to also go through existing tasks in the base graph
        for task in self.rx_graph.nodes():
            producer_index = self.type_to_index[task.__task_spec__.output_type]
            for input_type in task.__task_spec__.input_types:
                if input_type in self.type_to_index:
                    consumer_index = self.type_to_index[input_type]
                    self.rx_graph.add_edge(consumer_index, producer_index, None)


# The Graph object gets transformed into an ExecutionPlan when executing, which does a
# subgraph extraction to only keep the relevant part of the graph for the target type.
# This way the ExecutionPlan is as small as possible which is relevant for remote
# execution with ray where the plan needs to be serialized and sent to workers.
class ExecutionPlan:
    def __init__(self, target: TypeAliasType, graph: Graph):
        if target not in graph.type_to_index:
            raise MissingProducerError(
                f"Target type {target.__name__} is not produced by any task"
            )
        target_id = graph.type_to_index[target]
        relevant_ids = list(rx.ancestors(graph.rx_graph, target_id)) + [target_id]
        # Check that all inputs have producers
        for task in [graph.rx_graph[i] for i in relevant_ids]:
            spec = task.__task_spec__
            for input_type in spec.input_types:
                if input_type not in graph.type_to_index:
                    raise MissingProducerError(
                        f"No task produces required input type {input_type.__name__} "
                        f"(needed by task producing {spec.output_type.__name__})"
                    )
        self.target = target
        self.type_to_index: dict[TypeAliasType, int] = {}
        self.index_to_type: dict[int, TypeAliasType] = {}
        self.rx_graph, nodemap = graph.rx_graph.subgraph_with_nodemap(relevant_ids)
        for i in self.rx_graph.node_indices():
            original_i = nodemap[i]
            t = graph.index_to_type[original_i]
            self.type_to_index[t] = i
            self.index_to_type[i] = t
        # Check for cycles using topological sort
        try:
            rx.topological_sort(self.rx_graph)
        except rx.DAGHasCycle:
            raise CycleError("Workflow contains a cycle")
        self.keys: dict[TypeAliasType, bytes] = {}
        self._get_key_recursive(target, self.keys)

    def _get_key_recursive(
        self, target: TypeAliasType, keys: dict[TypeAliasType, bytes]
    ) -> bytes:
        if target in keys:
            return keys[target]
        task = self.rx_graph[self.type_to_index[target]]
        input_keys: list[bytes] = []
        for input_type in task.__task_spec__.input_types:
            input_key = self._get_key_recursive(input_type, keys)
            input_keys.append(input_key)
        node_key = xxh3_128_digest(task.__task_fingerprint__ + b"".join(input_keys))
        keys[target] = node_key
        return node_key

    def execute(self) -> Any:
        results: dict[int, Any] = {}
        return self._execute_recursive(self.type_to_index[self.target], results)

    def _execute_recursive(self, node_id, results):
        if node_id in results:
            return results[node_id]
        task = self.rx_graph[node_id]
        inputs: list[TypeAliasType] = []
        for input_type in task.__task_spec__.input_types:
            predecessor_id = self.type_to_index[input_type]
            input_value = self._execute_recursive(predecessor_id, results)
            inputs.append(input_value)
        result = task(*inputs)
        results[node_id] = result
        return result

    def profile(self) -> ProfilingReport:
        report = ProfilingReport()
        self._execute_recursive_with_profiling(
            self.type_to_index[self.target], {}, report
        )
        return report

    def _execute_recursive_with_profiling(self, node_id, results, report):
        if node_id in results:
            return results[node_id]
        task = self.rx_graph[node_id]
        inputs: list[TypeAliasType] = []
        for input_type in task.__task_spec__.input_types:
            predecessor_id = self.type_to_index[input_type]
            input_value = self._execute_recursive_with_profiling(
                predecessor_id, results, report
            )
            inputs.append(input_value)
        start_time = time.perf_counter()
        result = task(*inputs)
        results[node_id] = result
        report.compute_times[self.index_to_type[node_id]] = (
            time.perf_counter() - start_time
        )
        report.sizes[self.index_to_type[node_id]] = sys.getsizeof(result)
        return result


def _execute_plan(plan: ExecutionPlan) -> Any:
    return plan.execute()


def _profile_plan(plan: ExecutionPlan) -> ProfilingReport:
    return plan.profile()


class Workflow:
    """A workflow composed of tasks that form a DAG.

    Args:
        tasks: Task instances that make up the base workflow.
        use_ray: Execute workflow plans through Ray when set.
        db_dir: Optional LMDB directory for persisted final target results.
        lmdb_args: Optional keyword arguments passed to ``lmdb.open``.
        db_flush_interval_s: Minimum time between buffered LMDB flushes.
    """

    def __init__(
        self,
        tasks: Iterable[Any],
        use_ray: bool = False,
        db_dir: str | Path | None = None,
        lmdb_args: dict[str, Any] | None = None,
        db_flush_interval_s: float = 5.0,
    ):
        self.base_graph = Graph(tasks, None)
        self.db_writer: BufferedLMDBWriter | None = None
        self.lmdb_args = {} if lmdb_args is None else dict(lmdb_args)
        self.db_flush_interval_s = db_flush_interval_s

        if "path" in self.lmdb_args:
            raise ValueError("Cannot specify 'path' in lmdb_args; use db_dir instead.")
        if db_dir is None:
            self.use_database = False
        else:
            self.db_dir = initialize_lmdb_directory(db_dir, self.lmdb_args)
            self.use_database = True
            self.db_writer = BufferedLMDBWriter(
                self.db_dir,
                self.lmdb_args,
                flush_interval_s=self.db_flush_interval_s,
            )

        self.use_ray = use_ray
        if self.use_ray:
            ray = _require_ray()
            if not ray.is_initialized():
                ray.init(namespace="dagreon", log_to_driver=False)

    def _db_get(self, key: bytes) -> tuple[bool, Any]:
        if self.db_writer is None:
            return False, None
        raw = self.db_writer.get(key)
        if raw is not None:
            return True, pickle.loads(raw)
        return False, None

    def _db_put(self, key: bytes, value: bytes) -> None:
        if self.db_writer is None:
            return
        self.db_writer.put(key, value)

    def _db_flush_if_due(self) -> None:
        if self.db_writer is not None:
            self.db_writer.flush_if_due()

    @staticmethod
    def _require_task(obj: Any) -> TaskClass:
        if not is_task(obj):
            raise ValueError(
                f"{obj} is not a Task. Did you forget to decorate it with @task?"
            )
        return obj

    def validate(self, target: TypeAliasType) -> None:
        """
        Validate the workflow for correctness.

        Args:
            target: The output type to validate reachability for

        Raises:
            DuplicateOutputError: If multiple tasks produce the same output type
            MissingProducerError: If a required input type has no producer
            CycleError: If the workflow contains a cycle
        """
        ExecutionPlan(target, self.base_graph)

    def run(
        self,
        target: TypeAliasType,
        overrides: Iterable[Any] | None = None,
        load_only: bool = False,
        ray_remote_args: dict[str, Any] | None = None,
    ) -> Any:
        """
        Execute the workflow to compute the target type using recursive execution.

        Args:
            target: The output type to compute.
            overrides: Optional task instances to replace or add for this run.
            load_only: Return a persisted result without computing. Returns ``None``
                when no stored result exists or no database is configured.
            ray_remote_args: Optional Ray task options for this execution.

        Returns:
            The computed or loaded result of the target type.

        Raises:
            MissingProducerError: If a required type has no producer.
            CycleError: If the target subgraph contains a cycle.
        """
        overrides_tuple: tuple[Any, ...] = () if overrides is None else tuple(overrides)
        ray_options: dict[str, Any] = (
            {} if ray_remote_args is None else dict(ray_remote_args)
        )
        overridden_graph = Graph(overrides_tuple, self.base_graph)
        plan = ExecutionPlan(target, overridden_graph)

        was_found, loaded = self._db_get(plan.keys[plan.target])
        if was_found or load_only:
            return loaded

        # run the workflow
        if self.use_ray:
            ray = _require_ray()
            ref = ray.remote(_execute_plan).options(**ray_options).remote(plan)
            res = ray.get(ref)
        else:
            res = plan.execute()

        self._db_put(plan.keys[plan.target], pickle.dumps(res))
        self._db_flush_if_due()
        return res

    def profile(
        self,
        target: TypeAliasType,
        overrides: Iterable[Any] | None = None,
        ray_remote_args: dict[str, Any] | None = None,
    ) -> ProfilingReport:
        """
        Execute the workflow to compute the target type and return profiling data.

        Args:
            target: The output type to compute.
            overrides: Optional task instances to replace or add for this profile run.
            ray_remote_args: Optional Ray task options for this execution.

        Returns:
            A profiling report for the workflow execution.

        Raises:
            MissingProducerError: If a required type has no producer.
            CycleError: If the target subgraph contains a cycle.
        """
        overrides_tuple: tuple[Any, ...] = () if overrides is None else tuple(overrides)
        ray_options: dict[str, Any] = (
            {} if ray_remote_args is None else dict(ray_remote_args)
        )
        start_time = time.perf_counter()
        overridden_graph = Graph(overrides_tuple, self.base_graph)
        plan = ExecutionPlan(target, overridden_graph)

        # run the workflow
        if self.use_ray:
            ray = _require_ray()
            ref = ray.remote(_profile_plan).options(**ray_options).remote(plan)
            report = ray.get(ref)
        else:
            report = plan.profile()
        report.end_to_end_time = time.perf_counter() - start_time
        return report

    def run_variants(
        self,
        target: TypeAliasType,
        variants: Iterable[tuple[Any, ...] | Any],
        overrides: Iterable[Any] | None = None,
        load_only: bool = False,
        ray_remote_args: dict[str, Any] | None = None,
        tqdm_args: dict[str, Any] | None = None,
    ) -> Any:
        """
        Run multiple variants of this workflow and return all results.

        Args:
            target: The output type to run for each variant.
            variants: Variants to evaluate. Each variant is either a single task
                instance or a tuple of task instances.
            overrides: Optional task instances applied to every variant.
            load_only: Return persisted results without computing missing variants.
                Missing results are returned as ``None``.
            ray_remote_args: Optional Ray task options for variant executions.
            tqdm_args: Optional arguments passed to ``tqdm``. ``total`` is set
                automatically and cannot be supplied.

        Returns:
            List of results, one per variant.
        """
        # parse variants and overrides
        task_tuples: list[tuple[TaskClass, ...]] = []
        override_tuple: tuple[Any, ...] = () if overrides is None else tuple(overrides)
        override_tasks = tuple(self._require_task(o) for o in override_tuple)
        for v in variants:
            if isinstance(v, tuple):
                variant_tasks = tuple(self._require_task(t) for t in v)
                task_tuples.append(variant_tasks + override_tasks)
            else:
                task_tuples.append((self._require_task(v),) + override_tasks)
        progress_options: dict[str, Any] = {} if tqdm_args is None else dict(tqdm_args)
        ray_options: dict[str, Any] = (
            {} if ray_remote_args is None else dict(ray_remote_args)
        )
        if "total" in progress_options:
            raise ValueError(
                "Cannot specify 'total' in tqdm_args; it is set automatically."
            )

        plans: list[ExecutionPlan] = []
        for variant in task_tuples:
            graph = Graph(variant, self.base_graph)
            plans.append(ExecutionPlan(target, graph))

        with tqdm(total=len(task_tuples), **progress_options) as pbar:
            # load final target results from db
            results: list[Any] = [None] * len(task_tuples)
            unloaded_indices: list[int] = []
            for i, plan in enumerate(plans):
                was_found, loaded = self._db_get(plan.keys[plan.target])
                if was_found:
                    pbar.update(1)
                    results[i] = loaded
                else:
                    unloaded_indices.append(i)
            if load_only or not unloaded_indices:
                return results
            # Reset the pbar so that it does not have a long update interval
            n = pbar.n
            pbar.reset()
            pbar.n = n
            pbar.last_print_n = n
            pbar.refresh()
            pbar.miniters = 1

            # run the workflow
            if self.use_ray:
                ray = _require_ray()
                ref_to_index = {
                    ray.remote(_execute_plan).options(**ray_options).remote(plans[i]): i
                    for i in unloaded_indices
                }
                remaining = list(ref_to_index.keys())
                while remaining:
                    ready, remaining = ray.wait(remaining, num_returns=1)
                    for ready_ref in ready:
                        i = ref_to_index[ready_ref]
                        result = ray.get(ready_ref)
                        results[i] = result
                        self._db_put(
                            plans[i].keys[plans[i].target], pickle.dumps(result)
                        )
                        pbar.update(1)
            else:
                for i in unloaded_indices:
                    results[i] = plans[i].execute()
                    self._db_put(
                        plans[i].keys[plans[i].target], pickle.dumps(results[i])
                    )
                    pbar.update(1)

        self._db_flush_if_due()
        return results
