from typing import TypeAliasType


TIME_COLUMN_WIDTH = 24
SIZE_COLUMN_WIDTH = 28
TABLE_WIDTH = 32 + 1 + TIME_COLUMN_WIDTH + 1 + SIZE_COLUMN_WIDTH


def _percentage(value: float, total: float) -> float:
    if total == 0.0:
        return 0.0
    return value / total * 100.0


def _format_time(seconds: float, total_seconds: float) -> str:
    return f"{seconds:.6f}s ({_percentage(seconds, total_seconds):.1f}%)"


def _format_size(size: int, total_size: int) -> str:
    return f"{size:>15,d} B ({_percentage(float(size), float(total_size)):.1f}%)"


class ProfilingReport:
    """Timing and result-size data from :meth:`dagreon.Workflow.profile`.

    Attributes:
        compute_times: Mapping from each produced type alias to the time spent
            computing that result, in seconds.
        sizes: Mapping from each produced type alias to ``sys.getsizeof`` for
            that result, in bytes.
        end_to_end_time: Total elapsed profiling time, including planning and
            execution overhead, in seconds.
    """

    def __init__(self):
        self.compute_times: dict[TypeAliasType, float] = {}
        self.sizes: dict[TypeAliasType, int] = {}
        self.end_to_end_time = 0.0

    def summary(self, sort_by: str = "compute_time", threshold: float = 0.98) -> str:
        """Format profiling data as a text table.

        Args:
            sort_by: Metric used to order rows. Must be ``"compute_time"`` or
                ``"size"``.
            threshold: Fraction of the selected metric to show before grouping
                remaining rows into an ``Other`` row. Use ``1.0`` to show all
                rows.

        Returns:
            A multi-line text report containing per-type compute times, result
            sizes, percentages, and totals.

        Raises:
            ValueError: If ``sort_by`` or ``threshold`` is invalid.
        """

        if sort_by not in ("compute_time", "size"):
            raise ValueError("sort_by must be 'compute_time' or 'size'.")

        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be greater than 0.0 and at most 1.0.")

        total_compute_time = sum(self.compute_times.values())
        total_size = sum(self.sizes.values())
        rows = [
            (
                result_type,
                self.compute_times.get(result_type, 0.0),
                self.sizes.get(result_type, 0),
            )
            for result_type in set(self.compute_times) | set(self.sizes)
        ]

        if sort_by == "compute_time":
            rows.sort(key=lambda row: (-row[1], -row[2], str(row[0])))
            threshold_total = total_compute_time
            metric_index = 1
        else:
            rows.sort(key=lambda row: (-row[2], -row[1], str(row[0])))
            threshold_total = float(total_size)
            metric_index = 2

        visible_rows = rows
        hidden_rows: list[tuple[TypeAliasType, float, int]] = []
        if threshold < 1.0 and threshold_total > 0.0:
            visible_rows = []
            cumulative = 0.0
            cutoff = threshold_total * threshold
            for row in rows:
                if cumulative < cutoff:
                    visible_rows.append(row)
                    cumulative += row[metric_index]  # type: ignore
                else:
                    hidden_rows.append(row)

        lines = [
            "Profiling Report",
            f"{'Type':<32} {'Compute Time':>{TIME_COLUMN_WIDTH}} "
            f"{'Size':>{SIZE_COLUMN_WIDTH}}",
            "-" * TABLE_WIDTH,
        ]
        for result_type, compute_time, size in visible_rows:
            lines.append(
                f"{str(result_type):<32} "
                f"{_format_time(compute_time, total_compute_time):>{TIME_COLUMN_WIDTH}} "
                f"{_format_size(size, total_size):>{SIZE_COLUMN_WIDTH}}"
            )

        if hidden_rows:
            other_compute_time = sum(row[1] for row in hidden_rows)
            other_size = sum(row[2] for row in hidden_rows)
            lines.append(
                f"{'Other':<32} "
                f"{_format_time(other_compute_time, total_compute_time):>{TIME_COLUMN_WIDTH}} "
                f"{_format_size(other_size, total_size):>{SIZE_COLUMN_WIDTH}}"
            )

        lines.append("-" * TABLE_WIDTH)
        lines.append(
            f"TOTAL | end-to-end: {self.end_to_end_time:.6f}s | "
            f"compute: {_format_time(total_compute_time, total_compute_time)} | "
            f"size: {_format_size(total_size, total_size).strip()}"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
