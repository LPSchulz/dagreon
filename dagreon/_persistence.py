from pathlib import Path
from time import monotonic
from typing import Any

import lmdb


def initialize_lmdb_directory(
    directory: str | Path, lmdb_args: dict[str, Any] | None = None
) -> Path:
    lmdb_args = {} if lmdb_args is None else dict(lmdb_args)
    directory_path = Path(directory).resolve()
    data_file = directory_path / "data.mdb"
    lock_file = directory_path / "lock.mdb"

    if data_file.exists() or lock_file.exists():
        return directory_path

    directory_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(directory_path), **lmdb_args)
    env.sync()
    env.close()
    print(
        "[dagreon] Initialized new LMDB datastore at "
        f"{directory_path} (created {data_file.name} and {lock_file.name})."
    )
    return directory_path


class BufferedLMDBWriter:
    def __init__(
        self,
        directory: str | Path,
        lmdb_args: dict[str, Any] | None = None,
        flush_interval_s: float = 5.0,
    ):
        self.lmdb_args = {} if lmdb_args is None else dict(lmdb_args)
        self.directory = initialize_lmdb_directory(directory, self.lmdb_args)
        self.flush_interval_s = flush_interval_s
        self._env = lmdb.open(str(self.directory), **self.lmdb_args)
        self._queue: list[tuple[bytes, bytes]] = []
        self._last_flush = monotonic()
        self._closed = False

    def put(self, key: bytes, value: bytes) -> None:
        self._queue.append((key, value))
        self.flush_if_due()

    def get(self, key: bytes) -> bytes | None:
        for queued_key, queued_value in reversed(self._queue):
            if queued_key == key:
                return queued_value
        with self._env.begin(write=False) as txn:
            return txn.get(key)

    def flush_if_due(self) -> None:
        if monotonic() - self._last_flush >= self.flush_interval_s:
            self.flush_now()

    def flush_now(self) -> None:
        if not self._queue:
            self._last_flush = monotonic()
            return
        with self._env.begin(write=True) as txn:
            for key, value in self._queue:
                txn.put(key=key, value=value)
        self._queue.clear()
        self._env.sync()
        self._last_flush = monotonic()

    def close(self) -> None:
        if self._closed:
            return
        self.flush_now()
        self._env.close()
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
