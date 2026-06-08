import inspect
import json
from dataclasses import asdict, dataclass
from typing import (
    Callable,
    Protocol,
    TypeAliasType,
    TypeVar,
    dataclass_transform,
    runtime_checkable,
)

from typing_extensions import TypeIs
from xxhash import xxh3_128_digest


@dataclass(frozen=True)
class TaskSpec:
    input_types: list[TypeAliasType]
    output_type: TypeAliasType


@runtime_checkable
class TaskClass(Protocol):
    __task_spec__: TaskSpec
    __task_fingerprint__: bytes

    def __call__(self, *args: TypeAliasType) -> TypeAliasType: ...


def _stable_json(obj) -> str:
    """Best-effort stable encoding for common config values.

    Falls back to `repr()` for unknown types.
    """

    def _convert(v):
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, (list, tuple)):
            return [_convert(x) for x in v]
        if isinstance(v, dict):
            return {
                str(k): _convert(vv)
                for k, vv in sorted(v.items(), key=lambda kv: str(kv[0]))
            }
        if isinstance(v, (set, frozenset)):
            return sorted(
                (_convert(x) for x in v),
                key=lambda x: json.dumps(x, sort_keys=True, default=str),
            )
        # TODO: test for numpy arrays, maybe add something for those
        return {"__repr__": repr(v)}

    return json.dumps(_convert(obj), sort_keys=True, separators=(",", ":"), default=str)


def is_task(obj: Callable) -> TypeIs[TaskClass]:
    return hasattr(obj, "__task_spec__")


def _inspect_task(task: type) -> tuple[TypeAliasType, list[TypeAliasType]]:
    sig = inspect.signature(task.__call__)
    output_type, input_types = (
        sig.return_annotation,
        [v.annotation for v in sig.parameters.values()][1:],  # skip "self"
    )
    if output_type is inspect.Signature.empty:
        raise TypeError(f"Output type annotation missing in {task}")
    if type(output_type) is not TypeAliasType:
        raise TypeError(
            f"Output type annotation {output_type} in {task} is not a TypeAliasType"
        )
    for req in input_types:
        if req is inspect.Signature.empty:
            raise TypeError(f"Input type annotation missing in {task}")
        if type(req) is not TypeAliasType:
            raise TypeError(
                f"Input type annotation {req} in {task} is not a TypeAliasType"
            )
    if len(set(input_types)) != len(input_types):
        raise TypeError(f"Duplicate input type annotations in {task}")
    if output_type in input_types:
        raise TypeError(
            f"Output type annotation {output_type} in {task} duplicates an input type"
        )
    return output_type, input_types


# T = TypeVar("T")  # , bound=Task)
T = TypeVar("T", bound=type)


@dataclass_transform(frozen_default=True)
def task(cls: T) -> T:
    """Decorate a class as a Dagreon task.

    The decorated class must define its own ``__call__`` method. Every
    ``__call__`` input parameter after ``self`` and the return value must be a
    Python 3.12 type alias, for example ``type Samples = list[float]``. Dagreon
    uses those alias objects as dependency graph nodes.

    Args:
        cls: Class to convert into a task.

    Returns:
        The decorated class, converted to a frozen dataclass with Dagreon task
        metadata.

    Raises:
        TypeError: If the class does not define ``__call__``, if an input or
            output annotation is missing, if an annotation is not a Python 3.12
            type alias, if an input type is repeated, or if the output type is
            also used as an input type.
    """

    if "__call__" not in cls.__dict__:
        raise TypeError(f"Task {cls.__name__} must define its own __call__ method")
    output_type, input_types = _inspect_task(cls)
    cls.__task_spec__ = TaskSpec(input_types, output_type)

    # Ensure a __post_init__ exists *before* dataclass() runs so the generated
    # __init__ will call it.
    previous_post_init = cls.__dict__.get("__post_init__")

    def __post_init__(self, *args, **kwargs):
        if callable(previous_post_init):
            previous_post_init(self, *args, **kwargs)

        class_str = (
            f"{cls.__module__}.{cls.__qualname__}"
            + "|".join([f"{t.__module__}.{t.__name__}" for t in input_types])
            + "->"
            + f"{output_type.__module__}.{output_type.__name__}"
        )
        instance_str = _stable_json(asdict(self))
        # bypass frozen restriction
        object.__setattr__(
            self, "__task_fingerprint__", xxh3_128_digest(class_str + instance_str)
        )

    cls.__post_init__ = __post_init__
    cls = dataclass(frozen=True)(cls)
    return cls
