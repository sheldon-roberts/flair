import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Iterator, Optional, Tuple, Union

RecordType = Enum("RecordType", ["scalar", "image", "histogram", "string", "scalar_list"])


class MetricName:
    def __init__(self, name):
        self.parts: Tuple[str, ...]

        if isinstance(name, str):
            self.parts = tuple(name.split("/"))
        else:
            self.parts = tuple(name)

    def __str__(self) -> str:
        return "/".join(self.parts)

    def __repr__(self) -> str:
        return str(self)

    def __iter__(self) -> Iterator[str]:
        return iter(self.parts)

    def __getitem__(self, i) -> Union["MetricName", str]:
        item = self.parts[i]

        if isinstance(i, slice):
            item = self.__class__(item)

        return item

    def __add__(self, other) -> "MetricName":
        if isinstance(other, str):
            return self.__class__(self.parts + (other,))
        elif isinstance(other, MetricName):
            return self.__class__(self.parts + other.parts)
        else:
            return self.__class__(self.parts + tuple(other))

    def __radd__(self, other) -> "MetricName":
        if isinstance(other, str):
            return self.__class__((other,) + self.parts)
        else:
            # no need to check for MetricName, as __add__ of other would be called in this case
            return self.__class__(tuple(other) + self.parts)

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.parts == tuple(other.split("/"))
        elif isinstance(other, MetricName):
            return self.parts == other.parts
        elif other is None:
            return False
        else:
            return self.parts == tuple(other)

    def __hash__(self):
        return hash(self.parts)


@dataclass
class MetricRecord:
    """Represents a recorded metric value."""

    def __init__(
        self,
        name: Union[Iterable[str], str],
        value: Any,
        global_step: int,
        typ: RecordType,
        *,
        walltime: Optional[float] = None,
    ):
        """Create a metric record.

        :param name: Name of the metric.
        :param typ: Type of metric.
        :param value: Value of the metric (can be anything: scalar, tensor,
            image, etc.).
        :param walltime: Time of recording this metric.
        """
        self.name: MetricName = MetricName(name)
        self.typ: RecordType = typ
        self.value: Any = value
        self.global_step: int = global_step
        self.walltime: float = walltime if walltime is not None else time.time()

    @property
    def joined_name(self) -> str:
        return str(self.name)

    @classmethod
    def scalar(cls, name: Iterable[str], value: Any, global_step: int, *, walltime=None):
        return cls(name=name, value=value, global_step=global_step, typ=RecordType.scalar, walltime=walltime)

    @classmethod
    def scalar_list(cls, name: Iterable[str], value: list, global_step: int, *, walltime=None):
        return cls(name=name, value=value, global_step=global_step, typ=RecordType.scalar_list, walltime=walltime)

    @classmethod
    def string(cls, name: Iterable[str], value: str, global_step: int, *, walltime=None):
        return cls(name=name, value=value, global_step=global_step, typ=RecordType.string, walltime=walltime)

    @classmethod
    def histogram(cls, name: Iterable[str], value: str, global_step: int, *, walltime=None):
        return cls(name=name, value=value, global_step=global_step, typ=RecordType.histogram, walltime=walltime)

    def is_type(self, typ):
        return self.typ == typ

    @property
    def is_scalar(self):
        return self.is_type(RecordType.scalar)

    @property
    def is_scalar_list(self):
        return self.is_type(RecordType.scalar_list)

    @property
    def is_string(self):
        return self.is_type(RecordType.string)

    @property
    def is_histogram(self):
        return self.is_type(RecordType.histogram)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.joined_name} at step {self.global_step}, {self.walltime:.4f})"
