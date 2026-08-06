from __future__ import annotations

from collections.abc import Sequence

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider


class KubernetesBackendCapability(BackendCapability):
    def __init__(
        self,
        *,
        cells_provider: BaseWorkerProvider,
        cells_spec_names: Sequence[str],
        static_provider: SimpleWorkerProvider,
        cell_operations: BaseCellOperations,
    ) -> None:
        self._cells_provider = cells_provider
        self._cells_spec_names = list(cells_spec_names)
        self._static_provider = static_provider
        self._cell_operations = cell_operations

    def dynamic_worker_provider(self, *, spec_names: Sequence[str]) -> BaseWorkerProvider:
        unwatched = [name for name in spec_names if name not in self._cells_spec_names]
        assert not unwatched, (
            f"{unwatched} are not watched by this run's provider, which observes {self._cells_spec_names}; "
            f"their cells would never be reported"
        )
        return self._cells_provider

    def static_worker_provider(self, *, worker_name: str) -> BaseWorkerProvider:
        assert self._static_provider.knows_worker(
            worker_name
        ), f"worker {worker_name} is not in this run's static address book, so no address can be predicted for it"
        return self._static_provider

    def cell_operations(self) -> BaseCellOperations:
        return self._cell_operations
