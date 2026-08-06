from __future__ import annotations

import asyncio
from functools import partial

from pydantic import ConfigDict

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_provider.k8s import K8sWorkerProvider
from miles.utils.workers.worker_spec import NamedHostAndPorts


class _Subscription(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    reconcile: ReconcileFn
    spec_names: set[str]


class SharedK8sWorkerProvider(BaseWorkerProvider):
    def __init__(self, *, inner: K8sWorkerProvider, spec_names: list[str]) -> None:
        self._inner = inner
        self._spec_names = spec_names
        self._subscriptions: dict[int, _Subscription] = {}
        self._next_token = 0
        self._stop_inner: StopWatchFn | None = None
        self._spec_name_by_cell: dict[str, str] = {}
        self._start_lock = asyncio.Lock()

    async def watch_cells(self, reconcile: ReconcileFn, *, spec_names: list[str]) -> StopWatchFn:
        await self.start()

        token = self._next_token
        self._next_token += 1
        self._subscriptions[token] = _Subscription(reconcile=reconcile, spec_names=set(spec_names))

        for cell_id in self._inner.cell_ids():
            info = self._inner.cell_info(cell_id)
            if info is not None and info.alive and info.spec_name in spec_names:
                await reconcile(cell_id, info)

        return partial(self._unsubscribe, token)

    async def start(self) -> None:
        async with self._start_lock:
            if self._stop_inner is None:
                self._stop_inner = await self._inner.watch_cells(self._fan_out, spec_names=self._spec_names)

    async def stop(self) -> None:
        stop_inner, self._stop_inner = self._stop_inner, None
        self._subscriptions.clear()
        if stop_inner is not None:
            await stop_inner()

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return await self._inner.get_addrs(worker_name)

    def get_worker_infos_of(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return self._inner.get_worker_infos_of(cell_ids=cell_ids)

    def cell_info(self, cell_id: str) -> CellInfo | None:
        return self._inner.cell_info(cell_id)

    def cell_ids(self) -> list[str]:
        return self._inner.cell_ids()

    def pod_names(self, cell_id: str) -> list[str]:
        return self._inner.pod_names(cell_id)

    async def _fan_out(self, cell_id: str, info: CellInfo | None) -> None:
        if info is not None:
            self._spec_name_by_cell[cell_id] = info.spec_name
            spec_name: str | None = info.spec_name
        else:
            spec_name = self._spec_name_by_cell.pop(cell_id, None)

        for subscription in list(self._subscriptions.values()):
            if spec_name is None or spec_name in subscription.spec_names:
                await subscription.reconcile(cell_id, info)

    async def _unsubscribe(self, token: int) -> None:
        self._subscriptions.pop(token, None)
