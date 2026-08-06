from collections.abc import Awaitable, Callable

from miles.utils.function_registry import load_function
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import RPC_PORT_NAME, NamedHostAndPorts


async def apply_cell_observation(
    *,
    cell_id: str,
    observed: CellInfo | None,
    actual_workers_hash: str | None,
    add: Callable[[str, CellInfo], Awaitable[None]],
    remove: Callable[[str], Awaitable[None]],
) -> None:
    if observed is not None and actual_workers_hash is None:
        await add(cell_id, observed)
    elif observed is None and actual_workers_hash is not None:
        await remove(cell_id)
    elif observed is not None and actual_workers_hash is not None and actual_workers_hash != observed.workers_hash:
        await remove(cell_id)
        await add(cell_id, observed)


class WorkerClassLoader:
    def __init__(self, paths: dict[str, str]) -> None:
        self._paths = paths
        self._classes: dict[str, type] = {}

    def of_spec(self, spec_name: str) -> type:
        if spec_name not in self._classes:
            path = self._paths.get(spec_name)
            assert path is not None, (
                f"spec {spec_name} has no worker class, so its rpc methods are unknown; "
                f"known specs are {sorted(self._paths)}"
            )
            self._classes[spec_name] = load_function(path)
        return self._classes[spec_name]


def build_rpc_handle(*, worker_class: type, addrs: NamedHostAndPorts, spec_name: str) -> BaseWorkerHandle:
    assert RPC_PORT_NAME in addrs, f"spec {spec_name} has no {RPC_PORT_NAME!r} port to be called through"
    return RpcWorkerHandle(worker_class, server_url=addrs[RPC_PORT_NAME].addr, require_stable_boot_uuid=True)
