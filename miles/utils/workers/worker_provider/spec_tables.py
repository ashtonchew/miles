from __future__ import annotations

from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_provider.k8s_naming import static_worker_host
from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    BaseWorkerSpec,
    HostAndPort,
    NamedHostAndPorts,
    ServeWorkerSpec,
    SpecMetaFn,
)


def static_worker_provider(*, specs: list[BaseWorkerSpec], release: str) -> SimpleWorkerProvider:
    cells: dict[str, list[str]] = {}
    spec_names: dict[str, str] = {}
    for spec in specs:
        if is_fleet_spec(spec):
            continue
        for cell_index in range(spec.scheduling.num_cells):
            cell_id = compute_cell_id(spec_name=spec.name, cell_index=cell_index)
            cells[cell_id] = [
                compute_worker_name(spec_name=spec.name, cell_index=cell_index, worker_in_cell_index=index)
                for index in range(spec.scheduling.num_workers_per_cell)
            ]
            spec_names[cell_id] = spec.name

    return SimpleWorkerProvider(
        addrs=static_worker_addrs(specs=specs, release=release),
        cells=cells,
        spec_names=spec_names,
        worker_classes=static_worker_classes(specs=specs),
    )


def static_worker_addrs(*, specs: list[BaseWorkerSpec], release: str) -> dict[str, NamedHostAndPorts]:
    addrs: dict[str, NamedHostAndPorts] = {}
    for spec in specs:
        if is_fleet_spec(spec):
            continue
        for cell_index in range(spec.scheduling.num_cells):
            host = static_worker_host(release, spec.name, cell_index)
            for worker_in_cell_index in range(spec.scheduling.num_workers_per_cell):
                worker_name = compute_worker_name(
                    spec_name=spec.name, cell_index=cell_index, worker_in_cell_index=worker_in_cell_index
                )
                addrs[worker_name] = {
                    port.name: HostAndPort(host=host, port=port.static_port) for port in spec.port_infos
                }
    return addrs


def static_worker_classes(*, specs: list[BaseWorkerSpec]) -> dict[str, str]:
    return {
        spec.name: spec.worker_class for spec in specs if not is_fleet_spec(spec) and isinstance(spec, ServeWorkerSpec)
    }


def fleet_spec_names(*, specs: list[BaseWorkerSpec]) -> list[str]:
    return [spec.name for spec in specs if is_fleet_spec(spec)]


def fleet_worker_ports(*, specs: list[BaseWorkerSpec]) -> dict[str, dict[str, int]]:
    return {
        spec.name: {port.name: port.static_port for port in spec.port_infos} for spec in specs if is_fleet_spec(spec)
    }


def fleet_worker_classes(*, specs: list[BaseWorkerSpec]) -> dict[str, str]:
    return {
        spec.name: spec.worker_class for spec in specs if is_fleet_spec(spec) and isinstance(spec, ServeWorkerSpec)
    }


def fleet_spec_metas(*, specs: list[BaseWorkerSpec]) -> dict[str, SpecMetaFn]:
    return {spec.name: spec.meta for spec in specs if is_fleet_spec(spec) and spec.meta is not None}


def fleet_ranks_per_pod(*, specs: list[BaseWorkerSpec], num_gpus_per_node: int) -> dict[str, int]:
    return {
        spec.name: _ranks_per_pod_of(spec, num_gpus_per_node=num_gpus_per_node)
        for spec in specs
        if is_fleet_spec(spec)
    }


def is_fleet_spec(spec: BaseWorkerSpec) -> bool:
    return spec.scheduling.num_workers_per_cell * spec.scheduling.num_gpu_slots_per_worker > 0


def _ranks_per_pod_of(spec: BaseWorkerSpec, *, num_gpus_per_node: int) -> int:
    if not isinstance(spec, ServeWorkerSpec):
        return 1

    ranks_per_pod = min(spec.scheduling.num_workers_per_cell, num_gpus_per_node)
    _assert_rank_ports_are_free(spec, ranks_per_pod=ranks_per_pod)
    return ranks_per_pod


def _assert_rank_ports_are_free(spec: ServeWorkerSpec, *, ranks_per_pod: int) -> None:
    rpc_port = next(port.static_port for port in spec.port_infos if port.name == RPC_PORT_NAME)
    for port in spec.port_infos:
        if port.name == RPC_PORT_NAME:
            continue
        assert rpc_port + ranks_per_pod <= port.static_port or port.static_port + port.num_consecutive <= rpc_port, (
            f"spec '{spec.name}' serves {ranks_per_pod} ranks per pod from {RPC_PORT_NAME} port {rpc_port} "
            f"upwards, which reaches into the {port.num_consecutive} port(s) '{port.name}' claims from "
            f"{port.static_port}"
        )
