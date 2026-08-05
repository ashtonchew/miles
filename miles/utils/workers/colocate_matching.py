from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import NamedTuple

from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.naming import parse_cell_id


class GpuPlacement(FrozenStrictBaseModel):
    worker_name: str
    node_name: str
    gpu_ids: tuple[int, ...]

    def gpus(self) -> list[tuple[str, int]]:
        return [(self.node_name, gpu_id) for gpu_id in self.gpu_ids]


class ColocatePairing(FrozenStrictBaseModel):
    engine_gpus: tuple[tuple[str, int], ...]
    trainer_of_gpu: dict[tuple[str, int], str]

    def trainer_names_in_engine_order(self) -> list[str]:
        return [self.trainer_of_gpu[gpu] for gpu in self.engine_gpus]


def match_by_gpu(
    *, engine_placements: Iterable[GpuPlacement], trainer_placements: Iterable[GpuPlacement]
) -> ColocatePairing:
    engine_placements = list(engine_placements)
    trainer_placements = list(trainer_placements)

    trainer_of_gpu: dict[tuple[str, int], str] = {}
    for placement in trainer_placements:
        for gpu in placement.gpus():
            assert (
                gpu not in trainer_of_gpu
            ), f"{gpu[0]} gpu {gpu[1]} is claimed by both {trainer_of_gpu[gpu]} and {placement.worker_name}"
            trainer_of_gpu[gpu] = placement.worker_name

    engine_gpus = tuple(gpu for placement in engine_placements for gpu in placement.gpus())
    unmatched = [gpu for gpu in engine_gpus if gpu not in trainer_of_gpu]
    assert not unmatched, (
        f"no trainer shares {_describe(unmatched)}; colocate needs a trainer rank on every engine gpu, "
        f"and a weight update to an unshared gpu would transfer nothing"
    )

    duplicated = [gpu for gpu in set(engine_gpus) if engine_gpus.count(gpu) > 1]
    assert not duplicated, f"{_describe(duplicated)} is claimed by more than one engine rank"

    return ColocatePairing(engine_gpus=engine_gpus, trainer_of_gpu=trainer_of_gpu)


def assert_same_nodes(
    *, engine_placements: Iterable[GpuPlacement], trainer_placements: Iterable[GpuPlacement]
) -> None:
    engine_nodes = {placement.node_name for placement in engine_placements}
    trainer_nodes = {placement.node_name for placement in trainer_placements}
    stray = sorted(engine_nodes - trainer_nodes)
    assert not stray, f"engines run on {stray}, where no trainer of this group does"


def local_gpu_index(*, gpu_uuid: str, visible_uuids: list[str]) -> int:
    assert gpu_uuid in visible_uuids, f"gpu {gpu_uuid} is not visible here; this process sees {visible_uuids}"
    return visible_uuids.index(gpu_uuid)


def _describe(gpus: list[tuple[str, int]]) -> str:
    return ", ".join(f"{node} gpu {gpu_id}" for node, gpu_id in sorted(gpus))


class PairingLayout(FrozenStrictBaseModel):
    engine_cells: int = Field(ge=1)
    trainer_cells: int = Field(ge=1)
    pods_per_engine_cell: int = Field(ge=1)
    pods_per_trainer_cell: int = Field(ge=1)

    @property
    def engines_per_trainer_cell(self) -> int:
        return self.pods_per_trainer_cell // self.pods_per_engine_cell


def assert_colocate_supported(
    *, layout: PairingLayout, gpus_per_engine_pod: int, gpus_per_trainer_pod: int, gpus_per_node: int
) -> None:
    assert_layout_pairs(layout=layout)
    assert gpus_per_engine_pod == gpus_per_node, (
        f"an engine pod holding {gpus_per_engine_pod} of a node's {gpus_per_node} gpus is a sub-node cell, "
        f"which colocate does not support: the device plugin picks the cards, so the engine's base gpu id "
        f"cannot be rendered before the pod runs"
    )
    assert gpus_per_trainer_pod == gpus_per_node, (
        f"a trainer pod holding {gpus_per_trainer_pod} of a node's {gpus_per_node} gpus is a sub-node cell, "
        f"which colocate does not support: two trainer cells could then share a node and an engine would "
        f"have no single cell to pair with"
    )


def assert_layout_pairs(*, layout: PairingLayout) -> None:
    assert layout.pods_per_engine_cell <= layout.pods_per_trainer_cell, (
        f"an engine cell of {layout.pods_per_engine_cell} pods cannot fit in a trainer cell of "
        f"{layout.pods_per_trainer_cell}; colocate needs every engine rank to sit on a trainer's node"
    )
    assert layout.pods_per_trainer_cell % layout.pods_per_engine_cell == 0, (
        f"{layout.pods_per_trainer_cell} trainer pods per cell is not a whole number of "
        f"{layout.pods_per_engine_cell}-pod engine cells, so an engine would straddle two trainer cells"
    )
    assert layout.engine_cells <= layout.trainer_cells * layout.engines_per_trainer_cell, (
        f"{layout.engine_cells} engine cells do not fit in {layout.trainer_cells} trainer cells holding "
        f"{layout.engines_per_trainer_cell} each; an engine rank whose gpu no trainer shares would "
        f"receive nothing from a weight update"
    )


def target_trainer_pod(
    *, engine_cell_index: int, engine_pod_index: int, layout: PairingLayout, trainer_fleet: str
) -> str:
    assert 0 <= engine_cell_index < layout.engine_cells, f"{engine_cell_index=} outside {layout}"
    assert 0 <= engine_pod_index < layout.pods_per_engine_cell, f"{engine_pod_index=} outside {layout}"
    assert_layout_pairs(layout=layout)

    trainer_cell_index = engine_cell_index // layout.engines_per_trainer_cell
    offset_within_cell = engine_cell_index % layout.engines_per_trainer_cell
    trainer_pod_index = offset_within_cell * layout.pods_per_engine_cell + engine_pod_index
    return lws_pod_name(fleet=trainer_fleet, cell_index=trainer_cell_index, pod_index=trainer_pod_index)


def lws_pod_name(*, fleet: str, cell_index: int, pod_index: int) -> str:
    if pod_index == 0:
        return f"{fleet}-{cell_index}"
    return f"{fleet}-{cell_index}-{pod_index}"


class ParsedLwsPodName(NamedTuple):
    cell_index: int
    pod_index: int


def parse_lws_pod_name(*, pod_name: str, fleet: str) -> ParsedLwsPodName:
    prefix = f"{fleet}-"
    assert pod_name.startswith(prefix), f"pod '{pod_name}' is not a pod of leader worker set '{fleet}'"
    indices = pod_name[len(prefix) :].split("-")
    assert len(indices) in (1, 2) and all(
        index.isdigit() for index in indices
    ), f"pod '{pod_name}' does not name a cell and a pod of '{fleet}' the way leader worker set does"
    cell_index, pod_index = (*indices, "0")[:2]
    return ParsedLwsPodName(cell_index=int(cell_index), pod_index=int(pod_index))


def engine_pods_of_trainer_cell(
    *, trainer_cell_index: int, layout: PairingLayout, engine_fleet: str, trainer_fleet: str
) -> list[str]:
    return [
        lws_pod_name(fleet=engine_fleet, cell_index=engine_cell_index, pod_index=engine_pod_index)
        for engine_cell_index in range(layout.engine_cells)
        for engine_pod_index in range(layout.pods_per_engine_cell)
        if parse_lws_pod_name(
            pod_name=target_trainer_pod(
                engine_cell_index=engine_cell_index,
                engine_pod_index=engine_pod_index,
                layout=layout,
                trainer_fleet=trainer_fleet,
            ),
            fleet=trainer_fleet,
        ).cell_index
        == trainer_cell_index
    ]


def colocated_pods_of(
    *, layout: PairingLayout, engine_fleet: str, trainer_fleet: str, trainer_spec_name: str
) -> Callable[[str], list[str]]:
    def colocated_with(cell_id: str) -> list[str]:
        parsed = parse_cell_id(cell_id)
        if parsed.spec_name != trainer_spec_name:
            return []
        return engine_pods_of_trainer_cell(
            trainer_cell_index=parsed.cell_index,
            layout=layout,
            engine_fleet=engine_fleet,
            trainer_fleet=trainer_fleet,
        )

    return colocated_with
