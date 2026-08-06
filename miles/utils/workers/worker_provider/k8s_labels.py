from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from miles.utils.workers.naming import compute_cell_id

LWS_NAME_LABEL = "leaderworkerset.sigs.k8s.io/name"
LWS_GROUP_INDEX_LABEL = "leaderworkerset.sigs.k8s.io/group-index"
LWS_WORKER_INDEX_LABEL = "leaderworkerset.sigs.k8s.io/worker-index"
LWS_SIZE_LABEL = "leaderworkerset.sigs.k8s.io/size"

SPEC_NAME_LABEL = "miles.radixark.io/spec-name"
META_ANNOTATION_PREFIX = "miles.radixark.io/meta-"
GPU_IDS_META = "gpu_ids"


@dataclass(frozen=True)
class CellLabelKeys:
    fleet: str = LWS_NAME_LABEL
    cell_index: str = LWS_GROUP_INDEX_LABEL
    worker_index: str = LWS_WORKER_INDEX_LABEL
    spec_name: str = SPEC_NAME_LABEL
    cell_size: str = LWS_SIZE_LABEL
    meta_annotation_prefix: str = META_ANNOTATION_PREFIX


DEFAULT_LABEL_KEYS = CellLabelKeys()


@dataclass(frozen=True)
class ObservedWorker:
    name: str
    cell_id: str
    spec_name: str
    worker_index: int
    ready: bool
    pod_ip: str | None
    uid: str
    restart_count: int
    node_name: str | None
    cell_size: int = 0
    subdomain: str | None = None
    gpu_ids: tuple[int, ...] = field(default_factory=tuple)


def observe_pod(pod: Any, keys: CellLabelKeys = DEFAULT_LABEL_KEYS) -> ObservedWorker | None:
    metadata = pod.metadata
    labels = metadata.labels or {}
    fleet = labels.get(keys.fleet)
    cell_index = labels.get(keys.cell_index)
    if fleet is None or cell_index is None:
        return None

    status = pod.status
    return ObservedWorker(
        name=metadata.name,
        cell_id=compute_cell_id(spec_name=fleet, cell_index=int(cell_index)),
        spec_name=labels.get(keys.spec_name, fleet),
        worker_index=int(labels.get(keys.worker_index, 0)),
        ready=_is_ready(status),
        pod_ip=getattr(status, "pod_ip", None),
        uid=metadata.uid,
        restart_count=sum(
            container.restart_count for container in (getattr(status, "container_statuses", None) or [])
        ),
        node_name=getattr(pod.spec, "node_name", None),
        cell_size=int(labels.get(keys.cell_size, 0)),
        subdomain=getattr(pod.spec, "subdomain", None),
        gpu_ids=_parse_gpu_ids(read_meta(pod, keys).get(GPU_IDS_META, "")),
    )


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def _is_ready(status: Any) -> bool:
    conditions = getattr(status, "conditions", None) or []
    return any(condition.type == "Ready" and condition.status == "True" for condition in conditions)


def cell_members_hash(workers: list[ObservedWorker]) -> str:
    parts = [
        f"{worker.name}:{worker.uid}:{worker.restart_count}"
        for worker in sorted(workers, key=lambda worker: worker.worker_index)
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def read_meta(pod: Any, keys: CellLabelKeys = DEFAULT_LABEL_KEYS) -> dict[str, str]:
    annotations = pod.metadata.annotations or {}
    prefix = keys.meta_annotation_prefix
    return {key[len(prefix) :]: value for key, value in annotations.items() if key.startswith(prefix)}
