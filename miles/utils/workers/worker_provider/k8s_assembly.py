from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from miles.utils.workers.backend_capability.k8s import KubernetesBackendCapability
from miles.utils.workers.cell_operations.k8s import K8sCellOperations
from miles.utils.workers.types import DEFAULT_GPUS_PER_NODE
from miles.utils.workers.worker_provider.k8s import K8sWorkerProvider
from miles.utils.workers.worker_provider.k8s_labels import CellLabelKeys
from miles.utils.workers.worker_provider.k8s_shared import SharedK8sWorkerProvider
from miles.utils.workers.worker_provider.spec_tables import (
    fleet_ranks_per_pod,
    fleet_spec_metas,
    fleet_spec_names,
    fleet_worker_classes,
    fleet_worker_ports,
    static_worker_addrs,
    static_worker_provider,
)
from miles.utils.workers.worker_spec import BaseWorkerSpec

INSTANCE_LABEL = "app.kubernetes.io/instance"


def install_kubernetes_workers(
    *,
    specs: list[BaseWorkerSpec],
    namespace: str,
    release: str,
    kube_client_factory: Callable[[], Any],
    delete_pods: Callable[[list[str]], Awaitable[None]],
    num_gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
    label_keys: CellLabelKeys | None = None,
    colocated_with: Callable[[str], list[str]] | None = None,
) -> KubernetesBackendCapability:
    watched_spec_names = fleet_spec_names(specs=specs)
    provider = SharedK8sWorkerProvider(
        inner=K8sWorkerProvider(
            namespace=namespace,
            label_selector=f"{INSTANCE_LABEL}={release}",
            static_addrs=static_worker_addrs(specs=specs, release=release),
            worker_ports=fleet_worker_ports(specs=specs),
            worker_classes=fleet_worker_classes(specs=specs),
            spec_metas=fleet_spec_metas(specs=specs),
            ranks_per_pod=fleet_ranks_per_pod(specs=specs, num_gpus_per_node=num_gpus_per_node),
            kube_client_factory=kube_client_factory,
            label_keys=label_keys,
        ),
        spec_names=watched_spec_names,
    )

    return KubernetesBackendCapability(
        cells_provider=provider,
        cells_spec_names=watched_spec_names,
        static_provider=static_worker_provider(specs=specs, release=release),
        cell_operations=K8sCellOperations(
            provider=provider,
            spec_names=watched_spec_names,
            delete_pods=delete_pods,
            colocated_with=colocated_with,
        ),
    )
