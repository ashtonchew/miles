from __future__ import annotations

from miles.utils.workers.backend_capability.base import BackendCapability, DeferredBackendCapability
from miles.utils.workers.backend_capability.k8s import KubernetesBackendCapability
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_provider.k8s_assembly import install_kubernetes_workers
from miles.utils.workers.worker_provider.k8s_env import (
    current_colocated_with,
    current_label_keys,
    current_namespace,
    current_release,
)
from miles.utils.workers.worker_provider.kube_client import create_kube_client, pod_deleter


def create_backend_capability(args) -> BackendCapability:
    if ClusterBackend(args.cluster_backend) is ClusterBackend.KUBERNETES:
        return _install_kubernetes_workers_from_args(args)
    return RayBackendCapability(worker_manager_handle=_launch_ray_worker_manager(args))


def create_worker_backend_capability(*, worker_argv: list[str]) -> BackendCapability:
    return DeferredBackendCapability(create=lambda: _attach_backend_capability_from_argv(worker_argv))


def _attach_backend_capability_from_argv(worker_argv: list[str]) -> BackendCapability:
    from miles.utils.arguments import parse_args_from_argv

    args = parse_args_from_argv(worker_argv)
    if ClusterBackend(args.cluster_backend) is ClusterBackend.KUBERNETES:
        return _install_kubernetes_workers_from_args(args)

    from miles.utils.workers.ray_worker_manager import RayWorkerManager

    return RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())


def _install_kubernetes_workers_from_args(args) -> KubernetesBackendCapability:
    from miles.ray.specs.entrypoint import compute_specs

    namespace = current_namespace()
    return install_kubernetes_workers(
        specs=compute_specs(args),
        namespace=namespace,
        release=current_release(),
        kube_client_factory=create_kube_client,
        delete_pods=pod_deleter(namespace=namespace),
        num_gpus_per_node=args.num_gpus_per_node,
        colocated_with=current_colocated_with(),
        label_keys=current_label_keys(),
    )


def _launch_ray_worker_manager(args):
    from miles.ray.placement_group import create_placement_groups
    from miles.ray.specs.entrypoint import compute_specs
    from miles.utils.workers.ray_worker_manager import RayWorkerManager

    specs = compute_specs(args)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(specs, pgs)
