from __future__ import annotations

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.backend_capability.ray import RayBackendCapability


def create_backend_capability(args) -> BackendCapability:
    return RayBackendCapability(worker_manager_handle=_launch_ray_worker_manager(args))


def _launch_ray_worker_manager(args):
    from miles.ray.placement_group import create_placement_groups
    from miles.ray.specs.entrypoint import compute_specs
    from miles.utils.workers.ray_worker_manager import RayWorkerManager

    specs = compute_specs(args)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(specs, pgs)
