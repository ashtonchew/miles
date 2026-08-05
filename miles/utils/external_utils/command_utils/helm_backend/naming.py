from __future__ import annotations

from miles.utils.workers.worker_provider.k8s_naming import (
    CHART_NAME,
    NAME_BUDGET,
    component_name,
    fleet_leader_host,
    fleet_worker_host,
    fullname,
    release_name,
    static_worker_host,
)

__all__ = [
    "CHART_NAME",
    "NAME_BUDGET",
    "component_name",
    "fleet_leader_host",
    "fleet_worker_host",
    "fullname",
    "release_name",
    "static_worker_host",
]
