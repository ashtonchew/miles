from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from miles.ray import wiring
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.types import ClusterBackend


class TestCreateBackendCapability:
    def test_a_ray_run_still_launches_the_ray_worker_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Ray path is the one every existing run takes, and it must be untouched."""
        sentinel = object()
        launched: list[Any] = []
        monkeypatch.setattr(wiring, "_launch_ray_worker_manager", lambda args: launched.append(args) or sentinel)

        args = SimpleNamespace(cluster_backend=ClusterBackend.RAY.value)
        capability = wiring.create_backend_capability(args)

        assert launched == [args]
        assert isinstance(capability, RayBackendCapability)

    def test_a_kubernetes_run_installs_a_provider_rather_than_launching_workers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under Kubernetes the pods already exist, so launching actors would double the run."""
        installed: list[Any] = []
        sentinel = object()
        monkeypatch.setattr(
            wiring, "_install_kubernetes_workers_from_args", lambda args: installed.append(args) or sentinel
        )
        monkeypatch.setattr(wiring, "_launch_ray_worker_manager", _refuse_ray)

        args = SimpleNamespace(cluster_backend=ClusterBackend.KUBERNETES.value)

        assert wiring.create_backend_capability(args) is sentinel
        assert installed == [args]


def _refuse_ray(args: Any) -> None:
    raise AssertionError("the Kubernetes path must not launch Ray workers")
