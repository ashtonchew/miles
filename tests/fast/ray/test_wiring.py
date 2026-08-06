from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from tests.fast.utils.workers.worker_provider.test_k8s_assembly import install_workers

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

    def test_a_worker_process_builds_its_capability_only_when_something_asks_for_a_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every served worker builds this context, and most specs never look at it."""
        attached: list[list[str]] = []

        def _attach(worker_argv: list[str]):
            attached.append(worker_argv)
            return install_workers(deleted=[])

        monkeypatch.setattr(wiring, "_attach_backend_capability_from_argv", _attach)

        capability = wiring.create_worker_backend_capability(worker_argv=["--rollout-num-gpus", "8"])
        assert attached == []

        capability.dynamic_worker_provider(spec_names=["engine"])
        capability.dynamic_worker_provider(spec_names=["engine"])

        assert attached == [["--rollout-num-gpus", "8"]]


def _refuse_ray(args: Any) -> None:
    raise AssertionError("the Kubernetes path must not launch Ray workers")
