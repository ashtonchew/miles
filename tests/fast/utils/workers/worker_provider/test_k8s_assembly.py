from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest
from tests.fast.utils.workers.worker_provider.run_specs import RELEASE, make_engine_spec, make_router_spec
from tests.fast.utils.workers.worker_provider.test_k8s import FakePodApi
from tests.fast.utils.workers.worker_provider.test_k8s_labels import make_pod

from miles.utils.workers.backend_capability.k8s import KubernetesBackendCapability
from miles.utils.workers.worker_provider.k8s_assembly import install_kubernetes_workers
from miles.utils.workers.worker_provider.k8s_shared import SharedK8sWorkerProvider
from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider

NAMESPACE = "team-a"
RAY_WORKER_MANAGER_MODULE = "miles.utils.workers.ray_worker_manager"


def install_workers(*, deleted: list[list[str]], pods: list[Any] | None = None) -> KubernetesBackendCapability:
    api = FakePodApi(pods=list(pods or []))

    async def delete_pods(pod_names: list[str]) -> None:
        deleted.append(list(pod_names))

    return install_kubernetes_workers(
        specs=[make_router_spec(), make_engine_spec()],
        namespace=NAMESPACE,
        release=RELEASE,
        kube_client_factory=lambda: api,
        delete_pods=delete_pods,
    )


class TestKubernetesAssembly:
    def test_components_of_the_process_then_see_a_kubernetes_provider(self) -> None:
        """The whole point of the assembly: the capability must stop answering with Ray."""
        capability = install_workers(deleted=[])

        assert isinstance(capability.dynamic_worker_provider(spec_names=["engine"]), SharedK8sWorkerProvider)

    def test_a_static_worker_resolves_to_the_address_the_chart_gives_it(self) -> None:
        """A router has no cell to observe, so its address is predicted from the release name."""
        capability = install_workers(deleted=[])

        provider = capability.static_worker_provider(worker_name="inference-router-0-0-0")
        addr = asyncio.run(provider.get_addr("inference-router-0-0-0"))

        assert addr.host == f"{RELEASE}-inference-router-0-0.{RELEASE}-inference-router-0"
        assert addr.port == 8000

    def test_refuses_a_static_worker_the_run_never_deployed(self) -> None:
        """Answering with an invented address would send the caller at nothing at all."""
        capability = install_workers(deleted=[])

        with pytest.raises(AssertionError, match="static address book"):
            capability.static_worker_provider(worker_name="inference-router-9-0-0")

    def test_the_static_scope_answers_with_the_address_book_rather_than_with_the_watcher(self) -> None:
        """Statically addressed components need no watch, and a watch would never report them anyway."""
        capability = install_workers(deleted=[])

        static = capability.static_worker_provider(worker_name="inference-router-0-0-0")

        assert isinstance(static, SimpleWorkerProvider)
        assert static is not capability.dynamic_worker_provider(spec_names=["engine"])

    def test_every_component_shares_one_observation_of_the_namespace(self) -> None:
        """A second instance would open a second watch of the same pods and cache them twice."""
        capability = install_workers(deleted=[])

        assert capability.dynamic_worker_provider(spec_names=["engine"]) is capability.dynamic_worker_provider(
            spec_names=["engine"]
        )

    def test_suspending_a_cell_deletes_its_pods(self) -> None:
        """Kubernetes has no suspend: a cell heals because its workload recreates deleted pods."""
        deleted: list[list[str]] = []
        capability = install_workers(
            deleted=deleted, pods=[make_pod(name="engine-0-0", fleet="engine", cell_index="0")]
        )
        operations = capability.cell_operations()

        asyncio.run(operations.suspend(cell_id="engine-0"))

        assert deleted == [["engine-0-0"]]

    def test_listing_cells_starts_the_watch_it_needs(self) -> None:
        """The api server asks for cells without knowing that observation has to be started first."""
        capability = install_workers(deleted=[], pods=[make_pod(name="engine-0-0", fleet="engine", cell_index="0")])
        operations = capability.cell_operations()

        infos = asyncio.run(operations.cell_infos(spec_names=["engine"]))

        assert list(infos) == ["engine-0"]

    def test_never_reaches_for_the_ray_worker_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A namespace has no Ray cluster, so touching the manager would fail the run there."""
        monkeypatch.setitem(sys.modules, RAY_WORKER_MANAGER_MODULE, None)
        deleted: list[list[str]] = []
        capability = install_workers(
            deleted=deleted, pods=[make_pod(name="engine-0-0", fleet="engine", cell_index="0")]
        )

        operations = capability.cell_operations()
        asyncio.run(operations.suspend(cell_id="engine-0"))

        assert isinstance(capability.dynamic_worker_provider(spec_names=["engine"]), SharedK8sWorkerProvider)
        assert deleted == [["engine-0-0"]]
