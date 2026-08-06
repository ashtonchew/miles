import asyncio
from dataclasses import dataclass, field

from tests.fast.utils.workers.worker_provider.test_k8s_labels import make_pod

from miles.utils.workers.reconcile.k8s_api import PodListPage
from miles.utils.workers.worker_provider.k8s import K8sWorkerProvider
from miles.utils.workers.worker_provider.k8s_shared import SharedK8sWorkerProvider

NAMESPACE = "rl"
SELECTOR = "app.kubernetes.io/instance=r"


@dataclass
class CountingPodApi:
    pods: list = field(default_factory=list)
    lists: int = 0

    async def list_pods(self, *, namespace, label_selector):
        self.lists += 1
        return PodListPage(pods=list(self.pods), resource_version="1")

    async def stream_pods(self, *, namespace, label_selector, resource_version, timeout_seconds):
        await asyncio.sleep(3600)
        yield None


def _shared(api, spec_names=("engine",)):
    return SharedK8sWorkerProvider(
        inner=K8sWorkerProvider(
            namespace=NAMESPACE,
            label_selector=SELECTOR,
            static_addrs={},
            worker_ports={"engine": {"rpc": 8000}},
            kube_client_factory=lambda: api,
            resync_period=None,
        ),
        spec_names=list(spec_names),
    )


class TestOneObservation:
    def test_several_watchers_share_a_single_list_of_the_namespace(self):
        """Every component asks for its own watch, and a watch per component would list the pods again."""
        api = CountingPodApi(pods=[make_pod(name="engine-0-0")])
        provider = _shared(api)

        async def scenario():
            first = await provider.watch_cells(_ignore, spec_names=["engine"])
            second = await provider.watch_cells(_ignore, spec_names=["engine"])
            await first()
            await second()
            await provider.stop()
            return api.lists

        assert asyncio.run(scenario()) == 1

    def test_a_late_watcher_is_told_about_the_cells_that_already_exist(self):
        """The trainer group subscribes after the api server did, and must not miss the running cells."""
        api = CountingPodApi(pods=[make_pod(name="engine-0-0")])
        provider = _shared(api)
        seen: list[tuple[str, object]] = []

        async def scenario():
            first = await provider.watch_cells(_ignore, spec_names=["engine"])
            second = await provider.watch_cells(_record(seen), spec_names=["engine"])
            await first()
            await second()
            await provider.stop()

        asyncio.run(scenario())

        assert [cell_id for cell_id, _ in seen] == ["engine-0"]

    def test_a_watcher_only_hears_about_the_specs_it_asked_for(self):
        """Several controllers share one namespace, and each owns only its own cells."""
        api = CountingPodApi(pods=[make_pod(name="engine-0-0")])
        provider = _shared(api)
        seen: list[tuple[str, object]] = []

        async def scenario():
            stop = await provider.watch_cells(_record(seen), spec_names=["trainer-actor"])
            await stop()
            await provider.stop()

        asyncio.run(scenario())

        assert seen == []

    def test_stopping_one_watcher_leaves_the_others_running(self):
        """One component finishing its rollout must not blind the rest of the process."""
        api = CountingPodApi(pods=[make_pod(name="engine-0-0")])
        provider = _shared(api)

        async def scenario():
            first = await provider.watch_cells(_ignore, spec_names=["engine"])
            await provider.watch_cells(_ignore, spec_names=["engine"])
            await first()
            infos = provider.cell_info("engine-0")
            await provider.stop()
            return infos

        assert asyncio.run(scenario()) is not None

    def test_stopping_twice_is_harmless(self):
        """Shutdown paths overlap, and a second stop must not raise on an already stopped loop."""
        api = CountingPodApi(pods=[make_pod(name="engine-0-0")])
        provider = _shared(api)

        async def scenario():
            await provider.watch_cells(_ignore, spec_names=["engine"])
            await provider.stop()
            await provider.stop()

        asyncio.run(scenario())


async def _ignore(cell_id: str, info) -> None:
    return None


def _record(seen: list):
    async def reconcile(cell_id: str, info) -> None:
        seen.append((cell_id, info))

    return reconcile
