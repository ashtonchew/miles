import asyncio
from dataclasses import dataclass, field

import pytest
from tests.fast.utils.workers.worker_provider.test_k8s_labels import make_pod

from miles.utils.workers.reconcile.k8s_api import PodListPage, PodWatchEvent
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_provider.k8s import K8sWorkerProvider
from miles.utils.workers.worker_provider.k8s_labels import SPEC_NAME_LABEL
from miles.utils.workers.worker_spec import HostAndPort

NAMESPACE = "rl"
SELECTOR = "app.kubernetes.io/instance=r"


@dataclass
class FakePodApi:
    pods: list = field(default_factory=list)
    events: list = field(default_factory=list)
    resource_version: str = "1"
    event_delay: float = 0.0

    async def list_pods(self, *, namespace, label_selector):
        return PodListPage(pods=list(self.pods), resource_version=self.resource_version)

    async def stream_pods(self, *, namespace, label_selector, resource_version, timeout_seconds):
        await asyncio.sleep(self.event_delay)
        for event in self.events:
            yield event
        await asyncio.sleep(3600)


def _provider(api, static_addrs=None, worker_ports=None, **kwargs):
    return K8sWorkerProvider(
        namespace=NAMESPACE,
        label_selector=SELECTOR,
        static_addrs=static_addrs or {},
        worker_ports=worker_ports or {"engine": {"primary": 8000}},
        kube_client_factory=lambda: api,
        resync_period=None,
        **kwargs,
    )


def _cell_info(provider, cell_id="engine-0"):
    async def scenario():
        stop = await _watch(provider, [], ["engine"])
        try:
            return provider.cell_info(cell_id)
        finally:
            await stop()

    return asyncio.run(scenario())


async def _watch(provider, reported, spec_names):
    async def reconcile(cell_id, info):
        reported.append((cell_id, info))

    return await provider.watch_cells(reconcile, spec_names=spec_names)


def _relabelled(spec_name):
    pod = make_pod(name="engine-0-0", labels={SPEC_NAME_LABEL: spec_name})
    return PodWatchEvent(type="MODIFIED", obj=pod, resource_version="2", rejects_cursor=False)


async def _run_watch(provider, reported, spec_names, *, fail_when_gone=False, fail_when_alive=False):
    async def reconcile(cell_id, info):
        reported.append((cell_id, info))
        if (info is None and fail_when_gone) or (info is not None and fail_when_alive):
            raise RuntimeError("the consumer could not take this update")

    stop = await provider.watch_cells(reconcile, spec_names=spec_names)
    await asyncio.sleep(0.1)
    await stop()


class TestGetAddrs:
    def test_answers_a_static_worker_from_the_address_book(self):
        """A router's address is rendered by the launcher, so it is known before any pod exists."""
        addrs = {"inference-router-0": {"primary": HostAndPort(host="router.rl.svc", port=30000)}}
        provider = _provider(FakePodApi(), static_addrs=addrs)

        assert asyncio.run(provider.get_addr("inference-router-0")).port == 30000

    def test_answers_a_cell_member_from_its_pod(self):
        """A cell's members only exist once scheduled, so their addresses come from observation."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.1.2.3")])
        provider = _provider(api)

        async def scenario():
            stop = await _watch(provider, [], ["engine"])
            try:
                return await provider.get_addr("engine-0-0")
            finally:
                await stop()

        assert asyncio.run(scenario()) == HostAndPort(host="10.1.2.3", port=8000)

    def test_refuses_a_worker_it_has_never_seen(self):
        """Returning a guess would send traffic to whatever happens to answer at that address."""
        provider = _provider(FakePodApi())

        async def scenario():
            stop = await _watch(provider, [], ["engine"])
            try:
                await provider.get_addr("engine-9-9")
            finally:
                await stop()

        with pytest.raises(AssertionError, match="neither a static worker"):
            asyncio.run(scenario())


class TestWatchCells:
    def test_reports_the_cells_that_already_existed(self):
        """A controller starting against a running release must not think the cluster is empty."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0"), make_pod(name="engine-1-0", cell_index="1")])
        provider = _provider(api)
        reported = []

        async def scenario():
            stop = await _watch(provider, reported, ["engine"])
            await asyncio.sleep(0.05)
            await stop()

        asyncio.run(scenario())

        assert sorted(cell_id for cell_id, _ in reported) == ["engine-0", "engine-1"]

    def test_finishes_the_initial_listing_before_returning(self):
        """Otherwise a caller reading the cells right after cannot tell "not listed yet" from "not there"."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api)

        async def scenario():
            stop = await _watch(provider, [], ["engine"])
            try:
                return provider.cell_ids()
            finally:
                await stop()

        assert asyncio.run(scenario()) == ["engine-0"]

    def test_hides_a_cell_of_another_controller(self):
        """Several controllers share a namespace, and each must see only the specs it was given."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0"), make_pod(name="trainer-0-0", fleet="trainer")])
        provider = _provider(api)
        reported = []

        async def scenario():
            stop = await _watch(provider, reported, ["engine"])
            await asyncio.sleep(0.05)
            await stop()

        asyncio.run(scenario())

        assert [cell_id for cell_id, info in reported if info is not None] == ["engine-0"]

    def test_says_nothing_about_a_cell_that_was_never_alive(self):
        """The Ray provider reports only cells that came up; a cell still starting is not news."""
        api = FakePodApi(
            pods=[make_pod(name="engine-0-0"), make_pod(name="engine-0-1", worker_index="1", ready=False)]
        )
        provider = _provider(api)
        reported = []

        async def scenario():
            stop = await _watch(provider, reported, ["engine"])
            await asyncio.sleep(0.05)
            await stop()

        asyncio.run(scenario())

        assert reported == []

    def test_says_nothing_about_another_controller_cell_disappearing(self):
        """Its last pod is gone, so nothing says whose cell it was - and it was never this view's."""
        api = FakePodApi(pods=[make_pod(name="trainer-0-0", fleet="trainer")])
        provider = _provider(api)
        reported = []

        async def scenario():
            stop = await _watch(provider, reported, ["engine"])
            await asyncio.sleep(0.05)
            await stop()

        asyncio.run(scenario())

        assert reported == []

    def test_ignores_a_pod_that_is_not_a_worker(self):
        """A namespace holds other pods, and one of them must not become a phantom cell."""
        from tests.fast.utils.workers.worker_provider.test_k8s_labels import FakeMeta, FakePod

        api = FakePodApi(pods=[make_pod(name="engine-0-0"), FakePod(metadata=FakeMeta(name="prometheus-0"))])
        provider = _provider(api)

        async def scenario():
            stop = await _watch(provider, [], ["engine"])
            try:
                return provider.cell_ids()
            finally:
                await stop()

        assert asyncio.run(scenario()) == ["engine-0"]


class TestWatchCellsStateCommit:
    def test_a_reported_cell_leaving_the_wanted_set_is_reported_as_gone(self):
        """From this view the cell is gone, and only None says so; dropping it silently strands the consumer."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")], events=[_relabelled("someone-else")], event_delay=0.02)
        provider = _provider(api)
        reported = []

        asyncio.run(_run_watch(provider, reported, ["engine"]))

        assert [info is None for _, info in reported] == [False, True]
        assert [cell_id for cell_id, _ in reported] == ["engine-0", "engine-0"]

    def test_a_cell_that_was_never_reported_stays_silent_when_it_leaves_the_wanted_set(self):
        """It was never this view's cell, so announcing its removal would invent a cell the consumer never had."""
        api = FakePodApi(pods=[make_pod(name="trainer-0-0", fleet="trainer")])
        provider = _provider(api)
        reported = []

        asyncio.run(_run_watch(provider, reported, ["engine"]))

        assert reported == []

    def test_a_failing_gone_callback_keeps_the_cell_marked_as_reported(self):
        """Forgetting it before the callback lands turns the loop's retry into a second silent drop."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")], events=[_relabelled("someone-else")], event_delay=0.02)
        provider = _provider(api)

        asyncio.run(_run_watch(provider, [], ["engine"], fail_when_gone=True))

        assert provider._reported == {"engine-0"}

    def test_a_failing_alive_callback_does_not_count_as_reported(self):
        """The consumer never learnt of the cell, so a later removal must not be suppressed as old news."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api)

        asyncio.run(_run_watch(provider, [], ["engine"], fail_when_alive=True))

        assert provider._reported == set()


class TestCellInfo:
    def test_orders_the_workers_by_rank(self):
        """Consumers index this list by rank, so an arbitrary order would scramble the mapping."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-1", worker_index="1"),
                make_pod(name="engine-0-0", worker_index="0"),
            ]
        )
        provider = _provider(api)

        async def scenario():
            stop = await _watch(provider, [], ["engine"])
            try:
                return provider.cell_info("engine-0")
            finally:
                await stop()

        assert asyncio.run(scenario()).worker_names == ["engine-0-0", "engine-0-1"]

    def test_carries_the_meta_a_platform_annotated(self):
        """An engine's model id is a domain fact its consumers need and the pod is where it travels."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-model_id": "glm"})])
        provider = _provider(api)

        async def scenario():
            stop = await _watch(provider, [], ["engine"])
            try:
                return provider.cell_info("engine-0")
            finally:
                await stop()

        assert asyncio.run(scenario()).meta == {"model_id": "glm"}

    def test_names_every_rank_of_a_pod_that_serves_more_than_one(self):
        """Consumers index this list by rank, so a list of pods would hide every rank but the first of each."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0"), make_pod(name="engine-0-1", worker_index="1")])
        provider = _provider(api, ranks_per_pod={"engine": 2})

        assert _cell_info(provider).worker_names == ["engine-0-0", "engine-0-1", "engine-0-2", "engine-0-3"]

    def test_still_reports_the_pods_themselves_for_the_operations_that_delete_them(self):
        """Healing recreates pods, and a rank name is not something kubernetes could delete."""
        api = FakePodApi(pods=[make_pod(name="engine-0-1", worker_index="1"), make_pod(name="engine-0-0")])
        provider = _provider(api, ranks_per_pod={"engine": 2})

        async def scenario():
            stop = await _watch(provider, [], ["engine"])
            try:
                return provider.pod_names("engine-0")
            finally:
                await stop()

        assert asyncio.run(scenario()) == ["engine-0-0", "engine-0-1"]

    def test_is_absent_for_a_cell_with_no_pods(self):
        """A cell that was deleted must read as gone rather than as an empty cell."""
        provider = _provider(FakePodApi())

        async def scenario():
            stop = await _watch(provider, [], ["engine"])
            try:
                return provider.cell_info("engine-7")
            finally:
                await stop()

        assert asyncio.run(scenario()) is None


def _spec_meta(context) -> dict:
    return {"role": "actor", "cell_index": context.cell_index, "needs_offload": False, "model_id": "glm"}


class TestSpecMeta:
    def test_evaluates_the_meta_of_the_spec_for_every_cell_that_was_observed(self):
        """Cells of one fleet differ by their index alone, so one evaluation would collapse them into one."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0"), make_pod(name="engine-1-0", cell_index="1")])
        provider = _provider(api, spec_metas={"engine": _spec_meta})

        first = _cell_info(provider, cell_id="engine-0")
        second = _cell_info(provider, cell_id="engine-1")

        assert (first.meta["cell_index"], second.meta["cell_index"]) == (0, 1)
        assert (first.meta["role"], second.meta["role"]) == ("actor", "actor")

    def test_keeps_the_python_types_the_spec_computed(self):
        """A chart can only carry strings, which is why this meta is computed here rather than rendered."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api, spec_metas={"engine": _spec_meta})

        meta = _cell_info(provider).meta

        assert isinstance(meta["cell_index"], int)
        assert meta["needs_offload"] is False

    def test_reports_nothing_of_its_own_for_a_spec_that_declares_no_meta(self):
        """Most specs have no facts to add, and an invented key would look like a fact to a consumer."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])

        assert _cell_info(_provider(api)).meta == {}

    def test_lets_a_pod_annotation_override_a_key_the_spec_also_computed(self):
        """The pod is what a platform actually created, so its own account of itself wins."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-model_id": "qwen"})])
        provider = _provider(api, spec_metas={"engine": _spec_meta})

        assert _cell_info(provider).meta["model_id"] == "qwen"

    def test_refuses_a_cell_whose_pods_annotate_one_key_differently(self):
        """A cell reports one value per key, and picking one silently would depend on the store's order."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-model_id": "glm"}),
                make_pod(name="engine-0-1", worker_index="1", annotations={"miles.radixark.io/meta-model_id": "qwen"}),
            ]
        )

        with pytest.raises(AssertionError, match="annotates 'model_id'"):
            _cell_info(_provider(api))

    def test_accepts_pods_that_agree_about_a_key(self):
        """Every pod of a fleet carries the same values entry, so agreement is the normal case."""
        annotations = {"miles.radixark.io/meta-model_id": "glm"}
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-0", annotations=annotations),
                make_pod(name="engine-0-1", worker_index="1", annotations=annotations),
            ]
        )

        assert _cell_info(_provider(api)).meta == {"model_id": "glm"}


class _FakeTrainWorker:
    def init(self, rank: int) -> int:
        return rank

    def kill_self(self) -> None:
        return None


def _trainer_provider(api, **kwargs):
    return K8sWorkerProvider(
        namespace=NAMESPACE,
        label_selector=SELECTOR,
        static_addrs={},
        worker_ports={"engine": {"master": 9000, "rpc": 8000}},
        worker_classes={"engine": f"{__name__}._FakeTrainWorker"},
        kube_client_factory=lambda: api,
        resync_period=None,
        **kwargs,
    )


def _worker_infos(provider, cell_id="engine-0"):
    async def scenario():
        stop = await _watch(provider, [], ["engine"])
        try:
            return provider.get_worker_infos_of(cell_ids=[cell_id])[0]
        finally:
            await stop()

    return asyncio.run(scenario())


class TestGetWorkerInfos:
    def test_orders_the_workers_by_the_rank_label(self):
        """A trainer cell reads rank 0 as its master, so an arbitrary pod order would misconfigure it."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-2", worker_index="2", pod_ip="10.0.0.3"),
                make_pod(name="engine-0-0", worker_index="0", pod_ip="10.0.0.1"),
                make_pod(name="engine-0-1", worker_index="1", pod_ip="10.0.0.2"),
            ]
        )

        infos = _worker_infos(_trainer_provider(api))

        assert [info.name for info in infos] == ["engine-0-0", "engine-0-1", "engine-0-2"]

    def test_addresses_a_worker_at_its_pod_ip_on_the_spec_ports(self):
        """Every rank has its own network namespace, so each publishes the spec's ports at its own ip."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.1.2.3")])

        infos = _worker_infos(_trainer_provider(api))

        assert infos[0].self_addrs == {
            "master": HostAndPort(host="10.1.2.3", port=9000),
            "rpc": HostAndPort(host="10.1.2.3", port=8000),
        }

    def test_falls_back_to_the_headless_service_name_of_a_pod_without_an_ip(self):
        """A pod's ip appears late, but its dns name is stable from the moment the workload names it."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip=None, subdomain="engine")])

        infos = _worker_infos(_trainer_provider(api))

        assert infos[0].self_addrs["rpc"].host == f"engine-0-0.engine.{NAMESPACE}.svc"

    def test_hands_out_an_rpc_handle_pointed_at_the_worker(self):
        """A trainer cell drives its ranks through these handles, so they must talk to the right pod."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.1.2.3")])

        handle = _worker_infos(_trainer_provider(api))[0].handle

        assert isinstance(handle, RpcWorkerHandle)
        assert handle._transport._server_url == "http://10.1.2.3:8000"

    def test_the_handle_knows_the_methods_of_the_worker_class(self):
        """A typo would otherwise become a 404 at call time, deep inside a training step."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])

        handle = _worker_infos(_trainer_provider(api))[0].handle

        assert callable(handle.init)
        with pytest.raises(AttributeError):
            handle.__getattr__("nonexistent_method")

    def test_reports_the_gpus_a_platform_annotated(self):
        """Colocation is verified against these ids, so they travel with the worker rather than beside it."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-gpu_ids": "2,3"})])

        assert _worker_infos(_trainer_provider(api))[0].gpu_ids == [2, 3]

    def test_counts_a_pod_restart_as_a_new_worker_generation(self):
        """A restarted pod kept its name and lost its memory, and a consumer must be able to tell."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", restarts=2)])

        assert _worker_infos(_trainer_provider(api))[0].generation == 2

    def test_refuses_a_cell_that_is_missing_a_pod(self):
        """Driving half a cell would let the missing ranks' collective hang the whole run."""
        api = FakePodApi(pods=[make_pod(name="engine-0-1", worker_index="1")])

        with pytest.raises(AssertionError, match="missing pods"):
            _worker_infos(_trainer_provider(api))

    def test_refuses_a_cell_it_has_never_observed(self):
        """Returning no workers would read as a cell with nothing to do rather than as an error."""
        with pytest.raises(AssertionError, match="no observed worker pods"):
            _worker_infos(_trainer_provider(FakePodApi()), cell_id="engine-9")

    def test_refuses_a_spec_whose_worker_class_is_unknown(self):
        """Without the class the handle cannot type its calls, and a typo would reach the wire."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = K8sWorkerProvider(
            namespace=NAMESPACE,
            label_selector=SELECTOR,
            static_addrs={},
            worker_ports={"engine": {"rpc": 8000}},
            kube_client_factory=lambda: api,
            resync_period=None,
        )

        with pytest.raises(AssertionError, match="has no worker class"):
            _worker_infos(provider)

    def test_fans_a_pod_out_into_one_worker_per_rank_it_serves(self):
        """A supervised pod runs one worker process per rank, and each of them has to be driven separately."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.0.0.1")])

        infos = _worker_infos(_trainer_provider(api, ranks_per_pod={"engine": 3}))

        assert [info.name for info in infos] == ["engine-0-0", "engine-0-1", "engine-0-2"]

    def test_offsets_the_rpc_port_of_each_rank_the_way_the_process_binds_it(self):
        """serve_inner listens on port + rank_in_pod, so any other guess reaches the wrong rank or nothing."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.0.0.1")])

        infos = _worker_infos(_trainer_provider(api, ranks_per_pod={"engine": 2}))

        assert [info.self_addrs["rpc"].port for info in infos] == [8000, 8001]
        assert [info.self_addrs["master"].port for info in infos] == [9000, 9000]

    def test_numbers_the_ranks_of_the_second_pod_after_those_of_the_first(self):
        """A rank's name is its index in the cell, which spans the pods rather than restarting in each."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-1", worker_index="1", pod_ip="10.0.0.2"),
                make_pod(name="engine-0-0", worker_index="0", pod_ip="10.0.0.1"),
            ]
        )

        infos = _worker_infos(_trainer_provider(api, ranks_per_pod={"engine": 2}))

        assert [info.name for info in infos] == ["engine-0-0", "engine-0-1", "engine-0-2", "engine-0-3"]
        assert [info.self_addrs["rpc"].host for info in infos] == ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.2"]

    def test_gives_each_rank_its_own_share_of_the_gpus_of_the_pod(self):
        """A rank takes the gpu slots at its own offset, exactly as the pod's own process numbering does."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-gpu_ids": "0,1,2,3"})])

        infos = _worker_infos(_trainer_provider(api, ranks_per_pod={"engine": 2}))

        assert [info.gpu_ids for info in infos] == [[0, 1], [2, 3]]

    def test_refuses_a_pod_whose_gpus_do_not_divide_among_its_ranks(self):
        """Handing a rank a partial share would silently point two ranks at one gpu."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-gpu_ids": "0,1,2"})])

        with pytest.raises(AssertionError, match="equal share"):
            _worker_infos(_trainer_provider(api, ranks_per_pod={"engine": 2}))

    def test_resolves_the_address_of_a_rank_that_no_pod_is_named_after(self):
        """Consumers hold the rank names this provider handed out, so it has to answer for them."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.0.0.1")])
        provider = _trainer_provider(api, ranks_per_pod={"engine": 2})

        async def scenario():
            stop = await _watch(provider, [], ["engine"])
            try:
                return await provider.get_addrs("engine-0-1")
            finally:
                await stop()

        assert asyncio.run(scenario())["rpc"] == HostAndPort(host="10.0.0.1", port=8001)

    def test_answers_for_several_cells_at_once(self):
        """A controller resynchronises every cell it owns, and one round trip per cell is the shape."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-0", cell_index="0"),
                make_pod(name="engine-1-0", cell_index="1"),
            ]
        )
        provider = _trainer_provider(api)

        async def scenario():
            stop = await _watch(provider, [], ["engine"])
            try:
                return provider.get_worker_infos_of(cell_ids=["engine-0", "engine-1"])
            finally:
                await stop()

        assert [[info.name for info in infos] for infos in asyncio.run(scenario())] == [
            ["engine-0-0"],
            ["engine-1-0"],
        ]
