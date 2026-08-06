"""``RolloutManager`` cell dispatch + ``EnginesAndLock`` flow driven through
the production ``RolloutManager`` class.

We instantiate ``RolloutManager.__ray_actor_class__`` directly (the raw Python
class behind ``@ray.remote``) — that keeps the manager in the test process so
``monkeypatch`` reaches its dependencies, while the engines it spawns are
still real Ray actors (mocks). Methods are ``async`` and called with ``await``.

Pure routing/flag-flip helpers without Ray content live in
``tests/fast/ray/rollout/test_rollout_manager.py``."""

from __future__ import annotations

import asyncio
import textwrap
import threading
import time
from contextlib import nullcontext

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout.rollout_manager import RolloutManager
from miles.rollout.base_types import (
    LeasedRolloutFnTrainOutput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnLifecycle,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
    TrainAdmissionHold,
    TrainBatchLease,
    TrainBatchRollbackReason,
)
from miles.rollout.checkpoint_eval import CheckpointEvalFn
from miles.utils.async_utils import get_async_loop


class RecordingTrainBatchLease(TrainBatchLease):
    def __init__(self, rollout_id: int, events: list[str]) -> None:
        super().__init__(rollout_id=rollout_id)
        self._events = events

    def _commit(self) -> None:
        self._events.append("commit")

    def _rollback(self, reason: TrainBatchRollbackReason) -> None:
        self._events.append(f"rollback:{reason.name}")


class RecordingTrainAdmissionHold(TrainAdmissionHold):
    def __init__(self, hold_index: int, lifecycle: RecordingRolloutLifecycle) -> None:
        super().__init__()
        self._hold_index = hold_index
        self._lifecycle = lifecycle

    async def _wait_terminal(self) -> None:
        self._lifecycle.wait_started.set()
        while not self._lifecycle.allow_wait.is_set():
            await asyncio.sleep(0)
        self._lifecycle.loops.append(asyncio.get_running_loop())
        self._lifecycle.events.append(f"wait:{self._hold_index}")

    def _record_weight_update(self) -> None:
        self._lifecycle.record_started.set()
        if not self._lifecycle.allow_record.wait(timeout=5):
            raise TimeoutError("Timed out waiting to record the train weight update.")
        self._lifecycle.loops.append(asyncio.get_running_loop())
        self._lifecycle.events.append(f"record:{self._hold_index}")
        if self._lifecycle.record_failure is not None:
            raise self._lifecycle.record_failure

    def _release(self) -> None:
        self._lifecycle.release_started.set()
        if not self._lifecycle.allow_release.wait(timeout=5):
            raise TimeoutError("Timed out waiting to release the train admission hold.")
        self._lifecycle.loops.append(asyncio.get_running_loop())
        self._lifecycle.active_holds.remove(self._hold_index)
        self._lifecycle.events.append(f"release:{self._hold_index}")


class RecordingRolloutLifecycle(RolloutFnLifecycle):
    def __init__(self, events: list[str], label: str | None = None) -> None:
        self.events = events
        self.label = label
        self.loops: list[asyncio.AbstractEventLoop] = []
        self.holds: list[RecordingTrainAdmissionHold] = []
        self.active_holds: set[int] = set()
        self.prepare_started = threading.Event()
        self.allow_prepare = threading.Event()
        self.allow_prepare.set()
        self.wait_started = threading.Event()
        self.allow_wait = threading.Event()
        self.allow_wait.set()
        self.record_started = threading.Event()
        self.allow_record = threading.Event()
        self.allow_record.set()
        self.record_failure: BaseException | None = None
        self.release_started = threading.Event()
        self.allow_release = threading.Event()
        self.allow_release.set()

    async def prepare_checkpoint(self, rollout_id: int) -> None:
        self.prepare_started.set()
        while not self.allow_prepare.is_set():
            await asyncio.sleep(0)
        self.loops.append(asyncio.get_running_loop())
        self.events.append(f"prepare:{rollout_id}")

    async def acquire_train_admission_hold(self) -> TrainAdmissionHold:
        self.loops.append(asyncio.get_running_loop())
        hold_index = len(self.holds)
        hold = RecordingTrainAdmissionHold(hold_index, self)
        self.holds.append(hold)
        self.active_holds.add(hold_index)
        self.events.append(f"acquire:{hold_index}")
        return hold

    async def close(self) -> None:
        self.loops.append(asyncio.get_running_loop())
        self.events.append("close" if self.label is None else f"close:{self.label}")


class BlockingAcquireRolloutLifecycle(RecordingRolloutLifecycle):
    def __init__(
        self,
        events: list[str],
        acquire_started: threading.Event,
        allow_acquire: threading.Event,
    ) -> None:
        super().__init__(events)
        self._acquire_started = acquire_started
        self._allow_acquire = allow_acquire

    async def acquire_train_admission_hold(self) -> TrainAdmissionHold:
        self.events.append("acquire_started")
        self._acquire_started.set()
        while not self._allow_acquire.is_set():
            await asyncio.sleep(0)
        self.events.append("acquire_finished")
        return await super().acquire_train_admission_hold()


class CloseDominatedAcquireRolloutLifecycle(BlockingAcquireRolloutLifecycle):
    def __init__(
        self,
        events: list[str],
        acquire_started: threading.Event,
        allow_acquire: threading.Event,
    ) -> None:
        super().__init__(events, acquire_started, allow_acquire)
        self._closed = False

    async def acquire_train_admission_hold(self) -> TrainAdmissionHold:
        hold = await super().acquire_train_admission_hold()
        if self._closed:
            self.active_holds.clear()
        return hold

    async def close(self) -> None:
        self._closed = True
        self.active_holds.clear()
        self.events.append("close")


class FailingOnceCloseRolloutLifecycle(RecordingRolloutLifecycle):
    def __init__(self, events: list[str], failure: BaseException) -> None:
        super().__init__(events)
        self._failure: BaseException | None = failure

    async def close(self) -> None:
        self.loops.append(asyncio.get_running_loop())
        if self._failure is not None:
            failure = self._failure
            self._failure = None
            self.events.append("close_failed")
            raise failure
        self.events.append("close_succeeded")


class BlockingCloseRolloutLifecycle(RecordingRolloutLifecycle):
    def __init__(
        self,
        events: list[str],
        close_started: threading.Event,
        allow_close: threading.Event,
    ) -> None:
        super().__init__(events)
        self._close_started = close_started
        self._allow_close = allow_close

    async def close(self) -> None:
        self.loops.append(asyncio.get_running_loop())
        self.events.append("close_started")
        self._close_started.set()
        while not self._allow_close.is_set():
            await asyncio.sleep(0)
        self.events.append("close_finished")


@pytest.fixture
def patch_low_level(monkeypatch):
    """Replace, in the test process:
    - ``SGLangEngine`` → ``MockSGLangEngine`` so created actors are mocks.
    - addr allocator → deterministic stub.
    - ``init_tracking`` / ``init_http_client`` / ``start_session_server`` /
      ``load_function`` / ``load_rollout_function`` → no-ops (the production
      defaults touch wandb / network / not-importable default function paths)."""
    import miles.ray.rollout.rollout_manager as rmgr
    import miles.ray.rollout.rollout_server as rsrv
    import miles.ray.rollout.server_group as sg
    from miles.ray.rollout.addr_allocator import PortCursors
    from miles.utils.test_utils.mock_sglang_engine import MockSGLangEngine

    monkeypatch.setattr(sg, "SGLangEngine", MockSGLangEngine.__ray_actor_class__)
    # multi-model tests would otherwise spawn a real router subprocess for
    # ``model_idx > 0`` (force_new=True bypasses the args.sglang_router_ip cache).
    monkeypatch.setattr(
        rsrv,
        "start_router",
        lambda args, **kw: (args.sglang_router_ip, args.sglang_router_port),
    )

    def _fake_alloc(*args, **kwargs):
        engines = kwargs["rollout_engines"]
        return (
            {
                rank: dict(
                    host="127.0.0.1",
                    port=30000 + rank,
                    nccl_port=31000 + rank,
                    engine_info_bootstrap_port=32000 + rank,
                    dist_init_addr=f"127.0.0.1:{33000 + rank}",
                )
                for rank, _ in engines
            },
            PortCursors(_values={0: 34000}),
        )

    monkeypatch.setattr(sg, "allocate_rollout_engine_addr_and_ports_normal", _fake_alloc)
    monkeypatch.setattr(rmgr, "init_tracking", lambda *a, **kw: None)
    monkeypatch.setattr(rmgr, "init_http_client", lambda args: None)
    monkeypatch.setattr(rmgr, "start_session_server", lambda args: None)
    monkeypatch.setattr(rmgr, "load_function", lambda path: lambda *a, **kw: None)
    monkeypatch.setattr(rmgr, "load_rollout_function", lambda input, path: lambda *a, **kw: None)
    # generate()/eval() drive these — production hits wandb / tensorboard.
    monkeypatch.setattr(rmgr, "log_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rmgr, "log_eval_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rmgr, "save_debug_rollout_data", lambda *a, **kw: None)


def _make_manager(args, pg):
    return RolloutManager.__ray_actor_class__(args, pg)


def _install_train_rollout_lifecycle(manager, lifecycle: RolloutFnLifecycle) -> None:
    manager._train_rollout_lifecycle = lifecycle
    manager._lifecycle_async_loop = get_async_loop()


def _install_rollout_lifecycles(manager, *lifecycles: RolloutFnLifecycle) -> None:
    manager._rollout_lifecycles = lifecycles
    manager._closed_rollout_lifecycles = []
    manager._lifecycle_async_loop = get_async_loop()


@pytest.fixture
def lifecycle_manager(placement_group_factory, tmp_path, patch_low_level):
    args = _make_test_args(tmp_path, models=[("actor", True)])
    args.debug_train_only = True
    return _make_manager(args, placement_group_factory(2))


def _write_sglang_config(tmp_path, *, models: list[tuple[str, bool]]) -> str:
    """Write a multi-model sglang yaml — each entry ``(name, update_weights)``.
    Each model gets one regular group with 2 engines × 1 GPU = 2 GPUs. With N
    models, total GPUs = 2N; ``args.rollout_num_gpus`` must match."""
    lines = ["sglang:"]
    for name, update_weights in models:
        lines.extend(
            [
                f"  - name: {name}",
                f"    update_weights: {str(update_weights).lower()}",
                "    server_groups:",
                "      - worker_type: regular",
                "        num_gpus: 2",
                "        num_gpus_per_engine: 1",
            ]
        )
    cfg_path = tmp_path / "sglang.yaml"
    cfg_path.write_text(textwrap.dedent("\n".join(lines)) + "\n")
    return str(cfg_path)


def _make_test_args(tmp_path, *, models: list[tuple[str, bool]]):
    """Build args that drive ``RolloutManager.__init__`` →
    ``start_rollout_servers`` → N model servers each with 1 group of 2 mock
    engines."""
    cfg = _write_sglang_config(tmp_path, models=models)
    rollout_num_gpus = 2 * len(models)
    return make_args(
        sglang_config=cfg,
        rollout_num_gpus=rollout_num_gpus,
        # short-circuit start_router (returns early when ip+port already set)
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        # disable everything else that would spawn subprocesses or hit network
        use_session_server=False,
        use_fault_tolerance=False,
        use_wandb=False,
        use_tensorboard=False,
        use_mlflow=False,
        use_distributed_post=False,
        sglang_server_concurrency=1,
    )


async def _assert_engine_dies(actor_handle, *, deadline_s: float = 15.0, poll_interval_s: float = 0.2) -> None:
    deadline = time.monotonic() + deadline_s
    while True:
        try:
            ray.get(actor_handle.health_generate.remote(timeout=1.0), timeout=5.0)
        except (ray.exceptions.RayActorError, ray.exceptions.RayTaskError):
            return
        except ray.exceptions.GetTimeoutError:
            pass
        if time.monotonic() >= deadline:
            pytest.fail(f"engine actor still alive {deadline_s}s after stop_cell")
        await asyncio.sleep(poll_interval_s)


@pytest.mark.asyncio
class TestRolloutManagerInit:
    async def test_init_creates_live_mock_engines_via_real_start_rollout_servers(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """End-to-end smoke: production ``__init__`` + ``start_rollout_servers``
        runs against MockSGLangEngine; resulting engines are reachable as Ray
        actor handles via the public ``get_updatable_engines_and_lock``."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        assert len(eal.rollout_engines) == 2
        for h in eal.rollout_engines:
            assert isinstance(h, ray.actor.ActorHandle)
            assert ray.get(h.health_generate.remote(timeout=1.0)) is True


@pytest.mark.asyncio
class TestRolloutLifecycle:
    async def test_concurrent_lifecycle_calls_retain_one_event_loop(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr
        import miles.utils.async_utils as async_utils

        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.debug_train_only = True
        args.eval_function_path = args.rollout_function_path
        pg = placement_group_factory(2)
        lifecycle = RecordingRolloutLifecycle([])
        monkeypatch.setattr(rmgr, "load_rollout_function", lambda input, path: lifecycle)
        original_async_loop_thread = async_utils.AsyncLoopThread
        created_loops: list[async_utils.AsyncLoopThread] = []

        def create_async_loop() -> async_utils.AsyncLoopThread:
            async_loop = original_async_loop_thread()
            created_loops.append(async_loop)
            return async_loop

        monkeypatch.setattr(rmgr, "get_async_loop", create_async_loop)
        monkeypatch.setattr(async_utils, "get_async_loop", create_async_loop)
        manager = _make_manager(args, pg)

        try:
            hold_ids = await asyncio.gather(
                manager.acquire_train_admission_hold(),
                manager.acquire_train_admission_hold(),
            )
            for hold_id in hold_ids:
                await manager.release_train_admission_hold(hold_id)

            assert len(created_loops) == 1
            assert len({id(loop) for loop in lifecycle.loops}) == 1
        finally:
            for async_loop in created_loops:
                async_loop.loop.call_soon_threadsafe(async_loop.loop.stop)
                async_loop._thread.join(timeout=1)

    async def test_init_deduplicates_shared_lifecycle_by_identity(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.debug_train_only = True
        args.eval_function_path = args.rollout_function_path
        pg = placement_group_factory(2)
        lifecycle = RecordingRolloutLifecycle([])
        monkeypatch.setattr(rmgr, "load_rollout_function", lambda input, path: lifecycle)

        manager = _make_manager(args, pg)

        assert manager._train_rollout_lifecycle is lifecycle
        assert manager._rollout_lifecycles == (lifecycle,)

    async def test_lifecycle_calls_do_not_consume_manager_default_executor(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        events: list[str] = []
        lifecycle = RecordingRolloutLifecycle(events)
        _install_train_rollout_lifecycle(manager, lifecycle)
        _install_rollout_lifecycles(manager, lifecycle)
        monkeypatch.setattr(
            rmgr.event_analyzer,
            "run_analysis_from_args",
            lambda args: events.append("analysis"),
        )

        async def reject_to_thread(*args, **kwargs):
            raise AssertionError("lifecycle calls must not consume the manager default executor")

        monkeypatch.setattr(asyncio, "to_thread", reject_to_thread)

        hold_id = await manager.acquire_train_admission_hold()
        await manager.wait_train_admission_hold(hold_id)
        await manager.release_train_admission_hold(hold_id)
        await manager.dispose()

        assert events == ["acquire:0", "wait:0", "release:0", "close", "analysis"]

    async def test_dispose_rejects_hold_acquired_concurrently_with_lifecycle_close(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        events: list[str] = []
        acquire_started = threading.Event()
        allow_acquire = threading.Event()
        lifecycle = CloseDominatedAcquireRolloutLifecycle(events, acquire_started, allow_acquire)
        _install_train_rollout_lifecycle(manager, lifecycle)
        _install_rollout_lifecycles(manager, lifecycle)
        monkeypatch.setattr(
            rmgr.event_analyzer,
            "run_analysis_from_args",
            lambda args: events.append("analysis"),
        )
        acquire_task = asyncio.create_task(manager.acquire_train_admission_hold())
        assert await asyncio.to_thread(acquire_started.wait, 5)

        await manager.dispose()
        allow_acquire.set()

        with pytest.raises(RuntimeError) as acquisition_error:
            await acquire_task

        assert str(acquisition_error.value) == "Rollout manager lifecycle is closing."
        assert manager._train_admission_holds == {}
        assert lifecycle.active_holds == set()
        assert events == ["acquire_started", "close", "analysis", "acquire_finished", "acquire:0"]

    async def test_ordinary_close_bearing_rollout_remains_legacy_compatible(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.debug_train_only = True
        args.eval_function_path = args.rollout_function_path
        pg = placement_group_factory(2)
        events: list[str] = []

        class OrdinaryRollout:
            def __call__(self, input):
                raise AssertionError("ordinary rollout invocation is outside this test")

            def close(self) -> None:
                events.append("ordinary_close")

        ordinary_rollout = OrdinaryRollout()
        monkeypatch.setattr(rmgr, "load_rollout_function", lambda input, path: ordinary_rollout)
        manager = _make_manager(args, pg)

        assert manager._train_rollout_lifecycle is None
        assert manager._rollout_lifecycles == ()
        assert await manager.acquire_train_admission_hold() is None
        await manager.wait_train_admission_hold(None)
        await manager.record_train_weight_update(None)
        await manager.release_train_admission_hold(None)
        await manager.dispose()

        assert events == []

    async def test_save_prepares_lifecycle_before_checkpoint_publication(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        manager.args.rollout_global_dataset = True
        events: list[str] = []
        lifecycle = RecordingRolloutLifecycle(events)
        _install_train_rollout_lifecycle(manager, lifecycle)

        class RecordingDataSource:
            def save(self, rollout_id: int) -> None:
                assert lifecycle.active_holds == {0}
                events.append(f"source:{rollout_id}")

        manager.data_source = RecordingDataSource()

        def snapshot(args, rollout_id: int) -> None:
            assert lifecycle.active_holds == {0}
            events.append(f"event:{rollout_id}")

        monkeypatch.setattr(
            rmgr.event_logger_checkpoint,
            "snapshot",
            snapshot,
        )

        await manager.save(rollout_id=13)

        assert events == ["acquire:0", "prepare:13", "source:13", "event:13", "release:0"]
        assert lifecycle.active_holds == set()

    async def test_checkpoint_preparation_failure_prevents_publication(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        manager.args.rollout_global_dataset = True
        events: list[str] = []
        failure = RuntimeError("checkpoint preparation failed")

        class FailingPrepareLifecycle(RecordingRolloutLifecycle):
            async def prepare_checkpoint(self, rollout_id: int) -> None:
                events.append(f"prepare:{rollout_id}")
                raise failure

        class RecordingDataSource:
            def save(self, rollout_id: int) -> None:
                events.append(f"source:{rollout_id}")

        lifecycle = FailingPrepareLifecycle(events)
        _install_train_rollout_lifecycle(manager, lifecycle)
        manager.data_source = RecordingDataSource()
        monkeypatch.setattr(
            rmgr.event_logger_checkpoint,
            "snapshot",
            lambda args, rollout_id: events.append(f"event:{rollout_id}"),
        )

        with pytest.raises(RuntimeError) as error:
            await manager.save(rollout_id=14)

        assert error.value is failure
        assert events == ["acquire:0", "prepare:14", "release:0"]
        assert lifecycle.active_holds == set()

    @pytest.mark.parametrize("failure_point", ["source", "event"])
    async def test_checkpoint_publication_failure_releases_hold_and_skips_later_publication(
        self,
        lifecycle_manager,
        monkeypatch,
        failure_point: str,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        manager.args.rollout_global_dataset = True
        events: list[str] = []
        failure = RuntimeError(f"{failure_point} publication failed")
        lifecycle = RecordingRolloutLifecycle(events)
        _install_train_rollout_lifecycle(manager, lifecycle)

        class RecordingDataSource:
            def save(self, rollout_id: int) -> None:
                events.append(f"source:{rollout_id}")
                if failure_point == "source":
                    raise failure

        def snapshot(args, rollout_id: int) -> None:
            events.append(f"event:{rollout_id}")
            if failure_point == "event":
                raise failure

        manager.data_source = RecordingDataSource()
        monkeypatch.setattr(rmgr.event_logger_checkpoint, "snapshot", snapshot)

        with pytest.raises(RuntimeError) as error:
            await manager.save(rollout_id=16)

        assert error.value is failure
        expected_publication = ["source:16"]
        if failure_point == "event":
            expected_publication.append("event:16")
        assert events == ["acquire:0", "prepare:16", *expected_publication, "release:0"]
        assert lifecycle.active_holds == set()

    async def test_cancelled_checkpoint_hold_acquisition_settles_without_publication(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        manager.args.rollout_global_dataset = True
        events: list[str] = []
        acquire_started = threading.Event()
        allow_acquire = threading.Event()
        lifecycle = BlockingAcquireRolloutLifecycle(events, acquire_started, allow_acquire)
        _install_train_rollout_lifecycle(manager, lifecycle)

        class RecordingDataSource:
            def save(self, rollout_id: int) -> None:
                events.append(f"source:{rollout_id}")

        manager.data_source = RecordingDataSource()
        monkeypatch.setattr(
            rmgr.event_logger_checkpoint,
            "snapshot",
            lambda args, rollout_id: events.append(f"event:{rollout_id}"),
        )
        save_task = asyncio.create_task(manager.save(rollout_id=15))
        assert await asyncio.to_thread(acquire_started.wait, 5)

        try:
            save_task.cancel()
            await asyncio.sleep(0)
            assert save_task.done() is False
            allow_acquire.set()
            with pytest.raises(asyncio.CancelledError):
                await save_task
        finally:
            allow_acquire.set()
            await asyncio.gather(save_task, return_exceptions=True)

        assert events == ["acquire_started", "acquire_finished", "acquire:0", "release:0"]
        assert lifecycle.active_holds == set()

    async def test_cancelled_checkpoint_preparation_settles_without_publication(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        manager.args.rollout_global_dataset = True
        events: list[str] = []
        lifecycle = RecordingRolloutLifecycle(events)
        lifecycle.allow_prepare.clear()
        _install_train_rollout_lifecycle(manager, lifecycle)

        class RecordingDataSource:
            def save(self, rollout_id: int) -> None:
                events.append(f"source:{rollout_id}")

        manager.data_source = RecordingDataSource()
        monkeypatch.setattr(
            rmgr.event_logger_checkpoint,
            "snapshot",
            lambda args, rollout_id: events.append(f"event:{rollout_id}"),
        )
        save_task = asyncio.create_task(manager.save(rollout_id=15))
        assert await asyncio.to_thread(lifecycle.prepare_started.wait, 5)

        try:
            save_task.cancel()
            await asyncio.sleep(0)
            assert save_task.done() is False
            lifecycle.allow_prepare.set()
            with pytest.raises(asyncio.CancelledError):
                await save_task
        finally:
            lifecycle.allow_prepare.set()
            await asyncio.gather(save_task, return_exceptions=True)

        assert events == ["acquire:0", "prepare:15", "release:0"]
        assert lifecycle.active_holds == set()

    async def test_cancelled_checkpoint_release_settles_before_propagating(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        manager.args.rollout_global_dataset = True
        events: list[str] = []
        lifecycle = RecordingRolloutLifecycle(events)
        lifecycle.allow_release.clear()
        _install_train_rollout_lifecycle(manager, lifecycle)

        class RecordingDataSource:
            def save(self, rollout_id: int) -> None:
                events.append(f"source:{rollout_id}")

        manager.data_source = RecordingDataSource()
        monkeypatch.setattr(
            rmgr.event_logger_checkpoint,
            "snapshot",
            lambda args, rollout_id: events.append(f"event:{rollout_id}"),
        )
        save_task = asyncio.create_task(manager.save(rollout_id=17))
        assert await asyncio.to_thread(lifecycle.release_started.wait, 5)

        try:
            save_task.cancel()
            await asyncio.sleep(0)
            assert save_task.done() is False
            lifecycle.allow_release.set()
            with pytest.raises(asyncio.CancelledError):
                await save_task
        finally:
            lifecycle.allow_release.set()
            await asyncio.gather(save_task, return_exceptions=True)

        assert events == ["acquire:0", "prepare:17", "source:17", "event:17", "release:0"]
        assert lifecycle.active_holds == set()

    async def test_checkpoint_release_failure_surfaces(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        manager.args.rollout_global_dataset = False
        events: list[str] = []
        release_failure = RuntimeError("hold release failed")

        class FailingReleaseHold(TrainAdmissionHold):
            async def _wait_terminal(self) -> None:
                raise AssertionError("checkpoint save must not wait for terminal work")

            def _record_weight_update(self) -> None:
                raise AssertionError("checkpoint save must not record a weight update")

            def _release(self) -> None:
                events.append("release")
                raise release_failure

        class FailingReleaseLifecycle(RecordingRolloutLifecycle):
            async def acquire_train_admission_hold(self) -> TrainAdmissionHold:
                events.append("acquire")
                return FailingReleaseHold()

        _install_train_rollout_lifecycle(manager, FailingReleaseLifecycle(events))
        monkeypatch.setattr(
            rmgr.event_logger_checkpoint,
            "snapshot",
            lambda args, rollout_id: events.append(f"event:{rollout_id}"),
        )

        with pytest.raises(RuntimeError) as error:
            await manager.save(rollout_id=18)

        assert error.value is release_failure
        assert events == ["acquire", "prepare:18", "event:18", "release"]

    async def test_checkpoint_publication_failure_remains_primary_when_release_fails(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        manager.args.rollout_global_dataset = False
        events: list[str] = []
        publication_failure = RuntimeError("event publication failed")
        release_failure = RuntimeError("hold release failed")

        class FailingReleaseHold(TrainAdmissionHold):
            async def _wait_terminal(self) -> None:
                raise AssertionError("checkpoint save must not wait for terminal work")

            def _record_weight_update(self) -> None:
                raise AssertionError("checkpoint save must not record a weight update")

            def _release(self) -> None:
                events.append("release")
                raise release_failure

        class FailingReleaseLifecycle(RecordingRolloutLifecycle):
            async def acquire_train_admission_hold(self) -> TrainAdmissionHold:
                events.append("acquire")
                return FailingReleaseHold()

        def fail_snapshot(args, rollout_id: int) -> None:
            events.append(f"event:{rollout_id}")
            raise publication_failure

        _install_train_rollout_lifecycle(manager, FailingReleaseLifecycle(events))
        monkeypatch.setattr(rmgr.event_logger_checkpoint, "snapshot", fail_snapshot)

        with pytest.raises(RuntimeError) as error:
            await manager.save(rollout_id=19)

        assert error.value is publication_failure
        assert error.value.__cause__ is release_failure
        assert events == ["acquire", "prepare:19", "event:19", "release"]

    async def test_manager_owns_exact_admission_holds_behind_opaque_ids(self, lifecycle_manager):
        manager = lifecycle_manager
        events: list[str] = []
        lifecycle = RecordingRolloutLifecycle(events)
        _install_train_rollout_lifecycle(manager, lifecycle)
        manager_loop = asyncio.get_running_loop()

        first_hold_id = await manager.acquire_train_admission_hold()
        second_hold_id = await manager.acquire_train_admission_hold()
        await manager.wait_train_admission_hold(first_hold_id)
        await manager.record_train_weight_update(first_hold_id)
        await manager.release_train_admission_hold(first_hold_id)
        with pytest.raises(RuntimeError) as duplicate_release:
            await manager.release_train_admission_hold(first_hold_id)
        await manager.wait_train_admission_hold(second_hold_id)
        await manager.record_train_weight_update(second_hold_id)
        await manager.release_train_admission_hold(second_hold_id)

        assert (first_hold_id, second_hold_id) == (0, 1)
        assert str(duplicate_release.value) == "Unknown train admission hold 0."
        assert events == [
            "acquire:0",
            "acquire:1",
            "wait:0",
            "record:0",
            "release:0",
            "wait:1",
            "record:1",
            "release:1",
        ]
        assert len({id(loop) for loop in lifecycle.loops}) == 1
        assert lifecycle.loops[0] is not manager_loop

    async def test_cancelled_hold_acquisition_releases_the_unpublished_handle(self, lifecycle_manager):
        manager = lifecycle_manager
        events: list[str] = []
        acquire_started = threading.Event()
        allow_acquire = threading.Event()
        lifecycle = BlockingAcquireRolloutLifecycle(events, acquire_started, allow_acquire)
        _install_train_rollout_lifecycle(manager, lifecycle)
        acquire_task = asyncio.create_task(manager.acquire_train_admission_hold())
        assert await asyncio.to_thread(acquire_started.wait, 5)

        acquire_task.cancel()
        await asyncio.sleep(0)

        try:
            assert acquire_task.done() is False
            allow_acquire.set()
            with pytest.raises(asyncio.CancelledError):
                await acquire_task
            assert lifecycle.active_holds == set()
            assert manager._train_admission_holds == {}
            assert events == ["acquire_started", "acquire_finished", "acquire:0", "release:0"]
        finally:
            allow_acquire.set()

            async def wait_for_hold() -> None:
                while not lifecycle.holds:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_hold(), timeout=1)
            if lifecycle.active_holds:
                lifecycle.holds[0].release()

    async def test_cancelled_hold_wait_settles_and_leaves_handle_releasable(self, lifecycle_manager):
        manager = lifecycle_manager
        events: list[str] = []
        lifecycle = RecordingRolloutLifecycle(events)
        lifecycle.allow_wait.clear()
        _install_train_rollout_lifecycle(manager, lifecycle)
        hold_id = await manager.acquire_train_admission_hold()
        wait_task = asyncio.create_task(manager.wait_train_admission_hold(hold_id))
        assert await asyncio.to_thread(lifecycle.wait_started.wait, 5)

        wait_task.cancel()
        await asyncio.sleep(0)
        assert wait_task.done() is False
        lifecycle.allow_wait.set()
        with pytest.raises(asyncio.CancelledError):
            await wait_task
        await manager.release_train_admission_hold(hold_id)

        assert events == ["acquire:0", "wait:0", "release:0"]
        assert lifecycle.active_holds == set()

    async def test_cancelled_weight_update_record_settles_and_leaves_handle_releasable(self, lifecycle_manager):
        manager = lifecycle_manager
        events: list[str] = []
        lifecycle = RecordingRolloutLifecycle(events)
        lifecycle.allow_record.clear()
        _install_train_rollout_lifecycle(manager, lifecycle)
        hold_id = await manager.acquire_train_admission_hold()
        await manager.wait_train_admission_hold(hold_id)
        record_task = asyncio.create_task(manager.record_train_weight_update(hold_id))
        assert await asyncio.to_thread(lifecycle.record_started.wait, 5)

        record_task.cancel()
        await asyncio.sleep(0)
        assert record_task.done() is False
        lifecycle.allow_record.set()
        with pytest.raises(asyncio.CancelledError):
            await record_task
        await manager.release_train_admission_hold(hold_id)

        assert events == ["acquire:0", "wait:0", "record:0", "release:0"]
        assert lifecycle.active_holds == set()

    async def test_weight_update_record_failure_retains_its_handle_and_fence(self, lifecycle_manager):
        manager = lifecycle_manager
        events: list[str] = []
        failure = RuntimeError("weight update record failed")
        lifecycle = RecordingRolloutLifecycle(events)
        lifecycle.record_failure = failure
        _install_train_rollout_lifecycle(manager, lifecycle)
        hold_id = await manager.acquire_train_admission_hold()
        await manager.wait_train_admission_hold(hold_id)

        with pytest.raises(RuntimeError) as record_error:
            await manager.record_train_weight_update(hold_id)

        assert record_error.value is failure
        assert manager._weight_update_fence_hold_id == hold_id
        assert list(manager._train_admission_holds) == [hold_id]

        lifecycle.record_failure = None
        await manager.record_train_weight_update(hold_id)
        await manager.release_train_admission_hold(hold_id)

        assert events == ["acquire:0", "wait:0", "record:0", "record:0", "release:0"]
        assert lifecycle.active_holds == set()

    async def test_cancelled_hold_release_settles_and_consumes_handle(self, lifecycle_manager):
        manager = lifecycle_manager
        events: list[str] = []
        lifecycle = RecordingRolloutLifecycle(events)
        lifecycle.allow_release.clear()
        _install_train_rollout_lifecycle(manager, lifecycle)
        hold_id = await manager.acquire_train_admission_hold()
        release_task = asyncio.create_task(manager.release_train_admission_hold(hold_id))
        assert await asyncio.to_thread(lifecycle.release_started.wait, 5)

        release_task.cancel()
        await asyncio.sleep(0)
        assert release_task.done() is False
        with pytest.raises(RuntimeError) as duplicate_release:
            await manager.release_train_admission_hold(hold_id)
        lifecycle.allow_release.set()
        with pytest.raises(asyncio.CancelledError):
            await release_task

        assert str(duplicate_release.value) == f"Unknown train admission hold {hold_id}."
        assert events == ["acquire:0", "release:0"]
        assert lifecycle.active_holds == set()

    async def test_shared_eval_holds_train_admission_without_waiting_terminal(self, lifecycle_manager):
        manager = lifecycle_manager
        manager.args.debug_train_only = False
        events: list[str] = []
        lifecycle = RecordingRolloutLifecycle(events)
        _install_train_rollout_lifecycle(manager, lifecycle)

        def evaluate(input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
            events.append(f"eval:{input.rollout_id}")
            return RolloutFnEvalOutput(data={}, metrics=None)

        manager.eval_generate_rollout = evaluate

        await manager.eval(rollout_id=19)

        assert events == ["acquire:0", "eval:19", "release:0"]

    @pytest.mark.parametrize("first_operation", ["eval", "update"])
    async def test_shared_eval_and_weight_update_are_mutually_exclusive(
        self,
        lifecycle_manager,
        first_operation: str,
    ):
        manager = lifecycle_manager
        manager.args.debug_train_only = False
        events: list[str] = []
        _install_train_rollout_lifecycle(manager, RecordingRolloutLifecycle([]))
        eval_started = threading.Event()
        finish_eval = threading.Event()
        update_started = asyncio.Event()
        finish_update = asyncio.Event()

        def evaluate(input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
            events.append("eval:start")
            eval_started.set()
            assert finish_eval.wait(timeout=5)
            events.append("eval:end")
            return RolloutFnEvalOutput(data={}, metrics=None)

        async def update_weights() -> None:
            hold_id = await manager.acquire_train_admission_hold()
            assert hold_id is not None
            await manager.wait_train_admission_hold(hold_id)
            events.append("update:start")
            update_started.set()
            await finish_update.wait()
            events.append("update:end")
            await manager.release_train_admission_hold(hold_id)

        manager.eval_generate_rollout = evaluate
        eval_task = None
        update_task = None
        try:
            if first_operation == "eval":
                events.append("eval:requested")
                eval_task = asyncio.create_task(manager.eval(rollout_id=26))
                assert await asyncio.to_thread(eval_started.wait, 5)
                events.append("update:requested")
                update_task = asyncio.create_task(update_weights())
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(update_started.wait(), timeout=0.1)
                finish_eval.set()
                await eval_task
                await asyncio.wait_for(update_started.wait(), timeout=5)
                finish_update.set()
                await update_task
            else:
                events.append("update:requested")
                update_task = asyncio.create_task(update_weights())
                await asyncio.wait_for(update_started.wait(), timeout=5)
                events.append("eval:requested")
                eval_task = asyncio.create_task(manager.eval(rollout_id=26))
                assert await asyncio.to_thread(eval_started.wait, 0.1) is False
                finish_update.set()
                await update_task
                assert await asyncio.to_thread(eval_started.wait, 5)
                finish_eval.set()
                await eval_task
        finally:
            finish_eval.set()
            finish_update.set()
            tasks = [task for task in (eval_task, update_task) if task is not None]
            await asyncio.gather(*tasks, return_exceptions=True)

        if first_operation == "eval":
            assert events == [
                "eval:requested",
                "eval:start",
                "update:requested",
                "eval:end",
                "update:start",
                "update:end",
            ]
        else:
            assert events == [
                "update:requested",
                "update:start",
                "eval:requested",
                "update:end",
                "eval:start",
                "eval:end",
            ]

    async def test_failed_weight_update_keeps_shared_eval_excluded(self, lifecycle_manager):
        manager = lifecycle_manager
        manager.args.debug_train_only = False
        events: list[str] = []
        _install_train_rollout_lifecycle(manager, RecordingRolloutLifecycle([]))
        update_failure = RuntimeError("weight update failed")
        hold_id: int | None = None
        eval_started = threading.Event()
        finish_eval = threading.Event()

        async def update_weights() -> None:
            nonlocal hold_id
            hold_id = await manager.acquire_train_admission_hold()
            assert hold_id is not None
            await manager.wait_train_admission_hold(hold_id)
            events.append("update:start")
            raise update_failure

        def evaluate(input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
            events.append("eval:start")
            eval_started.set()
            assert finish_eval.wait(timeout=5)
            events.append("eval:end")
            return RolloutFnEvalOutput(data={}, metrics=None)

        manager.eval_generate_rollout = evaluate
        with pytest.raises(RuntimeError) as error:
            await update_weights()
        assert error.value is update_failure

        events.append("eval:requested")
        eval_task = asyncio.create_task(manager.eval(rollout_id=27))
        try:
            assert await asyncio.to_thread(eval_started.wait, 0.1) is False
            assert hold_id is not None
            await manager.release_train_admission_hold(hold_id)
            hold_id = None
            assert await asyncio.to_thread(eval_started.wait, 5)
            finish_eval.set()
            await eval_task
        finally:
            if hold_id is not None:
                await manager.release_train_admission_hold(hold_id)
            finish_eval.set()
            await asyncio.gather(eval_task, return_exceptions=True)

        assert events == [
            "update:start",
            "eval:requested",
            "eval:start",
            "eval:end",
        ]

    async def test_weight_update_release_failure_poison_fails_future_engine_users(
        self,
        lifecycle_manager,
    ):
        manager = lifecycle_manager
        manager.args.debug_train_only = False
        release_failure = RuntimeError("hold release failed")
        evaluated = False

        class FailingReleaseHold(TrainAdmissionHold):
            async def _wait_terminal(self) -> None:
                return

            def _record_weight_update(self) -> None:
                return

            def _release(self) -> None:
                raise release_failure

        class FailingOnceReleaseLifecycle(RecordingRolloutLifecycle):
            def __init__(self) -> None:
                super().__init__([])
                self._fail_next_release = True

            async def acquire_train_admission_hold(self) -> TrainAdmissionHold:
                if self._fail_next_release:
                    self._fail_next_release = False
                    return FailingReleaseHold()
                return await super().acquire_train_admission_hold()

        def evaluate(input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
            nonlocal evaluated
            evaluated = True
            return RolloutFnEvalOutput(data={}, metrics=None)

        _install_train_rollout_lifecycle(manager, FailingOnceReleaseLifecycle())
        manager.eval_generate_rollout = evaluate
        failed_hold_id = await manager.acquire_train_admission_hold()
        assert failed_hold_id is not None
        await manager.wait_train_admission_hold(failed_hold_id)
        await manager.record_train_weight_update(failed_hold_id)

        with pytest.raises(RuntimeError) as release_error:
            await manager.release_train_admission_hold(failed_hold_id)
        assert release_error.value is release_failure

        queued_hold_id = await manager.acquire_train_admission_hold()
        assert queued_hold_id is not None
        with pytest.raises(RuntimeError) as wait_error:
            await asyncio.wait_for(manager.wait_train_admission_hold(queued_hold_id), timeout=1)
        assert str(wait_error.value) == "Weight-update fence failed during train admission hold release."
        assert wait_error.value.__cause__ is release_failure
        await manager.release_train_admission_hold(queued_hold_id)

        with pytest.raises(RuntimeError) as eval_error:
            await asyncio.wait_for(manager.eval(rollout_id=29), timeout=1)
        assert str(eval_error.value) == "Weight-update fence failed during train admission hold release."
        assert eval_error.value.__cause__ is release_failure
        assert evaluated is False

    async def test_cancelled_queued_weight_update_wait_does_not_claim_exclusion(self, lifecycle_manager):
        manager = lifecycle_manager
        manager.args.debug_train_only = False
        events: list[str] = []
        _install_train_rollout_lifecycle(manager, RecordingRolloutLifecycle([]))

        def evaluate(input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
            events.append("eval")
            return RolloutFnEvalOutput(data={}, metrics=None)

        manager.eval_generate_rollout = evaluate
        active_hold_id = await manager.acquire_train_admission_hold()
        assert active_hold_id is not None
        await manager.wait_train_admission_hold(active_hold_id)
        queued_hold_id = await manager.acquire_train_admission_hold()
        assert queued_hold_id is not None
        queued_wait_task = asyncio.create_task(manager.wait_train_admission_hold(queued_hold_id))
        await asyncio.sleep(0)
        assert queued_wait_task.done() is False

        queued_wait_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued_wait_task
        await manager.release_train_admission_hold(queued_hold_id)
        await manager.release_train_admission_hold(active_hold_id)

        await asyncio.wait_for(manager.eval(rollout_id=28), timeout=5)

        assert events == ["eval"]

    async def test_cancelled_legacy_shared_eval_returns_before_worker_finishes(self, lifecycle_manager):
        manager = lifecycle_manager
        manager.args.debug_train_only = False
        events: list[str] = []
        eval_started = threading.Event()
        finish_eval = threading.Event()

        def evaluate(input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
            events.append("eval_started")
            eval_started.set()
            assert finish_eval.wait(timeout=5)
            events.append("eval_finished")
            return RolloutFnEvalOutput(data={}, metrics=None)

        manager.eval_generate_rollout = evaluate
        eval_task = asyncio.create_task(manager.eval(rollout_id=25))
        assert await asyncio.to_thread(eval_started.wait, 5)

        try:
            eval_task.cancel()
            done, pending = await asyncio.wait({eval_task}, timeout=0.5)

            assert done == {eval_task}
            assert pending == set()
            with pytest.raises(asyncio.CancelledError):
                await eval_task
            assert events == ["eval_started"]
        finally:
            finish_eval.set()
            await asyncio.gather(eval_task, return_exceptions=True)

    async def test_checkpoint_eval_does_not_acquire_train_admission_hold(self, lifecycle_manager):
        manager = lifecycle_manager
        manager.args.debug_train_only = False
        events: list[str] = []
        _install_train_rollout_lifecycle(manager, RecordingRolloutLifecycle(events))
        manager.args.eval_uses_snapshots = True

        async def checkpoint_eval(
            rollout_id: int,
            hf_dir: str | None,
            export_time_seconds: float | None,
            require_marker: bool,
        ) -> str:
            events.append(f"checkpoint_eval:{rollout_id}")
            return "checkpoint-result"

        manager._eval_checkpoint = checkpoint_eval

        result = await manager.eval(rollout_id=23, hf_dir="/snapshot")

        assert result == "checkpoint-result"
        assert events == ["checkpoint_eval:23"]

    async def test_shared_eval_preserves_failure_when_exact_release_also_fails(self, lifecycle_manager):
        manager = lifecycle_manager
        manager.args.debug_train_only = False
        events: list[str] = []
        eval_failure = RuntimeError("shared eval failed")
        release_failure = RuntimeError("hold release failed")

        class FailingReleaseHold(TrainAdmissionHold):
            async def _wait_terminal(self) -> None:
                raise AssertionError("shared eval must not wait for terminal work")

            def _record_weight_update(self) -> None:
                raise AssertionError("shared eval must not record a weight update")

            def _release(self) -> None:
                events.append("release")
                raise release_failure

        class FailingReleaseLifecycle(RecordingRolloutLifecycle):
            async def acquire_train_admission_hold(self) -> TrainAdmissionHold:
                events.append("acquire")
                return FailingReleaseHold()

        def evaluate(input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
            events.append("eval")
            raise eval_failure

        _install_train_rollout_lifecycle(manager, FailingReleaseLifecycle(events))
        manager.eval_generate_rollout = evaluate

        with pytest.raises(RuntimeError) as error:
            await manager.eval(rollout_id=24)

        assert error.value is eval_failure
        assert error.value.__cause__ is release_failure
        assert events == ["acquire", "eval", "release"]

    async def test_overlapping_shared_evals_release_only_their_exact_holds(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        manager.args.debug_train_only = False
        events: list[str] = []
        lifecycle = RecordingRolloutLifecycle(events)
        _install_train_rollout_lifecycle(manager, lifecycle)
        monkeypatch.setattr(rmgr, "timer", lambda name: nullcontext())
        started = {20: threading.Event(), 21: threading.Event()}
        finish = {20: threading.Event(), 21: threading.Event()}

        def evaluate(input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
            events.append(f"eval_started:{input.rollout_id}")
            started[input.rollout_id].set()
            assert finish[input.rollout_id].wait(timeout=5)
            events.append(f"eval_finished:{input.rollout_id}")
            return RolloutFnEvalOutput(data={}, metrics=None)

        manager.eval_generate_rollout = evaluate
        first_eval = asyncio.create_task(manager.eval(rollout_id=20))
        assert await asyncio.to_thread(started[20].wait, 5)
        second_eval = asyncio.create_task(manager.eval(rollout_id=21))
        assert await asyncio.to_thread(started[21].wait, 5)

        finish[20].set()
        await first_eval

        assert lifecycle.active_holds == {1}
        assert second_eval.done() is False

        finish[21].set()
        await second_eval

        assert lifecycle.active_holds == set()
        assert events == [
            "acquire:0",
            "eval_started:20",
            "acquire:1",
            "eval_started:21",
            "eval_finished:20",
            "release:0",
            "eval_finished:21",
            "release:1",
        ]

    async def test_cancelled_shared_eval_blocks_weight_update_until_invocation_settles(self, lifecycle_manager):
        manager = lifecycle_manager
        manager.args.debug_train_only = False
        events: list[str] = []
        _install_train_rollout_lifecycle(manager, RecordingRolloutLifecycle([]))
        eval_started = threading.Event()
        finish_eval = threading.Event()
        update_started = asyncio.Event()

        def evaluate(input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
            events.append("eval:start")
            eval_started.set()
            assert finish_eval.wait(timeout=5)
            events.append("eval:end")
            return RolloutFnEvalOutput(data={}, metrics=None)

        async def update_weights() -> None:
            hold_id = await manager.acquire_train_admission_hold()
            assert hold_id is not None
            await manager.wait_train_admission_hold(hold_id)
            events.append("update:start")
            update_started.set()
            await manager.release_train_admission_hold(hold_id)

        manager.eval_generate_rollout = evaluate
        eval_task = asyncio.create_task(manager.eval(rollout_id=22))
        assert await asyncio.to_thread(eval_started.wait, 5)

        eval_task.cancel()
        update_task = asyncio.create_task(update_weights())
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(update_started.wait(), timeout=0.1)
            assert eval_task.done() is False
            assert events == ["eval:start"]
        finally:
            finish_eval.set()
            await asyncio.gather(eval_task, update_task, return_exceptions=True)

        with pytest.raises(asyncio.CancelledError):
            await eval_task

        assert events == ["eval:start", "eval:end", "update:start"]

    async def test_dispose_closes_unique_rollout_lifecycles_before_manager_resources(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        events: list[str] = []
        train_lifecycle = RecordingRolloutLifecycle(events, label="train")
        eval_lifecycle = RecordingRolloutLifecycle(events, label="eval")
        _install_rollout_lifecycles(manager, train_lifecycle, eval_lifecycle)

        class RecordingDataSource:
            def close(self) -> None:
                events.append("source_close")

        manager.data_source = RecordingDataSource()
        monkeypatch.setattr(
            rmgr.event_analyzer,
            "run_analysis_from_args",
            lambda args: events.append("analysis"),
        )

        await manager.dispose()

        assert events == ["close:train", "close:eval", "source_close", "analysis"]

    async def test_dispose_continues_manager_cleanup_after_first_failure(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        events: list[str] = []
        failure = RuntimeError("source cleanup failed")

        class FailingOnceDataSource:
            def __init__(self) -> None:
                self._failure: BaseException | None = failure

            def close(self) -> None:
                events.append("source_close")
                if self._failure is not None:
                    source_failure = self._failure
                    self._failure = None
                    raise source_failure

        class RecordingDisposable:
            def __init__(self, event: str) -> None:
                self._event = event

            def dispose(self) -> None:
                events.append(self._event)

        class RecordingCheckpointEvalFn(CheckpointEvalFn):
            async def evaluate_checkpoint(
                self,
                checkpoint_dir: str,
                input: RolloutFnEvalInput,
            ) -> RolloutFnEvalOutput:
                raise AssertionError("cleanup must not invoke evaluation")

            def dispose(self) -> None:
                events.append("checkpoint_dispose")

        class RecordingMonitor:
            def stop(self) -> None:
                events.append("monitor_stop")

        manager.data_source = FailingOnceDataSource()
        manager._metric_checker = RecordingDisposable("metric_dispose")
        manager.eval_generate_rollout = RecordingCheckpointEvalFn()
        manager._health_monitors = [RecordingMonitor()]
        monkeypatch.setattr(
            rmgr.event_analyzer,
            "run_analysis_from_args",
            lambda args: events.append("analysis"),
        )

        with pytest.raises(RuntimeError) as cleanup_error:
            await manager.dispose()

        assert cleanup_error.value is failure
        assert events == [
            "source_close",
            "analysis",
            "metric_dispose",
            "checkpoint_dispose",
            "monitor_stop",
        ]

        await manager.dispose()
        await manager.dispose()

        assert events == [
            "source_close",
            "analysis",
            "metric_dispose",
            "checkpoint_dispose",
            "monitor_stop",
            "source_close",
        ]

    async def test_dispose_retries_failed_lifecycle_before_releasing_manager_resources(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        events: list[str] = []
        failure = RuntimeError("lifecycle close failed")
        lifecycle = FailingOnceCloseRolloutLifecycle(events, failure)
        _install_rollout_lifecycles(manager, lifecycle)

        class RecordingDataSource:
            def close(self) -> None:
                events.append("source_close")

        manager.data_source = RecordingDataSource()
        monkeypatch.setattr(
            rmgr.event_analyzer,
            "run_analysis_from_args",
            lambda args: events.append("analysis"),
        )

        with pytest.raises(RuntimeError) as close_error:
            await manager.dispose()

        assert close_error.value is failure
        assert events == ["close_failed"]

        await manager.dispose()

        assert events == ["close_failed", "close_succeeded", "source_close", "analysis"]

    async def test_cancelled_dispose_waits_for_lifecycle_then_cleans_manager_resources(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        events: list[str] = []
        close_started = threading.Event()
        allow_close = threading.Event()
        lifecycle = BlockingCloseRolloutLifecycle(events, close_started, allow_close)
        _install_rollout_lifecycles(manager, lifecycle)

        class RecordingDataSource:
            def close(self) -> None:
                events.append("source_close")

        manager.data_source = RecordingDataSource()
        monkeypatch.setattr(
            rmgr.event_analyzer,
            "run_analysis_from_args",
            lambda args: events.append("analysis"),
        )
        dispose_task = asyncio.create_task(manager.dispose())
        assert await asyncio.to_thread(close_started.wait, 5)

        dispose_task.cancel()
        await asyncio.sleep(0)

        assert dispose_task.done() is False
        assert events == ["close_started"]

        allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await dispose_task

        assert events == ["close_started", "close_finished", "source_close", "analysis"]

    async def test_overlapping_dispose_calls_close_and_clean_up_once(
        self,
        lifecycle_manager,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        manager = lifecycle_manager
        events: list[str] = []
        close_started = threading.Event()
        allow_close = threading.Event()
        lifecycle = BlockingCloseRolloutLifecycle(events, close_started, allow_close)
        _install_rollout_lifecycles(manager, lifecycle)

        class RecordingDataSource:
            def close(self) -> None:
                events.append("source_close")

        manager.data_source = RecordingDataSource()
        monkeypatch.setattr(
            rmgr.event_analyzer,
            "run_analysis_from_args",
            lambda args: events.append("analysis"),
        )
        first_dispose = asyncio.create_task(manager.dispose())
        assert await asyncio.to_thread(close_started.wait, 5)
        second_dispose = asyncio.create_task(manager.dispose())
        await asyncio.sleep(0)

        assert second_dispose.done() is False
        assert events == ["close_started"]

        allow_close.set()
        await asyncio.gather(first_dispose, second_dispose)

        assert events == ["close_started", "close_finished", "source_close", "analysis"]


@pytest.mark.asyncio
class TestStartStopCell:
    async def test_stop_cell_kills_target_engine_only(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``stop_cell(0)`` kills cell 0's actor; cell 1 untouched."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        actor0, actor1 = eal.rollout_engines

        await manager.stop_cell(0)

        await _assert_engine_dies(actor0)
        assert ray.get(actor1.health_generate.remote(timeout=1.0)) is True

    async def test_start_cell_recovers_after_stop_cell(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """stop_cell(0) → start_cell(0) drives a real ``recover()`` that spawns
        a fresh mock actor in place of the killed one."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal_before = await manager.get_updatable_engines_and_lock()
        actor0_before = eal_before.rollout_engines[0]

        await manager.stop_cell(0)
        await manager.start_cell(0)

        eal_after = await manager.get_updatable_engines_and_lock()
        actor0_after = eal_after.rollout_engines[0]

        assert actor0_after is not actor0_before, "start_cell must produce a fresh actor"
        assert ray.get(actor0_after.health_generate.remote(timeout=1.0)) is True

    async def test_stop_cell_targets_high_id_correctly(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``stop_cell(1)`` (not 0) must kill engine 1, leaving engine 0 alive —
        guards against off-by-one in ``get_cell_indexer_of_id_map``."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        actor0, actor1 = eal.rollout_engines

        await manager.stop_cell(1)

        assert ray.get(actor0.health_generate.remote(timeout=1.0)) is True
        await _assert_engine_dies(actor1)

    async def test_stop_cell_is_idempotent_on_already_stopped(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Calling ``stop_cell(0)`` twice does not raise — production code logs
        and proceeds when the engine is already de-allocated."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        await manager.get_updatable_engines_and_lock()  # ensure engines are alive

        await manager.stop_cell(0)
        await manager.stop_cell(0)  # must not raise


@pytest.mark.asyncio
class TestCellDispatchAcrossModels:
    async def test_cells_route_to_correct_model_by_sorted_srv_key(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Cells are flattened in sorted-srv-key order: with models ("actor",
        "ref") the cells map (0,1)→actor, (2,3)→ref. Stopping cell 2 must hit
        ref's first engine and leave actor's engines untouched."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        actor_handles = [e.actor_handle for e in manager.servers["actor"].server_groups[0].engines]
        ref_handles = [e.actor_handle for e in manager.servers["ref"].server_groups[0].engines]

        await manager.stop_cell(2)

        # actor untouched
        for h in actor_handles:
            assert ray.get(h.health_generate.remote(timeout=1.0)) is True
        # ref engine 0 dead, ref engine 1 alive
        await _assert_engine_dies(ref_handles[0])
        assert ray.get(ref_handles[1].health_generate.remote(timeout=1.0)) is True


@pytest.mark.asyncio
class TestGetUpdatableEnginesAndLock:
    async def test_returns_only_updatable_servers_engines_in_multi_model_setup(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """With actor (update_weights=True) + ref (update_weights=False), the
        returned EnginesAndLock contains the actor's engines only."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        assert len(eal.rollout_engines) == 2  # actor's 2, not ref's 2
        assert eal.engine_gpu_counts == [1, 1]
        assert all(isinstance(h, ray.actor.ActorHandle) for h in eal.rollout_engines)
        assert ray.get(eal.rollout_engines[0].health_generate.remote(timeout=1.0)) is True

    async def test_returns_empty_when_no_updatable_model(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """If every model has ``update_weights=False`` (e.g. inference-only
        deployment), the returned EnginesAndLock has empty engines list and
        the lock handle is still present (callers always need a lock)."""
        args = _make_test_args(tmp_path, models=[("ref", False)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        assert eal.rollout_engines == []
        assert eal.engine_gpu_counts == []
        assert eal.has_new_engines is False
        assert eal.rollout_engine_lock is not None

    async def test_has_new_engines_flag_lifecycle(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Lifecycle the trainer relies on: ``has_new_engines`` is True after
        init, False after ``clear_updatable_has_new_engines``, True again
        after ``start_cell`` spawns a fresh engine."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal_init = await manager.get_updatable_engines_and_lock()
        assert eal_init.has_new_engines is True

        manager.clear_updatable_has_new_engines()
        eal_cleared = await manager.get_updatable_engines_and_lock()
        assert eal_cleared.has_new_engines is False

        await manager.stop_cell(0)
        await manager.start_cell(0)
        eal_recovered = await manager.get_updatable_engines_and_lock()
        assert eal_recovered.has_new_engines is True

    async def test_clear_does_not_affect_non_updatable_server(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``clear_updatable_has_new_engines`` must touch only the updatable
        server's flag; non-updatable (ref) servers keep their flag intact."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        # Force ref's flag True so we can detect any erroneous clear.
        manager.servers["ref"].server_groups[0].has_new_engines = True

        manager.clear_updatable_has_new_engines()

        assert manager.servers["ref"].server_groups[0].has_new_engines is True
        assert manager.servers["actor"].server_groups[0].has_new_engines is False

    async def test_multiple_updatable_servers_raises_assertion(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Production guards against misconfiguration where two models both set
        ``update_weights=True``; that's ambiguous for the trainer."""
        args = _make_test_args(tmp_path, models=[("actor1", True), ("actor2", True)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        with pytest.raises(ValueError, match="Multiple servers"):
            await manager.get_updatable_engines_and_lock()


@pytest.mark.asyncio
class TestCheckWeights:
    async def test_check_weights_targets_only_updatable_model(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``check_weights`` targets only the updatable model. The snapshot/reset/
        compare round-trip is meaningless for a frozen model (restored from disk,
        never re-synced via update_weights), so it must be skipped there."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        await manager.get_updatable_engines_and_lock()  # wait for engines to be alive

        results = await manager.check_weights(action="pre_update")

        # Updatable server only: nested gather is [group][engine]; 1 group × 2 engines.
        assert len(results) == 1
        for per_group in results:
            assert len(per_group) == 2
            for engine_result in per_group:
                assert engine_result == {"_mock": True}

        # Frozen (non-updatable) servers must not have been touched.
        for srv in manager.servers.values():
            if srv.update_weights:
                continue
            for group in srv.server_groups:
                for engine in group.engines:
                    if not engine.is_allocated:
                        continue
                    calls = ray.get(engine.actor_handle.get_calls.remote())
                    assert not any(c[0] == "check_weights" for c in calls)


@pytest.mark.asyncio
class TestRecoverUpdatableEngines:
    async def test_skips_recovery_when_no_rollout_started(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``recover_updatable_engines`` is a no-op while ``rollout_id == -1``
        (initial state) — the trainer hasn't issued a rollout yet, so even if
        a slot looks dead the manager must not pre-emptively recover."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal_before = await manager.get_updatable_engines_and_lock()
        actor0_before = eal_before.rollout_engines[0]

        # Kill engine 0 directly + mark stopped (simulates a fault before any
        # rollout). recover_updatable_engines must not bring it back yet.
        ray.kill(actor0_before)
        manager.servers["actor"].server_groups[0].all_engines[0].mark_stopped()

        await manager.recover_updatable_engines()

        # Slot 0 is still de-allocated; recovery skipped because rollout_id=-1.
        assert not manager.servers["actor"].server_groups[0].all_engines[0].is_allocated

    async def test_recovers_dead_engine_after_rollout_started(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Once ``rollout_id`` advances past -1 (mid-training), a dead slot on
        the updatable server is brought back by ``recover_updatable_engines``."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal_before = await manager.get_updatable_engines_and_lock()
        actor0_before = eal_before.rollout_engines[0]

        ray.kill(actor0_before)
        manager.servers["actor"].server_groups[0].all_engines[0].mark_stopped()

        manager.rollout_id = 0  # simulates "rollout has started"
        await manager.recover_updatable_engines()

        slot0 = manager.servers["actor"].server_groups[0].all_engines[0]
        assert slot0.is_allocated
        assert slot0.actor_handle is not actor0_before
        assert ray.get(slot0.actor_handle.health_generate.remote(timeout=1.0)) is True


@pytest.mark.asyncio
class TestGenerate:
    """``generate(rollout_id)`` is the trainer's per-iteration rollout entry
    point. It must (1) advance ``self.rollout_id``, (2) call the rollout
    function with ``RolloutFnTrainInput(rollout_id=N)``, (3) postprocess +
    convert + DP-split the returned samples. Nothing else covers this path."""

    async def test_invokes_rollout_fn_with_correct_input_and_returns_dp_split(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        args = _make_test_args(tmp_path, models=[("actor", True)])
        # global_batch_size = number of samples we'll produce (postprocess
        # trims to a multiple, so equality avoids losing samples).
        args.global_batch_size = 8
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        manager.train_parallel_config = {"dp_size": 2}

        captured: list = []

        def fake_rollout_fn(input):
            captured.append(input)
            return RolloutFnTrainOutput(
                samples=[make_samples_grouped(n_groups=2, group_size=4)],
                metrics={"my_metric": 1.23},
            )

        manager.generate_rollout = fake_rollout_fn

        result = await manager.generate(rollout_id=42)

        assert manager.rollout_id == 42
        assert len(captured) == 1
        assert isinstance(captured[0], RolloutFnTrainInput)
        assert captured[0].rollout_id == 42
        # generate returns {"sample_indices": ..., "data_ref": ...};
        # split_train_data_by_dp returns Box(ObjectRef) per dp rank
        assert set(result) == {"sample_indices", "data_ref"}
        data_refs = result["data_ref"]
        assert len(data_refs) == 2
        partitions = ray.get([box.inner for box in data_refs])
        for partition in partitions:
            assert "tokens" in partition
            assert "rewards" in partition
            assert "loss_masks" in partition
            # 8 samples / 2 dp = 4 per rank
            assert len(partition["tokens"]) == 4

    async def test_commits_leased_output_after_every_dp_shard_is_published(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.global_batch_size = 8
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        manager.train_parallel_config = {"dp_size": 2}

        events: list[str] = []
        lease = RecordingTrainBatchLease(rollout_id=42, events=events)
        manager.generate_rollout = lambda input: LeasedRolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=2, group_size=4)],
            metrics={"my_metric": 1.23},
            lease=lease,
        )
        original_ray_put = ray.put

        def recording_ray_put(value):
            events.append("publish")
            return original_ray_put(value)

        monkeypatch.setattr(ray, "put", recording_ray_put)

        result = await manager.generate(rollout_id=42)

        assert events == ["publish", "publish", "commit"]
        assert set(result) == {"sample_indices", "data_ref"}
        assert len(result["data_ref"]) == 2


@pytest.mark.asyncio
class TestEval:
    async def test_invokes_eval_fn_with_eval_input(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)

        captured: list = []

        def fake_eval_fn(input):
            captured.append(input)
            return RolloutFnEvalOutput(
                data={"my_dataset": {"rewards": [0.5, 1.0]}},
                metrics={},
            )

        manager.eval_generate_rollout = fake_eval_fn

        await manager.eval(rollout_id=10)

        assert len(captured) == 1
        assert isinstance(captured[0], RolloutFnEvalInput)
        assert captured[0].rollout_id == 10

    async def test_skipped_in_debug_train_only_mode(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``debug_train_only=True`` must short-circuit ``eval`` before the
        rollout function is invoked — used by trainer-only debug runs that
        have no rollout cluster."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.debug_train_only = True
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)

        called: list = []
        manager.eval_generate_rollout = lambda inp: called.append(inp)

        await manager.eval(rollout_id=10)

        assert called == []
