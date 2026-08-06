import asyncio
from collections.abc import Callable, Coroutine
from types import SimpleNamespace

import pytest
import ray

import train_async


class RemoteMethod:
    def __init__(self, function: Callable[..., object]) -> None:
        self._function = function

    def remote(self, *args: object, **kwargs: object) -> object:
        return self._function(*args, **kwargs)


class RecordingRolloutManager:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._next_hold_id = 0
        self.active_holds: set[int] = set()
        self.acquire_train_admission_hold = RemoteMethod(self._acquire_train_admission_hold)
        self.wait_train_admission_hold = RemoteMethod(self._wait_train_admission_hold)
        self.record_train_weight_update = RemoteMethod(self._record_train_weight_update)
        self.release_train_admission_hold = RemoteMethod(self._release_train_admission_hold)
        self.generate = RemoteMethod(self._generate)
        self.check_weights = RemoteMethod(self._unexpected_check_weights)
        self.save = RemoteMethod(self._unexpected_save)
        self.dispose = RemoteMethod(self._dispose)

    async def _acquire_train_admission_hold(self) -> int:
        hold_id = self._next_hold_id
        self._next_hold_id += 1
        self.active_holds.add(hold_id)
        self._events.append(f"acquire:{hold_id}")
        return hold_id

    async def _wait_train_admission_hold(self, hold_id: int) -> None:
        assert hold_id in self.active_holds
        self._events.append(f"wait:{hold_id}")

    async def _record_train_weight_update(self, hold_id: int) -> None:
        assert hold_id in self.active_holds
        self._events.append(f"record:{hold_id}")

    async def _release_train_admission_hold(self, hold_id: int) -> None:
        self.active_holds.remove(hold_id)
        self._events.append(f"release:{hold_id}")

    def _generate(self, rollout_id: int) -> Coroutine[object, object, str]:
        self._events.append(f"generate:{rollout_id}")

        async def complete() -> str:
            self._events.append(f"handoff:{rollout_id}")
            return f"data:{rollout_id}"

        return complete()

    async def _unexpected_check_weights(self, **kwargs: object) -> None:
        raise AssertionError(f"unexpected weight check: {kwargs}")

    async def _unexpected_save(self, rollout_id: int) -> None:
        raise AssertionError(f"unexpected save for rollout {rollout_id}")

    async def _dispose(self) -> None:
        self._events.append("dispose")


class RecordingCheckpointRolloutManager(RecordingRolloutManager):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.save = RemoteMethod(self._save)

    async def _save(self, rollout_id: int) -> None:
        self._events.append(f"rollout_save:{rollout_id}")


class FailingWaitRolloutManager(RecordingRolloutManager):
    def __init__(self, events: list[str], failure: RuntimeError) -> None:
        self._failure = failure
        super().__init__(events)

    async def _wait_train_admission_hold(self, hold_id: int) -> None:
        assert hold_id in self.active_holds
        self._events.append(f"wait:{hold_id}")
        raise self._failure


class FailingRecordRolloutManager(RecordingRolloutManager):
    def __init__(self, events: list[str], failure: RuntimeError) -> None:
        self._failure = failure
        super().__init__(events)

    async def _record_train_weight_update(self, hold_id: int) -> None:
        assert hold_id in self.active_holds
        self._events.append(f"record:{hold_id}")
        raise self._failure


class RecordingActorModel:
    def __init__(
        self,
        events: list[str],
        update_failure: tuple[str, RuntimeError] | None,
    ) -> None:
        self._events = events
        self._update_failure = update_failure

    async def update_weights(self, *args: object, **kwargs: object) -> None:
        assert args == ()
        if kwargs:
            assert list(kwargs) == ["rollout_id"]
            rollout_id = kwargs["rollout_id"]
            assert isinstance(rollout_id, int)
            label = str(rollout_id)
        else:
            label = "initial"
        self._events.append(f"update:{label}")
        if self._update_failure is not None and self._update_failure[0] == label:
            _, failure = self._update_failure
            self._update_failure = None
            raise failure

    async def train(self, rollout_id: int, rollout_data: str) -> None:
        assert rollout_data == f"data:{rollout_id}"
        self._events.append(f"train:{rollout_id}")

    async def save_model(self, rollout_id: int, *, force_sync: bool) -> None:
        raise AssertionError(f"unexpected model save for rollout {rollout_id}, force_sync={force_sync}")


class RecordingCheckpointActorModel(RecordingActorModel):
    async def save_model(self, rollout_id: int, *, force_sync: bool) -> None:
        self._events.append(f"model_save:{rollout_id}:{force_sync}")


class RecordingEvalDispatcher:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def dispatch(self, rollout_id: int, **kwargs: object) -> None:
        raise AssertionError(f"unexpected eval for rollout {rollout_id}: {kwargs}")

    async def drain(self) -> None:
        self._events.append("eval_drain")


def make_args(
    *,
    num_rollout: int,
    update_weights_interval: int,
    save_interval: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        check_weight_update_equal=False,
        colocate=False,
        control_server_port=None,
        debug_exit_after_rollout=None,
        eval_interval=None,
        ft_components=[],
        hf_checkpoint=None,
        num_rollout=num_rollout,
        save_interval=save_interval,
        save_trigger_sentinel=None,
        skip_eval_before_train=True,
        start_rollout_id=0,
        update_weights_interval=update_weights_interval,
        use_critic=False,
    )


def patch_train_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    rollout_manager: object,
    actor_model: RecordingActorModel,
    events: list[str],
) -> None:
    monkeypatch.setattr(train_async, "configure_logger", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_async, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_async, "create_placement_groups", lambda args: {"rollout": object()})
    monkeypatch.setattr(train_async.object_store, "init_instance", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_async, "init_tracking", lambda args: None)
    monkeypatch.setattr(train_async, "create_rollout_manager", lambda args, pg: (rollout_manager, 1))

    async def create_training_models(args: object, pgs: object, manager: object) -> tuple[RecordingActorModel, None]:
        assert manager is rollout_manager
        return actor_model, None

    monkeypatch.setattr(train_async, "create_training_models", create_training_models)
    monkeypatch.setattr(train_async, "maybe_start_mini_ft_controller", lambda args: None)
    monkeypatch.setattr(train_async, "remove_rollout_data_refs", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_async, "should_run_periodic_action", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        train_async,
        "EvalDispatcher",
        lambda args, actor, manager: RecordingEvalDispatcher(events),
    )


@pytest.mark.asyncio
async def test_weight_updates_wait_for_and_release_exact_admission_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    rollout_manager = RecordingRolloutManager(events)
    actor_model = RecordingActorModel(events, update_failure=None)
    patch_train_dependencies(monkeypatch, rollout_manager, actor_model, events)

    await train_async.train(make_args(num_rollout=2, update_weights_interval=1))

    assert events == [
        "acquire:0",
        "wait:0",
        "update:initial",
        "record:0",
        "release:0",
        "generate:0",
        "handoff:0",
        "generate:1",
        "train:0",
        "handoff:1",
        "acquire:1",
        "wait:1",
        "update:0",
        "record:1",
        "release:1",
        "train:1",
        "acquire:2",
        "wait:2",
        "update:1",
        "record:2",
        "release:2",
        "eval_drain",
        "dispose",
    ]
    assert rollout_manager.active_holds == set()


@pytest.mark.asyncio
async def test_checkpoint_waits_for_prefetched_handoff_and_reuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    rollout_manager = RecordingCheckpointRolloutManager(events)
    actor_model = RecordingCheckpointActorModel(events, update_failure=None)
    patch_train_dependencies(monkeypatch, rollout_manager, actor_model, events)
    monkeypatch.setattr(
        train_async,
        "should_run_periodic_action",
        lambda rollout_id, interval, num_rollout_per_epoch, num_rollout: interval is not None,
    )

    await train_async.train(make_args(num_rollout=2, update_weights_interval=1, save_interval=1))

    assert events == [
        "acquire:0",
        "wait:0",
        "update:initial",
        "record:0",
        "release:0",
        "generate:0",
        "handoff:0",
        "generate:1",
        "train:0",
        "model_save:0:False",
        "handoff:1",
        "rollout_save:0",
        "acquire:1",
        "wait:1",
        "update:0",
        "record:1",
        "release:1",
        "train:1",
        "model_save:1:True",
        "rollout_save:1",
        "acquire:2",
        "wait:2",
        "update:1",
        "record:2",
        "release:2",
        "eval_drain",
        "dispose",
    ]
    assert rollout_manager.active_holds == set()


@pytest.mark.asyncio
async def test_weight_update_failure_retains_admission_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    failure = RuntimeError("weight update failed")
    rollout_manager = RecordingRolloutManager(events)
    actor_model = RecordingActorModel(events, update_failure=("initial", failure))
    patch_train_dependencies(monkeypatch, rollout_manager, actor_model, events)

    with pytest.raises(RuntimeError) as update_error:
        await train_async.train(make_args(num_rollout=1, update_weights_interval=1))

    assert update_error.value is failure
    assert events == ["acquire:0", "wait:0", "update:initial"]
    assert rollout_manager.active_holds == {0}


@pytest.mark.asyncio
async def test_wait_failure_retains_admission_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    failure = RuntimeError("terminal frontier failed")
    rollout_manager = FailingWaitRolloutManager(events, failure)
    actor_model = RecordingActorModel(events, update_failure=None)
    patch_train_dependencies(monkeypatch, rollout_manager, actor_model, events)

    with pytest.raises(RuntimeError) as wait_error:
        await train_async.train(make_args(num_rollout=1, update_weights_interval=1))

    assert wait_error.value is failure
    assert events == ["acquire:0", "wait:0"]
    assert rollout_manager.active_holds == {0}


@pytest.mark.asyncio
async def test_record_failure_retains_admission_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    failure = RuntimeError("weight update record failed")
    rollout_manager = FailingRecordRolloutManager(events, failure)
    actor_model = RecordingActorModel(events, update_failure=None)
    patch_train_dependencies(monkeypatch, rollout_manager, actor_model, events)

    with pytest.raises(RuntimeError) as record_error:
        await train_async.train(make_args(num_rollout=1, update_weights_interval=1))

    assert record_error.value is failure
    assert events == ["acquire:0", "wait:0", "update:initial", "record:0"]
    assert rollout_manager.active_holds == {0}


@pytest.mark.asyncio
async def test_periodic_weight_update_failure_retains_its_admission_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    failure = RuntimeError("periodic weight update failed")
    rollout_manager = RecordingRolloutManager(events)
    actor_model = RecordingActorModel(events, update_failure=("0", failure))
    patch_train_dependencies(monkeypatch, rollout_manager, actor_model, events)

    with pytest.raises(RuntimeError) as update_error:
        await train_async.train(make_args(num_rollout=2, update_weights_interval=1))

    assert update_error.value is failure
    assert events == [
        "acquire:0",
        "wait:0",
        "update:initial",
        "record:0",
        "release:0",
        "generate:0",
        "handoff:0",
        "generate:1",
        "train:0",
        "handoff:1",
        "acquire:1",
        "wait:1",
        "update:0",
    ]
    assert rollout_manager.active_holds == {1}


@pytest.mark.asyncio
async def test_cancelled_hold_acquisition_releases_completed_remote_hold(
    monkeypatch: pytest.MonkeyPatch,
    ray_local_mode: None,
) -> None:
    class CancellationRaceRolloutManager:
        def __init__(self) -> None:
            self._acquire_started = False
            self._allow_acquire = False
            self._acquire_finished = False
            self._acquire_cancelled = False
            self._active_holds: set[int] = set()

        async def acquire_train_admission_hold(self) -> int:
            self._acquire_started = True
            try:
                while not self._allow_acquire:
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                self._acquire_finished = True
                self._acquire_cancelled = True
                raise
            self._active_holds.add(0)
            self._acquire_finished = True
            return 0

        async def wait_train_admission_hold(self, hold_id: int) -> None:
            assert hold_id in self._active_holds

        async def record_train_weight_update(self, hold_id: int) -> None:
            assert hold_id in self._active_holds

        async def release_train_admission_hold(self, hold_id: int) -> None:
            self._active_holds.remove(hold_id)

        async def allow_acquire(self) -> None:
            self._allow_acquire = True

        async def status(self) -> tuple[bool, bool, bool, list[int]]:
            return (
                self._acquire_started,
                self._acquire_finished,
                self._acquire_cancelled,
                sorted(self._active_holds),
            )

    events: list[str] = []
    rollout_manager = ray.remote(CancellationRaceRolloutManager).options(max_concurrency=4).remote()
    actor_model = RecordingActorModel(events, update_failure=None)
    patch_train_dependencies(monkeypatch, rollout_manager, actor_model, events)
    train_task = asyncio.create_task(train_async.train(make_args(num_rollout=1, update_weights_interval=1)))

    while not (await rollout_manager.status.remote())[0]:
        await asyncio.sleep(0)

    train_task.cancel()
    await rollout_manager.allow_acquire.remote()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(train_task, timeout=5)

    while not (status := await rollout_manager.status.remote())[1]:
        await asyncio.sleep(0)

    assert status == (True, True, False, [])
    assert events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_phase", ["record", "release"])
async def test_cancelled_weight_update_commit_settles_before_propagation(
    monkeypatch: pytest.MonkeyPatch,
    ray_local_mode: None,
    blocked_phase: str,
) -> None:
    class BlockingReleaseRolloutManager:
        def __init__(self, phase: str) -> None:
            self._blocked_phase = phase
            self._active_holds: set[int] = set()
            self._phase_started = False
            self._allow_release = False
            self._release_finished = False
            self._phase_cancelled = False
            self._record_finished = False

        async def acquire_train_admission_hold(self) -> int:
            self._active_holds.add(0)
            return 0

        async def wait_train_admission_hold(self, hold_id: int) -> None:
            assert hold_id in self._active_holds

        async def record_train_weight_update(self, hold_id: int) -> None:
            assert hold_id in self._active_holds
            if self._blocked_phase == "record":
                self._phase_started = True
                try:
                    while not self._allow_release:
                        await asyncio.sleep(0)
                except asyncio.CancelledError:
                    self._phase_cancelled = True
                    raise
            self._record_finished = True

        async def release_train_admission_hold(self, hold_id: int) -> None:
            assert self._record_finished
            if self._blocked_phase == "release":
                self._phase_started = True
                try:
                    while not self._allow_release:
                        await asyncio.sleep(0)
                except asyncio.CancelledError:
                    self._phase_cancelled = True
                    raise
            self._active_holds.remove(hold_id)
            self._release_finished = True

        async def allow_release(self) -> None:
            self._allow_release = True

        async def status(self) -> tuple[bool, bool, bool, list[int]]:
            return (
                self._phase_started,
                self._release_finished,
                self._phase_cancelled,
                sorted(self._active_holds),
            )

    events: list[str] = []
    rollout_manager = ray.remote(BlockingReleaseRolloutManager).options(max_concurrency=4).remote(blocked_phase)
    actor_model = RecordingActorModel(events, update_failure=None)
    patch_train_dependencies(monkeypatch, rollout_manager, actor_model, events)
    train_task = asyncio.create_task(train_async.train(make_args(num_rollout=1, update_weights_interval=1)))

    while not (await rollout_manager.status.remote())[0]:
        await asyncio.sleep(0)

    train_task.cancel()
    await asyncio.sleep(0)
    cancellation_waited_for_commit = not train_task.done()
    await rollout_manager.allow_release.remote()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(train_task, timeout=5)

    status = await rollout_manager.status.remote()
    assert cancellation_waited_for_commit is True
    assert status == (True, True, False, [])
    assert events == ["update:initial"]
