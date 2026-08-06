from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
import gc
from argparse import Namespace
from collections import deque
from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest

import miles.rollout.fully_async_rollout as fully_async
import miles.rollout.inference_rollout.fully_async as inference_fully_async
from miles.rollout.base_types import (
    LeasedRolloutFnTrainOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnLifecycle,
    RolloutFnTrainInput,
    TrainBatchRollbackReason,
)
from miles.rollout.data_source import SourceReservation, SourceReservationId
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.fully_async.execution import (
    FullyAsyncExecutionRetry,
    FullyAsyncExecutionSuccess,
    FullyAsyncRetryReason,
    FullyAsyncTerminalPendingError,
)
from miles.rollout.fully_async.ownership import ReservationExecutorReceipt, ReservationStageId
from miles.utils.async_utils import AsyncLoopThread
from miles.utils.types import Sample

N_SAMPLES_PER_PROMPT = 2


@pytest.fixture
def lifecycle_loop() -> Iterator[AsyncLoopThread]:
    loop_thread = AsyncLoopThread()
    try:
        yield loop_thread
    finally:
        loop_thread.loop.call_soon_threadsafe(loop_thread.loop.stop)
        loop_thread._thread.join(timeout=1)
        assert not loop_thread._thread.is_alive()
        loop_thread.loop.close()


class FakeGenerateState:
    def __init__(self, args):
        self.args = args
        self.sampling_params = {}
        self.aborted = False


class FakeDataSource:
    """Serves scripted groups first, then manufactures completed groups forever."""

    def __init__(self, scripted=None):
        self.scripted = deque(scripted or [])
        self.next_group_index = 1000
        self.recycled = []
        self.num_get_calls = 0

    def get_samples(self, num_samples):
        assert num_samples == 1
        self.num_get_calls += 1
        if self.scripted:
            return [self.scripted.popleft()]
        self.next_group_index += 1
        return [make_group(self.next_group_index)]

    def add_samples(self, groups):
        self.recycled.extend(groups)


class FakeReservationDataSource:
    def __init__(self, reservations: list[SourceReservation]) -> None:
        self.reservations = deque(reservations)
        self.reserved: list[SourceReservation] = []
        self.acknowledged: list[tuple[list[SourceReservation], int]] = []
        self.requeued: list[list[SourceReservation]] = []
        self.next_group_index = 1000

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        raise AssertionError("owned fully async rollout must reserve source groups")

    def add_samples(self, groups: list[list[Sample]]) -> None:
        raise AssertionError("owned fully async rollout must settle source reservations")

    def reserve_samples(self, num_groups: int) -> list[SourceReservation]:
        assert num_groups == 1
        if self.reservations:
            reservation = self.reservations.popleft()
        else:
            self.next_group_index += 1
            reservation = SourceReservation(
                reservation_id=SourceReservationId(f"source-{self.next_group_index}"),
                samples=tuple(make_group(self.next_group_index)),
            )
        self.reserved.append(reservation)
        return [reservation]

    def acknowledge_reservations(
        self,
        reservations: Sequence[SourceReservation],
        *,
        rollout_id: int,
    ) -> None:
        self.acknowledged.append((list(reservations), rollout_id))

    def requeue_reservations(self, reservations: Sequence[SourceReservation]) -> None:
        self.requeued.append(list(reservations))

    def save(self, rollout_id: int) -> None:
        pass

    def load(self, rollout_id: int | None = None) -> None:
        pass


def make_group(
    group_index: int,
    status: Sample.Status = Sample.Status.COMPLETED,
    weight_versions: list[str] | None = None,
) -> list[Sample]:
    return [
        Sample(
            group_index=group_index,
            index=group_index * 10 + i,
            prompt=f"prompt {group_index}",
            response="ok",
            response_length=1,
            label="ok",
            reward=1,
            status=status,
            weight_versions=list(weight_versions or []),
        )
        for i in range(N_SAMPLES_PER_PROMPT)
    ]


def make_reservation(group_index: int) -> SourceReservation:
    return SourceReservation(
        reservation_id=SourceReservationId(f"source-{group_index}"),
        samples=tuple(make_group(group_index)),
    )


async def wait_until(predicate) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)


def make_args(**overrides) -> Namespace:
    defaults = dict(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
        max_weight_staleness=None,
        async_max_concurrent_samples=None,
        rollout_submission_granularity=None,
        async_data_buffer_max_batches=0,
        async_data_buffer_order="fifo",
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        sglang_server_concurrency=8,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=1,
        rollout_health_check_timeout=0.1,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        eval_num_gpus=0,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class FakeWeightVersion:
    def __init__(self, value: int | None = None):
        self.value = value
        self.requests = 0
        self.refreshes = 0

    async def get(self, args) -> int | None:
        self.requests += 1
        return self.value

    async def refresh(self, args) -> int | None:
        self.refreshes += 1
        return await self.get(args)


def make_fn(monkeypatch, args, data_source, generate=None):
    async def default_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await asyncio.sleep(0)
        return group

    monkeypatch.setattr(fully_async, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(fully_async, "generate_and_rm_group", generate or default_generate)
    monkeypatch.setattr(inference_fully_async, "generate_and_rm_group", generate or default_generate)
    fn = fully_async.FullyAsyncRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
    # Staleness accounting queries the router on every drain; fake it out.
    fn._weight_version = FakeWeightVersion()
    return fn


def make_owned_fn(
    monkeypatch,
    data_source,
    generate=None,
    *,
    batch_size=1,
    execution_samples=2,
    retained_groups=1,
    completed_groups=1,
    **overrides,
):
    overrides.setdefault("rollout_submission_granularity", "group")
    return make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=batch_size,
            fully_async_max_execution_samples=execution_samples,
            fully_async_max_retained_groups=retained_groups,
            fully_async_max_completed_prefetch_groups=completed_groups,
            **overrides,
        ),
        data_source,
        generate=generate,
    )


async def test_train_call_leases_one_to_many_output_by_parent_group(monkeypatch):
    reservation = make_reservation(1)
    first_parent, second_parent = reservation.samples
    first_child = deepcopy(first_parent)
    first_child.response = "first child"
    second_child = deepcopy(first_parent)
    second_child.response = "second child"
    completed_second_parent = deepcopy(second_parent)
    generated_group = [[first_child, second_child], completed_second_parent]
    data_source = FakeReservationDataSource([reservation])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        assert [(sample.group_index, sample.index) for sample in group] == [(1, 10), (1, 11)]
        return generated_group

    fn = make_owned_fn(monkeypatch, data_source, generate)

    output = await fn(RolloutFnTrainInput(rollout_id=17))

    assert output == LeasedRolloutFnTrainOutput(
        samples=[generated_group],
        metrics={
            "rollout/fully_async/queue_size": 0,
            "rollout/fully_async/aborted_groups_recycled": 0,
            "rollout/fully_async/stale_groups_recycled": 0,
            "rollout/fully_async/evicted_stale_groups": 0,
            "rollout/fully_async/evicted_overflow_groups": 0,
            "rollout/fully_async/evict_rate": 0.0,
        },
        lease=output.lease,
    )
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    output.lease.commit()

    assert data_source.acknowledged == [([reservation], 17)]
    assert data_source.requeued == []


async def test_owned_admission_rejects_missing_parent_identity(monkeypatch):
    reservation = make_reservation(37)
    reservation.samples[0].index = None
    data_source = FakeReservationDataSource([reservation])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        raise AssertionError("generation must not start for an invalid source reservation")

    fn = make_owned_fn(monkeypatch, data_source, generate)

    with pytest.raises(ValueError) as error:
        await fn(RolloutFnTrainInput(rollout_id=37))

    assert str(error.value) == "Source reservation source-37 has incomplete parent identity (37, None) at slot 0."
    assert data_source.reserved == [reservation]
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation


async def test_owned_admission_rejects_duplicate_parent_identities(monkeypatch):
    reservation = make_reservation(38)
    reservation.samples[1].index = reservation.samples[0].index
    data_source = FakeReservationDataSource([reservation])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        raise AssertionError("generation must not start for an invalid source reservation")

    fn = make_owned_fn(monkeypatch, data_source, generate)

    with pytest.raises(ValueError) as error:
        await fn(RolloutFnTrainInput(rollout_id=38))

    assert str(error.value) == (
        "Source reservation source-38 has duplicate parent identities: [(38, 380), (38, 380)]."
    )
    assert data_source.reserved == [reservation]
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation


async def test_terminal_failure_requeues_exact_source_reservation(monkeypatch):
    reservation = make_reservation(2)
    data_source = FakeReservationDataSource([reservation])
    failure = RuntimeError("generation failed")

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        raise failure

    fn = make_owned_fn(monkeypatch, data_source, generate)

    with pytest.raises(RuntimeError) as error:
        await fn(RolloutFnTrainInput(rollout_id=18))

    assert error.value is failure
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]


async def test_mismatched_execution_receipt_retains_ownership_fail_closed(monkeypatch):
    reservation = make_reservation(50)
    data_source = FakeReservationDataSource([reservation])
    release_terminal = asyncio.Event()

    class MismatchedReceiptExecution:
        def __init__(self, executor_receipt: ReservationExecutorReceipt) -> None:
            self.executor_receipt = executor_receipt
            self.cancellation_requests = 0

        def request_cancellation(self) -> None:
            self.cancellation_requests += 1

        async def wait_terminal(self) -> FullyAsyncExecutionSuccess:
            await release_terminal.wait()
            return FullyAsyncExecutionSuccess(
                executor_receipt=replace(self.executor_receipt),
                samples=[deepcopy(sample) for sample in reservation.samples],
            )

    fn = make_owned_fn(monkeypatch, data_source)
    assert fn._executor is not None
    executions: list[MismatchedReceiptExecution] = []

    def submit(
        source_reservation: SourceReservation,
        executor_receipt: ReservationExecutorReceipt,
    ) -> MismatchedReceiptExecution:
        assert source_reservation is reservation
        execution = MismatchedReceiptExecution(executor_receipt)
        executions.append(execution)
        return execution

    monkeypatch.setattr(fn._executor, "submit", submit)

    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=50)))
    await wait_until(lambda: len(executions) == 1)
    hold = await fn.acquire_train_admission_hold()
    release_terminal.set()

    with pytest.raises(RuntimeError) as terminal_error:
        await asyncio.wait_for(hold.wait_terminal(), timeout=1)
    with pytest.raises(RuntimeError) as train_error:
        await train

    assert terminal_error.value is train_error.value
    assert str(train_error.value) == "Execution receipt 0 did not return its exact terminal receipt."
    assert data_source.reserved == [reservation]
    assert data_source.acknowledged == []
    assert data_source.requeued == []
    assert len(fn._active_executions) == 1

    with pytest.raises(RuntimeError) as close_error:
        await fn.close()

    assert close_error.value is train_error.value
    assert executions[0].cancellation_requests == 0
    assert data_source.acknowledged == []
    assert data_source.requeued == []


async def test_terminal_failure_drains_and_requeues_active_siblings(monkeypatch):
    reservations = [make_reservation(index) for index in range(26, 29)]
    data_source = FakeReservationDataSource(reservations)
    all_started = asyncio.Event()
    release_successes = asyncio.Event()
    abort_requested = asyncio.Event()
    started: list[int] = []
    failure = RuntimeError("generation failed")

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = group[0].group_index
        started.append(group_index)
        if len(started) == 3:
            all_started.set()
        await all_started.wait()
        if group_index == 26:
            raise failure
        await release_successes.wait()
        return group

    async def request_abort(args) -> None:
        abort_requested.set()

    monkeypatch.setattr(inference_fully_async, "request_abort", request_abort)
    fn = make_owned_fn(monkeypatch, data_source, generate, execution_samples=6, retained_groups=3)
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=26)))
    await all_started.wait()

    try:
        await asyncio.wait_for(abort_requested.wait(), timeout=0.01)
        for _ in range(10):
            await asyncio.sleep(0)
        finished_before_siblings = drain.done()
    finally:
        release_successes.set()

    with pytest.raises(RuntimeError) as error:
        await drain
    for _ in range(10):
        await asyncio.sleep(0)

    requeued_reservations = sorted(
        [reservation for batch in data_source.requeued for reservation in batch],
        key=lambda reservation: reservation.reservation_id,
    )

    assert error.value is failure
    assert not finished_before_siblings
    assert data_source.acknowledged == []
    assert requeued_reservations == reservations
    assert all(
        actual is expected
        for actual, expected in zip(
            requeued_reservations,
            reservations,
            strict=True,
        )
    )


async def test_close_resurfaces_recorded_worker_failure_after_cancelling_worker(monkeypatch):
    reservations = [make_reservation(index) for index in range(80, 82)]
    data_source = FakeReservationDataSource(reservations)
    all_started = asyncio.Event()
    release_sibling = asyncio.Event()
    abort_requested = asyncio.Event()
    started: list[int] = []
    generation_error = RuntimeError("generation failed before close")

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = group[0].group_index
        started.append(group_index)
        if len(started) == 2:
            all_started.set()
        await all_started.wait()
        if group_index == 80:
            raise generation_error
        await release_sibling.wait()
        return group

    async def request_abort(args) -> None:
        abort_requested.set()

    monkeypatch.setattr(inference_fully_async, "request_abort", request_abort)
    fn = make_owned_fn(monkeypatch, data_source, generate, execution_samples=4, retained_groups=2)
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=80)))
    await all_started.wait()
    await abort_requested.wait()
    close = asyncio.create_task(fn.close())

    try:
        await asyncio.sleep(0)
        assert not close.done()
    finally:
        release_sibling.set()

    with pytest.raises(RuntimeError) as close_error:
        await close

    assert close_error.value is generation_error
    with pytest.raises(asyncio.CancelledError):
        await train
    assert data_source.acknowledged == []
    assert (
        sorted(
            (reservation for batch in data_source.requeued for reservation in batch),
            key=lambda reservation: reservation.reservation_id,
        )
        == reservations
    )

    await fn.close()
    assert fn._closed


async def test_submission_failure_drains_and_requeues_active_sibling(monkeypatch):
    reservation = make_reservation(31)
    data_source = FakeReservationDataSource([reservation])
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()
    second_reservation_attempted = asyncio.Event()
    failure = RuntimeError("reservation failed")

    original_reserve_samples = data_source.reserve_samples

    def reserve_samples(num_groups: int) -> list[SourceReservation]:
        if data_source.reserved:
            second_reservation_attempted.set()
            raise failure
        return original_reserve_samples(num_groups)

    data_source.reserve_samples = reserve_samples

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_started.set()
        await release_generation.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate, execution_samples=4, retained_groups=2)
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=31)))
    await generation_started.wait()
    await second_reservation_attempted.wait()
    for _ in range(10):
        await asyncio.sleep(0)

    try:
        assert not drain.done()
    finally:
        release_generation.set()

    with pytest.raises(RuntimeError) as error:
        await drain

    assert error.value is failure
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation


async def test_submission_failure_requeues_prefetched_terminal_group(monkeypatch):
    leased_reservation = make_reservation(35)
    prefetched_reservation = make_reservation(36)
    data_source = FakeReservationDataSource([leased_reservation, prefetched_reservation])
    release_prefetch = asyncio.Event()
    failure = RuntimeError("reservation failed")

    original_reserve_samples = data_source.reserve_samples

    def reserve_samples(num_groups: int) -> list[SourceReservation]:
        if len(data_source.reserved) == 2:
            raise failure
        return original_reserve_samples(num_groups)

    data_source.reserve_samples = reserve_samples

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        if group[0].group_index == 36:
            await release_prefetch.wait()
        return group

    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        retained_groups=3,
        completed_groups=3,
    )
    output = await fn(RolloutFnTrainInput(rollout_id=35))

    release_prefetch.set()
    await wait_until(lambda: fn._worker.done())

    with pytest.raises(RuntimeError) as error:
        fn._worker.result()

    assert error.value is failure
    assert data_source.acknowledged == []
    assert data_source.requeued == [[prefetched_reservation]]
    assert data_source.requeued[0][0] is prefetched_reservation

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)

    assert data_source.requeued == [[prefetched_reservation], [leased_reservation]]


async def test_close_retries_failed_requeue_after_executor_rejects_submission(monkeypatch):
    reservation = make_reservation(45)
    data_source = FakeReservationDataSource([reservation])
    submission_error = RuntimeError("submission rejected")
    requeue_error = RuntimeError("submission requeue failed")
    original_requeue = data_source.requeue_reservations
    requeue_attempts = 0

    def requeue_reservations(reservations: Sequence[SourceReservation]) -> None:
        nonlocal requeue_attempts
        requeue_attempts += 1
        if requeue_attempts == 1:
            raise requeue_error
        original_requeue(reservations)

    data_source.requeue_reservations = requeue_reservations
    fn = make_owned_fn(monkeypatch, data_source)
    assert fn._executor is not None

    def submit(reservation, executor_receipt):
        raise submission_error

    monkeypatch.setattr(fn._executor, "submit", submit)

    with pytest.raises(RuntimeError) as train_error:
        await fn(RolloutFnTrainInput(rollout_id=45))

    assert train_error.value is submission_error
    assert train_error.value.__cause__ is requeue_error
    assert data_source.requeued == []

    with pytest.raises(RuntimeError) as close_error:
        await fn.close()

    assert close_error.value is submission_error
    assert requeue_attempts == 2
    assert data_source.requeued == [[reservation]]

    await fn.close()


async def test_executor_rejection_does_not_charge_sample_backfill(monkeypatch):
    reservation = make_reservation(55)
    data_source = FakeReservationDataSource([reservation])
    submission_error = RuntimeError("submission rejected")
    fn = make_owned_fn(
        monkeypatch,
        data_source,
        rollout_submission_granularity="sample",
    )
    assert fn._executor is not None

    def submit(reservation, executor_receipt):
        raise submission_error

    monkeypatch.setattr(fn._executor, "submit", submit)

    with pytest.raises(RuntimeError) as train_error:
        await fn(RolloutFnTrainInput(rollout_id=55))

    assert train_error.value is submission_error
    assert fn._scheduler.samples_in_flight == 0
    assert data_source.requeued == [[reservation]]

    with pytest.raises(RuntimeError) as close_error:
        await fn.close()

    assert close_error.value is submission_error

    await fn.close()


async def test_failed_owned_buffer_handoff_restores_completed_capacity(monkeypatch):
    reservation = make_reservation(59)
    data_source = FakeReservationDataSource([reservation])
    buffer_error = RuntimeError("buffer handoff failed")
    fn = make_owned_fn(monkeypatch, data_source)

    async def reject_put(self, buffered_group, *, current_version=None):
        raise buffer_error

    monkeypatch.setattr(fully_async.DataBuffer, "put", reject_put)

    with pytest.raises(RuntimeError) as train_error:
        await fn(RolloutFnTrainInput(rollout_id=59))

    assert train_error.value is buffer_error
    assert data_source.requeued == [[reservation]]
    assert fn._active_executions == {}
    assert fn._pending_terminal_rollbacks == []
    assert fn._retained_slots is not None
    assert not fn._retained_slots.locked()
    assert fn._completed_slots is not None
    assert fn._completed_slots.qsize() == 1

    with pytest.raises(RuntimeError) as close_error:
        await fn.close()

    assert close_error.value is buffer_error

    await fn.close()


async def test_close_settles_pending_acquisition_rollback_before_releasing_capacity(monkeypatch):
    reservations = [make_reservation(56), make_reservation(57)]
    data_source = FakeReservationDataSource([])
    acquisition_requeue_error = RuntimeError("acquisition requeue failed")
    close_requeue_error = RuntimeError("close acquisition requeue failed")
    requeue_attempts = 0

    def reserve_samples(num_groups: int) -> list[SourceReservation]:
        assert num_groups == 1
        return reservations

    def requeue_reservations(requeued: Sequence[SourceReservation]) -> None:
        nonlocal requeue_attempts
        requeue_attempts += 1
        if requeue_attempts == 1:
            raise acquisition_requeue_error
        if requeue_attempts == 2:
            raise close_requeue_error
        data_source.requeued.append(list(requeued))

    data_source.reserve_samples = reserve_samples
    data_source.requeue_reservations = requeue_reservations
    fn = make_owned_fn(monkeypatch, data_source)

    with pytest.raises(RuntimeError) as train_error:
        await fn(RolloutFnTrainInput(rollout_id=56))

    assert train_error.value.__cause__ is acquisition_requeue_error
    assert fn._retained_slots is not None
    assert fn._retained_slots.locked()

    with pytest.raises(RuntimeError) as first_close_error:
        await fn.close()

    assert first_close_error.value is close_requeue_error
    assert requeue_attempts == 2
    assert fn._retained_slots.locked()
    assert not fn._closed

    with pytest.raises(RuntimeError) as second_close_error:
        await fn.close()

    assert second_close_error.value is train_error.value
    assert requeue_attempts == 3
    assert data_source.requeued == [reservations]
    assert not fn._retained_slots.locked()
    assert not fn._closed

    await fn.close()

    assert fn._closed


async def test_close_retries_failed_reserved_validation_rollback_before_releasing_capacity(monkeypatch):
    group = make_group(58)
    reservation = SourceReservation(
        reservation_id=SourceReservationId("source-58"),
        samples=(group[0],),
    )
    data_source = FakeReservationDataSource([reservation])
    validation_requeue_error = RuntimeError("validation requeue failed")
    close_requeue_error = RuntimeError("close validation requeue failed")
    requeue_attempts = 0

    def requeue_reservations(requeued: Sequence[SourceReservation]) -> None:
        nonlocal requeue_attempts
        requeue_attempts += 1
        if requeue_attempts == 1:
            raise validation_requeue_error
        if requeue_attempts == 2:
            raise close_requeue_error
        data_source.requeued.append(list(requeued))

    data_source.requeue_reservations = requeue_reservations
    fn = make_owned_fn(monkeypatch, data_source)

    with pytest.raises(ValueError) as train_error:
        await fn(RolloutFnTrainInput(rollout_id=58))

    assert str(train_error.value) == "Source reservation source-58 contains 1 parent slots; expected 2."
    assert train_error.value.__cause__ is validation_requeue_error
    assert fn._retained_slots is not None
    assert fn._retained_slots.locked()

    with pytest.raises(RuntimeError) as first_close_error:
        await fn.close()

    assert first_close_error.value is close_requeue_error
    assert requeue_attempts == 2
    assert data_source.requeued == []
    assert fn._retained_slots.locked()
    assert not fn._closed

    with pytest.raises(ValueError) as second_close_error:
        await fn.close()

    assert second_close_error.value is train_error.value
    assert requeue_attempts == 3
    assert data_source.requeued == [[reservation]]
    assert not fn._retained_slots.locked()
    assert not fn._closed

    await fn.close()

    assert fn._closed


async def test_terminal_local_execution_cancellation_requeues_exact_source_reservation(monkeypatch):
    reservation = make_reservation(29)
    data_source = FakeReservationDataSource([reservation])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        raise asyncio.CancelledError()

    fn = make_owned_fn(monkeypatch, data_source, generate)

    with pytest.raises(asyncio.CancelledError):
        await fn(RolloutFnTrainInput(rollout_id=29))

    assert data_source.reserved == [reservation]
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation


async def test_cancelled_train_waiter_does_not_consume_next_completed_group(monkeypatch):
    reservation = make_reservation(39)
    data_source = FakeReservationDataSource([reservation])
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_started.set()
        await release_generation.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate)
    cancelled_train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=39)))
    await generation_started.wait()

    cancelled_train.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_train

    release_generation.set()
    output = await asyncio.wait_for(fn(RolloutFnTrainInput(rollout_id=40)), timeout=1)

    assert output.samples == [list(reservation.samples)]
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)

    assert data_source.requeued == [[reservation]]


async def test_legacy_claimed_group_survives_output_queue_refill(monkeypatch):
    first_group = make_group(52)
    second_group = make_group(53)
    first_buffered_group = (first_group, first_group)
    second_buffered_group = (second_group, second_group)
    fn = make_fn(monkeypatch, make_args(), FakeDataSource())
    output = fully_async.DataBuffer(
        order="fifo",
        blocking_capacity=fully_async.OUTPUT_QUEUE_MAX_GROUPS,
        max_groups=None,
        max_staleness=None,
        on_evict=fn._recycle_buffer_source,
    )
    await output.put(first_buffered_group)
    fn._output = output
    hold_worker = asyncio.Event()
    worker = asyncio.create_task(hold_worker.wait())
    fn._worker = worker
    queue_get = asyncio.create_task(output.get())
    await queue_get
    await output.put(second_buffered_group)

    try:
        await fn._settle_failed_queue_get(queue_get, output)

        assert await fn._next_group() is first_buffered_group
        assert await fn._next_group() is second_buffered_group
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


async def test_worker_failure_wins_when_output_queue_completes_in_the_same_wait(monkeypatch):
    reservation = make_reservation(59)
    data_source = FakeReservationDataSource([reservation])
    fn = make_owned_fn(monkeypatch, data_source)
    ownership = fn._ownership
    retained_slots = fn._retained_slots
    completed_slots = fn._completed_slots
    assert ownership is not None
    assert retained_slots is not None
    assert completed_slots is not None

    await retained_slots.acquire()
    [reserved] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("execution-tie")
    [executor_receipt] = ownership.begin_execution([reserved], stage_id=stage_id)
    [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)
    completed_slots.get_nowait()
    output = fully_async.DataBuffer(
        order="fifo",
        blocking_capacity=fully_async.OUTPUT_QUEUE_MAX_GROUPS,
        max_groups=None,
        max_staleness=None,
        on_evict=fn._recycle_buffer_source,
    )
    completed = fully_async._OwnedCompletedGroup(
        terminal_receipt=terminal_receipt,
        samples=list(reservation.samples),
        expected_parent_identities=tuple((sample.group_index, sample.index) for sample in reservation.samples),
    )
    await output.put((completed, completed.samples))
    fn._output = output
    failure = RuntimeError("worker failed with output ready")

    async def fail_worker() -> None:
        raise failure

    worker = asyncio.create_task(fail_worker())
    await asyncio.sleep(0)
    fn._worker = worker

    async def complete_both(fs, **kwargs):
        await asyncio.gather(*fs, return_exceptions=True)
        return set(fs), set()

    monkeypatch.setattr(fully_async.asyncio, "wait", complete_both)

    with pytest.raises(RuntimeError) as next_group_error:
        await fn._next_group()

    assert next_group_error.value is failure
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert not retained_slots.locked()
    assert completed_slots.qsize() == 1

    fn._worker_failure_reported = True
    await fn.close()


async def test_close_waits_for_terminal_late_success_before_requeue(monkeypatch):
    reservation = make_reservation(40)
    data_source = FakeReservationDataSource([reservation])
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()
    abort_requested = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_started.set()
        await release_generation.wait()
        return group

    async def request_abort(args) -> None:
        abort_requested.set()

    monkeypatch.setattr(inference_fully_async, "request_abort", request_abort)
    fn = make_owned_fn(monkeypatch, data_source, generate)
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=40)))
    await generation_started.wait()
    close = None

    try:
        close = asyncio.create_task(fn.close())
        await abort_requested.wait()
        await asyncio.sleep(0)

        assert not close.done()
        assert data_source.acknowledged == []
        assert data_source.requeued == []

        close.cancel()
        await asyncio.sleep(0)
        assert not close.done()
        assert data_source.requeued == []

        release_generation.set()
        with pytest.raises(asyncio.CancelledError):
            await close
    finally:
        release_generation.set()
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        if fn._worker is not None and not fn._worker.done():
            fn._worker.cancel()
        train.cancel()
        await asyncio.gather(train, fn._worker, return_exceptions=True)

    with pytest.raises(asyncio.CancelledError):
        await train
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation

    await fn.close()


async def test_close_waits_for_legacy_in_flight_generation(monkeypatch):
    generation_started = asyncio.Event()
    cancellation_received = asyncio.Event()
    release_generation = asyncio.Event()
    generation_tasks: list[asyncio.Task[list[Sample]]] = []

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_task = asyncio.current_task()
        assert generation_task is not None
        generation_tasks.append(generation_task)
        generation_started.set()
        try:
            await release_generation.wait()
        except asyncio.CancelledError:
            cancellation_received.set()
            await release_generation.wait()
        return group

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1),
        FakeDataSource(),
        generate=generate,
    )
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=48)))
    await generation_started.wait()
    close = asyncio.create_task(fn.close())

    try:
        await asyncio.wait_for(cancellation_received.wait(), timeout=0.01)
        await asyncio.sleep(0)
        assert not close.done()
    finally:
        release_generation.set()
        await asyncio.gather(close, train, *generation_tasks, return_exceptions=True)

    await close
    with pytest.raises(asyncio.CancelledError):
        await train
    assert fn._closed


async def test_close_recycles_unconsumed_legacy_active_and_prefetched_groups(monkeypatch):
    active_group = make_group(54)
    cancelled_group = make_group(56)
    prefetched_group = make_group(55)
    claimed_group = make_group(58)
    data_source = FakeDataSource(scripted=[active_group, cancelled_group])
    cancellation_received = asyncio.Event()
    all_started = asyncio.Event()
    started: list[int] = []

    async def finish_after_cancellation(
        state,
        group,
        sampling_params,
        evaluation=False,
        sample_done_callback=None,
    ):
        started.append(group[0].group_index)
        if len(started) == 2:
            all_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            if group is active_group:
                cancellation_received.set()
                return group
            raise

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1),
        data_source,
        generate=finish_after_cancellation,
    )
    output = fully_async.DataBuffer(
        order="fifo",
        blocking_capacity=fully_async.OUTPUT_QUEUE_MAX_GROUPS,
        max_groups=None,
        max_staleness=None,
        on_evict=fn._recycle_buffer_source,
    )
    await output.put((prefetched_group, prefetched_group))
    fn._output = output
    fn._legacy_requeued_groups.append((claimed_group, claimed_group))
    fn._submit_one_group()
    fn._submit_one_group()
    await all_started.wait()

    await fn.close()

    assert cancellation_received.is_set()
    assert data_source.recycled == [active_group, cancelled_group, prefetched_group, claimed_group]
    assert fn._closed


async def test_close_recycles_legacy_group_blocked_on_output_put(monkeypatch):
    prefetched_group = make_group(59)
    blocked_group = make_group(60)
    data_source = FakeDataSource(scripted=[blocked_group])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)
    output = fully_async.DataBuffer(
        order="fifo",
        blocking_capacity=1,
        max_groups=None,
        max_staleness=None,
        on_evict=fn._recycle_buffer_source,
    )
    await output.put((prefetched_group, prefetched_group))
    fn._output = output
    fn._worker = asyncio.create_task(fn._worker_loop())
    await wait_until(lambda: data_source.num_get_calls == 1 and not fn._legacy_executions)

    await fn.close()

    assert data_source.recycled == [prefetched_group, blocked_group]
    assert fn._closed


async def test_close_recycles_legacy_partial_drain(monkeypatch):
    first_group = make_group(61)
    second_group = make_group(62)
    data_source = FakeDataSource(scripted=[first_group, second_group])
    release_second = asyncio.Event()
    first_claimed = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        if group[0].group_index == 62:
            await release_second.wait()
        return group

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=2, async_max_concurrent_samples=2),
        data_source,
        generate=generate,
    )
    original_next_group = fn._next_group

    async def next_group():
        buffered_group = await original_next_group()
        if buffered_group[1] is first_group:
            first_claimed.set()
        return buffered_group

    monkeypatch.setattr(fn, "_next_group", next_group)
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=61)))
    await first_claimed.wait()
    await asyncio.sleep(0)

    await fn.close()

    with pytest.raises(asyncio.CancelledError):
        await train
    assert sorted(group[0].group_index for group in data_source.recycled) == [61, 62]
    assert fn._closed


async def test_close_retries_failed_legacy_recycle(monkeypatch):
    prefetched_group = make_group(63)
    data_source = FakeDataSource()
    recycle_error = RuntimeError("legacy recycle failed")
    recycle_attempts = 0

    def add_samples(groups):
        nonlocal recycle_attempts
        recycle_attempts += 1
        if recycle_attempts == 1:
            raise recycle_error
        data_source.recycled.extend(groups)

    data_source.add_samples = add_samples
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)
    output = fully_async.DataBuffer(
        order="fifo",
        blocking_capacity=fully_async.OUTPUT_QUEUE_MAX_GROUPS,
        max_groups=None,
        max_staleness=None,
        on_evict=fn._recycle_buffer_source,
    )
    await output.put((prefetched_group, prefetched_group))
    fn._output = output

    with pytest.raises(RuntimeError) as first_close_error:
        await fn.close()

    assert first_close_error.value is recycle_error
    assert data_source.recycled == []
    assert not fn._closed

    await fn.close()

    assert recycle_attempts == 2
    assert data_source.recycled == [prefetched_group]
    assert fn._closed


async def test_close_does_not_abort_completed_terminal_observation(monkeypatch):
    reservation = make_reservation(48)
    data_source = FakeReservationDataSource([reservation])
    abort_requests = 0

    async def request_abort(args) -> None:
        nonlocal abort_requests
        abort_requests += 1

    monkeypatch.setattr(inference_fully_async, "request_abort", request_abort)
    fn = make_owned_fn(monkeypatch, data_source)
    retained_slots = fn._retained_slots
    assert retained_slots is not None
    await retained_slots.acquire()

    terminal_task = fn._submit_one_group()
    await terminal_task
    await fn.close()

    assert abort_requests == 0
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation


async def test_close_reobserves_cancelled_terminal_observer(monkeypatch):
    reservation = make_reservation(49)
    data_source = FakeReservationDataSource([reservation])
    first_observation_started = asyncio.Event()

    class CancelOnceExecution:
        def __init__(self) -> None:
            self.executor_receipt: ReservationExecutorReceipt | None = None
            self.cancellation_requests = 0
            self.observation_attempts = 0

        def request_cancellation(self) -> None:
            self.cancellation_requests += 1

        async def wait_terminal(self) -> FullyAsyncExecutionRetry:
            self.observation_attempts += 1
            if self.observation_attempts == 1:
                first_observation_started.set()
                await asyncio.Future()
                raise AssertionError("cancelled terminal observation resumed")
            assert self.executor_receipt is not None
            return FullyAsyncExecutionRetry(
                executor_receipt=self.executor_receipt,
                reason=FullyAsyncRetryReason.CANCELLATION_REQUESTED,
            )

    execution = CancelOnceExecution()
    fn = make_owned_fn(monkeypatch, data_source)
    assert fn._executor is not None

    def submit(
        reservation: SourceReservation,
        executor_receipt: ReservationExecutorReceipt,
    ) -> CancelOnceExecution:
        execution.executor_receipt = executor_receipt
        return execution

    monkeypatch.setattr(fn._executor, "submit", submit)
    retained_slots = fn._retained_slots
    assert retained_slots is not None
    await retained_slots.acquire()
    terminal_task = fn._submit_one_group()
    await first_observation_started.wait()
    terminal_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await terminal_task

    await fn.close()

    assert execution.cancellation_requests == 1
    assert execution.observation_attempts == 2
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation


async def test_close_retries_terminal_observation_without_releasing_ownership(monkeypatch):
    reservation = make_reservation(41)
    data_source = FakeReservationDataSource([reservation])
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()
    abort_requests = 0

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_started.set()
        await release_generation.wait()
        return group

    async def request_abort(args) -> None:
        nonlocal abort_requests
        abort_requests += 1

    monkeypatch.setattr(inference_fully_async, "request_abort", request_abort)
    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        rollout_health_check_timeout=0.01,
    )
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=41)))
    await generation_started.wait()

    try:
        with pytest.raises(FullyAsyncTerminalPendingError):
            await fn.close()

        assert abort_requests == 1
        assert data_source.acknowledged == []
        assert data_source.requeued == []

        release_generation.set()
        await fn.close()
    finally:
        release_generation.set()
        if fn._worker is not None and not fn._worker.done():
            fn._worker.cancel()
        train.cancel()
        await asyncio.gather(train, fn._worker, return_exceptions=True)

    assert abort_requests == 2
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation


async def test_close_retries_failed_prefetched_terminal_requeue(monkeypatch):
    reservation = make_reservation(42)
    data_source = FakeReservationDataSource([reservation])
    release_generation = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release_generation.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate)
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=42)))
    await wait_until(lambda: len(data_source.reserved) == 1)
    train.cancel()
    with pytest.raises(asyncio.CancelledError):
        await train

    release_generation.set()
    await wait_until(lambda: fn._output is not None and fn._output.qsize() == 1)
    requeue_error = RuntimeError("prefetched requeue failed")
    original_requeue = data_source.requeue_reservations
    requeue_attempts = 0

    def requeue_reservations(reservations: Sequence[SourceReservation]) -> None:
        nonlocal requeue_attempts
        requeue_attempts += 1
        if requeue_attempts == 1:
            raise requeue_error
        original_requeue(reservations)

    data_source.requeue_reservations = requeue_reservations

    with pytest.raises(RuntimeError) as first_close_error:
        await fn.close()

    assert first_close_error.value is requeue_error
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    await fn.close()

    assert requeue_attempts == 2
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation


async def test_close_waits_for_claimed_drain_and_retries_rollback(monkeypatch):
    reservation = make_reservation(51)
    data_source = FakeReservationDataSource([reservation])
    requeue_error = RuntimeError("claimed drain requeue failed")
    original_requeue = data_source.requeue_reservations
    requeue_attempts = 0

    def requeue_reservations(reservations: Sequence[SourceReservation]) -> None:
        nonlocal requeue_attempts
        requeue_attempts += 1
        if requeue_attempts == 1:
            raise requeue_error
        original_requeue(reservations)

    data_source.requeue_reservations = requeue_reservations
    fn = make_owned_fn(monkeypatch, data_source)
    claimed = asyncio.Event()
    release_claimed = asyncio.Event()
    original_next_group = fn._next_group

    async def next_group():
        completed = await original_next_group()
        claimed.set()
        await release_claimed.wait()
        return completed

    monkeypatch.setattr(fn, "_next_group", next_group)
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=51)))
    await claimed.wait()
    close = asyncio.create_task(fn.close())

    try:
        await asyncio.sleep(0)
        assert not close.done()
    finally:
        release_claimed.set()
        await asyncio.gather(close, train, return_exceptions=True)

    with pytest.raises(RuntimeError) as train_error:
        await train
    await close

    assert str(train_error.value) == "Fully async rollout closed before the train batch lease was issued."
    assert train_error.value.__cause__ is requeue_error
    assert requeue_attempts == 2
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation
    assert fn._pending_terminal_rollbacks == []
    assert fn._closed


async def test_cancelled_claimed_waiter_retains_failed_rollback_for_close(monkeypatch):
    reservation = make_reservation(43)
    data_source = FakeReservationDataSource([reservation])
    fn = make_owned_fn(monkeypatch, data_source)
    original_wait = asyncio.wait
    queue_get_completed = asyncio.Event()
    hold_wait_result = asyncio.Event()

    async def gated_wait(fs, **kwargs):
        done, pending = await original_wait(fs, **kwargs)
        if fn._worker is not None and fn._worker in fs and any(task is not fn._worker for task in done):
            queue_get_completed.set()
            await hold_wait_result.wait()
        return done, pending

    monkeypatch.setattr(fully_async.asyncio, "wait", gated_wait)
    requeue_error = RuntimeError("claimed rollback failed")
    original_requeue = data_source.requeue_reservations
    requeue_attempts = 0

    def requeue_reservations(reservations: Sequence[SourceReservation]) -> None:
        nonlocal requeue_attempts
        requeue_attempts += 1
        if requeue_attempts == 1:
            raise requeue_error
        original_requeue(reservations)

    data_source.requeue_reservations = requeue_reservations
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=43)))
    await queue_get_completed.wait()
    train.cancel()

    with pytest.raises(asyncio.CancelledError) as cancellation:
        await train

    assert cancellation.value.__cause__ is requeue_error
    assert requeue_attempts == 1
    assert data_source.requeued == []

    monkeypatch.setattr(fully_async.asyncio, "wait", original_wait)
    await fn.close()

    assert requeue_attempts == 2
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation


async def test_cancelled_partial_drain_retains_failed_batch_rollback_for_close(monkeypatch):
    reservations = [make_reservation(46), make_reservation(47)]
    data_source = FakeReservationDataSource(reservations)
    release_second = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        if group[0].group_index == 47:
            await release_second.wait()
        return group

    async def request_abort(args) -> None:
        pass

    monkeypatch.setattr(inference_fully_async, "request_abort", request_abort)
    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        batch_size=2,
        execution_samples=4,
        retained_groups=2,
        completed_groups=2,
    )
    first_group_claimed = asyncio.Event()
    original_next_group = fn._next_group

    async def next_group():
        completed = await original_next_group()
        if not first_group_claimed.is_set():
            first_group_claimed.set()
        return completed

    monkeypatch.setattr(fn, "_next_group", next_group)
    requeue_error = RuntimeError("partial drain rollback failed")
    original_requeue = data_source.requeue_reservations
    requeue_attempts = 0

    def requeue_reservations(requeued: Sequence[SourceReservation]) -> None:
        nonlocal requeue_attempts
        requeue_attempts += 1
        if requeue_attempts == 1:
            raise requeue_error
        original_requeue(requeued)

    data_source.requeue_reservations = requeue_reservations
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=46)))
    await first_group_claimed.wait()
    train.cancel()

    with pytest.raises(asyncio.CancelledError) as cancellation:
        await train

    assert cancellation.value.__cause__ is requeue_error
    assert data_source.requeued == []

    release_second.set()
    await fn.close()

    assert data_source.requeued == [[reservations[0]], [reservations[1]]]
    assert data_source.requeued[0][0] is reservations[0]
    assert data_source.requeued[1][0] is reservations[1]


async def test_close_retries_worker_terminal_requeue_before_resurfacing_failure(monkeypatch):
    reservation = make_reservation(44)
    data_source = FakeReservationDataSource([reservation])
    generation_error = RuntimeError("generation failed")
    requeue_error = RuntimeError("worker terminal requeue failed")
    original_requeue = data_source.requeue_reservations
    requeue_attempts = 0

    def requeue_reservations(reservations: Sequence[SourceReservation]) -> None:
        nonlocal requeue_attempts
        requeue_attempts += 1
        if requeue_attempts == 1:
            raise requeue_error
        original_requeue(reservations)

    data_source.requeue_reservations = requeue_reservations

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        raise generation_error

    fn = make_owned_fn(monkeypatch, data_source, generate)

    with pytest.raises(RuntimeError) as train_error:
        await fn(RolloutFnTrainInput(rollout_id=44))

    assert train_error.value is generation_error
    assert train_error.value.__cause__ is requeue_error
    assert requeue_attempts == 1
    assert data_source.requeued == []

    with pytest.raises(RuntimeError) as close_error:
        await fn.close()

    assert close_error.value is generation_error
    assert requeue_attempts == 2
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation

    await fn.close()


async def test_aborted_owned_group_requeues_pristine_reservation(monkeypatch):
    aborted_reservation = make_reservation(3)
    completed_reservation = make_reservation(4)
    data_source = FakeReservationDataSource([aborted_reservation, completed_reservation])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        if group[0].group_index == 3:
            for sample in group:
                sample.response = "aborted output"
                sample.status = Sample.Status.ABORTED
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate)

    output = await fn(RolloutFnTrainInput(rollout_id=19))

    assert isinstance(output, LeasedRolloutFnTrainOutput)
    assert output.samples == [list(completed_reservation.samples)]
    assert output.metrics == {
        "rollout/fully_async/queue_size": 0,
        "rollout/fully_async/aborted_groups_recycled": 1,
        "rollout/fully_async/stale_groups_recycled": 0,
        "rollout/fully_async/evicted_stale_groups": 0,
        "rollout/fully_async/evicted_overflow_groups": 0,
        "rollout/fully_async/evict_rate": 0.0,
    }
    assert data_source.acknowledged == []
    assert data_source.requeued == [[aborted_reservation]]
    assert [sample.response for sample in aborted_reservation.samples] == ["ok", "ok"]

    output.lease.commit()

    assert data_source.acknowledged == [([completed_reservation], 19)]


async def test_owned_group_rejects_missing_parent_slot_without_losing_reservation(monkeypatch):
    reservation = make_reservation(5)
    data_source = FakeReservationDataSource([reservation])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        return group[:1]

    fn = make_owned_fn(monkeypatch, data_source, generate)

    with pytest.raises(ValueError) as error:
        await fn(RolloutFnTrainInput(rollout_id=20))

    assert str(error.value) == "Source reservation source-5 returned 1 parent slots; expected 2."
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]


async def test_owned_group_rejects_reordered_parent_identity_without_losing_reservation(monkeypatch):
    reservation = make_reservation(24)
    data_source = FakeReservationDataSource([reservation])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        return [deepcopy(group[1]), deepcopy(group[0])]

    fn = make_owned_fn(monkeypatch, data_source, generate)
    output = None

    try:
        with pytest.raises(ValueError) as error:
            output = await fn(RolloutFnTrainInput(rollout_id=24))
    finally:
        if output is not None:
            output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)

    assert str(error.value) == (
        "Source reservation source-24 returned sample identities [(24, 241)] at parent slot 0; "
        "expected every sample to have identity (24, 240)."
    )
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation


async def test_owned_group_rejects_foreign_one_to_many_child_identity(monkeypatch):
    reservation = make_reservation(30)
    first_parent, second_parent = reservation.samples
    generated_group = [
        [deepcopy(first_parent), deepcopy(second_parent)],
        deepcopy(second_parent),
    ]
    data_source = FakeReservationDataSource([reservation])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        return generated_group

    fn = make_owned_fn(monkeypatch, data_source, generate)

    with pytest.raises(ValueError) as error:
        await fn(RolloutFnTrainInput(rollout_id=30))

    assert str(error.value) == (
        "Source reservation source-30 returned sample identities [(30, 300), (30, 301)] at parent slot 0; "
        "expected every sample to have identity (30, 300)."
    )
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation


async def test_owned_drain_failure_requeues_dequeued_terminal_receipt(monkeypatch):
    reservation = SourceReservation(
        reservation_id=SourceReservationId("source-25"),
        samples=tuple(make_group(25, weight_versions=["1"])),
    )
    data_source = FakeReservationDataSource([reservation])
    failure = RuntimeError("weight version lookup failed")
    fn = make_owned_fn(monkeypatch, data_source, max_weight_staleness=2)

    class FailingWeightVersion:
        async def get(self, args):
            raise failure

    fn._weight_version = FailingWeightVersion()

    with pytest.raises(RuntimeError) as error:
        await fn(RolloutFnTrainInput(rollout_id=25))

    assert error.value is failure
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation


async def test_owned_execution_capacity_bounds_started_source_samples(monkeypatch):
    release = asyncio.Event()
    started: list[int] = []
    data_source = FakeReservationDataSource([make_reservation(index) for index in range(6, 10)])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        started.append(group[0].group_index)
        await release.wait()
        return group

    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        execution_samples=4,
        retained_groups=4,
        completed_groups=4,
    )
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=21)))

    await wait_until(lambda: len(started) == 2)
    for _ in range(10):
        await asyncio.sleep(0)

    assert started == [6, 7]
    assert data_source.reserved == [make_reservation(6), make_reservation(7)]

    release.set()
    output = await drain
    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)


async def test_train_admission_hold_blocks_new_source_reservations_until_release(monkeypatch):
    reservation = make_reservation(54)
    data_source = FakeReservationDataSource([reservation])
    fn = make_owned_fn(monkeypatch, data_source)

    assert isinstance(fn, RolloutFnLifecycle)
    hold = await fn.acquire_train_admission_hold()
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=54)))
    for _ in range(10):
        await asyncio.sleep(0)

    assert (data_source.reserved, train.done()) == ([], False)

    hold.release()
    output = await asyncio.wait_for(train, timeout=1)

    assert output.samples == [list(reservation.samples)]

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    await fn.close()


async def test_overlapping_train_admission_holds_reopen_only_after_exact_releases(monkeypatch):
    reservation = make_reservation(57)
    data_source = FakeReservationDataSource([reservation])
    fn = make_owned_fn(monkeypatch, data_source)

    first_hold = await fn.acquire_train_admission_hold()
    second_hold = await fn.acquire_train_admission_hold()
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=57)))

    first_hold.release()
    for _ in range(10):
        await asyncio.sleep(0)

    assert (data_source.reserved, train.done()) == ([], False)

    second_hold.release()
    output = await asyncio.wait_for(train, timeout=1)

    assert output.samples == [list(reservation.samples)]

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    await fn.close()


async def test_train_admission_hold_waits_for_its_terminal_frontier_without_issuing_a_lease(monkeypatch):
    reservations = [make_reservation(55), make_reservation(56)]
    data_source = FakeReservationDataSource(reservations)
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_started.set()
        await release_generation.wait()
        return group

    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        retained_groups=2,
        completed_groups=2,
    )
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=55)))
    await generation_started.wait()

    hold = await fn.acquire_train_admission_hold()
    terminal = asyncio.create_task(hold.wait_terminal())
    for _ in range(10):
        await asyncio.sleep(0)

    assert (terminal.done(), data_source.reserved) == (False, reservations[:1])

    release_generation.set()
    await terminal
    assert fn._output is not None
    await wait_until(lambda: fn._output.qsize() == 0)

    assert (train.done(), data_source.reserved) == (False, reservations[:1])

    hold.release()
    output = await train
    await wait_until(lambda: data_source.reserved == reservations)

    assert output.samples == [list(reservations[0].samples)]

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    await fn.close()


async def test_train_admission_hold_does_not_consume_sample_wakes_while_ownership_blocks(monkeypatch):
    reservation = make_reservation(81)
    data_source = FakeReservationDataSource([reservation])
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()
    callbacks = []

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        callbacks.append(sample_done_callback)
        generation_started.set()
        await release_generation.wait()
        return group

    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        rollout_submission_granularity=None,
    )
    progress_waits = 0
    wait_for_progress = fn._scheduler.wait_for_progress

    async def count_progress_waits(pendings):
        nonlocal progress_waits
        progress_waits += 1
        return await wait_for_progress(pendings)

    fn._scheduler.wait_for_progress = count_progress_waits
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=81)))
    await generation_started.wait()
    for _ in range(10):
        await asyncio.sleep(0)

    assert progress_waits == 0

    hold = await fn.acquire_train_admission_hold()

    try:
        assert callbacks[0] is not None
        callbacks[0]()
        for _ in range(10):
            await asyncio.sleep(0)

        assert progress_waits == 0
    finally:
        hold.release()
        release_generation.set()

    output = await asyncio.wait_for(train, timeout=1)
    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    await fn.close()


async def test_train_admission_hold_settles_terminal_frontier_with_a_full_owned_buffer(monkeypatch):
    monkeypatch.setattr(fully_async, "OUTPUT_QUEUE_MAX_GROUPS", 1)
    reservations = [make_reservation(79), make_reservation(80)]
    data_source = FakeReservationDataSource(reservations)
    all_started = asyncio.Event()
    release_generation = asyncio.Event()
    started = 0

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        nonlocal started
        started += 1
        if started == len(reservations):
            all_started.set()
        await release_generation.wait()
        return group

    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        execution_samples=4,
        retained_groups=2,
        completed_groups=2,
    )
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=79)))
    await all_started.wait()
    hold = await fn.acquire_train_admission_hold()
    train.cancel()
    with pytest.raises(asyncio.CancelledError):
        await train

    release_generation.set()
    try:
        await asyncio.wait_for(hold.wait_terminal(), timeout=0.1)
        assert fn._output is not None
        assert fn._output.qsize() == 2
        assert fn._active_executions == {}
    finally:
        await fn.close()

    assert sorted(data_source.requeued, key=lambda batch: str(batch[0].reservation_id)) == [
        [reservations[0]],
        [reservations[1]],
    ]


async def test_train_admission_hold_preserves_legacy_frontier_compatibility(monkeypatch):
    data_source = FakeDataSource()
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_started.set()
        await release_generation.wait()
        return group

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1),
        data_source,
        generate=generate,
    )
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=67)))
    await generation_started.wait()
    hold = await fn.acquire_train_admission_hold()
    terminal = asyncio.create_task(hold.wait_terminal())
    await asyncio.sleep(0)

    assert (terminal.done(), data_source.num_get_calls) == (False, 1)

    release_generation.set()
    await terminal
    output = await train
    for _ in range(10):
        await asyncio.sleep(0)

    assert output.samples == [make_group(1001)]
    assert data_source.num_get_calls == 1

    hold.release()
    await wait_until(lambda: data_source.num_get_calls == 2)
    await fn.close()


async def test_legacy_admission_hold_does_not_wait_for_saturated_buffer_publication(monkeypatch):
    monkeypatch.setattr(fully_async, "OUTPUT_QUEUE_MAX_GROUPS", 1)
    data_source = FakeDataSource()
    started: list[int] = []
    generation_releases = {
        1001: asyncio.Event(),
        1002: asyncio.Event(),
    }

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = group[0].group_index
        started.append(group_index)
        await generation_releases[group_index].wait()
        return group

    fn = make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=1,
            async_max_concurrent_samples=2 * N_SAMPLES_PER_PROMPT,
            rollout_submission_granularity="group",
        ),
        data_source,
        generate=generate,
    )
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=68)))
    await wait_until(lambda: started == [1001, 1002])
    train.cancel()
    with pytest.raises(asyncio.CancelledError):
        await train

    assert fn._output is not None
    prefilled_group = make_group(90)
    await fn._output.put((prefilled_group, prefilled_group))
    generation_releases[1001].set()
    await wait_until(
        lambda: len(fn._legacy_executions) == 1 and next(iter(fn._legacy_executions.values()))[0].group_index == 1002
    )

    hold = await fn.acquire_train_admission_hold()
    generation_releases[1002].set()
    try:
        await asyncio.wait_for(hold.wait_terminal(), timeout=0.1)

        assert fn._output.qsize() == 1
        assert len(fn._legacy_executions) == 1
        assert next(iter(fn._legacy_executions)).done()

        published_groups = [(await fn._output.get())[1]]
        for _ in range(2):
            await wait_until(lambda: fn._output.qsize() == 1)
            published_groups.append((await fn._output.get())[1])

        assert [[sample.group_index for sample in group] for group in published_groups] == [
            [90, 90],
            [1001, 1001],
            [1002, 1002],
        ]
    finally:
        hold.release()
        await fn.close()


async def test_legacy_terminal_frontier_reports_the_canonical_worker_failure(monkeypatch):
    first_group = make_group(72)
    second_group = make_group(73)
    data_source = FakeDataSource(scripted=[first_group, second_group])
    all_started = asyncio.Event()
    release_first = asyncio.Event()
    release_second = asyncio.Event()
    first_cancelled = asyncio.Event()
    started: list[int] = []
    first_failure = RuntimeError("first captured legacy execution failed")
    canonical_failure = RuntimeError("second captured legacy execution failed first")

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = group[0].group_index
        started.append(group_index)
        if len(started) == 2:
            all_started.set()
        await all_started.wait()
        if group_index == 72:
            try:
                await release_first.wait()
            except asyncio.CancelledError:
                first_cancelled.set()
                await release_first.wait()
            raise first_failure
        await release_second.wait()
        raise canonical_failure

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=2),
        data_source,
        generate=generate,
    )
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=72)))
    await all_started.wait()
    hold = await fn.acquire_train_admission_hold()
    terminal = asyncio.create_task(hold.wait_terminal())

    release_second.set()
    await first_cancelled.wait()
    release_first.set()

    with pytest.raises(RuntimeError) as terminal_error:
        await terminal
    with pytest.raises(RuntimeError) as train_error:
        await train

    assert terminal_error.value is canonical_failure
    assert train_error.value is canonical_failure

    with pytest.raises(RuntimeError) as close_error:
        await fn.close()
    assert close_error.value is canonical_failure
    await fn.close()


async def test_train_admission_hold_fences_retained_capacity_waiter(monkeypatch):
    reservations = [make_reservation(58), make_reservation(59)]
    data_source = FakeReservationDataSource(reservations)
    fn = make_owned_fn(
        monkeypatch,
        data_source,
        retained_groups=1,
        completed_groups=2,
    )
    output = await fn(RolloutFnTrainInput(rollout_id=58))

    hold = await fn.acquire_train_admission_hold()
    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    for _ in range(10):
        await asyncio.sleep(0)

    assert data_source.reserved == reservations[:1]

    hold.release()
    await wait_until(lambda: data_source.reserved == reservations)
    await fn.close()


async def test_train_admission_hold_blocks_completed_batch_lease_until_checkpoint_prepares(monkeypatch):
    reservation = make_reservation(74)
    data_source = FakeReservationDataSource([reservation])
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_started.set()
        await release_generation.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate)
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=74)))
    await generation_started.wait()
    hold = await fn.acquire_train_admission_hold()

    release_generation.set()
    await hold.wait_terminal()
    assert fn._output is not None
    await wait_until(lambda: fn._output.qsize() == 0)

    assert train.done() is False
    assert await fn.prepare_checkpoint(rollout_id=74) is None

    hold.release()
    output = await asyncio.wait_for(train, timeout=1)

    assert output.samples == [list(reservation.samples)]
    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    assert data_source.requeued == [[reservation]]
    await fn.close()


async def test_train_admission_hold_fences_lease_during_buffer_metric_read(monkeypatch):
    reservation = make_reservation(78)
    data_source = FakeReservationDataSource([reservation])
    metric_read_started = asyncio.Event()
    release_metric_read = asyncio.Event()

    class BlockingWeightVersion:
        def __init__(self) -> None:
            self.requests = 0

        async def get(self, args) -> int | None:
            self.requests += 1
            if self.requests == 3:
                metric_read_started.set()
                await release_metric_read.wait()
            return None

        async def refresh(self, args) -> int | None:
            return await self.get(args)

    fn = make_owned_fn(monkeypatch, data_source)
    fn._weight_version = BlockingWeightVersion()
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=78)))
    await metric_read_started.wait()

    hold = await fn.acquire_train_admission_hold()
    release_metric_read.set()
    for _ in range(10):
        await asyncio.sleep(0)

    assert train.done() is False

    await hold.wait_terminal()
    hold.release()
    output = await asyncio.wait_for(train, timeout=1)

    assert output.samples == [list(reservation.samples)]

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    await fn.close()


async def test_train_admission_hold_revalidates_staleness_before_issuing_a_lease(monkeypatch):
    reservation = SourceReservation(
        reservation_id=SourceReservationId("source-76"),
        samples=tuple(make_group(76, weight_versions=["1"])),
    )
    data_source = FakeReservationDataSource([reservation])
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_started.set()
        await release_generation.wait()
        return group

    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        max_weight_staleness=0,
    )
    weight_version = FakeWeightVersion(1)
    fn._weight_version = weight_version
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=76)))
    await generation_started.wait()
    hold = await fn.acquire_train_admission_hold()

    release_generation.set()
    await hold.wait_terminal()
    # Publication performs the first query. Wait for the drain to validate the
    # group under this hold's admission epoch before releasing the hold.
    await wait_until(lambda: weight_version.requests >= 2)
    weight_version.value = 2
    hold.release()
    output = await asyncio.wait_for(train, timeout=1)

    assert output == LeasedRolloutFnTrainOutput(
        samples=[make_group(1001)],
        metrics={
            "rollout/fully_async/queue_size": 0,
            "rollout/fully_async/aborted_groups_recycled": 0,
            "rollout/fully_async/stale_groups_recycled": 1,
            "rollout/fully_async/evicted_stale_groups": 0,
            "rollout/fully_async/evicted_overflow_groups": 0,
            "rollout/fully_async/evict_rate": 0.0,
            "rollout/fully_async/avg_staleness": 0.5,
            "rollout/fully_async/max_staleness": 1,
        },
        lease=output.lease,
    )
    assert data_source.requeued == [[reservation]]
    assert weight_version.refreshes == 1

    fresh_reservation = data_source.reserved[1]
    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    assert data_source.requeued == [[reservation], [fresh_reservation]]
    await fn.close()


async def test_train_admission_hold_refreshes_cached_version_when_drain_starts_after_release(monkeypatch):
    reservation = SourceReservation(
        reservation_id=SourceReservationId("source-79"),
        samples=tuple(make_group(79, weight_versions=["1"])),
    )
    data_source = FakeReservationDataSource([reservation])
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()
    allow_drain = asyncio.Event()
    current_weight_version = 1
    weight_version_requests: list[str] = []

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_started.set()
        await release_generation.wait()
        return group

    async def get(url: str) -> dict[str, str]:
        weight_version_requests.append(url)
        return {"weight_version": str(current_weight_version)}

    monkeypatch.setattr(fully_async, "get", get)
    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        max_weight_staleness=0,
    )
    fn._weight_version = fully_async._CachedWeightVersion(ttl=60.0)
    next_group = fn._next_group

    async def gated_next_group():
        await allow_drain.wait()
        return await next_group()

    fn._next_group = gated_next_group
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=79)))
    await generation_started.wait()
    hold = await fn.acquire_train_admission_hold()

    release_generation.set()
    await hold.wait_terminal()
    assert fn._output is not None
    await wait_until(lambda: fn._output.qsize() == 1)

    current_weight_version = 2
    hold.release()
    allow_drain.set()
    output = await asyncio.wait_for(train, timeout=1)

    assert output == LeasedRolloutFnTrainOutput(
        samples=[make_group(1001)],
        metrics={
            "rollout/fully_async/queue_size": 0,
            "rollout/fully_async/aborted_groups_recycled": 0,
            "rollout/fully_async/stale_groups_recycled": 1,
            "rollout/fully_async/evicted_stale_groups": 0,
            "rollout/fully_async/evicted_overflow_groups": 0,
            "rollout/fully_async/evict_rate": 0.0,
            "rollout/fully_async/avg_staleness": 0.5,
            "rollout/fully_async/max_staleness": 1,
        },
        lease=output.lease,
    )
    assert data_source.requeued == [[reservation]]
    assert weight_version_requests == [
        "http://127.0.0.1:30000/model_info",
        "http://127.0.0.1:30000/model_info",
    ]

    fresh_reservation = data_source.reserved[1]
    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    assert data_source.requeued == [[reservation], [fresh_reservation]]
    await fn.close()


async def test_concurrent_drains_revalidate_their_groups_against_the_strict_epoch_version(monkeypatch):
    reservations = [
        SourceReservation(
            reservation_id=SourceReservationId(f"source-{group_index}"),
            samples=tuple(make_group(group_index, weight_versions=[weight_version])),
        )
        for group_index, weight_version in ((82, "1"), (83, "1"), (84, "2"), (85, "2"))
    ]
    data_source = FakeReservationDataSource(reservations)
    fn = make_owned_fn(
        monkeypatch,
        data_source,
        execution_samples=8,
        retained_groups=4,
        completed_groups=4,
        max_weight_staleness=0,
    )
    ownership = fn._ownership
    retained_slots = fn._retained_slots
    completed_slots = fn._completed_slots
    assert ownership is not None
    assert retained_slots is not None
    assert completed_slots is not None

    output = fully_async.DataBuffer(
        order="fifo",
        blocking_capacity=4,
        max_groups=None,
        max_staleness=0,
        on_evict=fn._recycle_buffer_source,
    )
    for index in range(4):
        await retained_slots.acquire()
        [reserved] = ownership.reserve_samples(1)
        stage_id = ReservationStageId(f"concurrent-drain-{index}")
        [executor_receipt] = ownership.begin_execution([reserved], stage_id=stage_id)
        [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)
        completed_slots.get_nowait()
        completed = fully_async._OwnedCompletedGroup(
            terminal_receipt=terminal_receipt,
            samples=list(reserved.samples),
            expected_parent_identities=tuple((sample.group_index, sample.index) for sample in reserved.samples),
        )
        await output.put((completed, completed.samples))
    fn._output = output
    worker_release = asyncio.Event()
    fn._worker = asyncio.create_task(worker_release.wait())

    class CoordinatedWeightVersion:
        def __init__(self) -> None:
            self.value = 1
            self.refreshes = 0
            self.gets_by_task: dict[asyncio.Task, int] = {}
            self.blocked_metric_task: asyncio.Task | None = None
            self.metric_read_started = asyncio.Event()
            self.release_metric_read = asyncio.Event()

        async def get(self, args) -> int:
            task = asyncio.current_task()
            assert task is not None
            request_count = self.gets_by_task.get(task, 0) + 1
            self.gets_by_task[task] = request_count
            value = self.value
            if request_count == 2 and self.blocked_metric_task is None:
                self.blocked_metric_task = task
                self.metric_read_started.set()
                await self.release_metric_read.wait()
            return value

        async def refresh(self, args) -> int:
            self.refreshes += 1
            self.value = 2
            self.release_metric_read.set()
            return self.value

    weight_version = CoordinatedWeightVersion()
    fn._weight_version = weight_version
    hold = await fn.acquire_train_admission_hold()
    hold.release()

    first_drain = asyncio.create_task(fn._drain(rollout_id=82))
    await weight_version.metric_read_started.wait()
    second_drain = asyncio.create_task(fn._drain(rollout_id=83))
    training_outputs = await asyncio.gather(first_drain, second_drain)

    leased_samples = sorted(
        (training_output.samples[0] for training_output in training_outputs),
        key=lambda group: group[0].group_index,
    )
    old_requeues = sorted(
        data_source.requeued,
        key=lambda requeued: requeued[0].samples[0].group_index,
    )
    assert leased_samples == [list(reservations[2].samples), list(reservations[3].samples)]
    assert old_requeues == [[reservations[0]], [reservations[1]]]
    assert weight_version.refreshes == 1

    for training_output in training_outputs:
        training_output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    all_requeues = sorted(
        data_source.requeued,
        key=lambda requeued: requeued[0].samples[0].group_index,
    )
    assert all_requeues == [[reservation] for reservation in reservations]
    await fn.close()


async def test_strict_epoch_snapshot_coalesces_concurrent_drain_refreshes(monkeypatch):
    fn = make_owned_fn(
        monkeypatch,
        FakeReservationDataSource([]),
        max_weight_staleness=0,
    )

    class BlockingWeightVersion:
        def __init__(self) -> None:
            self.refreshes = 0
            self.refresh_started = asyncio.Event()
            self.release_refresh = asyncio.Event()

        async def get(self, args) -> int:
            return 1

        async def refresh(self, args) -> int:
            self.refreshes += 1
            self.refresh_started.set()
            await self.release_refresh.wait()
            return 2

    weight_version = BlockingWeightVersion()
    fn._weight_version = weight_version
    hold = await fn.acquire_train_admission_hold()
    hold.release()
    admission_epoch = fn._train_admission_epoch

    first_refresh = asyncio.create_task(fn._strict_weight_version_for_admission_epoch(admission_epoch))
    await weight_version.refresh_started.wait()
    second_refresh = asyncio.create_task(fn._strict_weight_version_for_admission_epoch(admission_epoch))
    for _ in range(10):
        await asyncio.sleep(0)

    refreshes_while_blocked = weight_version.refreshes
    second_blocked = not second_refresh.done()
    weight_version.release_refresh.set()
    results = await asyncio.gather(first_refresh, second_refresh)

    assert (results, refreshes_while_blocked, weight_version.refreshes, second_blocked) == (
        [(2, True), (2, True)],
        1,
        1,
        True,
    )
    await fn.close()


async def test_train_admission_hold_fails_closed_when_numeric_staleness_refresh_fails(monkeypatch):
    reservation = SourceReservation(
        reservation_id=SourceReservationId("source-77"),
        samples=tuple(make_group(77, weight_versions=["1"])),
    )
    data_source = FakeReservationDataSource([reservation])
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()
    failure = httpx.ConnectError("router unavailable during admission revalidation")
    weight_version_requests: list[str] = []
    validation_read = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_started.set()
        await release_generation.wait()
        return group

    async def get(url: str) -> dict[str, str]:
        weight_version_requests.append(url)
        if len(weight_version_requests) == 1:
            return {"weight_version": "1"}
        raise failure

    class ObservedWeightVersion(fully_async._CachedWeightVersion):
        def __init__(self) -> None:
            super().__init__(ttl=60.0)
            self.get_calls = 0

        async def get(self, args) -> int | None:
            value = await super().get(args)
            self.get_calls += 1
            if self.get_calls == 2:
                validation_read.set()
            return value

    monkeypatch.setattr(fully_async, "get", get)
    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        max_weight_staleness=0,
    )
    fn._weight_version = ObservedWeightVersion()
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=77)))
    await generation_started.wait()
    hold = await fn.acquire_train_admission_hold()

    release_generation.set()
    await hold.wait_terminal()
    await validation_read.wait()
    hold.release()

    output = None
    try:
        with pytest.raises(httpx.ConnectError) as error:
            output = await asyncio.wait_for(train, timeout=1)
    finally:
        if output is not None:
            output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
        await fn.close()

    assert error.value is failure
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation], [data_source.reserved[1]]]
    assert weight_version_requests == [
        "http://127.0.0.1:30000/model_info",
        "http://127.0.0.1:30000/model_info",
    ]


async def test_close_wakes_lease_blocked_drain_and_rolls_back_once(monkeypatch):
    reservation = make_reservation(75)
    data_source = FakeReservationDataSource([reservation])
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_started.set()
        await release_generation.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate)
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=75)))
    await generation_started.wait()
    hold = await fn.acquire_train_admission_hold()

    release_generation.set()
    await hold.wait_terminal()
    assert fn._output is not None
    await wait_until(lambda: fn._output.qsize() == 0)
    assert train.done() is False

    await asyncio.wait_for(fn.close(), timeout=1)
    with pytest.raises(RuntimeError) as train_error:
        await train

    assert str(train_error.value) == "Fully async rollout closed before the train batch lease was issued."
    assert data_source.requeued == [[reservation]]
    assert data_source.requeued[0][0] is reservation

    await fn.close()
    assert data_source.requeued == [[reservation]]


async def test_checkpoint_preparation_rejects_an_open_train_batch_lease(monkeypatch):
    reservation = make_reservation(65)
    data_source = FakeReservationDataSource([reservation])
    fn = make_owned_fn(monkeypatch, data_source)
    output = await fn(RolloutFnTrainInput(rollout_id=65))
    hold = await fn.acquire_train_admission_hold()

    with pytest.raises(RuntimeError) as checkpoint_error:
        await fn.prepare_checkpoint(rollout_id=65)

    assert str(checkpoint_error.value) == "Cannot prepare checkpoint 65 with open train batch leases: [65]."

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)

    assert await fn.prepare_checkpoint(rollout_id=65) is None
    hold.release()
    await fn.close()


async def test_blocked_close_permanently_invalidates_admission_hold_then_retries(monkeypatch):
    reservations = [make_reservation(66), make_reservation(67)]
    data_source = FakeReservationDataSource(reservations)
    fn = make_owned_fn(
        monkeypatch,
        data_source,
        retained_groups=1,
        completed_groups=2,
    )
    output = await fn(RolloutFnTrainInput(rollout_id=66))
    hold = await fn.acquire_train_admission_hold()

    with pytest.raises(RuntimeError) as close_error:
        await fn.close()

    assert str(close_error.value) == "Cannot close fully async rollout with open train batch leases: [66]."
    with pytest.raises(RuntimeError) as release_error:
        hold.release()
    assert str(release_error.value) == "Train admission hold is not active on this rollout function."

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    for _ in range(10):
        await asyncio.sleep(0)

    assert data_source.reserved == reservations[:1]

    await fn.close()

    assert data_source.requeued[0] == [reservations[0]]
    assert data_source.requeued[0][0] is reservations[0]


async def test_cancelled_terminal_wait_preserves_the_hold_and_frontier(monkeypatch):
    reservation = make_reservation(60)
    data_source = FakeReservationDataSource([reservation])
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_started.set()
        await release_generation.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate)
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=60)))
    await generation_started.wait()
    hold = await fn.acquire_train_admission_hold()

    cancelled_wait = asyncio.create_task(hold.wait_terminal())
    await asyncio.sleep(0)
    cancelled_wait.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_wait

    retry_wait = asyncio.create_task(hold.wait_terminal())
    await asyncio.sleep(0)

    assert (retry_wait.done(), data_source.requeued) == (False, [])

    release_generation.set()
    await retry_wait
    assert fn._output is not None
    await wait_until(lambda: fn._output.qsize() == 0)
    assert train.done() is False

    hold.release()
    output = await train

    assert output.samples == [list(reservation.samples)]

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    await fn.close()


async def test_train_admission_hold_allows_staggered_frontier_processing(monkeypatch):
    reservations = [make_reservation(68), make_reservation(69)]
    data_source = FakeReservationDataSource(reservations)
    all_started = asyncio.Event()
    release_first = asyncio.Event()
    release_second = asyncio.Event()
    started: list[int] = []

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = group[0].group_index
        started.append(group_index)
        if len(started) == 2:
            all_started.set()
        await all_started.wait()
        await (release_first if group_index == 68 else release_second).wait()
        return group

    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        batch_size=2,
        execution_samples=4,
        retained_groups=2,
        completed_groups=2,
    )
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=68)))
    await all_started.wait()
    hold = await fn.acquire_train_admission_hold()
    terminal = asyncio.create_task(hold.wait_terminal())

    release_first.set()
    for _ in range(10):
        await asyncio.sleep(0)

    assert terminal.done() is False

    release_second.set()
    await asyncio.wait_for(terminal, timeout=1)
    assert train.done() is False

    hold.release()
    output = await asyncio.wait_for(train, timeout=1)

    assert output.samples == [list(reservations[0].samples), list(reservations[1].samples)]

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    await fn.close()


async def test_terminal_frontier_failure_waits_for_siblings_and_keeps_admission_held(monkeypatch):
    reservations = [make_reservation(61), make_reservation(62)]
    data_source = FakeReservationDataSource(reservations)
    all_started = asyncio.Event()
    release_failure = asyncio.Event()
    release_sibling = asyncio.Event()
    started: list[int] = []
    failure = RuntimeError("frontier execution failed")

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = group[0].group_index
        started.append(group_index)
        if len(started) == 2:
            all_started.set()
        await all_started.wait()
        if group_index == 61:
            await release_failure.wait()
            raise failure
        try:
            await release_sibling.wait()
        except asyncio.CancelledError:
            await release_sibling.wait()
        return group

    async def request_abort(args) -> None:
        pass

    monkeypatch.setattr(inference_fully_async, "request_abort", request_abort)
    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        batch_size=2,
        execution_samples=4,
        retained_groups=2,
        completed_groups=2,
    )
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=61)))
    await all_started.wait()
    hold = await fn.acquire_train_admission_hold()
    terminal = asyncio.create_task(hold.wait_terminal())

    release_failure.set()
    for _ in range(10):
        await asyncio.sleep(0)

    assert terminal.done() is False

    release_sibling.set()
    with pytest.raises(RuntimeError) as terminal_error:
        await terminal
    with pytest.raises(RuntimeError) as train_error:
        await train
    await wait_until(lambda: sum(len(batch) for batch in data_source.requeued) == 2)

    assert terminal_error.value is failure
    assert train_error.value is failure
    assert (
        sorted(
            (reservation for batch in data_source.requeued for reservation in batch),
            key=lambda reservation: reservation.reservation_id,
        )
        == reservations
    )

    with pytest.raises(RuntimeError) as close_error:
        await fn.close()
    assert close_error.value is failure
    await fn.close()


async def test_terminal_frontier_reports_the_canonical_worker_failure(monkeypatch):
    reservations = [make_reservation(70), make_reservation(71)]
    data_source = FakeReservationDataSource(reservations)
    all_started = asyncio.Event()
    release_first = asyncio.Event()
    release_second = asyncio.Event()
    abort_requested = asyncio.Event()
    started: list[int] = []
    first_failure = RuntimeError("first captured execution failed")
    canonical_failure = RuntimeError("second captured execution failed first")

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = group[0].group_index
        started.append(group_index)
        if len(started) == 2:
            all_started.set()
        await all_started.wait()
        await (release_first if group_index == 70 else release_second).wait()
        if group_index == 70:
            raise first_failure
        raise canonical_failure

    async def request_abort(args) -> None:
        abort_requested.set()

    monkeypatch.setattr(inference_fully_async, "request_abort", request_abort)
    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        batch_size=2,
        execution_samples=4,
        retained_groups=2,
        completed_groups=2,
    )
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=70)))
    await all_started.wait()
    hold = await fn.acquire_train_admission_hold()
    terminal = asyncio.create_task(hold.wait_terminal())

    release_second.set()
    await abort_requested.wait()
    release_first.set()

    with pytest.raises(RuntimeError) as terminal_error:
        await terminal
    with pytest.raises(RuntimeError) as train_error:
        await train

    assert terminal_error.value is canonical_failure
    assert train_error.value is canonical_failure

    with pytest.raises(RuntimeError) as close_error:
        await fn.close()
    assert close_error.value is canonical_failure
    await fn.close()


async def test_train_admission_hold_preserves_a_prior_worker_failure(monkeypatch):
    reservation = make_reservation(64)
    data_source = FakeReservationDataSource([reservation])
    failure = RuntimeError("worker failed before hold acquisition")

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        raise failure

    fn = make_owned_fn(monkeypatch, data_source, generate)
    with pytest.raises(RuntimeError) as train_error:
        await fn(RolloutFnTrainInput(rollout_id=64))

    hold = await fn.acquire_train_admission_hold()
    with pytest.raises(RuntimeError) as terminal_error:
        await hold.wait_terminal()

    assert train_error.value is failure
    assert terminal_error.value is failure

    with pytest.raises(RuntimeError) as close_error:
        await fn.close()
    assert close_error.value is failure
    await fn.close()


async def test_close_dominates_an_active_train_admission_hold(monkeypatch):
    reservation = make_reservation(63)
    data_source = FakeReservationDataSource([reservation])
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()
    abort_requested = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_started.set()
        try:
            await release_generation.wait()
        except asyncio.CancelledError:
            await release_generation.wait()
        return group

    async def request_abort(args) -> None:
        abort_requested.set()

    monkeypatch.setattr(inference_fully_async, "request_abort", request_abort)
    fn = make_owned_fn(monkeypatch, data_source, generate)
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=63)))
    await generation_started.wait()
    hold = await fn.acquire_train_admission_hold()
    close = asyncio.create_task(fn.close())
    await abort_requested.wait()
    terminal = asyncio.create_task(hold.wait_terminal())
    await asyncio.sleep(0)

    assert (terminal.done(), close.done()) == (False, False)

    release_generation.set()
    await terminal
    await close
    with pytest.raises(asyncio.CancelledError):
        await train

    assert data_source.requeued == [[reservation]]
    with pytest.raises(RuntimeError) as release_error:
        hold.release()
    assert str(release_error.value) == "Train admission hold is not active on this rollout function."
    with pytest.raises(RuntimeError) as acquisition_error:
        await fn.acquire_train_admission_hold()
    assert str(acquisition_error.value) == "Fully async rollout function is closed."


async def test_owned_retained_limit_does_not_block_active_completion(monkeypatch):
    reservation = make_reservation(32)
    data_source = FakeReservationDataSource([reservation])
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        generation_started.set()
        await release_generation.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate, execution_samples=4)
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=32)))
    await generation_started.wait()

    release_generation.set()
    output = await asyncio.wait_for(drain, timeout=1)

    assert output.samples == [list(reservation.samples)]
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)

    assert data_source.requeued == [[reservation]]


async def test_owned_retained_capacity_reopens_only_after_lease_settlement(monkeypatch):
    first_reservation = make_reservation(10)
    second_reservation = make_reservation(11)
    data_source = FakeReservationDataSource([first_reservation, second_reservation])
    fn = make_owned_fn(monkeypatch, data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=22))
    for _ in range(10):
        await asyncio.sleep(0)

    assert data_source.reserved == [first_reservation]

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    await wait_until(lambda: len(data_source.reserved) == 2)

    assert data_source.requeued == [[first_reservation]]
    assert data_source.requeued[0][0] is first_reservation
    assert [sample.response for sample in data_source.requeued[0][0].samples] == ["ok", "ok"]
    assert data_source.reserved == [first_reservation, second_reservation]


async def test_owned_lease_settlement_backfills_while_another_group_is_active(monkeypatch):
    reservations = [make_reservation(index) for index in range(10, 13)]
    data_source = FakeReservationDataSource(reservations)
    second_started = asyncio.Event()
    third_started = asyncio.Event()
    release_active = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = group[0].group_index
        if group_index == 10:
            await second_started.wait()
        elif group_index == 11:
            second_started.set()
            await release_active.wait()
        else:
            third_started.set()
            await release_active.wait()
        if sample_done_callback is not None:
            for _ in group:
                sample_done_callback()
        return group

    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        execution_samples=4,
        retained_groups=2,
        completed_groups=1,
        rollout_submission_granularity="sample",
    )
    output = await fn(RolloutFnTrainInput(rollout_id=10))

    assert data_source.reserved == reservations[:2]

    output.lease.commit()
    try:
        await asyncio.wait_for(third_started.wait(), timeout=1)
    finally:
        release_active.set()
        await fn.close()

    assert data_source.reserved == reservations


async def test_failed_owned_lease_rollback_transfers_cleanup_to_checkpoint_retry(monkeypatch, lifecycle_loop):
    reservations = [make_reservation(70), make_reservation(71)]
    data_source = FakeReservationDataSource(reservations)
    requeue_error = RuntimeError("lease rollback failed")
    original_requeue = data_source.requeue_reservations
    requeue_attempts = 0

    def requeue_reservations(requeued: Sequence[SourceReservation]) -> None:
        nonlocal requeue_attempts
        requeue_attempts += 1
        if requeue_attempts == 1:
            raise requeue_error
        original_requeue(requeued)

    data_source.requeue_reservations = requeue_reservations
    fn = make_owned_fn(monkeypatch, data_source)
    output = await asyncio.to_thread(
        lifecycle_loop.run,
        fn(RolloutFnTrainInput(rollout_id=70)),
    )

    with pytest.raises(RuntimeError) as rollback_error:
        output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)

    assert rollback_error.value is requeue_error
    with pytest.raises(RuntimeError) as repeated_rollback_error:
        output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    assert str(repeated_rollback_error.value) == "Train batch lease for rollout 70 already has a settlement attempt."
    assert fn._retained_slots is not None
    assert fn._completed_slots is not None
    assert (fn._retained_slots.locked(), fn._completed_slots.qsize()) == (True, 0)
    await asyncio.to_thread(lifecycle_loop.run, fn.acquire_train_admission_hold())
    assert await asyncio.to_thread(lifecycle_loop.run, fn.prepare_checkpoint(rollout_id=70)) is None
    await asyncio.to_thread(lifecycle_loop.run, asyncio.sleep(0.01))
    assert (fn._retained_slots.locked(), fn._completed_slots.qsize()) == (False, 1)
    assert (data_source.reserved, data_source.requeued, requeue_attempts) == (
        reservations[:1],
        [[reservations[0]]],
        2,
    )

    await asyncio.to_thread(lifecycle_loop.run, fn.close())
    await asyncio.to_thread(lifecycle_loop.run, fn.close())

    assert (fn._retained_slots.locked(), fn._completed_slots.qsize()) == (False, 1)
    assert (data_source.acknowledged, data_source.requeued, requeue_attempts) == (
        [],
        [[reservations[0]]],
        2,
    )


async def test_failed_owned_lease_commit_requeues_before_checkpoint(monkeypatch):
    reservation = make_reservation(75)
    data_source = FakeReservationDataSource([reservation])
    acknowledge_error = RuntimeError("lease commit failed")
    acknowledge_attempts = 0

    def acknowledge_reservations(
        reservations: Sequence[SourceReservation],
        *,
        rollout_id: int,
    ) -> None:
        nonlocal acknowledge_attempts
        acknowledge_attempts += 1
        raise acknowledge_error

    data_source.acknowledge_reservations = acknowledge_reservations
    fn = make_owned_fn(monkeypatch, data_source)
    output = await fn(RolloutFnTrainInput(rollout_id=75))

    with pytest.raises(RuntimeError) as commit_error:
        output.lease.commit()

    assert commit_error.value is acknowledge_error
    with pytest.raises(RuntimeError) as repeated_commit_error:
        output.lease.commit()
    assert str(repeated_commit_error.value) == "Train batch lease for rollout 75 already has a settlement attempt."
    assert fn._retained_slots is not None
    assert fn._completed_slots is not None
    assert (fn._retained_slots.locked(), fn._completed_slots.qsize()) == (False, 1)
    assert data_source.requeued == [[reservation]]
    await fn.acquire_train_admission_hold()

    assert await fn.prepare_checkpoint(rollout_id=75) is None

    assert acknowledge_attempts == 1
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert (fn._retained_slots.locked(), fn._completed_slots.qsize()) == (False, 1)
    await fn.close()


async def test_cross_thread_failed_owned_lease_commit_retries_rollback_through_checkpoint_and_close(
    monkeypatch,
    lifecycle_loop,
):
    reservation = make_reservation(77)
    data_source = FakeReservationDataSource([reservation])
    commit_error = RuntimeError("cross-thread lease commit failed")
    rollback_error = RuntimeError("cross-thread commit rollback failed")
    checkpoint_error = RuntimeError("checkpoint rollback retry failed")
    original_requeue = data_source.requeue_reservations
    acknowledge_attempts = 0
    requeue_attempts = 0

    def acknowledge_reservations(
        reservations: Sequence[SourceReservation],
        *,
        rollout_id: int,
    ) -> None:
        nonlocal acknowledge_attempts
        acknowledge_attempts += 1
        raise commit_error

    def requeue_reservations(reservations: Sequence[SourceReservation]) -> None:
        nonlocal requeue_attempts
        requeue_attempts += 1
        if requeue_attempts == 1:
            raise rollback_error
        if requeue_attempts == 2:
            raise checkpoint_error
        original_requeue(reservations)

    data_source.acknowledge_reservations = acknowledge_reservations
    data_source.requeue_reservations = requeue_reservations
    fn = make_owned_fn(monkeypatch, data_source)
    output = await asyncio.to_thread(
        lifecycle_loop.run,
        fn(RolloutFnTrainInput(rollout_id=77)),
    )

    with pytest.raises(RuntimeError) as first_commit_error:
        output.lease.commit()

    assert first_commit_error.value is commit_error
    assert first_commit_error.value.__cause__ is rollback_error
    await asyncio.to_thread(lifecycle_loop.run, fn.acquire_train_admission_hold())

    with pytest.raises(RuntimeError) as failed_checkpoint:
        await asyncio.to_thread(lifecycle_loop.run, fn.prepare_checkpoint(rollout_id=77))

    assert failed_checkpoint.value is checkpoint_error
    assert acknowledge_attempts == 1
    assert requeue_attempts == 2
    assert data_source.acknowledged == []
    assert data_source.requeued == []
    assert fn._retained_slots is not None
    assert fn._completed_slots is not None
    assert (fn._retained_slots.locked(), fn._completed_slots.qsize()) == (True, 0)

    await asyncio.to_thread(lifecycle_loop.run, fn.close())

    assert acknowledge_attempts == 1
    assert requeue_attempts == 3
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]
    assert (fn._retained_slots.locked(), fn._completed_slots.qsize()) == (False, 1)
    assert fn._closed


async def test_checkpoint_preparation_fails_while_terminal_rollback_remains_pending(monkeypatch):
    reservation = make_reservation(76)
    data_source = FakeReservationDataSource([reservation])
    lease_requeue_error = RuntimeError("lease rollback failed")
    checkpoint_requeue_error = RuntimeError("checkpoint rollback retry failed")
    original_requeue = data_source.requeue_reservations
    requeue_attempts = 0

    def requeue_reservations(requeued: Sequence[SourceReservation]) -> None:
        nonlocal requeue_attempts
        requeue_attempts += 1
        if requeue_attempts == 1:
            raise lease_requeue_error
        if requeue_attempts == 2:
            raise checkpoint_requeue_error
        original_requeue(requeued)

    data_source.requeue_reservations = requeue_reservations
    fn = make_owned_fn(monkeypatch, data_source)
    output = await fn(RolloutFnTrainInput(rollout_id=76))

    with pytest.raises(RuntimeError) as lease_error:
        output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)

    assert lease_error.value is lease_requeue_error
    await fn.acquire_train_admission_hold()

    with pytest.raises(RuntimeError) as checkpoint_error:
        await fn.prepare_checkpoint(rollout_id=76)

    assert checkpoint_error.value is checkpoint_requeue_error
    assert fn._retained_slots is not None
    assert fn._completed_slots is not None
    assert (fn._retained_slots.locked(), fn._completed_slots.qsize()) == (True, 0)
    assert data_source.requeued == []

    assert await fn.prepare_checkpoint(rollout_id=76) is None

    assert (fn._retained_slots.locked(), fn._completed_slots.qsize()) == (False, 1)
    assert data_source.requeued == [[reservation]]
    await fn.close()


@pytest.mark.parametrize(
    "rollback_reason",
    [None, TrainBatchRollbackReason.HANDOFF_FAILED],
)
async def test_owned_lease_settlement_wakes_capacity_waiter_on_lifecycle_loop(
    monkeypatch,
    lifecycle_loop,
    rollback_reason: TrainBatchRollbackReason | None,
):
    reservations = [make_reservation(20), make_reservation(21)]
    data_source = FakeReservationDataSource(reservations)
    fn = make_owned_fn(monkeypatch, data_source)
    output = await asyncio.to_thread(
        lifecycle_loop.run,
        fn(RolloutFnTrainInput(rollout_id=20)),
    )

    assert data_source.reserved == reservations[:1]

    if rollback_reason is None:
        output.lease.commit()
    else:
        output.lease.rollback(rollback_reason)
    await wait_until(lambda: data_source.reserved == reservations)

    if rollback_reason is None:
        assert data_source.acknowledged == [([reservations[0]], 20)]
        assert data_source.requeued == []
    else:
        assert data_source.acknowledged == []
        assert data_source.requeued == [[reservations[0]]]

    await asyncio.to_thread(lifecycle_loop.run, fn.close())


async def test_owned_batch_rolls_back_valid_and_identity_invalid_reservations_together(monkeypatch):
    reservations = [make_reservation(33), make_reservation(34)]
    data_source = FakeReservationDataSource(reservations)
    first_identity_validated = asyncio.Event()
    original_identity_error = fully_async._owned_group_identity_error

    def identity_error(completed):
        error = original_identity_error(completed)
        if error is None:
            first_identity_validated.set()
        return error

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        if group[0].group_index == 33:
            return group
        await first_identity_validated.wait()
        return [deepcopy(group[1]), deepcopy(group[0])]

    monkeypatch.setattr(fully_async, "_owned_group_identity_error", identity_error)
    fn = make_owned_fn(
        monkeypatch,
        data_source,
        generate,
        batch_size=2,
        execution_samples=4,
        retained_groups=2,
        completed_groups=2,
    )

    with pytest.raises(ValueError) as error:
        await fn(RolloutFnTrainInput(rollout_id=33))

    assert str(error.value) == (
        "Source reservation source-34 returned sample identities [(34, 341)] at parent slot 0; "
        "expected every sample to have identity (34, 340)."
    )
    assert data_source.acknowledged == []
    assert data_source.requeued == [reservations]
    assert all(actual is expected for actual, expected in zip(data_source.requeued[0], reservations, strict=True))


async def test_owned_completed_prefetch_overflow_requeues_and_blocks_admission(monkeypatch):
    release = asyncio.Event()
    started: list[int] = []
    reservations = [make_reservation(index) for index in range(12, 20)]
    data_source = FakeReservationDataSource(reservations)

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        started.append(group[0].group_index)
        await release.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate, execution_samples=6, retained_groups=10)
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=23)))
    await wait_until(lambda: len(started) == 3)

    release.set()
    output = await drain
    await wait_until(lambda: len(data_source.requeued) == 2)
    for _ in range(10):
        await asyncio.sleep(0)

    leased_group_index = output.samples[0][0].group_index
    leased_reservation = next(
        reservation for reservation in reservations[:3] if reservation.samples[0].group_index == leased_group_index
    )
    requeued_reservations = sorted(
        [reservation for batch in data_source.requeued for reservation in batch],
        key=lambda reservation: reservation.reservation_id,
    )
    expected_requeued = sorted(
        [reservation for reservation in reservations[:3] if reservation is not leased_reservation],
        key=lambda reservation: reservation.reservation_id,
    )

    assert output.samples == [list(leased_reservation.samples)]
    assert data_source.reserved == reservations[:3]
    assert data_source.acknowledged == []
    assert requeued_reservations == expected_requeued
    assert all(
        actual is expected
        for actual, expected in zip(
            requeued_reservations,
            expected_requeued,
            strict=True,
        )
    )

    output.lease.commit()
    await wait_until(lambda: len(data_source.reserved) >= 4)

    assert data_source.reserved[:4] == reservations[:4]
    assert data_source.acknowledged == [([leased_reservation], 23)]


async def test_owned_buffer_eviction_requeues_exact_source_reservation(monkeypatch):
    stale_reservation = make_reservation(39)
    fresh_reservation = make_reservation(40)
    for sample in stale_reservation.samples:
        sample.weight_versions = ["1"]
    for sample in fresh_reservation.samples:
        sample.weight_versions = ["9"]
    data_source = FakeReservationDataSource([stale_reservation, fresh_reservation])
    fn = make_owned_fn(
        monkeypatch,
        data_source,
        execution_samples=4,
        retained_groups=2,
        completed_groups=2,
    )
    buffer = fully_async.DataBuffer(
        order="fifo",
        blocking_capacity=fully_async.OUTPUT_QUEUE_MAX_GROUPS,
        max_groups=1,
        max_staleness=None,
        on_evict=fn._recycle_buffer_source,
    )

    async def complete_one():
        await fn._retained_slots.acquire()
        completed = await fn._submit_one_group()
        assert isinstance(completed, fully_async._OwnedCompletedGroup)
        assert fn._try_acquire_completed_slot()
        await buffer.put((completed, completed.samples))

    await complete_one()
    await complete_one()

    assert data_source.acknowledged == []
    assert data_source.requeued == [[stale_reservation]]
    assert data_source.requeued[0][0] is stale_reservation
    assert buffer.qsize() == 1
    assert buffer.evicted_overflow_groups == 1

    assert await fn._rollback_queued_owned_groups(buffer) is None
    assert data_source.requeued == [[stale_reservation], [fresh_reservation]]
    assert data_source.requeued[1][0] is fresh_reservation


async def test_owned_dynamic_filter_drop_commits_before_next_admission(monkeypatch):
    rejected_reservation = make_reservation(41)
    accepted_reservation = make_reservation(42)
    data_source = FakeReservationDataSource([rejected_reservation, accepted_reservation])
    fn = make_owned_fn(monkeypatch, data_source)

    def reject_first(args, group, **kwargs):
        keep = group[0].group_index != 41
        return DynamicFilterOutput(keep=keep, reason=None if keep else "rejected")

    fn._dynamic_filter = reject_first

    output = await asyncio.wait_for(fn(RolloutFnTrainInput(rollout_id=41)), timeout=1)

    assert output.samples == [list(accepted_reservation.samples)]
    assert output.metrics["rollout/dynamic_filter/drop_rejected"] == 1
    assert data_source.acknowledged == [([rejected_reservation], 41)]
    assert data_source.acknowledged[0][0][0] is rejected_reservation
    assert data_source.requeued == []

    output.lease.commit()

    assert data_source.acknowledged == [
        ([rejected_reservation], 41),
        ([accepted_reservation], 41),
    ]
    assert data_source.acknowledged[1][0][0] is accepted_reservation


async def test_drain_collects_batch_sorted_with_metrics(monkeypatch):
    args = make_args(rollout_batch_size=3)
    fn = make_fn(monkeypatch, args, FakeDataSource())

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 3
    indices = [group[0].index for group in output.samples]
    assert indices == sorted(indices)
    assert all(len(group) == N_SAMPLES_PER_PROMPT for group in output.samples)
    assert output.metrics["rollout/fully_async/aborted_groups_recycled"] == 0
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 0

    # The worker persists across calls; a second drain works on the same instance.
    output2 = await fn(RolloutFnTrainInput(rollout_id=1))
    assert len(output2.samples) == 3


async def test_eval_without_fleet_pauses_producer(monkeypatch):
    """Shared-engine eval: producer submissions pause during eval and resume after."""
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(
        monkeypatch, make_args(rollout_batch_size=2, eval_num_gpus=0), data_source, generate=blocking_generate
    )

    eval_started = asyncio.Event()
    eval_release = asyncio.Event()
    eval_results = {"fake_ds": {"rewards": [1.0], "truncated": [False], "samples": []}}

    async def fake_run_eval_datasets(state, cache):
        assert state is fn.state  # shared-engine eval uses the train state
        eval_started.set()
        await eval_release.wait()
        return eval_results

    monkeypatch.setattr(fully_async, "run_eval_datasets", fake_run_eval_datasets)

    # Start the producer via a train call, then run eval concurrently.
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    submitted_before_eval = data_source.num_get_calls

    eval_task = asyncio.create_task(fn(RolloutFnEvalInput(rollout_id=0)))
    await eval_started.wait()
    release.set()  # in-flight groups finish and buffer, but no NEW submissions
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == submitted_before_eval

    eval_release.set()
    output = await eval_task
    assert output.data == eval_results

    # Producer resumes and the train drain completes.
    assert (await drain).samples


async def test_eval_runs_on_dedicated_fleet(monkeypatch):
    """RolloutManager (not the fn) decides fleet-vs-shared and builds the fleet's
    GenerateState; it hands it in via RolloutFnEvalInput.generate_state. The fn must
    use that state as-is (not self.state) and must not touch the producer/data_source.
    Building/caching the fleet state itself is EvalFleetSession's job, covered in
    tests/fast/rollout/test_checkpoint_eval.py.
    """
    args = make_args(eval_num_gpus=1, eval_num_gpus_per_engine=1)
    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, args, data_source)

    fleet_state = FakeGenerateState(args)
    eval_results = {"fake_ds": {"rewards": [1.0], "truncated": [False], "samples": []}}
    seen_states = []

    async def fake_run_eval_datasets(state, cache):
        seen_states.append(state)
        return eval_results

    monkeypatch.setattr(fully_async, "run_eval_datasets", fake_run_eval_datasets)

    output = await fn(RolloutFnEvalInput(rollout_id=0, generate_state=fleet_state, weight_version="0"))

    assert output.data == eval_results
    assert seen_states == [fleet_state]  # used the fleet's state, not fn.state
    # Eval must not start the producer or consume training prompts.
    assert fn._worker is None
    assert data_source.num_get_calls == 0


async def test_aborted_group_recycled(monkeypatch):
    aborted = make_group(1, status=Sample.Status.ABORTED)
    data_source = FakeDataSource(scripted=[aborted])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [aborted]
    # reset_for_retry cleared generated outputs so the prompt can be re-sampled
    assert all(sample.response == "" and sample.weight_versions == [] for sample in aborted)
    assert output.samples[0][0].group_index != 1
    assert output.metrics["rollout/fully_async/aborted_groups_recycled"] == 1


async def test_stale_group_recycled(monkeypatch):
    stale = make_group(1, weight_versions=["5"])
    data_source = FakeDataSource(scripted=[stale])
    data_source_fresh_versions = ["10"]

    original_make = data_source.get_samples

    def get_samples_with_fresh_versions(num_samples):
        groups = original_make(num_samples)
        for group in groups:
            for sample in group:
                if not sample.weight_versions:
                    sample.weight_versions = list(data_source_fresh_versions)
        return groups

    data_source.get_samples = get_samples_with_fresh_versions

    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=2), data_source)

    class FakeWeightVersion:
        async def get(self, args):
            return 10

    fn._weight_version = FakeWeightVersion()

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [stale]
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 1
    assert output.metrics["rollout/fully_async/max_staleness"] == 5


def test_worker_error_propagates_without_leaking_sibling_failure(monkeypatch):
    unhandled_messages: list[str] = []

    async def run_failure() -> None:
        async def failing_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
            raise RuntimeError("generation exploded")

        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _, context: unhandled_messages.append(str(context["message"])))
        fn = make_fn(monkeypatch, make_args(), FakeDataSource(), generate=failing_generate)

        with pytest.raises(RuntimeError, match="generation exploded"):
            await fn(RolloutFnTrainInput(rollout_id=0))

        worker = fn._worker
        fn._worker = None
        del worker
        del fn
        gc.collect()
        await asyncio.sleep(0)

    asyncio.run(run_failure())

    assert unhandled_messages == []


async def test_legacy_worker_failure_retains_done_and_cancelled_siblings_for_close(monkeypatch):
    groups = [make_group(index) for index in range(77, 80)]
    data_source = FakeDataSource(scripted=groups)
    all_started = asyncio.Event()
    release_terminals = asyncio.Event()
    cancelled_sibling = asyncio.Event()
    started: list[int] = []
    generation_error = RuntimeError("legacy generation failed")

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = group[0].group_index
        started.append(group_index)
        if len(started) == 3:
            all_started.set()
        await all_started.wait()
        if group_index == 79:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled_sibling.set()
                return group
        await release_terminals.wait()
        if group_index == 78:
            raise generation_error
        return group

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=3),
        data_source,
        generate=generate,
    )
    train = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=77)))
    await all_started.wait()
    release_terminals.set()

    with pytest.raises(RuntimeError) as train_error:
        await train

    assert train_error.value is generation_error
    await cancelled_sibling.wait()

    with pytest.raises(RuntimeError) as close_error:
        await fn.close()

    assert close_error.value is generation_error
    await fn.close()

    assert sorted(group[0].group_index for group in data_source.recycled) == [77, 78, 79]
    assert len(data_source.recycled) == 3


async def test_worker_bounds_in_flight_groups(monkeypatch):
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == 2  # in-flight bound, not more

    release.set()
    output = await drain
    assert len(output.samples) == 2


async def test_async_max_concurrent_samples_caps_in_flight_groups(monkeypatch):
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release.wait()
        return group

    data_source = FakeDataSource()
    # 3 samples // 2 per group -> 1 group in flight, below rollout_batch_size
    args = make_args(rollout_batch_size=4, async_max_concurrent_samples=3)
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == 1

    release.set()
    output = await drain
    assert len(output.samples) == 4


async def test_worker_failure_beats_queued_groups(monkeypatch):
    """A dead worker fails the step even when it left completed groups behind."""
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

    async def boom():
        raise RuntimeError("generation exploded")

    fn._output = make_buffer()[0]
    group = make_group(1)
    await fn._output.put((group, group))
    fn._worker = asyncio.create_task(boom())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn(RolloutFnTrainInput(rollout_id=0))


async def test_nested_group_recycles_the_flat_prompt_group(monkeypatch):
    """A generate function may expand one trajectory into several samples; the retry
    must resubmit the flat prompt group the data source handed out."""
    prompt_group = make_group(1)
    data_source = FakeDataSource(scripted=[prompt_group])
    submitted = []

    async def multi_sample_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        assert all(isinstance(sample, Sample) for sample in group), "resubmitted a nested group"
        submitted.append(group)
        if len(submitted) > 1:
            return group
        expanded = []
        for sample in group:
            aborted = replace(sample, status=Sample.Status.ABORTED)
            expanded.append([aborted, replace(sample)])
        return expanded

    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source, generate=multi_sample_generate)
    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [prompt_group]
    assert all(isinstance(sample, Sample) for sample in data_source.recycled[0])
    assert len(submitted) > 1
    assert len(output.samples) == 1


async def test_dynamic_filter_drops_group_without_recycling(monkeypatch):
    rejected = make_group(1)
    data_source = FakeDataSource(scripted=[rejected])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)

    def reject_group_1(args, group, **kwargs):
        keep = group[0].group_index != 1
        return DynamicFilterOutput(keep=keep, reason=None if keep else "rejected")

    fn._dynamic_filter = reject_group_1

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 1
    assert output.samples[0][0].group_index != 1
    # Unlike a recycle, a filtered group is not returned to the data source for re-sampling.
    assert data_source.recycled == []
    assert output.metrics["rollout/dynamic_filter/drop_rejected"] == 1


async def test_sample_filter_marks_samples_without_shrinking_the_batch(monkeypatch):
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), FakeDataSource())

    def mark_first_of_each_group(args, data):
        for group in data:
            group[0].remove_sample = True

    fn._sample_filter = mark_first_of_each_group

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 2
    assert [sample.remove_sample for sample in output.samples[0]] == [True, False]


async def test_weight_version_throttles_failed_queries(monkeypatch):
    """A drain queries once per group, so an unreachable router must not cost one timeout each."""
    calls = []

    async def unreachable_router(url):
        calls.append(url)
        raise httpx.ConnectError("router down")

    monkeypatch.setattr(fully_async, "get", unreachable_router)
    args = make_args()

    throttled = fully_async._CachedWeightVersion(ttl=60.0)
    assert await throttled.get(args) is None
    assert await throttled.get(args) is None
    assert len(calls) == 1

    calls.clear()
    expired = fully_async._CachedWeightVersion(ttl=0.0)
    assert await expired.get(args) is None
    assert await expired.get(args) is None
    assert len(calls) == 2


async def test_weight_version_refresh_cannot_be_poisoned_by_older_query(monkeypatch):
    first_query_started = asyncio.Event()
    release_first_query = asyncio.Event()
    calls: list[str] = []

    async def reordered_router(url: str) -> dict[str, str]:
        calls.append(url)
        if len(calls) == 1:
            first_query_started.set()
            await release_first_query.wait()
            return {"weight_version": "1"}
        return {"weight_version": "2"}

    monkeypatch.setattr(fully_async, "get", reordered_router)
    args = make_args()
    weight_version = fully_async._CachedWeightVersion(ttl=60.0)

    older_query = asyncio.create_task(weight_version.get(args))
    await first_query_started.wait()
    admission_refresh = asyncio.create_task(weight_version.refresh(args))
    await asyncio.sleep(0.01)
    release_first_query.set()

    older_result, refresh_result = await asyncio.gather(older_query, admission_refresh)
    cached_result = await weight_version.get(args)

    assert (older_result, refresh_result, cached_result, calls) == (
        1,
        2,
        2,
        [
            "http://127.0.0.1:30000/model_info",
            "http://127.0.0.1:30000/model_info",
        ],
    )


async def test_worker_defaults_to_sample_granularity(monkeypatch):
    """Unset --rollout-submission-granularity: this driver backfills on sample completion."""
    callbacks = []
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        callbacks.append(sample_done_callback)
        await release.wait()
        return group

    data_source = FakeDataSource()
    args = make_args(rollout_batch_size=1)
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.01)
    assert data_source.num_get_calls == 1

    # Report every sample of the still-pending group as finished.
    for _ in range(N_SAMPLES_PER_PROMPT):
        callbacks[0]()
    await asyncio.sleep(0.01)

    # A replacement group went out even though the first group has not returned.
    assert data_source.num_get_calls == 2

    release.set()
    output = await drain
    assert len(output.samples) == 1


async def test_owned_backfill_submits_replacement_before_the_group_returns(monkeypatch):
    callbacks = []
    release = asyncio.Event()
    reservations = [make_reservation(90), make_reservation(91)]
    data_source = FakeReservationDataSource(reservations)

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        callbacks.append(sample_done_callback)
        await release.wait()
        return group

    fn = make_owned_fn(
        monkeypatch,
        data_source,
        blocking_generate,
        retained_groups=2,
        completed_groups=2,
        rollout_submission_granularity="sample",
    )
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=90)))
    await wait_until(lambda: len(data_source.reserved) == 1 and len(callbacks) == 1)

    for _ in range(N_SAMPLES_PER_PROMPT):
        callbacks[0]()
    await wait_until(lambda: len(data_source.reserved) == 2)

    release.set()
    output = await drain
    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    await fn.close()

    assert data_source.reserved[:2] == reservations


async def test_group_granularity_opts_the_worker_out_of_backfill(monkeypatch):
    callbacks = []
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        callbacks.append(sample_done_callback)
        await release.wait()
        return group

    data_source = FakeDataSource()
    args = make_args(rollout_batch_size=1, rollout_submission_granularity="group")
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.01)
    assert data_source.num_get_calls == 1
    # No callback is wired, so nothing can free a slot before the group task returns.
    assert callbacks == [None]

    await asyncio.sleep(0.01)
    assert data_source.num_get_calls == 1

    release.set()
    output = await drain
    assert len(output.samples) == 1


# ── DataBuffer: staleness-bounded buffering ─────────────────────────


def make_buffer(**overrides):
    evicted = []
    defaults = dict(
        order="fifo",
        blocking_capacity=fully_async.OUTPUT_QUEUE_MAX_GROUPS,
        max_groups=None,
        max_staleness=None,
        on_evict=evicted.append,
    )
    defaults.update(overrides)
    return fully_async.DataBuffer(**defaults), evicted


async def put_group(buffer, group, **kwargs):
    """The buffer holds (prompt group, finished group); these tests reuse one for both."""
    await buffer.put((group, group), **kwargs)


async def test_buffer_evicts_stalest_on_overflow():
    buffer, evicted = make_buffer(max_groups=2)
    oldest = make_group(1, weight_versions=["5"])
    await put_group(buffer, oldest)
    await put_group(buffer, make_group(2, weight_versions=["7"]))
    await put_group(buffer, make_group(3, weight_versions=["9"]))

    assert evicted == [oldest]
    assert buffer.qsize() == 2
    assert buffer.evicted_overflow_groups == 1
    _, group = await buffer.get()
    assert group[0].group_index == 2


async def test_buffer_eviction_recycles_the_original_prompt_group(monkeypatch):
    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, make_args(), data_source)
    buffer = fully_async.DataBuffer(
        order="fifo",
        blocking_capacity=fully_async.OUTPUT_QUEUE_MAX_GROUPS,
        max_groups=1,
        max_staleness=None,
        on_evict=fn._recycle_buffer_source,
    )
    prompt_group = make_group(1)
    generated_group = make_group(101, weight_versions=["5"])

    await buffer.put((prompt_group, generated_group))
    await buffer.put((make_group(2), make_group(102, weight_versions=["9"])))

    assert data_source.recycled[0] is prompt_group
    assert generated_group not in data_source.recycled


async def test_buffer_overflow_tie_broken_by_summed_staleness():
    buffer, evicted = make_buffer(max_groups=2)
    # Same stalest sample (version 5); the first group's other sample is also
    # at 5, so its summed staleness is larger and it loses the tie.
    all_old = make_group(1, weight_versions=["5"])
    one_old = make_group(2, weight_versions=["5"])
    one_old[1].weight_versions = ["9"]
    await put_group(buffer, all_old)
    await put_group(buffer, one_old)
    await put_group(buffer, make_group(3, weight_versions=["9"]))

    assert evicted == [all_old]


async def test_buffer_threshold_evicts_all_over_staleness_first():
    buffer, evicted = make_buffer(max_groups=3, max_staleness=2)
    over_a = make_group(1, weight_versions=["5"])
    over_b = make_group(2, weight_versions=["6"])
    await put_group(buffer, over_a, current_version=10)
    await put_group(buffer, over_b, current_version=10)
    await put_group(buffer, make_group(3, weight_versions=["9"]), current_version=10)
    await put_group(buffer, make_group(4, weight_versions=["10"]), current_version=10)

    assert evicted == [over_a, over_b]
    assert buffer.evicted_stale_groups == 2
    assert buffer.evicted_overflow_groups == 0
    assert buffer.qsize() == 2


async def test_buffer_threshold_evicts_existing_groups_stalest_first():
    buffer, evicted = make_buffer(max_groups=3, max_staleness=2)
    less_stale = make_group(1, weight_versions=["6"])
    most_stale = make_group(2, weight_versions=["5"])
    await put_group(buffer, less_stale, current_version=10)
    await put_group(buffer, most_stale, current_version=10)
    await put_group(buffer, make_group(3, weight_versions=["9"]), current_version=10)
    await put_group(buffer, make_group(4, weight_versions=["10"]), current_version=10)

    assert evicted == [most_stale, less_stale]


async def test_buffer_failed_eviction_returns_incoming_group_to_caller():
    first_group = make_group(1, weight_versions=["5"])
    incoming_group = make_group(2, weight_versions=["6"])
    first_entry = (first_group, first_group)
    incoming_entry = (incoming_group, incoming_group)
    settled = []
    failure = RuntimeError("incoming settlement failed")

    def settle(source):
        if source is incoming_group:
            raise failure
        settled.append(source)

    buffer = fully_async.DataBuffer(
        order="fifo",
        blocking_capacity=fully_async.OUTPUT_QUEUE_MAX_GROUPS,
        max_groups=1,
        max_staleness=2,
        on_evict=settle,
    )
    await buffer.put(first_entry, current_version=10)

    with pytest.raises(RuntimeError) as error:
        await buffer.put(incoming_entry, current_version=10)

    assert error.value is failure
    assert settled == [first_group]
    assert buffer.qsize() == 0
    assert buffer.entered_groups == 1


async def test_buffer_lifo_serves_freshest_first():
    buffer, _ = make_buffer(order="lifo")
    await put_group(buffer, make_group(1))
    await put_group(buffer, make_group(2))

    assert (await buffer.get())[1][0].group_index == 2
    assert (await buffer.get())[1][0].group_index == 1


async def test_buffer_staleness_stats():
    buffer, _ = make_buffer(max_groups=8)
    await put_group(buffer, make_group(1, weight_versions=["4"]))
    await put_group(buffer, make_group(2, weight_versions=["8"]))

    assert buffer.staleness_stats(None) is None
    assert buffer.staleness_stats(10) == (4.0, 6)


async def test_drain_reports_eviction_metrics(monkeypatch):
    fn = make_fn(monkeypatch, make_args(async_data_buffer_max_batches=4), FakeDataSource())
    await fn(RolloutFnTrainInput(rollout_id=0))

    # Evictions land in the buffer counters between drains; the racy overflow
    # path itself is covered by the DataBuffer tests above.
    assert fn._output._on_evict == fn._recycle_buffer_source
    fn._output.entered_groups += 8
    fn._output.evicted_stale_groups = 1
    fn._output.evicted_overflow_groups = 2
    output = await fn(RolloutFnTrainInput(rollout_id=1))

    assert output.metrics["rollout/fully_async/evicted_stale_groups"] == 1
    assert output.metrics["rollout/fully_async/evicted_overflow_groups"] == 2
    # 3 evictions over >= 8 seeded + 2 consumed entries
    assert 0 < output.metrics["rollout/fully_async/evict_rate"] <= 3 / 10
    assert fn._output.evicted_stale_groups == 0  # counters reset per drain

    output2 = await fn(RolloutFnTrainInput(rollout_id=2))
    assert output2.metrics["rollout/fully_async/evicted_overflow_groups"] == 0
