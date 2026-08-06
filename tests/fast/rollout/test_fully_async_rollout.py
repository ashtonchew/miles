from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace
from collections import deque
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest

import miles.rollout.fully_async_rollout as fully_async
from miles.rollout.base_types import (
    LeasedRolloutFnTrainOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
    TrainBatchRollbackReason,
)
from miles.rollout.data_source import SourceReservation, SourceReservationId
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.async_utils import AsyncLoopThread
from miles.utils.types import Sample

N_SAMPLES_PER_PROMPT = 2


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
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        eval_num_gpus=0,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class FakeWeightVersion:
    def __init__(self, value: int | None = None):
        self.value = value

    async def get(self, args) -> int | None:
        return self.value


def make_fn(monkeypatch, args, data_source, generate=None):
    async def default_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await asyncio.sleep(0)
        return group

    monkeypatch.setattr(fully_async, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(fully_async, "generate_and_rm_group", generate or default_generate)
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

    async def generate(state, group, sampling_params, evaluation=False):
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

    async def generate(state, group, sampling_params, evaluation=False):
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

    async def generate(state, group, sampling_params, evaluation=False):
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

    async def generate(state, group, sampling_params, evaluation=False):
        raise failure

    fn = make_owned_fn(monkeypatch, data_source, generate)

    with pytest.raises(RuntimeError) as error:
        await fn(RolloutFnTrainInput(rollout_id=18))

    assert error.value is failure
    assert data_source.acknowledged == []
    assert data_source.requeued == [[reservation]]


async def test_terminal_failure_drains_and_requeues_active_siblings(monkeypatch):
    reservations = [make_reservation(index) for index in range(26, 29)]
    data_source = FakeReservationDataSource(reservations)
    all_started = asyncio.Event()
    release_successes = asyncio.Event()
    started: list[int] = []
    failure = RuntimeError("generation failed")

    async def generate(state, group, sampling_params, evaluation=False):
        group_index = group[0].group_index
        started.append(group_index)
        if len(started) == 3:
            all_started.set()
        await all_started.wait()
        if group_index == 26:
            raise failure
        await release_successes.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate, execution_samples=6, retained_groups=3)
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=26)))
    await all_started.wait()
    for _ in range(10):
        await asyncio.sleep(0)
    finished_before_siblings = drain.done()

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

    async def generate(state, group, sampling_params, evaluation=False):
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

    async def generate(state, group, sampling_params, evaluation=False):
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


async def test_local_execution_cancellation_does_not_requeue_without_terminal_receipt(monkeypatch):
    reservation = make_reservation(29)
    data_source = FakeReservationDataSource([reservation])

    async def generate(state, group, sampling_params, evaluation=False):
        raise asyncio.CancelledError()

    fn = make_owned_fn(monkeypatch, data_source, generate)

    with pytest.raises(asyncio.CancelledError):
        await fn(RolloutFnTrainInput(rollout_id=29))

    assert data_source.reserved == [reservation]
    assert data_source.acknowledged == []
    assert data_source.requeued == []


async def test_aborted_owned_group_requeues_pristine_reservation(monkeypatch):
    aborted_reservation = make_reservation(3)
    completed_reservation = make_reservation(4)
    data_source = FakeReservationDataSource([aborted_reservation, completed_reservation])

    async def generate(state, group, sampling_params, evaluation=False):
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

    async def generate(state, group, sampling_params, evaluation=False):
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

    async def generate(state, group, sampling_params, evaluation=False):
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

    async def generate(state, group, sampling_params, evaluation=False):
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

    async def generate(state, group, sampling_params, evaluation=False):
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


async def test_owned_retained_limit_does_not_block_active_completion(monkeypatch):
    reservation = make_reservation(32)
    data_source = FakeReservationDataSource([reservation])
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False):
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

    assert data_source.reserved == reservations


@pytest.mark.parametrize(
    "rollback_reason",
    [None, TrainBatchRollbackReason.HANDOFF_FAILED],
)
async def test_owned_lease_settlement_wakes_capacity_waiter_on_lifecycle_loop(
    monkeypatch,
    rollback_reason: TrainBatchRollbackReason | None,
):
    reservations = [make_reservation(20), make_reservation(21)]
    data_source = FakeReservationDataSource(reservations)
    fn = make_owned_fn(monkeypatch, data_source)
    lifecycle_loop = AsyncLoopThread()

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

    async def teardown_on_lifecycle_loop() -> None:
        output_buffer = fn._output
        worker = fn._worker
        ownership = fn._ownership
        assert output_buffer is not None
        assert worker is not None
        assert ownership is not None

        await wait_until(lambda: output_buffer.qsize() == 1)
        worker.cancel()
        [worker_result] = await asyncio.gather(worker, return_exceptions=True)

        assert isinstance(worker_result, asyncio.CancelledError)
        assert await fn._rollback_queued_owned_groups(output_buffer) is None
        assert output_buffer.qsize() == 0
        assert ownership._records == {}

    await asyncio.to_thread(lifecycle_loop.run, teardown_on_lifecycle_loop())

    expected_requeued = [[reservations[1]]] if rollback_reason is None else [[reservations[0]], [reservations[1]]]
    assert data_source.requeued == expected_requeued
    assert data_source.requeued[-1][0] is reservations[1]


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

    async def generate(state, group, sampling_params, evaluation=False):
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

    async def generate(state, group, sampling_params, evaluation=False):
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
    await wait_until(lambda: len(data_source.reserved) == 4)

    assert data_source.reserved == reservations[:4]
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


async def test_worker_error_propagates(monkeypatch):
    async def failing_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        raise RuntimeError("generation exploded")

    fn = make_fn(monkeypatch, make_args(), FakeDataSource(), generate=failing_generate)

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn(RolloutFnTrainInput(rollout_id=0))


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
    defaults = dict(order="fifo", max_groups=None, max_staleness=None, on_evict=evicted.append)
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
