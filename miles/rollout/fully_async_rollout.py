"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
worker's output queue. Rollout production and training consumption run in parallel,
so per-iteration wall time moves from ``rollout_time + train_time`` toward
``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which also requires the class-based
rollout API (``MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1``).

Evaluation targets whatever ``GenerateState`` ``RolloutManager`` passes via
``RolloutFnEvalInput.generate_state`` (see ``miles/rollout/checkpoint_eval.py``
for how the dedicated-fleet state is built). When unset, eval shares the
rollout engines, pausing producer submissions for the duration of the
(blocking) eval.
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable, Coroutine, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TypeVar, cast

import httpx

from miles.rollout.base_types import (
    LeasedRolloutFnTrainOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnLifecycle,
    RolloutFnOutput,
    RolloutFnTrainOutput,
    TrainAdmissionHold,
    TrainBatchLease,
    TrainBatchRollbackReason,
)
from miles.rollout.data_source import SourceReservation
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.fully_async.execution import (
    FullyAsyncExecution,
    FullyAsyncExecutionFailure,
    FullyAsyncExecutionRetry,
    FullyAsyncExecutionSuccess,
    FullyAsyncRetryReason,
    FullyAsyncTerminalPendingError,
)
from miles.rollout.fully_async.ownership import ReservationOwnership, ReservationStageId, ReservationTerminalReceipt
from miles.rollout.inference_rollout.fully_async import InferenceFullyAsyncExecutor
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.http_utils import get
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

OUTPUT_QUEUE_MAX_GROUPS = 1000
NO_PROGRESS_WARN_SECS = 30.0
WEIGHT_VERSION_QUERY_TIMEOUT_SECS = 2.0

# A finished group is list[Sample], or list[list[Sample]] when a generate function
# returns multiple samples per trajectory (e.g. multi-agent).
Group = list[Sample | list[Sample]]
LegacyBufferedGroup = tuple[list[Sample], Group]


@dataclass(frozen=True)
class _OwnedCompletedGroup:
    terminal_receipt: ReservationTerminalReceipt
    samples: Group
    expected_parent_identities: tuple[tuple[int | None, int | None], ...]


@dataclass(frozen=True)
class _OwnedExecutionFailure:
    terminal_receipt: ReservationTerminalReceipt
    error: BaseException


@dataclass(frozen=True)
class _OwnedExecutionRetry:
    terminal_receipt: ReservationTerminalReceipt
    reason: FullyAsyncRetryReason


_OwnedTerminalResult = _OwnedCompletedGroup | _OwnedExecutionRetry | _OwnedExecutionFailure
_OwnedTerminalObserver = Callable[[], Coroutine[object, object, _OwnedTerminalResult]]
_WorkerResult = LegacyBufferedGroup | _OwnedTerminalResult
_TrainAdmissionFrontierTask = asyncio.Task[LegacyBufferedGroup] | asyncio.Task[_OwnedTerminalResult]


@dataclass(frozen=True)
class _ActiveOwnedExecution:
    execution: FullyAsyncExecution
    observe_terminal: _OwnedTerminalObserver


BufferSource = list[Sample] | _OwnedCompletedGroup
BufferEntry = tuple[BufferSource, Group]


class _OwnedTrainAdmissionHold(TrainAdmissionHold):
    def __init__(
        self,
        owner: "FullyAsyncRolloutFn",
        terminal_frontier: tuple[_TrainAdmissionFrontierTask, ...],
    ) -> None:
        super().__init__()
        self._owner = owner
        self._terminal_frontier = terminal_frontier

    async def _wait_terminal(self) -> None:
        await self._owner._wait_train_admission_frontier(self)

    def _release(self) -> None:
        self._owner._release_train_admission_hold(self)


class _OwnedTrainBatchLease(TrainBatchLease):
    def __init__(
        self,
        *,
        rollout_id: int,
        ownership: ReservationOwnership,
        terminal_receipts: list[ReservationTerminalReceipt],
        retained_slots: asyncio.BoundedSemaphore,
        completed_slots: asyncio.Queue[object],
        completed_slot_available: asyncio.Event,
        owned_capacity_released: asyncio.Event,
        on_settled: Callable[[TrainBatchLease], None],
        on_rollback_failed: Callable[[list[ReservationTerminalReceipt]], None],
    ) -> None:
        super().__init__(rollout_id=rollout_id)
        self._owner_loop = asyncio.get_running_loop()
        self._ownership = ownership
        self._terminal_receipts = terminal_receipts
        self._retained_slots = retained_slots
        self._completed_slots = completed_slots
        self._completed_slot_available = completed_slot_available
        self._owned_capacity_released = owned_capacity_released
        self._on_settled = on_settled
        self._on_rollback_failed = on_rollback_failed

    def _commit(self) -> None:
        self._run_on_owner_loop(self._commit_on_owner_loop)

    def _commit_on_owner_loop(self) -> None:
        try:
            self._ownership.commit_batch(self._terminal_receipts, rollout_id=self.rollout_id)
        except BaseException as commit_error:
            try:
                self._ownership.rollback_batch(self._terminal_receipts)
            except BaseException as rollback_error:
                self._on_rollback_failed(self._terminal_receipts)
                self._on_settled(self)
                raise commit_error from rollback_error
            self._release_capacity()
            self._on_settled(self)
            raise
        self._release_capacity()
        self._on_settled(self)

    def _rollback(self, reason: TrainBatchRollbackReason) -> None:
        self._run_on_owner_loop(lambda: self._rollback_on_owner_loop(reason))

    def _rollback_on_owner_loop(self, reason: TrainBatchRollbackReason) -> None:
        try:
            self._ownership.rollback_batch(self._terminal_receipts)
        except BaseException:
            self._on_rollback_failed(self._terminal_receipts)
            self._on_settled(self)
            raise
        self._release_capacity()
        self._on_settled(self)

    def _run_on_owner_loop(self, operation: Callable[[], None]) -> None:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is self._owner_loop:
            operation()
            return

        completion: Future[None] = Future()

        def run_operation() -> None:
            try:
                operation()
            except BaseException as error:
                completion.set_exception(error)
            else:
                completion.set_result(None)

        self._owner_loop.call_soon_threadsafe(run_operation)
        completion.result()

    def _release_capacity(self) -> None:
        for _ in self._terminal_receipts:
            self._retained_slots.release()
            self._completed_slots.put_nowait(object())
        self._completed_slot_available.set()
        self._owned_capacity_released.set()


def _iter_samples(group: Group) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def _first_sample(group: Group) -> Sample:
    return group[0][0] if isinstance(group[0], list) else group[0]


def _owned_group_identity_error(completed: _OwnedCompletedGroup) -> ValueError | None:
    reservation_id = completed.terminal_receipt.executor_receipt.reservation_id
    for position, expected_identity in enumerate(completed.expected_parent_identities):
        item = completed.samples[position]
        samples = item if isinstance(item, list) else [item]
        if any(not isinstance(sample, Sample) for sample in samples):
            return ValueError(
                f"Source reservation {reservation_id} returned non-Sample values at parent slot {position}."
            )
        actual_identities = [(sample.group_index, sample.index) for sample in samples]
        if not actual_identities or any(identity != expected_identity for identity in actual_identities):
            return ValueError(
                f"Source reservation {reservation_id} returned sample identities {actual_identities} "
                f"at parent slot {position}; expected every sample to have identity {expected_identity}."
            )
    return None


def group_oldest_weight_version(group: Group) -> int | None:
    """Return the minimum weight version across all trajectories and turns in a group."""
    versions = [v for s in _iter_samples(group) if (v := s.oldest_weight_version) is not None]
    return min(versions) if versions else None


class DataBuffer:
    """Finished groups waiting between rollout production and training consumption.

    Supported dataflow/staleness control options:

    (1) max groups: use ``--async-data-buffer-max-batches`` to set the max size
        of the buffer, in multiples of rollout_batch_size. On overflow the most
        stale groups are evicted and their sources settled for regeneration;
        0 disables eviction and blocks the producer when the buffer is full.
    (2) order: use ``--async-data-buffer-order`` to set the consumption order,
        fifo (default) or lifo. lifo trains on the freshest group first; pair
        it with (1) and/or ``--max-weight-staleness`` so old groups are evicted
        rather than eventually trained on.
    """

    def __init__(
        self,
        *,
        order: str,
        blocking_capacity: int,
        max_groups: int | None,
        max_staleness: int | None,
        on_evict: Callable[[BufferSource], None],
    ) -> None:
        assert order in ("fifo", "lifo"), f"unknown buffer order: {order}"
        assert max_groups is None or max_groups > 0, f"non-positive buffer capacity: {max_groups}"
        self._order = order
        self._capacity = max_groups if max_groups is not None else blocking_capacity
        self._evict_on_overflow = max_groups is not None
        self._max_staleness = max_staleness
        self._on_evict = on_evict
        self._entries: list[BufferEntry] = []
        self._cond = asyncio.Condition()
        self.entered_groups = 0
        self.evicted_stale_groups = 0
        self.evicted_overflow_groups = 0

    def qsize(self) -> int:
        return len(self._entries)

    async def put(self, entry: BufferEntry, *, current_version: int | None = None) -> None:
        async with self._cond:
            if not self._evict_on_overflow:
                while len(self._entries) >= self._capacity:
                    await self._cond.wait()
            self._entries.append(entry)
            self.entered_groups += 1
            try:
                if self._evict_on_overflow and len(self._entries) > self._capacity:
                    self._evict_overflow(current_version, incoming_entry=entry)
            except BaseException:
                for index, queued_entry in enumerate(self._entries):
                    if queued_entry is entry:
                        self._entries.pop(index)
                        self.entered_groups -= 1
                        break
                self._cond.notify_all()
                raise
            self._cond.notify_all()

    async def get(self) -> BufferEntry:
        async with self._cond:
            while not self._entries:
                await self._cond.wait()
            entry = self._entries.pop() if self._order == "lifo" else self._entries.pop(0)
            self._cond.notify_all()
            return entry

    @staticmethod
    def _eviction_key(group: Group) -> tuple[float, float]:
        """Stalest-first sort key: (min, sum) of weight versions; versionless groups rank freshest."""
        versions = [v for s in _iter_samples(group) if (v := s.oldest_weight_version) is not None]
        if not versions:
            return (float("inf"), float("inf"))
        return (min(versions), sum(versions))

    def _evict_overflow(
        self,
        current_version: int | None,
        *,
        incoming_entry: BufferEntry,
    ) -> None:
        if self._max_staleness is not None and current_version is not None:
            eviction_order = sorted(
                (entry for entry in self._entries if entry is not incoming_entry),
                key=lambda entry: self._eviction_key(entry[1]),
            )
            eviction_order.append(incoming_entry)
            for entry in eviction_order:
                source, group = entry
                oldest = group_oldest_weight_version(group)
                too_stale = oldest is not None and current_version - oldest > self._max_staleness
                if not too_stale:
                    continue
                index = next(
                    (index for index, queued_entry in enumerate(self._entries) if queued_entry is entry),
                    None,
                )
                if index is None:
                    continue
                self._on_evict(source)
                self._entries.pop(index)
                self.evicted_stale_groups += 1
        while len(self._entries) > self._capacity:
            keys = [self._eviction_key(group) for _, group in self._entries]
            index = keys.index(min(keys))
            source, _ = self._entries[index]
            self._on_evict(source)
            self._entries.pop(index)
            self.evicted_overflow_groups += 1

    def staleness_stats(self, current_version: int | None) -> tuple[float, int] | None:
        """(average, max) staleness across buffered groups, or None when unknown."""
        if current_version is None:
            return None
        values = [
            current_version - oldest
            for _, group in self._entries
            if (oldest := group_oldest_weight_version(group)) is not None
        ]
        if not values:
            return None
        return sum(values) / len(values), max(values)

    def reset_counters(self) -> None:
        self.entered_groups = 0
        self.evicted_stale_groups = 0
        self.evicted_overflow_groups = 0

    async def discard_all(self, on_discard: Callable[[BufferSource], None]) -> BaseException | None:
        """Discard buffered entries after their ownership settlement succeeds."""
        first_error: BaseException | None = None
        async with self._cond:
            index = 0
            while index < len(self._entries):
                source, _ = self._entries[index]
                try:
                    on_discard(source)
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                    index += 1
                else:
                    self._entries.pop(index)
            self._cond.notify_all()
        return first_error


class _CachedWeightVersion:
    """Throttled query of the current engine weight version via the router's /model_info."""

    def __init__(self, ttl: float = 1.0):
        self._ttl = ttl
        self._value: int | None = None
        self._last_query = float("-inf")
        self._query_lock = asyncio.Lock()

    async def _query(self, args) -> int:
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/model_info"
        data = await asyncio.wait_for(get(url), timeout=WEIGHT_VERSION_QUERY_TIMEOUT_SECS)
        return int(data["weight_version"])

    async def get(self, args) -> int | None:
        # Throttles failures too: the drain queries once per group, and an unreachable
        # router would otherwise cost every one of them the full timeout.
        if (time.monotonic() - self._last_query) < self._ttl:
            return self._value
        async with self._query_lock:
            if (time.monotonic() - self._last_query) < self._ttl:
                return self._value
            try:
                self._value = await self._query(args)
            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                # Transient router unavailability; the staleness filter is best-effort.
                logger.debug(f"Failed to query engine weight version: {e}")
            finally:
                # Stamped on completion, so a router slower than the TTL still gets throttled.
                self._last_query = time.monotonic()
        return self._value

    async def refresh(self, args) -> int | None:
        """Require a fresh router value without falling back to the cached version."""
        async with self._query_lock:
            self._last_query = -self._ttl
            self._value = None
            try:
                self._value = await self._query(args)
            finally:
                self._last_query = time.monotonic()
        return self._value


class FullyAsyncRolloutFn(RolloutFnLifecycle):
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Groups whose samples were aborted (e.g. by a
    weight update pausing generation) or whose weights are older than
    ``--max-weight-staleness`` are recycled back into the data source.
    """

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        # Default to sample-level backfill for fully async rollout.
        self._scheduler = make_submission_scheduler(input.args, default="sample")
        self._dynamic_filter = load_function(input.args.dynamic_sampling_filter_path)
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._weight_version = _CachedWeightVersion()
        self._worker: asyncio.Task | None = None
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._executor_closed = False
        self._worker_error: BaseException | None = None
        self._worker_failure_reported = False
        self._legacy_executions: dict[asyncio.Task[LegacyBufferedGroup], list[Sample]] = {}
        self._legacy_requeued_groups: deque[LegacyBufferedGroup] = deque()
        self._legacy_close_pending_groups: deque[list[Sample]] = deque()
        self._active_drains: set[asyncio.Task[RolloutFnTrainOutput]] = set()
        self._eval_prompt_dataset_cache: dict = {}
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        self._output: DataBuffer | None = None
        self._train_admission_open = asyncio.Event()
        self._train_admission_open.set()
        self._train_batch_lease_admission_open = asyncio.Event()
        self._train_batch_lease_admission_open.set()
        self._train_admission_epoch = 0
        self._strict_weight_version_epoch: int | None = None
        self._strict_weight_version: int | None = None
        self._strict_weight_version_lock = asyncio.Lock()
        self._train_admission_holds: set[_OwnedTrainAdmissionHold] = set()
        self._open_train_batch_leases: set[TrainBatchLease] = set()
        execution_samples = getattr(self.args, "fully_async_max_execution_samples", None)
        retained_groups = getattr(self.args, "fully_async_max_retained_groups", None)
        completed_groups = getattr(self.args, "fully_async_max_completed_prefetch_groups", None)
        self._uses_owned_capacity = execution_samples is not None
        if self._uses_owned_capacity and (retained_groups is None or completed_groups is None):
            raise ValueError("Fully async ownership requires execution, retained, and completed capacity limits.")
        self._owned_max_in_flight_groups = (
            cast(int, execution_samples) // self.args.n_samples_per_prompt if self._uses_owned_capacity else None
        )
        self._retained_slots = (
            asyncio.BoundedSemaphore(cast(int, retained_groups)) if self._uses_owned_capacity else None
        )
        self._completed_groups = cast(int, completed_groups) if self._uses_owned_capacity else None
        self._ownership = ReservationOwnership(self.data_source) if self._uses_owned_capacity else None
        self._executor = (
            InferenceFullyAsyncExecutor(
                self.state,
                sample_done_callback=self._scheduler.sample_done_callback,
            )
            if self._uses_owned_capacity
            else None
        )
        self._active_executions: dict[
            asyncio.Task[_OwnedTerminalResult],
            _ActiveOwnedExecution,
        ] = {}
        self._pending_reserved_rollbacks: list[SourceReservation] = []
        self._pending_terminal_rollbacks: list[tuple[ReservationTerminalReceipt, bool]] = []
        self._pending_aborted_groups_recycled = 0
        self._next_execution_id = 1
        self._completed_slots: asyncio.Queue[object] | None = None
        self._completed_slot_available = asyncio.Event()
        self._owned_capacity_released = asyncio.Event()
        if self._completed_groups is not None:
            self._completed_slots = asyncio.Queue(maxsize=self._completed_groups)
            for _ in range(self._completed_groups):
                self._completed_slots.put_nowait(object())
            self._completed_slot_available.set()

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if self._closing:
            raise RuntimeError("Fully async rollout function is closed.")
        if input.evaluation:
            return await self._call_eval(input)
        if self._worker is None:
            max_batches = self.args.async_data_buffer_max_batches
            blocking_capacity = (
                self._completed_groups if self._completed_groups is not None else OUTPUT_QUEUE_MAX_GROUPS
            )
            self._output = DataBuffer(
                order=self.args.async_data_buffer_order,
                blocking_capacity=blocking_capacity,
                max_groups=max_batches * self.args.rollout_batch_size if max_batches else None,
                max_staleness=self.args.max_weight_staleness,
                on_evict=self._recycle_buffer_source,
            )
            self._worker = asyncio.create_task(self._worker_loop())
            logger.info("Started fully-async rollout worker")
        drain_task = asyncio.create_task(self._drain(input.rollout_id))
        self._active_drains.add(drain_task)
        try:
            return await drain_task
        finally:
            self._active_drains.discard(drain_task)

    async def prepare_checkpoint(self, rollout_id: int) -> None:
        """Prepare rollout-owned state for checkpoint publication.

        Args:
            rollout_id: Rollout identifier that the checkpoint will publish.
        """
        if self._closing:
            raise RuntimeError("Fully async rollout function is closed.")
        if not self._train_admission_holds:
            raise RuntimeError("Checkpoint preparation requires an active train admission hold.")
        if self._open_train_batch_leases:
            open_rollout_ids = sorted(lease.rollout_id for lease in self._open_train_batch_leases)
            raise RuntimeError(
                f"Cannot prepare checkpoint {rollout_id} with open train batch leases: {open_rollout_ids}."
            )
        pending_rollback_error = self._retry_pending_terminal_rollbacks()
        if pending_rollback_error is not None:
            raise pending_rollback_error
        if self._worker_error is not None:
            raise self._worker_error

    async def acquire_train_admission_hold(self) -> TrainAdmissionHold:
        """Close training admission and return its owned claim."""
        if self._closing:
            raise RuntimeError("Fully async rollout function is closed.")
        self._train_admission_open.clear()
        self._train_batch_lease_admission_open.clear()
        self._train_admission_epoch += 1
        if self._uses_owned_capacity:
            terminal_frontier: tuple[_TrainAdmissionFrontierTask, ...] = tuple(self._active_executions)
        else:
            terminal_frontier = tuple(self._legacy_executions)
        hold = _OwnedTrainAdmissionHold(self, terminal_frontier)
        self._train_admission_holds.add(hold)
        return hold

    async def _wait_train_admission_frontier(self, hold: _OwnedTrainAdmissionHold) -> None:
        outcomes = await asyncio.gather(
            *(asyncio.shield(task) for task in hold._terminal_frontier),
            return_exceptions=True,
        )
        worker = self._worker
        # Owned tasks stay registered through receipt publication or rollback. Legacy
        # tasks have no receipt, so their terminal frontier ends with the task itself.
        while (
            self._worker_error is None
            and any(task in self._active_executions for task in hold._terminal_frontier)
            and (worker is None or not worker.done())
        ):
            await asyncio.sleep(0)
        if self._worker_error is not None:
            raise self._worker_error
        if worker is not None and worker.done() and not worker.cancelled():
            worker_error = worker.exception()
            if worker_error is not None:
                raise worker_error
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
            if isinstance(outcome, _OwnedExecutionFailure):
                raise outcome.error

    def _release_train_admission_hold(self, hold: _OwnedTrainAdmissionHold) -> None:
        if hold not in self._train_admission_holds:
            raise RuntimeError("Train admission hold is not active on this rollout function.")
        self._train_admission_holds.remove(hold)
        if not self._train_admission_holds and not self._closing:
            self._train_admission_epoch += 1
            self._train_admission_open.set()
            self._train_batch_lease_admission_open.set()

    async def _strict_weight_version_for_admission_epoch(
        self,
        admission_epoch: int,
    ) -> tuple[int | None, bool]:
        async with self._strict_weight_version_lock:
            if self._closing:
                raise RuntimeError("Fully async rollout closed before the train batch lease was issued.")
            if not self._train_batch_lease_admission_open.is_set() or admission_epoch != self._train_admission_epoch:
                return None, False
            if self._strict_weight_version_epoch != admission_epoch:
                current = await self._weight_version.refresh(self.args)
                if self._closing:
                    raise RuntimeError("Fully async rollout closed before the train batch lease was issued.")
                if (
                    not self._train_batch_lease_admission_open.is_set()
                    or admission_epoch != self._train_admission_epoch
                ):
                    return None, False
                self._strict_weight_version_epoch = admission_epoch
                self._strict_weight_version = current
            return self._strict_weight_version, True

    def _settle_train_batch_lease(self, lease: TrainBatchLease) -> None:
        if lease not in self._open_train_batch_leases:
            raise RuntimeError(f"Train batch lease for rollout {lease.rollout_id} is not open.")
        self._open_train_batch_leases.remove(lease)

    def _retain_failed_train_batch_rollback(
        self,
        terminal_receipts: list[ReservationTerminalReceipt],
    ) -> None:
        self._pending_terminal_rollbacks.extend((terminal_receipt, True) for terminal_receipt in terminal_receipts)

    async def close(self) -> None:
        """Stop rollout production and settle every retained reservation.

        Returns:
            None after active executions are terminal and retained reservations
            are requeued.

        Raises:
            BaseException: The first cleanup or terminal execution failure. A
                later call retries retained cleanup that did not settle.
        """
        if self._closed:
            return
        close_task = self._close_task
        if close_task is None:
            close_task = asyncio.create_task(self._close())
            self._close_task = close_task
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancellation:
            try:
                await _await_task_terminal(close_task)
            except BaseException as terminal_error:
                raise cancellation from terminal_error
            raise
        finally:
            if self._close_task is close_task and close_task.done() and not self._closed:
                self._close_task = None

    async def _close(self) -> None:
        self._closing = True
        self._train_admission_holds.clear()
        self._train_admission_open.clear()
        self._train_batch_lease_admission_open.set()
        if self._open_train_batch_leases:
            open_rollout_ids = sorted(lease.rollout_id for lease in self._open_train_batch_leases)
            raise RuntimeError(f"Cannot close fully async rollout with open train batch leases: {open_rollout_ids}.")
        cleanup_error: BaseException | None = None
        shutdown_error: BaseException | None = None

        worker = self._worker
        if worker is not None:
            if not worker.done():
                worker.cancel()
            worker_result: None | BaseException = (await asyncio.gather(worker, return_exceptions=True))[0]
            if not self._worker_failure_reported:
                recorded_worker_error = self._worker_error
                if recorded_worker_error is not None and not isinstance(recorded_worker_error, asyncio.CancelledError):
                    shutdown_error = recorded_worker_error
                elif isinstance(worker_result, BaseException) and not isinstance(
                    worker_result, asyncio.CancelledError
                ):
                    shutdown_error = worker_result

        legacy_executions = tuple(self._legacy_executions.items())
        for task, _ in legacy_executions:
            if not task.done():
                task.cancel()
        if legacy_executions:
            results = await asyncio.gather(
                *(task for task, _ in legacy_executions),
                return_exceptions=True,
            )
            for (task, source_group), _ in zip(legacy_executions, results, strict=True):
                self._legacy_close_pending_groups.append(source_group)
                self._legacy_executions.pop(task, None)

        active_drains = tuple(self._active_drains)
        if active_drains:
            await asyncio.gather(*active_drains, return_exceptions=True)

        output = self._output
        if not self._uses_owned_capacity and output is not None:

            def retain_legacy_source(source: BufferSource) -> None:
                if isinstance(source, _OwnedCompletedGroup):
                    raise RuntimeError("Legacy fully async output buffer contained an owned group.")
                self._legacy_close_pending_groups.append(source)

            buffered_retain_error = await output.discard_all(retain_legacy_source)
            if buffered_retain_error is not None:
                cleanup_error = buffered_retain_error
            while self._legacy_requeued_groups:
                source_group, _ = self._legacy_requeued_groups.popleft()
                self._legacy_close_pending_groups.append(source_group)

        legacy_recycle_error = self._retry_pending_legacy_recycles()
        if legacy_recycle_error is not None:
            cleanup_error = legacy_recycle_error

        acquisition_rollback_error = self._retry_pending_acquisition_rollback()
        if cleanup_error is None:
            cleanup_error = acquisition_rollback_error

        reserved_rollback_error = self._retry_pending_reserved_rollbacks()
        if cleanup_error is None:
            cleanup_error = reserved_rollback_error

        pending_rollback_error = self._retry_pending_terminal_rollbacks()
        if cleanup_error is None:
            cleanup_error = pending_rollback_error

        active = list(self._active_executions.items())
        terminal_waits: list[
            tuple[
                asyncio.Task[_OwnedTerminalResult],
                _ActiveOwnedExecution,
            ]
        ] = []
        for terminal_task, active_execution in active:
            if _terminal_observation_needs_retry(terminal_task):
                replacement: asyncio.Task[_OwnedTerminalResult] = asyncio.create_task(
                    active_execution.observe_terminal()
                )
                del self._active_executions[terminal_task]
                self._active_executions[replacement] = active_execution
                terminal_task = replacement
            if not terminal_task.done():
                try:
                    active_execution.execution.request_cancellation()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
                    continue
            terminal_waits.append((terminal_task, active_execution))

        for terminal_task, _ in terminal_waits:
            try:
                result = await asyncio.shield(terminal_task)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                continue
            if not isinstance(result, (_OwnedCompletedGroup, _OwnedExecutionRetry, _OwnedExecutionFailure)):
                if cleanup_error is None:
                    cleanup_error = RuntimeError(f"Fully async close observed unsupported {type(result).__name__}.")
                continue
            try:
                self._rollback_owned_terminal(result.terminal_receipt, completed_slot_held=False)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                continue
            self._active_executions.pop(terminal_task, None)
            if (
                isinstance(result, _OwnedExecutionFailure)
                and not isinstance(result.error, asyncio.CancelledError)
                and shutdown_error is None
            ):
                shutdown_error = result.error

        if self._uses_owned_capacity and output is not None:
            queued_error = await self._rollback_queued_owned_groups(output)
            if cleanup_error is None:
                cleanup_error = queued_error

        if not self._active_executions and not self._executor_closed and self._executor is not None:
            try:
                await self._executor.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            else:
                self._executor_closed = True

        if cleanup_error is not None:
            raise cleanup_error
        if self._active_executions:
            raise RuntimeError("Fully async close did not settle every active execution.")
        if self._pending_reserved_rollbacks:
            raise RuntimeError("Fully async close did not settle every retained reserved rollback.")
        if self._pending_terminal_rollbacks:
            raise RuntimeError("Fully async close did not settle every retained terminal rollback.")
        if self._ownership is not None and self._ownership.has_pending_acquisition_rollback:
            raise RuntimeError("Fully async close did not settle the retained acquisition rollback.")
        if self._legacy_close_pending_groups:
            raise RuntimeError("Fully async close did not recycle every retained legacy group.")
        if shutdown_error is not None:
            self._worker_failure_reported = True
            raise shutdown_error
        self._closed = True

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnOutput:
        if input.generate_state is not None:
            results = await run_eval_datasets(input.generate_state, self._eval_prompt_dataset_cache)
            return RolloutFnEvalOutput(data=results)

        logger.info("Pausing fully-async producer submissions for shared-engine eval")
        self._producer_resumed.clear()
        try:
            results = await run_eval_datasets(self.state, self._eval_prompt_dataset_cache)
        finally:
            self._producer_resumed.set()
            logger.info("Resumed fully-async producer submissions after eval")
        return RolloutFnEvalOutput(data=results)

    # -------------------------- producer --------------------------

    def _max_in_flight_groups(self) -> int:
        if self._owned_max_in_flight_groups is not None:
            return self._owned_max_in_flight_groups
        if (x := self.args.async_max_concurrent_samples) is not None:
            # Whole groups are submitted, so the sample budget floors to a group count.
            return max(1, x // self.args.n_samples_per_prompt)
        return self.args.rollout_batch_size

    async def _generate_group(self, prompt_group: list[Sample]) -> Group:
        """Return the submitted prompt group next to its result.

        A retry has to resubmit the prompt group: a generate function may expand one
        trajectory into several samples, and ``generate_and_rm_group`` does not accept
        that shape back.
        """
        callback = self._scheduler.sample_done_callback
        if callback is None:
            return cast(
                Group,
                await generate_and_rm_group(
                    self.state,
                    prompt_group,
                    sampling_params=self.state.sampling_params.copy(),
                    evaluation=False,
                ),
            )
        return cast(
            Group,
            await generate_and_rm_group(
                self.state,
                prompt_group,
                sampling_params=self.state.sampling_params.copy(),
                evaluation=False,
                sample_done_callback=callback,
            ),
        )

    def _submit_one_group(
        self,
    ) -> asyncio.Task[_WorkerResult]:
        if not self._uses_owned_capacity:
            prompt_groups = self.data_source.get_samples(1)
            self._scheduler.on_submit(prompt_groups)
            [prompt_group] = prompt_groups

            async def execute_legacy() -> LegacyBufferedGroup:
                return prompt_group, await self._generate_group(prompt_group)

            task = asyncio.create_task(execute_legacy())
            self._legacy_executions[task] = prompt_group
            return task

        ownership = self._ownership
        retained_slots = self._retained_slots
        if ownership is None or retained_slots is None:
            raise RuntimeError("Fully async ownership is not initialized.")
        try:
            [reservation] = ownership.reserve_samples(1)
        except Exception:
            if not ownership.has_pending_acquisition_rollback:
                retained_slots.release()
            raise
        try:
            expected_parents = self.args.n_samples_per_prompt
            if len(reservation.samples) != expected_parents:
                raise ValueError(
                    f"Source reservation {reservation.reservation_id} contains {len(reservation.samples)} "
                    f"parent slots; expected {expected_parents}."
                )
            expected_parent_identities = tuple((sample.group_index, sample.index) for sample in reservation.samples)
            for position, identity in enumerate(expected_parent_identities):
                if identity[0] is None or identity[1] is None:
                    raise ValueError(
                        f"Source reservation {reservation.reservation_id} has incomplete parent identity "
                        f"{identity} at slot {position}."
                    )
            if len(expected_parent_identities) != len(set(expected_parent_identities)):
                raise ValueError(
                    f"Source reservation {reservation.reservation_id} has duplicate parent identities: "
                    f"{list(expected_parent_identities)}."
                )
            stage_id = ReservationStageId(f"execution-{self._next_execution_id}")
            self._next_execution_id += 1
            [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
        except Exception as validation_error:
            try:
                ownership.rollback_reserved([reservation])
            except BaseException as rollback_error:
                self._pending_reserved_rollbacks.append(reservation)
                raise validation_error from rollback_error
            retained_slots.release()
            raise

        executor = self._executor
        if executor is None:
            raise RuntimeError("Fully async executor is not initialized.")
        try:
            execution = executor.submit(reservation, executor_receipt)
        except BaseException as submission_error:
            try:
                [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)
            except BaseException as terminal_error:
                raise submission_error from terminal_error
            try:
                ownership.rollback_batch([terminal_receipt])
            except BaseException as settlement_error:
                self._pending_terminal_rollbacks.append((terminal_receipt, False))
                raise submission_error from settlement_error
            retained_slots.release()
            raise
        self._scheduler.on_submit([list(reservation.samples)])

        async def observe_terminal() -> _OwnedCompletedGroup | _OwnedExecutionRetry | _OwnedExecutionFailure:
            outcome = await execution.wait_terminal()
            if outcome.executor_receipt is not executor_receipt:
                # A foreign receipt cannot prove this reservation terminal; retain ownership fail-closed.
                raise RuntimeError(
                    f"Execution receipt {executor_receipt.receipt_id} did not return its exact terminal receipt."
                )
            [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)
            if isinstance(outcome, FullyAsyncExecutionFailure):
                return _OwnedExecutionFailure(terminal_receipt=terminal_receipt, error=outcome.error)
            if isinstance(outcome, FullyAsyncExecutionRetry):
                return _OwnedExecutionRetry(
                    terminal_receipt=terminal_receipt,
                    reason=outcome.reason,
                )
            if not isinstance(outcome, FullyAsyncExecutionSuccess):
                raise RuntimeError(f"Fully async execution returned unsupported {type(outcome).__name__}.")
            return _OwnedCompletedGroup(
                terminal_receipt=terminal_receipt,
                samples=outcome.samples,
                expected_parent_identities=expected_parent_identities,
            )

        terminal_task = asyncio.create_task(observe_terminal())
        self._active_executions[terminal_task] = _ActiveOwnedExecution(
            execution=execution,
            observe_terminal=observe_terminal,
        )
        return terminal_task

    async def _acquire_retained_slot(self) -> bool:
        if self._retained_slots is None:
            return False
        await self._retained_slots.acquire()
        if self._producer_resumed.is_set() and self._train_admission_open.is_set():
            return True
        self._retained_slots.release()
        return False

    async def _submit_active_group(
        self,
        active: set[asyncio.Task[_WorkerResult]],
    ) -> bool:
        retained_slot_acquired = await self._acquire_retained_slot()
        if self._uses_owned_capacity and not retained_slot_acquired:
            return False
        if not self._producer_resumed.is_set() or not self._train_admission_open.is_set():
            if retained_slot_acquired:
                retained_slots = self._retained_slots
                if retained_slots is None:
                    raise RuntimeError("Fully async retained capacity is not initialized.")
                retained_slots.release()
            return False
        active.add(self._submit_one_group())
        return True

    def _owned_retained_capacity_available(self) -> bool:
        return self._retained_slots is None or not self._retained_slots.locked()

    def _owned_completed_capacity_available(self) -> bool:
        return self._completed_slots is None or not self._completed_slots.empty()

    def _try_acquire_completed_slot(self) -> bool:
        completed_slots = self._completed_slots
        if completed_slots is None:
            return False
        try:
            completed_slots.get_nowait()
        except asyncio.QueueEmpty:
            self._completed_slot_available.clear()
            return False
        if completed_slots.empty():
            self._completed_slot_available.clear()
        return True

    def _release_owned_capacity(
        self,
        *,
        terminal_receipts: list[ReservationTerminalReceipt],
        completed_slots: int,
    ) -> None:
        retained_slots = self._retained_slots
        owned_completed_slots = self._completed_slots
        if retained_slots is None or owned_completed_slots is None:
            raise RuntimeError("Fully async ownership capacity is not initialized.")
        for _ in terminal_receipts:
            retained_slots.release()
        for _ in range(completed_slots):
            owned_completed_slots.put_nowait(object())
        if completed_slots:
            self._completed_slot_available.set()
        if terminal_receipts or completed_slots:
            self._owned_capacity_released.set()

    def _rollback_owned_terminal(
        self,
        terminal_receipt: ReservationTerminalReceipt,
        *,
        completed_slot_held: bool,
    ) -> None:
        self._rollback_owned_terminals(
            [terminal_receipt],
            completed_slots=int(completed_slot_held),
        )

    def _commit_owned_terminal(
        self,
        terminal_receipt: ReservationTerminalReceipt,
        *,
        rollout_id: int,
        completed_slot_held: bool,
    ) -> None:
        ownership = self._ownership
        if ownership is None:
            raise RuntimeError("Fully async ownership is not initialized.")
        ownership.commit_batch([terminal_receipt], rollout_id=rollout_id)
        self._release_owned_capacity(
            terminal_receipts=[terminal_receipt],
            completed_slots=int(completed_slot_held),
        )

    def _rollback_owned_terminals(
        self,
        terminal_receipts: list[ReservationTerminalReceipt],
        *,
        completed_slots: int,
    ) -> None:
        ownership = self._ownership
        if ownership is None:
            raise RuntimeError("Fully async ownership is not initialized.")
        ownership.rollback_batch(terminal_receipts)
        self._release_owned_capacity(
            terminal_receipts=terminal_receipts,
            completed_slots=completed_slots,
        )

    async def _rollback_queued_owned_groups(
        self,
        output: DataBuffer,
    ) -> BaseException | None:
        def rollback(source: BufferSource) -> None:
            if not isinstance(source, _OwnedCompletedGroup):
                raise RuntimeError(f"Owned fully async output buffer contained unsupported {type(source).__name__}.")
            self._rollback_owned_terminal(source.terminal_receipt, completed_slot_held=True)

        return await output.discard_all(rollback)

    def _retry_pending_terminal_rollbacks(self) -> BaseException | None:
        while self._pending_terminal_rollbacks:
            terminal_receipt, completed_slot_held = self._pending_terminal_rollbacks[0]
            try:
                self._rollback_owned_terminal(
                    terminal_receipt,
                    completed_slot_held=completed_slot_held,
                )
            except BaseException as error:
                return error
            del self._pending_terminal_rollbacks[0]
        return None

    def _retry_pending_reserved_rollbacks(self) -> BaseException | None:
        ownership = self._ownership
        retained_slots = self._retained_slots
        if ownership is None or retained_slots is None:
            if self._pending_reserved_rollbacks:
                return RuntimeError("Fully async ownership capacity is not initialized.")
            return None
        while self._pending_reserved_rollbacks:
            reservation = self._pending_reserved_rollbacks[0]
            try:
                ownership.rollback_reserved([reservation])
            except BaseException as error:
                return error
            retained_slots.release()
            del self._pending_reserved_rollbacks[0]
        return None

    def _retry_pending_acquisition_rollback(self) -> BaseException | None:
        ownership = self._ownership
        retained_slots = self._retained_slots
        if ownership is None or not ownership.has_pending_acquisition_rollback:
            return None
        if retained_slots is None:
            return RuntimeError("Fully async ownership capacity is not initialized.")
        try:
            ownership.retry_failed_acquisition_rollback()
        except BaseException as error:
            return error
        retained_slots.release()
        return None

    def _retry_pending_legacy_recycles(self) -> BaseException | None:
        while self._legacy_close_pending_groups:
            try:
                self._recycle(self._legacy_close_pending_groups[0])
            except BaseException as error:
                return error
            self._legacy_close_pending_groups.popleft()
        return None

    def _record_worker_error(self, error: BaseException) -> BaseException:
        if self._worker_error is None:
            self._worker_error = error
        return self._worker_error

    async def _worker_loop(self) -> None:
        output = self._output
        if output is None:
            raise RuntimeError("Fully async output buffer is not initialized.")
        active: set[asyncio.Task[_WorkerResult]] = set()
        cancellation_requested: set[asyncio.Task[_OwnedTerminalResult]] = set()
        fatal_error: BaseException | None = None
        fatal_settlement_error: BaseException | None = None
        while True:
            scheduler_blocked = False
            ownership_blocked = False
            self._owned_capacity_released.clear()
            if fatal_error is None:
                if not active:
                    await self._producer_resumed.wait()
                    await self._train_admission_open.wait()
                if self._producer_resumed.is_set() and self._train_admission_open.is_set():
                    while True:
                        if self._uses_owned_capacity and not self._owned_completed_capacity_available():
                            ownership_blocked = True
                            break
                        if self._uses_owned_capacity and active and not self._owned_retained_capacity_available():
                            ownership_blocked = True
                            break
                        if not self._scheduler.has_capacity(
                            pending_groups=len(active),
                            group_budget=self._max_in_flight_groups(),
                        ):
                            scheduler_blocked = True
                            break
                        try:
                            submitted = await self._submit_active_group(active)
                        except Exception as submission_error:
                            if not self._uses_owned_capacity:
                                self._record_worker_error(submission_error)
                                raise
                            fatal_error = self._record_worker_error(submission_error)
                            break
                        if not submitted:
                            break
                        if self._uses_owned_capacity:
                            # Let terminal work claim reopened prefetch capacity before
                            # another reservation is admitted.
                            await asyncio.sleep(0)
                            if any(task.done() for task in active):
                                break
            if fatal_error is not None:
                queued_settlement_error = await self._rollback_queued_owned_groups(output)
                if fatal_settlement_error is None:
                    fatal_settlement_error = queued_settlement_error
                cancellation_error: BaseException | None = None
                if self._uses_owned_capacity:
                    for task in active:
                        owned_task = cast(asyncio.Task[_OwnedTerminalResult], task)
                        if owned_task in cancellation_requested:
                            continue
                        active_execution = self._active_executions.get(owned_task)
                        if active_execution is None:
                            if cancellation_error is None:
                                cancellation_error = RuntimeError(
                                    "Fully async worker lost an active execution record."
                                )
                            continue
                        try:
                            active_execution.execution.request_cancellation()
                        except BaseException as error:
                            if cancellation_error is None:
                                cancellation_error = error
                        else:
                            cancellation_requested.add(owned_task)
                if cancellation_error is not None:
                    raise fatal_error from cancellation_error
            if not active:
                if fatal_error is not None:
                    if fatal_settlement_error is not None:
                        raise fatal_error from fatal_settlement_error
                    raise fatal_error
                if self._uses_owned_capacity and not self._owned_completed_capacity_available():
                    await self._completed_slot_available.wait()
                    continue
                if not self._producer_resumed.is_set():
                    await self._producer_resumed.wait()
                    continue
                if not self._train_admission_open.is_set():
                    await self._train_admission_open.wait()
                    continue
                raise RuntimeError("Fully async scheduler has admission capacity but no active work.")
            if fatal_error is None and self._producer_resumed.is_set():
                if self._train_admission_open.is_set():
                    if ownership_blocked:
                        capacity_waiter = asyncio.create_task(self._owned_capacity_released.wait())
                        try:
                            ready, _ = await asyncio.wait(
                                [*active, capacity_waiter],
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                        finally:
                            capacity_waiter.cancel()
                            await asyncio.gather(capacity_waiter, return_exceptions=True)
                        done = active.intersection(ready)
                        active.difference_update(done)
                    elif scheduler_blocked:
                        done, active = await self._scheduler.wait_for_progress(active)
                    else:
                        done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                else:
                    admission_waiter = asyncio.create_task(self._train_admission_open.wait())
                    try:
                        ready, _ = await asyncio.wait(
                            [*active, admission_waiter],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        admission_waiter.cancel()
                        await asyncio.gather(admission_waiter, return_exceptions=True)
                    done = active.intersection(ready)
                    active.difference_update(done)
            else:
                done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                continue
            if not self._uses_owned_capacity:
                completed_groups: list[LegacyBufferedGroup] = []
                done_source_groups: list[list[Sample]] = []
                task_error: BaseException | None = None
                for task in done:
                    legacy_task = cast(asyncio.Task[LegacyBufferedGroup], task)
                    source_group = self._legacy_executions[legacy_task]
                    done_source_groups.append(source_group)
                    try:
                        completed_group = legacy_task.result()
                    except BaseException as error:
                        if task_error is None:
                            task_error = error
                    else:
                        completed_groups.append(completed_group)
                if task_error is not None:
                    active_tasks = [cast(asyncio.Task[LegacyBufferedGroup], task) for task in active]
                    active_source_groups = [self._legacy_executions[task] for task in active_tasks]
                    for task in active_tasks:
                        task.cancel()
                    await asyncio.gather(*active_tasks, return_exceptions=True)
                    self._legacy_close_pending_groups.extend(done_source_groups)
                    self._legacy_close_pending_groups.extend(active_source_groups)
                    for task in done:
                        self._legacy_executions.pop(cast(asyncio.Task[LegacyBufferedGroup], task), None)
                    for task in active_tasks:
                        self._legacy_executions.pop(task, None)
                    self._record_worker_error(task_error)
                    raise task_error
                for task in done:
                    self._legacy_executions.pop(cast(asyncio.Task[LegacyBufferedGroup], task), None)
                for position, completed_group in enumerate(completed_groups):
                    try:
                        version = await self._weight_version.get(self.args)
                        await output.put(completed_group, current_version=version)
                    except BaseException as error:
                        self._legacy_requeued_groups.extend(completed_groups[position:])
                        self._record_worker_error(error)
                        raise
                continue
            for task in done:
                owned_task = cast(asyncio.Task[_OwnedTerminalResult], task)
                try:
                    result = owned_task.result()
                except BaseException as task_error:
                    if fatal_error is None:
                        fatal_error = self._record_worker_error(task_error)
                    continue
                if isinstance(result, _OwnedExecutionRetry):
                    try:
                        self._rollback_owned_terminal(result.terminal_receipt, completed_slot_held=False)
                    except BaseException as settlement_error:
                        self._pending_terminal_rollbacks.append((result.terminal_receipt, False))
                        if fatal_error is None:
                            fatal_error = self._record_worker_error(settlement_error)
                    else:
                        if result.reason is FullyAsyncRetryReason.EXECUTION_ABORTED:
                            self._pending_aborted_groups_recycled += 1
                    self._active_executions.pop(owned_task, None)
                    continue
                if isinstance(result, _OwnedExecutionFailure):
                    try:
                        self._rollback_owned_terminal(result.terminal_receipt, completed_slot_held=False)
                    except BaseException as settlement_error:
                        self._pending_terminal_rollbacks.append((result.terminal_receipt, False))
                        if fatal_settlement_error is None:
                            fatal_settlement_error = settlement_error
                    if fatal_error is None:
                        fatal_error = self._record_worker_error(result.error)
                    self._active_executions.pop(owned_task, None)
                    continue
                if not isinstance(result, _OwnedCompletedGroup):
                    raise RuntimeError(f"Owned fully async execution returned unsupported {type(result).__name__}.")
                if fatal_error is not None or not self._try_acquire_completed_slot():
                    try:
                        self._rollback_owned_terminal(result.terminal_receipt, completed_slot_held=False)
                    except BaseException as settlement_error:
                        self._pending_terminal_rollbacks.append((result.terminal_receipt, False))
                        if fatal_error is None:
                            fatal_error = self._record_worker_error(settlement_error)
                        elif fatal_settlement_error is None:
                            fatal_settlement_error = settlement_error
                    self._active_executions.pop(owned_task, None)
                    continue
                try:
                    version = await self._weight_version.get(self.args)
                    await output.put((result, result.samples), current_version=version)
                except BaseException as buffer_error:
                    try:
                        self._rollback_owned_terminal(result.terminal_receipt, completed_slot_held=True)
                    except BaseException as settlement_error:
                        self._pending_terminal_rollbacks.append((result.terminal_receipt, True))
                        if fatal_settlement_error is None:
                            fatal_settlement_error = settlement_error
                    self._active_executions.pop(owned_task, None)
                    if fatal_error is None:
                        fatal_error = self._record_worker_error(buffer_error)
                else:
                    self._active_executions.pop(owned_task, None)

    # -------------------------- consumer --------------------------

    async def _next_group(self) -> BufferEntry:
        output = self._output
        worker = self._worker
        if output is None or worker is None:
            raise RuntimeError("Fully async worker is not initialized.")
        if self._legacy_requeued_groups:
            return self._legacy_requeued_groups.popleft()
        queue_get = asyncio.create_task(output.get())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {queue_get, worker},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=NO_PROGRESS_WARN_SECS,
                )
                # A dead worker fails the step before its buffered backlog is consumed.
                if worker in done:
                    worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                if queue_get in done:
                    return queue_get.result()
                logger.warning(f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s (queued: {output.qsize()})")
        except BaseException as error:
            try:
                await self._settle_failed_queue_get(queue_get, output)
            except BaseException as settlement_error:
                raise error from settlement_error
            raise

    async def _settle_failed_queue_get(
        self,
        queue_get: asyncio.Task[BufferEntry],
        output: DataBuffer,
    ) -> None:
        if not queue_get.done():
            queue_get.cancel()
        await asyncio.gather(queue_get, return_exceptions=True)
        if queue_get.cancelled():
            return

        completed = queue_get.result()
        source, _ = completed
        if isinstance(source, _OwnedCompletedGroup):
            try:
                self._rollback_owned_terminal(source.terminal_receipt, completed_slot_held=True)
            except BaseException:
                self._pending_terminal_rollbacks.append((source.terminal_receipt, True))
                raise
            return
        self._legacy_requeued_groups.append(completed)

    async def _drain(self, rollout_id: int) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset
        output = self._output
        if output is None:
            raise RuntimeError("Fully async output queue is not initialized.")

        target_data_size = args.rollout_batch_size
        data: list[Group] = []
        accepted_legacy_groups: list[LegacyBufferedGroup] = []
        claimed_legacy_group: LegacyBufferedGroup | None = None
        terminal_receipts: list[ReservationTerminalReceipt] = []
        strict_validation_epochs: list[int | None] = []
        aborted_groups_recycled = 0
        stale_groups_recycled = 0
        staleness_values: list[int] = []
        metric_gatherer = MetricGatherer()
        do_print = True
        buffer_stats_weight_version: int | None = None

        try:
            while True:
                while len(data) < target_data_size:
                    buffered_group = await self._next_group()
                    source, group = buffered_group
                    if isinstance(source, _OwnedCompletedGroup):
                        prompt_group = None
                        terminal_receipt = source.terminal_receipt
                        terminal_receipts.append(terminal_receipt)
                    else:
                        prompt_group = source
                        terminal_receipt = None
                        claimed_legacy_group = buffered_group

                    if len(group) != args.n_samples_per_prompt:
                        if terminal_receipt is None:
                            raise AssertionError(
                                f"Generated group contains {len(group)} parent slots; "
                                f"expected {args.n_samples_per_prompt}."
                            )
                        raise ValueError(
                            f"Source reservation {terminal_receipt.executor_receipt.reservation_id} returned "
                            f"{len(group)} parent slots; expected {args.n_samples_per_prompt}."
                        )
                    if isinstance(source, _OwnedCompletedGroup):
                        identity_error = _owned_group_identity_error(source)
                        if identity_error is not None:
                            raise identity_error

                    # A weight update paused generation mid-group: return it for re-sampling.
                    if any(s.status == Sample.Status.ABORTED for s in _iter_samples(group)):
                        if terminal_receipt is None:
                            assert prompt_group is not None
                            self._recycle(prompt_group)
                            claimed_legacy_group = None
                        else:
                            self._rollback_owned_terminal(terminal_receipt, completed_slot_held=True)
                            assert terminal_receipts.pop() is terminal_receipt
                        aborted_groups_recycled += 1
                        continue

                    validation_epoch = self._train_admission_epoch
                    strict_validation_epoch = (
                        validation_epoch if self._strict_weight_version_epoch == validation_epoch else None
                    )
                    oldest = group_oldest_weight_version(group)
                    current = (
                        self._strict_weight_version
                        if strict_validation_epoch is not None
                        else await self._weight_version.get(args)
                    )
                    if oldest is not None and current is not None:
                        staleness = current - oldest
                        staleness_values.append(staleness)
                        if args.max_weight_staleness is not None and staleness > args.max_weight_staleness:
                            if terminal_receipt is None:
                                assert prompt_group is not None
                                self._recycle(prompt_group)
                                claimed_legacy_group = None
                            else:
                                self._rollback_owned_terminal(terminal_receipt, completed_slot_held=True)
                                assert terminal_receipts.pop() is terminal_receipt
                            stale_groups_recycled += 1
                            logger.info(
                                f"Recycled stale group (oldest_version={oldest}, current={current}, "
                                f"staleness={staleness} > max={args.max_weight_staleness})"
                            )
                            continue

                    filter_output = call_dynamic_filter(self._dynamic_filter, args, group)
                    if not filter_output.keep:
                        if terminal_receipt is not None:
                            self._commit_owned_terminal(
                                terminal_receipt,
                                rollout_id=rollout_id,
                                completed_slot_held=True,
                            )
                            assert terminal_receipts.pop() is terminal_receipt
                        else:
                            claimed_legacy_group = None
                        # Filtered groups are consumed, not replayed: they have no usable gradient signal.
                        metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
                        continue

                    if do_print:
                        sample = group[0][0] if isinstance(group[0], list) else group[0]
                        logger.info(
                            f"First rollout sample: {[str(sample.prompt) + sample.response]}, "
                            f"label: {sample.label}, reward: {sample.reward}"
                        )
                        do_print = False

                    data.append(group)
                    if terminal_receipt is None:
                        assert not isinstance(source, _OwnedCompletedGroup)
                        accepted_legacy_groups.append(buffered_group)
                    if args.max_weight_staleness is not None:
                        strict_validation_epochs.append(strict_validation_epoch)
                    claimed_legacy_group = None

                buffer_stats_weight_version = await self._weight_version.get(args)
                if self._retained_slots is None:
                    break
                await self._train_batch_lease_admission_open.wait()
                if self._closing:
                    raise RuntimeError("Fully async rollout closed before the train batch lease was issued.")
                admission_epoch = self._train_admission_epoch
                if args.max_weight_staleness is None:
                    break
                if admission_epoch == 0:
                    break

                current, admission_current = await self._strict_weight_version_for_admission_epoch(admission_epoch)
                if not admission_current:
                    continue

                revalidation_indexes = [
                    index for index, epoch in enumerate(strict_validation_epochs) if epoch != admission_epoch
                ]
                stale_groups: list[tuple[int, int, int]] = []
                if current is not None:
                    for index in revalidation_indexes:
                        group = data[index]
                        oldest = group_oldest_weight_version(group)
                        if oldest is None:
                            continue
                        staleness = current - oldest
                        staleness_values.append(staleness)
                        if staleness > args.max_weight_staleness:
                            stale_groups.append((index, oldest, staleness))
                for index in revalidation_indexes:
                    strict_validation_epochs[index] = admission_epoch
                for index, oldest, staleness in reversed(stale_groups):
                    terminal_receipt = terminal_receipts[index]
                    self._rollback_owned_terminal(terminal_receipt, completed_slot_held=True)
                    assert terminal_receipts.pop(index) is terminal_receipt
                    del data[index]
                    del strict_validation_epochs[index]
                    stale_groups_recycled += 1
                    logger.info(
                        f"Recycled stale group (oldest_version={oldest}, current={current}, "
                        f"staleness={staleness} > max={args.max_weight_staleness})"
                    )
                if stale_groups:
                    continue
                buffer_stats_weight_version = current
                break

            sample = _first_sample(data[-1])
            logger.info(
                f"Finish rollout: {[str(sample.prompt) + sample.response]}, "
                f"label: {sample.label}, reward: {sample.reward}"
            )

            data.sort(key=lambda group: _first_sample(group).index)

            if self._uses_owned_capacity and self._closing:
                raise RuntimeError("Fully async rollout closed before the train batch lease was issued.")

            if self._sample_filter is not None:
                self._sample_filter(args, data)

            aborted_groups_recycled += self._pending_aborted_groups_recycled
            self._pending_aborted_groups_recycled = 0
            metrics: dict[str, int | float] = {
                "rollout/fully_async/queue_size": output.qsize() + len(self._legacy_requeued_groups),
                "rollout/fully_async/aborted_groups_recycled": aborted_groups_recycled,
                "rollout/fully_async/stale_groups_recycled": stale_groups_recycled,
                "rollout/fully_async/evicted_stale_groups": output.evicted_stale_groups,
                "rollout/fully_async/evicted_overflow_groups": output.evicted_overflow_groups,
                **metric_gatherer.collect(),
            }
            if output.entered_groups:
                evicted = output.evicted_stale_groups + output.evicted_overflow_groups
                metrics["rollout/fully_async/evict_rate"] = evicted / output.entered_groups
            output.reset_counters()
            if staleness_values:
                metrics["rollout/fully_async/avg_staleness"] = sum(staleness_values) / len(staleness_values)
                metrics["rollout/fully_async/max_staleness"] = max(staleness_values)
            if (stats := output.staleness_stats(buffer_stats_weight_version)) is not None:
                (
                    metrics["rollout/fully_async/buffer_avg_staleness"],
                    metrics["rollout/fully_async/buffer_max_staleness"],
                ) = stats

            if self._retained_slots is not None:
                ownership = self._ownership
                completed_slots = self._completed_slots
                if ownership is None or completed_slots is None:
                    raise RuntimeError("Fully async ownership is not initialized.")
                lease = _OwnedTrainBatchLease(
                    rollout_id=rollout_id,
                    ownership=ownership,
                    terminal_receipts=terminal_receipts,
                    retained_slots=self._retained_slots,
                    completed_slots=completed_slots,
                    completed_slot_available=self._completed_slot_available,
                    owned_capacity_released=self._owned_capacity_released,
                    on_settled=self._settle_train_batch_lease,
                    on_rollback_failed=self._retain_failed_train_batch_rollback,
                )
                self._open_train_batch_leases.add(lease)
                return LeasedRolloutFnTrainOutput(
                    samples=cast(list[list[Sample]], data),
                    metrics=metrics,
                    lease=lease,
                )
            return RolloutFnTrainOutput(samples=data, metrics=metrics)
        except BaseException as error:
            if not self._uses_owned_capacity:
                worker = self._worker
                retained_groups = list(accepted_legacy_groups)
                if claimed_legacy_group is not None:
                    retained_groups.append(claimed_legacy_group)
                if self._closing or (worker is not None and worker.done()):
                    self._legacy_close_pending_groups.extend(source_group for source_group, _ in retained_groups)
                else:
                    self._legacy_requeued_groups.extend(retained_groups)
            if terminal_receipts:
                try:
                    self._rollback_owned_terminals(
                        terminal_receipts,
                        completed_slots=len(terminal_receipts),
                    )
                except BaseException as settlement_error:
                    self._pending_terminal_rollbacks.extend(
                        (terminal_receipt, True) for terminal_receipt in terminal_receipts
                    )
                    raise error from settlement_error
            raise

    def _recycle_buffer_source(self, source: BufferSource) -> None:
        if isinstance(source, _OwnedCompletedGroup):
            self._rollback_owned_terminal(source.terminal_receipt, completed_slot_held=True)
            return
        self._recycle(source)

    def _recycle(self, prompt_group: list[Sample]) -> None:
        for sample in prompt_group:
            sample.reset_for_retry()
        self.data_source.add_samples([prompt_group])


async def _await_task_terminal(task: asyncio.Task[_T]) -> _T:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


def _terminal_observation_needs_retry(task: asyncio.Task[_OwnedTerminalResult]) -> bool:
    if not task.done():
        return False
    if task.cancelled():
        return True
    return isinstance(task.exception(), FullyAsyncTerminalPendingError)
