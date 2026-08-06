from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from miles.rollout.data_source import DataSource
from miles.utils.types import Sample

if TYPE_CHECKING:
    from miles.rollout.inference_rollout.inference_rollout_common import GenerateState


@dataclass(frozen=True)
class RolloutFnConstructorInput:
    args: Namespace
    # TODO may refactor DataSource API
    data_source: DataSource


@dataclass(frozen=True)
class RolloutFnBaseInput:
    rollout_id: int

    @property
    def evaluation(self):
        raise NotImplementedError


# subclassing for different data in the future
@dataclass(frozen=True)
class RolloutFnTrainInput(RolloutFnBaseInput):
    @property
    def evaluation(self):
        return False


@dataclass(frozen=True)
class RolloutFnEvalInput(RolloutFnBaseInput):
    generate_state: GenerateState | None = None
    weight_version: str | None = None
    hf_dir: str | None = None

    @property
    def evaluation(self):
        return True


class TrainBatchRollbackReason(Enum):
    """Reason that a manager could not hand a leased batch to training."""

    HANDOFF_FAILED = auto()


class TrainAdmissionHold(ABC):
    """Own one claim that keeps training admission closed.

    Active holds block both new source reservations and owned train-batch
    lease issuance. Admission remains closed until every active hold is
    released or the rollout lifecycle begins closing.
    """

    def __init__(self) -> None:
        self._release_attempted = False

    async def wait_terminal(self) -> None:
        """Wait until every execution before this hold's frontier is terminal.

        This does not consume completed groups, settle train-batch leases, or
        request execution cancellation. Calls may be repeated while the hold
        remains unreleased, including after lifecycle close begins.

        Raises:
            RuntimeError: If release was already attempted.
            BaseException: A terminal execution or lifecycle failure.
        """
        if self._release_attempted:
            raise RuntimeError("Train admission hold already has a release attempt.")
        await self._wait_terminal()

    @abstractmethod
    async def _wait_terminal(self) -> None:
        """Implement terminal observation for this hold's admission frontier."""

    def release(self) -> None:
        """Release this hold's claim on training admission.

        A release attempt claims the handle even if its implementation raises.
        Source reservation and owned lease issuance reopen only after every
        active hold is released.

        Raises:
            RuntimeError: If release was already attempted.
        """
        if self._release_attempted:
            raise RuntimeError("Train admission hold already has a release attempt.")
        self._release_attempted = True
        self._release()

    @abstractmethod
    def _release(self) -> None:
        """Implement release of this exact admission claim."""


class RolloutFnLifecycle(ABC):
    """Expose optional ownership and resource lifecycle controls."""

    @abstractmethod
    async def prepare_checkpoint(self, rollout_id: int) -> None:
        """Prepare rollout-owned state for checkpoint publication.

        The caller must own an active train-admission hold so no owned batch
        lease can be issued while checkpoint publication is prepared.

        Args:
            rollout_id: Rollout identifier that the checkpoint will publish.

        Raises:
            RuntimeError: If no train-admission hold is active.
            RuntimeError: If train-batch ownership remains unsettled.
            BaseException: A rollout lifecycle failure.
        """

    @abstractmethod
    async def acquire_train_admission_hold(self) -> TrainAdmissionHold:
        """Close training admission and return its owned claim.

        The return linearizes after every later source reservation and owned
        train-batch lease issuance is blocked. Work admitted before that point
        continues until terminal.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close rollout-owned resources.

        Repeated calls have no effect after a successful close. A failed close
        may be retried when its reported ownership or cleanup blocker changes.
        Close permanently dominates outstanding admission holds.
        """


class TrainBatchLease(ABC):
    """Own a rollout batch until its train-data handoff settles.

    Args:
        rollout_id: Training rollout that requested the batch.

    A successful commit transfers ownership to downstream train data. A failed
    commit must recover ownership or retain it for lifecycle cleanup because
    the caller discards the failed handoff. Settlement may be attempted only
    once, including when its implementation raises.
    """

    def __init__(self, rollout_id: int) -> None:
        self._rollout_id = rollout_id
        self._settlement_attempted = False

    @property
    def rollout_id(self) -> int:
        """Return the rollout that acquired this batch."""
        return self._rollout_id

    def commit(self) -> None:
        """Transfer batch ownership to downstream train data.

        Raises:
            RuntimeError: If any settlement was already attempted.
        """
        self._claim_settlement()
        self._commit()

    @abstractmethod
    def _commit(self) -> None:
        """Implement the ownership transfer after settlement is claimed."""

    def rollback(self, reason: TrainBatchRollbackReason) -> None:
        """Return ownership after a train-data handoff fails.

        Args:
            reason: Why the manager could not complete the handoff.

        Raises:
            RuntimeError: If any settlement was already attempted.
        """
        self._claim_settlement()
        self._rollback(reason)

    @abstractmethod
    def _rollback(self, reason: TrainBatchRollbackReason) -> None:
        """Implement ownership recovery after settlement is claimed."""

    def _claim_settlement(self) -> None:
        if self._settlement_attempted:
            raise RuntimeError(f"Train batch lease for rollout {self.rollout_id} already has a settlement attempt.")
        self._settlement_attempted = True


# TODO make it frozen
@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] | None = None


@dataclass
class LeasedRolloutFnTrainOutput(RolloutFnTrainOutput):
    """Carry ordinary train output data with its required settlement lease.

    Args:
        samples: Generated samples grouped by source prompt.
        metrics: Optional rollout metrics.
        lease: Ownership to settle after the train-data handoff.
    """

    lease: TrainBatchLease = field(kw_only=True)


# TODO make it frozen
@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] | None = None


RolloutFnInput = RolloutFnTrainInput | RolloutFnEvalInput
RolloutFnOutput = RolloutFnTrainOutput | RolloutFnEvalOutput


@dataclass(frozen=True)
class GenerateFnInput:
    state: GenerateState
    sample: Sample
    sampling_params: dict[str, Any]
    evaluation: bool

    @property
    def args(self) -> Namespace:
        return self.state.args


@dataclass(frozen=True)
class GenerateFnOutput:
    # One generate may lead to multiple samples, such as multi-agent, tree-like exploration, or
    # multi-turn with removing thinking tokens.
    samples: Sample | list[Sample]


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    """Legacy rollout function call interface. Used when MILES_EXPERIMENTAL_ROLLOUT_REFACTOR is disabled."""
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    return output
