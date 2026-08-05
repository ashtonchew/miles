import asyncio

import pytest

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.k8s import K8sCellOperations
from miles.utils.workers.worker_provider.base import CellInfo


class FakeProvider:
    def __init__(self, infos: dict[str, CellInfo]) -> None:
        self._infos = infos
        self.watched_spec_names: list[list[str]] = []

    async def watch_cells(self, reconcile, *, spec_names):
        self.watched_spec_names.append(list(spec_names))
        return _stop_watching

    def cell_ids(self) -> list[str]:
        return sorted(self._infos)

    def cell_info(self, cell_id: str) -> CellInfo | None:
        return self._infos.get(cell_id)

    def pod_names(self, cell_id: str) -> list[str]:
        info = self._infos.get(cell_id)
        return list(info.worker_names) if info is not None else []


def _info(cell_id="trainer-actor-0", spec_name="trainer-actor", workers=("trainer-actor-0-0",)):
    return CellInfo(
        cell_id=cell_id,
        spec_name=spec_name,
        alive=True,
        worker_names=list(workers),
        workers_hash="h",
        meta={},
    )


def _operations(infos, deleted):
    async def delete_pods(names):
        deleted.append(list(names))

    return K8sCellOperations(provider=FakeProvider(infos), spec_names=["trainer-actor"], delete_pods=delete_pods)


class TestCellInfos:
    def test_reports_the_cells_of_the_specs_it_was_asked_about(self):
        """A trainer handler must not list the inference cells that share the namespace."""
        infos = {"trainer-actor-0": _info(), "engine-0": _info(cell_id="engine-0", spec_name="engine")}
        operations = _operations(infos, [])

        listed = asyncio.run(operations.cell_infos(spec_names=["trainer-actor"]))

        assert list(listed) == ["trainer-actor-0"]

    def test_reports_nothing_when_no_cell_exists_yet(self):
        """A run whose pods are still being scheduled has no cells, which is not an error."""
        assert asyncio.run(_operations({}, []).cell_infos(spec_names=["trainer-actor"])) == {}


class TestSuspend:
    def test_deletes_the_pods_of_the_cell(self):
        """Deleting them is the whole operation: the workload brings the group back by itself."""
        deleted = []
        operations = _operations({"trainer-actor-0": _info(workers=("p0", "p1"))}, deleted)

        asyncio.run(operations.suspend(cell_id="trainer-actor-0"))

        assert deleted == [["p0", "p1"]]

    def test_touches_no_other_cell(self):
        """Healing one dp group must leave the others training."""
        deleted = []
        infos = {
            "trainer-actor-0": _info(workers=("a",)),
            "trainer-actor-1": _info(cell_id="trainer-actor-1", workers=("b",)),
        }

        asyncio.run(_operations(infos, deleted).suspend(cell_id="trainer-actor-0"))

        assert deleted == [["a"]]

    def test_refuses_a_cell_with_no_pods(self):
        """There is nothing to delete, and silently succeeding would report a heal that never happened."""
        with pytest.raises(AssertionError, match="no pods"):
            asyncio.run(_operations({}, []).suspend(cell_id="trainer-actor-0"))


class TestResume:
    def test_does_nothing_because_the_workload_already_did(self):
        """Kubernetes recreates a deleted group on its own, so resume has nothing left to do."""
        deleted = []

        asyncio.run(_operations({"trainer-actor-0": _info()}, deleted).resume(cell_id="trainer-actor-0"))

        assert deleted == []


class TestInjectFault:
    def test_says_it_cannot_reach_into_a_worker_process(self):
        """Silently doing nothing would make a fault-injection test pass while injecting no fault."""
        with pytest.raises(NotImplementedError, match="rpc layer"):
            asyncio.run(
                _operations({"trainer-actor-0": _info()}, []).inject_fault(
                    cell_id="trainer-actor-0", mode=list(FailureMode)[0], sub_index=0
                )
            )


class TestColocateFallback:
    def test_takes_the_paired_engine_pods_down_with_the_cell(self):
        """A healed trainer cell may return on different nodes, and a scheduled engine cannot follow it."""
        deleted = []
        operations = K8sCellOperations(
            provider=FakeProvider({"trainer-actor-0": _info(workers=("t0",))}),
            spec_names=["trainer-actor"],
            delete_pods=lambda names: _record(deleted, names),
            colocated_with=lambda cell_id: ["engine-0-0"],
        )

        asyncio.run(operations.suspend(cell_id="trainer-actor-0"))

        assert deleted == [["t0", "engine-0-0"]]

    def test_deletes_only_the_cell_when_nothing_is_colocated(self):
        """A disaggregated run has no engines pinned to trainer nodes, so none should be disturbed."""
        deleted = []
        operations = K8sCellOperations(
            provider=FakeProvider({"trainer-actor-0": _info(workers=("t0",))}),
            spec_names=["trainer-actor"],
            delete_pods=lambda names: _record(deleted, names),
        )

        asyncio.run(operations.suspend(cell_id="trainer-actor-0"))

        assert deleted == [["t0"]]


async def _record(deleted, names):
    deleted.append(list(names))


async def _stop_watching() -> None:
    return None
