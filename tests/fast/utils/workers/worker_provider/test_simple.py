import asyncio

import pytest

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort


class FakeController:
    def health(self) -> int:
        return 1


def _provider() -> SimpleWorkerProvider:
    return SimpleWorkerProvider(
        addrs={
            "trainer-controller-0-0": {
                "primary": HostAndPort(host="controller.rl.svc", port=7000),
                "rpc": HostAndPort(host="controller.rl.svc", port=8000),
            }
        },
        cells={"trainer-controller-0": ["trainer-controller-0-0"]},
        spec_names={"trainer-controller-0": "trainer-controller"},
        worker_classes={"trainer-controller": f"{__name__}.FakeController"},
    )


class TestAddresses:
    def test_answers_from_the_address_book_it_was_handed(self):
        """A statically addressed component needs no cluster at all, only the address the chart rendered."""
        assert asyncio.run(_provider().get_addr("trainer-controller-0-0")).port == 7000

    def test_refuses_a_worker_nobody_declared(self):
        """Guessing an address would send calls to whatever happens to answer there."""
        with pytest.raises(AssertionError, match="not in the address book"):
            asyncio.run(_provider().get_addr("trainer-controller-9-9"))


class TestWatchCells:
    def test_reports_every_declared_cell_once_and_never_again(self):
        """Static workers do not come and go, so one report of the whole world is the entire watch."""
        seen: list[tuple[str, object]] = []

        async def scenario():
            async def reconcile(cell_id, info):
                seen.append((cell_id, info))

            stop = await _provider().watch_cells(reconcile, spec_names=["trainer-controller"])
            await stop()

        asyncio.run(scenario())

        assert [cell_id for cell_id, _ in seen] == ["trainer-controller-0"]
        assert seen[0][1].alive is True

    def test_hides_the_cells_of_another_spec(self):
        """One provider serves one group of workers, and reporting the rest would confuse its owner."""
        seen: list[str] = []

        async def scenario():
            async def reconcile(cell_id, info):
                seen.append(cell_id)

            await _provider().watch_cells(reconcile, spec_names=["inference-controller"])

        asyncio.run(scenario())

        assert seen == []


class TestHandles:
    def test_builds_an_rpc_handle_at_the_declared_rpc_port(self):
        """The caller drives the component over rpc, which is only reachable at the port it declared."""
        handle = _provider().get_handle("trainer-controller-0-0")

        assert isinstance(handle, RpcWorkerHandle)
        assert handle._transport._server_url == "http://controller.rl.svc:8000"

    def test_worker_infos_carry_the_handles_of_a_cell(self):
        """A consumer that owns a static cell drives it exactly like an observed one."""
        infos = _provider().get_worker_infos_of(cell_ids=["trainer-controller-0"])[0]

        assert [info.name for info in infos] == ["trainer-controller-0-0"]
        assert isinstance(infos[0].handle, RpcWorkerHandle)
