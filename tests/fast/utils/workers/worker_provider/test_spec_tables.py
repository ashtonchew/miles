from __future__ import annotations

import asyncio

import pytest
from tests.fast.utils.workers.worker_provider.run_specs import (
    RELEASE,
    make_engine_spec,
    make_router_spec,
    make_trainer_spec,
)

from miles.utils.workers.worker_provider import spec_tables
from miles.utils.workers.worker_spec import PortInfo


class TestSpecDerivedValues:
    def test_only_gpu_bearing_specs_are_watched_as_cells(self) -> None:
        """A router is one pod behind a service, so it is addressed rather than healed."""
        assert spec_tables.fleet_spec_names(specs=[make_router_spec(), make_engine_spec()]) == ["engine"]

    def test_the_ports_of_an_observed_worker_come_from_its_spec(self) -> None:
        """A pod has a network namespace of its own, so every rank publishes the spec's ports."""
        assert spec_tables.fleet_worker_ports(specs=[make_router_spec(), make_engine_spec()]) == {
            "engine": {"primary": 8000, "nccl": 10000}
        }

    def test_the_specs_that_compute_meta_are_the_fleet_specs_that_declare_it(self) -> None:
        """A cell's meta is evaluated per observation, so only the fleets that have one may be listed."""
        specs = [make_router_spec(), make_engine_spec(), make_trainer_spec(num_workers_per_cell=8)]

        assert list(spec_tables.fleet_spec_metas(specs=specs)) == ["trainer-actor"]

    def test_the_address_book_covers_every_static_worker(self) -> None:
        """A spec with several cells still has one address per cell, and all of them are needed."""
        addrs = spec_tables.static_worker_addrs(specs=[make_router_spec(), make_engine_spec()], release=RELEASE)

        assert list(addrs) == ["inference-router-0-0-0"]


class TestStaticProvider:
    def test_serves_the_statically_addressed_workers_of_a_run(self) -> None:
        """A router is one pod behind a service, so it is addressed rather than observed."""
        provider = spec_tables.static_worker_provider(specs=[make_router_spec(), make_engine_spec()], release=RELEASE)

        assert provider.cell_ids() == ["inference-router-0-0"]
        assert asyncio.run(provider.get_addr("inference-router-0-0-0")).port == 8000


class TestRanksPerPod:
    def test_an_engine_pod_runs_one_command_and_therefore_holds_one_rank(self) -> None:
        """An engine pod is a single server process spanning its gpus, whatever its spec counts as a worker."""
        assert spec_tables.fleet_ranks_per_pod(specs=[make_engine_spec()], num_gpus_per_node=8) == {"engine": 1}

    def test_ignores_a_spec_that_is_addressed_rather_than_observed(self) -> None:
        """A router has no cell and therefore no pod to fan out."""
        assert spec_tables.fleet_ranks_per_pod(specs=[make_router_spec()], num_gpus_per_node=8) == {}

    def test_refuses_a_spec_whose_rank_ports_would_reach_into_another_port(self) -> None:
        """Rank n binds the rpc port plus n, so a neighbour close above it would be taken from under it."""
        spec = make_trainer_spec(
            num_workers_per_cell=8,
            port_infos=[
                PortInfo(name="rpc", static_port=8000, allow_dynamic=True),
                PortInfo(name="dist_init", static_port=8004, num_consecutive=30),
            ],
        )

        with pytest.raises(AssertionError, match="reaches into"):
            spec_tables.fleet_ranks_per_pod(specs=[spec], num_gpus_per_node=8)
