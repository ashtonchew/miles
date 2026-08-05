import json
from typing import Any

from tests.fast.charts.utils import named_object, objects_of_kind, render_run, render_run_error, requires_helm

ENGINES = [
    {
        "name": "engine",
        "replicas": 2,
        "size": 4,
        "command": ["python"],
        "resources": {"limits": {"nvidia.com/gpu": 8}},
    }
]
TRAINERS = [{"name": "trainer", "replicas": 3, "size": 2, "command": ["python"]}]

FLEETS = (
    "--set-json",
    f"run.inferenceEngines={json.dumps(ENGINES)}",
    "--set-json",
    f"run.trainers={json.dumps(TRAINERS)}",
)

PAIRING = "myrun-miles-run-colocate-pairing"
ORCHESTRATOR_ROLE = "myrun-miles-run-orchestrator"

ENABLE = (
    *FLEETS,
    "--set",
    "run.colocate.enabled=true",
    "--set",
    "run.colocate.engineFleet=engine",
    "--set",
    "run.colocate.trainerFleet=trainer",
)

GATE = {"name": "miles.radixark.io/colocate-pairing"}


def fleet_pod(objects: list[dict[str, Any]], name: str) -> dict[str, Any]:
    fleet = named_object(objects, "LeaderWorkerSet", name)
    return fleet["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]


def colocated_engine_pod(*args: str) -> dict[str, Any]:
    return fleet_pod(render_run(*ENABLE, *args), "myrun-miles-run-engine")


@requires_helm
class TestColocatedEngineFleet:
    def test_is_held_back_from_the_scheduler_by_the_chart_itself(self):
        """Nothing else can keep a pod unscheduled until another pod's node is known."""
        assert colocated_engine_pod()["schedulingGates"] == [GATE]

    def test_shares_the_host_ipc_namespace(self):
        """A CUDA IPC handle's reference counter lives in shared memory, so both pods need the same one."""
        assert colocated_engine_pod()["hostIPC"] is True

    def test_sees_every_gpu_on_the_node_it_lands_on(self):
        """It requests no gpu of its own, so only the device plugin bypass makes the trainer's gpus visible."""
        assert {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"} in colocated_engine_pod()["containers"][0]["env"]

    def test_requests_no_gpus_of_its_own(self):
        """The trainer requests the whole node, and two claims on one gpu would never both schedule."""
        assert colocated_engine_pod()["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 0

    def test_carries_no_affinity_at_all_but_keeps_the_node_selector(self):
        """Any affinity would contradict the node the controller picks; the selector it only adds to."""
        pod = colocated_engine_pod(
            "--set-json", 'infra.scheduling={"nodeSelector":{"pool":"gpu"},"affinity":{"nodeAffinity":{}}}'
        )

        assert "affinity" not in pod
        assert pod["nodeSelector"] == {"pool": "gpu"}


@requires_helm
class TestColocatedTrainerFleet:
    def test_shares_the_host_ipc_namespace_too(self):
        """The engine's CUDA IPC handles are only usable from a trainer in the same IPC namespace."""
        assert fleet_pod(render_run(*ENABLE), "myrun-miles-run-trainer")["hostIPC"] is True

    def test_is_scheduled_normally_and_gets_none_of_the_engine_treatment(self):
        """It is the pod that claims the node, so gating it would leave nothing for the engine to pair with."""
        pod = fleet_pod(render_run(*ENABLE), "myrun-miles-run-trainer")

        assert "schedulingGates" not in pod
        assert "env" not in pod["containers"][0]


@requires_helm
class TestDisaggregatedRun:
    def test_leaves_the_engine_fleet_ungated_and_holding_its_own_gpus(self):
        """The same fleet values must render an ordinary engine when the run is not colocated."""
        pod = fleet_pod(render_run(*FLEETS), "myrun-miles-run-engine")

        assert "schedulingGates" not in pod
        assert "hostIPC" not in pod
        assert "env" not in pod["containers"][0]
        assert pod["containers"][0]["resources"] == {"limits": {"nvidia.com/gpu": 8}}

    def test_installs_no_pairing_controller(self):
        """A run whose engines have their own nodes must not gain a controller with pod write rights."""
        objects = render_run(*FLEETS)

        assert [obj["metadata"]["name"] for obj in objects_of_kind(objects, "Role")] == [ORCHESTRATOR_ROLE]
        assert [obj["metadata"]["name"] for obj in objects_of_kind(objects, "RoleBinding")] == [ORCHESTRATOR_ROLE]
        assert objects_of_kind(objects, "Deployment") == []
        assert [obj["metadata"]["name"] for obj in objects_of_kind(objects, "ServiceAccount")] == [ORCHESTRATOR_ROLE]


@requires_helm
class TestPairingController:
    def test_holds_only_namespaced_rights_over_pods(self):
        """Releasing a gate is an ordinary pod update, so nothing cluster-scoped is needed."""
        role = named_object(render_run(*ENABLE), "Role", PAIRING)

        assert role["rules"] == [
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch", "patch", "update"]}
        ]

    def test_never_asks_for_the_binding_subresource(self):
        """That belongs to the scheduler, and asking for it would make this a scheduler replacement."""
        rules = named_object(render_run(*ENABLE), "Role", PAIRING)["rules"]

        assert not any("binding" in resource for rule in rules for resource in rule["resources"])

    def test_runs_as_a_single_replica(self):
        """If it dies, new engine pods stay Pending, which is safe and visible; two would race each other."""
        assert named_object(render_run(*ENABLE), "Deployment", PAIRING)["spec"]["replicas"] == 1

    def test_is_told_the_object_names_and_cell_counts_the_chart_derived(self):
        """Values name the fleet entries, so only the chart knows the object names and their group shapes."""
        deployment = named_object(render_run(*ENABLE), "Deployment", PAIRING)
        command = deployment["spec"]["template"]["spec"]["containers"][0]["command"]
        arguments = dict(zip(command[3::2], command[4::2], strict=True))

        assert arguments["--engine-fleet"] == "myrun-miles-run-engine"
        assert arguments["--trainer-fleet"] == "myrun-miles-run-trainer"
        assert arguments["--engine-cells"] == "2"
        assert arguments["--trainer-cells"] == "3"
        assert arguments["--pods-per-engine-cell"] == "4"
        assert arguments["--pods-per-trainer-cell"] == "2"

    def test_carries_the_cluster_environment(self):
        """It talks to the api server, so a cluster that needs a proxy needs it here as well."""
        objects = render_run(*ENABLE, "--set", "infra.env.HTTP_PROXY=http://proxy:7890")
        deployment = named_object(objects, "Deployment", PAIRING)

        assert deployment["spec"]["template"]["spec"]["containers"][0]["env"] == [
            {"name": "HTTP_PROXY", "value": "http://proxy:7890"}
        ]


@requires_helm
class TestUnknownFleetNames:
    def test_an_engine_fleet_naming_no_entry_fails_the_render(self):
        """A typo would otherwise install a controller pairing a fleet that does not exist."""
        error = render_run_error(
            *FLEETS,
            "--set",
            "run.colocate.enabled=true",
            "--set",
            "run.colocate.engineFleet=nope",
            "--set",
            "run.colocate.trainerFleet=trainer",
        )

        assert "run.colocate.engineFleet" in error

    def test_a_trainer_fleet_naming_no_entry_fails_the_render(self):
        """Same for the other side of the pairing, which the controller equally cannot find."""
        error = render_run_error(
            *FLEETS,
            "--set",
            "run.colocate.enabled=true",
            "--set",
            "run.colocate.engineFleet=engine",
            "--set",
            "run.colocate.trainerFleet=nope",
        )

        assert "run.colocate.trainerFleet" in error
