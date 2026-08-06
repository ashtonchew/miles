from dataclasses import dataclass, field
from typing import Any

from miles.utils.workers.worker_provider import k8s_labels


@dataclass
class FakeMeta:
    name: str
    uid: str = "uid-1"
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)


@dataclass
class FakeContainerStatus:
    restart_count: int = 0


@dataclass
class FakeStatus:
    phase: str = "Running"
    pod_ip: str | None = "10.0.0.1"
    conditions: list[Any] = field(default_factory=list)
    container_statuses: list[FakeContainerStatus] = field(default_factory=list)


@dataclass
class FakeCondition:
    type: str
    status: str
    reason: str | None = None


@dataclass
class FakePodSpec:
    node_name: str | None = "gpu-1"
    subdomain: str | None = None


@dataclass
class FakePod:
    metadata: FakeMeta
    status: FakeStatus = field(default_factory=FakeStatus)
    spec: FakePodSpec = field(default_factory=FakePodSpec)


def make_pod(name="engine-0-0", cell_index="0", worker_index="0", fleet="engine", ready=True, **kwargs):
    labels = {
        k8s_labels.LWS_NAME_LABEL: fleet,
        k8s_labels.LWS_GROUP_INDEX_LABEL: cell_index,
        k8s_labels.LWS_WORKER_INDEX_LABEL: worker_index,
    }
    labels.update(kwargs.pop("labels", {}))
    status = FakeStatus(
        conditions=[FakeCondition(type="Ready", status="True" if ready else "False")],
        container_statuses=[FakeContainerStatus(restart_count=kwargs.pop("restarts", 0))],
        pod_ip=kwargs.pop("pod_ip", "10.0.0.1"),
    )
    return FakePod(
        metadata=FakeMeta(
            name=name, uid=kwargs.pop("uid", f"uid-{name}"), labels=labels, annotations=kwargs.pop("annotations", {})
        ),
        status=status,
        spec=FakePodSpec(node_name=kwargs.pop("node_name", "gpu-1"), subdomain=kwargs.pop("subdomain", None)),
    )


class TestObservePod:
    def test_reads_the_cell_a_pod_belongs_to(self):
        """A cell is a fleet and a group index, which is what its consumers address."""
        observed = k8s_labels.observe_pod(make_pod(fleet="inference-engine-0-0", cell_index="2"))

        assert observed.cell_id == "inference-engine-0-0-2"

    def test_ignores_a_pod_that_carries_no_cell_labels(self):
        """A namespace holds other pods, and treating one as a worker would invent a cell."""
        assert k8s_labels.observe_pod(FakePod(metadata=FakeMeta(name="prometheus-0"))) is None

    def test_falls_back_to_the_workload_name_as_the_spec(self):
        """The bundled charts name a workload after its spec, so no extra label is needed."""
        assert k8s_labels.observe_pod(make_pod(fleet="trainer-actor")).spec_name == "trainer-actor"

    def test_lets_a_platform_name_the_spec_apart_from_the_workload(self):
        """A platform may group several specs into one workload, or name its workloads its own way."""
        pod = make_pod(labels={k8s_labels.SPEC_NAME_LABEL: "trainer-actor"}, fleet="team-a-training")

        assert k8s_labels.observe_pod(pod).spec_name == "trainer-actor"

    def test_reports_a_pod_that_is_not_ready_yet(self):
        """A cell whose ranks are still loading must not be given work."""
        assert k8s_labels.observe_pod(make_pod(ready=False)).ready is False

    def test_reads_the_node_a_pod_landed_on(self):
        """Colocate needs to know that a trainer and an engine really share a machine."""
        assert k8s_labels.observe_pod(make_pod()).node_name == "gpu-1"

    def test_reads_the_keys_a_platform_configured(self):
        """A platform that already labels its pods should not have to relabel them for miles."""
        keys = k8s_labels.CellLabelKeys(fleet="acme.io/group", cell_index="acme.io/index")
        pod = FakePod(metadata=FakeMeta(name="p", labels={"acme.io/group": "engine", "acme.io/index": "3"}))

        assert k8s_labels.observe_pod(pod, keys).cell_id == "engine-3"

    def test_reads_how_many_workers_the_cell_should_have(self):
        """A group still being created has ready pods but not all of them, and must not be given work."""
        pod = make_pod(labels={k8s_labels.LWS_SIZE_LABEL: "4"})

        assert k8s_labels.observe_pod(pod).cell_size == 4

    def test_reports_no_size_when_the_platform_publishes_none(self):
        """A platform that does not say cannot be second-guessed, so the cell is judged on readiness alone."""
        assert k8s_labels.observe_pod(make_pod()).cell_size == 0


class TestCellMembersHash:
    def test_is_stable_for_the_same_membership(self):
        """A consumer compares this across polls, so noise would look like a healed cell every time."""
        workers = [
            k8s_labels.observe_pod(make_pod(name=f"engine-0-{index}", worker_index=str(index))) for index in range(2)
        ]

        assert k8s_labels.cell_members_hash(workers) == k8s_labels.cell_members_hash(list(reversed(workers)))

    def test_changes_when_a_pod_is_replaced(self):
        """A new pod lost whatever the old one held in memory, and its consumers must resynchronise."""
        before = [k8s_labels.observe_pod(make_pod(uid="uid-a"))]
        after = [k8s_labels.observe_pod(make_pod(uid="uid-b"))]

        assert k8s_labels.cell_members_hash(before) != k8s_labels.cell_members_hash(after)

    def test_changes_when_a_pod_restarts_in_place(self):
        """The uid survives a restart but the process does not, so the hash has to notice."""
        before = [k8s_labels.observe_pod(make_pod(restarts=0))]
        after = [k8s_labels.observe_pod(make_pod(restarts=1))]

        assert k8s_labels.cell_members_hash(before) != k8s_labels.cell_members_hash(after)


class TestReadMeta:
    def test_reads_the_domain_facts_a_platform_attached(self):
        """An engine's model id reaches miles through the pod, not through the launcher's memory."""
        pod = make_pod(annotations={f"{k8s_labels.META_ANNOTATION_PREFIX}model_id": "glm", "other": "ignored"})

        assert k8s_labels.read_meta(pod) == {"model_id": "glm"}

    def test_reads_nothing_from_a_pod_without_annotations(self):
        """Most pods carry none, and a missing annotation block is not an error."""
        assert k8s_labels.read_meta(make_pod()) == {}
