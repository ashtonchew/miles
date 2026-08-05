import shlex

from tests.fast.cluster_backends import both_backends, require_backend

import miles.utils.external_utils.command_utils as command_utils
from miles.utils.external_utils.command_utils import base_backend
from miles.utils.workers.types import ClusterBackend


def _config(cluster_backend: str, namespace: str) -> command_utils.ExecuteTrainConfig:
    return command_utils.ExecuteTrainConfig(
        cluster_backend=cluster_backend,
        namespace=namespace,
        run_id="260101-000000-000",
    )


class TestEveryBackendHonoursTheSameLauncherContract:
    @both_backends
    def test_the_config_installs_the_backend_it_names(self, cluster_backend, monkeypatch):
        """The backend is a run's property, so choosing it in the config has to be what actually takes effect."""
        namespace = require_backend(cluster_backend)
        monkeypatch.setattr(base_backend, "_active_backend", None)

        command_utils.install_cluster_backend(_config(cluster_backend, namespace))

        assert (
            base_backend.active_backend().__class__.__name__
            == {
                ClusterBackend.RAY.value: "RayCommandBackend",
                ClusterBackend.KUBERNETES.value: "KubernetesCommandBackend",
            }[cluster_backend]
        )

    @both_backends
    def test_a_cpu_command_runs_without_reaching_the_cluster(self, cluster_backend, monkeypatch):
        """Both backends run cpu work where the launcher already sits, so a script reads the same either way."""
        namespace = require_backend(cluster_backend)
        monkeypatch.setattr(base_backend, "_active_backend", None)
        command_utils.install_cluster_backend(_config(cluster_backend, namespace))

        recorded: list[str] = []
        monkeypatch.setattr(
            base_backend.active_backend(),
            "exec_command_cpu",
            lambda cmd, capture_output=False: recorded.append(cmd),
        )
        base_backend.exec_command_cpu("echo hello")

        assert recorded == ["echo hello"]

    @both_backends
    def test_the_worker_flag_and_the_config_must_agree(self, cluster_backend):
        """The config drives the launcher and the flag drives the workers; disagreeing installs one and drives the other."""
        require_backend(cluster_backend)
        other = next(backend.value for backend in ClusterBackend if backend.value != cluster_backend)

        try:
            command_utils._resolve_backend(cluster_backend, train_args=f"--cluster-backend {other}")
        except AssertionError as error:
            assert shlex.quote(other).strip("'") in str(error)
        else:
            raise AssertionError("a config and a train flag naming different backends must be refused")
