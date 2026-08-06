from __future__ import annotations

import os
from pathlib import Path

from miles.utils.workers.worker_provider.k8s_labels import CellLabelKeys

NAMESPACE_ENV_VAR = "MILES_K8S_NAMESPACE"
RELEASE_ENV_VAR = "MILES_K8S_RELEASE"
NAMESPACE_FILE = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")

LABEL_KEY_ENV_VARS = {
    "fleet": "MILES_K8S_FLEET_LABEL",
    "cell_index": "MILES_K8S_CELL_INDEX_LABEL",
    "worker_index": "MILES_K8S_WORKER_INDEX_LABEL",
    "spec_name": "MILES_K8S_SPEC_NAME_LABEL",
    "cell_size": "MILES_K8S_CELL_SIZE_LABEL",
    "meta_annotation_prefix": "MILES_K8S_META_ANNOTATION_PREFIX",
}


def current_namespace() -> str:
    if namespace := os.environ.get(NAMESPACE_ENV_VAR, ""):
        return namespace
    assert NAMESPACE_FILE.exists(), (
        f"the driver runs outside a pod, so it cannot tell which namespace holds its workers: "
        f"set {NAMESPACE_ENV_VAR}"
    )
    namespace = NAMESPACE_FILE.read_text().strip()
    assert namespace, f"{NAMESPACE_FILE} is empty, so no namespace can be observed"
    return namespace


def current_release() -> str:
    release = os.environ.get(RELEASE_ENV_VAR, "")
    assert release, (
        f"the orchestrator cannot tell which release created its workers, so it cannot select their pods: "
        f"set {RELEASE_ENV_VAR}"
    )
    return release


def current_label_keys() -> CellLabelKeys:
    overrides = {
        field: value for field, env_var in LABEL_KEY_ENV_VARS.items() if (value := os.environ.get(env_var, ""))
    }
    return CellLabelKeys(**overrides)
