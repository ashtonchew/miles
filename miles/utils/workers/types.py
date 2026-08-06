from enum import Enum


class ClusterBackend(Enum):
    RAY = "ray"
    KUBERNETES = "kubernetes"


DEFAULT_GPUS_PER_NODE = 8
