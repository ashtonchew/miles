---
title: "Running e2e tests on Kubernetes"
description: "Step by step: install a workbench, launch a run, read the verdict, clean up."
---

You need a kubectl context and an `infra.yaml` for the cluster; see [Kubernetes](../../charts/miles-run/README.md)
for its shape and for what a platform owes a run. Everything testable without a cluster is tested
without one, so this runbook covers only what cannot be: real scheduling, real gpus, and CUDA IPC
between two pods.

## 1. Install a workbench

```bash
export MILES_NS="miles-e2e-$USER-$(date +%m%d%H%M)"
python charts/miles-workbench/cli.py install -n "$MILES_NS" -r workbench -f infra.yaml
```

`install` does four things in order, and a failure in any of them is a real problem with the cluster or
the namespace, so fix what it reports:

- creates the namespace if it is missing,
- checks that your identity may install the chart,
- installs it,
- waits for the workbench pod to be Ready.

`charts/miles-workbench/README.md` lists the flags of every subcommand.

## 2. Point the run at the code you want to test

The image carries a copy of miles, so a test of your branch needs your branch mounted over it:

```yaml
infra:
  paths:
    repos: {miles: alice/miles}
```

- Put the checkout under the shared storage root, and name that sub-path in `infra.paths.repos`.
- Every pod of the run mounts it over the image's copy, so the checkout's HEAD is what the run executes.
- Do not copy anything into a pod: the run's pods are separate containers, and the mount is declarative.

## 3. Launch

```bash
export MILES_RUN_ID="e2e-$(date +%m%d%H%M)"
python charts/miles-workbench/cli.py exec -n "$MILES_NS" -r workbench -- bash -lc \
  "cd /root/miles && python scripts/run_qwen3_4b.py train \
     --cluster-backend kubernetes \
     --namespace $MILES_NS \
     --run-id $MILES_RUN_ID \
     --infra-values /cluster-storage/infra.yaml"
```

`exec` shells into the release's pod; with no command it gives you `bash`. Every field of the launch
script's `ExecuteTrainConfig` is an option of each of its subcommands, and each also reads
`MILES_SCRIPT_<FIELD_NAME_UPPER>` from the environment. Required here:

- `--namespace`: the namespace the release is installed into.
- `--run-id`: names both the release and the run directory, so it has to be a valid kubernetes object
  name. Relaunching the same run id upgrades that run in place, which is how a run grows or shrinks, so
  do not generate it per launch.
- `--infra-values` (repeatable): the per-cluster file the pods are rendered from.
- `--cluster-backend kubernetes`: a config option. The launcher refuses a config and a train argument
  that disagree, and when they agree it appends the flag to the argv it renders into the pods, because
  the orchestrator inside the pod dispatches on it too.

Optional, Kubernetes-only:

- `--shared-root`: assert the storage root derived from the infra values instead of trusting it.
- `--stage-to-local source:destination` (repeatable): copy inputs onto the node-local disk once per node.
- `--node-local-root`: that disk's mount path.
- `--ci-run`: first uninstall leftover CI releases in the namespace.
- `--force`: apply a relaunch that changes more than a fleet's replica count, accepting that the changed
  pods restart.

The launcher prints a one-line pod summary until the run settles, then follows the orchestrator's log.
`ctrl+c` stops watching, not the run. While it starts, read the summary:

- `pending`: the scheduler has not placed a pod. Check quotas and taints.
- `gated`: expected only for a colocate run, where engine pods wait for their trainers.
- `starting`: Running but not ready, usually a model loading.
- `failed` or `restarted` above zero: look at that pod now.

## 4. Read the verdict

```bash
export MILES_RUN_DIR="/cluster-storage/miles_data/miles-runs/$MILES_RUN_ID"
python charts/miles-workbench/cli.py exec -n "$MILES_NS" -r workbench -- \
  cat "$MILES_RUN_DIR/state/orchestrator.exit"
```

- The run's outcome is that exit file.
- The launcher reports it too, when it stops following.

## 5. When it fails

```bash
python charts/miles-workbench/cli.py collect-diagnosis -n "$MILES_NS" -r workbench \
  --output-dir ~/artifacts/miles --run-dir "$MILES_RUN_DIR"
```

This collects pod logs, describes and events into one directory and prints its path, plus the verdict
when `--run-dir` is visible from where cli.py runs; archive that directory before step 6 deletes the
evidence, including the pods of a failed adhoc Job, which is left in place on purpose.

## 6. Clean up

```bash
python charts/miles-workbench/cli.py uninstall -n "$MILES_NS" -r workbench
kubectl delete namespace "$MILES_NS"
```

- Uninstall removes the release only.
- Deleting the namespace is what frees the gpus, so confirm it is gone rather than stuck Terminating.
- Delete only the namespace `$MILES_NS` names: its suffix is unique per invocation, so never reuse a
  namespace someone else exported.
