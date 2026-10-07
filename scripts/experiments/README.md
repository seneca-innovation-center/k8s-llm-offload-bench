# Sweep scripts

The scripts that produced the data in [`results/`](../../results/), exactly as run, adapted only to this repository's
layout (they `cd` into `src/engine`, call `../k8s/deploy.py`, and write under `src/engine/`). The mapping from script to
data file is in [`results/README.md`](../../results/README.md).

They assume a running k3d cluster with `src/engine` mounted at `/exp`, the `cyclic:v1` image imported, and the model
weights split into `src/engine/weights_<name>` (see [`src/k8s/README.md`](../../src/k8s/README.md)). `moe_chain.sh`
and `post_chain.sh` chained several sweeps unattended; they wait for log lines of earlier steps.
