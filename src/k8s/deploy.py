"""Generate + apply k8s manifests for one config.

    python3 deploy.py G K MODE [KEY=VALUE ...]
    DRY_RUN=1 python3 deploy.py G K MODE [KEY=VALUE ...] > manifest.yaml     # print the manifest, touch no cluster

G = number of layer groups, K = number of stage pods, MODE = resident | cyclic.
Extra KEY=VALUE pairs become environment variables of all pods (WEIGHTS, RESIDENT, SLOTS, LOAD_MODE, BATCH, MAXLEN,
DRAFT, DRAFT_LAYERS, TREE, GAMMA, PROMPT_FILE, CHAT). MEM_LIMIT=<k8s quantity> sets a memory limit on the stage pods.
The pods run the code from a hostPath /exp on the node (see src/k8s/README.md); IMAGE overrides the image name.
"""
import os, subprocess, sys

G, K, MODE = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
EXTRA = dict(a.split("=", 1) for a in sys.argv[4:])
MEM_LIMIT = EXTRA.pop("MEM_LIMIT", None)
IMG = os.environ.get("IMAGE", "cyclic:v1")

env = {"G": str(G), "K": str(K), "MODE": MODE, "THREADS": "16", "OMP_WAIT_POLICY": "PASSIVE", **EXTRA}


def env_block(indent):
    pad = " " * indent
    lines = [f'{pad}- {{name: {k}, value: "{v}"}}' for k, v in env.items()]
    lines.append(f"{pad}- {{name: POD_NAME, valueFrom: {{fieldRef: {{fieldPath: metadata.name}}}}}}")
    return "\n" + "\n".join(lines)


RES = f"\n        resources: {{limits: {{memory: {MEM_LIMIT}}}}}" if MEM_LIMIT else ""
Y = f"""
apiVersion: v1
kind: Service
metadata: {{name: stage}}
spec:
  clusterIP: None
  publishNotReadyAddresses: true
  selector: {{app: stage}}
  ports: [{{port: 9000}}]
---
apiVersion: v1
kind: Service
metadata: {{name: coordinator}}
spec:
  clusterIP: None
  publishNotReadyAddresses: true
  selector: {{app: coordinator}}
  ports: [{{port: 9000}}]
---
apiVersion: apps/v1
kind: StatefulSet
metadata: {{name: stage}}
spec:
  serviceName: stage
  replicas: {K}
  podManagementPolicy: Parallel
  selector: {{matchLabels: {{app: stage}}}}
  template:
    metadata: {{labels: {{app: stage}}}}
    spec:
      terminationGracePeriodSeconds: 1
      containers:
      - name: stage
        image: {IMG}
        imagePullPolicy: Never
        command: [python, /exp/stage_server.py]
        workingDir: /exp
        readinessProbe: {{tcpSocket: {{port: 9000}}, periodSeconds: 1}}{RES}
        env:{env_block(8)}
        volumeMounts: [{{name: exp, mountPath: /exp}}]
      volumes: [{{name: exp, hostPath: {{path: /exp}}}}]
---
apiVersion: v1
kind: Pod
metadata: {{name: coordinator, labels: {{app: coordinator}}}}
spec:
  terminationGracePeriodSeconds: 1
  containers:
  - name: coordinator
    image: {IMG}
    imagePullPolicy: Never
    command: [python, /exp/coordinator.py]
    workingDir: /exp
    readinessProbe: {{httpGet: {{path: /, port: 8000}}, periodSeconds: 1, failureThreshold: 600}}
    env:{env_block(4)}
    volumeMounts: [{{name: exp, mountPath: /exp}}]
  volumes: [{{name: exp, hostPath: {{path: /exp}}}}]
"""
if os.environ.get("DRY_RUN") == "1":
    print(Y.strip())
    sys.exit(0)
sh = lambda c: subprocess.run(c, shell=True, check=False)
sh("kubectl delete statefulset/stage pod/coordinator --ignore-not-found --wait=true >/dev/null")
sh("kubectl wait --for=delete pod -l 'app in (stage,coordinator)' --timeout=120s >/dev/null 2>&1")
suffix = "".join(f"_{k}{v}".replace("/", "").replace("-", "to") for k, v in sorted(EXTRA.items()))
if MEM_LIMIT:
    suffix += f"_mem{MEM_LIMIT}"
os.makedirs("k8s", exist_ok=True)
open(f"k8s/G{G}_K{K}_{MODE}{suffix}.yaml", "w").write(Y)
subprocess.run(["kubectl", "apply", "-f", "-"], input=Y.encode(), check=True, stdout=subprocess.DEVNULL)
sh("sleep 2; kubectl wait --for=condition=Ready pod -l 'app in (stage,coordinator)' --timeout=900s >/dev/null")
