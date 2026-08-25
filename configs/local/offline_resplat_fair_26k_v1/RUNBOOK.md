# Offline U8/U12/U26 versus official ReSplat R4 runbook

This protocol compares one ordinary pristine-Unblur 26K Adam trajectory at
8K, 12K, and 26K with an **independent** official ReSplat terminal renderer.
R4 is not fused with an Unblur map. All 113 formal RGB queries are context
mapped-training views; this protocol makes no novel-view claim.

All managed GPU work, including the BSD validation pipeline, is serialized by
the shared `/srv/szha0669/unblur-slam/locks/physical_gpu1.lock` and exposes only physical GPU 1
(`GPU-3501b285-78cd-1494-87f1-ccac2136866e`) as logical `cuda:0`. Never bypass
the pinned wrapper or the plan executor.

## 1. CPU fail-closed preflight

From `/home/szha0669/Unblur-SLAM-framecrafter-resplat`:

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/srv/szha0669/unblur-slam/tmp \
python scripts/preflight_offline_resplat_fair_26k.py \
  --contract configs/local/offline_resplat_fair_26k_v1/preregistered_contract.json \
  --report /srv/szha0669/unblur-slam/audits/offline_resplat_fair_26k_v1/preflight.json
```

Do not continue unless `passed=true`, `gpu_started=false`, and the output root
is absent.

## 2. Three serial pristine-Unblur trajectories on pinned GPU1

Run from `/srv/szha0669/unblur-slam/worktrees/offline_resplat_fair_26k_v1`.
Execute the following command once per scene, strictly in this order, replacing
both occurrences of `SCENE` with `freiburg1_desk`, then `freiburg2_xyz`, then
`freiburg3_office`:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/srv/szha0669/unblur-slam/.tmp/offline_resplat_fair_26k_v1/pycache \
TMPDIR=/srv/szha0669/unblur-slam/.tmp/offline_resplat_fair_26k_v1 \
UNBLUR_SKIP_NR_IQA=1 \
/srv/szha0669/unblur-slam/env/bin/python \
  /home/szha0669/Unblur-SLAM-framecrafter-resplat/scripts/execute_pinned_gpu_command.py \
  --lock-file /srv/szha0669/unblur-slam/locks/physical_gpu1.lock \
  --audit-report /srv/szha0669/unblur-slam/offline_resplat_fair_26k_v1/gpu_audits/unblur_SCENE.json \
  --expected-physical-index 1 \
  --expected-cuda-visible-devices 1 \
  --expected-gpu-name 'NVIDIA RTX A6000' \
  --expected-gpu-uuid GPU-3501b285-78cd-1494-87f1-ccac2136866e \
  --expected-gpu-serial 1711224002341 \
  -- /srv/szha0669/unblur-slam/env/bin/python run.py \
  configs/local/offline_resplat_fair_26k_v1/SCENE.yaml
```

The wrapper refuses a busy GPU, holds the exclusive lock for the whole child,
sets `CUDA_VISIBLE_DEVICES=1`, and writes a non-overwriting audit containing its
own hash plus the exact child argv, argv hash, and worktree cwd. The final
reporter rejects an audit not bound to this exact command. The hook also
rechecks the UUID/serial, exact keyframe/submap count, and records U8/U12/U26
from one trajectory.

## 3. Materialize the three frozen R4 plans

From `/home/szha0669/Unblur-SLAM-framecrafter-resplat`, run this command once
for each row in the table:

| SCENE | SOURCE_COUNT | EVAL_COUNT | SUBMAP_COUNT |
|---|---:|---:|---:|
| freiburg1_desk | 89 | 14 | 12 |
| freiburg2_xyz | 42 | 42 | 6 |
| freiburg3_office | 153 | 57 | 20 |

```bash
/srv/szha0669/unblur-slam/env/bin/python \
  scripts/materialize_offline_resplat_frozen_bundle.py \
  --bundle /srv/szha0669/unblur-slam/offline_resplat_fair_26k_v1/unblur_pristine/SCENE/offline_fair_26k/frozen_inputs/bundle.json \
  --output-dir /srv/szha0669/unblur-slam/offline_resplat_fair_26k_v1/resplat_tasks/SCENE \
  --scene-dir /srv/szha0669/unblur-slam/offline_resplat_fair_26k_v1/resplat_scenes/SCENE \
  --scene-name SCENE \
  --expected-source-count SOURCE_COUNT \
  --expected-eval-count EVAL_COUNT \
  --expected-submap-count SUBMAP_COUNT \
  --workspace /home/szha0669/Unblur-SLAM-framecrafter-resplat \
  --resplat-repo /srv/szha0669/unblur-slam/external/resplat \
  --resplat-python /srv/szha0669/unblur-slam/envs/resplat-official-py312-torch270-cu128/bin/python \
  --unblur-python /srv/szha0669/unblur-slam/env/bin/python \
  --checkpoint /srv/szha0669/unblur-slam/pretrained/resplat-official/resplat-small-dl3dv-256x448-view8-548993fe.pth \
  --checkpoint-sha256 548993fede0d9536d2d914cbe51e0ebea0ad6f88c898c909e02127d59bb2be9a \
  --paired-runner /home/szha0669/Unblur-SLAM-framecrafter-resplat/scripts/run_paired_official_resplat_smoke.py \
  --paired-runner-sha256 f9b40b8324fa044251056bbf3dfd40f17b77146f2a26a553d038510326a55c6e \
  --scene-exporter /home/szha0669/Unblur-SLAM-framecrafter-resplat/scripts/export_tum_official_resplat_scene.py \
  --scene-exporter-sha256 0cbc437009d25372911d223811e6aa5add73af9c3232b0dd0dd6e19f806d676a \
  --resplat-commit cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1 \
  --gpu-wrapper /home/szha0669/Unblur-SLAM-framecrafter-resplat/scripts/execute_pinned_gpu_command.py \
  --lock-file /srv/szha0669/unblur-slam/locks/physical_gpu1.lock \
  --expected-physical-index 1 \
  --expected-cuda-visible-devices 1 \
  --expected-gpu-name 'NVIDIA RTX A6000' \
  --expected-gpu-uuid GPU-3501b285-78cd-1494-87f1-ccac2136866e \
  --expected-gpu-serial 1711224002341
```

The runner SHA above is frozen in the preregistered contract and checked by
preflight. The materializer rejects any count drift, wrong GPU identity,
wrong quantizer, mutable checkpoint/runner, non-`/srv` output, or overwrite.

## 4. Export, run 38 fresh R4 processes, and score common RGB

For each scene plan, execute the exact argv arrays in this order:

1. `export_command` (CPU);
2. `sequential_execution_command` (12 + 6 + 20 fresh serial GPU processes);
3. `common_rgb_evaluation_command` (pinned GPU wrapper).

Use this argv-safe dispatcher from the repository root, replacing `SCENE` and
`COMMAND_KEY`; it performs no shell re-parsing:

```bash
/srv/szha0669/unblur-slam/env/bin/python -c \
  'import json, pathlib, subprocess, sys; p=json.loads(pathlib.Path(sys.argv[1]).read_text()); subprocess.run(p[sys.argv[2]], check=True)' \
  /srv/szha0669/unblur-slam/offline_resplat_fair_26k_v1/resplat_tasks/SCENE/plan.json \
  COMMAND_KEY
```

For each scene, use `export_command`, then `sequential_execution_command`, then
`common_rgb_evaluation_command`. Finish all three keys for one scene before
moving to the next. Existing output or audit paths cause a hard failure.

The primary R4 timer conservatively includes repository/checkpoint verification,
model load, input preprocessing, init0, four updates, and query rendering. It
stops before metric-reference loading, metric computation, and output PNG/PLY
artifact I/O. The executor also records complete fresh-process wall time,
including startup, metrics, and all I/O, as a secondary diagnostic only. The
common-RGB wrapper audit is likewise bound to the exact evaluator argv and
repository cwd stored in the plan.

## 5. Atomic final report and claim gate

After all three common-RGB reports exist, run on CPU:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/srv/szha0669/unblur-slam/env/bin/python \
  scripts/report_offline_resplat_fair_26k.py \
  --contract configs/local/offline_resplat_fair_26k_v1/preregistered_contract.json \
  --output-dir /srv/szha0669/unblur-slam/offline_resplat_fair_26k_v1/final_report
```

Only `final_report/report.json` may authorize a speedup statement. It requires
all three per-scene gates, the pooled 113-view gate, R4 primary time below U26,
exactly 38 fresh serial subprocesses, matching hashes/commit/checkpoint/refine4,
and the same physical A6000. Depth-L1 remains unavailable because R4 has no
audited raw metric-depth artifact.
