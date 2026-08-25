# TURTLE × Unblur-SLAM cloud reproduction

This package reproduces the preregistered three-stage, from-scratch TURTLE
training protocol without using validation or test pixels for training.

## GPU topology

The scientific contract fixes the global batch to six.  Therefore:

| Cloud node | Training GPUs | Local batch | Global batch | Spare GPUs |
|---|---:|---:|---:|---:|
| 4 GPUs | 3 | 2 | 6 | 1 |
| 8 GPUs | 6 | 1 | 6 | 2 |

Using all four or all eight GPUs would change the global batch, gradient
trajectory, and effective optimization protocol.  Spare GPUs may run held-out
evaluation after a checkpoint is frozen, but must not share training outputs.

## Fixed stages

1. `motion_base`: REDS 0.55 + GoPro blur-gamma 0.45, random-scratch TURTLE,
   300,000 optimizer steps, LR `1e-3`.
2. `replica`: initialize from completed motion-base, ReplicaBlurry Office3,
   300,000 steps, LR `1e-5`.
3. `defocus_rehearsal`: initialize from completed Replica stage, Unblur-SLAM
   defocus 0.80 + ReplicaBlurry rehearsal 0.20, 300,000 steps, LR `1e-7`.

All video batches use five-frame causal BPTT.  Single-image defocus batches
update only the spatial parameters; Replica rehearsal continues to update the
history-attention and spatial parameters.  AdamW, cosine scheduling, global
batch six, exact sRGB-to-linear conversion, L1 + 0.1 FFT loss, and checkpoint
frequency 6,000 are unchanged across supported GPU counts.

## Install

```bash
git clone https://github.com/SlamMate/Unblur-SLAM.git
cd Unblur-SLAM
git checkout <GITHUB_RELEASE_COMMIT>

export UNBLUR_ARTIFACT_ROOT=/workspace/unblur-artifacts
mkdir -p "$UNBLUR_ARTIFACT_ROOT/external" "$UNBLUR_ARTIFACT_ROOT/pretrained/turtle"
git clone https://github.com/Ascend-Research/Turtle.git \
  "$UNBLUR_ARTIFACT_ROOT/external/TURTLE"
git -C "$UNBLUR_ARTIFACT_ROOT/external/TURTLE" checkout \
  7094f4221b64ad0962b4f27ff1b76d788836e804

conda env create -f environment.yml
conda activate unblur-slam
```

The official GoPro TURTLE checkpoint is not used to initialize the new model;
stage one is random scratch.  The upstream checkout and config are still
content-pinned because they define the architecture.

## Materialize training data

The cloud bundle publishes content-addressed manifests and the derived
ReplicaBlurry Office3 training stream.  REDS and GoPro pixels are downloaded
from their pinned public Hugging Face repositories instead of being
republished.  The Unblur-SLAM defocus source is downloaded from its existing
owner repository.

```bash
python scripts/prepare_turtle_unblur_cloud_data.py \
  --artifact-root "$UNBLUR_ARTIFACT_ROOT" \
  --bundle-revision <HF_BUNDLE_COMMIT>
```

Expected source identities:

- `snah/REDS@62dc25d16e6f43d2214f1b365023abda86f7a0ae`
- `snah/GOPRO_Large@592978466ae510d2734b199cad2fc79a346bda1c`
- `qizhangslam/Unblur_slam_traning_dataset@1f9d98158c3f27f6ec6de45ee2874c9caf2a2c59`
- preprocessing bundle: `qizhangslam/Unblur_slam_traning_dataset@<HF_BUNDLE_COMMIT>`

The GoPro archive contains test members, but the preparation script extracts
only `train/*`.  No GoPro test image is decoded, hashed, evaluated, or used.

## Build and preflight one stage

Set `WORLD=3` on a four-GPU node, or `WORLD=6` on an eight-GPU node.

```bash
export WORLD=6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export MANIFEST_ROOT="$UNBLUR_ARTIFACT_ROOT/turtle_unblur_cloud_v1/manifests"
export OUTPUT_ROOT="$UNBLUR_ARTIFACT_ROOT/outputs/turtle_unblur_three_stage_v1"
mkdir -p "$UNBLUR_ARTIFACT_ROOT/contracts" "$UNBLUR_ARTIFACT_ROOT/receipts"

python scripts/build_turtle_unblur_stable_contract.py \
  --stage motion_base \
  --ddp-world-size "$WORLD" \
  --data-root "$UNBLUR_ARTIFACT_ROOT" \
  --manifest-root "$MANIFEST_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --output "$UNBLUR_ARTIFACT_ROOT/contracts/motion_base.json"

CONTRACT="$UNBLUR_ARTIFACT_ROOT/contracts/motion_base.json"
CONTRACT_SHA=$(sha256sum "$CONTRACT" | awk '{print $1}')
python scripts/preflight_turtle_unblur_stable.py \
  --contract "$CONTRACT" --contract-sha256 "$CONTRACT_SHA" \
  --output "$UNBLUR_ARTIFACT_ROOT/receipts/motion_base.preflight.json"

python scripts/run_turtle_unblur_ddp_probe_and_train.py \
  --contract "$CONTRACT" --contract-sha256 "$CONTRACT_SHA"
```

The runner first executes exactly one DDP optimizer step, verifies the
checkpoint contains every rank's RNG state, then resumes that exact checkpoint
to step 300,000.  AMP overflow is synchronized across ranks: no rank updates,
the scheduler and samplers do not advance, loss scale is reduced, and the same
batch is recomputed.

## Stage two and three

```bash
MOTION_CKPT="$OUTPUT_ROOT/motion_base_seed42/checkpoints/step_300000.pth"
OFFICE_ROOT="$UNBLUR_ARTIFACT_ROOT/turtle_unblur_cloud_v1/replica_blurry_office3"
OFFICE_MANIFEST="$OFFICE_ROOT/manifests/train.jsonl"

python scripts/build_turtle_unblur_stable_contract.py \
  --stage replica --ddp-world-size "$WORLD" \
  --data-root "$UNBLUR_ARTIFACT_ROOT" --manifest-root "$MANIFEST_ROOT" \
  --output-root "$OUTPUT_ROOT" --initialization "$MOTION_CKPT" \
  --office-root "$OFFICE_ROOT" --office-manifest "$OFFICE_MANIFEST" \
  --output "$UNBLUR_ARTIFACT_ROOT/contracts/replica.json"

# Run the same preflight + probe-and-train commands with replica.json.

REPLICA_CKPT="$OUTPUT_ROOT/replica_seed42/checkpoints/step_300000.pth"
python scripts/build_turtle_unblur_stable_contract.py \
  --stage defocus_rehearsal --ddp-world-size "$WORLD" \
  --data-root "$UNBLUR_ARTIFACT_ROOT" --manifest-root "$MANIFEST_ROOT" \
  --output-root "$OUTPUT_ROOT" --initialization "$REPLICA_CKPT" \
  --office-root "$OFFICE_ROOT" --office-manifest "$OFFICE_MANIFEST" \
  --output "$UNBLUR_ARTIFACT_ROOT/contracts/defocus_rehearsal.json"

# Run the same preflight + probe-and-train commands with defocus_rehearsal.json.
```

Stages are strictly sequential.  A partial or non-300k predecessor checkpoint
is rejected.  Training performs no validation or model selection.

## Required returned artifacts

For every stage, return:

- contract JSON and SHA256;
- preflight JSON and SHA256;
- `training.jsonl`;
- every 6,000-step checkpoint plus `.sha256` sidecar;
- stdout/stderr and scheduler job receipt;
- GPU model/count, CUDA, PyTorch, NCCL, driver, wall time, and peak memory.

After stage three, run the preregistered BSD validation, DPDD validation,
Replica temporal controls, and TUM SLAM evaluation.  BSD/DPDD test and Replica
Room2 remain sealed until their respective validation gates pass.
