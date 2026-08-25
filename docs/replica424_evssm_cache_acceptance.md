# Replica424 official-EVSSM cache acceptance

This is the production provenance gate for the frozen single-frame EVSSM
teacher cache.  It is deliberately separate from model/training code and runs
entirely on CPU.  Passing it does not claim that a causal model is good; it
only proves that the cached teacher inputs are exactly the registered official
Unblur-SLAM EVSSM outputs for the declared data roles.

## Registered inputs

```text
data root: /srv/szha0669/unblur-slam/causal_video_data
contract:  configs/local/causal_evssm_v3_replica424_contract.json
checkpoint: /srv/szha0669/unblur-slam/pretrained/net_g_latest_batch_8_no_NYU.pth
checkpoint SHA-256: 4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41
train:        replica424_v1/manifests/train.jsonl        (234 frames)
selection:    replica424_v1/manifests/val_temporal.jsonl (16 frames)
one-shot test: replica424_v1/manifests/test_room2.jsonl  (174 frames)
```

GoPro and TURTLE checkpoints are forbidden by the registered contract and by
the acceptance script.

## Cache-generation commands

The following commands are recorded for the GPU worker; this acceptance task
does not execute them.  Each output directory must be new or empty because the
precompute program refuses to overwrite artifacts.

```bash
cd /home/szha0669/Unblur-SLAM-framecrafter-resplat

/srv/szha0669/unblur-slam/env/bin/python \
  scripts/precompute_video_deblur_evssm.py \
  --input-manifest /srv/szha0669/unblur-slam/causal_video_data/replica424_v1/manifests/train.jsonl \
  --data-root /srv/szha0669/unblur-slam/causal_video_data \
  --checkpoint /srv/szha0669/unblur-slam/pretrained/net_g_latest_batch_8_no_NYU.pth \
  --output-dir /srv/szha0669/unblur-slam/causal_video_data/replica424_v1/evssm_train \
  --device cuda:N

/srv/szha0669/unblur-slam/env/bin/python \
  scripts/precompute_video_deblur_evssm.py \
  --input-manifest /srv/szha0669/unblur-slam/causal_video_data/replica424_v1/manifests/val_temporal.jsonl \
  --data-root /srv/szha0669/unblur-slam/causal_video_data \
  --checkpoint /srv/szha0669/unblur-slam/pretrained/net_g_latest_batch_8_no_NYU.pth \
  --output-dir /srv/szha0669/unblur-slam/causal_video_data/replica424_v1/evssm_val_temporal \
  --device cuda:N
```

Do not generate or inspect the room2 cache until the registered temporal-
validation selection gate permits opening the one-shot test.  At that point:

```bash
/srv/szha0669/unblur-slam/env/bin/python \
  scripts/precompute_video_deblur_evssm.py \
  --input-manifest /srv/szha0669/unblur-slam/causal_video_data/replica424_v1/manifests/test_room2.jsonl \
  --data-root /srv/szha0669/unblur-slam/causal_video_data \
  --checkpoint /srv/szha0669/unblur-slam/pretrained/net_g_latest_batch_8_no_NYU.pth \
  --output-dir /srv/szha0669/unblur-slam/causal_video_data/replica424_v1/evssm_test_room2 \
  --device cuda:N
```

Replace `cuda:N` only with the GPU assigned by the experiment owner.  Never
guess a device or interfere with another process.

## CPU-only acceptance command

Run only after all three `precompute_report.json` files exist:

```bash
cd /home/szha0669/Unblur-SLAM-framecrafter-resplat
CUDA_VISIBLE_DEVICES="" /srv/szha0669/unblur-slam/env/bin/python \
  scripts/audit_replica424_evssm_cache.py \
  --contract configs/local/causal_evssm_v3_replica424_contract.json \
  --source-inventory /srv/szha0669/unblur-slam/causal_video_data/replica424_source_inventory.json \
  --split-ranges /srv/szha0669/unblur-slam/causal_video_data/replica424_v1/split_ranges.json \
  --data-root /srv/szha0669/unblur-slam/causal_video_data \
  --train-report /srv/szha0669/unblur-slam/causal_video_data/replica424_v1/evssm_train/precompute_report.json \
  --val-temporal-report /srv/szha0669/unblur-slam/causal_video_data/replica424_v1/evssm_val_temporal/precompute_report.json \
  --test-room2-report /srv/szha0669/unblur-slam/causal_video_data/replica424_v1/evssm_test_room2/precompute_report.json \
  --output /srv/szha0669/unblur-slam/causal_video_data/replica424_v1/evssm_cache_acceptance.json
```

The command exits nonzero and writes `production_eligible: false` if any check
fails.  It writes `production_eligible: true` only when all of the following
hold simultaneously:

- the pre-registered experiment, source-validation report, split-ranges file,
  and all three source-manifest SHA-256 digests are unchanged;
- every source path and byte hash matches the pinned Hugging Face inventory;
- every cache uses the exact official Unblur-SLAM EVSSM checkpoint SHA;
- output sequence/frame ordering is identical to its source manifest;
- every reported blurry, sharp, and teacher SHA matches the current file;
- every teacher is an RGB PNG with the same dimensions as its input;
- every teacher stays inside its cache-local `teacher/` directory;
- source pairs, cache roots, output manifests, and teacher artifacts do not
  overlap across train, temporal validation, and room2 test.

Eligibility is scoped to the cache's declared role.  `test_room2` remains a
one-shot test-only cache and is never training eligible.
