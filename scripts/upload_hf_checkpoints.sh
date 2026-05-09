#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:-qizhangslam/Unblur-SLAM-checkpoints}"

huggingface-cli upload "$REPO_ID" /var/scratch/qzhang2/Deblur-SLAM/thirdparty/EVSSM/experiments EVSSM/experiments --repo-type model
huggingface-cli upload "$REPO_ID" /var/scratch/qzhang2/Deblur-SLAM/thirdparty/EVSSM_defocus/experiments EVSSM_defocus/experiments --repo-type model
huggingface-cli upload "$REPO_ID" /var/scratch/qzhang2/Deblur-SLAM/thirdparty/EVSSM_replica/experiments EVSSM_replica/experiments --repo-type model
huggingface-cli upload "$REPO_ID" /var/scratch/qzhang2/Deblur-SLAM/thirdparty/EVSSM_Scannet/experiments EVSSM_Scannet/experiments --repo-type model
