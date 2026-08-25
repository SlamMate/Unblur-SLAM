#!/usr/bin/env python3
"""Contract-check and (explicitly) launch experimental FrameCrafter LoRA SFT.

The default action is a CPU dataset contract check plus a printed command; it
does not allocate a GPU or start training.  The only enabled real launch
profile is deliberately conservative and still experimental: 2x RTX A6000,
192x336 or smaller, DeepSpeed ZeRO-3, parameter+optimizer CPU offload, and
gradient-checkpointing offload.  It is not evidence that training completed.

The worker executes the official ``external/FrameCrafter/model_training/train.py``
unchanged after replacing only its dataset symbol with the role-aware paired
adapter.  The official model, flow-matching loss, logger, and runner remain in
control.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.framecrafter_lora_dataset import (  # noqa: E402
    PairedFrameCrafterDataset,
    canonical_sha256,
    official_dataset_repeat_is_broken,
    validate_cpu_contract,
)


DEFAULT_FRAMECRAFTER_ROOT = Path("/srv/szha0669/unblur-slam/external/FrameCrafter")
DEFAULT_MODEL_ROOT = Path("/srv/szha0669/unblur-slam/framecrafter_models")
DEFAULT_CHECKPOINT = DEFAULT_MODEL_ROOT / "ckpt/framecrafter.safetensors"
DEFAULT_TARGETS = "q,k,v,o,ffn.0,ffn.2"


def _model_files(root: Path) -> tuple[list[Any], Path]:
    wan = root / "Wan-AI/Wan2.1-I2V-14B-480P"
    shards = sorted(wan.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    t5 = wan / "models_t5_umt5-xxl-enc-bf16.pth"
    vae = wan / "Wan2.1_VAE.pth"
    clip = wan / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    tokenizer = root / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
    missing = [path for path in (t5, vae, clip, tokenizer) if not path.exists()]
    if not shards:
        missing.append(wan / "diffusion_pytorch_model-*-of-*.safetensors")
    if missing:
        raise FileNotFoundError("missing local FrameCrafter model assets: " + ", ".join(map(str, missing)))
    # DiffSynth accepts a list of shard paths as one ModelConfig, matching the
    # official inference loader, followed by the three common model weights.
    return [[str(path) for path in shards], str(t5), str(vae), str(clip)], tokenizer


def _accelerate_executable(value: str | None) -> str:
    if value:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    preferred = Path("/srv/szha0669/unblur-slam/framecrafter_env/bin/accelerate")
    if preferred.is_file():
        return str(preferred)
    found = shutil.which("accelerate")
    if found is None:
        raise FileNotFoundError("accelerate executable not found")
    return found


def _gpu_names(visible: str) -> list[str]:
    ids = [part.strip() for part in visible.split(",") if part.strip()]
    if len(ids) != 2:
        raise ValueError("the experimental launch contract requires exactly two visible GPUs")
    command = [
        "nvidia-smi",
        "--query-gpu=index,name",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    names: dict[str, str] = {}
    for line in result.stdout.splitlines():
        index, name = line.split(",", 1)
        names[index.strip()] = name.strip()
    selected = [names.get(index, "") for index in ids]
    if not all("A6000" in name for name in selected):
        raise RuntimeError(f"real launch is restricted to 2x A6000; selected devices are {selected}")
    return selected


def _run_worker(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    expected = spec.pop("spec_sha256", None)
    if expected != canonical_sha256(spec):
        raise RuntimeError("worker launch specification hash mismatch")
    framecrafter_root = Path(spec["framecrafter_root"])
    train_py = Path(spec["official_train_py"])
    if train_py != framecrafter_root / "model_training/train.py" or not train_py.is_file():
        raise RuntimeError("launch spec does not point to official FrameCrafter train.py")
    sys.path.insert(0, str(framecrafter_root))
    # Import the real package first, then replace only the dataset class that
    # train.py imports by name.  This preserves the official training stack.
    import diffsynth.core.data.dataset as official_dataset  # type: ignore

    official_dataset.WanNVSDataset = PairedFrameCrafterDataset
    sys.argv = [str(train_py), *spec["official_args"]]
    runpy.run_path(str(train_py), run_name="__main__")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-spec", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dataset-root", type=Path, help="Output of build_framecrafter_lora_dataset.py.")
    parser.add_argument("--samples-path", type=Path, default=None)
    parser.add_argument("--framecrafter-root", type=Path, default=DEFAULT_FRAMECRAFTER_ROOT)
    parser.add_argument("--base-model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--framecrafter-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-path", type=Path, default=Path("models/train/unblur-framecrafter-lora"))
    parser.add_argument("--accelerate", default=None, help="Path to accelerate executable.")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=336)
    parser.add_argument("--num-input-frames", type=int, default=6)
    parser.add_argument("--num-output-frames", type=int, default=1)
    parser.add_argument("--dataset-repeat", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-targets", default=DEFAULT_TARGETS)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default="0,1")
    parser.add_argument("--execute", action="store_true", help="Actually start the experimental training command.")
    parser.add_argument(
        "--acknowledge-experimental-2xa6000",
        action="store_true",
        help="Required with --execute; acknowledges that this 14B/2-GPU profile may still OOM.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_spec is not None:
        return _run_worker(args.worker_spec.resolve())
    if args.dataset_root is None:
        raise ValueError("--dataset-root is required")
    dataset_root = args.dataset_root.expanduser().resolve()
    samples_path = (args.samples_path or dataset_root / "samples.jsonl").expanduser().resolve()
    framecrafter_root = args.framecrafter_root.expanduser().resolve()
    model_root = args.base_model_root.expanduser().resolve()
    checkpoint = args.framecrafter_checkpoint.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    train_py = framecrafter_root / "model_training/train.py"
    if not train_py.is_file():
        raise FileNotFoundError(train_py)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.height > 192 or args.width > 336 or args.height % 16 or args.width % 16:
        raise ValueError("2xA6000 experimental mode is restricted to <=192x336 and dimensions divisible by 16")
    if args.num_input_frames < 1 or args.num_output_frames < 1:
        raise ValueError("M and N must be positive")
    if args.dataset_repeat < 1 or args.num_epochs < 1:
        raise ValueError("dataset repeat and epochs must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if args.lora_rank < 1:
        raise ValueError("LoRA rank must be positive")

    contract_dataset = PairedFrameCrafterDataset(
        base_path=str(dataset_root),
        metadata_path=str(samples_path),
        repeat=args.dataset_repeat,
        num_frames=args.num_input_frames + args.num_output_frames,
        height=args.height,
        width=args.width,
        height_division_factor=8,
        width_division_factor=8,
        time_division_factor=4,
        time_division_remainder=1,
        num_input_frames=args.num_input_frames,
        num_output_frames=args.num_output_frames,
    )
    cpu_contract = validate_cpu_contract(contract_dataset)
    repeat_bug = official_dataset_repeat_is_broken(framecrafter_root)
    model_paths, tokenizer_path = _model_files(model_root)
    accelerate = _accelerate_executable(args.accelerate)
    output_path.mkdir(parents=True, exist_ok=True)

    official_args = [
        "--dataset_base_path", str(dataset_root),
        "--dataset_metadata_path", str(samples_path),
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_frames", str(args.num_input_frames + args.num_output_frames),
        "--dataset_repeat", str(args.dataset_repeat),
        "--dataset_num_workers", "0",
        "--model_paths", json.dumps(model_paths, separators=(",", ":")),
        "--tokenizer_path", str(tokenizer_path),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--num_epochs", str(args.num_epochs),
        "--remove_prefix_in_ckpt", "pipe.dit.",
        "--output_path", str(output_path),
        "--lora_base_model", "dit",
        "--lora_target_modules", args.lora_targets,
        "--lora_rank", str(args.lora_rank),
        "--resume_checkpoint", str(checkpoint),
        "--extra_inputs", "input_image",
        "--modify_channels",
        "--new_in_dim", "420",
        "--gradient_accumulation_steps", "1",
        "--use_gradient_checkpointing",
        "--use_gradient_checkpointing_offload",
        "--initialize_model_on_cpu",
        "--individual_encoding",
        "--sampling_strategy", "all_random",
        "--num_input_frames", str(args.num_input_frames),
        "--num_output_frames", str(args.num_output_frames),
    ]
    if args.save_steps is not None:
        official_args.extend(("--save_steps", str(args.save_steps)))

    spec: dict[str, Any] = {
        "schema_version": "framecrafter-lora-launch-v1",
        "framecrafter_root": str(framecrafter_root),
        "official_train_py": str(train_py),
        "official_args": official_args,
        "dataset_adapter": "src.framecrafter_lora_dataset.PairedFrameCrafterDataset",
        "official_repeat_bug_detected": repeat_bug,
        "repeat_fix": "adapter __len__ == sample_count * dataset_repeat",
        "cpu_contract": cpu_contract,
        "profile": {
            "status": "experimental_not_trained",
            "num_processes": 2,
            "mixed_precision": "bf16",
            "zero_stage": 3,
            "offload_optimizer": "cpu",
            "offload_parameters": "cpu",
            "max_resolution": [192, 336],
        },
    }
    spec["spec_sha256"] = canonical_sha256(spec)
    spec_path = output_path / "framecrafter_lora_launch.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    command = [
        accelerate,
        "launch",
        "--num_processes", "2",
        "--mixed_precision", "bf16",
        "--use_deepspeed",
        "--zero_stage", "3",
        "--offload_optimizer_device", "cpu",
        "--offload_param_device", "cpu",
        "--zero3_init_flag", "true",
        str(Path(__file__).resolve()),
        "--worker-spec", str(spec_path),
    ]
    report = {
        "status": "ready_to_launch" if args.execute else "dry_run_only_no_training_started",
        "official_train_py": str(train_py),
        "official_repeat_bug_detected": repeat_bug,
        "adapter_repeat_length": len(contract_dataset),
        "cpu_contract": cpu_contract,
        "defaults": {
            "lora_rank": args.lora_rank,
            "lora_targets": args.lora_targets,
            "learning_rate": args.learning_rate,
        },
        "experimental_profile": "2xA6000, low-resolution, ZeRO-3 CPU parameter+optimizer offload",
        "launch_spec": str(spec_path),
        "command": f"CUDA_VISIBLE_DEVICES={shlex.quote(args.cuda_visible_devices)} "
        + shlex.join(command),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    if not args.acknowledge_experimental_2xa6000:
        raise RuntimeError("--execute requires --acknowledge-experimental-2xa6000")
    names = _gpu_names(args.cuda_visible_devices)
    print(json.dumps({"launching_on": names, "warning": "experimental; completion is not guaranteed"}))
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    subprocess.run(command, check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
