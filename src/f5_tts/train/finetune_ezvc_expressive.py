from __future__ import annotations

import argparse
import json
import shutil
from importlib.resources import files
from pathlib import Path

from cached_path import cached_path

from f5_tts.model import CFM, DiT, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer


DEFAULT_DATASET_NAME = "Expressive_EZVC"
DEFAULT_PRETRAINED_CKPT = "hf://SPRINGLab/EZ-VC/model_2700000.safetensors"


def resolve_cached_path(path: str) -> Path:
    if path.startswith("hf://"):
        return Path(cached_path(path))
    return Path(path).expanduser()


def default_dataset_dir(dataset_name: str) -> Path:
    return Path(str(files("f5_tts").joinpath(f"../../data/{dataset_name}_custom")))


def default_checkpoint_dir(dataset_name: str) -> Path:
    return Path(str(files("f5_tts").joinpath(f"../../ckpts/F5TTS_Base_bigvgan_custom_{dataset_name}")))


def copy_pretrained_checkpoint(pretrained_ckpt: str, checkpoint_dir: Path) -> Path:
    ckpt_path = resolve_cached_path(pretrained_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_name = ckpt_path.name
    if not output_name.startswith("pretrained_"):
        output_name = f"pretrained_{output_name}"
    output_path = checkpoint_dir / output_name
    if not output_path.exists():
        shutil.copy2(ckpt_path, output_path)
        print(f"Copied pretrained checkpoint to {output_path}")
    else:
        print(f"Using existing pretrained checkpoint at {output_path}")
    return output_path


def mel_spec_kwargs() -> dict:
    return {
        "target_sample_rate": 16000,
        "n_mel_channels": 80,
        "hop_length": 160,
        "win_length": 640,
        "n_fft": 1024,
        "mel_spec_type": "bigvgan",
    }


def model_arch() -> dict:
    return {
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "text_mask_padding": False,
        "conv_layers": 4,
        "pe_attn_head": 1,
        "attn_backend": "torch",
        "attn_mask_enabled": False,
        "checkpoint_activations": False,
    }


def read_prepare_summary(dataset_dir: Path) -> dict | None:
    summary_path = dataset_dir / "prepare_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def check_style_metadata(dataset_dir: Path, mode: str) -> None:
    labels_path = dataset_dir / "style_labels.jsonl"
    if mode == "none":
        return
    if labels_path.exists():
        print(
            "Found style label metadata. Current training keeps it as metadata only; "
            "no style-control tokens or style encoder are consumed by this checkpoint."
        )
    else:
        print("No style label metadata found. Continuing with unlabeled expressive fine-tuning.")


def check_source_prosody_sidecars(dataset_dir: Path, architecture_mode: str) -> None:
    if architecture_mode == "unit-only":
        return

    prosody_manifest = dataset_dir / "prosody_manifest.jsonl"
    if not prosody_manifest.exists():
        raise FileNotFoundError(
            f"{architecture_mode} requires {prosody_manifest}. "
            "Prepare the dataset with --source-prosody-mode sidecar first."
        )

    total = 0
    missing = []
    with open(prosody_manifest, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            prosody_path = Path(row["prosody_path"])
            if not prosody_path.exists():
                missing.append(prosody_path.as_posix())
                if len(missing) >= 5:
                    break

    if missing:
        raise FileNotFoundError(f"Missing prosody sidecars: {missing}")

    print(
        f"Validated {total} source-prosody sidecars. "
        "This run still trains the current unit-conditioned decoder; the flag records and validates "
        "the future source-prosody architecture contract."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the EZ-VC decoder from the current pretrained checkpoint on expressive unit datasets.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Prepared dataset directory. Defaults to data/<dataset-name>_custom.",
    )
    parser.add_argument("--pretrained-ckpt", default=DEFAULT_PRETRAINED_CKPT)
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Output checkpoint directory. Defaults to ckpts/F5TTS_Base_bigvgan_custom_<dataset-name>.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Custom vocab path. Defaults to <dataset-dir>/vocab.txt.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-warmup-updates", type=int, default=100)
    parser.add_argument("--batch-size-per-gpu", type=int, default=2400)
    parser.add_argument("--batch-size-type", default="frame", choices=["frame", "sample"])
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--mixed-precision",
        default=None,
        choices=["no", "none", "fp16", "bf16", "fp8"],
        help="Accelerate mixed precision mode. Defaults to the accelerate launch/config setting.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-per-updates", type=int, default=1000)
    parser.add_argument("--last-per-updates", type=int, default=100)
    parser.add_argument("--keep-last-n-checkpoints", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--logger", default="none", choices=["none", "wandb", "tensorboard"])
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--bnb-optimizer", action="store_true")
    parser.add_argument(
        "--style-conditioning",
        default="metadata-only",
        choices=["none", "metadata-only"],
        help="Reserved hook for labeled expressive datasets. Current checkpoint only supports metadata-only labels.",
    )
    parser.add_argument(
        "--architecture-mode",
        default="unit-only",
        choices=["unit-only", "source-prosody-sidecar"],
        help=(
            "unit-only trains the current EZ-VC decoder. source-prosody-sidecar validates F0/UV/energy sidecars "
            "for the future source-prosody branch while keeping this checkpoint's model shape unchanged."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build dataset/model objects and stage the pretrained checkpoint, but do not start training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = None if args.logger == "none" else args.logger
    dataset_dir = Path(args.dataset_dir).expanduser() if args.dataset_dir else default_dataset_dir(args.dataset_name)
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser() if args.checkpoint_dir else default_checkpoint_dir(args.dataset_name)
    )
    tokenizer_path = Path(args.tokenizer_path).expanduser() if args.tokenizer_path else dataset_dir / "vocab.txt"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {dataset_dir}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer vocab not found: {tokenizer_path}")

    summary = read_prepare_summary(dataset_dir)
    if summary:
        print("Prepared dataset summary:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    check_style_metadata(dataset_dir, args.style_conditioning)
    check_source_prosody_sidecars(dataset_dir, args.architecture_mode)

    copy_pretrained_checkpoint(args.pretrained_ckpt, checkpoint_dir)

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path.as_posix(), "custom")
    print(f"Vocab size: {vocab_size}")
    print("Loading dataset ...")
    if args.dataset_dir:
        train_dataset = load_dataset(
            dataset_dir.as_posix(),
            "custom",
            dataset_type="CustomDatasetPath",
            mel_spec_kwargs=mel_spec_kwargs(),
        )
    else:
        train_dataset = load_dataset(args.dataset_name, "custom", mel_spec_kwargs=mel_spec_kwargs())
    print(f"Samples: {len(train_dataset)}")

    model = CFM(
        transformer=DiT(**model_arch(), text_num_embeds=vocab_size, mel_dim=mel_spec_kwargs()["n_mel_channels"]),
        mel_spec_kwargs=mel_spec_kwargs(),
        vocab_char_map=vocab_char_map,
    )

    trainer = Trainer(
        model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_dir.as_posix(),
        batch_size_per_gpu=args.batch_size_per_gpu,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logger=logger,
        wandb_project="EZ-VC-Expressive",
        wandb_run_name=args.dataset_name,
        last_per_updates=args.last_per_updates,
        log_samples=args.log_samples,
        bnb_optimizer=args.bnb_optimizer,
        mel_spec_type="bigvgan",
        model_cfg_dict={
            "dataset_name": args.dataset_name,
            "dataset_dir": dataset_dir.as_posix(),
            "pretrained_ckpt": args.pretrained_ckpt,
            "checkpoint_dir": checkpoint_dir.as_posix(),
            "style_conditioning": args.style_conditioning,
            "architecture_mode": args.architecture_mode,
            "mixed_precision": args.mixed_precision,
            "mel_spec": mel_spec_kwargs(),
            "model_arch": model_arch(),
        },
        accelerate_kwargs={"mixed_precision": args.mixed_precision},
    )

    if args.dry_run:
        print("Dry run complete. Training was not started.")
        return

    trainer.train(
        train_dataset,
        num_workers=args.num_workers,
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()
