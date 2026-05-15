from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from cached_path import cached_path
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


AUDIO_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}
DEFAULT_DATA_DIR = Path("data")
DEFAULT_VOCAB = "hf://SPRINGLab/EZ-VC/vocab.txt"
PROSODY_SAMPLE_RATE = 16000
PROSODY_HOP_LENGTH = 160
PROSODY_FRAME_LENGTH = 640


@dataclass(frozen=True)
class ManifestItem:
    audio_path: Path
    style_label: str | None = None


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_cached_path(path: str) -> Path:
    if path.startswith("hf://"):
        return Path(cached_path(path))
    return Path(path).expanduser()


def sniff_delimiter(manifest_path: Path) -> str:
    sample = manifest_path.read_text(encoding="utf-8-sig")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="|,\t")
        return dialect.delimiter
    except csv.Error:
        return "|"


def resolve_audio_path(raw_path: str, audio_root: Path | None, manifest_parent: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    candidates = []
    if audio_root is not None:
        candidates.append(audio_root / path)
    if manifest_parent is not None:
        candidates.append(manifest_parent / path)
    candidates.append(path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_jsonl_manifest(
    manifest_path: Path,
    audio_column: str,
    label_column: str | None,
    audio_root: Path | None,
    require_labels: bool,
) -> list[ManifestItem]:
    items = []
    with open(manifest_path, encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if audio_column not in row:
                raise ValueError(f"{manifest_path}:{line_number} is missing audio column '{audio_column}'")
            label = row.get(label_column) if label_column else None
            if require_labels and not label:
                raise ValueError(f"{manifest_path}:{line_number} is missing label column '{label_column}'")
            items.append(
                ManifestItem(
                    audio_path=resolve_audio_path(str(row[audio_column]), audio_root, manifest_path.parent),
                    style_label=str(label).strip() if label else None,
                )
            )
    return items


def read_csv_manifest(
    manifest_path: Path,
    audio_column: str,
    label_column: str | None,
    audio_root: Path | None,
    require_labels: bool,
    delimiter: str | None,
) -> list[ManifestItem]:
    items = []
    actual_delimiter = delimiter or sniff_delimiter(manifest_path)
    with open(manifest_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=actual_delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{manifest_path} has no header row")
        if audio_column not in reader.fieldnames:
            raise ValueError(f"{manifest_path} is missing audio column '{audio_column}'")
        if label_column and label_column not in reader.fieldnames:
            raise ValueError(f"{manifest_path} is missing label column '{label_column}'")

        for line_number, row in enumerate(reader, start=2):
            raw_audio_path = row.get(audio_column)
            if not raw_audio_path:
                raise ValueError(f"{manifest_path}:{line_number} has an empty audio path")
            label = row.get(label_column) if label_column else None
            if require_labels and not label:
                raise ValueError(f"{manifest_path}:{line_number} is missing label column '{label_column}'")
            items.append(
                ManifestItem(
                    audio_path=resolve_audio_path(raw_audio_path, audio_root, manifest_path.parent),
                    style_label=label.strip() if label else None,
                )
            )
    return items


def scan_audio_dir(audio_dir: Path) -> list[ManifestItem]:
    return [
        ManifestItem(audio_path=path)
        for path in sorted(audio_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]


def load_manifest(args: argparse.Namespace) -> list[ManifestItem]:
    audio_root = Path(args.audio_root).expanduser() if args.audio_root else None
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser()
        if manifest_path.suffix.lower() == ".jsonl":
            return read_jsonl_manifest(
                manifest_path,
                args.audio_column,
                args.label_column,
                audio_root,
                args.require_labels,
            )
        return read_csv_manifest(
            manifest_path,
            args.audio_column,
            args.label_column,
            audio_root,
            args.require_labels,
            args.delimiter,
        )

    if not args.audio_dir:
        raise ValueError("Provide either --manifest or --audio-dir")
    return scan_audio_dir(Path(args.audio_dir).expanduser())


def audio_duration_seconds(audio_path: Path) -> float:
    info = sf.info(audio_path)
    return float(info.frames / info.samplerate)


def extract_source_prosody(
    audio_path: Path,
    prosody_dir: Path,
    index: int,
    f0_min_hz: float,
    f0_max_hz: float,
) -> Path:
    y, _ = librosa.load(audio_path, sr=PROSODY_SAMPLE_RATE, mono=True)
    f0_hz, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=f0_min_hz,
        fmax=f0_max_hz,
        sr=PROSODY_SAMPLE_RATE,
        frame_length=PROSODY_FRAME_LENGTH,
        hop_length=PROSODY_HOP_LENGTH,
    )
    rms = librosa.feature.rms(
        y=y,
        frame_length=PROSODY_FRAME_LENGTH,
        hop_length=PROSODY_HOP_LENGTH,
        center=True,
    )[0]

    length = min(len(f0_hz), len(voiced_flag), len(voiced_prob), len(rms))
    f0_hz = f0_hz[:length]
    voiced_flag = voiced_flag[:length]
    voiced_prob = voiced_prob[:length]
    rms = rms[:length]

    clean_f0_hz = np.nan_to_num(f0_hz, nan=0.0).astype(np.float32)
    log_f0 = np.zeros_like(clean_f0_hz, dtype=np.float32)
    voiced = (voiced_flag.astype(bool) & (clean_f0_hz > 0)).astype(np.float32)
    log_f0[voiced > 0] = np.log(clean_f0_hz[voiced > 0])

    prosody_dir.mkdir(parents=True, exist_ok=True)
    prosody_path = prosody_dir / f"{index:08d}_{audio_path.stem}.npz"
    np.savez_compressed(
        prosody_path,
        f0_hz=clean_f0_hz,
        log_f0=log_f0,
        voiced=voiced,
        voiced_prob=np.nan_to_num(voiced_prob, nan=0.0).astype(np.float32),
        energy=rms.astype(np.float32),
        sample_rate=np.array(PROSODY_SAMPLE_RATE, dtype=np.int32),
        hop_length=np.array(PROSODY_HOP_LENGTH, dtype=np.int32),
        frame_length=np.array(PROSODY_FRAME_LENGTH, dtype=np.int32),
    )
    return prosody_path


def prepare_dataset(args: argparse.Namespace) -> None:
    from f5_tts.infer.utils_xeus import ApplyKmeans, extract_units, load_xeus_model

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else DEFAULT_DATA_DIR / f"{args.dataset_name}_custom"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_manifest(args)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise RuntimeError("No audio files found.")

    device = select_device(args.device)
    print(f"Loading frozen EZ-VC frontend on {device} ...")
    xeus_model = load_xeus_model(device).eval()
    apply_kmeans = ApplyKmeans(device)

    rows = []
    durations = []
    style_sidecar = []
    prosody_sidecar = []
    skipped = []
    write_style_label = args.label_column is not None
    prosody_dir = out_dir / "prosody"
    extract_prosody = args.source_prosody_mode == "sidecar"

    for index, item in enumerate(tqdm(items, desc="Extracting EZ-VC units")):
        audio_path = item.audio_path
        if not audio_path.exists():
            skipped.append({"audio_path": audio_path.as_posix(), "reason": "missing"})
            continue

        try:
            duration = audio_duration_seconds(audio_path)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"audio_path": audio_path.as_posix(), "reason": f"duration error: {exc}"})
            continue

        if duration < args.min_duration or duration > args.max_duration:
            skipped.append(
                {
                    "audio_path": audio_path.as_posix(),
                    "reason": f"duration {duration:.2f}s outside {args.min_duration}-{args.max_duration}s",
                }
            )
            continue

        try:
            unit_text = extract_units(
                audio_path.as_posix(),
                xeus_model,
                apply_kmeans,
                device,
                layer_index=args.xeus_layer,
                deduplicate=False,
                max_consecutive_units=args.repeat_cap,
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append({"audio_path": audio_path.as_posix(), "reason": f"unit extraction error: {exc}"})
            continue

        if not unit_text:
            skipped.append({"audio_path": audio_path.as_posix(), "reason": "empty unit string"})
            continue

        row = {
            "audio_path": audio_path.resolve().as_posix(),
            "text": unit_text,
            "duration": duration,
        }
        if extract_prosody:
            try:
                prosody_path = extract_source_prosody(
                    audio_path,
                    prosody_dir,
                    index,
                    args.f0_min_hz,
                    args.f0_max_hz,
                )
            except Exception as exc:  # noqa: BLE001
                skipped.append({"audio_path": audio_path.as_posix(), "reason": f"prosody extraction error: {exc}"})
                continue
            row["prosody_path"] = prosody_path.resolve().as_posix()
            prosody_sidecar.append(
                {
                    "audio_path": row["audio_path"],
                    "prosody_path": row["prosody_path"],
                    "kind": "f0_uv_energy",
                }
            )
        if write_style_label:
            row["style_label"] = item.style_label or ""
        if item.style_label:
            style_sidecar.append({"audio_path": row["audio_path"], "style_label": item.style_label})
        rows.append(row)
        durations.append(duration)

    if not rows:
        raise RuntimeError("No usable samples were prepared.")

    raw_arrow_path = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix()) as writer:
        for row in tqdm(rows, desc="Writing raw.arrow"):
            writer.write(row)
        writer.finalize()

    with open(out_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f, ensure_ascii=False, indent=2)

    vocab_path = resolve_cached_path(args.vocab_file)
    shutil.copy2(vocab_path, out_dir / "vocab.txt")

    with open(out_dir / "expressive_manifest.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if style_sidecar:
        with open(out_dir / "style_labels.jsonl", "w", encoding="utf-8") as f:
            for row in style_sidecar:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if prosody_sidecar:
        with open(out_dir / "prosody_manifest.jsonl", "w", encoding="utf-8") as f:
            for row in prosody_sidecar:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset_name": args.dataset_name,
        "out_dir": out_dir.as_posix(),
        "samples": len(rows),
        "skipped": len(skipped),
        "hours": sum(durations) / 3600,
        "xeus_layer": args.xeus_layer,
        "repeat_cap": args.repeat_cap,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "style_labels": bool(style_sidecar),
        "source_prosody_mode": args.source_prosody_mode,
        "prosody_sidecars": len(prosody_sidecar),
        "prosody_sample_rate": PROSODY_SAMPLE_RATE if prosody_sidecar else None,
        "prosody_hop_length": PROSODY_HOP_LENGTH if prosody_sidecar else None,
        "note": (
            "style_label and prosody_path are stored as metadata/sidecars only; "
            "the current EZ-VC decoder does not consume them until a source-prosody branch is added."
        ),
    }
    with open(out_dir / "prepare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if skipped:
        with open(out_dir / "skipped.jsonl", "w", encoding="utf-8") as f:
            for row in skipped:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare expressive speech/audio clips for EZ-VC decoder fine-tuning.",
    )
    parser.add_argument("--dataset-name", default="Expressive_EZVC", help="Dataset name under data/<name>_custom")
    parser.add_argument("--out-dir", default=None, help="Prepared dataset output directory")
    parser.add_argument("--manifest", default=None, help="CSV/TSV/pipe-delimited or JSONL manifest")
    parser.add_argument("--audio-dir", default=None, help="Directory to recursively scan when no manifest is provided")
    parser.add_argument("--audio-root", default=None, help="Base directory for relative manifest audio paths")
    parser.add_argument("--audio-column", default="audio_path", help="Manifest column/key containing audio path")
    parser.add_argument("--label-column", default=None, help="Optional expressive style label column/key")
    parser.add_argument("--require-labels", action="store_true", help="Require every manifest row to contain a label")
    parser.add_argument("--delimiter", default=None, help="CSV delimiter override, e.g. '|' or ','")
    parser.add_argument("--vocab-file", default=DEFAULT_VOCAB, help="EZ-VC vocab file or hf:// path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps", "xpu"])
    parser.add_argument("--xeus-layer", type=int, default=14, help="XEUS hidden-state layer for k-means extraction")
    parser.add_argument(
        "--repeat-cap",
        type=int,
        default=2,
        help="Max consecutive identical unit tokens to keep. Use 1 for dedup, 0 to keep all.",
    )
    parser.add_argument("--min-duration", type=float, default=0.3, help="Minimum clip length in seconds")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Maximum clip length in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke testing")
    parser.add_argument(
        "--source-prosody-mode",
        default="none",
        choices=["none", "sidecar"],
        help="Extract source F0/voicing/energy sidecars for future source-prosody conditioning.",
    )
    parser.add_argument("--f0-min-hz", type=float, default=50.0, help="Minimum F0 for pyin sidecar extraction")
    parser.add_argument("--f0-max-hz", type=float, default=1100.0, help="Maximum F0 for pyin sidecar extraction")
    return parser.parse_args()


def main() -> None:
    prepare_dataset(parse_args())


if __name__ == "__main__":
    main()
