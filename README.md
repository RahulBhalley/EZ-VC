# EZ-VC: Easy Zero-shot Any-to-Any Voice Conversion

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/EZ-VC/EZ-VC)
[![arXiv](https://img.shields.io/badge/arXiv-2505.16691-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2505.16691)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://ez-vc.github.io/EZ-VC-Demo/)
[![huggingface](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/SPRINGLab/EZ-VC)
[![lab](https://img.shields.io/badge/SPRING-Lab-grey?labelColor=lightgrey)](https://asr.iitm.ac.in/)
<!-- <img src="https://github.com/user-attachments/assets/12d7749c-071a-427c-81bf-b87b91def670" alt="Watermark" style="width: 40px; height: auto"> -->


### Our paper has been accepted to the Findings of EMNLP 2025!

## Installation

### Upstream Python 3.10 setup

```bash
# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n ez-vc python=3.10
conda activate ez-vc
```

### Install PyTorch with matched device

<details>
<summary>NVIDIA GPU</summary>

> ```bash
> # Install pytorch with your CUDA version, e.g.
> pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
> ```

</details>

<details>
<summary>AMD GPU</summary>

> ```bash
> # Install pytorch with your ROCm version (Linux only), e.g.
> pip install torch==2.5.1+rocm6.2 torchaudio==2.5.1+rocm6.2 --extra-index-url https://download.pytorch.org/whl/rocm6.2
> ```

</details>

<details>
<summary>Intel GPU</summary>

> ```bash
> # Install pytorch with your XPU version, e.g.
> # Intel® Deep Learning Essentials or Intel® oneAPI Base Toolkit must be installed
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/test/xpu
> 
> # Intel GPU support is also available through IPEX (Intel® Extension for PyTorch)
> # IPEX does not require the Intel® Deep Learning Essentials or Intel® oneAPI Base Toolkit
> # See: https://pytorch-extension.intel.com/installation?request=platform
> ```

</details>

<details>
<summary>Apple Silicon</summary>

> ```bash
> # Install the stable pytorch, e.g.
> pip install torch torchaudio
> ```

</details>

### Local installation

```bash
git clone https://github.com/EZ-VC/EZ-VC
cd EZ-VC
git submodule update --init --recursive
pip install -e .

# Install espnet for xeus (Exactly this version)
pip install 'espnet @ git+https://github.com/wanchichen/espnet.git@ssl'
```

### Tested local setup: Python 3.12 on Apple Silicon

This checkout has also been tested locally with a separate Python 3.12 venv.
The ESPnet fork used for XEUS has old dependency pins, so install its code
without forcing its incompatible pins, then install the runtime packages.

```bash
cd /Users/rahulb/s2s-vc/ez-vc
git submodule update --init --recursive

python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e .

# ESPnet/XEUS workaround for Python 3.12.
.venv/bin/python -m pip install --no-deps 'espnet @ git+https://github.com/wanchichen/espnet.git@ssl'
.venv/bin/python -m pip install \
  configargparse typeguard humanfriendly librosa==0.9.2 jamo h5py kaldiio \
  torch_complex nltk g2p_en espnet_tts_frontend opt-einsum editdistance \
  sentencepiece resampy inflect distance more_itertools jaconv
```

Notes:

- `torchcodec` is required for current `torchaudio.load()` behavior and is
  included in `pyproject.toml`.
- `pip check` may still report ESPnet's old pins, especially `numpy<1.24`.
  This is expected for the Python 3.12 workaround; the local inference and
  tiny training smoke test still ran successfully.
- The vendored BigVGAN loader has been patched for current
  `huggingface_hub` arguments.

### Hugging Face access

The pretrained EZ-VC assets are hosted at
[SPRINGLab/EZ-VC](https://huggingface.co/SPRINGLab/EZ-VC). Request/accept
access there first, then provide a token at runtime. Do not commit tokens.

```bash
export HF_TOKEN="hf_..."
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
```

## Inference

### Notebook

Open [Inference notebook](src/f5_tts/infer/infer.ipynb).

Run all. 

The converted audio will be available at the last cell.

### Gradio UI

A local Gradio UI is available for source/reference voice conversion.

```bash
cd /Users/rahulb/s2s-vc/ez-vc
scripts/start_inference_gradio.sh
```

Open [http://localhost:7861](http://localhost:7861).

The legacy shortcut still starts the same inference UI:

```bash
./run_ez_vc.zh
```

The training/fine-tuning Gradio UI runs on port 7862 by default:

```bash
scripts/start_training_gradio.sh
```

Open [http://localhost:7862](http://localhost:7862).

For 16 GB Apple Silicon Macs, the UI defaults to low-memory behavior:

- one queued conversion at a time
- lazy model loading
- unload models after each conversion by default
- optional `Keep models loaded` checkbox for faster repeated conversions
- explicit `Unload Models` button

## Kaggle Notebook

Use [notebooks/ezvc_kaggle_inference.ipynb](notebooks/ezvc_kaggle_inference.ipynb)
for Kaggle inference. Add a Kaggle secret named `HF_TOKEN` before running it.
The notebook handles clone, submodules, Python package setup, gated asset
download, conversion, and memory cleanup.

For Gradio-first cloud use, use:

- [notebooks/ezvc_kaggle_gradio_ui.ipynb](notebooks/ezvc_kaggle_gradio_ui.ipynb)
  on Kaggle.
- [notebooks/ezvc_runpod_gradio_ui.ipynb](notebooks/ezvc_runpod_gradio_ui.ipynb)
  on RunPod.

Both notebooks clone this repo, set up `.venv`, and provide separate cells to
start the inference UI and training UI with `--host 0.0.0.0 --share`, so Gradio
prints a public `gradio.live` URL. Provide `HF_TOKEN` as a Kaggle secret,
RunPod environment variable, or paste it into the UI's `HF Token` field.

## Training

Short answer: yes, this repo can run training from scratch, but real EZ-VC
training needs EZ-VC-style unit targets, not normal transcripts.

### What the model expects

The training dataset still has an audio field and a `text` field because EZ-VC
uses the F5-TTS training path. For normal F5-TTS, `text` is natural language.
For EZ-VC, `text` should be a string of discrete speech units:

```text
raw audio -> XEUS features -> multilingual k-means IDs -> unit-token string
```

So you do not need to manually write transcripts for each audio file. You do
need a preprocessing step that turns your audio into XEUS + k-means unit
strings and writes the prepared dataset files expected by the trainer:

```text
data/<Dataset>_custom/raw.arrow
data/<Dataset>_custom/duration.json
data/<Dataset>_custom/vocab.txt
```

For expressive audio-only fine-tuning, use
`src/f5_tts/train/datasets/prepare_ezvc_expressive.py`. It loads the frozen
XEUS checkpoint and EZ-VC multilingual k-means model, extracts unit strings for
each clip, copies the official EZ-VC vocab, and writes the trainer-ready
dataset.

### Expressive EZ-VC fine-tuning

This workflow is for adapting the current EZ-VC decoder toward expressive
speech such as movie dialogue, podcasts, interviews, dramatic reads, and other
media speech. It does not add a new prosody encoder yet. The goal is to
fine-tune the existing decoder so it becomes less biased toward flat/read
speech and better uses source-unit timing at inference.

What is loaded:

- Frozen frontend during dataset prep:
  - `hf://espnet/xeus/model/xeus_checkpoint_old.pth`
  - `hf://SPRINGLab/EZ-VC/kmeans_xeus_500_multilingual.pkl`
- Trainable decoder during fine-tuning:
  - `hf://SPRINGLab/EZ-VC/model_2700000.safetensors`
- Vocab:
  - `hf://SPRINGLab/EZ-VC/vocab.txt`
- Vocoder:
  - BigVGAN remains pretrained. It is used for sample logging/inference, not
    trained by these scripts.

Important limitation: with the current architecture, expressiveness can only be
learned from what survives the unit string and duration path. This should help
with expressive speech distributions, but it is not the same as adding explicit
F0, energy, rhythm, or emotion-conditioning branches. See
[`SOURCE_PROSODY_TRANSFER_ARCHITECTURE.md`](SOURCE_PROSODY_TRANSFER_ARCHITECTURE.md)
for the target architecture that preserves source expression while transferring
target timbre. See [`SINGING_ARCHITECTURE_NOTES.md`](SINGING_ARCHITECTURE_NOTES.md)
for the singing-specific extension of the same idea.

#### Dataset input

Use short phrase-level clips, ideally isolated speech without music or heavy
background noise.

Recommended clip constraints:

- `0.3s` to `30s`; longer clips are skipped by default because the existing
  dataset loader filters long samples.
- mono or stereo is fine; training converts to mono.
- clips should preserve natural pauses, emphasis, and emotional variation.
- for movies/podcasts, separate or reject clips with strong music, effects, or
  overlapping speakers when possible.

Manifest CSV example:

```csv
audio_path|style
wavs/movie_angry_001.wav|angry
wavs/podcast_excited_002.wav|excited
wavs/interview_soft_003.wav|soft
```

Labels are optional. If you pass `--label-column`, labels are stored as metadata
in `style_labels.jsonl`, but the current EZ-VC checkpoint does not consume them.
That flag is a forward-compatible hook for a later style-conditioned model.

You can also skip labels entirely:

```csv
audio_path
wavs/movie_001.wav
wavs/podcast_002.wav
wavs/audiobook_003.wav
```

Or scan a directory recursively with `--audio-dir`.

#### Prepare expressive units

```bash
cd /Users/rahulb/s2s-vc/ez-vc

scripts/prepare_ezvc_dataset.sh \
  --manifest /path/to/expressive_manifest.csv \
  --audio-root /path/to/expressive_dataset \
  --audio-column audio_path \
  --label-column style \
  --source-prosody-sidecar
```

Directory scan:

```bash
scripts/prepare_ezvc_dataset.sh --audio-dir /path/to/expressive_clips
```

The wrapper defaults to `--dataset-name Expressive_EZVC`, `--device auto`,
`--xeus-layer 14`, `--repeat-cap 2`, `--min-duration 0.3`, and
`--max-duration 30.0`. Use `--limit N` for a quick preprocessing smoke run.

Outputs:

```text
data/Expressive_EZVC_custom/raw.arrow
data/Expressive_EZVC_custom/duration.json
data/Expressive_EZVC_custom/vocab.txt
data/Expressive_EZVC_custom/expressive_manifest.jsonl
data/Expressive_EZVC_custom/prepare_summary.json
data/Expressive_EZVC_custom/style_labels.jsonl      # only if labels are present
data/Expressive_EZVC_custom/prosody/*.npz           # if --source-prosody-mode sidecar
data/Expressive_EZVC_custom/prosody_manifest.jsonl  # if --source-prosody-mode sidecar
data/Expressive_EZVC_custom/skipped.jsonl           # only if samples were skipped
```

`--source-prosody-mode sidecar` extracts F0, log-F0, voiced/unvoiced,
voicing-probability, and energy arrays into compressed `.npz` files. These are
not consumed by the current EZ-VC checkpoint yet; they establish the data
contract for the future source-prosody branch described in
`SOURCE_PROSODY_TRANSFER_ARCHITECTURE.md`.

Repeat cap controls adjacent unit collapse:

- `--repeat-cap 1`: original dedup-like behavior.
- `--repeat-cap 2` or `3`: keeps some timing/prosody density while limiting
  token explosion. This is the recommended first expressive setting.
- `--repeat-cap 0`: keeps all repeated unit frames. This may preserve more
  duration detail but can create very long conditioning strings.

Use the same repeat-cap family at inference when evaluating the fine-tuned
model. If you train on capped repeats and infer with full dedup, the model will
not see the same source-timing signal.

#### Fine-tune from the current EZ-VC checkpoint

Conservative local smoke run:

```bash
cd /Users/rahulb/s2s-vc/ez-vc

HOME=/Users/rahulb/s2s-vc/ez-vc/.home \
MPLCONFIGDIR=/Users/rahulb/s2s-vc/ez-vc/.matplotlib \
NUMBA_CACHE_DIR=/Users/rahulb/s2s-vc/ez-vc/.numba_cache \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  .venv-py312/bin/python src/f5_tts/train/finetune_ezvc_expressive.py \
    --dataset-name Expressive_EZVC \
    --epochs 1 \
    --batch-size-per-gpu 800 \
    --batch-size-type frame \
    --max-samples 1 \
    --num-workers 0 \
    --save-per-updates 50 \
    --last-per-updates 10 \
    --keep-last-n-checkpoints 1 \
    --architecture-mode source-prosody-sidecar \
    --dry-run
```

Remove `--dry-run` to start training.

Real GPU run starting point:

```bash
HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  .venv/bin/accelerate launch --num_processes 1 src/f5_tts/train/finetune_ezvc_expressive.py \
    --dataset-name Expressive_EZVC \
    --epochs 5 \
    --learning-rate 1e-5 \
    --batch-size-per-gpu 2400 \
    --batch-size-type frame \
    --max-samples 16 \
    --grad-accumulation-steps 1 \
    --num-warmup-updates 100 \
    --save-per-updates 1000 \
    --last-per-updates 100 \
    --keep-last-n-checkpoints 2 \
    --architecture-mode source-prosody-sidecar
```

Multi-GPU, low-precision, and gradient accumulation are Accelerate-backed:

```bash
HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  .venv/bin/accelerate launch --num_processes 4 --mixed_precision bf16 \
    src/f5_tts/train/finetune_ezvc_expressive.py \
      --dataset-name Expressive_EZVC \
      --epochs 5 \
      --learning-rate 1e-5 \
      --batch-size-per-gpu 2400 \
      --batch-size-type frame \
      --max-samples 16 \
      --grad-accumulation-steps 2 \
      --mixed-precision bf16 \
      --num-warmup-updates 100 \
      --save-per-updates 1000 \
      --last-per-updates 100 \
      --keep-last-n-checkpoints 2
```

`--num_processes` controls distributed workers, normally one per GPU.
`--mixed_precision` on `accelerate launch` controls the launcher; the script
also accepts `--mixed-precision no|fp16|bf16|fp8` for explicit runs. Leave it
unset to follow your Accelerate config. Effective batch size is approximately:

```text
batch_size_per_gpu * num_processes * grad_accumulation_steps
```

Use `bf16` on GPUs with stable bfloat16 support, `fp16` otherwise, and `no` if
you see numerical instability.

`--architecture-mode unit-only` trains the current EZ-VC decoder from unit
strings only. `--architecture-mode source-prosody-sidecar` validates that
prosody sidecars are present and records the intended future architecture in the
training config, while keeping the current checkpoint shape unchanged.

The script stages the pretrained checkpoint here by default:

```text
ckpts/F5TTS_Base_bigvgan_custom_Expressive_EZVC/pretrained_model_2700000.safetensors
```

The Gradio training UI has the same default as a checkbox:
`Use EZ-VC pretrained checkpoint`. Leave the override box empty to use
`hf://SPRINGLab/EZ-VC/model_2700000.safetensors`, or paste a local path / other
`hf://` URI if you intentionally want a different starting point.

Training checkpoints are written beside it:

```text
ckpts/F5TTS_Base_bigvgan_custom_Expressive_EZVC/model_last.pt
ckpts/F5TTS_Base_bigvgan_custom_Expressive_EZVC/model_<update>.pt
```

#### Labeled expressive dataset hook

The current labeled-mode flag is intentionally metadata-only:

```bash
.venv-py312/bin/python src/f5_tts/train/finetune_ezvc_expressive.py \
  --dataset-name Expressive_EZVC \
  --style-conditioning metadata-only
```

This lets you prepare and preserve labels now without pretending the current
checkpoint can use them. A true labeled expressive model needs one of these
future changes:

- style tokens plus resized text embeddings and careful checkpoint surgery;
- a separate style/emotion embedding branch;
- a source-prosody encoder with F0, energy, and rhythm side channels.

#### What to expect

Good signs:

- source pauses and speaking rate are better preserved when inference uses
  source repeat caps and source-duration matching;
- output has less read-speech flatness;
- dramatic/podcast-style contours become more likely without destroying target
  timbre.

Failure signs:

- source content gets less intelligible: repeat cap may be too high or data too
  noisy;
- target timbre drifts: dataset may be too small or dominated by a few speakers;
- output becomes unstable: lower learning rate, reduce expressive/noisy outliers,
  use `--repeat-cap 1` or `2`, and keep training short.

For the user's goal of source expressiveness plus target timbre, the next real
architecture step is explicit source-prosody conditioning. This fine-tuning path
is the best practical baseline before adding new model branches.

### Training configs

- `src/f5_tts/configs/F5TTS_Base_EZ-VC.yaml`: full/base EZ-VC training config.
  It is sized for serious training hardware, not a 16 GB laptop.
- `src/f5_tts/configs/F5TTS_EZVC_Smoke.yaml`: tiny local smoke config. This is
  only for checking that the trainer, dataset loading, and checkpoint writing
  work. It is not a useful voice conversion model.

The training UI values are optimization/runtime controls, not model qualities
that the network learns. The learned parts are the decoder weights saved in
`model_last.pt` / `model_<update>.pt`. The UI values answer operational questions:
how many epochs to run, how large each batch can be, how fast to update weights,
how much gradient clipping to apply, how long to warm up the learning rate, how
often to save checkpoints, and whether to use fp16/bf16 or a logger.

### Environment

This checkout uses `pyproject.toml` as the dependency source instead of a root
`requirements.txt`.

```bash
cd /Users/rahulb/s2s-vc/ez-vc
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
```

Additional dependency lists only exist inside vendored/subproject folders such
as `src/third_party/BigVGAN/requirements.txt`; the main EZ-VC package
requirements are in `pyproject.toml`.

### Tiny local smoke training

After creating dummy `metadata.csv` + wav files under `tmp/dummy_ezvc_smoke`,
prepare and train with:

```bash
cd /Users/rahulb/s2s-vc/ez-vc

.venv/bin/python src/f5_tts/train/datasets/prepare_csv_wavs.py \
  tmp/dummy_ezvc_smoke \
  data/Dummy_EZVC_Smoke_custom \
  --pretrain \
  --workers 1

HOME=/Users/rahulb/s2s-vc/ez-vc/.home \
MPLCONFIGDIR=/Users/rahulb/s2s-vc/ez-vc/.matplotlib \
NUMBA_CACHE_DIR=/Users/rahulb/s2s-vc/ez-vc/.numba_cache \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  .venv/bin/python src/f5_tts/train/train.py --config-name F5TTS_EZVC_Smoke.yaml
```

The tested smoke run completed 3 updates and wrote:

```text
ckpts/F5TTS_EZVC_Smoke_bigvgan_custom_Dummy_EZVC_Smoke/model_last.pt
```

### Real training guidance

For a real dataset expansion:

1. Collect and clean audio.
2. Run `prepare_ezvc_expressive.py` or another EZ-VC preprocessing step that
   extracts XEUS + k-means units for every audio file.
3. Write the prepared dataset under `data/<Dataset>_custom/`.
4. Copy a config and point `datasets.name` and `model.tokenizer_path` to the
   prepared dataset.
5. Start very small locally, then move full training to GPU hardware.

For 16 GB Apple Silicon experiments, keep these settings conservative:

```yaml
datasets:
  batch_size_per_gpu: 1
  batch_size_type: sample
  max_samples: 1
  num_workers: 0

ckpts:
  logger: null
  log_samples: false
  keep_last_n_checkpoints: 0
```

The trainer has been adjusted so `num_workers: 0` works without persistent
DataLoader workers. Full from-scratch EZ-VC training with the base model is
still not realistic on a 16 GB Mac; use it for smoke tests, small experiments,
or preprocessing validation, then train the real model on a larger GPU machine.

## Acknowledgements

- [F5-TTS](https://arxiv.org/abs/2410.06885) for opensourcing their code which has made EZ-VC possible.

## Citation
If our work and codebase is useful for you, please cite as:
```
@misc{joglekar2025ezvceasyzeroshotanytoany,
      title={EZ-VC: Easy Zero-shot Any-to-Any Voice Conversion}, 
      author={Advait Joglekar and Divyanshu Singh and Rooshil Rohit Bhatia and S. Umesh},
      year={2025},
      eprint={2505.16691},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2505.16691}, 
}
```
## License

Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license. Sorry for any inconvenience this may cause.
