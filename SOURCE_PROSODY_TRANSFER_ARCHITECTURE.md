# Source Prosody Transfer Architecture

Date: 2026-05-15

Purpose: adapt the singing-model architecture idea to the actual EZ-VC target
application: transfer target-speaker timbre while retaining the source speech in
every other sense, including language, content, timing, pauses, emotion,
intonation, energy, and cross-lingual behavior.

## Position

Yes, the singing architecture plan is a good fit for this broader voice
conversion goal. The key idea is not "singing" itself. The key idea is explicit
source-side prosody conditioning.

Current EZ-VC mostly does this:

```text
source audio -> XEUS -> k-means units -> unit string
target audio -> XEUS -> k-means units -> prompt units
target audio + target units + source units -> decoder -> target-timbre output
```

That is useful for cross-lingual content transfer, but it compresses source
speech too aggressively. If the source has anger, hesitation, sarcasm, breathy
delivery, rising intonation, fast/slow rhythm, or dramatic pauses, some of that
can be lost before the decoder sees it.

The desired architecture should do this instead:

```text
source audio
  -> content units/features
  -> F0 / intonation contour
  -> voiced/unvoiced mask
  -> energy/loudness contour
  -> rhythm/duration/pauses
  -> optional style embedding

target/reference audio
  -> timbre/speaker embedding

decoder
  -> generated speech in target timbre
  -> preserves source content, language, timing, and expression
```

## Design Rule

Source audio owns:

- linguistic content;
- language/accent realization when doing audio-driven conversion;
- timing and pauses;
- pitch contour / intonation;
- energy contour;
- emotion and delivery;
- singing melody, if the source is singing.

Target/reference audio owns:

- speaker identity;
- timbre;
- vocal texture;
- channel/acoustic reference only if explicitly desired.

These roles should not be merged into one generic embedding. If they are merged,
the model will either leak source speaker identity or overwrite source
expressiveness with target-reference style.

## Recommended Architecture

### 1. Content path

Keep an audio-derived content path so the system remains transcript-free and
cross-lingual:

```text
source audio -> XEUS/HuBERT/Whisper content features -> content adapter
```

The current hard k-means unit string is the simplest version. A stronger future
version should support soft/bottleneck features because they preserve more
information than hard units.

Recommended stages:

1. Keep existing EZ-VC units for compatibility.
2. Add side-channel prosody conditioning.
3. Replace or augment hard units with soft content features once the side
   channels are working.

### 2. Source prosody path

Add frame-level source prosody features:

```text
source audio -> F0, log-F0, voiced mask, energy -> prosody encoder
```

The current implementation now supports a data hook:

```bash
--source-prosody-mode sidecar
```

This extracts `.npz` files containing:

- `f0_hz`
- `log_f0`
- `voiced`
- `voiced_prob`
- `energy`
- `sample_rate`
- `hop_length`
- `frame_length`

These are sidecars for the future architecture. The current EZ-VC decoder does
not consume them yet.

### 3. Rhythm and duration path

Duration should be explicit, not inferred only from unit-string length.

Useful signals:

- source frame count;
- pause mask;
- voiced/unvoiced runs;
- source syllable or unit duration, if available;
- output duration override at inference.

The current UI's source repeat cap and source-duration matching are a small
inference-time proxy for this. The future model should consume duration/rhythm
features directly.

### 4. Target timbre path

Add a target speaker/timbre encoder independent of source prosody:

```text
target/reference audio -> speaker encoder -> timbre embedding
```

Training should discourage this embedding from carrying target-reference
prosody. The target reference should answer "who is speaking?", not "how should
the sentence be performed?"

### 5. Decoder conditioning

The decoder should condition on:

```text
content features + source prosody features + source duration/rhythm + target timbre
```

Two practical decoder directions:

- Continue from the F5-TTS/EZ-VC flow decoder and add frame-level prosody
  conditioning adapters.
- Build a VITS/SVC-style decoder that already expects content + F0 + speaker
  embedding.

The first path is less disruptive to this repo. The second path may be more
natural for singing or very pitch-sensitive transfer.

## Training Objective

For a paired/self-reconstruction phase:

```text
source audio == target audio
content(source) + prosody(source) + timbre(source) -> reconstruct source
```

For timbre-transfer training:

```text
content(source) + prosody(source) + timbre(reference) -> source performance in reference timbre
```

If true parallel target-timbre data is unavailable, use consistency losses:

- mel/audio reconstruction for self pairs;
- speaker similarity to reference;
- content consistency to source;
- F0/prosody consistency to source;
- anti-leakage loss so content/prosody branches do not carry speaker identity.

## Current Implementation Flag

This repo now has a practical staging flag:

```bash
.venv-py312/bin/python src/f5_tts/train/datasets/prepare_ezvc_expressive.py \
  --dataset-name Expressive_EZVC \
  --audio-dir /path/to/clips \
  --source-prosody-mode sidecar \
  --repeat-cap 2
```

It writes:

```text
data/Expressive_EZVC_custom/prosody/*.npz
data/Expressive_EZVC_custom/prosody_manifest.jsonl
```

The fine-tuning launcher has a matching architecture flag:

```bash
.venv-py312/bin/python src/f5_tts/train/finetune_ezvc_expressive.py \
  --dataset-name Expressive_EZVC \
  --architecture-mode source-prosody-sidecar \
  --dry-run
```

Today this validates and records the source-prosody sidecar contract while
keeping the checkpoint shape unchanged. The next implementation step is to add a
model branch that consumes those sidecars.

## Implementation Roadmap

### Phase 1: sidecar data contract

Status: implemented.

- Extract EZ-VC units.
- Extract F0/UV/energy sidecars.
- Preserve optional style labels.
- Validate sidecars from the fine-tuning launcher.

### Phase 2: dataset loader support

Add dataset/collate support for:

```python
prosody = {
    "log_f0": Tensor[T],
    "voiced": Tensor[T],
    "energy": Tensor[T],
}
```

Pad prosody to mel length in `collate_fn`.

### Phase 3: source prosody encoder

Add a small temporal encoder:

```text
[log_f0, voiced, energy] -> Conv/Transformer prosody embedding
```

Project it to the decoder hidden dimension and inject it with either:

- additive frame conditioning;
- cross-attention;
- FiLM/adaptive layer norm.

### Phase 4: timbre/content disentanglement

Add speaker leakage checks:

- source speaker classifier should fail on content/prosody embeddings;
- generated output should match target speaker embedding;
- generated output should preserve source F0/energy/rhythm.

### Phase 5: inference UI

Expose controls:

- preserve source F0 strength;
- preserve source energy strength;
- preserve source duration strength;
- transpose F0;
- target-timbre strength;
- source-prosody debug plots.

## Expected Outcome

With only unit fine-tuning:

- some expressiveness may improve;
- source timing may improve with repeat caps;
- exact emotion/F0/energy transfer remains unreliable.

With the source-prosody architecture:

- target timbre should come from reference audio;
- source content and language should remain audio-driven;
- source pauses, rhythm, pitch contour, and energy should transfer more
  directly;
- cross-lingual behavior should be retained because content remains
  self-supervised/audio-derived rather than transcript-bound.

## Non-Goals

This architecture is not the same as text-to-speech with emotion labels. Labels
can help later, but the primary goal is source-performance transfer: the source
audio itself should tell the model how the line is performed.
