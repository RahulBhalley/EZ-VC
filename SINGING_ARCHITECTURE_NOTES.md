# Singing Architecture Notes

Date: 2026-05-15

Purpose: future architecture guidance for adapting the EZ-VC-style pipeline to
expressive speech and singing data. This is not an implementation plan for the
current checkpoint; these changes require training or fine-tuning new modules.

## Short Answer

An expressive speech dataset can help EZ-VC-like training, but singing needs
more than expressive data. Singing requires explicit control over melody,
duration, sustained vowels, vibrato, phrasing, and target singer timbre. The
current EZ-VC path compresses audio into XEUS+k-means content units, which is
useful for textless voice conversion but is not enough to preserve or generate
musical pitch accurately.

For singing future work, keep the audio-in/audio-out spirit, but stop treating
the discrete unit stream as the only source condition.

Recommended decomposition:

```text
source singing audio
  -> content encoder: soft units or XEUS/HuBERT/Whisper bottleneck features
  -> melody encoder: continuous F0, voiced/unvoiced, vibrato descriptors
  -> rhythm encoder: frame durations, pauses, note/syllable timing
  -> energy encoder: loudness/energy contour

reference singer audio
  -> singer/timbre encoder

optional score or lyrics
  -> note pitch, note duration, rests, slurs, lyric/phoneme content

decoder
  -> F0-conditioned acoustic model or VITS/flow/diffusion decoder
  -> singing-capable vocoder/post-processor
```

## Why The Current Unit Path Is Not Enough

EZ-VC combines discrete speech representations from self-supervised models with
a non-autoregressive Diffusion Transformer conditional flow-matching decoder in
a textless, self-supervised VC setup. That is a good design for zero-shot,
cross-lingual speech conversion, but it deliberately compresses the source.

The singing-specific failure point is that exact F0, sustained note duration,
vibrato, and phrase-level dynamics are not guaranteed to survive k-means unit
quantization. De-duplicating adjacent units makes this worse because repeated
frames are exactly where sustained vowels and note lengths live.

## Recommended Future Architecture

### 1. Replace hard units with soft content features where possible

Keep content separate from singer identity, but avoid relying only on hard
k-means IDs. A useful direction is soft speech units or bottleneck features:

- hard units are strong for speaker removal, but can discard content and
  pronunciation detail;
- soft units preserve uncertainty and tend to improve naturalness and
  intelligibility;
- singing should benefit from not forcing all frames into one hard cluster ID.

Practical options:

- XEUS hidden states before k-means, optionally projected by a trainable content
  adapter;
- HuBERT soft units;
- Whisper bottleneck features for robust linguistic content, especially if the
  dataset is multilingual or noisy.

Do not remove the content bottleneck entirely. Otherwise source singer timbre
will leak into the target singer.

### 2. Add explicit F0 and voicing conditioning

This is the highest-priority singing change.

Add a melody branch:

```text
source audio -> F0 extractor -> log-F0 + voiced/unvoiced mask -> melody encoder
```

Condition the acoustic decoder on the F0 contour at frame level. For conversion,
use the source F0 as the default. For synthesis from score, use note pitch plus
a learned F0 predictor or diffusion pitch model.

Recommended details:

- normalize F0 by singer range for training stability;
- allow transposition at inference;
- keep unvoiced/breath/noise regions separate from pitch;
- model vibrato as local F0 modulation, not as random decoder noise;
- use multi-scale F0 features if the decoder struggles with pitch variation.

### 3. Add explicit duration/rhythm conditioning

Singing timing is not speech timing. A model needs to know how long vowels,
notes, rests, and syllables should last.

For voice conversion:

- preserve source frame duration;
- avoid unit de-duplication for singing;
- optionally cap repeated units, but keep a separate duration signal.

For score/lyrics synthesis:

- condition on note duration and note type;
- add syllable-level or note-level duration losses;
- predict phoneme-to-note duration ratios when phoneme alignment exists;
- support rests, slurs, grace notes, and long held vowels if the dataset has
  those annotations.

### 4. Keep singer/timbre conditioning separate from melody

Reference audio should identify the target singer. Source audio should provide
content, melody, rhythm, and expression. Do not use one embedding to carry all
of this.

Use:

```text
reference audio -> singer encoder -> global timbre embedding
source audio -> content + F0 + rhythm + energy
```

Add anti-leakage measures:

- pitch perturbation before content-feature extraction;
- speaker adversarial loss on content features;
- target-singer classification or speaker similarity loss on generated audio;
- source-singer leakage evaluation.

### 5. Use a singing-aware decoder and vocoder path

Two viable decoder families:

- VITS-style SVC decoder: strong baseline for singing voice conversion because
  many SVC systems already use content features plus F0 plus singer embedding.
- Flow/diffusion decoder: closer to F5-TTS/EZ-VC and likely easier to integrate
  conceptually, but it should be explicitly conditioned on frame-level F0,
  duration, energy, and singer embedding.

For the vocoder, prefer a pitch-aware or singing-proven vocoder/post-processor
over a generic speech vocoder when singing quality matters. Singing exposes
pitch errors, phase artifacts, and high-frequency harshness more aggressively
than normal speech.

## Dataset Requirements

Minimum useful singing dataset fields:

```text
audio_path
singer_id
language_id
isolated_vocal_path
source_lyrics_or_units
f0_path
uv_path
energy_path
duration_or_alignment_path
```

Better if available:

```text
lyrics
phoneme_sequence
phoneme_durations
note_pitch
note_duration
note_type
rests
slurs
breath/noise labels
style labels
```

Preprocessing rules:

- use isolated vocals, not full mixes;
- remove accompaniment leakage where possible;
- segment by phrase, not arbitrary short chunks;
- keep long sustained vowels;
- do not collapse repeated units for singing experiments;
- extract F0 with a singing-robust method and inspect errors manually;
- store F0, UV, energy, and durations as first-class training artifacts.

## Suggested Experiment Stages

### Stage 0: diagnose current bottleneck

Use the current UI controls:

- source repeat cap: 0, 2, 3;
- target repeat cap: 1;
- match source duration enabled;
- CFG around 1.0 to 2.0.

This only tests whether any useful timing signal survived the unit path.

### Stage 1: add side-channel conditioning without changing content units

Keep XEUS+k-means units, but train a new decoder with:

- source F0 contour;
- source energy contour;
- explicit generated duration;
- reference singer embedding.

This is the smallest architecture change that directly targets singing.

### Stage 2: switch from hard units to soft/bottleneck content

Replace the hard unit string with frame-level soft content features, then train
the decoder to use:

```text
soft content + F0 + UV + energy + singer embedding -> mel/waveform
```

This should reduce pronunciation loss and improve naturalness.

### Stage 3: add score-aware control

If lyrics or MIDI/music-score data is available, add:

```text
lyrics/phonemes + note pitch + note duration + note type -> acoustic model
```

This turns the project from pure voice conversion into singing voice synthesis.
It is more work, but it gives direct control over melody and rhythm.

## What Singing Data Would Teach The Model

With only the current EZ-VC-style unit path:

- likely learns singer-like timbre and broad sustained-vowel patterns;
- may learn some singing-like phrasing;
- will not reliably follow melody;
- may flatten or hallucinate pitch;
- may trade intelligibility for musicality.

With F0/duration/singer conditioning:

- can preserve source melody during voice conversion;
- can learn target singer timbre while keeping source pitch;
- can handle sustained vowels and vibrato better;
- can support transposition and melody editing;
- can become a real SVC model rather than speech VC exposed to singing data.

With score/lyrics conditioning:

- can synthesize singing from notes and lyrics;
- can learn note-level rhythm, rests, slurs, and pitch transitions;
- can support controllable singing rather than only conversion.

## Paper Notes And Links

- EZ-VC: textless voice conversion using discrete speech representations plus a
  Diffusion Transformer conditional flow-matching decoder.
  https://arxiv.org/abs/2505.16691

- F5-TTS: non-autoregressive flow-matching TTS with DiT. Useful decoder lineage,
  but singing needs additional pitch/duration conditioning.
  https://arxiv.org/abs/2410.06885

- Soft speech units for VC: hard discrete units can discard information; soft
  units improve intelligibility and naturalness.
  https://arxiv.org/abs/2111.02392

- XiaoiceSing: adds musical score features, F0 modeling, and duration modeling
  to a FastSpeech-style singing system.
  https://www.microsoft.com/en-us/research/publication/xiaoicesing-a-high-quality-and-integrated-singing-voice-synthesis-system/

- DiffSinger: diffusion acoustic model conditioned on music score for realistic
  mel-spectrogram generation.
  https://arxiv.org/abs/2105.02446

- VISinger: VITS-like end-to-end SVS with frame-level prior, F0 predictor, and
  note-aware duration modeling.
  https://arxiv.org/abs/2110.08813

- RMSSinger: realistic score-based SVS with word-level modeling and
  diffusion-based pitch modeling.
  https://aclanthology.org/2023.findings-acl.16/

- VITS-based SVC with DSPGAN post-processing: uses HuBERT content plus F0
  contours, then recomposes target timbre, F0, and content.
  https://arxiv.org/abs/2310.05118

- VITS-based SVC with Whisper and multi-scale F0: uses Whisper bottleneck
  features, pitch perturbation to reduce timbre leakage, and multi-scale F0.
  https://arxiv.org/abs/2310.02802

## Decision

For this codebase, the most practical future direction is:

```text
EZ-VC content frontend
+ source F0/UV branch
+ source rhythm/duration branch
+ source energy branch
+ reference singer encoder
+ F0-conditioned flow/VITS decoder
```

Do not start by training on singing data with the current unit-only decoder and
expect good singing. Use singing data only after the conditioning path can carry
melody and duration.
