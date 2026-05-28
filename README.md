# select-corpus

Analyze a directory of speech clips and select a balanced subset for transcription.

Designed for building diverse speech corpora from found audio (radio, podcasts, field recordings). Takes pre-segmented clips, produces per-sample metadata (gender, language, speaker ID, audio quality), then selects a subset that maximizes speaker and gender diversity.

## How it works

**`analyze.py`** — Runs three analysis passes on each clip:

1. **[inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter)** — classifies speech/music/noise, tags speaker gender
2. **[Whisper](https://github.com/openai/whisper)** — detects language (English vs non-English)
3. **[SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)** — extracts speaker embeddings, clusters into speaker groups

Also computes SNR (signal-to-noise ratio) for each clip.

**`select_subset.py`** — Reads the metadata and selects a balanced subset:

- Filters out music, non-speech, English, low-SNR clips
- Caps samples per speaker to reduce dominant-speaker bias
- Prioritizes female samples to improve gender balance
- All parameters configurable via CLI

## Installation

Requires Python 3.10+ and ffmpeg.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Step 1: Analyze

```bash
python analyze.py /path/to/clips/
python analyze.py /path/to/clips/ -o results/
python analyze.py /path/to/clips/ --skip-whisper    # faster, no language detection
python analyze.py /path/to/clips/ --skip-embeddings  # faster, no speaker clustering
```

Produces `metadata.csv` in the output directory.

### Step 2: Select

```bash
python select_subset.py metadata.csv
python select_subset.py metadata.csv -n 3000 --max-per-speaker 100 --min-female 0.3
python select_subset.py metadata.csv -n 5000 --max-per-speaker-pct 0.1
```

Produces `selected.csv` — a filtered, balanced subset of the input.

### Analyze options

```
--skip-whisper          Skip language detection (faster)
--skip-embeddings       Skip speaker clustering (faster)
--whisper-model SIZE    Whisper model: tiny/base/small/medium (default: base)
--en-threshold FLOAT    English probability threshold (default: 0.3)
--n-clusters INT        Force number of speaker clusters (auto if not set)
--max-clusters INT      Max clusters to try in auto mode (default: 30)
```

### Select options

```
-n, --target INT          Target number of samples
--max-per-speaker INT     Max samples from any single speaker
--max-per-speaker-pct F   Max per speaker as fraction of target (e.g. 0.1)
--min-female FLOAT        Target minimum female ratio (0-1)
--include-music           Don't exclude music clips
--include-english         Don't exclude English clips
--include-nonspeech       Don't exclude non-speech clips
--min-snr FLOAT           Minimum SNR in dB
--min-duration FLOAT      Minimum clip duration in seconds
--max-duration FLOAT      Maximum clip duration in seconds
--seed INT                Random seed (default: 42)
```

## Metadata format

| Field | Source | Description |
|---|---|---|
| `file` | — | Filename |
| `duration_sec` | ina | Clip duration |
| `gender` | ina | `male` / `female` / `unknown` |
| `speech_ratio` | ina | Fraction of clip that is speech |
| `music_ratio` | ina | Fraction of clip that is music |
| `is_speech` | ina | Speech > 50% of clip |
| `is_music` | ina | Music > 50% of clip |
| `snr_db` | signal | Estimated signal-to-noise ratio |
| `whisper_lang` | whisper | Top detected language code |
| `whisper_en_prob` | whisper | English probability |
| `is_english` | whisper | English prob > threshold |
| `speaker_id` | ecapa | Speaker cluster label |

## Works with segment-found-audio

This tool is designed to work as a second step after [segment-found-audio](../segment-found-audio/), but also works on any directory of pre-segmented audio clips from other sources.

```
Raw audio → [segment-found-audio] → clips/ → [select-corpus] → metadata.csv + selected.csv
                                      ↑
              Or: pre-segmented clips from HuggingFace, field recordings, etc.
```

## License

MIT
