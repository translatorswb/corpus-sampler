# select-corpus

Two-step tool for building balanced speech corpora from pre-segmented audio clips.

## What this does

Takes a directory of short speech clips (from any source — radio segmentation, field recordings, HuggingFace datasets) and:

1. **Analyzes** them (`analyze.py`): gender, speech/music/noise, English detection, speaker clustering, SNR
2. **Selects** a balanced subset (`select_subset.py`): filters bad clips, caps dominant speakers, boosts female representation

## Architecture

```
clips/ → analyze.py → metadata.csv → select_subset.py → selected.csv + report.md
```

- `analyze.py` — Three analysis passes: inaSpeechSegmenter (gender + speech/music), Whisper (English detection), SpeechBrain ECAPA-TDNN (speaker embeddings + clustering). Also computes SNR. Outputs `metadata.csv` + `embeddings.npy`.
- `select_subset.py` — Reads metadata, applies filters (music, English, non-speech, SNR, duration), then selects a diverse subset respecting speaker caps and female ratio targets. Outputs `selected.csv` + `selected_report.md`.

## Current project: Rohingya VOA corpus (R2)

The immediate use case is selecting samples from the VOA Rohingya dataset for transcription as part of the Kobo/UNHCR/GMU Rohingya ASR project.

- Source dataset: [freococo/rohingya_asr_audio](https://huggingface.co/datasets/freococo/rohingya_asr_audio) on HuggingFace
- 100-sample test set in `rohingya_samples/` with manual review notes in `rohingya_samples_notes.txt`
- Analysis output from the 100 samples in `rohingya_samples_output/`
- Full project context is in `/Users/alp/Professional/CLEAR/play/KOBO-UNHCR-Rohingya-Kurdish/`

### Known issues from the 100-sample test

- Language ID: Whisper doesn't know Rohingya, detects it as Bengali (bn), Farsi (fa), Nepali (ne), etc. We only use English/non-English distinction. Threshold is 0.3 (catches most English but missed sample_025 which has brief English in mostly Rohingya).
- Gender: inaSpeechSegmenter found 15 female (vs 3 in quick manual listen) — probably correct, manual review was partial.
- Music detection missed sample_002 (faint music under voice). SNR filtering may help catch borderline cases.
- Speaker clustering found 15 clusters with silhouette=0.223 (modest). Largest cluster = 21 samples, all male — likely the dominant VOA presenter.

### Next steps

- Run the full pipeline on the complete HuggingFace dataset (will need GPU for speaker embeddings at scale)
- Tune English detection threshold and validate against more manual listening
- Generate a recommended subset of 3K-10K samples for transcription upload to TWB Voice

## Tech notes

- The `.venv` is a symlink to `../segment-found-audio/.venv` to avoid numpy version conflicts with inaSpeechSegmenter. If setting up fresh, pin `numpy<2`.
- Speaker embedding extraction is CPU-intensive (~3 min for 100 clips). Budget GPU time for the full dataset.
- The tool is language-agnostic. The only language-specific logic is the English detection threshold in Whisper.

## Related

- `../segment-found-audio/` — upstream tool that segments long audio into clips (not needed when clips are pre-segmented)
- `/Users/alp/Professional/CLEAR/play/KOBO-UNHCR-Rohingya-Kurdish/` — the broader Kobo/UNHCR project files (workplan, contracts, ToR, transcription guidelines)
