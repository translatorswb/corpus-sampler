#!/usr/bin/env python3
"""
Analyze a directory of speech clips and produce per-sample metadata.

Runs up to three analysis passes:
  1. inaSpeechSegmenter — speech/music/noise + gender
  2. Whisper — language detection (English vs non-English)
  3. SpeechBrain ECAPA-TDNN — speaker embeddings + clustering

Also computes SNR for each clip.

Input:  directory of audio files (WAV, MP3, etc.)
Output: metadata.csv with one row per clip

Usage:
  python analyze.py /path/to/clips/
  python analyze.py /path/to/clips/ -o results/
  python analyze.py /path/to/clips/ --skip-whisper --skip-embeddings  # ina only
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SAMPLING_RATE = 16000
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".mp4", ".mpeg", ".wma"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze speech clips and produce per-sample metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s /path/to/clips/
  %(prog)s /path/to/clips/ -o results/ --whisper-model small
  %(prog)s /path/to/clips/ --skip-whisper --skip-embeddings
""")
    p.add_argument("input_dir", help="Directory containing audio clips")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory (default: input_dir)")
    p.add_argument("--skip-whisper", action="store_true",
                   help="Skip Whisper language detection")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="Skip speaker embedding extraction and clustering")
    p.add_argument("--whisper-model", default="base",
                   help="Whisper model size: tiny/base/small/medium (default: base)")
    p.add_argument("--en-threshold", type=float, default=0.3,
                   help="Whisper English probability threshold (default: 0.3)")
    p.add_argument("--n-clusters", type=int, default=None,
                   help="Force number of speaker clusters (auto if not set)")
    p.add_argument("--max-clusters", type=int, default=30,
                   help="Maximum clusters to try in auto mode (default: 30)")
    return p.parse_args()


def find_audio_files(input_dir):
    files = []
    for f in sorted(Path(input_dir).iterdir()):
        if f.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(str(f))
    return files


def load_audio(path):
    """Load audio as mono 16kHz waveform."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLING_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(waveform)
    return waveform.squeeze()


def compute_snr(waveform, sr=SAMPLING_RATE, frame_ms=30, top_pct=0.9, bottom_pct=0.1):
    """Estimate SNR by comparing loud vs quiet frames."""
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(waveform) // frame_len
    if n_frames < 2:
        return 0.0
    frames = waveform[:n_frames * frame_len].reshape(n_frames, frame_len)
    energies = (frames ** 2).mean(dim=1)
    energies_sorted = energies.sort().values
    n_top = max(1, int(n_frames * (1 - top_pct)))
    n_bottom = max(1, int(n_frames * bottom_pct))
    signal_energy = energies_sorted[-n_top:].mean()
    noise_energy = energies_sorted[:n_bottom].mean()
    if noise_energy < 1e-10:
        return 40.0  # effectively clean
    return float(10 * torch.log10(signal_energy / noise_energy))


# ─── Pass 1: inaSpeechSegmenter ─────────────────────────────────────

def run_ina_pass(wav_files):
    from inaSpeechSegmenter import Segmenter
    print(f"\n  Pass 1/3: inaSpeechSegmenter")
    print(f"  Loading model...")
    seg = Segmenter()

    results = []
    for i, wav_path in enumerate(wav_files):
        segmentation = seg(wav_path)
        df = pd.DataFrame(segmentation, columns=["label", "start", "stop"])
        df["duration"] = df["stop"] - df["start"]
        total_dur = df["duration"].sum()
        label_durs = df.groupby("label")["duration"].sum()

        male_dur = float(label_durs.get("male", 0))
        female_dur = float(label_durs.get("female", 0))
        speech_dur = float(label_durs.get("speech", 0))
        music_dur = float(label_durs.get("music", 0))
        noise_dur = float(label_durs.get("noise", 0) + label_durs.get("noEnergy", 0))
        speech_total = male_dur + female_dur + speech_dur

        if male_dur > 0 or female_dur > 0:
            gender = "male" if male_dur >= female_dur else "female"
        else:
            gender = "unknown"

        # SNR
        waveform = load_audio(wav_path)
        snr = compute_snr(waveform)

        results.append({
            "file": os.path.basename(wav_path),
            "duration_sec": round(total_dur, 2),
            "gender": gender,
            "speech_ratio": round(speech_total / total_dur, 2) if total_dur > 0 else 0,
            "music_ratio": round(music_dur / total_dur, 2) if total_dur > 0 else 0,
            "noise_ratio": round(noise_dur / total_dur, 2) if total_dur > 0 else 0,
            "is_speech": bool(speech_total / total_dur > 0.5) if total_dur > 0 else False,
            "is_music": bool(music_dur / total_dur > 0.5) if total_dur > 0 else False,
            "snr_db": round(snr, 1),
        })

        if (i + 1) % 20 == 0 or i == len(wav_files) - 1:
            print(f"    {i+1}/{len(wav_files)}")

    return pd.DataFrame(results)


# ─── Pass 2: Whisper language detection ──────────────────────────────

def run_whisper_pass(wav_files, model_size="base", en_threshold=0.3):
    import whisper
    print(f"\n  Pass 2/3: Whisper language detection (model={model_size}, en_threshold={en_threshold})")
    print(f"  Loading model...")
    model = whisper.load_model(model_size)

    results = []
    for i, wav_path in enumerate(wav_files):
        audio = whisper.load_audio(wav_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)

        top_lang = max(probs, key=probs.get)
        en_prob = probs.get("en", 0)

        results.append({
            "file": os.path.basename(wav_path),
            "whisper_lang": top_lang,
            "whisper_lang_prob": round(probs[top_lang], 3),
            "whisper_en_prob": round(en_prob, 3),
            "is_english": bool(en_prob > en_threshold),
        })

        if (i + 1) % 20 == 0 or i == len(wav_files) - 1:
            print(f"    {i+1}/{len(wav_files)}")

    return pd.DataFrame(results)


# ─── Pass 3: Speaker embeddings + clustering ─────────────────────────

def run_embedding_pass(wav_files, n_clusters=None, max_clusters=30):
    from speechbrain.inference.speaker import EncoderClassifier
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    print(f"\n  Pass 3/3: Speaker embeddings (ECAPA-TDNN)")
    print(f"  Loading model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}
    )

    embeddings = []
    filenames = []
    for i, wav_path in enumerate(wav_files):
        waveform = load_audio(wav_path).unsqueeze(0)  # [1, time]
        emb = classifier.encode_batch(waveform)
        embeddings.append(emb.squeeze().detach().numpy())
        filenames.append(os.path.basename(wav_path))

        if (i + 1) % 20 == 0 or i == len(wav_files) - 1:
            print(f"    {i+1}/{len(wav_files)}")

    X = np.stack(embeddings)

    # Cluster
    print(f"  Clustering ({X.shape[0]} embeddings, dim={X.shape[1]})...")
    max_k = min(max_clusters, len(X) // 2)
    if n_clusters:
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)
        k = n_clusters
        sil = silhouette_score(X, labels) if n_clusters > 1 else 0
    else:
        best_k, best_score, best_labels = 2, -1, None
        for k in range(2, max_k + 1):
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels
        labels = best_labels
        k = best_k
        sil = best_score
    print(f"  → {k} clusters (silhouette={sil:.3f})")

    results = pd.DataFrame({
        "file": filenames,
        "speaker_id": [f"spk_{l:03d}" for l in labels],
    })
    return results, X


# ─── Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    wav_files = find_audio_files(args.input_dir)
    if not wav_files:
        print(f"No audio files found in {args.input_dir}")
        sys.exit(1)

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Input: {args.input_dir} ({len(wav_files)} files)")
    print(f"  Output: {output_dir}/")

    # Pass 1
    df = run_ina_pass(wav_files)

    # Pass 2
    if not args.skip_whisper:
        df_w = run_whisper_pass(wav_files, args.whisper_model, args.en_threshold)
        df = df.merge(df_w, on="file")

    # Pass 3
    if not args.skip_embeddings:
        df_spk, embeddings = run_embedding_pass(wav_files, args.n_clusters, args.max_clusters)
        df = df.merge(df_spk, on="file")
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)

    # Save
    csv_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)

    # Print summary
    n = len(df)
    total_dur = df["duration_sec"].sum()
    print(f"\n{'='*50}")
    print(f"  {n} clips, {total_dur/60:.1f} min total")
    gc = df["gender"].value_counts()
    for g, c in gc.items():
        print(f"  {g}: {c} ({c/n*100:.0f}%)")
    print(f"  speech: {df['is_speech'].sum()}  music: {df['is_music'].sum()}")
    if "is_english" in df.columns:
        print(f"  english: {df['is_english'].sum()}  non-english: {(~df['is_english']).sum()}")
    if "speaker_id" in df.columns:
        sc = df["speaker_id"].value_counts()
        print(f"  speakers: {len(sc)}  largest: {sc.iloc[0]} clips ({sc.iloc[0]/n*100:.0f}%)")
    print(f"  snr range: {df['snr_db'].min():.0f}–{df['snr_db'].max():.0f} dB")
    print(f"  → {csv_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
