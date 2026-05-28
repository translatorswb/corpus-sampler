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

Supports crash recovery: re-run the same command and it will skip
already-processed files, resuming from where it left off.

Usage:
  python analyze.py /path/to/clips/
  python analyze.py /path/to/clips/ -o results/
  python analyze.py /path/to/clips/ --skip-whisper --skip-embeddings  # ina only
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import argparse
import sys
import time
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
CHECKPOINT_INTERVAL = 50


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
                   help="Output directory (default: <input_dir>_output)")
    p.add_argument("--skip-whisper", action="store_true",
                   help="Skip Whisper language detection")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="Skip speaker embedding extraction and clustering")
    p.add_argument("--whisper-model", default="base",
                   help="Whisper model size: tiny/base/small/medium (default: base)")
    p.add_argument("--en-threshold", type=float, default=0.3,
                   help="Whisper English probability threshold (default: 0.3)")
    p.add_argument("--cluster-method", default="hdbscan",
                   choices=["hdbscan", "agglomerative"],
                   help="Speaker clustering algorithm (default: hdbscan). "
                        "hdbscan auto-detects cluster count and scales to large datasets. "
                        "agglomerative tries all k up to --max-clusters but needs O(n^2) memory.")
    p.add_argument("--n-clusters", type=int, default=None,
                   help="Force number of speaker clusters (agglomerative only, auto if not set)")
    p.add_argument("--max-clusters", type=int, default=30,
                   help="Max clusters to try in auto mode (agglomerative only, default: 30)")
    p.add_argument("--min-cluster-size", type=int, default=3,
                   help="Minimum cluster size (hdbscan only, default: 3)")
    return p.parse_args()


def find_audio_files(input_dir):
    files = []
    for f in sorted(Path(input_dir).rglob("*")):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
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


def _file_key(wav_path, input_dir):
    """Relative path from input_dir, used as the file identifier in CSVs."""
    return os.path.relpath(wav_path, input_dir)


def _load_checkpoint(path):
    """Load a checkpoint CSV and return the set of already-processed file keys."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        return set(df["file"].tolist()), df.to_dict("records")
    return set(), []


def _save_checkpoint(records, path):
    pd.DataFrame(records).to_csv(path, index=False)


def _eta_str(elapsed, done, total):
    if done == 0:
        return "estimating..."
    rate = elapsed / done
    remaining = rate * (total - done)
    if remaining > 3600:
        return f"{remaining/3600:.1f}h remaining"
    return f"{remaining/60:.0f}m remaining"


# ─── Pass 1: inaSpeechSegmenter ─────────────────────────────────────

def run_ina_pass(wav_files, input_dir, checkpoint_path):
    from inaSpeechSegmenter import Segmenter

    done_keys, results = _load_checkpoint(checkpoint_path)
    remaining = [(f, _file_key(f, input_dir)) for f in wav_files
                 if _file_key(f, input_dir) not in done_keys]

    total = len(wav_files)
    n_done = total - len(remaining)
    print(f"\n  Pass 1/3: inaSpeechSegmenter ({n_done}/{total} already done)")
    if not remaining:
        return pd.DataFrame(results)

    print(f"  Loading model...")
    seg = Segmenter()
    t0 = time.time()

    for i, (wav_path, fkey) in enumerate(remaining):
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

        waveform = load_audio(wav_path)
        snr = compute_snr(waveform)

        results.append({
            "file": fkey,
            "duration_sec": round(total_dur, 2),
            "gender": gender,
            "speech_ratio": round(speech_total / total_dur, 2) if total_dur > 0 else 0,
            "music_ratio": round(music_dur / total_dur, 2) if total_dur > 0 else 0,
            "noise_ratio": round(noise_dur / total_dur, 2) if total_dur > 0 else 0,
            "is_speech": bool(speech_total / total_dur > 0.5) if total_dur > 0 else False,
            "is_music": bool(music_dur / total_dur > 0.5) if total_dur > 0 else False,
            "snr_db": round(snr, 1),
        })

        count = n_done + i + 1
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(results, checkpoint_path)
        if (i + 1) % 20 == 0 or i == len(remaining) - 1:
            eta = _eta_str(time.time() - t0, i + 1, len(remaining))
            print(f"    {count}/{total}  ({eta})")

    _save_checkpoint(results, checkpoint_path)
    return pd.DataFrame(results)


# ─── Pass 2: Whisper language detection ──────────────────────────────

def run_whisper_pass(wav_files, input_dir, checkpoint_path, model_size="base", en_threshold=0.3):
    import whisper

    done_keys, results = _load_checkpoint(checkpoint_path)
    remaining = [(f, _file_key(f, input_dir)) for f in wav_files
                 if _file_key(f, input_dir) not in done_keys]

    total = len(wav_files)
    n_done = total - len(remaining)
    print(f"\n  Pass 2/3: Whisper language detection (model={model_size}, en_threshold={en_threshold})")
    print(f"  ({n_done}/{total} already done)")
    if not remaining:
        return pd.DataFrame(results)

    print(f"  Loading model...")
    model = whisper.load_model(model_size)
    t0 = time.time()

    for i, (wav_path, fkey) in enumerate(remaining):
        audio = whisper.load_audio(wav_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)

        top_lang = max(probs, key=probs.get)
        en_prob = probs.get("en", 0)

        results.append({
            "file": fkey,
            "whisper_lang": top_lang,
            "whisper_lang_prob": round(probs[top_lang], 3),
            "whisper_en_prob": round(en_prob, 3),
            "is_english": bool(en_prob > en_threshold),
        })

        count = n_done + i + 1
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(results, checkpoint_path)
        if (i + 1) % 20 == 0 or i == len(remaining) - 1:
            eta = _eta_str(time.time() - t0, i + 1, len(remaining))
            print(f"    {count}/{total}  ({eta})")

    _save_checkpoint(results, checkpoint_path)
    return pd.DataFrame(results)


# ─── Pass 3: Speaker embeddings + clustering ─────────────────────────

def _cluster_hdbscan(X, min_cluster_size=3):
    """Cluster with HDBSCAN — auto-detects cluster count, scales to large datasets."""
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import silhouette_score

    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2)
    labels = hdb.fit_predict(X)
    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()

    # Silhouette on assigned points only
    mask = labels != -1
    if n_clusters > 1 and mask.sum() > n_clusters:
        sil = silhouette_score(X[mask], labels[mask])
    else:
        sil = 0.0

    # Assign noise points to nearest cluster centroid
    if n_noise > 0 and n_clusters > 0:
        centroids = np.array([X[labels == c].mean(axis=0) for c in range(n_clusters)])
        for i in np.where(labels == -1)[0]:
            dists = np.linalg.norm(centroids - X[i], axis=1)
            labels[i] = int(np.argmin(dists))

    print(f"  → {n_clusters} clusters, {n_noise} reassigned from noise (silhouette={sil:.3f})")
    return labels


def _cluster_agglomerative(X, n_clusters=None, max_clusters=30):
    """Cluster with agglomerative — tries all k, picks best silhouette. O(n^2) memory."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

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
    return labels


def run_embedding_pass(wav_files, input_dir, output_dir,
                       cluster_method="hdbscan", n_clusters=None,
                       max_clusters=30, min_cluster_size=3):
    from speechbrain.inference.speaker import EncoderClassifier

    emb_path = os.path.join(output_dir, "embeddings.npy")
    names_path = os.path.join(output_dir, "embedding_files.txt")

    if os.path.exists(emb_path) and os.path.exists(names_path):
        print(f"\n  Pass 3/3: Speaker embeddings — loading from checkpoint")
        X = np.load(emb_path)
        with open(names_path) as f:
            filenames = [line.strip() for line in f]
    else:
        print(f"\n  Pass 3/3: Speaker embeddings (ECAPA-TDNN)")
        print(f"  Loading model...")
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )

        embeddings = []
        filenames = []
        t0 = time.time()
        total = len(wav_files)
        for i, wav_path in enumerate(wav_files):
            waveform = load_audio(wav_path).unsqueeze(0)
            emb = classifier.encode_batch(waveform)
            embeddings.append(emb.squeeze().detach().numpy())
            filenames.append(_file_key(wav_path, input_dir))

            if (i + 1) % 20 == 0 or i == total - 1:
                eta = _eta_str(time.time() - t0, i + 1, total)
                print(f"    {i+1}/{total}  ({eta})")

            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                np.save(emb_path, np.stack(embeddings))
                with open(names_path, "w") as f:
                    f.write("\n".join(filenames))

        X = np.stack(embeddings)
        np.save(emb_path, X)
        with open(names_path, "w") as f:
            f.write("\n".join(filenames))

    # Cluster
    print(f"  Clustering ({X.shape[0]} embeddings, dim={X.shape[1]}, method={cluster_method})...")
    if cluster_method == "hdbscan":
        labels = _cluster_hdbscan(X, min_cluster_size=min_cluster_size)
    else:
        labels = _cluster_agglomerative(X, n_clusters=n_clusters, max_clusters=max_clusters)

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

    output_dir = args.output_dir or (args.input_dir.rstrip("/") + "_output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Input: {args.input_dir} ({len(wav_files)} files)")
    print(f"  Output: {output_dir}/")

    t_start = time.time()

    # Pass 1
    ina_ckpt = os.path.join(output_dir, "_checkpoint_ina.csv")
    df = run_ina_pass(wav_files, args.input_dir, ina_ckpt)

    # Pass 2
    if not args.skip_whisper:
        whisper_ckpt = os.path.join(output_dir, "_checkpoint_whisper.csv")
        df_w = run_whisper_pass(wav_files, args.input_dir, whisper_ckpt,
                                args.whisper_model, args.en_threshold)
        df = df.merge(df_w, on="file")

    # Pass 3
    if not args.skip_embeddings:
        df_spk, embeddings = run_embedding_pass(
            wav_files, args.input_dir, output_dir,
            cluster_method=args.cluster_method,
            n_clusters=args.n_clusters,
            max_clusters=args.max_clusters,
            min_cluster_size=args.min_cluster_size)
        df = df.merge(df_spk, on="file")

    # Save final output
    csv_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - t_start

    # Print summary
    n = len(df)
    total_dur = df["duration_sec"].sum()
    print(f"\n{'='*50}")
    print(f"  {n} clips, {total_dur/60:.1f} min total")
    print(f"  completed in {elapsed/60:.0f}m ({elapsed/n:.1f}s per clip)")
    gc = df["gender"].value_counts()
    for g, c in gc.items():
        print(f"  {g}: {c} ({c/n*100:.0f}%)")
    print(f"  speech: {df['is_speech'].sum()}  music: {df['is_music'].sum()}")
    if "whisper_lang" in df.columns:
        top_langs = df["whisper_lang"].value_counts().head(5)
        langs_str = ", ".join(f"{l}={c}" for l, c in top_langs.items())
        print(f"  top languages: {langs_str}")
    if "is_english" in df.columns:
        print(f"  english: {df['is_english'].sum()}  non-english: {(~df['is_english']).sum()}")
    if "speaker_id" in df.columns:
        sc = df["speaker_id"].value_counts()
        print(f"  speakers: {len(sc)}  largest: {sc.iloc[0]} clips ({sc.iloc[0]/n*100:.0f}%)")
    print(f"  snr range: {df['snr_db'].min():.0f}–{df['snr_db'].max():.0f} dB")
    print(f"  → {csv_path}")
    print(f"{'='*50}")

    # Clean up checkpoint files on successful completion
    for ckpt in [ina_ckpt,
                 os.path.join(output_dir, "_checkpoint_whisper.csv"),
                 os.path.join(output_dir, "embedding_files.txt")]:
        if os.path.exists(ckpt):
            os.remove(ckpt)


if __name__ == "__main__":
    main()
