#!/usr/bin/env python3
"""
Select a balanced subset from analyzed speech clips.

Reads metadata.csv (produced by analyze.py) and selects samples that
maximize speaker and gender diversity while filtering out unusable clips.

Input:  metadata.csv
Output: selected.csv (subset) + selection report to stdout

Usage:
  python select.py metadata.csv
  python select.py metadata.csv -n 3000 --min-female 0.3
  python select.py metadata.csv --max-per-speaker 50 --min-snr 15
"""

import argparse
import sys

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Select a balanced subset from analyzed speech metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s metadata.csv
  %(prog)s metadata.csv -n 5000 --min-female 0.4
  %(prog)s metadata.csv -n 3000 --max-per-speaker 100 --exclude-langs en,bn
""")
    p.add_argument("metadata", help="Path to metadata.csv from analyze.py")
    p.add_argument("-o", "--output", default=None,
                   help="Output CSV path (default: selected.csv in same dir)")
    p.add_argument("-n", "--target", type=int, default=None,
                   help="Target number of samples (default: all that pass filters)")
    p.add_argument("--max-per-speaker", type=int, default=None,
                   help="Max samples from any single speaker. "
                        "Use as absolute count or as percentage with %%: e.g. 10%%")
    p.add_argument("--max-per-speaker-pct", type=float, default=None,
                   help="Max samples per speaker as fraction of target (e.g. 0.1 = 10%%)")
    p.add_argument("--min-female", type=float, default=0.0,
                   help="Target minimum female ratio (0-1). Will include all available "
                        "female samples and report achieved ratio. (default: 0, no target)")
    p.add_argument("--exclude-music", action="store_true", default=True,
                   help="Exclude music-dominant clips (default: yes)")
    p.add_argument("--include-music", action="store_true",
                   help="Include music-dominant clips")
    p.add_argument("--exclude-langs", type=str, default="en",
                   help="Comma-separated language codes to exclude (default: en). "
                        "Set to empty string to disable: --exclude-langs ''")
    p.add_argument("--exclude-nonspeech", action="store_true", default=True,
                   help="Exclude non-speech clips (default: yes)")
    p.add_argument("--include-nonspeech", action="store_true",
                   help="Include non-speech clips")
    p.add_argument("--min-snr", type=float, default=0,
                   help="Minimum SNR in dB (default: 0, no filter)")
    p.add_argument("--min-duration", type=float, default=0,
                   help="Minimum clip duration in seconds (default: 0)")
    p.add_argument("--max-duration", type=float, default=999,
                   help="Maximum clip duration in seconds (default: no limit)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.metadata)
    n_total = len(df)
    print(f"  Loaded {n_total} clips from {args.metadata}")

    # ── Filtering ────────────────────────────────────────────────
    mask = pd.Series(True, index=df.index)

    # Music
    if not args.include_music and "is_music" in df.columns:
        music_mask = df["is_music"] == True
        mask &= ~music_mask
        print(f"  Excluded {music_mask.sum()} music clips")

    # Non-speech
    if not args.include_nonspeech and "is_speech" in df.columns:
        nonspeech_mask = df["is_speech"] == False
        # Don't double-count with music
        new_exclusions = nonspeech_mask & mask & ~(df.get("is_music", False) == True)
        mask &= df["is_speech"] == True
        print(f"  Excluded {(~df['is_speech'] & ~df.get('is_music', False)).sum()} non-speech clips")

    # Language exclusion
    exclude_langs = [l.strip() for l in args.exclude_langs.split(",") if l.strip()]
    if exclude_langs and "whisper_lang" in df.columns:
        lang_mask = df["whisper_lang"].isin(exclude_langs)
        mask &= ~lang_mask
        print(f"  Excluded {lang_mask.sum()} clips by language ({', '.join(exclude_langs)})")

    # SNR
    if args.min_snr > 0 and "snr_db" in df.columns:
        snr_mask = df["snr_db"] < args.min_snr
        mask &= ~snr_mask
        print(f"  Excluded {snr_mask.sum()} low-SNR clips (<{args.min_snr} dB)")

    # Duration
    if args.min_duration > 0:
        dur_mask = df["duration_sec"] < args.min_duration
        mask &= ~dur_mask
        print(f"  Excluded {dur_mask.sum()} short clips (<{args.min_duration}s)")
    if args.max_duration < 999:
        dur_mask = df["duration_sec"] > args.max_duration
        mask &= ~dur_mask
        print(f"  Excluded {dur_mask.sum()} long clips (>{args.max_duration}s)")

    pool = df[mask].copy()
    print(f"  After filtering: {len(pool)} clips available")

    if len(pool) == 0:
        print("  No clips remain after filtering. Adjust thresholds.")
        sys.exit(1)

    # ── Selection ────────────────────────────────────────────────
    target = args.target or len(pool)

    # Resolve max-per-speaker
    max_per_speaker = None
    if args.max_per_speaker is not None:
        max_per_speaker = args.max_per_speaker
    elif args.max_per_speaker_pct is not None:
        max_per_speaker = max(1, int(target * args.max_per_speaker_pct))

    has_speakers = "speaker_id" in pool.columns

    if has_speakers and max_per_speaker:
        print(f"  Max per speaker: {max_per_speaker}")

    # Strategy: prioritize female samples, then fill with speaker-diverse males
    selected_indices = []
    speaker_counts = {}  # track across both phases

    if args.min_female > 0 and "gender" in pool.columns:
        female_pool = pool[pool["gender"] == "female"]
        n_female_target = int(target * args.min_female)
        # Take all female samples (they're likely scarce)
        if has_speakers and max_per_speaker:
            female_selected = _diverse_sample(female_pool, len(female_pool), max_per_speaker, args.seed)
        else:
            female_selected = female_pool
        selected_indices.extend(female_selected.index.tolist())
        # Track speaker counts from female phase
        if has_speakers:
            for _, row in female_selected.iterrows():
                spk = row["speaker_id"]
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
        print(f"  Female: {len(female_selected)} selected (target was {n_female_target})")

    # Fill remaining with speaker-diverse sampling from the rest
    remaining_pool = pool.drop(index=selected_indices, errors="ignore")
    n_remaining = target - len(selected_indices)

    if n_remaining > 0 and len(remaining_pool) > 0:
        if has_speakers and max_per_speaker:
            rest = _diverse_sample(remaining_pool, n_remaining, max_per_speaker, args.seed,
                                   existing_counts=speaker_counts)
        elif n_remaining < len(remaining_pool):
            rest = remaining_pool.sample(n=n_remaining, random_state=args.seed)
        else:
            rest = remaining_pool
        selected_indices.extend(rest.index.tolist())

    selected = df.loc[selected_indices].copy()

    # ── Output ───────────────────────────────────────────────────
    output_path = args.output
    if not output_path:
        from pathlib import Path
        output_path = str(Path(args.metadata).parent / "selected.csv")

    selected.to_csv(output_path, index=False)

    # Console summary
    n_sel = len(selected)
    dur = selected["duration_sec"].sum()
    print(f"\n{'='*50}")
    print(f"  SELECTED: {n_sel} clips ({dur/60:.1f} min)")
    print(f"  from {n_total} total → {n_sel/n_total*100:.0f}% kept")

    if "gender" in selected.columns:
        gc = selected["gender"].value_counts()
        print(f"\n  Gender:")
        for g, c in gc.items():
            print(f"    {g}: {c} ({c/n_sel*100:.0f}%)")

    if has_speakers:
        sc = selected["speaker_id"].value_counts()
        print(f"\n  Speakers: {len(sc)}")
        print(f"    largest: {sc.index[0]} = {sc.iloc[0]} clips ({sc.iloc[0]/n_sel*100:.0f}%)")
        print(f"    smallest: {sc.index[-1]} = {sc.iloc[-1]} clips")
        if len(sc) <= 20:
            print(f"    distribution: {dict(sc)}")

    # Write markdown report
    report_path = output_path.replace(".csv", "_report.md")
    _write_report(df, selected, args, report_path)

    print(f"\n  → {output_path}")
    print(f"  → {report_path}")
    print(f"{'='*50}")


def _dataset_stats(df, label):
    """Generate a stats block for a dataset."""
    n = len(df)
    dur = df["duration_sec"].sum()
    lines = []
    lines.append(f"### {label}")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Samples | {n} |")
    lines.append(f"| Total duration | {dur/60:.1f} min ({dur/3600:.2f} h) |")
    lines.append(f"| Mean duration | {df['duration_sec'].mean():.1f}s |")
    lines.append(f"| Duration range | {df['duration_sec'].min():.1f}s – {df['duration_sec'].max():.1f}s |")

    if "snr_db" in df.columns:
        lines.append(f"| SNR range | {df['snr_db'].min():.0f} – {df['snr_db'].max():.0f} dB |")
        lines.append(f"| Mean SNR | {df['snr_db'].mean():.0f} dB |")

    lines.append("")

    # Gender
    if "gender" in df.columns:
        lines.append("**Gender distribution:**")
        lines.append("")
        lines.append("| Gender | Count | % |")
        lines.append("|--------|-------|---|")
        for g, c in df["gender"].value_counts().items():
            lines.append(f"| {g} | {c} | {c/n*100:.0f}% |")
        lines.append("")

    # Speech/music classification
    if "is_speech" in df.columns:
        lines.append("**Content classification:**")
        lines.append("")
        lines.append("| Category | Count | % |")
        lines.append("|----------|-------|---|")
        speech_n = int(df["is_speech"].sum())
        lines.append(f"| Usable speech | {speech_n} | {speech_n/n*100:.0f}% |")
        if "is_music" in df.columns:
            music_n = int(df["is_music"].sum())
            lines.append(f"| Music/jingles | {music_n} | {music_n/n*100:.0f}% |")
        nonspeech = n - speech_n
        lines.append(f"| Non-speech total | {nonspeech} | {nonspeech/n*100:.0f}% |")
        lines.append("")

    # Speakers
    if "speaker_id" in df.columns:
        sc = df["speaker_id"].value_counts()
        lines.append("**Speaker distribution:**")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Distinct speakers | {len(sc)} |")
        lines.append(f"| Largest cluster | {sc.index[0]} ({sc.iloc[0]} clips, {sc.iloc[0]/n*100:.0f}%) |")
        lines.append(f"| Smallest cluster | {sc.index[-1]} ({sc.iloc[-1]} clips) |")
        lines.append(f"| Median cluster size | {sc.median():.0f} clips |")
        lines.append("")

        # Top speakers table
        top_n = min(10, len(sc))
        lines.append(f"Top {top_n} speakers:")
        lines.append("")
        lines.append("| Speaker | Clips | % | Gender |")
        lines.append("|---------|-------|---|--------|")
        for spk in sc.index[:top_n]:
            spk_df = df[df["speaker_id"] == spk]
            count = len(spk_df)
            if "gender" in df.columns:
                genders = spk_df["gender"].value_counts()
                gender_str = ", ".join(f"{g}={c}" for g, c in genders.items())
            else:
                gender_str = "—"
            lines.append(f"| {spk} | {count} | {count/n*100:.0f}% | {gender_str} |")
        lines.append("")

    # Language breakdown (if whisper data present)
    if "whisper_lang" in df.columns:
        lines.append("**Detected languages (Whisper):**")
        lines.append("")
        lines.append("| Language | Count | % |")
        lines.append("|----------|-------|---|")
        for lang, c in df["whisper_lang"].value_counts().head(10).items():
            lines.append(f"| {lang} | {c} | {c/n*100:.0f}% |")
        lines.append("")

    return "\n".join(lines)


def _write_report(df_full, df_selected, args, report_path):
    """Write a markdown report comparing input dataset and selected subset."""
    lines = []
    lines.append("# Corpus Analysis and Selection Report")
    lines.append("")
    lines.append(f"Generated from `{args.metadata}`")
    lines.append("")

    # Selection parameters
    lines.append("## Selection parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Target samples | {args.target or 'all passing filters'} |")
    if args.max_per_speaker:
        lines.append(f"| Max per speaker | {args.max_per_speaker} |")
    elif args.max_per_speaker_pct:
        lines.append(f"| Max per speaker | {args.max_per_speaker_pct*100:.0f}% of target |")
    if args.min_female > 0:
        lines.append(f"| Min female ratio | {args.min_female} |")
    lines.append(f"| Exclude music | {not args.include_music} |")
    lines.append(f"| Exclude languages | {args.exclude_langs or 'none'} |")
    lines.append(f"| Exclude non-speech | {not args.include_nonspeech} |")
    if args.min_snr > 0:
        lines.append(f"| Min SNR | {args.min_snr} dB |")
    if args.min_duration > 0:
        lines.append(f"| Min duration | {args.min_duration}s |")
    if args.max_duration < 999:
        lines.append(f"| Max duration | {args.max_duration}s |")
    lines.append(f"| Random seed | {args.seed} |")
    lines.append("")

    # Input dataset stats
    lines.append("## Input dataset")
    lines.append("")
    lines.append(_dataset_stats(df_full, "Full corpus"))

    # Selected subset stats
    lines.append("## Selected subset")
    lines.append("")
    lines.append(_dataset_stats(df_selected, "Selected"))

    # Comparison
    n_full = len(df_full)
    n_sel = len(df_selected)
    lines.append("## Comparison")
    lines.append("")
    lines.append("| Metric | Input | Selected | Change |")
    lines.append("|--------|-------|----------|--------|")
    lines.append(f"| Samples | {n_full} | {n_sel} | {n_sel/n_full*100:.0f}% kept |")

    dur_full = df_full["duration_sec"].sum()
    dur_sel = df_selected["duration_sec"].sum()
    lines.append(f"| Duration | {dur_full/60:.1f} min | {dur_sel/60:.1f} min | {dur_sel/dur_full*100:.0f}% |")

    if "gender" in df_full.columns:
        fem_full = (df_full["gender"] == "female").sum()
        fem_sel = (df_selected["gender"] == "female").sum()
        pct_full = fem_full / n_full * 100
        pct_sel = fem_sel / n_sel * 100 if n_sel > 0 else 0
        lines.append(f"| Female % | {pct_full:.0f}% | {pct_sel:.0f}% | {'+'  if pct_sel > pct_full else ''}{pct_sel - pct_full:.0f}pp |")

    if "speaker_id" in df_full.columns:
        spk_full = df_full["speaker_id"].nunique()
        spk_sel = df_selected["speaker_id"].nunique()
        largest_full = df_full["speaker_id"].value_counts().iloc[0]
        largest_sel = df_selected["speaker_id"].value_counts().iloc[0]
        lines.append(f"| Speakers | {spk_full} | {spk_sel} | |")
        lines.append(f"| Largest speaker % | {largest_full/n_full*100:.0f}% | {largest_sel/n_sel*100:.0f}% | {'improved' if largest_sel/n_sel < largest_full/n_full else 'same'} |")

    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def _diverse_sample(pool, n, max_per_speaker, seed, existing_counts=None):
    """Sample up to n clips, capping each speaker at max_per_speaker."""
    if "speaker_id" not in pool.columns:
        return pool.head(n)

    selected = []
    speaker_counts = dict(existing_counts) if existing_counts else {}

    # Shuffle to avoid ordering bias
    shuffled = pool.sample(frac=1, random_state=seed)

    for idx, row in shuffled.iterrows():
        if len(selected) >= n:
            break
        spk = row["speaker_id"]
        count = speaker_counts.get(spk, 0)
        if count < max_per_speaker:
            selected.append(idx)
            speaker_counts[spk] = count + 1

    return pool.loc[selected]


if __name__ == "__main__":
    main()
