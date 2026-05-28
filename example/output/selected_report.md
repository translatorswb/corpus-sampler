# Corpus Analysis and Selection Report

Generated from `/Users/alp/Professional/CLEAR/play/LT4Crisis/_Workspaces/select-corpus/rohingya_samples_output/metadata.csv`

## Selection parameters

| Parameter | Value |
|-----------|-------|
| Target samples | 50 |
| Max per speaker | 5 |
| Min female ratio | 0.3 |
| Exclude music | True |
| Exclude English | True |
| Exclude non-speech | True |
| Random seed | 42 |

## Input dataset

### Full corpus

| Metric | Value |
|--------|-------|
| Samples | 100 |
| Total duration | 15.9 min (0.26 h) |
| Mean duration | 9.5s |
| Duration range | 4.9s – 15.0s |
| SNR range | 9 – 84 dB |
| Mean SNR | 35 dB |

**Gender distribution:**

| Gender | Count | % |
|--------|-------|---|
| male | 79 | 79% |
| female | 15 | 15% |
| unknown | 6 | 6% |

**Content classification:**

| Category | Count | % |
|----------|-------|---|
| Usable speech | 90 | 90% |
| Music/jingles | 7 | 7% |
| Non-speech total | 10 | 10% |
| English | 14 | 14% |

**Speaker distribution:**

| Metric | Value |
|--------|-------|
| Distinct speakers | 15 |
| Largest cluster | spk_000 (21 clips, 21%) |
| Smallest cluster | spk_013 (2 clips) |
| Median cluster size | 6 clips |

Top 10 speakers:

| Speaker | Clips | % | Gender |
|---------|-------|---|--------|
| spk_000 | 21 | 21% | male=21 |
| spk_007 | 11 | 11% | male=11 |
| spk_011 | 9 | 9% | male=9 |
| spk_003 | 9 | 9% | male=9 |
| spk_014 | 9 | 9% | male=8, unknown=1 |
| spk_005 | 7 | 7% | male=6, female=1 |
| spk_010 | 7 | 7% | male=7 |
| spk_006 | 6 | 6% | female=6 |
| spk_002 | 4 | 4% | male=3, female=1 |
| spk_001 | 3 | 3% | male=3 |

**Detected languages (Whisper):**

| Language | Count | % |
|----------|-------|---|
| bn | 30 | 30% |
| en | 18 | 18% |
| fa | 10 | 10% |
| ne | 8 | 8% |
| tr | 5 | 5% |
| ml | 5 | 5% |
| ar | 4 | 4% |
| hi | 4 | 4% |
| si | 3 | 3% |
| ur | 2 | 2% |

## Selected subset

### Selected

| Metric | Value |
|--------|-------|
| Samples | 50 |
| Total duration | 8.0 min (0.13 h) |
| Mean duration | 9.6s |
| Duration range | 4.9s – 15.0s |
| SNR range | 10 – 79 dB |
| Mean SNR | 33 dB |

**Gender distribution:**

| Gender | Count | % |
|--------|-------|---|
| male | 38 | 76% |
| female | 12 | 24% |

**Content classification:**

| Category | Count | % |
|----------|-------|---|
| Usable speech | 50 | 100% |
| Music/jingles | 0 | 0% |
| Non-speech total | 0 | 0% |
| English | 0 | 0% |

**Speaker distribution:**

| Metric | Value |
|--------|-------|
| Distinct speakers | 12 |
| Largest cluster | spk_006 (5 clips, 10%) |
| Smallest cluster | spk_004 (1 clips) |
| Median cluster size | 5 clips |

Top 10 speakers:

| Speaker | Clips | % | Gender |
|---------|-------|---|--------|
| spk_006 | 5 | 10% | female=5 |
| spk_005 | 5 | 10% | male=4, female=1 |
| spk_014 | 5 | 10% | male=5 |
| spk_000 | 5 | 10% | male=5 |
| spk_011 | 5 | 10% | male=5 |
| spk_007 | 5 | 10% | male=5 |
| spk_003 | 5 | 10% | male=5 |
| spk_010 | 5 | 10% | male=5 |
| spk_008 | 3 | 6% | female=3 |
| spk_009 | 3 | 6% | female=3 |

**Detected languages (Whisper):**

| Language | Count | % |
|----------|-------|---|
| bn | 14 | 28% |
| fa | 8 | 16% |
| ne | 7 | 14% |
| ml | 5 | 10% |
| tr | 4 | 8% |
| si | 3 | 6% |
| en | 3 | 6% |
| ar | 1 | 2% |
| hi | 1 | 2% |
| kk | 1 | 2% |

## Comparison

| Metric | Input | Selected | Change |
|--------|-------|----------|--------|
| Samples | 100 | 50 | 50% kept |
| Duration | 15.9 min | 8.0 min | 51% |
| Female % | 15% | 24% | +9pp |
| Speakers | 15 | 12 | |
| Largest speaker % | 21% | 10% | improved |
