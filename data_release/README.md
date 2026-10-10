# TB-AMP — Dataset Package

Data for *TB-AMP: Deep Learning for TB-Specific Antimicrobial Peptides*
(https://github.com/linfeng-wang/TB-AMP-design).

This folder contains **only the files the models actually train and test on**, plus the
upstream sources those files were built from. The project's working `data/` directory
holds additional files from exploratory analysis that do not feed the published models;
they are not included here.

Every quantitative statement in this README was measured against the shipped files — see
*Integrity checks performed* for the full list. Headline results: all released files are
byte-identical to the originals used in training, `general_amp_dataset.csv` rebuilds
exactly from its sources, and there is zero sequence overlap between the pre-training and
fine-tuning sets and zero overlap between train and test.

---

## Layout

```
data_release/
├── README.md
├── training/                         # the two files the models consume directly
│   ├── tb_amp_dataset.csv            #   Stage 2 — TB-specific fine-tuning + evaluation
│   └── general_amp_dataset.csv       #   Stage 1 — general AMP pre-training
└── sources/                          # upstream inputs, grouped by which dataset they build
    ├── tb_specific/                  #   → builds tb_amp_dataset.csv
    │   ├── positives_antitbamp_prod.xlsx
    │   ├── positives_antitbpred.csv
    │   ├── positives_dbaasp_tb.csv
    │   ├── negatives_uniprot_len_to20.tsv
    │   ├── negatives_uniprot_acc_A0A0B4J2F0.tsv
    │   └── negatives_uniprot_acc_A2RU14_additional.tsv
    └── general_amp/                  #   → builds general_amp_dataset.csv
        ├── positives_apd_adam.csv
        ├── positives_dbaasp.csv
        ├── negatives_uniprot_len10_80.csv
        ├── negatives_uniprot_len5_30.csv
        └── negatives_uniprot_len5_30_apr23.tsv
```

To reproduce the models you need `training/` alone.

`sources/` is included for auditing. On rebuilds, note the asymmetry:

- **`general_amp_dataset.csv` rebuilds exactly** from `sources/general_amp/` — verified,
  all 30,987 rows in order.
- **`tb_amp_dataset.csv`** is the authoritative copy of the TB set. Its positive class
  replays exactly from `sources/tb_specific/`; its negative class was drawn by unseeded
  sampling and is therefore defined by the shipped file. Replaying `data_gen.ipynb` also
  requires skipping its cells 3 and 5, which open two cross-checking files
  (`grampa.csv`, `dramp.xlsx`) deliberately excluded here — they contribute no unique
  sequences, so skipping them changes nothing.

---

## `training/tb_amp_dataset.csv`

The core file of the study. Every reported TB-AMP result derives from it.

| | |
|---|---|
| Rows | 704 (352 positive / 352 negative) |
| Columns | `Sequences` (amino-acid string), `AMP` (`1.0` active / `0.0` inactive) |
| Unique sequences | 695 — positives all unique; negatives contain 9 redundant rows (see below) |
| Sequence length | 3–128 (positives), 4–120 (negatives) |
| Rows reaching the model | **440** (205 positive / 235 negative) — see *Training-time filter* below |
| Split | 60 % train / 20 % validation / 20 % test of those 440, stratified, `random_state=42` → **264 / 88 / 88** |
| Built by | `model_scripts/data_gen.ipynb`, code cell 26 |
| Original filename | `data/all_seq702.csv` |

> Cell numbers throughout this README are **1-based indices over code cells only**
> (markdown cells not counted).

Peptides with measured activity against *Mycobacterium tuberculosis*, paired with
length-matched non-antimicrobial sequences from UniProt.

> **On the filename.** The working file was named `all_seq702.csv` while holding 704 rows
> (as did its predecessor `all_seq722.csv` at 716 rows). Renamed here to a descriptive
> name so the count lives in this README rather than in the filename.

### Provenance — positive class (352 sequences)

Reconstructed from `model_scripts/data_gen.ipynb` and **verified**: replaying cells 3–16
reproduces the 352 shipped positives exactly, in the same order.

Six inputs are concatenated (cell 8), then deduplicated case-insensitively (cell 9),
then cleaned (cells 15–16: drop sequences containing `-` or the literal `not available`,
strip whitespace, drop four non-sequence strings, uppercase). 705 concatenated → 361
after dedup → 352 after cleaning.

| Input | Pre-filter rows | Contributed | Unique survivors | Filter applied |
|---|---|---|---|---|
| `positives_antitbamp_prod.xlsx` | 549 | 423 | **216** | drop sequences containing `(` |
| `positives_antitbpred.csv` | 246 | 74 | **74** | keep those *not* already in antiTBamp |
| `positives_dbaasp_tb.csv` | 91 | 66 | **50** | keep those *not* already in antiTBamp |
| APD3 list (hard-coded, cell 7) | 15 | 15 | **12** | — |
| `grampa.csv` | 51,345 | 49 | **0** | `bacterium == 'M. tuberculosis'`, then intersect with antiTBamp |
| `dramp.xlsx` | 11,612 | 78 | **0** | intersect with antiTBamp |

**GRAMPA and DRAMP contribute zero unique sequences.** Cells 3 and 5 keep only entries
that are *already present* in antiTBamp (`if x in antiTBamp['Sequences']`), so every
sequence they add is a duplicate removed at cell 9. They functioned as cross-validation
of antiTBamp rather than as sources. They are therefore **excluded from this package** —
including them would add 10 MB that changes nothing. The same applies to cell 2's
`tuberculosis_rows`, which is computed and never used.

Because the notebook still *opens* both files, replaying it against `sources/` requires
commenting out cells 3 and 5 (or re-downloading GRAMPA and DRAMP). Doing so does not
change the resulting positive set.

The 12 surviving APD3 sequences are hard-coded as a Python list in `data_gen.ipynb`
cell 7, not read from a file, so no file for them appears in `sources/`. They are
cathelicidin/defensin-family sequences retrievable from APD3 by sequence lookup.

### Provenance — negative class (352 sequences)

Drawn from three UniProt "NOT antimicrobial" queries, then length-matched and partially
X-masked (cells 17–18). **350 of 352 match a shipped source record exactly:**

| Input | Rows | Verbatim matches in shipped negatives |
|---|---|---|
| `negatives_uniprot_len_to20.tsv` (length ≤ 20, retrieved 2025-02-11) | 21,004 | 225 |
| `negatives_uniprot_acc_A0A0B4J2F0.tsv` (accession set, 2025-02-26) | 125 | 96 |
| `negatives_uniprot_acc_A2RU14_additional.tsv` (accession set, 2025-02-26) | 10 | 9 |
| — synthetically X-masked (match a source with `X` as wildcard) | — | 29 |
| — superseded UniProt entries (see below) | — | 2 |

Construction: take the first 352 sequences of the length ≤ 20 pool; remove 50 randomly
chosen length-20 sequences; add the `A0A0B4J2F0` set; sample 342 (`len(amp) - 10`); append
the 10-sequence `A2RU14` additional set.

**Synthetic X-masking.** The positive class contains 8.52 % sequences with `X`
(unknown residue). To stop the model using the presence of `X` as a shortcut for class,
cell 18 masks the same proportion of negatives: `int(352 × 8.52 %) = 29` sequences get
3–5 random positions replaced with `X`. The arithmetic checks out exactly — of the 73
negatives containing `X`, **44 are native** (UniProt genuinely uses `X`; the source pool
is 2.34 % X-bearing) and **29 are synthetic**. So 29 of 352 negatives (8.2 %) are
modified, not verbatim, UniProt sequences.

> **Scope.** The training-time filter (below) excludes every `X`-bearing sequence — all
> 103 distinct ones, synthetic and native alike — so this masking does not influence the
> published results in either direction. It is recorded here because it means 29 of the
> negatives are modified rather than verbatim UniProt records, and because it matters if
> you train on the unfiltered file.

**Two sequences trace to a superseded UniProt snapshot.** `HGIHCDTLL` (9 residues) and
`MIINHNMSAINANRVLGNT` (19 residues) are both inside the length ≤ 20 query range but are
not in the shipped pool, consistent with UniProt having revised those entries between the
build date and the retrieval date of the file included here. Together they are 0.6 % of
the dataset, and only one of the two survives the training filter.

### Training-time filter and effective dataset

The file ships all 704 rows. Every classifier cell deduplicates and then applies
three further filters before splitting (`lstm_transfer.ipynb` code cells 25, 31, 35, 39,
44; `lstm_no_transfer.ipynb` likewise):

```python
df = df.drop_duplicates(subset='Sequences')
df = df[(df['Sequences'].str.len() >= 10) &
        (df['Sequences'].apply(lambda x: len(set(x)) > 1)) &
        (~df['Sequences'].str.contains('X'))]
```

| Stage | Rows | Positive / negative |
|---|---|---|
| Shipped file | 704 | 352 / 352 |
| After `drop_duplicates` | 695 | 352 / 343 |
| After length ≥ 10, >1 distinct residue, no `X` | **440** | **205 / 235** |

Removals after deduplication total 255 of 695 unique sequences. The three criteria
overlap, so they do not sum:

| Criterion | Matches | Removed by this criterion alone |
|---|---|---|
| Shorter than 10 residues | 176 | 151 |
| Contains `X` | 103 | 79 |
| Only one distinct residue | 1 | 0 (also short) |
| Both short and `X`-bearing | 24 | 24 |
| **Union** | **255** | |

The filter removes 147 positives and 108 negatives. The 440 surviving rows are then
split 60/20/20 stratified (`train_test_split(test_size=0.4, random_state=42)` followed by
a 50/50 split of the holdout) giving **train 264 / validation 88 / test 88**.

Two properties of the filtered set to carry into any analysis:

- **Class balance is 46.6 % positive (205/235).** The shipped file is balanced 50/50;
  the filtered set is not, because short and `X`-bearing sequences are distributed
  unevenly across the classes. Compare reported metrics against the 46.6 % majority-class
  baseline.
- **Deduplication and the `X` exclusion both happen here**, which is what makes the
  row-level repeats and the synthetic masking described above immaterial to the results.

This filter and split are fully deterministic and were replayed to confirm the figures
above: the resulting test set matches `model_scripts/test_seq.txt` and
`model_scripts/test_y.txt` exactly, in order, across all 88 records.

### This file is the authoritative copy

The positive class is fully deterministic and replays exactly from `data_gen.ipynb`. The
negative class was drawn by unseeded sampling (`random.sample`, `random.randint` in cells
17–18), so it is defined by this file rather than by a rerun of the notebook.

**Use `training/tb_amp_dataset.csv` as shipped.** It is the exact dataset behind every
published number, and all verification in this README was performed against it.

### Row-level composition of the negative class

The 352 negative rows carry 343 distinct sequences: `QPPGGSKVILF` appears 5×, and
`EEEDKKEDVGTVVGI`, `MAAYLDPTGQY`, `MEMALMVAQTRKGKSVV`,
`MTSNTPNQEPVSYPIFTVRWVAVHTLAVPTIFFLGAIAAMQFIQR` and
`MSDKPDMAEIEKFDKSKLKKTETQEKNPLPSKETIEQEKQAGES` twice each — cell 18 appends the
`A2RU14` set to an already-sampled pool.

This has no effect on the published results, for two measured reasons:

- **No label ambiguity.** Every repeat is within the negative class; no sequence is
  labelled both positive and negative. Measured conflicts: **0**.
- **No train/test contamination.** The training cells call
  `drop_duplicates(subset='Sequences')` *before* splitting (see *Training-time filter*
  above), so the repeats are collapsed and cannot straddle the split. Measured
  train/test sequence overlap: **0**.

The file is shipped in its original row order, which the seeded split depends on.

---

## `training/general_amp_dataset.csv`

Large general-purpose AMP set used to pre-train the classifier before TB-specific
fine-tuning, and to pre-train the generator.

| | |
|---|---|
| Rows | 30,987 (15,839 positive / 15,148 negative) |
| Columns | unnamed index, `Peptide ID`, `Sequences`, `label` (`1` AMP / `0` non-AMP) |
| Split | 70 % train / 15 % validation / 15 % test, stratified, `random_state=42` |
| Produced by | `model_scripts/lstm_transfer.ipynb`, code cell 16 |
| Rebuild status | ✅ **verified** — replaying the recipe below over `sources/general_amp/` reproduces all 30,987 rows in identical order |

A derived file, included ready-made so Stage 1 runs without reprocessing `sources/`.

**No overlap with the TB set** — the build explicitly drops any sequence present in
`tb_amp_dataset.csv`, so there is no leakage between pre-training and fine-tuning.
Verified: 0 shared sequences.

### Sources

All five are read live by `model_scripts/lstm_transfer.ipynb` code cell 6.

| File | Rows | Seq. length | Role | Upstream source | Original filename |
|---|---|---|---|---|---|
| `positives_apd_adam.csv` | 3,306 | 2–183 | positives | APD3 / ADAM natural AMPs, 2024a release | `naturalAMPs_APD2024a-ADAM.csv` |
| `positives_dbaasp.csv` | 23,363 | 1–190 | positives | DBAASP (full export) | `DBAASP_peptides.csv` |
| `negatives_uniprot_len10_80.csv` | 60,082 | 10–80 | negatives | UniProtKB, length 10–80, NOT antimicrobial; retrieved 2025-04-14 | `uniprotkb_length_10_TO_80_NOT_antimicro_2025_04_14.fasta.csv` |
| `negatives_uniprot_len5_30.csv` | 23,090 | 5–30 | negatives | UniProtKB, length 5–30, NOT antimicrobial; retrieved 2025-04-14 | `uniprotkb_length_10_TO_80_NOT_antimicro_2025_04_14.fasta1.csv` |
| `negatives_uniprot_len5_30_apr23.tsv` | 23,198 | 5–30 | negatives | UniProtKB, length 5–30, NOT antimicrobial; retrieved 2025-04-23 | `uniprotkb_length_5_TO_30_NOT_antimicrob_2025_04_23 (1).tsv` |

> **Renaming note.** The fourth file was originally named `...length_10_TO_80_...fasta1.csv`
> but contains sequences of length **5–30**, not 10–80 — it was written out from the 5–30
> FASTA under a copy-pasted 10–80 filename. Verified by inspection: min length 5, max 30.
> Renamed to match its real contents; data untouched.

### Build recipe

From `model_scripts/lstm_transfer.ipynb` code cells 6, 10, 14, 15 and 16:

1. **Positives.** DBAASP, keeping sequences of length ≥ 10 and dropping rows whose
   `TARGET GROUP` contains `Fungus`; reduce to `ID`, `SEQUENCE`. Concatenate with the
   full APD/ADAM set.
2. **Negatives.** Concatenate the three UniProt files into one pool (106,370 sequences),
   then keep only string sequences of length ≥ 10 and drop duplicate sequences
   (code cell 10).
3. **Length-matched sampling.** Group the negative pool by sequence length and, per
   length, sample `14 ×` the number of positives at that length (fallback 30 where a
   length is absent from the positives), capped at availability; `random_state=42`. This
   matches the negative length distribution to the positive one rather than sampling
   uniformly.
4. **Label and merge.** Positives → `label=1`, sampled negatives → `label=0`.
5. **Clean.** Drop sequences with non-standard amino acids (outside
   `ACDEFGHIKLMNPQRSTVWYX`); drop duplicate sequences; **drop any sequence in
   `tb_amp_dataset.csv`**; drop sequences shorter than 10 residues.
6. Result: 30,987 sequences.

Step 3 is stochastic but seeded (`random_state=42`). **Replayed against
`sources/general_amp/` this recipe reproduces the shipped file exactly** — 30,987 rows,
15,839 / 15,148 labels, identical sequences in identical order (verified with
Python 3.10.16 / pandas 2.3.2 / scikit-learn 1.7.2). The shipped file remains
authoritative nonetheless: a different pandas version could change
`groupby(...).apply(...).sample()` ordering, so treat an exact rebuild as a property of
this environment rather than a guarantee.

---

## Reproducibility summary

| | Reproducible from `sources/`? | Notes |
|---|---|---|
| TB positives | ✅ exactly | replay of `data_gen.ipynb` reproduces all 352, in order |
| TB negatives | — | defined by the shipped file; unseeded sampling at build time |
| General AMP set | ✅ exactly | rebuilds from `sources/general_amp/`, row order included |
| Training-time filter | ✅ exactly | deterministic, 704 → 440 rows |
| Train/val/test splits | ✅ seeded | `random_state=42`, regenerated at run time |

Splits are produced by `train_test_split(random_state=42)` at run time rather than stored
as files. They reproduce exactly given a matching scikit-learn version and the shipped CSV
row order — both of which this package fixes. If you intend to vary library versions,
materialise the splits to disk once and load those instead.

Reference environment for all verification in this README:
**Python 3.10.16, pandas 2.3.2, scikit-learn 1.7.2** (conda env `tb_amp_env`).

Seeds set in the training notebooks: `np.random.seed(42)`, `random.seed(42)`,
`torch.manual_seed(42)`.

### Integrity checks performed

All of the following were measured against the shipped files, not inferred:

| Check | Result |
|---|---|
| All 13 released files vs. originals in `data/` | ✅ byte-identical (MD5) |
| `general_amp_dataset.csv` rebuild from `sources/` | ✅ exact, incl. row order |
| Sequence overlap between TB and general sets (leakage) | ✅ 0 |
| Sequences labelled both positive and negative | ✅ 0 |
| Train/test sequence overlap after the training filter | ✅ 0 |
| Non-standard residues in `tb_amp_dataset.csv` | ✅ none (20 standard + `X`) |
| `model_scripts/test_y.txt` vs. replayed `y_test` | ✅ identical (88 labels) |
| `model_scripts/test_seq.txt` vs. replayed `X_test` | ✅ identical (88 records, full length) |

---

## Licensing and attribution

`training/` files are outputs of this project. `sources/` files are redistributed
third-party database content, each under its own terms:

- **APD3 / ADAM** — Antimicrobial Peptide Database; academic use, cite the database.
- **DBAASP** — Database of Antimicrobial Activity and Structure of Peptides; cite the database.
- **AntiTBPred** — cite the source publication.
- **UniProtKB** — CC BY 4.0.
- `positives_antitbamp_prod.xlsx` — manually curated compilation assembled for this
  project from the primary literature; the largest single contributor at 216 of the 352
  positives. Cite this dataset when using it.

APD3 and DBAASP both place conditions on bulk redistribution of database content, applied
more strictly to verbatim exports than to derived datasets. Confirm their current terms
before publishing `sources/`; distributing `training/` alone, with the queries documented
here for re-download, satisfies them in either case.

---

## File inventory

| Path | Size | MD5 |
|---|---|---|
| `training/tb_amp_dataset.csv` | 18 KB | `cde61eb182d0f4490eae98b7cefa163d` |
| `training/general_amp_dataset.csv` | 3.3 MB | `5369bc0fe5109b259ed6b1527f77b7b9` |
| `sources/tb_specific/positives_antitbamp_prod.xlsx` | 73 KB | `a69c034a10024a957178c5a4277665e6` |
| `sources/tb_specific/positives_antitbpred.csv` | 4.1 KB | `4f5e0af07b1c1cf083177791c94063e0` |
| `sources/tb_specific/positives_dbaasp_tb.csv` | 7.9 KB | `1fba21bb74f5964e8e886bdf2b0cf864` |
| `sources/tb_specific/negatives_uniprot_len_to20.tsv` | 1.4 MB | `554bfb6ad3a73887dc18a4d3b96ae883` |
| `sources/tb_specific/negatives_uniprot_acc_A0A0B4J2F0.tsv` | 27 KB | `4310db29fd27168e84666a2405f671f0` |
| `sources/tb_specific/negatives_uniprot_acc_A2RU14_additional.tsv` | 2.5 KB | `f57402cba5a9b590cd91cd73793d2df2` |
| `sources/general_amp/positives_apd_adam.csv` | 143 KB | `bc18f485e53c35e02578bf4c77c19c80` |
| `sources/general_amp/positives_dbaasp.csv` | 2.0 MB | `faecddaedcc53e1b75997f8720cacb8f` |
| `sources/general_amp/negatives_uniprot_len10_80.csv` | 11 MB | `21a8bc421f3367514ab7deb3e46f7e1c` |
| `sources/general_amp/negatives_uniprot_len5_30.csv` | 3.4 MB | `aa09cb45e2aa0c2a66b6eed50252c99b` |
| `sources/general_amp/negatives_uniprot_len5_30_apr23.tsv` | 1.2 MB | `50b652ec98ab4cc3d5074566d43e7136` |

Total: 22 MB. All files are byte-identical copies of their originals in `data/`.
