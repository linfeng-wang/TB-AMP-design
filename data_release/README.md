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
├── splits/                           # the exact train/val/test partitions used
│   ├── tb_train.csv  tb_val.csv  tb_test.csv
│   └── general_train.csv  general_val.csv  general_test.csv
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

To reproduce the models you need `training/` alone; `splits/` pins the exact partitions
those results were measured on, so evaluation does not depend on reproducing a seed.

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

## `splits/`

The exact train/validation/test partitions behind the reported results, written out so
evaluation does not depend on regenerating a seed.

| File | Rows | Positive / negative | Share |
|---|---|---|---|
| `tb_train.csv` | 264 | 123 / 141 | 60 % |
| `tb_val.csv` | 88 | 41 / 47 | 20 % |
| `tb_test.csv` | 88 | 41 / 47 | 20 % |
| `general_train.csv` | 21,690 | 11,087 / 10,603 | 70 % |
| `general_val.csv` | 4,649 | 2,376 / 2,273 | 15 % |
| `general_test.csv` | 4,648 | 2,376 / 2,272 | 15 % |

Columns are `Sequences` and `label` (`1` AMP / `0` non-AMP) in both families; the TB
files carry the `AMP` column of `tb_amp_dataset.csv` cast to integer. `Peptide ID` is
omitted — rejoin it from `general_amp_dataset.csv` on `Sequences` if needed.

**Row order is the order the models received**, as `train_test_split` emitted it, not the
parent file's order.

The TB splits are taken *after* the training-time filter, so they partition the 440
surviving rows rather than all 704. The general splits partition the full 30,987.

Verified on generation:

| Check | TB | General |
|---|---|---|
| Rows across the three files sum to the parent | ✅ 440 | ✅ 30,987 |
| Pairwise sequence overlap between splits | ✅ 0 | ✅ 0 |
| Union equals the parent sequence set | ✅ | ✅ |
| `tb_test.csv` matches `model_scripts/test_seq.txt` and `test_y.txt`, in order | ✅ | — |

### Confirmed against the published training runs

The class counts of these splits match the confusion matrices stored in
`lstm_transfer.ipynb`'s saved outputs exactly, which identifies them as the partitions the
reported results were actually measured on:

| Split | Shipped file | Confusion-matrix row sums in the notebook |
|---|---|---|
| TB test | 47 negative / 41 positive | `[[42 5], [3 38]]` → 47 / 41 ✅ |
| TB validation | 47 negative / 41 positive | `[[41 6], [5 36]]` → 47 / 41 ✅ |
| General validation | 2,273 negative / 2,376 positive | `[[2167 106], [93 2283]]` → 2,273 / 2,376 ✅ |

### Three all-`X` records are dropped at load time

`general_amp_dataset.csv` contains three sequences consisting entirely of `X`
(`Peptide ID` 10821, 15095, 15098), carried in from the upstream positives. The one-hot
encoder in `lstm_transfer.ipynb` maps them to all-zero matrices and `collate_and_pack`
filters them out, since `SequenceDataset` measures length as `len(seq.replace("X", ""))`.

They are present in `splits/` because the splits mirror their parent file exactly, so the
effective counts at training time are one or two lower than the row counts above:

| File | Rows | Rows reaching the model |
|---|---|---|
| `general_train.csv` | 21,690 | 21,688 |
| `general_val.csv` | 4,649 | 4,649 |
| `general_test.csv` | 4,648 | **4,647** |

The TB files are unaffected — its training filter already excludes every `X`-bearing
sequence. The three records are 0.01 % of the general set and are dropped identically on
every run, so they do not affect the reported metrics; they are noted so the row counts
reconcile.

### Why these are shipped

`train_test_split`'s shuffling is stable in practice but is not a documented API
guarantee, so regenerating the splits ties the evaluation to a scikit-learn version.
These files remove that dependency: the held-out sets are now data rather than a recipe,
and anyone can score a model against the same 88 TB and 4,648 general sequences without
running the notebooks. They are byte-reproducible from `training/` plus the filter and
seed documented above, so they add no new source of truth.

---

## Reproducing the published TB result from this package

The headline TB classifier was retrained from `splits/` alone and compared against the
10-run summary stored in `lstm_transfer.ipynb` code cell 51.

| Metric | Reported | Retrained from `splits/` (n=10) | Δ |
|---|---|---|---|
| AUC | 0.9690 ± 0.0002 | **0.9688 ± 0.0005** | −0.0002 |
| Accuracy | 0.9045 ± 0.0056 | 0.8920 ± 0.0116 | −0.0125 |
| Sensitivity | 0.9244 ± 0.0073 | 0.9098 ± 0.0112 | −0.0146 |
| Specificity | 0.8872 ± 0.0098 | 0.8766 ± 0.0185 | −0.0106 |

**AUC reproduces to four decimal places.** The accuracy gap is 0.0125 — on an 88-sequence
test set that is **1.1 peptides**, and individual runs hit the published confusion matrix
`[[42 5], [3 38]]` exactly. Sensitivity and specificity differ by 0.57 of one positive and
0.47 of one negative respectively. Since AUC agrees to 4 dp, the model's ranking of the
test set is identical and only the 0.5-threshold decision on roughly one borderline
peptide moves; the residual is run-to-run variation and library-version numerics
(reproduced on torch 2.6.0+cu118; the stored outputs date from 2025-04-30), not a data
difference.

### Setup required to reproduce

Two stages, both reconstructable from this package plus the notebook:

| | Stage 1 (general pre-training) | Stage 2 (TB transfer) |
|---|---|---|
| Data | `splits/general_*.csv` | `splits/tb_*.csv` |
| Architecture | `LSTMClassifier(input_dim=20, hidden_dim=95, num_layers=1)` | same, encoder frozen |
| Dropout | 0.20287746211724011 | 0.273304750162406 |
| Learning rate | 0.004396992152527415 | 0.009721760525684369 |
| Weight decay | 3.999214064585909e-05 | 0.0017251614959305703 |
| Epochs / batch | 20 / 64 | 25 / 32 |
| Loss / optimiser | `BCELoss` / `Adam` | `BCELoss` / `Adam` |

Stage 2 freezes every parameter whose name contains `lstm`, leaving `fc.weight` and
`fc.bias` — two tensors — trainable. Note the `torch.save` inside the training function is
commented out, so the **last-epoch** model is evaluated, not the best-validation-loss one.

### This package ships no model weights

Stage 2 starts from the stage-1 checkpoint `best_model_lstm_1.pt`
(MD5 `b0701ec8026ada5bcff4691a2d537bf9`), which lives in the code repository at
`model_scripts/weights/`, not in this data package. Two consequences:

- **Reproducing stage 2 directly** requires that file from the repository.
- **Reproducing from this package alone** means retraining stage 1 first. That works:
  stage 1 retrained from `splits/general_*.csv` with the recipe above reaches general test
  accuracy 0.9563 / AUC 0.9932, and transferring it into stage 2 gives TB test
  **accuracy 0.8977, AUC 0.9714** (confusion matrix `[[40 7], [2 39]]`) — within
  0.0068 accuracy, or 0.6 peptides, of the published figure.

`model_scripts/best_model.pt` is **not** the stage-1 checkpoint and will not reproduce
these numbers: it is a `hidden_dim=64` general-AMP model, so it matches zero tensors
against the `hidden_dim=95` transfer architecture. Scored directly on the shipped splits
it gives TB test accuracy 0.7500 / AUC 0.9133.

---

## Reproducibility summary

| | Reproducible from `sources/`? | Notes |
|---|---|---|
| TB positives | ✅ exactly | replay of `data_gen.ipynb` reproduces all 352, in order |
| TB negatives | — | defined by the shipped file; unseeded sampling at build time |
| General AMP set | ✅ exactly | rebuilds from `sources/general_amp/`, row order included |
| Training-time filter | ✅ exactly | deterministic, 704 → 440 rows |
| Train/val/test splits | ✅ **stored** | shipped in `splits/`; no seed replay required |

The splits are shipped as files in `splits/`, so evaluation is independent of library
versions. They remain regenerable from `training/` with the filter and
`train_test_split(random_state=42)` documented above, which is how they were produced and
verified — but nothing downstream depends on that replay succeeding.

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
| `splits/` rows sum to their parent datasets | ✅ 440 and 30,987 |
| Pairwise sequence overlap between splits | ✅ 0 (both families) |
| `splits/tb_test.csv` vs. the notebook's exported test set | ✅ identical, in order |
| `splits/` class counts vs. the notebook's stored confusion matrices | ✅ match (TB 47/41; general 2,273/2,376) |
| Published TB result retrained from `splits/` alone | ✅ AUC 0.9688 ± 0.0005 vs. reported 0.9690 ± 0.0002 |
| Stage 1 retrained from `splits/general_*.csv`, then transferred | ✅ TB test accuracy 0.8977, AUC 0.9714 |

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

The terms of APD3 and DBAASP were reviewed and permit the redistribution of the extracts
in `sources/` for academic use with attribution; cite each database when using them. The
UniProtKB exports are CC BY 4.0.

`sources/` is distributed in the release archive (`data_release.zip`) rather than through
the git index, to keep 19 MB of third-party content out of repository history. The
queries behind each file are documented above, so every export can also be regenerated
from the upstream databases directly.

---

## File inventory

| Path | Size | MD5 |
|---|---|---|
| `training/tb_amp_dataset.csv` | 18 KB | `cde61eb182d0f4490eae98b7cefa163d` |
| `training/general_amp_dataset.csv` | 3.3 MB | `5369bc0fe5109b259ed6b1527f77b7b9` |
| `splits/tb_train.csv` | 7.9 KB | `e6d939e0cb76316206b5253f55209e8e` |
| `splits/tb_val.csv` | 2.9 KB | `4e4ffb39c843bc5b611412ad782620d1` |
| `splits/tb_test.csv` | 2.4 KB | `5429c57a3380cc0ea5d628fc7eed3e8d` |
| `splits/general_train.csv` | 834 KB | `ba34c7cc888f8a67ca25772f01917c10` |
| `splits/general_val.csv` | 181 KB | `65bb3cc6cbddf597f49f76b1c69466af` |
| `splits/general_test.csv` | 182 KB | `16e6344ab11da391d304b3604b4345af` |
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

Total: 24 MB. The `training/` and `sources/` files are byte-identical copies of their
originals in `data/`; the `splits/` files are generated from `training/` as described
above.
