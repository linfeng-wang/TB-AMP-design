
# 🧬 TB-AMP: Deep Learning for TB-Specific Antimicrobial Peptides

This repository contains the code and data pipelines used in our study on deep learning models for the classification and generation of **tuberculosis-specific antimicrobial peptides (TB-AMPs)**. The project leverages LSTM-based architectures, attention mechanisms, and transfer learning to improve AMP discovery targeting *Mycobacterium tuberculosis*.

---

**Reference: [Paper Title] (DOI link)
**

## How to install

### Prerequisites
- Python 3.10+
- Conda (or Miniconda)
- Git
- Jupyter Notebook (optional, for running notebooks)
- PyTorch (version compatible with your CUDA version, if applicable)
- Other dependencies will be installed via the provided `env.yml` file.


### 1. Clone the repo
```bash
# Clone the repository
git clone https://github.com/linfeng-wang/TB-AMP-design.git
# Navigate to the project directory
cd TB-AMP-design
```

### 2. Create the Conda environment

```bash
# Create a new conda environment using the provided YAML file
conda env create -f env.yml
# Activate the environment
conda activate tb_amp_env  # activate the environment you created
```

### 3. Run the AMP generator/classifier


## 📁 File System Structure

```
data/                       # Contains the datasets used for training and evaluation.
model_scripts/              # Model architectures and training scripts.
    data_gen.ipynb                      # Data generation and preprocessing notebook.
    gene-vanilla-lstm.py                # Vanilla LSTM model for generation.
    lstm_no_transfer.py                 # LSTM classification without transfer learning.
    lstm_transfer.py                    # LSTM classification with transfer learning.
    seq_eval.py                         # Evaluation of generated sequences.
    weights/                            # Pretrained model weights.
    database_check/                     # Scripts for database validation used in seq_eval.
model_implementation/        # Executable model scripts.
    tb_AMP_classification.py            # Classify TB-AMPs from input sequences.
    tb_AMP_generation.py                # Generate TB-AMP candidates.
    best_model_lstm_frozen.pt           # Weights of the best classifier (frozen encoder).
    best_model_lstm_generator.pt # Weights of the best generator (no transfer).
    peptides_25mers.fastq               # Generated peptides in FASTA format.
    test_tbamp.fasta                    # Test set of TB-AMPs in FASTA format.
    test_tbamp_pred.txt                 # Predictions for test TB-AMPs.
env_droplet.yml              # Environment YAML file for project setup.
```

---

## 🚀 Example Usage

### 🔄 Generation

```bash
python tb_AMP_generation.py \
  --output peptides_25mers.fasta \
  --num_seqs 20 \
  --temperature 1 \
  --length 25
```

| Argument        | Description                                                                                        |
|----------------|----------------------------------------------------------------------------------------------------|
| `--output`      | Output FASTA file to save generated peptides.                                                      |
| `--num_seqs`    | Number of peptide sequences to generate (e.g., 20).                                                |
| `--temperature` | Sampling temperature; controls diversity (1 = default, >1 = more diverse, <1 = more conservative). |
| `--length`      | Fixed length of each generated peptide (e.g., 25 amino acids).                                     |

---

### 🧪 Classification

```bash
python tb_AMP_classification.py \
  --fasta test_tbamp.fasta \
  --output test_tbamp_pred.txt
```

| Argument   | Description                                                |
|------------|------------------------------------------------------------|
| `--fasta`  | Input FASTA file containing peptide sequences to classify. |
| `--output` | Output text file to save AMP prediction results.           |

