import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# --- Generative Model Definition ---
class GenerativeLSTM(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=149, num_layers=2, dropout=0.3):
        super(GenerativeLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

# --- Amino Acid Utilities ---
aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: i for i, aa in enumerate(aa_vocab)}
idx_to_aa = {i: aa for aa, i in aa_to_idx.items()}

def one_hot_encode_amino_acid(aa):
    vec = torch.zeros(len(aa_vocab))
    vec[aa_to_idx[aa]] = 1.0
    return vec

def sample_start_amino_acid():
    return random.choice(aa_vocab)

def generate_sequence_from_seed(model, seed, max_length=30, temperature=1.0, device='cpu'):
    model.eval()
    input_seq = [one_hot_encode_amino_acid(aa).to(device) for aa in seed]
    input_tensor = torch.stack(input_seq).unsqueeze(0)  # [1, L, 20]

    generated = seed.copy()
    with torch.no_grad():
        for _ in range(max_length - len(seed)):
            output = model(input_tensor)
            logits = output[0, -1, :]
            probs = F.softmax(logits / temperature, dim=-1).cpu().numpy()
            next_idx = np.random.choice(len(aa_vocab), p=probs)
            next_aa = idx_to_aa[next_idx]
            next_vec = one_hot_encode_amino_acid(next_aa).to(device).unsqueeze(0).unsqueeze(0)
            input_tensor = torch.cat([input_tensor, next_vec], dim=1)
            generated.append(next_aa)
    return ''.join(generated)

# --- Generation Function ---
def generate_fixed_length_peptides(model_path, output_fasta, num_seqs, temperature, fixed_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenerativeLSTM()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    generated_peptides = []
    for i in range(num_seqs):
        start = sample_start_amino_acid()
        peptide = generate_sequence_from_seed(model, [start], max_length=fixed_length, temperature=temperature, device=device)
        generated_peptides.append(peptide)

    with open(output_fasta, "w") as f:
        for i, pep in enumerate(generated_peptides):
            f.write(f">peptide{i}\n{pep}\n")

# --- CLI Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AMP-like peptides of fixed length using a trained LSTM generator.")
    parser.add_argument("--output", required=True, help="Output FASTA file path")
    parser.add_argument("--num_seqs", type=int, default=100, help="Number of sequences to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--length", type=int, required=True, help="Fixed length of generated sequences")
    args = parser.parse_args()
    
    model_path = "./best_model_lstm_generator-notrans-tb.pt"
    model_path = "./best_model_lstm_generator.pt"
    generate_fixed_length_peptides(model_path, args.output, args.num_seqs, args.temperature, args.length)
    print(f"> Generated {args.num_seqs} peptides of length {args.length} and saved to {args.output} with temperature = {args.temperature}.")

# python tb_AMP_generation.py \
#   --output peptides_25mers.fasta \
#   --num_seqs 20 \
#   --temperature 1 \
#   --length 25
