import argparse
from Bio import SeqIO
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import os

# --- LSTMClassifier Definition ---
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim=20, hidden_dim=95, num_layers=1, dropout=0.2733):
        super(LSTMClassifier, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, packed_input):
        _, (hn, _) = self.lstm(packed_input)
        last_hidden = hn[-1]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        out = self.sigmoid(out).squeeze(1)
        return out

# --- One-Hot Encoding ---
def one_hot_torch(seq: str, dtype=torch.float32):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq_bytes = torch.ByteTensor(list(bytes(seq, "utf-8")))
    aa_bytes = torch.ByteTensor(list(bytes(amino_acids, "utf-8")))
    arr = torch.zeros(len(amino_acids), len(seq_bytes), dtype=dtype)
    for i, aa in enumerate(aa_bytes):
        arr[i, seq_bytes == aa] = 1
    return arr

# --- Collation for a batch ---
def prepare_sequences(sequences):
    valid = [(s.id, str(s.seq)) for s in sequences if len(s.seq) > 0 and set(s.seq) <= set("ACDEFGHIKLMNPQRSTVWY")]
    ids, seqs = zip(*valid)
    tensor_seqs = [one_hot_torch(s).T for s in seqs]  # [L, 20]
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(tensor_seqs, batch_first=False)
    packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=False, enforce_sorted=False)
    return ids, packed

# --- Prediction Script ---
def predict_from_fasta(fasta_path, model_path, output_path):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    # Load model
    model = LSTMClassifier()
    # checkpoint = torch.load(model_path, map_location=device)  
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Read FASTA
    records = list(SeqIO.parse(fasta_path, "fasta"))
    ids, packed_input = prepare_sequences(records)
    packed_input = packed_input.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(packed_input).cpu().numpy()
        pred_labels = [1 if p > 0.5 else 0 for p in outputs]

    # Write output
    with open(output_path, "w") as f:
        f.write("ID\tPrediction\n")
        for seq_id, pred in zip(ids, pred_labels):
            f.write(f"{seq_id}\t{pred:.4f}\n")
    print(f"> Predictions saved to {output_path}")
    return output_path

# --- Main Script with argparse ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict AMP scores from protein FASTA sequences using trained LSTM.")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file")
    # parser.add_argument("--model", required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Path to output predictions file (.txt)")
    args = parser.parse_args()

    predict_from_fasta(args.fasta, './best_model_lstm_frozen.pt', args.output)


## run bash code exaample
# python tb_AMP_classification.py \
#   --fasta test_tbamp.fasta \
#   --output test_tbamp_pred.txt