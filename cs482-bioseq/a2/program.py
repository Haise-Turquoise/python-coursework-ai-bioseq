#!/usr/bin/env python3
"""
ASSN 2: Distinguish Natural and Random Peptides
Reads peptide sequences from stdin, outputs score1 score2 peptide to stdout.
Requires: score_params.json, peptide_cnn3Btwo.pt (same directory)
"""
import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np

# prepare parameter files addr to wait load
scriptAddr = os.path.dirname(os.path.abspath(__file__))

# CNN architecture (need match training result config, need carefully check)
class PeptideCNN(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=64, num_filters=64,
                 kernel_sizes=(3, 5, 7), num_classes=2, pad_idx=20, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters,
                      kernel_size=ks, padding=0)
            for ks in kernel_sizes
        ])
        filters_total = num_filters * len(kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters_total, num_classes)

    def forward(self, x):
        emb = self.embedding(x)          # (batch, seq_len, embed_dim)
        emb = emb.permute(0, 2, 1)       # (batch, embed_dim, seq_len)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(emb))     # (batch, num_filters, reduced_len)
            p = c.max(dim=2).values       # (batch, num_filters)
            pooled.append(p)
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        logits_score = self.fc(cat)
        return logits_score

# Load k-mer score table
scoreParaPath = os.path.join(scriptAddr, "score_params.json")
with open(scoreParaPath, "r") as f:
    loadedDict = json.load(f)

threeMerLogTable = loadedDict["score_tables"]["3"]

# load multi-scale kmer tables + weights for ensemble score2
kmerTables = {int(k): v for k, v in loadedDict["score_tables"].items()}   # int key for faster lookup
kmerWeights = {int(k): v for k, v in loadedDict["score2_weights"].items()}

# Load trained finished CNN model
AA_seq_list = sorted("ACDEFGHIKLMNPQRSTVWY")
AA_map_idx_dict = {aa: i for i, aa in enumerate(AA_seq_list)}
PADDING_IDX = 20
MAX_LEN = 40

model = PeptideCNN()
score2_model_path = os.path.join(scriptAddr, "peptide_cnn3Btwo.pt")
model.load_state_dict(torch.load(score2_model_path, map_location="cpu", weights_only=True))
model.eval()

# Score functions score1 is 3mer lookuptable, score2 is ensemble CNN + weighted kmer
def calc_score1(peptide):
    """3-mer log2(p/q) sum"""
    total = 0.0
    for i in range(len(peptide) - 2):
        kmer = peptide[i:i+3]  # sliding window k=3
        total += threeMerLogTable.get(kmer, 0.0)
    return total

def encode_peptide(peptide):
    """peptide str to int list, pad to MAX_LEN"""
    enc = [AA_map_idx_dict[c] for c in peptide]
    enc += [PADDING_IDX] * (MAX_LEN - len(enc))  # right side padding
    return enc

# weighted kmer score for single peptide, used in ensemble
def calc_wtkmer(peptide):
    total = 0.0
    for k, w in kmerWeights.items():  # sum all k=2,3,4 weighted scores
        table = kmerTables[k]
        for i in range(len(peptide) - k + 1):
            total += w * table.get(peptide[i:i+k], 0.0)
    return total

def calc_score2_batch(peptides):
    """ensemble: z-norm CNN + z-norm weighted kmer, alpha=0.5"""
    # CNN part
    encoded = [encode_peptide(p) for p in peptides]
    X = torch.LongTensor(encoded)
    with torch.no_grad():
        logits_score = model(X)
        # class 1 = natural, class 0 = random, so diff = confidence
        cnn_raw = (logits_score[:, 1] - logits_score[:, 0]).numpy()

    # weighted kmer part
    kmer_raw = np.array([calc_wtkmer(p) for p in peptides])

    # z-score normalize both, mix 50/50
    def znorm(arr):
        return (arr - arr.mean()) / (arr.std() + 1e-8)

    cnn_z = znorm(cnn_raw)
    kmer_z = znorm(kmer_raw)
    ensemble = (0.5 * cnn_z + 0.5 * kmer_z).tolist()
    return ensemble

# Main IO part loop
peptides = []
for line in sys.stdin:
    line = line.strip()
    if line:
        peptides.append(line)

# batch for faster speed processing
if peptides:
    scores2 = calc_score2_batch(peptides)
    for pep, s2 in zip(peptides, scores2):
        s1 = calc_score1(pep)
        print(f"{s1:.4f} {s2:.4f} {pep}")