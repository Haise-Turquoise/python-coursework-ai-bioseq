#!/usr/bin/env python3

#a3-PREDICT.PY
"""
ASSN 3: Peptide MS/MS Spectrum Prediction - predict script
load weights, run inference on input seqs, write jsonl output
"""
import json
import math
import torch
import torch.nn as nn

# ===== Sequence Encoding Config =====

AA_LIST    = sorted("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX  = {aa: i for i, aa in enumerate(AA_LIST)}
PAD_IDX    = 20
MAX_LEN    = 30
MAX_CHARGE = 6

def encodeSeq(seq):
    enc = [AA_TO_IDX[aa] for aa in seq]
    enc += [PAD_IDX] * (MAX_LEN - len(enc))
    return enc

def encodeCharge(charge):
    return max(0, min(charge - 1, MAX_CHARGE - 1))

# ===== Positional Encoding =====

class PositionalEncoding(nn.Module):
    def __init__(self, dModel, maxLen=MAX_LEN):
        super().__init__()
        pe    = torch.zeros(maxLen, dModel)
        pos   = torch.arange(0, maxLen).unsqueeze(1).float()
        denom = torch.exp(torch.arange(0, dModel, 2).float() * (-math.log(10000.0) / dModel))
        pe[:, 0::2] = torch.sin(pos * denom)
        pe[:, 1::2] = torch.cos(pos * denom)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ===== Transformer Model =====

class SpectrumTransformer(nn.Module):
    def __init__(self,
                 vocabSize  = len(AA_LIST) + 1,
                 dModel     = 256,
                 nHead      = 8,
                 numLayers  = 4,
                 dimFF      = 512,
                 maxCharge  = MAX_CHARGE,
                 dropout    = 0.1):
        super().__init__()
        self.seqEmbed    = nn.Embedding(vocabSize, dModel, padding_idx=PAD_IDX)
        self.chargeEmbed = nn.Embedding(maxCharge, dModel)
        self.posEnc      = PositionalEncoding(dModel)
        self.dropout     = nn.Dropout(dropout)
        encLayer         = nn.TransformerEncoderLayer(
                               d_model=dModel, nhead=nHead,
                               dim_feedforward=dimFF, dropout=dropout,
                               batch_first=True)
        self.encoder     = nn.TransformerEncoder(encLayer, num_layers=numLayers)
        self.bHead = nn.Sequential(nn.Linear(dModel, 64), nn.ReLU(), nn.Linear(64, 1))
        self.yHead = nn.Sequential(nn.Linear(dModel, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, aaIdxSeq, chargeTokenIdx, seqLen):
        batch  = aaIdxSeq.size(0)
        seqEmb = self.posEnc(self.seqEmbed(aaIdxSeq))
        chgEmb = self.chargeEmbed(chargeTokenIdx).unsqueeze(1)
        x      = self.dropout(torch.cat([chgEmb, seqEmb], dim=1))
        padMask = torch.zeros(batch, MAX_LEN + 1, dtype=torch.bool, device=aaIdxSeq.device)
        for i in range(batch):
            padMask[i, seqLen[i] + 1:] = True
        hidden     = self.encoder(x, src_key_padding_mask=padMask)
        siteHidden = hidden[:, 1:MAX_LEN, :]
        out = torch.stack([self.bHead(siteHidden).squeeze(-1),
                           self.yHead(siteHidden).squeeze(-1)], dim=-1)
        return torch.sigmoid(out)


# ===== Inference Entry =====

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("wrong para format, do this please: python a3-predict.py seq_file out_file [param_file]")
        sys.exit(1)
    seqFile   = sys.argv[1]
    outFile   = sys.argv[2]
    paramFile = sys.argv[3] if len(sys.argv) > 3 else 'a3_weights.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load weights
    checkpoint = torch.load(paramFile, map_location=device, weights_only=True)
    model      = SpectrumTransformer().to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"loaded weights from {paramFile}")

    # read seq file, parse each line
    entries = []
    with open(seqFile, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: seq=PEPTIDE,charge=2
            parts  = line.split(',')
            seq    = parts[0].split('=')[1].strip()
            charge = int(parts[1].split('=')[1].strip())
            entries.append((seq, charge))

    # batch forward
    BATCH = 64
    results = []
    for start in range(0, len(entries), BATCH):
        batch    = entries[start:start + BATCH]
        aaIdxSeq   = torch.tensor([encodeSeq(s)       for s, _ in batch], dtype=torch.long).to(device)
        chargeTokenIdx   = torch.tensor([encodeCharge(c)    for _, c in batch], dtype=torch.long).to(device)
        seqLens  = torch.tensor([len(s)             for s, _ in batch], dtype=torch.long).to(device)

        with torch.no_grad():
            pred = model(aaIdxSeq, chargeTokenIdx, seqLens)  # (batch, MAX_LEN-1, 2)

        for i, (seq, charge) in enumerate(batch):
            L     = len(seq)
            sites = []
            for j in range(L - 1):
                b = float(pred[i, j, 0])
                y = float(pred[i, j, 1])
                # threshold 0.01, below -> 0.0
                b = round(b, 6) if b >= 0.01 else 0.0
                y = round(y, 6) if y >= 0.01 else 0.0
                sites.append([b, y])
            results.append({
                'sequence':         seq,
                'precursor_charge': charge,
                'length':           L,
                'sites':            sites
            })

    # write JSONL
    with open(outFile, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"wrote {len(results)} records to {outFile}")