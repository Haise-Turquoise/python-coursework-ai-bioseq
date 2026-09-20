#!/usr/bin/env python3
"""
ASSN 3: Peptide MS/MS Spectrum Prediction - train script
load msp, train transformer model, save weights
"""
import re
import json

# ===== Residue Mass Table (monoisotopic) =====

RESIDUE_MASS = {
    'A': 71.03711,  'R': 156.10111, 'N': 114.04293,
    'D': 115.02694, 'C': 103.00919, 'E': 129.04259,
    'Q': 128.05858, 'G': 57.02146,  'H': 137.05891,
    'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P': 97.05276,
    'S': 87.03203,  'T': 101.04768, 'W': 186.07931,
    'Y': 163.06333, 'V': 99.06841,
}
CAM_MASS   = 103.00919 + 57.02146  # C with CAM modification
PROTON     = 1.00728
WATER      = 18.01056
MZ_TOL     = 0.02                  # matching tolerance in Da

# ===== Name Line Parser =====

def decodeNameStr(nameLine):
    """
    parse Name line, return (seq, charge, modNumRaw, camCount)
    输入格式: SEQUENCE/charge_N(pos,C,CAM)..._energy
    """
    # strip "Name: " prefix
    raw = nameLine.strip()
    if raw.startswith("Name:"):
        raw = raw[5:].strip()

    slashIdx = raw.index('/')
    seq      = raw[:slashIdx]
    after    = raw[slashIdx + 1:]

    # charge is first segment before first underscore
    charge   = int(after.split('_')[0])

    # modNumRaw is numeric prefix of second segment (before any bracket)
    secondSeg = after.split('_')[1]
    modNumRaw  = int(re.match(r'(\d+)', secondSeg).group(1))

    # count actual CAM brackets to verify
    camBrackets = re.findall(r'\(\d+,C,CAM\)', after)
    camCount    = len(camBrackets)

    return seq, charge, modNumRaw, camCount

# ===== Filter Check =====

def filterPassCheck(seq, modNumRaw, camCount):
    """
    return True if record passes all filters
    过滤规则: 长度5-30, C数量==modNumRaw==camCount, 无其他修饰
    """
    seqLen  = len(seq)
    cCount  = seq.count('C')

    if not (5 <= seqLen <= 30):
        return False
    if cCount != modNumRaw:           # mod数量和C个数必须对上
        return False
    if camCount != modNumRaw:         # 所有修饰必须是CAM
        return False
    return True

# ===== b/y Theoretical m/z Calculation =====

def calcTheoMz(seq):
    """
    get b/y ion mz list for charge=1
    返回 (b_mz_list, y_mz_list), 各长度 L-1 / len L-1 each
    """
    L      = len(seq)
    masses = [CAM_MASS if aa == 'C' else RESIDUE_MASS[aa] for aa in seq]

    # prefix sum for b ions / b离子从N端累加
    prefixSum = []
    acc = 0.0
    for m in masses:
        acc += m
        prefixSum.append(acc)

    bMz = [prefixSum[i] + PROTON          for i in range(L - 1)]
    yMz = [prefixSum[L-1] - prefixSum[i] + WATER + PROTON for i in range(L - 1)]

    return bMz, yMz

# ===== Peak Matching =====

def matchIntensity(peakPairList, targetMz):
    """
    find closest peak within MZ_TOL, return intensity or 0.0
    在峰列表里按容差窗口匹配目标mz
    """
    best      = 0.0
    bestDelta = MZ_TOL
    for mz, intensity in peakPairList:
        delta = abs(mz - targetMz)
        if delta < bestDelta:
            bestDelta = delta
            best      = intensity
    return best



# ===== Sequence Encoding Config =====

AA_LIST    = sorted("ACDEFGHIKLMNPQRSTVWY")   # 20 standard amino acids, sorted
AA_TO_IDX  = {aa: i for i, aa in enumerate(AA_LIST)}  # letter -> int index
PAD_IDX    = 20                                # padding token index (outside 0-19)
MAX_LEN    = 30                                # max seq length, 5-30 per a3 rules
MAX_CHARGE = 6                                 # charge range seen in dataset (1-6)

def encodeSeq(seq):
    """
    encode peptide string to fixed-length int tensor, pad to MAX_LEN
    返回长度30的list, 超出部分用PAD_IDX填充
    """
    enc = [AA_TO_IDX[aa] for aa in seq]
    enc += [PAD_IDX] * (MAX_LEN - len(enc))   # right-pad to MAX_LEN
    return enc

def encodeCharge(charge):
    """
    encode precursor charge as 0-indexed int for embedding lookup
    charge 1-6 -> index 0-5, clip anything outside range
    """
    return max(0, min(charge - 1, MAX_CHARGE - 1))

# ===== Transformer Model Definition =====
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    # positional encoding layer, fixed weights / 固定位置编码不训练
    def __init__(self, dModel, maxLen=MAX_LEN):
        super().__init__()
        # build pe table once, not a trained param
        pe    = torch.zeros(maxLen, dModel)
        pos   = torch.arange(0, maxLen).unsqueeze(1).float()
        denom = torch.exp(torch.arange(0, dModel, 2).float() * (-math.log(10000.0) / dModel))
        pe[:, 0::2] = torch.sin(pos * denom)
        pe[:, 1::2] = torch.cos(pos * denom)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, maxLen, dModel)

    def forward(self, x):
        # x shape: (batch, seqLen, dModel)
        return x + self.pe[:, :x.size(1), :]


class SpectrumTransformer(nn.Module):
    """
    transformer model for b/y ion intensity prediction
    input: seq + charge, output: each site [b, y]
    """
    def __init__(self,
                 vocabSize  = len(AA_LIST) + 1,   # 20 AAs + 1 padding
                 dModel     = 256,
                 nHead      = 8,
                 numLayers  = 4,
                 dimFF      = 512,
                 maxCharge  = MAX_CHARGE,
                 dropout    = 0.1):
        super().__init__()

        # sequence token embedding
        self.seqEmbed    = nn.Embedding(vocabSize, dModel, padding_idx=PAD_IDX)

        # charge embedding: encode charge as a single extra token prepended to sequence
        # charge 0-5 -> learned vector of size dModel
        self.chargeEmbed = nn.Embedding(maxCharge, dModel)

        self.posEnc      = PositionalEncoding(dModel)
        self.dropout     = nn.Dropout(dropout)

        # transformer encoder stack
        encLayer         = nn.TransformerEncoderLayer(
                               d_model        = dModel,
                               nhead          = nHead,
                               dim_feedforward= dimFF,
                               dropout        = dropout,
                               batch_first    = True)   # (batch, seq, dim) convention
        self.encoder     = nn.TransformerEncoder(encLayer, num_layers=numLayers)

        # per-site output heads: b and y each get their own 2-layer MLP
        # input is hidden state at each position, output is single intensity float
        self.bHead = nn.Sequential(
            nn.Linear(dModel, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.yHead = nn.Sequential(
            nn.Linear(dModel, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, aaIdxSeq, chargeTokenIdx, seqLen):
        # input: encoded seq + charge idx + actual length, output: per-site b/y intensity
        batch = aaIdxSeq.size(0)

        # embed sequence tokens + add positional encoding
        seqEmb = self.seqEmbed(aaIdxSeq)                    # (batch, MAX_LEN, dModel)
        seqEmb = self.posEnc(seqEmb)

        # prepend charge token as position 0, shift sequence right by 1
        # charge token shape: (batch, 1, dModel)
        chgEmb = self.chargeEmbed(chargeTokenIdx).unsqueeze(1)
        x      = torch.cat([chgEmb, seqEmb], dim=1)      # (batch, MAX_LEN+1, dModel)
        x      = self.dropout(x)

        # build padding mask: True = ignore this position
        # charge token (pos 0) always attended; seq padding positions masked
        padMask = torch.zeros(batch, MAX_LEN + 1, dtype=torch.bool, device=aaIdxSeq.device)
        for i in range(batch):
            # positions after actual seq end should be masked (seqLen+1 because of charge token)
            padMask[i, seqLen[i] + 1:] = True

        # transformer encoder
        hidden = self.encoder(x, src_key_padding_mask=padMask)  # (batch, MAX_LEN+1, dModel)

        # take positions 1..MAX_LEN (skip charge token at position 0)
        # site i prediction uses hidden state at position i (1-indexed)
        siteHidden = hidden[:, 1:MAX_LEN, :]               # (batch, MAX_LEN-1, dModel)

        bOut = self.bHead(siteHidden).squeeze(-1)           # (batch, MAX_LEN-1)
        yOut = self.yHead(siteHidden).squeeze(-1)           # (batch, MAX_LEN-1)

        # stack to (batch, MAX_LEN-1, 2), apply sigmoid to bound output 0-1
        out  = torch.stack([bOut, yOut], dim=-1)
        return torch.sigmoid(out)
    

# ===== B4a: Dataset + DataLoader + Train/Val Split =====

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random

class SpectrumDataset(Dataset):
    # dataset for msp records, returns encoded seq/charge/len + target sites
    def __init__(self, mspRecordList):
        self.mspRecordList = mspRecordList

    def __len__(self):
        return len(self.mspRecordList)

    def __getitem__(self, idx):
        r         = self.mspRecordList[idx]
        seq       = r['seq']
        charge    = r['charge']
        sites     = r['sites']          # list of [b, y], length L-1
        L         = len(seq)

        # encode sequence to fixed-length int list
        aaIdxSeq    = torch.tensor(encodeSeq(seq),      dtype=torch.long)
        chargeTokenIdx = torch.tensor(encodeCharge(charge), dtype=torch.long)
        seqLen    = torch.tensor(L,                    dtype=torch.long)

        # build target tensor shape (MAX_LEN-1, 2), pad beyond L-1 with 0
        target    = torch.zeros(MAX_LEN - 1, 2,        dtype=torch.float)
        for i, (b, y) in enumerate(sites):
            target[i, 0] = b
            target[i, 1] = y

        return aaIdxSeq, chargeTokenIdx, seqLen, target


def splitLoadersTVpair(mspRecordList, valFrac=0.1, batchSize=64, seed=482):
    # split into train/val loaders, same seed = same split
    random.seed(seed)
    torch.manual_seed(seed)

    dataset  = SpectrumDataset(mspRecordList)
    nVal     = int(len(dataset) * valFrac)
    nTrain   = len(dataset) - nVal

    # random_split uses torch generator for reproducibility
    gen      = torch.Generator().manual_seed(seed)
    trainSet, valSet = random_split(dataset, [nTrain, nVal], generator=gen)

    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)
    valLoader   = DataLoader(valSet,   batch_size=batchSize, shuffle=False)

    print(f"train: {nTrain}  val: {nVal}  batch_size: {batchSize}")
    return trainLoader, valLoader

# ===== end B4a =====


# ===== B5: Training Loop =====

def genSiteBoolMask(seqLen, device):
    # make bool mask for valid sites, True = keep, False = skip
    batch   = seqLen.size(0)
    mask    = torch.zeros(batch, MAX_LEN - 1, dtype=torch.bool, device=device)
    for i in range(batch):
        mask[i, :seqLen[i] - 1] = True   # L-1 valid sites for peptide of length L
    return mask


def stepOneEpoch(model, loader, optimizer, device, train=True):
    # one step epoch loop, returns acc + mse over valid sites
    model.train() if train else model.eval()

    bceLoss  = torch.nn.BCELoss(reduction='none')   # per-element, masked manually
    totalBce = 0.0
    totalMse = 0.0
    totalCorrect = 0
    totalSites   = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for aaIdxSeq, chargeTokenIdx, seqLen, target in loader:
            aaIdxSeq    = aaIdxSeq.to(device)
            chargeTokenIdx = chargeTokenIdx.to(device)
            seqLen    = seqLen.to(device)
            target    = target.to(device)          # (batch, MAX_LEN-1, 2)

            pred      = model(aaIdxSeq, chargeTokenIdx, seqLen)  # (batch, MAX_LEN-1, 2)
            mask      = genSiteBoolMask(seqLen, device)     # (batch, MAX_LEN-1)

            # expand mask to cover both b and y channels
            byMaskExpanded   = mask.unsqueeze(-1).expand_as(pred)  # (batch, MAX_LEN-1, 2)

            # BCE loss on binary presence (threshold 0.01)
            targetBin = (target >= 0.01).float()
            bceAll    = bceLoss(pred, targetBin)           # (batch, MAX_LEN-1, 2)
            bceMasked = (bceAll * byMaskExpanded).sum() / byMaskExpanded.sum()

            # MSE loss on raw intensity including zeros
            mseAll    = ((pred - target) ** 2)
            mseMasked = (mseAll * byMaskExpanded).sum() / byMaskExpanded.sum()

            # combined loss: BCE + MSE at nonzero positions (lec11 dual loss)
            presentIonMask = byMaskExpanded & (target >= 0.01)
            mseNonzero  = ((pred - target) ** 2 * presentIonMask).sum() / (presentIonMask.sum() + 1e-8)
            loss        = bceMasked + mseNonzero

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # accuracy: correct binary classification over valid sites
            predBin      = (pred >= 0.5).float()
            correct      = ((predBin == targetBin) * byMaskExpanded).sum().item()
            totalCorrect += correct
            totalSites   += byMaskExpanded.sum().item()
            totalBce     += bceMasked.item()
            totalMse     += mseMasked.item()

    acc = totalCorrect / (totalSites + 1e-8)
    mse = totalMse / len(loader)
    return acc, mse


def trainModel(mspRecordList, paramFile, numEpoch=20, batchSize=64, lr=1e-3, seed=2819):
    # run full training, print metrics each epoch, save to paramFile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    trainLoader, valLoader = splitLoadersTVpair(mspRecordList, batchSize=batchSize, seed=seed)

    model     = SpectrumTransformer().to(device)
    paramCount = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {paramCount:,}")
    assert paramCount <= 4_000_000, f"exceeds 4M limit"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in range(1, numEpoch + 1):
        trainAcc, trainMse = stepOneEpoch(model, trainLoader, optimizer, device, train=True)
        valAcc,   valMse   = stepOneEpoch(model, valLoader,   optimizer, device, train=False)
        scheduler.step(valMse)
        print(f"epoch {epoch:02d}  "
              f"train acc={trainAcc:.4f} mse={trainMse:.4f}  "
              f"val   acc={valAcc:.4f}  mse={valMse:.4f}")

    # save weights + config to paramFile
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'dModel':    256,
            'nHead':     8,
            'numLayers': 4,
            'dimFF':     512,
        }
    }, paramFile)
    print(f"saved: {paramFile}")

# ===== end B5 =====


# ===== MSP Parser (main entry) =====

def readMspRecordsToList(filepath):
    """
    parse full MSP file, return list of filtered mspRecordList
    每条记录: {seq, charge, sites: [[b1,y1],[b2,y2],...]}
    """
    mspRecordList = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # state: IDLE -> HEADER -> PEAKS
    state     = 'IDLE'
    curSeq    = ''
    curCharge = 0
    curMod    = 0
    curCam    = 0
    peakLines = 0
    peakPairList  = []

    for line in lines:
        line = line.rstrip('\n').rstrip('\r')

        if state == 'IDLE':
            if line.startswith('Name:'):
                curSeq, curCharge, curMod, curCam = decodeNameStr(line)
                state = 'HEADER'

        elif state == 'HEADER':
            if line.startswith('Num peaks:'):
                peakLines = int(line.split(':')[1].strip())
                peakPairList  = []
                state     = 'PEAKS'
            # MW / Comment lines: skip silently

        elif state == 'PEAKS':
            if peakLines > 0:
                parts = line.split('\t')
                if len(parts) >= 2:
                    peakPairList.append((float(parts[0]), float(parts[1])))
                peakLines -= 1

            if peakLines == 0:
                # record complete, run filter + extraction
                if filterPassCheck(curSeq, curMod, curCam):
                    bMz, yMz = calcTheoMz(curSeq)
                    L        = len(curSeq)
                    sites    = []
                    for i in range(L - 1):
                        bInt = matchIntensity(peakPairList, bMz[i])
                        yInt = matchIntensity(peakPairList, yMz[i])
                        sites.append([bInt, yInt])

                    # normalize by max charge-1 b/y intensity
                    allInts = [v for pair in sites for v in pair]
                    maxInt  = max(allInts) if max(allInts) > 0 else 1.0
                    sites   = [[round(b/maxInt, 6) if b/maxInt >= 0.01 else 0.0,
                                 round(y/maxInt, 6) if y/maxInt >= 0.01 else 0.0]
                                for b, y in sites]

                    mspRecordList.append({
                        'seq':    curSeq,
                        'charge': curCharge,
                        'sites':  sites
                    })

                state = 'IDLE'

    return mspRecordList









# ===== Main Entry =====

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('msp_file')
    parser.add_argument('param_file')
    parser.add_argument('--epoch', type=int, default=20)
    args = parser.parse_args()
    recs = readMspRecordsToList(args.msp_file)
    trainModel(recs, args.param_file, numEpoch=args.epoch)