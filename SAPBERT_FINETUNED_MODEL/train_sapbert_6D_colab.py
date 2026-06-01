# -*- coding: utf-8 -*-

# Fine-Tuning SapBERT v5g -- Clasificador ESI (6D, sin cross-features)
# Script de entrenamiento ejecutado en Google Colab con GPU T4.
# El modelo entrenado se guarda como: modelo_sapbert_finetuned_v5g_6D_ablation.pt

# Arquitectura:
#   Modelo base: SapBERT (cambridgeltl/SapBERT-from-PubMedBERT-fulltext)
#   Fusion: Cross-Attention text-dominant (4 heads, 128 dim)
#   Concatenacion: [CLS_768, text_proj_128, attended_128] -> 1024D
#   Vector clinico: 6 features (pain, temperature, heartrate,
#   resprate, age, gender_M) sin cross-features
#   Loss: 0.55*CE + 0.20*EMD + 0.10*Binary + 0.15*Contrastive_UMLS

# -- Dependencias
!pip install -q transformers accelerate scikit-learn seaborn

import torch
print(f"GPU disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Dispositivo: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    raise RuntimeError("Se requiere GPU. Runtime -> Change runtime type -> T4 GPU")

from google.colab import files
import os

for fname in ["triage_with_demographics.csv", "umls_symptoms.csv"]:
    if not os.path.exists(fname):
        print(f"Sube el archivo: {fname}")
        files.upload()
    else:
        print(f"Archivo encontrado: {fname}")


# 1. Limpieza de datos

import pandas as pd
import numpy as np
import re

def normalize_chiefcomplaint(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text.strip()

print("Cargando triage_with_demographics.csv...")
df = pd.read_csv("triage_with_demographics.csv", low_memory=False)
print(f"  {len(df):,} filas cargadas")

cols_numericas = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity", "age"]
for col in cols_numericas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["acuity"])
df["acuity"] = df["acuity"].astype(int)
df = df[df["acuity"].between(1, 5)]

df = df.dropna(subset=["chiefcomplaint"])
df["chiefcomplaint"] = df["chiefcomplaint"].apply(normalize_chiefcomplaint)
df = df[df["chiefcomplaint"].str.len() > 0]

# Rangos fisiologicos validos -- solo para las 4 variables que usa el modelo
RANGOS_VALIDOS = {
    "pain":        (0.0,  10.0),
    "temperature": (95.0, 107.0),
    "heartrate":   (20.0, 300.0),
    "resprate":    (4.0,  60.0),
}

for col, (vmin, vmax) in RANGOS_VALIDOS.items():
    if col in df.columns:
        mask = df[col].notna() & ~df[col].between(vmin, vmax)
        if mask.sum() > 0:
            df = df[~mask]
            print(f"  {col}: {mask.sum():,} filas con valores fuera de rango eliminadas")

vitals = ["pain", "temperature", "heartrate", "resprate"]
n_before = len(df)
any_null = df[vitals].isna().any(axis=1)
df = df[~any_null]
print(f"  Eliminadas {n_before - len(df):,} filas con vitales nulas")

df["age"] = df["age"].fillna(df["age"].median()).clip(18, 91)
df["gender_M"] = df["gender"].map({"M": 1.0, "F": 0.0}).fillna(0.5)

print(f"  Dataset final: {len(df):,} casos")
print("\nDistribucion ESI:")
for level in range(1, 6):
    n = (df["acuity"] == level).sum()
    print(f"  ESI {level}: {n:7,} ({n/len(df)*100:5.1f}%)")


# 2. Preparar datos MIMIC + pares UMLS contrastivos

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

texts = df["chiefcomplaint"].values
labels = df["acuity"].values - 1

clin_cols = ['pain', 'temperature', 'heartrate', 'resprate', 'age', 'gender_M']
clin_raw = df[clin_cols].values.astype(np.float32)

idx = np.arange(len(labels))
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
idx_train, idx_val = train_test_split(idx_train, test_size=0.1, random_state=42, stratify=labels[idx_train])

print(f"Train: {len(idx_train):,}  Val: {len(idx_val):,}  Test: {len(idx_test):,}")

scaler_clin = StandardScaler()
clin_scaled = np.zeros_like(clin_raw)
clin_scaled[idx_train] = scaler_clin.fit_transform(clin_raw[idx_train])
clin_scaled[idx_val]   = scaler_clin.transform(clin_raw[idx_val])
clin_scaled[idx_test]  = scaler_clin.transform(clin_raw[idx_test])

# v5g usa solo las 6 features base, sin cross-features
clin_crosses = clin_scaled.copy()
CLIN_DIM = clin_crosses.shape[1]
print(f"Dimension vector clinico: {CLIN_DIM}D")

# Pares contrastivos UMLS
print("\nCargando umls_symptoms.csv para entrenamiento contrastivo...")
df_umls = pd.read_csv("umls_symptoms.csv")
df_umls = df_umls[df_umls['synonyms'].notna() & (df_umls['n_synonyms'] >= 2)]
print(f"  Conceptos con >=2 sinonimos: {len(df_umls):,}")

umls_groups = []
for _, row in df_umls.iterrows():
    texts_cui = [row['canonical_name'].strip()]
    for syn in str(row['synonyms']).split('|'):
        syn = syn.strip()
        if syn and len(syn) >= 2:
            texts_cui.append(syn)
    if len(texts_cui) >= 2:
        umls_groups.append(texts_cui)

total_texts = sum(len(g) for g in umls_groups)
total_pairs = sum(len(g) * (len(g) - 1) // 2 for g in umls_groups)
print(f"  Grupos CUI: {len(umls_groups):,}")
print(f"  Total textos: {total_texts:,}")
print(f"  Pares posibles: {total_pairs:,}")
print(f"  Pares por epoca (sampling online): configurado en seccion 3")


# 3. Modelo, Datasets y DataLoaders

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import random

SAPBERT_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


class SapBERTCrossAttentionModel(nn.Module):
    """
    SapBERT con fusion por Cross-Attention para clasificacion ESI.

    El vector clinico actua como query y atiende al embedding de texto (key/value).
    La fusion es text-dominant: concatena [CLS_768, text_proj_128, attended_128] -> 1024D.
    El entrenamiento incluye una loss contrastiva sobre sinonimos UMLS para mantener
    la alineacion semantica del modelo base durante el fine-tuning.
    """
    def __init__(self, clin_dim=6, dropout=0.3, n_heads=4, attn_dim=128):
        super().__init__()
        self.bert = AutoModel.from_pretrained(SAPBERT_NAME)
        bert_dim = 768

        self.clin_project = nn.Sequential(
            nn.Linear(clin_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_project = nn.Sequential(
            nn.Linear(bert_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(attn_dim)
        self.cls_norm  = nn.LayerNorm(bert_dim)

        fusion_dim = bert_dim + attn_dim * 2  # 768 + 128 + 128 = 1024
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_esi    = nn.Linear(128, 5)
        self.head_binary = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask, clinical):
        outputs  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb  = outputs.last_hidden_state[:, 0, :]
        text_proj = self.text_project(cls_emb)
        clin_proj = self.clin_project(clinical)
        text_seq  = text_proj.unsqueeze(1)
        clin_seq  = clin_proj.unsqueeze(1)
        attended, _ = self.cross_attn(query=clin_seq, key=text_seq, value=text_seq)
        attended  = self.attn_norm(attended.squeeze(1) + clin_proj)
        cls_normed = self.cls_norm(cls_emb)
        fused = torch.cat([cls_normed, text_proj, attended], dim=1)
        h = self.fusion(fused)
        return self.head_esi(h), self.head_binary(h)

    def encode_text(self, input_ids, attention_mask):
        """Devuelve el embedding CLS del texto (usado en la loss contrastiva)."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]


class TriageTextDataset(Dataset):
    """Dataset MIMIC para clasificacion ESI."""
    def __init__(self, texts, clinical, labels, tokenizer, max_length=64):
        self.texts    = texts
        self.clinical = torch.tensor(clinical, dtype=torch.float32)
        self.labels   = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx], padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids':      encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'clinical':       self.clinical[idx],
            'labels':         self.labels[idx],
        }


class UMLSPairDataset(Dataset):
    """
    Dataset de pares de sinonimos UMLS para entrenamiento contrastivo.
    Cada item devuelve dos textos del mismo CUI, muestreados aleatoriamente.
    Los pares se regeneran al inicio de cada epoca.
    """
    def __init__(self, umls_groups, tokenizer, max_length=64, pairs_per_epoch=20000):
        self.umls_groups     = umls_groups
        self.tokenizer       = tokenizer
        self.max_length      = max_length
        self.pairs_per_epoch = pairs_per_epoch
        # Grupos mas grandes tienen mayor probabilidad de ser muestreados
        self.weights = np.array([len(g) for g in umls_groups], dtype=np.float32)
        self.weights /= self.weights.sum()
        self._resample()

    def _resample(self):
        """Genera nuevos pares aleatorios para la epoca actual."""
        self.pairs = []
        group_indices = np.random.choice(
            len(self.umls_groups), size=self.pairs_per_epoch,
            p=self.weights, replace=True)
        for gi in group_indices:
            group = self.umls_groups[gi]
            a, b = random.sample(range(len(group)), 2)
            self.pairs.append((group[a], group[b]))

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, idx):
        text_a, text_b = self.pairs[idx]
        enc_a = self.tokenizer(
            text_a, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt')
        enc_b = self.tokenizer(
            text_b, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids_a':      enc_a['input_ids'].squeeze(0),
            'attention_mask_a': enc_a['attention_mask'].squeeze(0),
            'input_ids_b':      enc_b['input_ids'].squeeze(0),
            'attention_mask_b': enc_b['attention_mask'].squeeze(0),
        }


# Instanciar tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(SAPBERT_NAME)
model     = SapBERTCrossAttentionModel(clin_dim=CLIN_DIM, dropout=0.3).cuda()

total_params = sum(p.numel() for p in model.parameters())
print(f"Parametros totales: {total_params:,}")
print(f"Vector clinico: {CLIN_DIM}D")
print(f"Fusion: 768 + 128 + 128 = 1024D")

# DataLoaders MIMIC
train_ds = TriageTextDataset(texts[idx_train], clin_crosses[idx_train], labels[idx_train], tokenizer)
val_ds   = TriageTextDataset(texts[idx_val],   clin_crosses[idx_val],   labels[idx_val],   tokenizer)
test_ds  = TriageTextDataset(texts[idx_test],  clin_crosses[idx_test],  labels[idx_test],  tokenizer)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=64,                num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=64,                num_workers=2, pin_memory=True)

# DataLoader UMLS contrastivo
PAIRS_PER_EPOCH = 20000
umls_ds = UMLSPairDataset(umls_groups, tokenizer, pairs_per_epoch=PAIRS_PER_EPOCH)
umls_dl = DataLoader(umls_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

print(f"\nMIMIC  -- Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
print(f"UMLS   -- {PAIRS_PER_EPOCH:,} pares/epoca de {len(umls_groups):,} conceptos")
print(f"Batches -- MIMIC: {len(train_dl):,} | UMLS: {len(umls_dl):,}")


# 4. Loss multi-objetivo + Optimizer

# Loss total: 0.55*CE + 0.20*EMD + 0.10*Binary + 0.15*Contrastive
# - CE: Cross-entropy ponderada por clase
# - EMD: Earth Mover's Distance sobre CDFs (penaliza errores ordinales)
# - Binary: Cross-entropy urgente (ESI 1-2) vs no urgente (ESI 3-5)
# - Contrastive: Cosine similarity loss sobre pares de sinonimos UMLS
#                Mantiene la alineacion semantica durante el fine-tuning

from transformers import get_linear_schedule_with_warmup


class MultiTaskOrdinalLoss(nn.Module):
    """Loss compuesta para clasificacion ESI (CE + EMD + Binary)."""
    def __init__(self, n_classes, class_weights):
        super().__init__()
        self.n_classes    = n_classes
        self.class_weights = class_weights
        self.ce        = nn.CrossEntropyLoss(weight=class_weights)
        self.binary_ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).cuda())

    def forward(self, logits_esi, logits_binary, targets):
        ce_loss = self.ce(logits_esi, targets)

        # EMD: penaliza segun distancia ordinal entre prediccion y etiqueta real
        probs      = F.softmax(logits_esi, dim=1)
        targets_oh = F.one_hot(targets, self.n_classes).float()
        cdf_pred   = torch.cumsum(probs, dim=1)
        cdf_target = torch.cumsum(targets_oh, dim=1)
        emd        = torch.sum((cdf_pred - cdf_target) ** 2, dim=1)
        emd_loss   = (emd * self.class_weights[targets]).mean()

        # Binary: clasificacion urgente vs no urgente
        binary_targets = (targets <= 1).long()
        binary_loss    = self.binary_ce(logits_binary, binary_targets)

        return 0.55 * ce_loss + 0.20 * emd_loss + 0.10 * binary_loss


def contrastive_loss(emb_a, emb_b):
    """
    Loss contrastiva por similitud coseno.
    Minimiza la distancia entre embeddings de sinonimos del mismo CUI.
    """
    emb_a  = F.normalize(emb_a, p=2, dim=1)
    emb_b  = F.normalize(emb_b, p=2, dim=1)
    cos_sim = (emb_a * emb_b).sum(dim=1)
    return (1.0 - cos_sim).mean()


# Pesos de clase (inversamente proporcionales a frecuencia en MIMIC)
CLASS_WEIGHTS     = torch.tensor([5.0, 1.5, 1.0, 3.0, 12.0], dtype=torch.float32).cuda()
criterion_esi     = MultiTaskOrdinalLoss(5, CLASS_WEIGHTS)
CONTRASTIVE_WEIGHT = 0.15

optimizer = torch.optim.AdamW([
    {"params": model.bert.parameters(),        "lr": 2e-5,  "weight_decay": 0.01},
    {"params": model.clin_project.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
    {"params": model.text_project.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
    {"params": model.cross_attn.parameters(),   "lr": 1e-3, "weight_decay": 1e-4},
    {"params": model.attn_norm.parameters(),    "lr": 1e-3, "weight_decay": 1e-4},
    {"params": model.fusion.parameters(),       "lr": 1e-3, "weight_decay": 1e-4},
    {"params": model.head_esi.parameters(),     "lr": 1e-3, "weight_decay": 1e-4},
    {"params": model.head_binary.parameters(),  "lr": 1e-3, "weight_decay": 1e-4},
])

NUM_EPOCHS    = 5
total_steps   = len(train_dl) * NUM_EPOCHS
warmup_steps  = len(train_dl)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

print(f"Epocas: {NUM_EPOCHS} | Warmup: {warmup_steps} steps")
print(f"Loss ESI: 0.55*CE + 0.20*EMD + 0.10*Binary")
print(f"Loss total: ESI_loss + {CONTRASTIVE_WEIGHT}*Contrastive")
print(f"Pesos de clase: {CLASS_WEIGHTS.cpu().tolist()}")


# 5. Entrenamiento

# Cada paso de entrenamiento:
# 1. Forward sobre batch MIMIC -> loss ESI (CE + EMD + Binary)
# 2. Cada 3 batches: forward sobre par UMLS -> loss contrastiva
# 3. Loss total = ESI_loss + 0.15 * contrastive_loss
# 4. Backward + optimizer step

from sklearn.metrics import roc_auc_score
import time

scaler_amp  = torch.amp.GradScaler('cuda')
best_val_auc = 0.0
best_state   = None

print("=" * 70)
print("Entrenamiento SapBERT v5g -- Multi-Objetivo (ESI + Contrastive UMLS)")
print("=" * 70)

t0_total = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    t0_epoch = time.time()
    model.train()
    epoch_loss_esi, epoch_loss_ctr, n_batches = 0.0, 0.0, 0

    umls_ds._resample()
    umls_iter = iter(umls_dl)

    for batch_idx, batch in enumerate(train_dl):
        input_ids      = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        clinical       = batch["clinical"].cuda()
        labels_batch   = batch["labels"].cuda()

        optimizer.zero_grad()

        # Loss ESI sobre batch MIMIC
        with torch.amp.autocast('cuda'):
            logits_esi, logits_binary = model(input_ids, attention_mask, clinical)
            loss_esi = criterion_esi(logits_esi, logits_binary, labels_batch)

        # Loss contrastiva sobre pares UMLS (cada 3 batches)
        loss_ctr = torch.tensor(0.0, device='cuda')
        if (batch_idx + 1) % 3 == 0:
            try:
                umls_batch = next(umls_iter)
            except StopIteration:
                umls_iter  = iter(umls_dl)
                umls_batch = next(umls_iter)

            with torch.amp.autocast('cuda'):
                ids_a  = umls_batch['input_ids_a'].cuda()
                mask_a = umls_batch['attention_mask_a'].cuda()
                ids_b  = umls_batch['input_ids_b'].cuda()
                mask_b = umls_batch['attention_mask_b'].cuda()
                emb_a  = model.encode_text(ids_a, mask_a)
                emb_b  = model.encode_text(ids_b, mask_b)
                loss_ctr = contrastive_loss(emb_a, emb_b)

        with torch.amp.autocast('cuda'):
            total_loss = loss_esi + CONTRASTIVE_WEIGHT * loss_ctr

        scaler_amp.scale(total_loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        scheduler.step()

        epoch_loss_esi += loss_esi.item()
        epoch_loss_ctr += loss_ctr.item()
        n_batches      += 1

        if (batch_idx + 1) % 500 == 0:
            elapsed  = time.time() - t0_epoch
            pct      = (batch_idx + 1) / len(train_dl) * 100
            eta      = elapsed / (batch_idx + 1) * (len(train_dl) - batch_idx - 1)
            avg_esi  = epoch_loss_esi / n_batches
            avg_ctr  = epoch_loss_ctr / n_batches
            print(f"  Epoca {epoch} | Batch {batch_idx+1}/{len(train_dl)} ({pct:.0f}%) | "
                  f"ESI: {avg_esi:.4f} | CTR: {avg_ctr:.4f} | ETA: {eta/60:.1f}min")

    avg_loss_esi = epoch_loss_esi / n_batches
    avg_loss_ctr = epoch_loss_ctr / n_batches

    # Validacion
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in val_dl:
            input_ids      = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            clinical       = batch["clinical"].cuda()
            with torch.amp.autocast('cuda'):
                logits_esi, _ = model(input_ids, attention_mask, clinical)
            probs = F.softmax(logits_esi.float(), dim=1)
            all_preds.append(logits_esi.argmax(1).cpu())
            all_labels.append(batch["labels"])
            all_probs.append(probs.cpu())

    val_preds  = torch.cat(all_preds).numpy()
    val_labels = torch.cat(all_labels).numpy()
    val_probs  = torch.cat(all_probs).numpy()
    val_acc    = (val_preds == val_labels).mean()
    val_auc    = roc_auc_score(val_labels, val_probs, multi_class="ovr", average="macro")

    epoch_time = time.time() - t0_epoch
    improved   = val_auc > best_val_auc
    marker     = " <- mejor hasta ahora" if improved else ""
    print(f"\n  Epoca {epoch}/{NUM_EPOCHS} | ESI loss: {avg_loss_esi:.4f} | CTR loss: {avg_loss_ctr:.4f} | "
          f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | {epoch_time/60:.1f}min{marker}\n")

    if improved:
        best_val_auc = val_auc
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

total_time = time.time() - t0_total


# 6. Evaluacion en test

from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt

model.load_state_dict(best_state)
model.eval().cuda()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for batch in test_dl:
        input_ids      = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        clinical       = batch["clinical"].cuda()
        with torch.amp.autocast('cuda'):
            logits_esi, _ = model(input_ids, attention_mask, clinical)
        probs = F.softmax(logits_esi.float(), dim=1)
        all_preds.append(logits_esi.argmax(1).cpu())
        all_labels.append(batch["labels"])
        all_probs.append(probs.cpu())

preds      = torch.cat(all_preds).numpy()
y_test     = torch.cat(all_labels).numpy()
probs_test = torch.cat(all_probs).numpy()

target_names = [f"ESI {i+1}" for i in range(5)]
print("Classification Report:")
print(classification_report(y_test, preds, target_names=target_names, digits=4))

accuracy  = (preds == y_test).mean()
auc_macro = roc_auc_score(y_test, probs_test, multi_class="ovr", average="macro")
kappa_q   = cohen_kappa_score(y_test, preds, weights="quadratic")

print(f"Accuracy:         {accuracy:.4f}")
print(f"AUC Macro:        {auc_macro:.4f}")
print(f"Kappa cuadratico: {kappa_q:.4f}")

diff  = np.abs(preds - y_test)
exact = (diff == 0).sum()
adj   = (diff == 1).sum()
grave = (diff > 1).sum()
total = len(y_test)
undertriage = ((y_test <= 1) & (preds >= 3)).sum()

print(f"\nErrores:")
print(f"  Exactos:      {exact:,} ({exact/total*100:.1f}%)")
print(f"  Adyacentes:   {adj:,} ({adj/total*100:.1f}%)")
print(f"  Graves (>1):  {grave:,} ({grave/total*100:.1f}%)")
print(f"  Undertriage:  {undertriage}")

cm     = confusion_matrix(y_test, preds)
cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_pct, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title(f"SapBERT v5g -- Acc: {accuracy:.3f}, AUC: {auc_macro:.3f}")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.show()


# 7. Verificacion de alineacion de sinonimos (contrastive loss)

# Comprueba que el modelo mantiene proximidad entre expresiones coloquiales
# y sus equivalentes clinicos formales tras el fine-tuning.

test_pairs = [
    ("the runs",       "diarrhea"),
    ("trouble peeing", "difficulty urinating"),
    ("belly hurts",    "abdominal pain"),
    ("throwing up",    "vomiting"),
    ("feeling woozy",  "dizziness"),
    ("stuffed up nose","nasal congestion"),
    ("chest hurts",    "chest pain"),
    ("can't breathe",  "difficulty breathing"),
    ("seeing spots",   "visual disturbance"),
    ("heart racing",   "tachycardia"),
]

model.eval()
print("Similitud coseno: expresiones coloquiales vs. formales")
print(f"{'Coloquial':<25} {'Formal':<25} {'Similitud':>10}")
print(f"{'-'*62}")

with torch.no_grad():
    for colloquial, formal in test_pairs:
        enc_c = tokenizer(colloquial, padding='max_length', truncation=True,
                          max_length=64, return_tensors='pt')
        enc_f = tokenizer(formal,     padding='max_length', truncation=True,
                          max_length=64, return_tensors='pt')
        emb_c = model.encode_text(enc_c['input_ids'].cuda(), enc_c['attention_mask'].cuda())
        emb_f = model.encode_text(enc_f['input_ids'].cuda(), enc_f['attention_mask'].cuda())
        emb_c = F.normalize(emb_c, p=2, dim=1)
        emb_f = F.normalize(emb_f, p=2, dim=1)
        sim   = (emb_c * emb_f).sum().item()
        print(f"{colloquial:<25} {formal:<25} {sim:>10.4f}")

print(f"\nReferencia pre-fine-tuning (v5d): similitud 'the runs' / 'diarrhea' ~ 0.35")
print(f"Con contrastive loss activa se espera similitud > 0.60 en estos pares.")


# 8. Guardado del modelo

import json

torch.save({
    "model_state":    best_state,
    "model_class":    "SapBERTCrossAttentionModel",
    "sapbert_name":   SAPBERT_NAME,
    "clin_dim":       CLIN_DIM,
    "dropout":        0.3,
    "n_heads":        4,
    "attn_dim":       128,
    "scaler_clin":    scaler_clin,
    "test_accuracy":  float(accuracy),
    "test_auc_macro": float(auc_macro),
    "class_weights":  CLASS_WEIGHTS.cpu().tolist(),
    "max_length":     64,
    "feature_order":  ['pain', 'temperature', 'heartrate', 'resprate', 'age', 'gender_M'],
    "training_config": {
        "epochs":                   NUM_EPOCHS,
        "lr_bert":                  2e-5,
        "lr_head":                  1e-3,
        "loss":                     "0.55*CE + 0.20*EMD + 0.10*Binary + 0.15*Contrastive",
        "contrastive_pairs_per_epoch": PAIRS_PER_EPOCH,
        "contrastive_concepts":     len(umls_groups),
        "clin_features":            6,
        "crosses_total":            0,
        "fusion":                   "cross-attention (4 heads, 128 dim) + text-dominant",
        "dataset":                  "MIMIC-IV-ED limpio, 6D sin cross-features",
    }
}, "modelo_sapbert_finetuned_v5g_6D_ablation.pt")
print("\nModelo guardado: modelo_sapbert_finetuned_v5g_6D_ablation.pt")

metrics = {
    "accuracy":          float(accuracy),
    "auc_macro":         float(auc_macro),
    "kappa_quadratic":   float(kappa_q),
    "exact":             int(exact),
    "adjacent":          int(adj),
    "grave":             int(grave),
    "undertriage":       int(undertriage),
    "dataset_size":      len(labels),
    "contrastive_pairs": PAIRS_PER_EPOCH,
    "total_time_min":    total_time / 60,
}
with open("metrics_finetuned_v5g_6D_ablation.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metricas guardadas: metrics_finetuned_v5g_6D_ablation.json")

from google.colab import files
files.download("modelo_sapbert_finetuned_v5g_6D_ablation.pt")
files.download("metrics_finetuned_v5g_6D_ablation.json")
