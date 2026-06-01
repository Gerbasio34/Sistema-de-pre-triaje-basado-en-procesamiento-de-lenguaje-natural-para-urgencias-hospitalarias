# -*- coding: utf-8 -*-
# Evaluación de separabilidad lineal de embeddings SapBERT base vs fine-tuned
# sobre el test set completo de MIMIC-IV-ED (~74.000 casos)
#
# Pregunta: ¿Son los embeddings linealmente separables por nivel ESI?
# Esto determina si el colapso geométrico observado es funcional o solo geométrico.
#
# Ejecutar desde la carpeta que contiene:
#   - triage_with_demographics.csv
#   - modelo_sapbert_finetuned_v5g_6D_ablation.pt

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import time
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

SAPBERT_NAME    = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
CLASSIFIER_PATH = "../SAPBERT_FINETUNED_MODEL/modelo_sapbert_finetuned_v5g_6D_ablation.pt"
CSV_PATH        = "triage_with_demographics.csv"
BATCH_SIZE      = 128   # grande para aprovechar la RTX 4060
MAX_LENGTH      = 64
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}\n")


# ── Arquitectura idéntica al entrenamiento ──

class SapBERTCrossAttentionModel(nn.Module):
    def __init__(self, clin_dim=6, dropout=0.3, n_heads=4, attn_dim=128):
        super().__init__()
        self.bert = AutoModel.from_pretrained(SAPBERT_NAME)
        bert_dim  = 768
        self.clin_project = nn.Sequential(
            nn.Linear(clin_dim, attn_dim), nn.LayerNorm(attn_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.text_project = nn.Sequential(
            nn.Linear(bert_dim, attn_dim), nn.LayerNorm(attn_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True)
        self.attn_norm  = nn.LayerNorm(attn_dim)
        self.cls_norm   = nn.LayerNorm(bert_dim)
        fusion_dim = bert_dim + attn_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout))
        self.head_esi    = nn.Linear(128, 5)
        self.head_binary = nn.Linear(128, 2)

    def encode_text(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]


# ── Dataset mínimo para extracción de embeddings ──

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], padding='max_length', truncation=True,
            max_length=MAX_LENGTH, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          self.labels[idx],
        }


# ── 1. Reproducir el preprocesado y split exactos del entrenamiento ──

print("="*65)
print("  PASO 1 — Cargando y preprocesando MIMIC-IV-ED")
print("="*65)

def normalize_chiefcomplaint(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text.strip()

df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"  Filas cargadas: {len(df):,}")

cols_numericas = ["temperature","heartrate","resprate","o2sat","sbp","dbp",
                  "pain","acuity","age"]
for col in cols_numericas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["acuity"])
df["acuity"] = df["acuity"].astype(int)
df = df[df["acuity"].between(1, 5)]
df = df.dropna(subset=["chiefcomplaint"])
df["chiefcomplaint"] = df["chiefcomplaint"].apply(normalize_chiefcomplaint)
df = df[df["chiefcomplaint"].str.len() > 0]

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

vitals = ["pain", "temperature", "heartrate", "resprate"]
df = df[~df[vitals].isna().any(axis=1)]
df["age"]      = df["age"].fillna(df["age"].median()).clip(18, 91)
df["gender_M"] = df["gender"].map({"M": 1.0, "F": 0.0}).fillna(0.5)

print(f"  Dataset final: {len(df):,} casos")

# Split idéntico al entrenamiento
texts  = df["chiefcomplaint"].values
labels = df["acuity"].values - 1  # 0-4

idx = np.arange(len(labels))
idx_train, idx_test = train_test_split(
    idx, test_size=0.2, random_state=42, stratify=labels)
idx_train, idx_val = train_test_split(
    idx_train, test_size=0.1, random_state=42, stratify=labels[idx_train])

print(f"  Train: {len(idx_train):,} | Val: {len(idx_val):,} | Test: {len(idx_test):,}")


# ── 2. Cargar modelos ──

print("\n" + "="*65)
print("  PASO 2 — Cargando modelos")
print("="*65)

tokenizer = AutoTokenizer.from_pretrained(SAPBERT_NAME)

print("  Cargando SapBERT BASE...")
base_bert = AutoModel.from_pretrained(SAPBERT_NAME).to(device).eval()

print("  Cargando SapBERT FINE-TUNED...")
checkpoint = torch.load(CLASSIFIER_PATH, map_location=device, weights_only=False)
ft_model   = SapBERTCrossAttentionModel(
    clin_dim=checkpoint['clin_dim'],
    dropout=checkpoint.get('dropout', 0.3),
    n_heads=checkpoint.get('n_heads', 4),
    attn_dim=checkpoint.get('attn_dim', 128),
)
ft_model.load_state_dict(checkpoint['model_state'])
ft_model.to(device).eval()
print("  Modelos cargados.\n")


# ── 3. Extracción de embeddings ──

def extract_embeddings(model_type, texts_subset, labels_subset):
    """Extrae embeddings CLS para un subset de textos."""
    ds = TextDataset(texts_subset, labels_subset, tokenizer)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=4, pin_memory=True)

    all_embs, all_labels = [], []
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(dl):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)

            if model_type == 'base':
                out  = base_bert(input_ids=ids, attention_mask=mask)
                embs = out.last_hidden_state[:, 0, :]
            else:
                embs = ft_model.encode_text(ids, mask)

            all_embs.append(embs.cpu().float().numpy())
            all_labels.append(batch['label'].numpy())

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                pct     = (i + 1) / len(dl) * 100
                eta     = elapsed / (i + 1) * (len(dl) - i - 1)
                print(f"    Batch {i+1}/{len(dl)} ({pct:.0f}%) | "
                      f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")

    return np.concatenate(all_embs), np.concatenate(all_labels)


print("="*65)
print("  PASO 3 — Extrayendo embeddings del test set completo")
print(f"  ({len(idx_test):,} casos × 768D × 2 modelos)")
print("="*65)

texts_test  = texts[idx_test]
labels_test = labels[idx_test]

# También necesitamos train para entrenar la regresión logística
texts_train  = texts[idx_train]
labels_train = labels[idx_train]

print("\n  [BASE] Test set...")
X_test_base, y_test = extract_embeddings('base', texts_test, labels_test)
print(f"  → Shape: {X_test_base.shape}")

print("\n  [BASE] Train set...")
X_train_base, y_train = extract_embeddings('base', texts_train, labels_train)
print(f"  → Shape: {X_train_base.shape}")

print("\n  [FINE-TUNED] Test set...")
X_test_ft, _ = extract_embeddings('ft', texts_test, labels_test)
print(f"  → Shape: {X_test_ft.shape}")

print("\n  [FINE-TUNED] Train set...")
X_train_ft, _ = extract_embeddings('ft', texts_train, labels_train)
print(f"  → Shape: {X_train_ft.shape}")


# ── 4. Separabilidad lineal ──

print("\n" + "="*65)
print("  PASO 4 — Evaluando separabilidad lineal")
print("  Clasificador: Regresión Logística (sin capas ocultas)")
print("="*65)

def eval_linear_separability(X_train, X_test, y_train, y_test, nombre):
    from sklearn.preprocessing import label_binarize
    print(f"\n  Entrenando regresión logística sobre embeddings {nombre}...")
    t0  = time.time()
    lr  = LogisticRegression(
        max_iter=1000, random_state=42,
        solver='lbfgs', C=1.0)
    lr.fit(X_train, y_train)

    y_probs = lr.predict_proba(X_test)
    y_preds = lr.predict(X_test)

    auc = roc_auc_score(
        y_test, y_probs, multi_class='ovr', average='macro')
    acc = (y_preds == y_test).mean()

    y_bin = label_binarize(y_test, classes=[0,1,2,3,4])
    aucs_per_class = []
    for i in range(5):
        if y_bin[:, i].sum() > 0:
            a = roc_auc_score(y_bin[:, i], y_probs[:, i])
            aucs_per_class.append(a)
        else:
            aucs_per_class.append(float('nan'))

    elapsed = time.time() - t0
    print(f"\n  ── {nombre} ──")
    print(f"  AUC Macro (OvR):  {auc:.4f}")
    print(f"  Accuracy:         {acc:.4f}")
    print(f"  Tiempo:           {elapsed:.1f}s")
    print(f"  AUC por clase ESI:")
    for i, a in enumerate(aucs_per_class):
        print(f"    ESI {i+1}: {a:.4f}")
    print(f"  {'─'*45}")
    return auc, acc, aucs_per_class

auc_base, acc_base, aucs_base = eval_linear_separability(
    X_train_base, X_test_base, y_train, y_test, "SapBERT BASE")

# Resultados intermedios BASE antes de continuar
print("\n   BASE completado. Continuando con FINE-TUNED...")

auc_ft, acc_ft, aucs_ft = eval_linear_separability(
    X_train_ft, X_test_ft, y_train, y_test, "SapBERT FINE-TUNED")