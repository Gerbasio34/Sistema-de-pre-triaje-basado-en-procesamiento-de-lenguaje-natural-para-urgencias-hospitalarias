# Pipeline de pre-triaje ESI.
# Recibe el texto libre del paciente y sus vitales, y devuelve una prediccion
# del nivel ESI (1-5) junto con un urgency score para ordenar la cola de espera.

# Componentes:
#   NER: BioBERT + CRF, extrae sintomas del texto libre
#   Clasificador: SapBERT con cross-attention, 7 features clinicas

import torch
import torch.nn as nn
import numpy as np
import re
import time
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF

# 1. CONFIGURATION

NER_MODEL_PATH  = "../BioBERT_NER/biobert_sintomas_ner_v4_crf"
NER_BASE_MODEL  = "dmis-lab/biobert-base-cased-v1.2"

CLASSIFIER_PATH = "../SAPBERT_FINETUNED_MODEL/modelo_sapbert_finetuned_v5g_6D_ablation.pt"
SAPBERT_NAME    = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

VITAL_MAP = {
    'temperature': {1: 97.8, 2: 98.2, 3: 99.0, 4: 100.4, 5: 102.0},
    'heartrate':   {1: 70.0, 2: 85.0, 3: 100.0, 4: 110.0, 5: 120.0},
    'resprate': {1: 18.0, 2: 20.0, 3: 22.0, 4: 26.0, 5: 30.0},
}

# 2. NER MODEL: BioBERT + CRF

class BioBertCRF(nn.Module):
    """BioBERT with CRF layer for valid BIO sequence enforcement."""
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)
        if labels is not None:
            crf_labels = labels.clone()
            crf_mask = (labels != -100)
            crf_labels[~crf_mask] = 0
            crf_mask[:, 0] = True
            loss = -self.crf(emissions, crf_labels, mask=crf_mask, reduction='mean')
            return {"loss": loss, "logits": emissions}
        else:
            mask = attention_mask.bool()
            decoded = self.crf.decode(emissions, mask=mask)
            return {"decoded": decoded}


# 3. CLASSIFIER MODEL — Cross-Attention Fusion (IDENTICAL architecture to v5c)

class SapBERTCrossAttentionModel(nn.Module):
    """SapBERT + Clinical features with Cross-Attention + Text-Dominant fusion.
    
    """
    def __init__(self, clin_dim=7, dropout=0.3, n_heads=4, attn_dim=128):
        super().__init__()
        self.bert = AutoModel.from_pretrained(SAPBERT_NAME)
        bert_dim = 768

        # Project clinical features to attention dimension
        self.clin_project = nn.Sequential(
            nn.Linear(clin_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Project text [CLS] to attention dimension
        self.text_project = nn.Sequential(
            nn.Linear(bert_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cross-attention: clinical queries attend to text keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(attn_dim)

        # Fusion includes original CLS (768) + text_proj (128) + attended (128) = 1024
        self.cls_norm = nn.LayerNorm(bert_dim)
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

        # Output heads
        self.head_esi = nn.Linear(128, 5)
        self.head_binary = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask, clinical):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]

        text_proj = self.text_project(cls_emb)
        clin_proj = self.clin_project(clinical)

        text_seq = text_proj.unsqueeze(1)
        clin_seq = clin_proj.unsqueeze(1)

        attended, _ = self.cross_attn(
            query=clin_seq, key=text_seq, value=text_seq,
        )
        attended = self.attn_norm(attended.squeeze(1) + clin_proj)

        cls_normed = self.cls_norm(cls_emb)
        fused = torch.cat([cls_normed, text_proj, attended], dim=1)
        h = self.fusion(fused)

        return self.head_esi(h), self.head_binary(h)

# 5. NER POST-PROCESSING

def fix_tokenizer_contractions(text):

    text = re.sub(r"\bcan\s+'\s+t\b", "cannot", text)
    text = re.sub(r"\bdon\s+'\s+t\b", "do not", text)
    text = re.sub(r"\bdoesn\s+'\s+t\b", "does not", text)
    text = re.sub(r"\bdidn\s+'\s+t\b", "did not", text)
    text = re.sub(r"\bwon\s+'\s+t\b", "will not", text)
    text = re.sub(r"\bwouldn\s+'\s+t\b", "would not", text)
    text = re.sub(r"\bshouldn\s+'\s+t\b", "should not", text)
    text = re.sub(r"\bisn\s+'\s+t\b", "is not", text)
    text = re.sub(r"\baren\s+'\s+t\b", "are not", text)
    text = re.sub(r"\bwasn\s+'\s+t\b", "was not", text)
    text = re.sub(r"\bweren\s+'\s+t\b", "were not", text)
    text = re.sub(r"\bhasn\s+'\s+t\b", "has not", text)
    text = re.sub(r"\bhaven\s+'\s+t\b", "have not", text)
    text = re.sub(r"\bcouldn\s+'\s+t\b", "could not", text)

    for p, r in [("can't","cannot"),("don't","do not"),("doesn't","does not"),
                 ("didn't","did not"),("won't","will not"),("wouldn't","would not"),
                 ("shouldn't","should not"),("isn't","is not"),("aren't","are not"),
                 ("was't","was not"),("weren't","were not"),("hasn't","has not"),
                 ("haven't","have not"),("couldn't","could not")]:
        text = text.replace(p, r)
    text = re.sub(r"n\s*'\s*t", " not", text)
    return text

NEGATION_WORDS = {
    'no', 'not', 'without', 'never', 'nor', 'deny', 'denies',
    'denied', 'negative', 'absent', 'absence', 'none', 'neither',
    'cannot', 'do not', 'does not', 'did not', 'will not',
    'would not', 'should not', 'could not', 'is not', 'are not',
}
CLAUSE_SEPARATORS = {',', '.', ';', ' and ', ' but ', ' just ', ' only ',
                     ' however ', ' although ', ' though ', ' yet ', ' except '}
DOUBLE_NEG_PATTERNS = {
    'not stop', 'not quit', 'not go away', 'not goes away',
    'not gone away', 'not going away', 'not get rid',
    'not getting rid', 'never stop', 'never goes away',
    'never go away', 'not resolve', 'not resolved',
    'not let up', 'not ease up', 'not subside',
}

def is_negated(symptom_text, full_text_raw):
    full_text = fix_tokenizer_contractions(full_text_raw).lower()
    symptom_lower = fix_tokenizer_contractions(symptom_text).lower()
    symptom_pos = full_text.find(symptom_lower)
    if symptom_pos == -1:
        symptom_pos = full_text.find(symptom_text.lower())
    if symptom_pos == -1:
        return False
    text_before = full_text[:symptom_pos]
    last_separator_pos = -1
    for sep in CLAUSE_SEPARATORS:
        pos = text_before.rfind(sep)
        if pos > last_separator_pos:
            last_separator_pos = pos + len(sep)
    text_same_clause = text_before[last_separator_pos:] if last_separator_pos > 0 else text_before
    for pattern in DOUBLE_NEG_PATTERNS:
        if pattern in text_same_clause.strip().lower():
            return False
    words_before = text_same_clause.strip().split()[-5:]
    for word in words_before:
        clean_word = re.sub(r'[^\w\s]', '', word).strip()
        if clean_word in NEGATION_WORDS:
            return True
    return False

SPAN_SPLITTERS = [' and ', ' that ', ' which ', ' where ', ' when ',
                  ' because ', ' since ', ' while ', ' until ', ' so ', ' then ']

def split_compound_spans(symptoms):
    result = []
    for s in symptoms:
        if len(s.split()) <= 5:
            result.append(s); continue
        s_lower = s.lower()
        best_pos = len(s_lower)
        for sp in SPAN_SPLITTERS:
            pos = s_lower.find(sp)
            if pos > 2 and pos < best_pos: best_pos = pos
        if best_pos < len(s_lower):
            p1 = s[:best_pos].strip(); p2 = s[best_pos:].strip()
            for sp in SPAN_SPLITTERS:
                if p2.lower().startswith(sp.strip()):
                    p2 = p2[len(sp.strip()):].strip(); break
            if len(p1) > 2: result.append(p1)
            if len(p2) > 2: result.append(p2)
        else:
            result.append(s)
    return result

NOISE_WORDS = {
    'i', 'me', 'my', 'you', 'he', 'she', 'we', 'they',
    'a', 'an', 'the', 'in', 'on', 'of', 'for', 'with', 'to',
    'or', 'but', 'is', 'am', 'are', 'was', 'were', 'been', 'be',
    'have', 'has', 'had', 'do', 'does',
    'this', 'that', 'it', 'some', 'about', 'just', 'like',
    'got', 'getting',
    'very', 'really', 'extremely', 'terribly', 'quite', 'so',
    'unbearable', 'terrible', 'horrible', 'awful',
    'all', 'time', 'day', 'days', 'week', 'weeks',
    'morning', 'night', 'ago', 'since', 'lately', 'recently',
    'yesterday', 'today', 'hours', 'hour',
    'kind', 'sort', 'thing', 'stuff', 'everything',
    'lot', 'lots', 'bit', 'much', 'little', 'even',
    'keep', 'keeps', 'keeping', 'started', 'start',
    'goes', 'going', 'went', 'come', 'comes',
    'also', 'still', 'already', 'now',
}
INVALID_SINGLE_WORDS = {
    'something', 'anything', 'everything', 'nothing',
    'weird', 'strange', 'odd', 'funny', 'bad', 'worse',
    'wincing', 'moaning', 'screaming',
    'cold', 'hot', 'warm', 'eat', 'eating',
    'not', 'do', 'can', 'feel', 'feeling',
    'up', 'down', 'out', 'off', 'over',
}

def clean_symptom(raw_symptom):
    text = fix_tokenizer_contractions(raw_symptom.lower().strip())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    cleaned = [w for w in words if w not in NOISE_WORDS and len(w) >= 2]
    if not cleaned: return None
    if len(cleaned) == 1 and cleaned[0] in INVALID_SINGLE_WORDS: return None
    if len(cleaned) > 5: cleaned = cleaned[:5]
    return ' '.join(cleaned)

def deduplicate_symptoms(symptoms):
    if len(symptoms) <= 1: return symptoms
    seen = set(); unique = []
    for s in symptoms:
        if s not in seen: seen.add(s); unique.append(s)
    sorted_s = sorted(unique, key=len, reverse=True)
    final = []
    for sym in sorted_s:
        if not any(set(sym.split()).issubset(set(e.split())) for e in final):
            final.append(sym)
    return final

def extract_and_clean_symptoms(text_en, ner_model, ner_tokenizer, device, verbose=False):
    id2label = {0: "O", 1: "B-SINTOMA", 2: "I-SINTOMA"}
    inputs = ner_tokenizer(text_en, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = ner_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        decoded = outputs["decoded"][0]
    labels = [id2label[tag_id] for tag_id in decoded]
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    BRIDGE_WORDS = {'my', 'the', 'a', 'an', 'in', 'on', 'of', 'to', 'is', 'am',
                    'are', 'her', 'his', 'its', 'and', 'or', 'very', 'so', 'not',
                    'no', 'some', 'this', 'that', 'with', 'for'}
    MAX_BRIDGE_GAP = 2

    raw_symptoms = []; current = []; gap_tokens = []; gap_count = 0
    for token, label in zip(tokens, labels):
        if token in ["[CLS]", "[SEP]", "[PAD]"]: continue
        if label in ["B-SINTOMA", "I-SINTOMA"]:
            if gap_tokens and current:
                current.extend(gap_tokens)
            elif not current and label == "B-SINTOMA":
                pass
            gap_tokens = []
            gap_count = 0
            current.append(token)
        else:
            if current:
                clean_tok = token.lower().replace('##', '')
                if clean_tok in BRIDGE_WORDS and gap_count < MAX_BRIDGE_GAP:
                    gap_tokens.append(token)
                    gap_count += 1
                else:
                    s = ner_tokenizer.convert_tokens_to_string(current).strip()
                    if len(s) > 2: raw_symptoms.append(s)
                    current = []
                    gap_tokens = []
                    gap_count = 0
            else:
                gap_tokens = []
                gap_count = 0
    if current:
        s = ner_tokenizer.convert_tokens_to_string(current).strip()
        if len(s) > 2: raw_symptoms.append(s)
    if verbose: print(f"   NER raw: {raw_symptoms}")
    fixed = [fix_tokenizer_contractions(s) for s in raw_symptoms]
    affirmed = [s for s in fixed if not is_negated(s, text_en)]
    split = split_compound_spans(affirmed)
    cleaned = [c for s in split if (c := clean_symptom(s))]
    return deduplicate_symptoms(cleaned)


# 6. CLASSIFICATION + URGENCY SCORE 

def compute_urgency_score(probs, prob_urgent=0.0):
    base = sum((i + 1) * p for i, p in enumerate(probs))
    return base - 0.10 * prob_urgent

def build_fallback_cc(text_en):
    text = fix_tokenizer_contractions(text_en)
    parts = re.split(r'[,;.]+', text)
    kept = []
    for part in parts:
        part = part.strip()
        if not part: continue
        words = part.lower().split()
        if words and words[0] in {'no', 'not', 'never', 'without', 'neither', 'nor'}:
            continue
        if any(neg in part.lower() for neg in ['do not have', 'does not have',
                                                'did not have', 'no sign of']):
            continue
        kept.append(part)
    return ', '.join(kept) if kept else text_en

def run_pipeline(text_en, pain, temperature, heartrate, resprate, models,
                 age=50, gender="M", brief_cc=None, has_symptoms=True, verbose=False):
    t0 = time.time()

    if isinstance(temperature, int) and 1 <= temperature <= 5:
        temperature = VITAL_MAP['temperature'][temperature]
    if isinstance(heartrate, int) and 1 <= heartrate <= 5:
        heartrate = VITAL_MAP['heartrate'][heartrate]
    if isinstance(resprate, int) and 1 <= resprate <= 5:
        resprate = VITAL_MAP['resprate'][resprate]

    vitals_normal = (pain <= 1 and
                     temperature <= 97.8 and
                     heartrate <= 70.0 and
                     resprate <= 18.0)

    if not has_symptoms and vitals_normal:
        return {
            'esi_level': 5, 'esi_secondary': None, 'esi_display': 'ESI 5',
            'margin': 1.0, 'confidence': 0.80,
            'probabilities': [0.01, 0.02, 0.05, 0.12, 0.80],
            'symptoms': [], 'chiefcomplaint': brief_cc if brief_cc else (text_en[:50] + '...' if len(text_en) > 50 else text_en),
            'cc_source': 'shortcircuit_no_symptoms',
            'is_urgent': False, 'prob_urgent': 0.0, 'urgency_score': 4.80,
            'full_text': text_en, 'processing_time': round(time.time() - t0, 2),
        }

    if brief_cc is None or not brief_cc.strip():
        symptoms = extract_and_clean_symptoms(
        text_en, models['ner_model'], models['ner_tokenizer'],
        models['device'], verbose=verbose)
    else:
        symptoms = []

    if brief_cc is not None and brief_cc.strip():
        chiefcomplaint = brief_cc.strip().lower()
        cc_source = "direct"
    else:
        cc_source = "ner"
        use_fallback = (len(symptoms) == 0 or
                        (len(symptoms) == 1 and len(symptoms[0].split()) == 1))

        if not use_fallback:
            chiefcomplaint = ", ".join(symptoms)
        else:
            chiefcomplaint = build_fallback_cc(text_en)
            cc_source = "fallback"

    gender_M = 1.0 if str(gender).upper() == "M" else 0.0
    age_val = float(max(18, min(91, age)))

    raw = np.array([[pain, temperature, heartrate, resprate, age_val, gender_M]], dtype=np.float32)
    scaled = models['scaler_clin'].transform(raw)
    clinical = scaled.astype(np.float32)

    models['classifier_model'].eval()
    encoded = models['classifier_tokenizer'](
        chiefcomplaint, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
    input_ids = encoded['input_ids'].to(models['device'])
    attention_mask = encoded['attention_mask'].to(models['device'])
    clinical_tensor = torch.tensor(clinical, dtype=torch.float32).to(models['device'])

    with torch.no_grad():
        logits_esi, logits_binary = models['classifier_model'](input_ids, attention_mask, clinical_tensor)
        probs = torch.softmax(logits_esi, dim=1).cpu().numpy()[0].astype(float)
        pred_esi = int(np.argmax(probs) + 1)
        prob_urgent = float(torch.softmax(logits_binary, dim=1).cpu().numpy()[0][0])

    MARGIN_THRESHOLD = 0.10
    sorted_indices = np.argsort(probs)[::-1]
    top1_idx, top2_idx = sorted_indices[0], sorted_indices[1]
    margin = float(probs[top1_idx] - probs[top2_idx])

    esi_primary = pred_esi
    esi_secondary = None
    if margin < MARGIN_THRESHOLD:
        candidate = int(top2_idx + 1)
        esi_secondary = candidate
        lo, hi = sorted([esi_primary, esi_secondary])
        esi_display = f"ESI {lo}-{hi}"
    else:
        esi_display = f"ESI {esi_primary}"

    return {
        'esi_level': pred_esi, 'esi_secondary': esi_secondary,
        'esi_display': esi_display, 'margin': round(margin, 4),
        'confidence': float(probs[pred_esi - 1]),
        'probabilities': probs.tolist(), 'symptoms': symptoms,
        'chiefcomplaint': chiefcomplaint, 'cc_source': cc_source,
        'is_urgent': torch.argmax(logits_binary, dim=1).item() == 0,
        'prob_urgent': round(float(prob_urgent), 3),
        'urgency_score': round(compute_urgency_score(probs, prob_urgent), 3),
        'full_text': text_en,
        'processing_time': round(time.time() - t0, 2),
    }


# 7. MODEL LOADING 

def load_all_models():
    print()
    print("LOADING PRE-TRIAGE ESI MODELS (v8.2 — Clean, No Missing)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    print("   Loading NER (BioBERT+CRF v4)...")
    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
    ner_checkpoint = torch.load(
        f"{NER_MODEL_PATH}/biobert_crf_model.pt",
        map_location=device, weights_only=False)
    ner_model = BioBertCRF(NER_BASE_MODEL, ner_checkpoint['num_labels'])
    ner_model.load_state_dict(ner_checkpoint['model_state_dict'])
    ner_model.to(device).eval()
    print(f"NER loaded (CRF transitions: {ner_checkpoint['num_labels']}x{ner_checkpoint['num_labels']})")

    print("   Loading classifier (SapBERT Cross-Attention v5g clean)...")
    checkpoint = torch.load(CLASSIFIER_PATH, map_location=device, weights_only=False)
    classifier_model = SapBERTCrossAttentionModel(
        clin_dim=checkpoint['clin_dim'],
        dropout=checkpoint.get('dropout', 0.3),
        n_heads=checkpoint.get('n_heads', 4),
        attn_dim=checkpoint.get('attn_dim', 128))
    classifier_model.load_state_dict(checkpoint['model_state'])
    classifier_model.to(device).eval()
    classifier_tokenizer = AutoTokenizer.from_pretrained(SAPBERT_NAME)
    scaler_clin = checkpoint['scaler_clin']
    
    print(f"Classifier loaded (clin_dim={checkpoint['clin_dim']}, cross-attention {checkpoint.get('n_heads', 4)} heads)")

    return {
        'device': device,
        'ner_model': ner_model, 'ner_tokenizer': ner_tokenizer,
        'classifier_model': classifier_model, 'classifier_tokenizer': classifier_tokenizer,
        'scaler_clin': scaler_clin,
    }