#API REST — Pre-Triage ESI (v8.0 — Cross-Attention + UMLS Autocomplete)

import os, time, sys, re
import torch
import numpy as np
import uvicorn
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from transformers import AutoTokenizer, AutoModel


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_pretriaje_esi import load_all_models, run_pipeline

# ── Config ──
FRONTEND_FILE = "./INDEX.html"
UMLS_EMBEDDINGS_PATH = "../UMLS_BIOLORD/umls_embeddings_full_biolord.pt"
BIOLORD_NAME = "FremyCompany/BioLORD-2023"

# ── FastAPI ──
app = FastAPI(title="Pre-Triage ESI", version="8.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class TriageRequest(BaseModel):
    nombre: str
    edad: int
    sexo: str
    brief_cc: str
    full_text: Optional[str] = ""
    pain: float
    temperature: int
    heartrate: int
    resprate: int

class SuggestRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

patient_queue = []
patient_counter = 0
models = {}
umls_index = {}
biolord = {}  # BioLORD model + tokenizer for UMLS search


# ── UMLS Entity Linker ──

def clean_synonym(text):
    """Clean a UMLS synonym for display."""
    text = text.replace(';', ' ').replace('(', '').replace(')', '')
    text = re.sub(r'\[.*?\]', '', text)
    text = text.replace(' NOS', '').replace(' NEC', '')
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Fix ALL CAPS → Title Case
    if text.isupper() and len(text) > 3:
        text = text.title()
        for word in ['Of', 'In', 'On', 'At', 'To', 'Or', 'And', 'The', 'A', 'An', 'By', 'For', 'With']:
            text = text.replace(f' {word} ', f' {word.lower()} ')
    
    # Ensure first letter uppercase
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    # Clean trailing artifacts
    text = text.strip(' ,-/')
    
    return text


def load_umls_index():
    """Load pre-computed UMLS embeddings for entity linking."""
    if not os.path.exists(UMLS_EMBEDDINGS_PATH):
        print(f"   ⚠️  UMLS embeddings not found at {UMLS_EMBEDDINGS_PATH}")
        return {}
    
    print("   Loading UMLS entity linking index...")
    data = torch.load(UMLS_EMBEDDINGS_PATH, map_location='cpu', weights_only=False)
    
    print(f"   ✅ UMLS loaded: {data['n_concepts']:,} concepts, {data['n_texts']:,} synonyms, PCA {data['pca_dim']}D")
    return data


def search_umls(query, top_k=5):
    """Search UMLS concepts by semantic similarity with exact match priority.
    Uses BioLORD-2023 for better colloquial → medical matching."""
    if not umls_index or not biolord:
        return []
    
    bl_tokenizer = biolord['tokenizer']
    bl_model = biolord['model']
    device = models['device']
    
    synonym_texts = umls_index['synonym_texts']
    cui_indices = umls_index['cui_indices']
    concept_cuis = umls_index['concept_cuis']
    concept_canonical = umls_index['concept_canonical']
    concept_stypes = umls_index['concept_semantic_types']
    
    # STEP 1: Check for exact matches first
    query_lower = query.lower().strip()
    exact_results = []
    seen_cuis_exact = set()
    
    for i, syn in enumerate(synonym_texts):
        if syn.lower() == query_lower:
            concept_idx = cui_indices[i]
            cui = concept_cuis[concept_idx]
            if cui not in seen_cuis_exact:
                seen_cuis_exact.add(cui)
                display = clean_synonym(syn)
                if len(display) < 3:
                    display = clean_synonym(concept_canonical[concept_idx])
                exact_results.append({
                    'cui': cui,
                    'display_name': display,
                    'classifier_name': display.lower(),
                    'score': 1.0,
                    'semantic_type': concept_stypes[concept_idx],
                })
                if len(exact_results) >= 2:
                    break
    
    # STEP 2: Semantic search with BioLORD (mean pooling)
    encoded = bl_tokenizer(query, padding=True, truncation=True,
                           max_length=64, return_tensors='pt').to(device)
    with torch.no_grad():
        output = bl_model(**encoded)
        token_emb = output.last_hidden_state
        mask = encoded['attention_mask'].unsqueeze(-1).expand(token_emb.size()).float()
        q_emb_768 = ((token_emb * mask).sum(1) / mask.sum(1)).cpu().numpy()
    
    pca_components = umls_index['pca_components'].numpy()
    pca_mean = umls_index['pca_mean'].numpy()
    q_emb_256 = (q_emb_768 - pca_mean) @ pca_components.T
    q_norm = np.maximum(np.linalg.norm(q_emb_256, axis=1, keepdims=True), 1e-8)
    q_emb_256 = q_emb_256 / q_norm
    q_tensor = torch.tensor(q_emb_256, dtype=torch.float16)
    
    embeddings = umls_index['embeddings']
    similarities = torch.mm(q_tensor, embeddings.t()).squeeze(0)
    top_indices = similarities.argsort(descending=True)[:top_k * 5]
    
    # Deduplicate: exclude CUIs already in exact, but within semantic allow
    # multiple synonyms of the same CUI if semantically distinct
    seen_cuis_exact = set(r['cui'] for r in exact_results)
    seen_synonyms = set()
    semantic_results = []

    for idx in top_indices:
        idx = idx.item()
        concept_idx = cui_indices[idx]
        cui = concept_cuis[concept_idx]
        if cui in seen_cuis_exact:
            continue
        matched = synonym_texts[idx]
        display = clean_synonym(matched)
        if display in seen_synonyms:
            continue
        seen_synonyms.add(display)
        canonical = concept_canonical[concept_idx]
        
        canonical_clean = clean_synonym(canonical)
        if len(display) < 3:
            display = canonical_clean
        
        semantic_results.append({
            'cui': cui,
            'display_name': display,
            'classifier_name': display.lower(),
            'score': round(similarities[idx].item(), 3),
            'semantic_type': concept_stypes[concept_idx],
        })
        if len(semantic_results) >= top_k:
            break
    
    combined = exact_results + semantic_results
    return combined[:top_k]


# ── Startup ──

@app.on_event("startup")
async def startup():
    global models, umls_index, biolord
    models = load_all_models()
    umls_index = load_umls_index()
    
    # Load BioLORD for UMLS entity linking (separate from SapBERT classifier)
    if umls_index:
        print("   Loading BioLORD-2023 for entity linking...")
        biolord_tokenizer = AutoTokenizer.from_pretrained(BIOLORD_NAME)
        biolord_model = AutoModel.from_pretrained(BIOLORD_NAME).to(models['device']).eval()
        biolord['tokenizer'] = biolord_tokenizer
        biolord['model'] = biolord_model
        print("   ✅ BioLORD-2023 loaded")


# ── Frontend routes ──

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if os.path.exists(FRONTEND_FILE):
        with open(FRONTEND_FILE, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>")

@app.get("/patient", response_class=HTMLResponse)
async def serve_patient():
    if os.path.exists(FRONTEND_FILE):
        with open(FRONTEND_FILE, 'r', encoding='utf-8') as f:
            html = f.read().replace("view:'select'", "view:'patient'")
            return HTMLResponse(content=html)

@app.get("/nurse", response_class=HTMLResponse)
async def serve_nurse():
    if os.path.exists(FRONTEND_FILE):
        with open(FRONTEND_FILE, 'r', encoding='utf-8') as f:
            html = f.read().replace("view:'select'", "view:'nurse'")
            return HTMLResponse(content=html)


# ── API endpoints ──

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "models_loaded": bool(models), 
        "umls_loaded": bool(umls_index),
        "umls_concepts": umls_index.get('n_concepts', 0),
        "version": "8.0"
    }

@app.get("/pacientes")
async def get_pacientes():
    return {"patients": patient_queue}

@app.delete("/pacientes/{pid}")
async def delete_paciente(pid: str):
    global patient_queue
    before = len(patient_queue)
    patient_queue = [p for p in patient_queue if p['id'] != pid]
    return {"success": len(patient_queue) < before}

@app.post("/suggest")
async def suggest(req: SuggestRequest):
    """UMLS entity linking: returns top-K symptom suggestions for autocomplete."""
    if not req.query or len(req.query.strip()) < 2:
        return {"suggestions": []}
    
    t0 = time.time()
    results = search_umls(req.query.strip(), top_k=req.top_k or 5)
    elapsed = time.time() - t0
    
    return {
        "query": req.query,
        "suggestions": results,
        "time_ms": round(elapsed * 1000, 1),
    }

@app.post("/triaje")
async def triaje(req: TriageRequest):
    global patient_counter
    try:
        has_symptoms = bool((req.brief_cc and req.brief_cc.strip()) or
                    (req.full_text and req.full_text.strip()))
        text_for_ner = req.full_text if req.full_text and req.full_text.strip() else req.brief_cc

        result = run_pipeline(
            text_en=text_for_ner,
            pain=req.pain,
            temperature=req.temperature,
            heartrate=req.heartrate,
            resprate=req.resprate,
            models=models,
            age=req.edad,
            gender=req.sexo,
            brief_cc=req.brief_cc if has_symptoms else None,
            has_symptoms=has_symptoms,
        )

        patient_counter += 1
        patient = {
            "id": f"p-{patient_counter}-{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "ts": int(time.time() * 1000),
            "data": {
                "nombre": req.nombre,
                "edad": req.edad,
                "sexo": req.sexo,
                "brief_cc": req.brief_cc,
                "full_text": req.full_text or "",
                "pain": req.pain,
                "temperature": req.temperature,
                "heartrate": req.heartrate,
                "resprate": req.resprate,
            },
            "result": result,
        }
        patient_queue.append(patient)

        fb = f" [{result.get('cc_source', '')}]"
        mg = f" [margin:{result['margin']:.2f}]" if result.get('esi_secondary') else ""
        print(f"   ✅ {req.nombre} ({req.edad}{req.sexo[0].upper()}) → "
              f"{result['esi_display']} (score: {result['urgency_score']:.2f}, "
              f"CC: {result['chiefcomplaint'][:50]}{fb}{mg})")

        return {"success": True, **patient}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("\n🏥 Starting Pre-Triage ESI server (v8.0 — Cross-Attention + UMLS)...")
    print("   Frontend: http://localhost:8000")
    print("   Patient:  http://localhost:8000/patient")
    print("   Nurse:    http://localhost:8000/nurse")
    print("   API docs: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)