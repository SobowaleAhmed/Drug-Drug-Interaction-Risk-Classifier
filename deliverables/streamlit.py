import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
import os, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DDI Risk Classifier — Nigeria",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SEV_MAP    = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}
SEV_COLORS = {
    "None"    : "#2ecc71",
    "Mild"    : "#f1c40f",
    "Moderate": "#e67e22",
    "Severe"  : "#e74c3c"
}
SEV_EMOJI  = {
    "None"    : "✅",
    "Mild"    : "🟡",
    "Moderate": "🟠",
    "Severe"  : "🔴"
}
SEV_DESC = {
    "None"    : "No known clinically relevant interaction between these drugs.",
    "Mild"    : "Minor interaction. Monitor the patient but dose adjustment is usually not required.",
    "Moderate": "Clinically significant interaction. Consider dose adjustment or closer monitoring.",
    "Severe"  : "Life-threatening interaction. This combination should generally be avoided."
}

# Models folder — adjust if running locally
MODELS_DIR = os.environ.get("MODELS_DIR", "models")

# Nigerian drug list (used for autocomplete)
NIGERIAN_DRUGS = sorted([
    "artemether","lumefantrine","artesunate","amodiaquine","chloroquine",
    "quinine","mefloquine","primaquine","dihydroartemisinin","piperaquine",
    "sulphadoxine","pyrimethamine",
    "efavirenz","nevirapine","lamivudine","zidovudine","tenofovir",
    "emtricitabine","lopinavir","ritonavir","atazanavir","darunavir",
    "dolutegravir","abacavir","stavudine",
    "rifampicin","isoniazid","pyrazinamide","ethambutol","streptomycin",
    "rifabutin","moxifloxacin","linezolid",
    "amlodipine","lisinopril","enalapril","losartan","hydrochlorothiazide",
    "atenolol","metoprolol","nifedipine","verapamil","diltiazem",
    "ramipril","furosemide","spironolactone","methyldopa",
    "metformin","glibenclamide","glimepiride","glipizide",
    "insulin","sitagliptin","pioglitazone",
    "amoxicillin","ampicillin","ciprofloxacin","metronidazole","fluconazole",
    "doxycycline","azithromycin","cotrimoxazole","sulfamethoxazole",
    "trimethoprim","ceftriaxone","gentamicin","erythromycin",
    "clarithromycin","nitrofurantoin","clindamycin",
    "aspirin","ibuprofen","diclofenac","paracetamol",
    "tramadol","codeine","morphine","naproxen",
    "warfarin","heparin","enoxaparin","clopidogrel",
    "ketoconazole","itraconazole","griseofulvin","nystatin",
    "phenytoin","carbamazepine","valproic acid","phenobarbitone",
    "lamotrigine","levetiracetam",
    "omeprazole","ranitidine","metoclopramide","loperamide",
    "prednisolone","dexamethasone","hydrocortisone","betamethasone"
])

DRUG_TO_CLASS = {
    "artemether":"Antimalarials","lumefantrine":"Antimalarials",
    "artesunate":"Antimalarials","amodiaquine":"Antimalarials",
    "chloroquine":"Antimalarials","quinine":"Antimalarials",
    "mefloquine":"Antimalarials","primaquine":"Antimalarials",
    "efavirenz":"Antiretrovirals","nevirapine":"Antiretrovirals",
    "lamivudine":"Antiretrovirals","zidovudine":"Antiretrovirals",
    "tenofovir":"Antiretrovirals","lopinavir":"Antiretrovirals",
    "ritonavir":"Antiretrovirals","dolutegravir":"Antiretrovirals",
    "abacavir":"Antiretrovirals","stavudine":"Antiretrovirals",
    "rifampicin":"Anti_TB","isoniazid":"Anti_TB",
    "pyrazinamide":"Anti_TB","ethambutol":"Anti_TB",
    "streptomycin":"Anti_TB","moxifloxacin":"Anti_TB",
    "amlodipine":"Antihypertensives","lisinopril":"Antihypertensives",
    "enalapril":"Antihypertensives","losartan":"Antihypertensives",
    "hydrochlorothiazide":"Antihypertensives","atenolol":"Antihypertensives",
    "metoprolol":"Antihypertensives","nifedipine":"Antihypertensives",
    "verapamil":"Antihypertensives","furosemide":"Antihypertensives",
    "metformin":"Antidiabetics","glibenclamide":"Antidiabetics",
    "glimepiride":"Antidiabetics","insulin":"Antidiabetics",
    "amoxicillin":"Antibiotics","ciprofloxacin":"Antibiotics",
    "metronidazole":"Antibiotics","fluconazole":"Antibiotics",
    "doxycycline":"Antibiotics","azithromycin":"Antibiotics",
    "cotrimoxazole":"Antibiotics","gentamicin":"Antibiotics",
    "erythromycin":"Antibiotics","clarithromycin":"Antibiotics",
    "aspirin":"Analgesics_NSAIDs","ibuprofen":"Analgesics_NSAIDs",
    "diclofenac":"Analgesics_NSAIDs","paracetamol":"Analgesics_NSAIDs",
    "tramadol":"Analgesics_NSAIDs","codeine":"Analgesics_NSAIDs",
    "warfarin":"Anticoagulants","heparin":"Anticoagulants",
    "clopidogrel":"Anticoagulants",
    "ketoconazole":"Antifungals","itraconazole":"Antifungals",
    "phenytoin":"Anticonvulsants","carbamazepine":"Anticonvulsants",
    "valproic acid":"Anticonvulsants","phenobarbitone":"Anticonvulsants",
    "lamotrigine":"Anticonvulsants","levetiracetam":"Anticonvulsants",
    "omeprazole":"GI_Drugs","ranitidine":"GI_Drugs",
    "prednisolone":"Corticosteroids","dexamethasone":"Corticosteroids",
    "hydrocortisone":"Corticosteroids","betamethasone":"Corticosteroids",
}

CYP_FLAGS = {
    "rifampicin"    :{"cyp3a4_inducer":1,"cyp2c9_inducer":1,"narrow_ti":0,"protein_binding":80,"half_life":3},
    "isoniazid"     :{"cyp2c9_inhibitor":1,"narrow_ti":0,"protein_binding":10,"half_life":5},
    "fluconazole"   :{"cyp3a4_inhibitor":1,"cyp2c9_inhibitor":1,"narrow_ti":0,"protein_binding":11,"half_life":30},
    "ketoconazole"  :{"cyp3a4_inhibitor":1,"narrow_ti":0,"protein_binding":99,"half_life":8},
    "itraconazole"  :{"cyp3a4_inhibitor":1,"narrow_ti":0,"protein_binding":99,"half_life":24},
    "erythromycin"  :{"cyp3a4_inhibitor":1,"narrow_ti":0,"protein_binding":73,"half_life":1.5},
    "clarithromycin":{"cyp3a4_inhibitor":1,"narrow_ti":0,"protein_binding":70,"half_life":5},
    "ciprofloxacin" :{"cyp1a2_inhibitor":1,"narrow_ti":0,"protein_binding":30,"half_life":5},
    "metronidazole" :{"cyp2c9_inhibitor":1,"narrow_ti":0,"protein_binding":20,"half_life":8},
    "warfarin"      :{"cyp2c9_substrate":1,"narrow_ti":1,"protein_binding":99,"half_life":40},
    "phenytoin"     :{"cyp2c9_substrate":1,"narrow_ti":1,"protein_binding":90,"half_life":22},
    "carbamazepine" :{"cyp3a4_inducer":1,"narrow_ti":1,"protein_binding":75,"half_life":15},
    "phenobarbitone":{"cyp3a4_inducer":1,"cyp2c9_inducer":1,"narrow_ti":1,"protein_binding":50,"half_life":100},
    "valproic acid" :{"cyp2c9_inhibitor":1,"narrow_ti":1,"protein_binding":90,"half_life":14},
    "efavirenz"     :{"cyp3a4_inducer":1,"cyp3a4_inhibitor":1,"narrow_ti":0,"protein_binding":99,"half_life":52},
    "nevirapine"    :{"cyp3a4_inducer":1,"narrow_ti":0,"protein_binding":60,"half_life":25},
    "ritonavir"     :{"cyp3a4_inhibitor":1,"narrow_ti":0,"protein_binding":98,"half_life":4},
    "lopinavir"     :{"cyp3a4_substrate":1,"narrow_ti":0,"protein_binding":99,"half_life":6},
    "verapamil"     :{"cyp3a4_inhibitor":1,"narrow_ti":0,"protein_binding":90,"half_life":7},
    "amlodipine"    :{"cyp3a4_substrate":1,"narrow_ti":0,"protein_binding":97,"half_life":45},
    "atenolol"      :{"narrow_ti":0,"protein_binding":3,"half_life":7},
    "metoprolol"    :{"cyp2d6_substrate":1,"narrow_ti":0,"protein_binding":12,"half_life":4},
    "metformin"     :{"narrow_ti":0,"protein_binding":0,"half_life":5},
    "glibenclamide" :{"cyp2c9_substrate":1,"narrow_ti":0,"protein_binding":99,"half_life":10},
    "aspirin"       :{"narrow_ti":0,"protein_binding":90,"half_life":0.25},
    "ibuprofen"     :{"cyp2c9_substrate":1,"narrow_ti":0,"protein_binding":99,"half_life":2},
    "quinine"       :{"cyp3a4_substrate":1,"cyp2d6_inhibitor":1,"narrow_ti":1,"protein_binding":80,"half_life":11},
    "cotrimoxazole" :{"cyp2c9_inhibitor":1,"narrow_ti":0,"protein_binding":65,"half_life":10},
    "doxycycline"   :{"narrow_ti":0,"protein_binding":90,"half_life":20},
    "furosemide"    :{"narrow_ti":0,"protein_binding":99,"half_life":0.5},
    "prednisolone"  :{"cyp3a4_substrate":1,"narrow_ti":0,"protein_binding":70,"half_life":3},
    "dexamethasone" :{"cyp3a4_substrate":1,"narrow_ti":0,"protein_binding":77,"half_life":5},
    "chloroquine"   :{"narrow_ti":0,"protein_binding":55,"half_life":720},
    "artemether"    :{"cyp3a4_substrate":1,"narrow_ti":0,"protein_binding":95,"half_life":2},
    "lumefantrine"  :{"cyp3a4_substrate":1,"narrow_ti":0,"protein_binding":99,"half_life":96},
    "lisinopril"    :{"narrow_ti":0,"protein_binding":25,"half_life":12},
    "omeprazole"    :{"cyp2c19_inhibitor":1,"narrow_ti":0,"protein_binding":95,"half_life":1},
    "amoxicillin"   :{"narrow_ti":0,"protein_binding":18,"half_life":1.5},
    "paracetamol"   :{"narrow_ti":0,"protein_binding":25,"half_life":2},
    "gentamicin"    :{"narrow_ti":1,"protein_binding":10,"half_life":2},
    "tenofovir"     :{"narrow_ti":0,"protein_binding":7,"half_life":17},
    "lamivudine"    :{"narrow_ti":0,"protein_binding":36,"half_life":18},
    "zidovudine"    :{"narrow_ti":0,"protein_binding":38,"half_life":1},
    "dolutegravir"  :{"cyp3a4_substrate":1,"narrow_ti":0,"protein_binding":99,"half_life":14},
}

CYP_DEFAULT = {
    "cyp3a4_inhibitor":0,"cyp3a4_inducer":0,"cyp3a4_substrate":0,
    "cyp2c9_inhibitor":0,"cyp2c9_inducer":0,"cyp2c9_substrate":0,
    "cyp2d6_inhibitor":0,"cyp2d6_substrate":0,"cyp1a2_inhibitor":0,
    "cyp2c19_inhibitor":0,"narrow_ti":0,"protein_binding":50.0,"half_life":6.0
}

FEATURE_COLS = [
    "mw_a","logp_a","tpsa_a","hbond_donors_a","hbond_acceptors_a",
    "mw_b","logp_b","tpsa_b","hbond_donors_b","hbond_acceptors_b",
    "cyp3a4_inhibitor_a","cyp3a4_inducer_a","cyp2c9_inhibitor_a","cyp2c9_substrate_a",
    "narrow_ti_a","protein_binding_a","half_life_a",
    "cyp3a4_inhibitor_b","cyp3a4_inducer_b","cyp2c9_inhibitor_b","cyp2c9_substrate_b",
    "narrow_ti_b","protein_binding_b","half_life_b",
    "same_class","metabolic_conflict","inducer_substrate_conflict",
    "both_narrow_ti","both_high_protein_binding",
    "class_a_enc","class_b_enc"
]

CLASS_LIST = sorted(set(DRUG_TO_CLASS.values()))
CLASS_ENC  = {cls: i for i, cls in enumerate(CLASS_LIST)}

# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH MODEL CLASS (must match training definition)
# ─────────────────────────────────────────────────────────────────────────────
class DDIClassifier(nn.Module):
    def __init__(self, n_features, n_classes=4, dropout_rates=(0.3, 0.25, 0.2)):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(n_features, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(dropout_rates[0])
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout_rates[1])
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(dropout_rates[2])
        )
        self.output = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.output(self.block3(self.block2(self.block1(x))))


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    models = {}
    try:
        models["XGBoost"]   = joblib.load(os.path.join(MODELS_DIR, "xgboost_best.pkl"))
        models["Random Forest"] = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
        models["Logistic Regression"] = joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl"))
        models["scaler"]    = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        models["imputer"]   = joblib.load(os.path.join(MODELS_DIR, "imputer.pkl"))
    except Exception as e:
        st.warning(f"Some ML models not found: {e}")

    try:
        from tensorflow import keras
        models["Keras DNN"] = keras.models.load_model(
            os.path.join(MODELS_DIR, "keras_dnn.h5"))
    except Exception:
        pass

    try:
        ckpt = torch.load(
            os.path.join(MODELS_DIR, "pytorch_dnn_full.pt"),
            map_location="cpu")
        n_feat = ckpt.get("n_features", len(FEATURE_COLS))
        pt_m   = DDIClassifier(n_features=n_feat)
        pt_m.load_state_dict(ckpt["model_state_dict"])
        pt_m.eval()
        models["PyTorch DNN"] = pt_m
    except Exception:
        pass

    return models


@st.cache_data(show_spinner=False)
def fetch_pubchem(drug_name):
    PROPS = "MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount"
    url   = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
             f"{requests.utils.quote(drug_name)}/property/{PROPS}/JSON")
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            p = r.json()["PropertyTable"]["Properties"][0]
            return {
                "molecular_weight": float(p.get("MolecularWeight", np.nan)),
                "logp"            : p.get("XLogP", np.nan),
                "tpsa"            : p.get("TPSA", np.nan),
                "hbond_donors"    : p.get("HBondDonorCount", np.nan),
                "hbond_acceptors" : p.get("HBondAcceptorCount", np.nan),
            }
    except Exception:
        pass
    return {k: np.nan for k in ["molecular_weight","logp","tpsa","hbond_donors","hbond_acceptors"]}


def get_cyp(drug):
    d = dict(CYP_DEFAULT)
    d.update(CYP_FLAGS.get(drug.lower(), {}))
    return d


def build_feature_vector(drug_a, drug_b, props_a, props_b):
    ca = get_cyp(drug_a)
    cb = get_cyp(drug_b)
    cls_a = DRUG_TO_CLASS.get(drug_a.lower(), "Unknown")
    cls_b = DRUG_TO_CLASS.get(drug_b.lower(), "Unknown")
    met = int(
        (ca["cyp3a4_inhibitor"] and cb["cyp3a4_substrate"]) or
        (cb["cyp3a4_inhibitor"] and ca["cyp3a4_substrate"]) or
        (ca["cyp2c9_inhibitor"] and cb["cyp2c9_substrate"]) or
        (cb["cyp2c9_inhibitor"] and ca["cyp2c9_substrate"])
    )
    ind = int(
        (ca["cyp3a4_inducer"] and cb["cyp3a4_substrate"]) or
        (cb["cyp3a4_inducer"] and ca["cyp3a4_substrate"])
    )
    same = int(cls_a == cls_b and cls_a != "Unknown")
    vec = {
        "mw_a": props_a.get("molecular_weight", np.nan),
        "logp_a": props_a.get("logp", np.nan),
        "tpsa_a": props_a.get("tpsa", np.nan),
        "hbond_donors_a": props_a.get("hbond_donors", np.nan),
        "hbond_acceptors_a": props_a.get("hbond_acceptors", np.nan),
        "mw_b": props_b.get("molecular_weight", np.nan),
        "logp_b": props_b.get("logp", np.nan),
        "tpsa_b": props_b.get("tpsa", np.nan),
        "hbond_donors_b": props_b.get("hbond_donors", np.nan),
        "hbond_acceptors_b": props_b.get("hbond_acceptors", np.nan),
        "cyp3a4_inhibitor_a": ca["cyp3a4_inhibitor"],
        "cyp3a4_inducer_a"  : ca["cyp3a4_inducer"],
        "cyp2c9_inhibitor_a": ca["cyp2c9_inhibitor"],
        "cyp2c9_substrate_a": ca["cyp2c9_substrate"],
        "narrow_ti_a"       : ca["narrow_ti"],
        "protein_binding_a" : ca["protein_binding"],
        "half_life_a"       : ca["half_life"],
        "cyp3a4_inhibitor_b": cb["cyp3a4_inhibitor"],
        "cyp3a4_inducer_b"  : cb["cyp3a4_inducer"],
        "cyp2c9_inhibitor_b": cb["cyp2c9_inhibitor"],
        "cyp2c9_substrate_b": cb["cyp2c9_substrate"],
        "narrow_ti_b"       : cb["narrow_ti"],
        "protein_binding_b" : cb["protein_binding"],
        "half_life_b"       : cb["half_life"],
        "same_class"                  : same,
        "metabolic_conflict"          : met,
        "inducer_substrate_conflict"  : ind,
        "both_narrow_ti"              : int(ca["narrow_ti"] and cb["narrow_ti"]),
        "both_high_protein_binding"   : int(ca["protein_binding"]>85 and cb["protein_binding"]>85),
        "class_a_enc"                 : CLASS_ENC.get(cls_a, 0),
        "class_b_enc"                 : CLASS_ENC.get(cls_b, 0),
    }
    return pd.DataFrame([vec])[FEATURE_COLS]


def predict_all_models(models, X_raw):
    results = {}
    imputer = models.get("imputer")
    scaler  = models.get("scaler")

    X_imp = pd.DataFrame(
        imputer.transform(X_raw) if imputer else X_raw.fillna(0),
        columns=FEATURE_COLS
    )
    X_sc = scaler.transform(X_imp) if scaler else X_imp.values

    for name in ["XGBoost","Random Forest","Logistic Regression"]:
        if name not in models: continue
        mdl   = models[name]
        proba = mdl.predict_proba(
            X_sc if name == "Logistic Regression" else X_imp
        )[0]
        pred  = int(np.argmax(proba))
        results[name] = {"pred": pred, "proba": proba}

    if "Keras DNN" in models:
        proba = models["Keras DNN"].predict(X_sc, verbose=0)[0]
        results["Keras DNN"] = {"pred": int(np.argmax(proba)), "proba": proba}

    if "PyTorch DNN" in models:
        pt_m = models["PyTorch DNN"]
        pt_m.eval()
        with torch.no_grad():
            t     = torch.tensor(X_sc, dtype=torch.float32)
            logits= pt_m(t)
            proba = torch.softmax(logits, dim=1).numpy()[0]
        results["PyTorch DNN"] = {"pred": int(np.argmax(proba)), "proba": proba}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.risk-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin: 8px 0;
}
.badge-None     { background:#d5f5e3; color:#1e8449; border:2px solid #2ecc71; }
.badge-Mild     { background:#fef9e7; color:#9a7d0a; border:2px solid #f1c40f; }
.badge-Moderate { background:#fef0e7; color:#9c4a00; border:2px solid #e67e22; }
.badge-Severe   { background:#fdedec; color:#922b21; border:2px solid #e74c3c; }

.drug-card {
    background:#f8f9fa; border-radius:12px;
    padding:16px 20px; margin:8px 0;
    border-left:4px solid #3498db;
}
.metric-row {
    display:flex; gap:10px; flex-wrap:wrap; margin:10px 0;
}
.metric-box {
    background:white; border-radius:8px;
    padding:10px 16px; border:1px solid #dee2e6;
    flex:1; min-width:100px; text-align:center;
}
.metric-label { font-size:0.7rem; color:#6c757d; text-transform:uppercase; letter-spacing:1px; }
.metric-value { font-size:1.1rem; font-weight:600; color:#212529; margin-top:2px; }
.section-header {
    font-size:0.75rem; font-weight:600; letter-spacing:2px;
    text-transform:uppercase; color:#6c757d;
    margin:16px 0 6px; border-bottom:1px solid #dee2e6; padding-bottom:6px;
}
.disclaimer {
    background:#fff3cd; border-radius:8px;
    padding:10px 16px; font-size:0.78rem;
    color:#856404; border:1px solid #ffc107;
    margin-top:16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/Flag_of_Nigeria.svg/320px-Flag_of_Nigeria.svg.png",
             width=60)
    st.markdown("## 💊 DDI Risk Classifier")
    st.markdown("**Nigeria-Focused** | v1.0")
    st.markdown("---")

    st.markdown("### Model Selection")
    selected_model = st.radio(
        "Primary model for prediction:",
        ["XGBoost","Random Forest","Logistic Regression","Keras DNN","PyTorch DNN"],
        index=0
    )
    show_all = st.checkbox("Show all model predictions", value=True)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
This tool predicts **Drug-Drug Interaction (DDI)** severity for drug pairs
common in Nigerian clinical practice.

**Severity Classes:**
- ✅ **None** — No known interaction
- 🟡 **Mild** — Monitor only
- 🟠 **Moderate** — Dose adjustment may be needed
- 🔴 **Severe** — Avoid combination

**Data:** NAFDAC Greenbook · TWOSIDES · OpenFDA · PubChem · DrugBank
    """)
    st.markdown("---")
    st.markdown('<p class="disclaimer">⚠️ For educational and research purposes only. Not a substitute for clinical judgement.</p>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 💊 Drug-Drug Interaction Risk Classifier")
st.markdown("**Nigeria-Focused Clinical Decision Support Tool**")
st.markdown("---")

models = load_models()
available_models = [k for k in models if k not in ("scaler","imputer")]

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Risk Checker", "📊 Model Performance", "ℹ️ About the Model"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RISK CHECKER
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Select Two Drugs to Check")
    st.caption("Start typing to search NAFDAC-registered Nigerian drugs")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Drug A</p>', unsafe_allow_html=True)
        drug_a = st.selectbox(
            "Select Drug A", NIGERIAN_DRUGS,
            index=NIGERIAN_DRUGS.index("rifampicin"),
            key="drug_a", label_visibility="collapsed"
        )
    with col2:
        st.markdown('<p class="section-header">Drug B</p>', unsafe_allow_html=True)
        drug_b = st.selectbox(
            "Select Drug B", NIGERIAN_DRUGS,
            index=NIGERIAN_DRUGS.index("efavirenz"),
            key="drug_b", label_visibility="collapsed"
        )

    if drug_a == drug_b:
        st.warning("Please select two different drugs.")
        st.stop()

    check_btn = st.button("🔍 Check Interaction", type="primary", use_container_width=True)

    if check_btn or ("last_pair" in st.session_state
                     and st.session_state.last_pair == (drug_a, drug_b)):

        st.session_state.last_pair = (drug_a, drug_b)

        with st.spinner("Fetching molecular data & running models..."):
            props_a = fetch_pubchem(drug_a)
            props_b = fetch_pubchem(drug_b)
            X       = build_feature_vector(drug_a, drug_b, props_a, props_b)

            if not available_models:
                st.error("No models loaded. Check the models/ folder path.")
                st.stop()

            all_results = predict_all_models(models, X)

        # ── Primary result ────────────────────────────────────────────────────
        primary = all_results.get(selected_model) or list(all_results.values())[0]
        sev_class = SEV_MAP[primary["pred"]]
        sev_color = SEV_COLORS[sev_class]
        sev_emoji = SEV_EMOJI[sev_class]
        confidence = float(primary["proba"][primary["pred"]]) * 100

        st.markdown("---")
        st.markdown("### Interaction Risk Assessment")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown(
                f'<div class="risk-badge badge-{sev_class}">'
                f'{sev_emoji} {sev_class.upper()}</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"**Confidence:** {confidence:.1f}%")
            st.markdown(f"**Model:** {selected_model}")
            st.markdown(f"*{SEV_DESC[sev_class]}*")

        with res_col2:
            # Probability bar chart
            proba_df = pd.DataFrame({
                "Severity": list(SEV_MAP.values()),
                "Probability": [float(p)*100 for p in primary["proba"]],
                "Color": [SEV_COLORS[v] for v in SEV_MAP.values()]
            })
            fig_proba = go.Figure(go.Bar(
                x=proba_df["Severity"],
                y=proba_df["Probability"],
                marker_color=proba_df["Color"],
                marker_line_width=0,
                text=[f"{p:.1f}%" for p in proba_df["Probability"]],
                textposition="outside"
            ))
            fig_proba.update_layout(
                title=f"Predicted Probability by Severity Class ({selected_model})",
                yaxis=dict(title="Probability (%)", range=[0, 115]),
                xaxis_title="Severity Class",
                plot_bgcolor="white", paper_bgcolor="white",
                height=300, margin=dict(t=40, b=20, l=20, r=20),
                font=dict(size=12)
            )
            st.plotly_chart(fig_proba, use_container_width=True)

        # ── Drug Property Cards ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Drug Properties")
        prop_col1, prop_col2 = st.columns(2)

        def render_drug_card(drug, props, cyp, col):
            cls = DRUG_TO_CLASS.get(drug.lower(), "Unknown")
            with col:
                st.markdown(
                    f'<div class="drug-card">'
                    f'<b style="font-size:1.05rem">{drug.title()}</b><br>'
                    f'<span style="font-size:0.8rem;color:#6c757d">{cls}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("MW (g/mol)",  f'{props.get("molecular_weight","—"):.0f}'
                          if not pd.isna(props.get("molecular_weight",np.nan)) else "—")
                m2.metric("LogP",        f'{props.get("logp","—"):.1f}'
                          if not pd.isna(props.get("logp",np.nan)) else "—")
                m3.metric("TPSA (Å²)",   f'{props.get("tpsa","—"):.0f}'
                          if not pd.isna(props.get("tpsa",np.nan)) else "—")

                cyp_flags = []
                if cyp.get("cyp3a4_inhibitor"): cyp_flags.append("CYP3A4 inhibitor")
                if cyp.get("cyp3a4_inducer"):   cyp_flags.append("CYP3A4 inducer")
                if cyp.get("cyp2c9_inhibitor"):  cyp_flags.append("CYP2C9 inhibitor")
                if cyp.get("cyp3a4_substrate"):  cyp_flags.append("CYP3A4 substrate")
                if cyp.get("cyp2c9_substrate"):  cyp_flags.append("CYP2C9 substrate")
                if cyp.get("narrow_ti"):         cyp_flags.append("⚠️ Narrow TI")

                if cyp_flags:
                    st.markdown(
                        " · ".join([f"`{f}`" for f in cyp_flags])
                    )
                else:
                    st.markdown("`No major CYP flags`")

                st.caption(
                    f"Protein binding: {cyp.get('protein_binding',50):.0f}% · "
                    f"Half-life: {cyp.get('half_life',6):.0f}h"
                )

        render_drug_card(drug_a, props_a, get_cyp(drug_a), prop_col1)
        render_drug_card(drug_b, props_b, get_cyp(drug_b), prop_col2)

        # ── Interaction mechanism flags ────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Interaction Mechanism Flags")
        ca, cb = get_cyp(drug_a), get_cyp(drug_b)

        flags = []
        if ca["cyp3a4_inhibitor"] and cb["cyp3a4_substrate"]:
            flags.append(f"🔴 **{drug_a.title()} inhibits CYP3A4** — reduces metabolism of {drug_b.title()} → **elevated {drug_b.title()} levels**")
        if cb["cyp3a4_inhibitor"] and ca["cyp3a4_substrate"]:
            flags.append(f"🔴 **{drug_b.title()} inhibits CYP3A4** — reduces metabolism of {drug_a.title()} → **elevated {drug_a.title()} levels**")
        if ca["cyp3a4_inducer"] and cb["cyp3a4_substrate"]:
            flags.append(f"🟠 **{drug_a.title()} induces CYP3A4** — accelerates metabolism of {drug_b.title()} → **reduced {drug_b.title()} efficacy**")
        if cb["cyp3a4_inducer"] and ca["cyp3a4_substrate"]:
            flags.append(f"🟠 **{drug_b.title()} induces CYP3A4** — accelerates metabolism of {drug_a.title()} → **reduced {drug_a.title()} efficacy**")
        if ca["cyp2c9_inhibitor"] and cb["cyp2c9_substrate"]:
            flags.append(f"🔴 **{drug_a.title()} inhibits CYP2C9** — raises {drug_b.title()} levels → **toxicity risk**")
        if cb["cyp2c9_inhibitor"] and ca["cyp2c9_substrate"]:
            flags.append(f"🔴 **{drug_b.title()} inhibits CYP2C9** — raises {drug_a.title()} levels → **toxicity risk**")
        if ca["narrow_ti"] and cb["narrow_ti"]:
            flags.append(f"⚠️ **Both drugs have a narrow therapeutic index** — any interaction is high-risk")
        if ca["narrow_ti"]:
            flags.append(f"⚠️ **{drug_a.title()} has a narrow therapeutic index** — small changes in levels have big consequences")
        if cb["narrow_ti"]:
            flags.append(f"⚠️ **{drug_b.title()} has a narrow therapeutic index** — small changes in levels have big consequences")

        if flags:
            for f in flags:
                st.markdown(f)
        else:
            st.success("No major CYP enzyme interaction flags detected for this pair.")

        # ── All-model comparison ───────────────────────────────────────────────
        if show_all and len(all_results) > 1:
            st.markdown("---")
            st.markdown("### All Model Predictions")

            model_rows = []
            for mname, res in all_results.items():
                s = SEV_MAP[res["pred"]]
                model_rows.append({
                    "Model"     : mname,
                    "Prediction": f'{SEV_EMOJI[s]} {s}',
                    "Confidence": f'{float(res["proba"][res["pred"]])*100:.1f}%',
                    "None %"    : f'{float(res["proba"][0])*100:.1f}%',
                    "Mild %"    : f'{float(res["proba"][1])*100:.1f}%',
                    "Moderate %": f'{float(res["proba"][2])*100:.1f}%',
                    "Severe %"  : f'{float(res["proba"][3])*100:.1f}%',
                })

            st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

            # Model agreement chart
            agree_preds = [SEV_MAP[r["pred"]] for r in all_results.values()]
            agreement   = len(set(agree_preds)) == 1
            if agreement:
                st.success(f"✅ All {len(all_results)} models agree: **{agree_preds[0]}**")
            else:
                from collections import Counter
                vote = Counter(agree_preds).most_common(1)[0]
                st.info(f"Models disagree. Majority vote: **{vote[0]}** ({vote[1]}/{len(all_results)} models)")

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown(
            '<div class="disclaimer">⚠️ <b>Disclaimer:</b> This tool is for educational and '
            'research purposes only. Predictions are based on statistical models and should '
            'not replace clinical pharmacist or physician judgement. Always consult a '
            'qualified healthcare professional before making prescribing decisions.</div>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Model Performance Dashboard")
    st.caption("Results from training on Nigeria-focused DDI dataset (TWOSIDES + NAFDAC + curated)")

    lb_path = os.path.join(os.path.dirname(MODELS_DIR), "final_leaderboard.csv")
    if os.path.exists(lb_path):
        lb = pd.read_csv(lb_path)
        st.dataframe(lb, use_container_width=True, hide_index=True)

        numeric_lb = lb[pd.to_numeric(lb["Macro F1"], errors="coerce").notna()].copy()
        numeric_lb["Macro F1"] = pd.to_numeric(numeric_lb["Macro F1"])
        numeric_lb["ROC-AUC"]  = pd.to_numeric(numeric_lb["ROC-AUC"])

        fig_lb = go.Figure()
        fig_lb.add_trace(go.Bar(
            name="Macro F1", x=numeric_lb["Model"], y=numeric_lb["Macro F1"],
            marker_color="#3498db",
            text=numeric_lb["Macro F1"].round(4), textposition="outside"
        ))
        fig_lb.add_trace(go.Bar(
            name="ROC-AUC", x=numeric_lb["Model"], y=numeric_lb["ROC-AUC"],
            marker_color="#e67e22",
            text=numeric_lb["ROC-AUC"].round(4), textposition="outside"
        ))
        fig_lb.update_layout(
            barmode="group",
            title="All Models — Macro F1 vs ROC-AUC",
            yaxis=dict(range=[0,1.15], title="Score"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=420
        )
        st.plotly_chart(fig_lb, use_container_width=True)
    else:
        st.info("Run Notebook 03 first to generate final_leaderboard.csv")

    # Training curves
    keras_hist_path = os.path.join(os.path.dirname(MODELS_DIR), "keras_training_history.csv")
    pt_hist_path    = os.path.join(os.path.dirname(MODELS_DIR), "pytorch_training_history.csv")

    if os.path.exists(keras_hist_path):
        st.markdown("#### Keras DNN Training Curves")
        kh = pd.read_csv(keras_hist_path)
        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(y=kh["loss"],    name="Train Loss", line=dict(color="#e74c3c")))
        fig_k.add_trace(go.Scatter(y=kh["val_loss"],name="Val Loss",   line=dict(color="#3498db", dash="dash")))
        fig_k.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                             height=300, title="Keras DNN Loss")
        st.plotly_chart(fig_k, use_container_width=True)

    if os.path.exists(pt_hist_path):
        st.markdown("#### PyTorch DNN Training Curves")
        ph = pd.read_csv(pt_hist_path)
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(y=ph["train_loss"], name="Train Loss", line=dict(color="#e74c3c")))
        fig_p.add_trace(go.Scatter(y=ph["val_loss"],   name="Val Loss",   line=dict(color="#e67e22", dash="dash")))
        fig_p.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                             height=300, title="PyTorch DNN Loss")
        st.plotly_chart(fig_p, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### About This Model")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Project:** Nigeria-Focused DDI Risk Classifier

**Author:** Ahmed

**Domain:** Clinical Pharmacology / Drug Safety

**Geography:** Nigeria — NAFDAC-registered drug pairs

**Task:** Multi-class classification (4 severity levels)

**Models:** Logistic Regression, Random Forest, XGBoost, Keras DNN, PyTorch DNN
        """)
    with c2:
        st.markdown("""
**Data Sources:**
- 🇳🇬 NAFDAC Greenbook (scraped — 2,200+ registered drugs)
- TWOSIDES dataset (Tatonetti et al., 4.6M DDI records)
- OpenFDA Drug Label API
- PubChem REST API (molecular descriptors)
- DrugBank (CYP enzyme flags, pharmacokinetics)

**Key Nigeria Context:**
- TB-HIV co-treatment (Rifampicin + ARVs) — highest-risk DDI scenario
- Antimalarial combinations (QT prolongation risk)
- Anticoagulant sensitivity (Warfarin + CYP2C9 inhibitors)
        """)

    st.markdown("---")
    st.markdown("#### Feature Engineering")
    feat_df = pd.DataFrame([
        {"Feature Group":"Molecular (PubChem)","Features":"MW, LogP, TPSA, H-bond donors/acceptors","Why It Matters":"Physical behavior of drug in the body"},
        {"Feature Group":"CYP Enzymes","Features":"CYP3A4/2C9/2D6 inhibitor, inducer, substrate flags","Why It Matters":"Primary mechanism of metabolic DDIs"},
        {"Feature Group":"Pharmacokinetics","Features":"Protein binding %, half-life (h)","Why It Matters":"Duration and intensity of interactions"},
        {"Feature Group":"Narrow TI","Features":"Narrow therapeutic index flag","Why It Matters":"Small level changes → big clinical consequences"},
        {"Feature Group":"Pair-level","Features":"Metabolic conflict, inducer-substrate conflict, same class","Why It Matters":"Synthesised features capturing pair dynamics"},
    ])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
#### Limitations
- Labels derived from FAERS adverse event reports — not RCTs
- Nigerian-specific factors (herbal medicines, West African CYP polymorphisms) not modelled
- For educational and portfolio use only — not validated for clinical deployment
    """)