# 🧬 Model Card — Drug-Drug Interaction Risk Classifier
**About This Model | Version 1.0 | Nigeria-Focused DDI Classification**

---

## 📌 Model Overview

| Attribute | Detail |
|-----------|--------|
| **Task** | Multi-class classification (4 severity classes) |
| **Domain** | Clinical Pharmacology / Drug Safety |
| **Geography** | Nigeria — NAFDAC-registered drug pairs |
| **Input** | Feature vector of 31 features representing a drug pair (Drug A + Drug B) |
| **Output** | Severity class: None / Mild / Moderate / Severe |
| **Models** | Logistic Regression · Random Forest · XGBoost · Keras DNN · PyTorch DNN |
| **Framework** | scikit-learn · XGBoost · TensorFlow/Keras · PyTorch |
| **Author** | Ahmed |
| **Version Date** | 2026 |

---

## 🎯 Problem Statement

When two or more drugs are taken simultaneously, they can interact in ways that amplify, reduce, or completely alter each other's effects — sometimes with life-threatening consequences. In Nigeria, this risk is particularly acute due to:

- High rates of **polypharmacy** (patients managing multiple chronic conditions simultaneously)
- **TB-HIV co-infection** requiring concurrent treatment with Rifampicin + ARVs — one of the most dangerous known DDI scenarios globally
- Widespread use of antimalarials alongside other medications across diverse disease backgrounds
- Limited access to real-time clinical decision support tools in many Nigerian healthcare settings

This model provides a fast, data-driven tool to flag potentially dangerous drug combinations before they are prescribed or dispensed.

---

## 🗃️ Training Data

### Primary Dataset
- **TWOSIDES** (Tatonetti et al., 2012) — 4.6 million drug-drug interaction records derived from the FDA Adverse Event Reporting System (FAERS), accessed via **PyTDC** (`DDI(name='TWOSIDES')`). Each record contains a drug pair, associated adverse effect label (Y), and is mapped to a severity class using side effect keyword matching.

### Nigerian Drug Filtering
- Drug pairs were filtered and enriched to include drugs present in the **NAFDAC Greenbook** (`greenbook.nafdac.gov.ng/ingredients`) — scraped across 45 pages (2,200+ active ingredient entries).
- A curated clinical seed list of 50 known high-priority Nigerian DDIs was always included regardless of TWOSIDES overlap, anchoring the dataset to Nigeria's specific disease burden.

**Priority drug classes:**
- Antimalarials (ACTs, Chloroquine, Quinine, Artesunate)
- Antiretrovirals (Efavirenz, Nevirapine, Lopinavir/Ritonavir, Dolutegravir, Tenofovir)
- Anti-tuberculosis agents (Rifampicin, Isoniazid, Pyrazinamide, Ethambutol)
- Antihypertensives (Amlodipine, Lisinopril, Atenolol, Hydrochlorothiazide)
- Antidiabetics (Metformin, Glibenclamide, Insulin)
- Antibiotics (Amoxicillin, Ciprofloxacin, Metronidazole, Fluconazole, Cotrimoxazole)
- Anticoagulants (Warfarin, Heparin, Clopidogrel)
- Anticonvulsants (Phenytoin, Carbamazepine, Valproic acid, Phenobarbitone)

### Supplementary Sources
- **OpenFDA Drug Label API** — drug interaction warnings from official prescribing information
- **PubChem REST API** — computed molecular descriptors (MW, LogP, TPSA, H-bond counts)
- **DrugBank-derived lookup table** — CYP enzyme flags, protein binding %, half-life values

### Class Imbalance Handling
- **SMOTE** (Synthetic Minority Oversampling Technique) applied to training data only
- `k_neighbors` set dynamically to `min(5, minority_class_count - 1)` to avoid errors on small classes
- Class weights (`balanced`) also applied to ML model loss functions as a secondary safeguard
- Test set preserves original imbalanced distribution for honest evaluation

---

## 🔧 Feature Description

Each sample represents one **drug pair (Drug A + Drug B)**. The feature vector has **31 features** combining properties of both drugs.

### Molecular Features (PubChem)
| Feature | Description |
|---------|-------------|
| `mw_a` / `mw_b` | Molecular weight in g/mol |
| `logp_a` / `logp_b` | XLogP3 lipophilicity |
| `tpsa_a` / `tpsa_b` | Topological polar surface area (Å²) |
| `hbond_donors_a` / `hbond_donors_b` | Number of hydrogen bond donors |
| `hbond_acceptors_a` / `hbond_acceptors_b` | Number of hydrogen bond acceptors |

### CYP Enzyme Flags (DrugBank-derived)
| Feature | Description |
|---------|-------------|
| `cyp3a4_inhibitor_a` / `_b` | Does this drug inhibit CYP3A4? |
| `cyp3a4_inducer_a` / `_b` | Does this drug induce CYP3A4? |
| `cyp2c9_inhibitor_a` / `_b` | Does this drug inhibit CYP2C9? |
| `cyp2c9_substrate_a` / `_b` | Is this drug metabolised by CYP2C9? |

### Pharmacokinetic Features
| Feature | Description |
|---------|-------------|
| `narrow_ti_a` / `narrow_ti_b` | Narrow therapeutic index flag (1 = yes) |
| `protein_binding_a` / `protein_binding_b` | % plasma protein binding |
| `half_life_a` / `half_life_b` | Elimination half-life (hours) |

### Pair-Level Interaction Features
| Feature | Description |
|---------|-------------|
| `same_class` | Both drugs from same therapeutic class? |
| `metabolic_conflict` | Does one drug inhibit an enzyme the other is substrate of? |
| `inducer_substrate_conflict` | Does one drug induce an enzyme the other is substrate of? |
| `both_narrow_ti` | Both drugs have narrow therapeutic index? |
| `both_high_protein_binding` | Both drugs >85% protein bound? |
| `class_a_enc` / `class_b_enc` | Label-encoded therapeutic class |

---

## 🏗️ Model Architecture

### ML Models

```
Logistic Regression
  → multi_class='multinomial', solver='lbfgs'
  → class_weight='balanced', max_iter=1000
  → Role: Interpretable baseline

Random Forest
  → n_estimators=300, class_weight='balanced'
  → Role: Non-linear baseline + Gini feature importance

XGBoost (Tuned)
  → GridSearchCV over: n_estimators, max_depth, learning_rate,
                       subsample, colsample_bytree
  → StratifiedKFold(n_splits=5), scoring='f1_macro'
  → sample_weight='balanced'
  → Role: Primary ML model, SHAP-native explainability
```

### Keras DNN

```
Input(31 features)
  → Dense(256) → BatchNormalization → ReLU → Dropout(0.30)
  → Dense(128) → BatchNormalization → ReLU → Dropout(0.25)
  → Dense(64)  → BatchNormalization → ReLU → Dropout(0.20)
  → Dense(4, Softmax)

Optimizer  : Adam (lr=1e-3)
Loss       : sparse_categorical_crossentropy
Class weights: compute_class_weight('balanced')
Callbacks  : EarlyStopping(patience=15, restore_best_weights=True)
             ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6)
             ModelCheckpoint (save best only)
Max epochs : 150 | Batch size: 64
Val split  : 15% of training data (post-SMOTE)
```

### PyTorch DNN

```
DDIClassifier(nn.Module)
  block1: Linear(31→256) → BatchNorm1d → ReLU → Dropout(0.30)
  block2: Linear(256→128) → BatchNorm1d → ReLU → Dropout(0.25)
  block3: Linear(128→64)  → BatchNorm1d → ReLU → Dropout(0.20)
  output: Linear(64→4)   [raw logits → CrossEntropyLoss handles softmax]

Weight init   : He (Kaiming) normal — optimal for ReLU networks
Loss          : CrossEntropyLoss(weight=class_weights)
Optimizer     : Adam(lr=1e-3, weight_decay=1e-4)
Scheduler     : StepLR(step_size=20, gamma=0.5)
Early stopping: Manual (patience=15, saves best state dict)
Max epochs    : 150 | Batch size: 64
```

---

## 📏 Evaluation Metrics

| Metric | Why It's Used Here |
|--------|-------------------|
| **Macro F1 Score** | Treats all 4 classes equally — critical so the rare Severe class isn't ignored |
| **ROC-AUC (OvR)** | One-vs-Rest AUC curve per severity class + macro average |
| **Precision-Recall Curve** | More informative than ROC for imbalanced class distributions |
| **Confusion Matrix (4×4)** | Reveals which severity levels the model confuses |
| **Stratified K-Fold CV** | 5-fold; reports mean ± std F1 to confirm stability across data splits |
| **SHAP Explainability** | Beeswarm (global), dot (Severe class), waterfall (single prediction) |

---

## ⚠️ Limitations

1. **Label quality** — Severity classes are derived from keyword-matching of side effect names in FAERS adverse event reports, not from randomised controlled trials. Labels may reflect pharmacovigilance reporting bias rather than true causal severity.

2. **Nigerian-specific gap** — The drug list is NAFDAC-filtered, but interaction severity labels come from global databases (predominantly US/EU adverse event data). Nigerian-specific factors are not modelled:
   - Higher prevalence of herbal medicine co-use
   - West African CYP2D6 and CYP2C19 polymorphism frequencies
   - Nigeria-specific co-morbidity patterns

3. **Missing herbal interactions** — A significant proportion of Nigerian patients combine conventional drugs with traditional herbal medicines. This dataset does not capture those interactions.

4. **TWOSIDES name-matching gap** — TWOSIDES uses DrugBank IDs and naming conventions that may not match Nigerian generic drug names perfectly. A DrugBank ID → generic name lookup table bridges this gap, but coverage is not 100%.

5. **Class imbalance residual** — Severe interactions are rare by nature. Despite SMOTE and class weighting, performance on the Severe class will likely be lower than on the None/Mild classes.

6. **Not a clinical tool** — Built for educational and portfolio purposes. Should not be used to make real clinical prescribing decisions without full clinical validation.

---

## 🔬 Intended Use

| ✅ Appropriate Use | ❌ Inappropriate Use |
|-------------------|---------------------|
| Portfolio demonstration | Direct clinical prescribing decisions |
| Research & data science exploration | Replacing pharmacist or physician judgement |
| Educational tool for pharmacology/data science students | Regulatory or clinical submission |
| Foundation for a properly validated clinical tool | Patient self-medication guidance |
| Teaching CYP enzyme pharmacology concepts | Emergency clinical triage |

---

## 🌍 Societal Impact & Relevance

Nigeria carries one of the world's highest burdens of TB-HIV co-infection. Rifampicin — the backbone of TB treatment — is a potent inducer of CYP3A4 and CYP2C9, dramatically reducing plasma concentrations of most ARVs. A data-driven tool that flags this interaction and quantifies its severity class could support clinical decision-making in resource-limited settings where pharmacists may not have access to real-time reference tools.

Beyond TB-HIV, the malaria-endemic context means many patients receive artemisinin-based combination therapies (ACTs) concurrently with medications for HIV, hypertension, or diabetes. Systematic DDI screening across this combinatorial space is exactly the kind of problem data science can address at scale.

This project demonstrates how open global pharmacological data can be localised to Nigeria's drug registry to create contextually relevant AI tools — a reproducible model for building scientific AI in African healthcare contexts.

---

## 📚 References

1. Tatonetti, N.P. et al. (2012). Data-Driven Prediction of Drug Effects and Interactions. *Science Translational Medicine*, 4(125).
2. NAFDAC Greenbook — https://greenbook.nafdac.gov.ng
3. Therapeutics Data Commons (PyTDC) — https://tdcommons.ai
4. OpenFDA — https://open.fda.gov/apis/drug/
5. DrugBank — https://go.drugbank.com
6. PubChem — https://pubchem.ncbi.nlm.nih.gov
7. WHO ATC Classification — https://www.whocc.no/atc_ddd_index/

---

*Model Card prepared by Ahmed | 2026*
