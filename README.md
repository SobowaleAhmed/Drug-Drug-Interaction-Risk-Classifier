# 💊 Drug-Drug Interaction Risk Classifier
### A Nigeria-Focused Machine Learning & Deep Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?style=flat-square&logo=streamlit)
![PyTDC](https://img.shields.io/badge/PyTDC-TWOSIDES-blueviolet?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🧠 Overview

This project builds an end-to-end **Drug-Drug Interaction (DDI) Risk Classification system** grounded in Nigerian healthcare context. It scrapes drugs registered by **NAFDAC (National Agency for Food and Drug Administration and Control)** from the public Greenbook, maps them against global interaction databases (TWOSIDES via PyTDC, OpenFDA, PubChem), engineers pharmacological features, and classifies interaction severity into 4 classes:

| Class | Severity | Description |
|-------|----------|-------------|
| 0 | ✅ None | No known clinically relevant interaction |
| 1 | 🟡 Mild | Minor — monitoring recommended |
| 2 | 🟠 Moderate | Clinically significant — dose adjustment may be needed |
| 3 | 🔴 Severe | Life-threatening — combination should be avoided |

The project prioritises drugs common in Nigerian clinical practice: **antimalarials, antihypertensives, antibiotics, antiretrovirals (HIV/AIDS), antidiabetics, and antituberculosis agents.**

---

## 🗂️ Project Structure

```
drug-drug-interaction-nigeria/
│
├── notebooks/
│   ├── 00_nafdac_scraper.ipynb     # Scrape NAFDAC Greenbook → CSV (run first)
│   ├── 01_eda.ipynb                # Data loading, feature engineering, 7 EDA charts
│   ├── 02_ml_models.ipynb          # LR, RF, XGBoost + SHAP + full validation suite
│   └── 03_lstm_notebook.ipynb      # Keras DNN + PyTorch DNN (both frameworks)
│
├── app/
│   └── streamlit_app.py            # Interactive DDI Risk Checker (3-tab Streamlit app)
│
├── models/                         # Populated after running notebooks 02 & 03
│   ├── lr_model.pkl                # Logistic Regression
│   ├── rf_model.pkl                # Random Forest
│   ├── xgboost_best.pkl            # XGBoost (tuned)
│   ├── keras_dnn.h5                # Keras DNN weights
│   ├── keras_dnn_savedmodel/       # Keras SavedModel format
│   ├── pytorch_dnn_best.pt         # PyTorch best state dict
│   ├── pytorch_dnn_full.pt         # PyTorch full checkpoint
│   ├── scaler.pkl                  # StandardScaler
│   └── imputer.pkl                 # SimpleImputer (median)
│
├── README.md                       # This file
├── ABOUT_MODEL.md                  # Model card — methodology & limitations
└── requirements.txt                # All dependencies
```

> **Data files** (`nafdac_ingredients.csv`, `nafdac_individual_drugs.csv`, `ddi_nigeria_features.csv`) are stored in Google Drive and loaded by the notebooks at runtime. They are not committed to the repository.

---

## 📦 Data Sources

| Source | Access Method | What It Provides |
|--------|---------------|-----------------|
| [NAFDAC Greenbook](https://greenbook.nafdac.gov.ng/ingredients) | BeautifulSoup scraping (45 pages) | 2,200+ registered Nigerian drug ingredient names & IDs |
| [TWOSIDES](https://tdcommons.ai/multi_pred_tasks/ddi/) | PyTDC — `DDI(name='TWOSIDES')` | 4.6M labeled drug-pair interaction records |
| [OpenFDA API](https://api.fda.gov/drug/label.json) | REST API | Drug labels, interaction warnings, adverse events |
| [PubChem REST API](https://pubchem.ncbi.nlm.nih.gov/rest/pug) | REST API | Molecular descriptors per drug (MW, LogP, TPSA) |
| [DrugBank](https://go.drugbank.com) | Curated lookup table (in-code) | CYP enzyme flags, protein binding %, half-life |

> **No Kaggle account needed.** TWOSIDES is loaded directly via PyTDC in 3 lines of code.

---

## ⚙️ Tech Stack

### Data Collection
- `beautifulsoup4`, `lxml`, `requests` — NAFDAC Greenbook scraping (45 HTML pages)
- `PyTDC` — TWOSIDES dataset (4.6M DDI records, no manual download)
- `requests` — OpenFDA & PubChem REST API calls

### Machine Learning
- `pandas`, `numpy` — wrangling & feature engineering
- `scikit-learn` — Logistic Regression, Random Forest, metrics, StratifiedKFold, GridSearchCV
- `xgboost` — gradient boosted classifier
- `imbalanced-learn` — SMOTE oversampling
- `shap` — SHAP beeswarm, dot, and waterfall plots
- `joblib` — model serialisation

### Deep Learning
- `tensorflow` / `keras` — DNN (3 blocks: Dense→BatchNorm→ReLU→Dropout), Adam, EarlyStopping, ReduceLROnPlateau
- `torch` / `torch.nn` — PyTorch DNN, custom training loop, StepLR scheduler, early stopping

### Visualisation
- `plotly` — interactive EDA, ROC curves, PR curves, leaderboard bar charts
- `seaborn`, `matplotlib` — confusion matrices, training curves, CYP heatmaps

### Deployment
- `streamlit` — 3-tab interactive risk checker (Risk Checker · Model Performance · About)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/drug-drug-interaction-nigeria.git
cd drug-drug-interaction-nigeria
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Notebooks in Order (Google Colab)

| Step | Notebook | What It Does | Key Output |
|------|----------|--------------|------------|
| 1 | `00_nafdac_scraper.ipynb` | Scrapes NAFDAC Greenbook (45 pages) | `nafdac_ingredients.csv`, `nafdac_individual_drugs.csv` |
| 2 | `01_eda.ipynb` | Loads data, builds feature dataset, runs 7 EDA charts | `ddi_nigeria_features.csv` |
| 3 | `02_ml_models.ipynb` | Trains & validates LR, RF, XGBoost + SHAP | `models/*.pkl`, `ml_leaderboard.csv` |
| 4 | `03_lstm_notebook.ipynb` | Trains Keras & PyTorch DNNs, full comparison | `models/*.h5`, `models/*.pt`, `final_leaderboard.csv` |

> All notebooks mount Google Drive automatically and locate files with `os.walk()` — just run top to bottom.

### 4. Launch the Streamlit App
```bash
cd app
streamlit run streamlit_app.py
```

Set `MODELS_DIR` env variable if running locally:
```bash
MODELS_DIR=/path/to/your/models streamlit run streamlit_app.py
```

---

## 📊 Model Results Summary

| Model | Macro F1 | ROC-AUC | Role |
|-------|----------|---------|------|
| Logistic Regression | — | — | Baseline |
| Random Forest | — | — | Ensemble / Gini feature importance |
| XGBoost (Tuned) | — | — | Primary ML model |
| Keras DNN | — | — | Deep Learning (TF/Keras) |
| PyTorch DNN | — | — | Deep Learning (PyTorch) |

> Results populate after running notebooks 02 & 03. The final leaderboard is saved as `final_leaderboard.csv` and rendered live in the Streamlit **Model Performance** tab.

---

## 🌍 Nigerian Healthcare Context

This project targets the most clinically relevant drug combinations in Nigeria's disease burden:

| Disease Area | Key Drugs |
|---|---|
| **Malaria** | Artemether-Lumefantrine, Chloroquine, Quinine, Artesunate-Amodiaquine |
| **HIV/AIDS** | Efavirenz, Nevirapine, Lopinavir/Ritonavir, Dolutegravir, Tenofovir |
| **Tuberculosis** | Rifampicin, Isoniazid, Pyrazinamide, Ethambutol |
| **Hypertension** | Amlodipine, Lisinopril, Atenolol, Hydrochlorothiazide |
| **Diabetes** | Metformin, Glibenclamide, Insulin |
| **Infections** | Amoxicillin, Ciprofloxacin, Metronidazole, Fluconazole |

**Rifampicin** — backbone of TB treatment — is one of the most potent CYP3A4/CYP2C9 inducers known. It dramatically reduces plasma concentrations of most ARVs, making **TB-HIV co-treatment** the highest-stakes DDI scenario in Nigerian clinical practice and a core focus of this classifier.

---

## 🗃️ Pipeline Diagram

```
00_nafdac_scraper.ipynb
         │
         ▼
nafdac_ingredients.csv + nafdac_individual_drugs.csv
         │
         ▼
01_eda.ipynb  ◄── PyTDC (TWOSIDES) + PubChem API + OpenFDA API
         │
         ▼
ddi_nigeria_features.csv
         │
    ┌────┴────┐
    ▼         ▼
02_ml_models  03_lstm_notebook
    │              │
    ▼              ▼
*.pkl models    *.h5 / *.pt models
         │
         ▼
   streamlit_app.py
```

---

## 👤 Author

**Ahmed**
Production Chemist | Data Scientist | Computational Chemistry Enthusiast
- 🔗 [LinkedIn](#)
- 🐙 [GitHub](#)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- NAFDAC for the public Greenbook drug registry at `greenbook.nafdac.gov.ng`
- Tatonetti et al. (2012) for the TWOSIDES dataset
- Therapeutics Data Commons (PyTDC) for open dataset access
- OpenFDA for free access to drug label data
- PubChem for the free molecular descriptor REST API

