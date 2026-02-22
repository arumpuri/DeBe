# 🦟 DeBe: Agentic AI Triage for Dengue (Demam Berdarah)

> **Bilingual clinical decision support for Indonesia's Puskesmas — powered by fine-tuned MedGemma 4B**



---

## 📌 What is DeBe?

DeBe (**De**mam **Be**rdarah) is an agentic AI triage system that helps nurses and community health workers at Indonesian Puskesmas rapidly assess dengue severity using routine clinical data — CBC lab results, symptoms, and an optional skin photo. It produces bilingual (English + Bahasa Indonesia) recommendations aligned with **WHO 2009 Dengue Classification** in 15–20 seconds, and is designed to run **fully offline** on modest hardware.

Built for the [MedGemma Impact Challenge 2026](https://kaggle.com/competitions/med-gemma-impact-challenge) — Agentic Workflow Track.

---

## 🏥 Why DeBe?

Indonesia recorded **161,752 confirmed dengue cases and 673 deaths in 2025** — with an estimated true burden 5–8× higher. With only 0.76 doctors per 1,000 people and over 10,000 Puskesmas serving 270 million citizens, frontline triage is largely performed by nurses using manual WHO checklists that are slow, subjective, and error-prone under high caseloads.

Published WHO literature shows **25–40% of current dengue referrals are low-risk** and manageable at the community level. DeBe is built to close that gap — reducing unnecessary hospital burden, supporting early detection of severe cases, and helping Indonesia reach its **zero dengue death target by 2030**.

---

## 🤖 System Architecture

DeBe orchestrates four specialist agents in sequence, plus a deterministic risk scorer:

```
Patient Input (symptoms + labs + optional photo)
        │
        ▼
┌─────────────────────┐
│  Agent 1: Vision    │  ← Base MedGemma (no LoRA) — stronger vision priors
│  Petechiae / Rash   │    Skin photo analysis for dengue warning signs
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Agent 2: Symptom   │  ← DeBe LoRA Adapter
│  Clinical Phase     │    Febrile / Critical / Recovery mapping
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐   ┌──────────────────────────┐
│  Agent 3: Lab       │ + │  Deterministic Risk Score │  ← Pure Python
│  Hematology         │   │  WHO Risk Score 0–15      │    LLM never calculates
└────────┬────────────┘   └──────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Agent 4: Decision  │  ← DeBe LoRA Adapter
│  Orchestrator       │    Final WHO classification + bilingual action plan
└─────────────────────┘
         │
         ▼
Bilingual Triage Report (English + Bahasa Indonesia)
WHO Classification: No Warning Signs / Warning Signs / Severe Dengue
```

**Key architectural principle:** The Vision Agent uses *base MedGemma without the LoRA adapter* — preserving the base model's stronger visual priors for image tasks. The three text agents use the DeBe specialist adapter. The risk scorer is 100% deterministic Python — the LLM explains, never calculates.

---

## 🧠 Model

| Component | Details |
|-----------|---------|
| Base model | `google/medgemma-4b-it` |
| Adapter | `arumpuri/medgemma-4b-debe-specialist` |
| Adapter type | LoRA (r=16, α=32, dropout=0.05) |
| Target modules | Attention (q, k, v, o) + MLP (gate, up, down) |
| Trainable parameters | 29,802,496 (1.29% of total) |
| Quantisation | 4-bit NF4 + double quantisation |
| Adapter size | ~119 MB |
| Quantised model size | ~2.2 GB (offline deployment) |
| Training time | 39 minutes on 1× Kaggle T4 GPU |
| Final training loss | 0.219 |

---

## 📊 Dataset Pipeline

The training dataset was built from the public [Mendeley Dengue Hematological Dataset](https://data.mendeley.com/datasets/6fsrsk3mb8/2) through a four-stage pipeline:

| Stage | Notebook | Output |
|-------|----------|--------|
| EDA & Quality Assessment | `notebook1_eda.ipynb` | Dataset characterisation, quality score 75/100 |
| Differential Count Normalisation | `notebook2_normalisation.ipynb` | Validity: 13.8% → 100% |
| Synthetic Clinical Augmentation | `notebook3_synthetic_features.ipynb` | +17 clinical features (WHO-grounded) |
| Feature Engineering + Augmentation | `notebook4_feature_engineering.ipynb` | +6 engineered markers, +400 severe/pediatric cases |
| LoRA Fine-tuning | `notebook5_finetuning.ipynb` | `arumpuri/medgemma-4b-debe-specialist` |

**Final dataset:** 1,923 cases — 28% no warning signs, 26% warning signs, 21% severe dengue, 25% controls. Age range 5–78 years. Full platelet range including critical <20,000 /μL.

**Six engineered features:**
- `NLR` — Neutrophil-to-Lymphocyte Ratio (Jayaratne et al. 2019: NLR >2.5 predicts severe dengue)
- `Platelet_to_WBC_Ratio` — captures dual cytopenia signal
- `HCT_Change_from_Baseline` — gender-specific hemoconcentration (WHO plasma leakage criterion)
- `Platelet_Severity_Score` — ordinal 0–4 (WHO platelet thresholds)
- `Hemoconcentration_Score` — ordinal 0–3
- `WHO_Risk_Score` — composite 0–15 (matches app real-time display)
![1feaa854-774f-4d6c-b259-495f7eedd43e](https://github.com/user-attachments/assets/e6d96797-90ca-412b-a570-cf91c55c0b06)
---

## 🖥️ Gradio Application

The DeBe Triage Engine is a fully bilingual Gradio application with:

- Real-time streaming output — each agent's reasoning appears as it completes
- Bilingual labels, tooltips, and helper text (English + Bahasa Indonesia)
- Visual WHO Risk Badge — colour-coded severity indicator
- Four clickable test scenarios covering the full WHO severity spectrum
- 100% fallback to deterministic rule engine if LLM generation fails



---

## 🚀 Quick Start

### Run the Gradio Demo (Kaggle / Cloud)

```python
# Step 1: Install dependencies
!pip install -q -U "pillow<11.0" "transformers>=4.49.0" accelerate bitsandbytes gradio huggingface_hub peft

# Step 2: Authenticate
from huggingface_hub import login
login("your_hf_token")

# Step 3: Run the app
# Open app/debe_triage_v3.py and execute all cells
# The Gradio interface will launch with a public share link
```

### Run Locally (Offline / Edge Deployment)

```bash
git clone https://github.com/arumpuri/debe.git
cd debe
pip install -r requirements.txt
python app/debe_triage_v3.py
```

> **Hardware requirement:** GPU with ≥8 GB VRAM for inference (16 GB for fine-tuning).
> For CPU-only inference, set `device_map="cpu"` — slower but functional.

---

## 📋 Requirements

```
torch>=2.0
transformers>=4.49.0
peft>=0.10.0
accelerate
bitsandbytes
gradio
huggingface_hub
pillow<11.0
pandas
numpy
scikit-learn
matplotlib
seaborn
datasets
```

---

## 🗺️ Deployment Roadmap

| Phase | Target | Status |
|-------|--------|--------|
| Cloud demo | Kaggle / HuggingFace Spaces | ✅ Complete |
| 4-bit quantised offline model | ~2.2 GB, T4-ready | ✅ Complete |
| On-premise Puskesmas server | Local Linux server, no internet | 🔄 Roadmap |
| Android tablet deployment | Remote health posts | 🔄 Roadmap |
| Prospective pilot study | Real Puskesmas validation | 🔄 Roadmap |

---

## 📚 References

- WHO (2009). *Dengue: Guidelines for Diagnosis, Treatment, Prevention and Control.* Geneva: World Health Organization.
- Jayaratne et al. (2019). Neutrophil-to-lymphocyte ratio as a predictor of dengue severity. *PLOS ONE.*
- Indonesian Ministry of Health. Dengue Surveillance Data 2024–2025.
- BPJS Kesehatan. Hospitalization cost data (dengue admissions).
- Mendeley Dataset: [Dengue Fever Hematological Dataset](https://data.mendeley.com/datasets/6fsrsk3mb8/2)
- Mendeley Dataset: [Predictive Clinical Dataset for Dengue Fever](https://data.mendeley.com/datasets/xrsbyjs24t/1)

---

## ⚠️ Disclaimer

DeBe is a clinical decision **support** system intended for use by trained healthcare workers. It is not a replacement for physician judgement. All outputs should be interpreted by qualified medical personnel in the context of the full clinical picture.

*Alat ini adalah sistem pendukung keputusan klinis untuk tenaga kesehatan terlatih — bukan pengganti penilaian dokter.*

---

## 👤 Author

**Arum Puri** — AI Engineer
MedGemma Impact Challenge 2026

- 🤗 HuggingFace: [arumpuri](https://huggingface.co/arumpuri)
- 🐙 GitHub: [arumpuri](https://github.com/arumpuri)

---

---

*Built with ❤️ for Indonesia's frontline health workers, using [Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations).*
