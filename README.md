# Language as a Hidden Variable: Measuring Behavioral Divergence in Multilingual LLMs

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange.svg)](https://colab.research.google.com/)
[![API](https://img.shields.io/badge/API-Groq-green.svg)](https://console.groq.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Overview

This repository contains the complete experimental pipeline for **Project 1** of the course *Fundamentals of NLP (CS3233)*, RV University, Bengaluru.

> **Core Claim:** When the same prompt is issued to a multilingual LLM in different languages, the model produces systematically divergent responses — not randomly, but in ways that are category-dependent, language-pair-dependent, and partially model-consistent. Language itself acts as a **hidden variable** that silently modulates model behavior even when prompt semantics are held constant.

---

## Members

| Name | USN |
|---|---|
| Aman Chandra H | 1RVU23CSE039 |
| S Nithin | 1RVU23CSE388 |
| Sameer Kulkarni | 1RVU23CSE404 |

**Guide:** Prof. Khushboo Pandey

---

## Repository Structure

```
.
├── notebooks/              ← 4 sequential Colab notebooks (NB0 → NB3)
│   ├── NB0_Setup_Translation.ipynb
│   ├── NB1_Data_Collection.ipynb
│   ├── NB2_Metrics.ipynb
│   └── NB3_Analysis.ipynb
│
├── figures/                ← 8 publication-quality PNG figures (300 DPI)
│   ├── fig1_category_bds1.png
│   ├── fig2_prompt_heatmap.png
│   ├── fig3_sentiment_divergence.png
│   ├── fig4_refusal_distribution.png
│   ├── fig5_variance_baseline.png
│   ├── fig6_model_consistency.png
│   ├── fig7_bds1_vs_bdsfull.png
│   └── fig8_sentiment_polarity.png
│
├── results/                ← All output files from NB2 and NB3
│   ├── paper_results.txt
│   ├── case_studies.txt
│   ├── statistical_tests.csv
│   ├── descriptive_stats.csv
│   ├── sanity_check_correlation.txt
│   ├── kappa_score.txt
│   └── figures/            ← Mirror of /figures (Drive-synced copies)
│
└── README.md               ← This file
```

---

## Experimental Design

| Dimension | Values |
|---|---|
| **Languages** | English (baseline), Hindi, French |
| **Models** | Llama-3.3-70B-Versatile, GPT-OSS-20B (via Groq) |
| **Prompt Categories** | Factual (×10), Normative (×10), Safety-sensitive (×10) |
| **Total Prompts** | 30 |
| **Main API Calls** | 30 × 3 languages × 2 models = **180** |
| **Baseline Calls** | 30 × 3 runs × 2 models = **180** |
| **Sanity Check Calls** | 10 × 2 pairs × 2 models = **40** |
| **Total Responses** | **360** (+ 40 sanity) |
| **Inference Params** | temperature=0.1, max_tokens=512, top_p=1.0 |

---

## Metrics

### BDS-1 — Primary Semantic Divergence
```
BDS-1 = 1 - cosine_similarity(SBERT_embedding_EN, SBERT_embedding_target)
```
Model: `paraphrase-multilingual-mpnet-base-v2`

### S_sent — Sentiment Divergence
```
S_sent = |sentiment_score_EN - sentiment_score_target|
```
Model: `cardiffnlp/twitter-xlm-roberta-base-sentiment`

### S_ref — Refusal Mismatch
```
S_ref = 1 if refusal_class_EN ≠ refusal_class_target, else 0
```
Classes: 0 = Full Answer, 1 = Partial, 2 = Explicit Refusal

### BDS-full — Composite (pre-registered weights)
```
BDS-full = 0.5 × BDS-1  +  0.3 × S_sent  +  0.2 × S_ref
```

---

## Key Results at a Glance

| Metric | EN-HI | EN-FR |
|---|---|---|
| Mean BDS-1 | 0.225 ± 0.246 | 0.264 ± 0.291 |
| Mean BDS-full | 0.190 | 0.221 |
| Mean S_sent | 0.170 | 0.220 |
| Safety refusal mismatch | 10.0% | 10.0% |
| Normative refusal mismatch | 30.0% | 25.0% |
| Cross-lang divergence vs noise | **2.5× (Safety)** | **2.4× (Factual)** |

**Hypotheses:** H1 (Safety > Factual divergence): Not supported · H2 (Hindi > French divergence): Not supported · H3 (Model consistency): Not supported — all null results are discussed as substantive findings in the paper.

**Annotation quality:** Cohen's κ = 1.00 (perfect inter-annotator agreement, n=180)  
**SBERT validity:** Spearman r = 0.727 (p = 0.0006) vs. LLM judge

---

## How to Reproduce

### Prerequisites
- Google account with Google Drive
- Groq API key (free tier at [console.groq.com](https://console.groq.com))
- `prompts_master_csv_utf.csv` and `refusal_labels.csv` in your Drive (see notebooks/README)

### Execution Order

```
NB0 → NB1 → [Manual: fill refusal_labels.csv] → NB2 → NB3
```

Each notebook is self-contained and saves its outputs to Google Drive before the next one begins. **Never skip steps or re-run NB1 without backing up `raw_responses.csv` first.**

Full instructions are in [`notebooks/README.md`](notebooks/README.md).

---

## Paper

The IEEE-format conference paper (8 pages, double-column) is written based on the outputs of this pipeline. All figures referenced in the paper correspond directly to files in the [`figures/`](figures/) folder.

**Venue target:** ICON 2025 (International Conference on Natural Language Processing)

---

## Citation

If you use this codebase or experimental framework, please cite:

```
@inproceedings{chandra2025language,
  title     = {Language as a Hidden Variable: Measuring Behavioral Divergence
               in Multilingual Large Language Models},
  author    = {Aman Chandra H , Sameer Kulkarni and S Nithin},
  booktitle = {Proceedings of ICON 2025},
  year      = {2025},
  institution = {RV University, Bengaluru}
}
```
