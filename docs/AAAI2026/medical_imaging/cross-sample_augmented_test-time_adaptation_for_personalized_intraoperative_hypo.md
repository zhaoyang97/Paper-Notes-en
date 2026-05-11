---
title: >-
  [Paper Note] Cross-Sample Augmented Test-Time Adaptation for Personalized Intraoperative Hypotension Prediction
description: >-
  [AAAI 2026][Medical Imaging][intraoperative hypotension prediction] This paper proposes the CSA-TTA framework, which enhances personalized intraoperative hypotension prediction at test time by constructing a cross-sample…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "intraoperative hypotension prediction"
  - "test-time adaptation"
  - "cross-sample retrieval"
  - "time series forecasting"
  - "personalized medicine"
date: 2026-05-08
content_hash: 6e517ef4b39122a7
---

# Cross-Sample Augmented Test-Time Adaptation for Personalized Intraoperative Hypotension Prediction

**Conference**: AAAI 2026
**arXiv**: [2512.15762](https://arxiv.org/abs/2512.15762)
**Code**: [GitHub](https://github.com/kanxueli/CSA-TTA)
**Area**: Medical Imaging
**Keywords**: intraoperative hypotension prediction, test-time adaptation, cross-sample retrieval, time series forecasting, personalized medicine

## TL;DR

This paper proposes the CSA-TTA framework, which enhances personalized intraoperative hypotension prediction at test time by constructing a cross-sample bank, performing coarse-to-fine retrieval, and applying multi-task optimization to retrieve hypotension event signals from other patients' data.

## Background & Motivation

- **Intraoperative Hypotension (IOH)**: MAP < 65 mmHg lasting ≥ 1 minute, which can lead to acute kidney injury, myocardial infarction, stroke, or death.
- Accurate IOH prediction is critical for early intraoperative intervention, yet inter-patient physiological variability is substantial.
- Limitations of existing methods:
    - Population-level models such as CMA (attention mechanism) and HMF (multi-feature fusion) fail to capture individual differences.
    - Clinical interventions (anesthesia, drug administration) introduce covert distribution shifts, degrading generalization of population models.
- **Test-Time Adaptation (TTA)** is a promising direction: TTT and TTT++ fine-tune models at inference time via self-supervised auxiliary tasks.
- **TTA challenges in IOH**: hypotensive events are extremely sparse.
    - In the VitalDB dataset, hypotension accounts for only 12.6% of samples.
    - Most patients experience hypotension during less than 10% of surgery time.
    - Standard TTA relies on single-sample adaptation, failing to capture sudden blood pressure drops and producing overly smooth predictions.
- **Core Insight**: Patients with similar physiological characteristics exhibit similar intraoperative responses → hypotensive events from other patients can be retrieved to enrich the adaptation signal.

## Method

### Overall Architecture

CSA-TTA consists of three core steps (as shown in Figure 2):

1. **Cross-Sample Bank Construction**
2. **Coarse-to-Fine Retrieval**
3. **Multi-task Optimization**

Two operational modes are supported: fine-tuning mode (offline fine-tuning followed by TTA) and zero-shot mode (direct TTA without prior fine-tuning).

### Key Designs

**1. Cross-Sample Bank Construction**

Physiological time series from all patients in the historical dataset are segmented into fixed-length clips to form the cross-sample bank $\mathcal{B}$:

$$\mathcal{B} = \mathcal{B}_{\text{hypo}} \cup \mathcal{B}_{\text{non-hypo}}$$

- Clips are partitioned into two subsets based on whether they contain hypotensive events.
- Hypotension is defined as MAP < 65 mmHg lasting ≥ 1 minute.
- This partition enables targeted retrieval of hypotensive samples during adaptation.

**2. Coarse-to-Fine Retrieval Strategy**

An adaptive context window is used to process streaming patient data. At time step $t$, the historical window $\mathcal{W}_{t-m:t}^{\text{hist}}$ is defined.

**Coarse-grained retrieval**:
- K-Shape clustering is applied separately to the hypotensive subset $\mathcal{B}_{\text{hypo}}$ and the non-hypotensive subset.
- K-Shape relies on shape-based matching, making it well suited to physiological time series (no explicit temporal alignment or amplitude normalization required).
- A query sample is first assigned to a category, then matched to the most similar cluster centroid, rapidly narrowing the search space.

**Fine-grained retrieval**:
- Within the cluster identified by coarse retrieval, Dynamic Time Warping (DTW) is used to compute semantic similarity.
- The top-$K$ most similar samples are selected to form the candidate set $\mathcal{D}_{\text{retrieval}}$.
- Retrieved samples are augmented with perturbations (Gaussian noise, time scaling) to increase diversity.

**Constructing the adaptation dataset**:

$$\mathcal{D}_t^{\text{CSA-TTA}} = \mathcal{W}_{t-m:t}^{\text{hist}} \cup \text{Aug}(\mathcal{D}_{\text{retrieval}})$$

**3. Multi-task Optimization**

The model $F_\theta = (f_\theta, h_\theta, g_\theta)$ comprises:
- A shared feature encoder $f_\theta$
- A prediction branch $h_\theta$ (primary task: sequence prediction)
- A self-supervised branch $g_\theta$ (auxiliary task: masked reconstruction)

Total loss:

$$\min_{f_\theta, h_\theta, g_\theta} \frac{1}{N} \sum_{n=1}^{N} \mathcal{L}_{\text{Pred}}(X_n, Y_n; f_\theta, h_\theta) + \mathcal{L}_{\text{Recon}}(X_n; f_\theta, g_\theta)$$

- Masked reconstruction enhances time series representation learning and helps capture subtle signal variations.
- Retrospective sequence prediction uses known historical data for self-supervised training.

### Loss & Training

- **Partial fine-tuning strategy**: only input layers, output layers, and LayerNorm parameters are updated, balancing adaptability and generalization.
- Fine-tuning mode: 1 epoch of TTA updates; zero-shot mode: 3 epochs.
- Offline fine-tuning: 10 epochs, lr = 1e-4, batch size = 64, dropout = 0.01.
- **Hybrid trigger mechanism** for IOH prediction:
    - Hard trigger: detects sustained hypotensive episodes.
    - Soft trigger: average risk assessment within a sliding window.
    - The combination produces the final probability estimate.

## Key Experimental Results

### Main Results (Zero-shot & Fine-tuning)

**Dataset**: VitalDB (2,150 non-cardiac surgeries, 2s/30s sampling) + in-house dataset (130 test cases).

**Zero-shot setting** (VitalDB 30S):

| Model | F1↑ | Recall↑ | MAE↓ | MSE↓ |
|-------|-----|---------|------|------|
| TimesFM | 64.17 | 58.87 | 6.49 | 92.77 |
| TimesFM + CSA-TTA | **64.90** | **59.27** | **6.28** | **85.28** |
| UniTS | 52.23 | 43.24 | 7.32 | 99.96 |
| UniTS + CSA-TTA | **57.30** | **50.70** | **7.19** | **95.84** |

UniTS + CSA-TTA: Recall +7.46%, F1 +5.07%.

**Fine-tuning setting** (VitalDB 2S):

| Model | F1↑ | Recall↑ | MAE↓ | MSE↓ |
|-------|-----|---------|------|------|
| TimesFM | 64.20 | 64.93 | 6.03 | 77.87 |
| TTT | 64.00 | 64.77 | 6.02 | 77.70 |
| TTT++ | 64.10 | 64.80 | 6.02 | 77.68 |
| CSA-TTA | **64.83** | **65.99** | **5.94** | **76.19** |

**In-house dataset** (zero-shot):

| Model | F1↑ | Recall↑ | MAE↓ | MSE↓ |
|-------|-----|---------|------|------|
| UniTS | 56.10 | 43.77 | 6.45 | 91.97 |
| UniTS + CSA-TTA | **63.80** | **53.33** | **6.30** | **88.69** |

Recall +9.56%, F1 +7.70%.

### Ablation Study

**Multi-task optimization** (TimesFM fine-tuned, 5-minute prediction):

| Pred | Recon | F1 | MAE | MSE |
|------|-------|----|-----|-----|
| ✗ | ✓ | 70.00 | 4.82 | 55.81 |
| ✓ | ✗ | 70.60 | 4.79 | 54.08 |
| ✓ | ✓ | **70.60** | **4.77** | **53.17** |

**Data augmentation strategy** (fine-tuning setting):

| Bank | Aug | F1 | MAE | MSE |
|------|-----|----|-----|-----|
| ✗ | ✗ | 65.90 | 5.87 | 74.79 |
| ✓ | ✗ | 66.03 | 5.85 | 74.19 |
| ✓ | ✓ | **66.07** | **5.82** | **72.93** |

**Top-K selection**: $K=3$ yields F1 = 64.90 and MSE = 85.28, balancing relevance and diversity.

### Key Findings

- CSA-TTA yields larger gains on weaker models (UniTS zero-shot F1 +5.07%), suggesting that cross-sample signals are particularly valuable for underfitting models.
- The combination of a cross-sample bank and perturbation augmentation outperforms either component used alone.
- Case studies show that CSA-TTA captures abrupt blood pressure drops and rebounds that standard models smooth over.
- Computational overhead is manageable: only 1.06% of TimesFM parameters are updated, with approximately 6.6 seconds per epoch.

## Highlights & Insights

1. **First application of TTA to personalized IOH prediction**: addresses the failure of standard TTA under sparse event conditions.
2. **Elegant cross-sample retrieval design**: the two-stage coarse-to-fine pipeline balances efficiency and precision; the K-Shape → DTW combination is well suited to physiological signals.
3. **Plug-and-play compatibility**: applicable to arbitrary time series foundation models (TimesFM, UniTS), effective in both zero-shot and fine-tuning settings.
4. **Clinical relevance**: improvements in Recall directly reduce the miss rate of hypotensive events.

## Limitations & Future Work

- Zero-shot Precision occasionally degrades (e.g., UniTS on VitalDB 30S: Precision −4.58%), indicating that recall gains come at a partial cost to precision.
- The cross-sample bank requires sufficiently large historical data, which may be unavailable for newly established clinical units.
- DTW computational cost scales rapidly with data volume, potentially requiring approximate algorithms.
- Validation is limited to MAP prediction and has not been extended to other vital signs.
- The retrieval strategy assumes that similar physiological features imply similar responses, without accounting for specific surgical type or medication differences.

## Related Work & Insights

- TTT/TTT++ provide the foundational TTA paradigm; CSA-TTA extends this by incorporating external information sources.
- The application of K-Shape clustering to time series retrieval is worth adopting, as it is far more efficient than exhaustive DTW search.
- The cross-sample bank concept parallels Retrieval-Augmented Generation (RAG), adapted here to the time series adaptation setting.
- The partial parameter update strategy is generalizable to other online adaptation scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — The cross-sample TTA concept is original and addresses a genuine clinical pain point.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Two datasets, two backbone models, multiple ablations, and case analysis.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear and method description is complete.
- Value: ⭐⭐⭐⭐ — High practical value with direct applicability to clinical decision support.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains](mirage_scaling_test-time_inference_with_parallel_graph-retrieval-augmented_reaso.md)
- [\[ICCV 2025\] Progressive Test Time Energy Adaptation for Medical Image Segmentation](../../ICCV2025/medical_imaging/progressive_test_time_energy_adaptation_for_medical_image_segmentation.md)
- [\[AAAI 2026\] GP-MoLFormer-Sim: Test Time Molecular Optimization through Contextual Similarity Guidance](gp-molformer-sim_test_time_molecular_optimization_through_contextual_similarity_.md)
- [\[CVPR 2026\] SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation](../../CVPR2026/medical_imaging/spegc_continual_test-time_adaptation_via_semantic-prompt-enhanced_graph_clusteri.md)
- [\[AAAI 2026\] Provably Minimum-Length Conformal Prediction Sets for Ordinal Classification](provably_minimum-length_conformal_prediction_sets_for_ordinal_classification.md)

</div>

<!-- RELATED:END -->
