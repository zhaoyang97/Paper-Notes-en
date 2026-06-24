---
title: >-
  [Paper Note] DiN: Diffusion Model for Robust Medical VQA with Semantic Noisy Labels
description: >-
  [CVPR 2025][Medical Imaging][Medical VQA] This paper proposes the DiN framework, applying diffusion models to the noisy-label medical VQA (NM-VQA) scenario for the first time. Through a diffusion-based answer classifier, it screens answers from coarse to fine from a generative perspective. Combined with a noisy label refinement module to dynamically correct labels, DiN achieves an accuracy of 74.24% on VQA-RAD under 10% semantic noise, outperforming SNLC's 69.65%.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Medical VQA"
  - "Diffusion Models"
  - "Semantic Noisy Labels"
  - "Label Denoising"
  - "Multimodal Fusion"
date: 2026-05-08
content_hash: fcafb9dab18f7c54
---

# DiN: Diffusion Model for Robust Medical VQA with Semantic Noisy Labels

**Conference**: CVPR 2025  
**arXiv**: [2503.18536](https://arxiv.org/abs/2503.18536)  
**Code**: [Erjian96/DiN](https://github.com/Erjian96/DiN)  
**Area**: Medical Imaging  
**Keywords**: Medical VQA, Diffusion Models, Semantic Noisy Labels, Label Denoising, Multimodal Fusion

## TL;DR

This paper proposes the DiN framework, applying diffusion models to the noisy-label medical VQA (NM-VQA) scenario for the first time. Through a diffusion-based answer classifier, it screens answers from coarse to fine from a generative perspective. Combined with a noisy label refinement module to dynamically correct labels, DiN achieves an accuracy of 74.24% on VQA-RAD under 10% semantic noise, outperforming SNLC's 69.65%.

## Background & Motivation

Medical VQA tasks aim to answer clinical questions by integrating medical images and textual information. Current challenges include:

1. **Ignored Label Noise**: Medical annotation requires professional expertise, and inter-annotator agreement is low. However, existing Med-VQA methods (e.g., MMBERT, Q2ATransformer) assume clean training labels.
2. **Inapplicability of Traditional Noise Models**: Symmetric/asymmetric noise models do not match medical scenarios, where annotator errors are typically "semantically similar mistakes" rather than random noise (e.g., mislabeling "pneumonia" as "pneumonitis").
3. **Limitations of Prior Work**:
    - NTM (Noise Transition Matrix): Difficult to estimate when the number of Med-VQA categories is large.
    - Co-teaching: Dual-branch networks double the parameters, incurring higher overhead in Med-VQA models that already contain image and text encoders.
    - SNLC (Noise Method for Natural VQA): Uses only robust contrastive loss, resulting in coarse handling of medical semantic noise.

**Key Challenge**: Classification methods are limited by predefined categories but robust to noise, whereas generative methods are flexible but may generate non-existent answers. This paper bridges the two using a diffusion model.

**Key Insight**: Frame classification as diffusion—the forward process adds noise to the answer distribution, while the reverse process progressively refines it back to the correct answer. This constrains the output to the predefined answer space (classification) while allowing flexible distribution adjustments through step-by-step refinement (generation).

**Core Idea**: Utilize a conditional diffusion model to perform "noising-denoising" on answer probability distributions for classification, combined with BERT-nearest-neighbor-based semantic noise simulation, robust focal loss, and dynamic pseudo-labels for noisy label refinement.

## Method

### Overall Architecture

DiN consists of three core modules:
1. **Answer Condition Generator (ACG)**: Generates answer-aware conditional features.
2. **Noisy Label Refinement (NLR)**: Refines noisy labels into pseudo-labels (used only during training).
3. **Answer Diffuser (AD)**: A conditional diffusion model-based answer classifier.

### Key Designs

**1. Semantic Noise Benchmark Construction**

- For closed-end questions (Yes/No): Symmetric noise is used to randomly flip answers.
- For open-end questions: A pretrained BERT model is used to construct an answer semantic space, and ground truth is replaced with semantically closest noisy labels.
- Evaluation settings: 10% and 20% semantic noise ratios.

This semantic noise is significantly more challenging than random noise—under the same 20% noise rate, SNLC achieves 62% accuracy under random noise but drops to 58% under semantic noise.

**2. Answer Condition Generator (ACG)**

- Image features $f_k^v$ are extracted using a visual encoder (Swin Transformer), and question features $f_k^q$ are extracted using a text encoder (BERT).
- Key and Value embeddings are generated post-fusion.
- $L$ learnable candidate Answer Condition Embeddings are introduced to capture relationships between answers via self-attention.
- Cross-attention interacts the answer embeddings with the image-question features to generate the conditional feature $f_k^c$.

**3. Noisy Label Refinement (NLR)**

Includes two sub-strategies:

**a) Robust Focal Loss (RFL)**: Integrates symmetric cross-entropy with focal loss to simultaneously resist noise and mitigate class imbalance:
$$\mathcal{L}_{RFL} = -\sum_{l=1}^{L} a_k(1-\hat{p}_k)^\gamma \log\hat{p}_k - \sum_{l=1}^{L} \hat{p}_k \log a_k$$

**b) Answer Adaptation (AA)**:
- If the auxiliary classifier's prediction $\hat{p}_k$ matches the original label $a_k$ $\rightarrow$ Trust the original label.
- Otherwise, generate soft pseudo-labels: $\bar{y}_k = w_t \hat{p}_k + (1-w_t) a_k$.
- The weight $w_t$ tracks the batch average confidence via EMA ($\tau=0.99$), progressively increasing reliance on model predictions as training proceeds.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{dif} + \alpha \mathcal{L}_{RFL}$$

- $\mathcal{L}_{dif}$: MSE loss supervising the AD output to match the pseudo-label distribution.
- $\mathcal{L}_{RFL}$: Robust Focal Loss used to train the auxiliary classifier.
- $\alpha = 0.5$

## Key Experimental Results

### Main Results

| Noise Type | Method | Open | Close | Overall |
|----------|------|------|-------|---------|
| 10%-Semantic | Baseline | 67.86 | 68.85 | 68.25 |
| 10%-Semantic | MMBERT | 63.36 | 68.21 | 66.42 |
| 10%-Semantic | CoDis | 68.02 | 70.54 | 69.53 |
| 10%-Semantic | SNLC | 67.83 | 71.35 | 69.65 |
| **10%-Semantic** | **DiN (Ours)** | **72.68** | **75.81** | **74.24** |
| 20%-Semantic | Baseline | 54.23 | 57.14 | 56.01 |
| 20%-Semantic | SNLC | 56.45 | 61.30 | 58.88 |
| **20%-Semantic** | **DiN (Ours)** | **58.06** | **64.52** | **63.17** |

- Outperforms SNLC by **+4.59%** and CoDis by **+4.71%** under 10% semantic noise.
- With a Clean Label upper bound of 79.13%, DiN reaches 74.24% under 10% noise, a gap of only 4.89 percentage points.
- Semantic noise is more challenging than random noise (63.17 vs 63.93 under 20%), validating the value of semantic noise simulation.

### Ablation Study

| Configuration | Effect |
|------|------|
| Remove AD module | Diffusion classification is the core driver of performance improvement |
| Remove RFL | Performance drops significantly, reducing noise robustness |
| Remove AA strategy | Pseudo-label quality decreases |
| ACG condition quality | Directly affects diffusion accuracy |

## Highlights & Insights

1. **Forward-looking Problem Definition**: First systematic study of the noisy label issue in Med-VQA, introducing a semantic noise benchmark that is closer to real clinical annotation errors.
2. **Novel Use of Diffusion Models**: Porting diffusion models from generative tasks to classification tasks, utilizing their progressive refinement property to handle label noise—conceptually elegant and highly effective.
3. **Adaptive Weight Design in NLR**: Tracking confidence via EMA allows trusting the original labels early in training and leaning towards model predictions later, preventing early error accumulation.
4. **Inference Efficiency**: Discarding the NLR module during inference incurs no additional parameter overhead.

## Limitations & Future Work

1. Inference in diffusion models requires multi-step denoising ($T$ steps), increasing inference time, which is less suitable for real-time clinical systems.
2. The construction of semantic noise pairs relies on a pretrained BERT model, which may have insufficient coverage of specialized medical terminology.
3. Validated only on VQA-RAD and PathVQA, both of which are relatively small-scale datasets.
4. The candidate answer set must be predefined, failing to handle completely open-ended answers.

## Related Work & Insights

- **Med-VQA Classification Methods**: MMBERT $\rightarrow$ Q2ATransformer $\rightarrow$ MMQ
- **Noisy Label Learning**: SimT (NTM) $\rightarrow$ CoDis (Co-teaching) $\rightarrow$ SNLC (noise method for natural VQA) $\rightarrow$ DivideMix
- **Diffusion Classifiers**: CARD (conditional label generation) $\rightarrow$ DiffusionDet (detection) — DiN applies it to VQA for the first time.
- **Standard Med-VQA**: Assumes clean labels, whereas DiN achieves performance close to clean labels even under 20% noise.

## Rating

- **Novelty**: 5/5 — Pioneering combination of diffusion classification, semantic noise, and Med-VQA.
- **Effectiveness**: 4/5 — Consistently outperforms baseline methods across multiple noise levels on two datasets.
- **Clarity**: 4/5 — Clear description of the three-module synergy and convincing analysis of noise types.
- **Significance**: 4/5 — Fills the gap in noisy label research for Med-VQA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation](sealion_semantic_part-aware_latent_point_diffusion_models_for_3d_generation.md)
- [\[CVPR 2025\] TopoCellGen: Generating Histopathology Cell Topology with a Diffusion Model](topocellgen_generating_histopathology_cell_topology_with_a_diffusion_model.md)
- [\[CVPR 2026\] MedFG-VQA: Low-Frequency Memory and Graph Attention for Lightweight Medical VQA](../../CVPR2026/medical_imaging/medfg-vqa_low-frequency_memory_and_graph_attention_for_lightweight_medical_vqa.md)
- [\[CVPR 2025\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](../../NeurIPS2025/medical_imaging/semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)

</div>

<!-- RELATED:END -->
