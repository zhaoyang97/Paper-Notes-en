---
title: >-
  [Paper Note] ViLU: Learning Vision-Language Uncertainties for Failure Prediction
description: >-
  [ICCV2025][Information Retrieval & RAG][Uncertainty Quantification] This paper proposes ViLU, a post-hoc uncertainty quantification framework for VLM zero-shot prediction. By fusing visual embeddings…
tags:
  - "ICCV2025"
  - "Information Retrieval & RAG"
  - "Uncertainty Quantification"
  - "Failure Prediction"
  - "VLM"
  - "Cross-Attention"
  - "Post-Hoc Estimation"
date: 2026-05-08
content_hash: 82364fe62c36842a
---

# ViLU: Learning Vision-Language Uncertainties for Failure Prediction

**Conference**: ICCV2025
**arXiv**: [2507.07620](https://arxiv.org/abs/2507.07620)  
**Code**: [GitHub](https://github.com/ViLU-uncertainty/ViLU)  
**Area**: Information Retrieval
**Keywords**: Uncertainty Quantification, Failure Prediction, VLM, Cross-Attention, Post-Hoc Estimation

## TL;DR

This paper proposes ViLU, a post-hoc uncertainty quantification framework for VLM zero-shot prediction. By fusing visual embeddings, predicted text embeddings, and image-conditioned text representations via cross-attention, ViLU constructs uncertainty-aware multimodal representations that significantly outperform existing failure prediction methods across 13 classification datasets and large-scale image-text datasets.

## Background & Motivation

Vision-language models (VLMs, e.g., CLIP) achieve strong performance in zero-shot classification, yet reliably quantifying the uncertainty of their predictions remains an open challenge. Robust uncertainty quantification (UQ) is critical in safety-sensitive applications.

Limitations of existing methods:

**Maximum Concept Matching (MCM)**: The VLM analogue of maximum class probability. While simple and effective, it is inherently prone to assigning high confidence to incorrect predictions. For example, when "American Eskimo dog" is misclassified as "Siberian husky," MCM still yields a high confidence score and fails to detect the error.

**Learning Visual Uncertainties (LVU)**: Learns a predictor for the classifier's loss value using visual features alone. When applied to VLMs, it does not model relationships among downstream concepts, limiting failure prediction capability. LVU similarly cannot distinguish the aforementioned dog breed error.

**BayesVLM**: Models embedding uncertainty via Laplace approximation, but is not optimized specifically for failure prediction.

Core insight: Uncertainty in VLMs originates from **two modalities** — ambiguity in visual patterns and ambiguity among textual concepts. Effective UQ must jointly model both sources of uncertainty and their interactions.

## Method

### Overall Architecture

ViLU is a **post-hoc** framework that operates solely on the visual and text embeddings output by a VLM, requiring no access to internal model parameters. The framework comprises three core components:

1. Vision-text cross-attention module
2. ViLU embedding construction
3. Failure prediction classification head

### Key Design 1: Vision-Text Cross-Attention

Given a visual embedding $\bm{z}_v$ and embeddings $Z_t = \{\bm{z}_{t_j}\}_{1 \leq j \leq K}$ for $K$ candidate text concepts, an image-conditioned text representation is produced via cross-attention:

$$\bm{z}_t^\alpha = h_{\theta_{\text{XA}}}(\bm{z}_v, \bm{z}_{t_1}, ..., \bm{z}_{t_K})$$

Specifically, the visual representation serves as the Query and text embeddings as Keys/Values:

$$\bm{\alpha} = \text{softmax}\left(\frac{(W_Q \bm{z}_v)^\top (W_K Z_t)}{\sqrt{d}}\right), \quad \bm{z}_t^\alpha = \sum_{j=1}^K \bm{\alpha}_j (W_V \bm{z}_{t_j})$$

This attention-weighted text embedding re-contextualizes candidate concepts according to the model's predicted distribution, enabling ViLU to capture fine-grained inter-concept ambiguity.

### Key Design 2: ViLU Embedding

A triplet uncertainty embedding is constructed as:

$$\bm{z}_{\text{ViLU}} = (\bm{z}_v, \bm{z}_{\hat{t}}, \bm{z}_t^\alpha)$$

- $\bm{z}_v$: visual embedding (captures visual ambiguity)
- $\bm{z}_{\hat{t}}$: predicted text embedding (the model's best guess, approximating MCM information)
- $\bm{z}_t^\alpha$: cross-attention output (captures weighted relationships among all candidate concepts)

Using only the first two components approximates the behavior of MCM but ignores confounding alternative concepts; incorporating $\bm{z}_t^\alpha$ effectively captures multi-concept ambiguity.

### Key Design 3: Failure Prediction Objective

Rather than predicting the classifier's loss value as in conventional UQ methods (a regression task), ViLU formulates the problem as a **binary classification task** — directly distinguishing correct from incorrect predictions:

$$\hat{y}_i = \sigma(g_{\theta_{\text{MLP}}}(\bm{z}_{\text{ViLU}}))$$

Training employs a weighted binary cross-entropy loss:

$$\mathcal{L}_{\text{wBCE}} = -\frac{1}{B}\sum_i \left[w y_i \log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

where the weight $w$ is adaptively adjusted based on the ratio of correct to incorrect samples in each mini-batch:

$$w = \log\left(1 + \frac{\sum_i(1-y_i)}{\sum_i y_i}\right)$$

This design renders ViLU fully **loss-agnostic**: it does not require knowledge of the VLM's pretraining loss (contrastive or sigmoid), making it particularly suitable for post-hoc settings with black-box VLMs.

## Key Experimental Results

### Failure Prediction on Image Classification Datasets (CLIP ViT-B/32)

| Method | Avg. AUC↑ | Avg. FPR95↓ |
|--------|----------|------------|
| MCM | 81.8 | 70.6 |
| Entropy | 80.2 | 74.0 |
| Doctor | 81.6 | 71.2 |
| Rel-U | 80.7 | 68.6 |
| LVU | 85.0 | 57.4 |
| BayesVLM | 84.2 | 65.1 |
| **ViLU** | **93.2** | **29.9** |

ViLU outperforms all baselines in average AUC across 13 datasets (+8.2 vs. LVU) and reduces FPR95 by 27.5 percentage points. On Flowers102, it achieves 98.7% AUC / 5.1% FPR95.

### Large-Scale Image-Text Datasets

| Method | CC3M AUC↑ | CC12M AUC↑ | LAION-400M AUC↑ |
|--------|----------|-----------|----------------|
| MCM | 83.9 | 88.8 | 91.7 |
| LVU | 69.3 | 74.4 | 80.2 |
| BayesVLM | 87.1 | 90.9 | 95.1 |
| **ViLU** | **91.4** | **95.2** | **97.3** |

In open-vocabulary settings (CC12M, LAION-400M), LVU performs even worse than MCM, demonstrating that modeling visual uncertainty alone is insufficient. ViLU achieves an FPR95 of only 25.2% on CC12M (vs. BayesVLM: 53.3%).

### Ablation Study

| Visual Emb. | Cross-Attn. | Pred. Text | CIFAR-10 AUC | ImageNet AUC | CC12M AUC |
|-------------|-------------|------------|-------------|-------------|----------|
| ✓ | ✗ | ✗ | 96.4 | 78.7 | 74.0 |
| ✓ | ✗ | ✓ | 97.9 | 88.8 | 88.9 |
| ✓ | ✓ | ✗ | 97.7 | 86.1 | 93.6 |
| ✓ | ✓ | ✓ | **98.3** | **89.5** | **95.2** |
| MCM | - | - | 89.9 | 80.8 | 88.8 |

- Adding the predicted text embedding yields +10 AUC on ImageNet and +14.9 AUC on CC12M.
- Cross-attention improves CC12M from 88.9 to 95.2 (+6.3), as text concepts in that dataset vary dynamically across batches.
- Binary BCE loss substantially outperforms regression MSE loss (ImageNet AUC: 89.5 vs. 85.7).

## Highlights & Insights

1. **Precise problem formulation**: The paper clearly identifies that the core challenge of VLM UQ lies in jointly modeling visual ambiguity and inter-concept textual ambiguity.
2. **Elegant cross-attention design**: Using the visual representation as a Query to attend over all candidate text embeddings naturally produces image-conditioned text representations and supports a variable number of candidate concepts.
3. **Binary classification outperforms loss regression**: Directly framing UQ as failure prediction (binary classification) is more effective than predicting loss values (regression), as the latter requires knowledge of the VLM's pretraining loss.
4. **Loss-agnostic post-hoc design**: Operating on embeddings alone, without access to VLM weights or training details, makes ViLU applicable to black-box settings.
5. **Robust under low-accuracy regimes**: While MCM and BayesVLM degrade sharply when VLM zero-shot accuracy is low (e.g., 64.1% AUC on EuroSAT), ViLU maintains 90.1% AUC.

## Limitations & Future Work

- ViLU requires training a separate model for each target dataset (though it is data-efficient — using only 2.5% of ImageNet data suffices to surpass MCM).
- Cross-dataset zero-shot transfer (training on CC12M and evaluating on other datasets) leaves room for improvement.
- Main experiments are conducted on CLIP ViT-B/32; applicability to larger-scale VLMs (e.g., EVA-CLIP, SigLIP) remains to be verified.
- The computational complexity of cross-attention scales linearly with the number of candidate concepts $K$.

## Related Work & Insights

- **Output distribution methods**: MCM, Entropy, Doctor, Rel-U — training-free but limited in expressiveness.
- **Data-driven predictors**: ConfidNet, LVU — learn only visual uncertainty, ignoring the language modality.
- **VLM-specific UQ**: ProbVLM (probabilistic embedding adapter), BayesVLM (Laplace approximation) — not optimized for failure prediction.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Technical Quality | 5 |
| Experimental Thoroughness | 5 |
| Writing Quality | 5 |
| Value | 4 |
| Overall | 4.6 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation](aligning_information_capacity_between_vision_and_language_via_dense-to-sparse_fe.md)
- [\[NeurIPS 2025\] Scaling Language-Centric Omnimodal Representation Learning](../../NeurIPS2025/information_retrieval/scaling_language-centric_omnimodal_representation_learning.md)
- [\[ICCV 2025\] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation for Image-Text Matching](aligning_information_capacity_between_vision_and_language_via_dense_to_sparse_feature_distillation.md)
- [\[AAAI 2026\] HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment](../../AAAI2026/information_retrieval/himo-clip_modeling_semantic_hierarchy_and_monotonicity_in_vi.md)
- [\[ICLR 2026\] Revela: Dense Retriever Learning via Language Modeling](../../ICLR2026/information_retrieval/revela_dense_retriever_learning_via_language_modeling.md)

</div>

<!-- RELATED:END -->
