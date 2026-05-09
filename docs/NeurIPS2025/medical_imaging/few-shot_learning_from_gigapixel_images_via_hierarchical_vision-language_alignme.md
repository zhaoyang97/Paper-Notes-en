---
title: >-
  [Paper Note] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling
description: >-
  [NeurIPS 2025][Medical Imaging][Multiple Instance Learning] This paper proposes HiVE-MIL, a hierarchical vision-language MIL framework that constructs a unified heterogeneous graph to model cross-scale hierarchical relationships (5× and 20×) and intra-scale multimodal alignment. Combined with a text-guided dynamic filtering mechanism and a hierarchical contrastive loss, HiVE-MIL consistently outperforms existing methods under the 16-shot setting on three TCGA datasets (lung, breast, and renal cancer), achieving up to 4.1% improvement in Macro F1.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Multiple Instance Learning
  - Vision-Language Models
  - Hierarchical Graph
  - Whole Slide Images
  - Few-Shot Classification
  - Pathology
date: 2026-05-08
content_hash: 9240438f465fca80
---

# Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2505.17982](https://arxiv.org/abs/2505.17982)
**Code**: [GitHub](https://github.com/bryanwong17/HiVE-MIL)
**Area**: Computational Pathology / Few-Shot Learning
**Keywords**: Multiple Instance Learning, Vision-Language Models, Hierarchical Graph, Whole Slide Images, Few-Shot Classification, Pathology

## TL;DR

This paper proposes HiVE-MIL, a hierarchical vision-language MIL framework that constructs a unified heterogeneous graph to model cross-scale hierarchical relationships (5× and 20×) and intra-scale multimodal alignment. Combined with a text-guided dynamic filtering mechanism and a hierarchical contrastive loss, HiVE-MIL consistently outperforms existing methods under the 16-shot setting on three TCGA datasets (lung, breast, and renal cancer), achieving up to 4.1% improvement in Macro F1.

## Background & Motivation

**Whole Slide Image Classification**: WSIs exhibit gigapixel resolution and encode multi-scale spatial information spanning coarse tissue architecture to fine-grained cellular morphology. The scarcity of annotations necessitates weakly supervised MIL frameworks.

**Limitations of Conventional MIL**: These methods rely on large amounts of annotated data, depend solely on visual features, and are sensitive to staining variations and domain shifts, resulting in poor performance in few-shot scenarios.

**Progress and Bottlenecks of VLM-MIL**: Recent multi-scale VLM-MIL approaches introduce scale-specific prompts but suffer from two critical drawbacks:
   - **Insufficient cross-scale intra-modal interaction**: Visual and textual features at each scale are processed independently, with simple summation or averaging at the prediction stage, discarding the coarse-to-fine semantic hierarchy.
   - **Inadequate intra-scale cross-modal alignment**: Fine-grained alignment between visual and textual features within the same scale remains underexplored.

## Method

### 1. Multi-Scale Hierarchical Feature Extraction

**Visual hierarchy**: WSIs are processed at the low scale (5×) to extract $N$ patches $z_n^{(l)} = f_{\text{img}}(x_n^{(l)}) \in \mathbb{R}^D$; each low-scale patch is subdivided into $M = (20/5)^2 = 16$ high-scale (20×) sub-patches $z_r^{(h)} = f_{\text{img}}(x_{n,m}^{(h)})$.

**Text hierarchy**: GPT-4o is used to generate hierarchical text descriptions — $O=4$ low-scale descriptions per class (coarse tissue features, e.g., "acinar pattern"), each paired with $K=3$ high-scale sub-descriptions (fine-grained cellular features, e.g., "nuclear hyperchromatism"). CoOp-style learnable token prefixes of length $L=16$ are prepended:

$$t_o^{(l)} = [v_1^{(l)}] \dots [v_L^{(l)}] [\text{Low-scale Text}_o]$$

### 2. Text-Guided Dynamic Filtering (TGDF)

**Stage 1 (Low-scale filtering)**: A patch-text cosine similarity matrix $S^{(l)} \in \mathbb{R}^{N \times O}$ is computed. For each text $o$, the mean $\mu_o$ and standard deviation $\sigma_o$ are used to define a text-adaptive threshold:

$$S_{\text{filtered}}^{(l)}(n,o) = \mathbb{I}\left(S^{(l)}(n,o) \geq \mu_o + \alpha \cdot \sigma_o\right)$$

**Stage 2 (High-scale refinement)**: The same filtering is applied to high-scale sub-patches corresponding to retained low-scale patches, masked by the low-scale filtering results to ensure consistency: $S_{\text{masked}}^{(h)}(r,s) = S^{(h)}(r,s) \cdot S_{\text{filtered}}^{(l)}(n,o)$

### 3. Hierarchical Heterogeneous Graph (HHG)

Node types: $\mathcal{T} = \{\text{img}^{(l)}, \text{img}^{(h)}, \text{text}^{(l)}, \text{text}^{(h)}\}$

**Intra-scale edges**: Connect visual-text node pairs within the same scale based on the TGDF-filtered similarity matrix.

**Hierarchical edges**: Connect same-modality nodes across low and high scales (visual hierarchy + text hierarchy), based on spatial parent-child relationships.

### 4. Modality-Scale Attention (MSA)

An attention mechanism is applied to hierarchical edges to enhance cross-scale information propagation. Node features incorporate learnable scale embeddings, and QKV projections are relation-specific:

$$\beta_{vu} = \text{softmax}\left(\frac{q_v^\top k_u}{\sqrt{d}}\right), \quad h_v^{\text{hier}} = q_v + \sum_{u \in \mathcal{N}_r(v)} \beta_{vu} v_u$$

### 5. Training Objectives

**Hierarchical Text Contrastive Loss (HTCL)**: Aligns cross-scale text semantics, with positive pairs defined as same-class parent-child text pairs and negatives from different classes:

$$\mathcal{L}_{\text{HTCL}} = \frac{1}{N}\sum_{i=1}^{N}\left(-\frac{1}{|\mathcal{P}_s|}\sum_{j \in \mathcal{P}_s}\log\sigma(\text{sim}_{o,s}) - \frac{1}{|\mathcal{N}_s|}\sum_{j \in \mathcal{N}_s}\log\sigma(-\text{sim}_{o,s})\right)$$

**Total loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(z_i, y_i) + \lambda \mathcal{L}_{\text{HTCL}}$, with $\lambda = 0.5$

## Key Experimental Results

### 16-Shot Results (PLIP Feature Encoder)

| Method | NSCLC ACC | NSCLC F1 | BRCA ACC | BRCA F1 | RCC ACC | RCC F1 |
|------|----------|---------|---------|--------|--------|-------|
| ABMIL | 70.64 | 70.37 | 65.83 | 65.29 | 80.00 | 77.95 |
| TransMIL | 73.21 | 72.98 | 72.08 | 71.94 | 87.05 | 84.96 |
| MSCPT | 76.86 | 76.82 | 72.71 | 72.58 | 86.21 | 84.20 |
| ViLa-MIL | 74.17 | 73.90 | 71.04 | 70.56 | 85.06 | 82.51 |
| FOCUS | 71.73 | 71.65 | 71.66 | 71.36 | 87.82 | 85.54 |
| **HiVE-MIL** | **80.13** | **80.08** | **75.21** | **74.99** | **88.89** | **87.18** |
| Δ vs 2nd | +3.27 | +3.26 | +2.50 | +2.41 | +1.07 | +1.64 |

### Consistency Across VLM Encoders

HiVE-MIL achieves state-of-the-art performance consistently across three pathology VLMs: PLIP (208K image-text pairs), QuiltNet (1M image-text pairs), and CONCH.

### Ablation Study

- Removing TGDF → F1 drops by 1.5–3%, demonstrating the necessity of filtering weakly matched patch-text pairs.
- Removing hierarchical edges → significant performance degradation, validating the importance of cross-scale modeling.
- Removing HTCL → reduced cross-scale consistency of textual semantics.

## Highlights & Insights

1. ⭐⭐⭐ **Unified Hierarchical Heterogeneous Graph**: The first approach to simultaneously model cross-scale intra-modal hierarchical interactions and intra-scale cross-modal alignment in WSIs, yielding substantial gains over naive fusion strategies.
2. ⭐⭐⭐ **Text-Guided Dynamic Filtering**: A top-down two-stage filtering mechanism that effectively removes irrelevant patch-text pairs and is updated dynamically during training.
3. ⭐⭐ **Consistent and Substantial Improvements**: Outperforms all baselines across 3 datasets × 3 VLMs × multiple shot settings.
4. ⭐⭐ **Well-Motivated Design**: Each component (MSA, TGDF, HTCL) is thoroughly validated through ablation experiments.

## Limitations & Future Work

1. **Computational Overhead**: Constructing and processing the hierarchical heterogeneous graph — particularly the 16× high-scale patch expansion — introduces significant memory and computational costs.
2. **Dependence on GPT-4o for Text Generation**: The quality of hierarchical text descriptions relies on LLM outputs; generalizability across different cancer types requires further validation.
3. **Limited to Binary/Ternary Classification**: Experiments cover only 2–3 class settings; performance on finer-grained subtype classification remains unverified.
4. **Sensitivity of Threshold Hyperparameter $\alpha$**: Whether the filtering threshold $\alpha=0.5$ in TGDF is optimal across all datasets and VLMs is not thoroughly investigated.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

## Related Work & Insights

## Highlights & Insights

## Rating

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](../../AAAI2026/medical_imaging/gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)
- [\[NeurIPS 2025\] RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray](radzero_similarity-based_cross-attention_for_explainable_vision-language_alignme.md)
- [\[ICCV 2025\] GECKO: Gigapixel Vision-Concept Contrastive Pretraining in Histopathology](../../ICCV2025/medical_imaging/gecko_gigapixel_vision-concept_contrastive_pretraining_in_histopathology.md)
- [\[NeurIPS 2025\] Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment](multimodal_disease_progression_modeling_via_spatiotemporal_disentanglement_and_m.md)
- [\[ICCV 2025\] AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images](../../ICCV2025/medical_imaging/aczerots_active_learning_for_zeroshot_tissue_segmentation_in.md)

<!-- RELATED:END -->
