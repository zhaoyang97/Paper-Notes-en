---
title: >-
  [Paper Note] CLoE: Expert Consistency Learning for Missing Modality Segmentation
description: >-
  [CVPR 2026][Medical Imaging][Missing modality] This paper proposes CLoE (Consistency Learning of Experts), which reformulates missing-modality robustness as a decision-level expert consistency control problem. It reduces expert drift via two complementary consistency branches—Modality Expert Consistency (MEC) and Region Expert Consistency (REC)—and achieves reliability-weighted fusion through a consistency-score-driven gating network.
tags:
  - CVPR 2026
  - Medical Imaging
  - Missing modality
  - multimodal segmentation
  - consistency learning
  - brain tumor segmentation
  - reliability gating
date: 2026-05-08
content_hash: 00d683697fefbd13
---

# CLoE: Expert Consistency Learning for Missing Modality Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.09316](https://arxiv.org/abs/2603.09316)
**Code**: Unavailable
**Area**: Medical Imaging
**Keywords**: Missing modality, multimodal segmentation, consistency learning, brain tumor segmentation, reliability gating

## TL;DR

This paper proposes CLoE (Consistency Learning of Experts), which reformulates missing-modality robustness as a decision-level expert consistency control problem. It reduces expert drift via two complementary consistency branches—Modality Expert Consistency (MEC) and Region Expert Consistency (REC)—and achieves reliability-weighted fusion through a consistency-score-driven gating network.

## Background & Motivation

Multimodal MRI segmentation (e.g., brain tumor) frequently encounters missing modalities in clinical practice due to equipment failure or varying scanning protocols. Limitations of prior work:

- **Generative methods** (GAN-based missing modality synthesis): Unstable generation quality inevitably introduces artifacts.
- **Fixed-weight fusion / attention mechanisms** (e.g., SE, CBAM): When missing modalities are filled with zero tensors, attention mechanisms become ineffective—magnitude-based attention cannot produce meaningful weights for zero inputs.
- **Consistency learning** (e.g., Mean Teacher): Suffers from background dominance in volumetric MRI—global consistency can be satisfied without aligning small tumor regions.

**Key Challenge**: Prior methods lack an explicit mechanism for determining "which modality expert should be trusted for a given case and region." Different modalities provide unequal evidence, yet no distinction is made during fusion.

**Key Insight**: CLoE **redefines missing-modality robustness as a decision-level consistency problem**—if predictions from all modality experts are consistent, the fused result is stable; inconsistency indicates that certain experts are unreliable and should be down-weighted.

## Method

### Overall Architecture

CLoE consists of three components: (1) parallel modality encoders $\Phi_m$ for per-modality feature extraction; (2) weight-shared expert decoders $D^{\text{sep}}$ that independently predict segmentation for each modality; and (3) a consistency-driven gating module that converts consistency scores into reliability weights, which are then used for weighted fusion before being passed to the fusion decoder $D^{\text{fuse}}$.

### Key Designs

1. **Modality Expert Consistency (MEC)**: For all available modality pairs $(a,b)$, the cosine similarity between prediction maps is computed to enforce global distribution alignment: $\mathcal{L}_{\text{MEC}} = \frac{1}{|\mathcal{P}|}\sum_{(a,b)\in\mathcal{P}}(1 - \mathcal{S}(\mathbf{p}^{(a)}, \mathbf{p}^{(b)}))$. **Design Motivation**: When certain modalities are absent, inconsistent predictions among remaining experts amplify fusion errors. MEC improves robustness by reducing case-wise drift.

2. **Region Expert Consistency (REC)**: Since global consistency is easily dominated by background pixels, a learnable foreground region map is introduced, $r = \sigma(\pi(\frac{1}{|\mathcal{A}|}\sum_{m\in\mathcal{A}}f_1^{(m)}))$, and consistency is computed on region-weighted predictions: $\mathcal{L}_{\text{REC}} = \frac{1}{|\mathcal{P}|}\sum_{(a,b)\in\mathcal{P}}(1 - \mathcal{S}(\mathbf{p}_r^{(a)}, \mathbf{p}_r^{(b)}))$. **Design Motivation**: In brain tumor segmentation, the enhancing tumor (ET) region occupies a very small volume, rendering global consistency constraints nearly ineffective; REC explicitly emphasizes alignment in foreground regions.

3. **Consistency-Driven Dynamic Gating**: For each modality $m$, global consistency $u_m$ and region consistency $v_m$ with respect to other experts are computed and fed into a lightweight gating network $\mathcal{G}$ to obtain reliability weights $w_m = \text{softmax}(\mathcal{G}(u_m, v_m))$. Multi-scale features are fused according to these weights: $f_\ell = \sum_m w_m \odot f_\ell^{(m)}$. Weights for missing modalities automatically collapse to zero. **Design Motivation**: Inconsistent experts equate to unreliable experts; directly deriving fusion weights from consistency measures is more principled than feature-magnitude-based attention.

### Loss & Training

The total loss is a sum of three terms:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \alpha \mathcal{L}_{\text{ECL}} + \beta \mathcal{L}_{\text{contrast}}$$

- $\mathcal{L}_{\text{seg}}$: Segmentation loss on fused features (WCE + Dice)
- $\mathcal{L}_{\text{ECL}}$: Independent supervision for each expert $+ \eta(\mathcal{L}_{\text{MEC}} + \lambda_{\text{rec}}\mathcal{L}_{\text{REC}})$
- $\mathcal{L}_{\text{contrast}}$: Contrastive representation learning loss (SSIM for content alignment + cosine for style alignment + KL regularization)

Training: Adam optimizer, lr=0.0002, weight decay=0.0001, 500 epochs, batch size=1. Modalities are randomly dropped during training to simulate missing modality scenarios.

## Key Experimental Results

### Main Results

**BraTS 2020 (15 missing modality combinations, average Dice %)**

| Region | Metric | CLoE | DC-Seg | M³AE | Gain (vs DC-Seg) |
|--------|--------|------|--------|------|------------------|
| WT | Avg Dice | **88.09** | 87.54 | 86.90 | +0.55 |
| TC | Avg Dice | **80.23** | 79.63 | 79.10 | +0.60 |
| ET | Avg Dice | **65.06** | 65.00 | 61.70 | +0.06 |

**MSD Prostate PZ (3 modality combinations)**

| Setting | CLoE | DC-Seg | RFNet |
|---------|------|--------|-------|
| T2 | **80.33** | 79.21 | 75.18 |
| ADC | **77.12** | 75.89 | 72.07 |
| T2&ADC | **82.91** | 81.67 | 78.00 |
| Average | **80.12** | 79.59 | 77.35 |

### Ablation Study

| Configuration | WT Dice | TC Dice | ET Dice | Notes |
|---------------|---------|---------|---------|-------|
| w/o MEC | 87.75 | 80.01 | 63.50 | Moderate contribution from global consistency |
| w/o REC | 86.40 | 79.39 | 61.65 | ET drops by 3.41%; region consistency is critical |
| w/o Gating | 87.99 | 80.08 | 63.90 | Gating provides fine-grained refinement |
| w/o Weight Fusion | 86.52 | 78.33 | 61.10 | ET drops by 3.96%; fusion is the most important component |
| **CLoE (full)** | **88.09** | **80.23** | **65.06** | — |

### Key Findings

- REC and Weight Fusion are the two most critical components; removing either causes a significant drop in ET (the most challenging small-region class).
- Removing MEC alone has a relatively modest effect, indicating that global consistency provides less precise constraints than region-level consistency.
- A single model handles all 15 missing modality combinations without requiring separate models for each configuration.

## Highlights & Insights

- Reformulating missing-modality robustness as a consistency control problem is conceptually clear and operationally tractable.
- The foreground-weighted strategy in REC effectively addresses background dominance and yields notable improvements for small-target segmentation (ET).
- The consistency → reliability → fusion weight pipeline is logically coherent; the gating network is extremely lightweight and introduces no additional inference overhead.
- Cross-dataset generalization is demonstrated from BraTS (4 modalities) to MSD Prostate (2 modalities).

## Limitations & Future Work

- Average Dice for ET remains at only 65%, indicating that small-target segmentation under missing modalities remains an open problem.
- The gating network takes only two scalar inputs ($u_m, v_m$), which may carry limited information; richer feature representations could be explored.
- Validation is conducted on only two datasets (BraTS and Prostate); other organ types and modality combinations are not covered.
- No comprehensive comparison with SAM-based methods (e.g., MedSAM) is provided.

## Related Work & Insights

- **Complementarity with DC-Seg** (latent disentanglement): CLoE emphasizes decision-level consistency, whereas DC-Seg focuses on representation-level disentanglement; the two methods operate at different levels of abstraction.
- The consistency learning paradigm (Mean Teacher) has proven highly effective in semi-supervised learning; this work adapts it to the missing modality setting and resolves the background dominance problem.
- **General insight for multimodal fusion**: Assessing the reliability of each modality prior to fusion is more principled than naive attention-based weighting.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The consistency → reliability formulation is novel; REC addresses a genuine problem
- **Experimental Thoroughness**: ⭐⭐⭐ BraTS + Prostate provide adequate but limited coverage
- **Writing Quality**: ⭐⭐⭐⭐ Method motivation is well-articulated; ablation design is sound
- **Value**: ⭐⭐⭐⭐ Missing modality is a genuine clinical need; the approach is practical and conceptually clear

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[ICCV 2025\] SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality](../../ICCV2025/medical_imaging/simmlm_a_simple_framework_for_multi-modal_learning_with_missing_modality.md)
- [\[CVPR 2026\] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)
- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[CVPR 2026\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)

<!-- RELATED:END -->
