---
title: >-
  [Paper Note] Heterogeneous Uncertainty-Guided Composed Image Retrieval with Fine-Grained Probabilistic Learning
description: >-
  [AAAI 2026][Multimodal VLM][Composed Image Retrieval] This paper proposes the HUG paradigm, which leverages fine-grained Gaussian probabilistic embeddings and heterogeneous uncertainty estimation—distinguishing query-sid…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Composed Image Retrieval"
  - "Uncertainty Modeling"
  - "Probabilistic Embedding"
  - "Fine-Grained Matching"
  - "Gaussian Representation"
date: 2026-05-08
content_hash: 15f605ec93805a34
---

# Heterogeneous Uncertainty-Guided Composed Image Retrieval with Fine-Grained Probabilistic Learning

**Conference**: AAAI 2026 Oral  
**arXiv**: [2601.11393](https://arxiv.org/abs/2601.11393)  
**Code**: [https://github.com/tanghme0w/AAAI26-HUG](https://github.com/tanghme0w/AAAI26-HUG)  
**Area**: Multimodal VLM
**Keywords**: Composed Image Retrieval, Uncertainty Modeling, Probabilistic Embedding, Fine-Grained Matching, Gaussian Representation

## TL;DR

This paper proposes the HUG paradigm, which leverages fine-grained Gaussian probabilistic embeddings and heterogeneous uncertainty estimation—distinguishing query-side multimodal coordination uncertainty from target-side content quality uncertainty—combined with dynamic weighted fusion and uncertainty-guided contrastive learning, achieving state-of-the-art performance on the Fashion-IQ and CIRR benchmarks.

## Background & Motivation

### State of the Field
Composed Image Retrieval (CIR) is an emerging multimedia retrieval direction in which users search for target images via multimodal queries consisting of a reference image paired with modification text. This paradigm holds significant value in e-commerce and social media, enabling users to express complex visual preferences such as "find a dress similar to this one but in a different color."

### Limitations of Prior Work
CIR tasks suffer from inherent **data noise** issues:

**Content quality uncertainty**: Blurry images and uninformative texts are unavoidable in training data.

**Multimodal coordination uncertainty**: Even when the image and text are individually of high quality, their correspondence may be ambiguous or misaligned (e.g., the modification described in the text bears little relevance to the reference image).

### Root Cause
Existing probabilistic learning methods exhibit two major deficiencies when applied to CIR:
- **Coarse-grained instance-level modeling**: They fail to capture the complex fine-grained user intent in CIR (e.g., "change the color to red while keeping the style" involves matching across multiple attribute dimensions).
- **Homogeneous treatment of query and target**: Existing methods apply identical uncertainty estimation strategies to both the query side (multimodal) and the target side (unimodal), neglecting the cross-modal coordination challenges unique to the query side.

### Starting Point
The paper proposes the **Heterogeneous Uncertainty-Guided (HUG)** paradigm: each query and target is represented as fine-grained Gaussian embeddings, with distinct uncertainty estimation strategies designed separately for multimodal queries and unimodal targets. A theoretically grounded dynamic weighting mechanism is used to fuse uncertainties from different sources.

## Method

### Overall Architecture
HUG builds upon the Q-Former architecture of BLIP-2. Each query and target image is represented as a sequence of $K=32$ Gaussian embeddings, where each Gaussian $z_q^k \sim \mathcal{N}(\mu_q^k, \sigma_q^{k2} \mathbf{I})$ describes a fine-grained concept (e.g., color, style, logo), with variance reflecting the uncertainty over that concept.

- **Query side**: Q-Former receives a reference image (visual backbone features injected via cross-attention) and modification text, outputting means $\mu_q \in \mathbb{R}^{32 \times D}$ through 32 learnable query tokens.
- **Target side**: Q-Former receives the target image (with text input left empty), outputting $\mu_c \in \mathbb{R}^{32 \times D}$.
- The query and target sides share the same Q-Former weights.

### Key Designs

#### 1. **Heterogeneous Uncertainty Estimation**
This is the central design of the paper—distinct uncertainty modeling strategies are applied to the query and target sides.

**Target side** (unimodal; focuses solely on content quality):
- A lightweight 1-layer Transformer is used as variance estimator $g_V$: $\sigma_c^2 = g_V(\mu_c)$.
- Only visual content quality and informativeness need to be modeled.

**Query side** (multimodal; additionally accounts for cross-modal coordination):
- **Reference image uncertainty** $\sigma_r^2 = g_V(h(x_{[LQ]}, \emptyset, x_r))$: visual content quality, sharing $g_V$ with the target side.
- **Modification text uncertainty** $\sigma_t^2 = g_T(h(x_{[LQ]}, x_t, \emptyset))$: clarity and specificity of the textual modification.
- **Multimodal coordination uncertainty** $\sigma_m^2 = g_M(\mu_q)$: degree of alignment between the reference image and modification text, requiring both modalities to be observed simultaneously for estimation.

Design Motivation: Cross-modal coordination uncertainty arises from the intrinsic interaction between image and text and cannot be trivially derived from unimodal uncertainties alone. For instance, "change the color" carries low uncertainty for a plain-colored garment but high uncertainty for one with a complex pattern.

#### 2. **Multimodal Coordination Loss**
To enable $g_M$ to learn meaningful coordination uncertainty, a ranking loss is introduced:

$$\mathcal{L}_{\text{Cord.}} = -\mathbb{E}_{(x_r,x_t,x_c) \neq (x_r',x_t',x_c')} \log \mathcal{S}(\bar{\sigma}_m^2(x_r,x_t) - \bar{\sigma}_m^2(x_r,x_t'))$$

Intuition: The image-text correspondence within a given triplet should be stronger than that across different triplets; hence, the former should exhibit lower coordination uncertainty.

#### 3. **Dynamic Weighted Fusion**
The three uncertainty sources are fused into a unified query uncertainty via dynamic weights:

$$w_x^k[i] = \frac{\exp(-\sigma_x^{k}[i]^2)}{\sum_{x' \in \{r,t,m\}} \exp(-\sigma_{x'}^{k}[i]^2)}$$

Components with higher uncertainty receive lower weights, realizing the principle of "more confident components contribute more." The paper provides a theoretical proof that, under reasonable assumptions, dynamic fusion yields a tighter generalization error bound than any static weighting scheme (Proposition 1 + Corollary 1).

#### 4. **Uncertainty-Guided Contrastive Learning**
- **Holistic contrast** $\mathcal{L}_{\text{HC}}$: Employs sigmoid contrastive loss, where the distance metric is the expected Euclidean distance between two Gaussians $d(z_q, z_c) = \|\mu_q - \mu_c\|_F^2 + \|\sigma_q\|_F^2 + \|\sigma_c\|_F^2$ (closed-form solution; no sampling required).
- **Fine-grained contrast** $\mathcal{L}_{\text{FC}}$: Promotes orthogonality and diversity among the 32 fine-grained uncertainty components using three negative sampling strategies:
    - Component-level: other components within the same side/instance.
    - Instance-level: components from different instances on the same side.
    - Modality-level: arbitrary components from the opposite side.

### Loss & Training

$$\mathcal{L}_{\text{HUG}} = \mathcal{L}_{\text{HC}} + \lambda_{\text{FC}} \mathcal{L}_{\text{FC}} + \lambda_{\text{Cord.}} \mathcal{L}_{\text{Cord.}}$$

Default hyperparameters: $\lambda_{\text{FC}} = 0.5$, $\lambda_{\text{Cord.}} = 0.1$. Training uses the AdamW optimizer with batch size 32, learning rate $3 \times 10^{-5}$, on a single A100-80G GPU.

## Key Experimental Results

### Main Results

**Fashion-IQ dataset** (primary results):

| Method | Dress R@10 | Shirt R@10 | Top R@10 | Avg R@10 | Avg R@50 | Overall Avg |
|--------|-----------|-----------|---------|---------|---------|-------------|
| CLIP4CIR | 33.81 | 39.99 | 41.41 | 38.40 | 61.74 | 50.07 |
| FAME-ViL | 42.19 | 47.64 | 50.69 | 46.84 | 69.75 | 58.29 |
| QuRe | 46.80 | 53.53 | 57.47 | 52.60 | 73.48 | 63.04 |
| **HUG** | **48.37** | **51.62** | **58.26** | **52.75** | **74.73** | **63.74** |

**CIRR dataset**:

| Method | R@5 | R@10 | R_s@1 | (R@5+R_s@1)/2 |
|--------|-----|------|-------|--------------|
| QuRe | 82.53 | 90.31 | 78.51 | 80.52 |
| **HUG** | **83.20** | **92.03** | **80.65** | **81.93** |

Notably, HUG outperforms methods that utilize additional data (★) and LLM augmentation (♠), demonstrating that principled uncertainty modeling can obviate the need for extra curated annotations or LLM-based data augmentation.

### Ablation Study

| Configuration | Avg R@10 | Avg R@50 | Overall Avg | Note |
|--------------|---------|---------|-------------|------|
| (0) Point-matching baseline | 41.15 | 63.38 | 52.26 | InfoNCE, no uncertainty |
| (1) + Probabilistic embedding | 45.00 | 65.89 | 55.44 | GPO global uncertainty, +3.18 |
| (4) + Three fine-grained contrasts | 49.42 | 69.24 | 59.33 | Progressive gains from fine-grained modeling |
| (6) + Multimodal coordination loss | 52.26 | 73.95 | 63.11 | Key leap, +3.78 |
| (7) + Dynamic weighting | **52.75** | **74.73** | **63.74** | Final model |

Key findings:
- Naively introducing cross-modal uncertainty in (5) alone degrades performance, whereas adding the coordination loss in (6) yields substantial improvement—demonstrating that a dedicated loss is necessary to disentangle cross-modal from unimodal uncertainty.
- Inference latency: 21.35 ms/query vs. 7.51 ms for the baseline, approximately a 3× increase, which remains acceptable.

### Key Findings

- The learned uncertainties exhibit **interpretability**: different fine-grained components correspond to distinct sub-concepts (color, logo, sleeve length, etc.), and uncertainty magnitudes are positively correlated with the ambiguity of these attributes.
- Dynamic weighting outperforms static weighting both theoretically and empirically.
- The coordination loss is the key component enabling effective utilization of cross-modal uncertainty.

## Highlights & Insights

1. **Heterogeneous design philosophy**: The structural asymmetry between multimodal queries and unimodal targets calls for distinct uncertainty modeling strategies—a simple yet effective insight.
2. **Fine-grained probabilistic representation**: The 32 Gaussians (Q-Former query tokens) naturally map to attribute-level granularity, capturing the complex intent in CIR more effectively than coarse instance-level representations.
3. **Theoretical guarantees**: The generalization error bound proof for dynamic weighting provides not only engineering justification but also theoretical grounding.
4. **Interpretability**: The learned uncertainty components can be mapped to human-interpretable visual concepts.

## Limitations & Future Work

- Inference time increases approximately 3× (21 ms vs. 7 ms per query), which may become a bottleneck in large-scale retrieval scenarios.
- The number of Gaussian components (32) is predetermined by the Q-Former design; the optimal number of components warrants further exploration.
- Validation is conducted solely on supervised CIR; applicability to zero-shot CIR remains to be investigated.
- The theoretical proof for dynamic weighting relies on the convexity assumption of the loss function; alternative loss functions may require re-analysis.

## Related Work & Insights

- Probabilistic embedding learning has been explored in cross-modal retrieval (PCME, PCME++), but this paper is the first to introduce it into CIR while explicitly accounting for heterogeneity.
- Using Q-Former's 32 query tokens as carriers of fine-grained representations is an elegant design choice.
- The uncertainty-guided contrastive learning strategy is generalizable to other multimodal matching tasks.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](../../ACL2026/multimodal_vlm/tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)
- [\[CVPR 2026\] ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval](../../CVPR2026/multimodal_vlm/recall_recalibrating_capability_degradation_for_mllm-based_composed_image_retrie.md)
- [\[CVPR 2026\] EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval](../../CVPR2026/multimodal_vlm/eaglenet_energy-aware_fine-grained_relationship_learning_network_for_text-video_.md)
- [\[CVPR 2026\] G-MIXER: Geodesic Mixup-based Implicit Semantic Expansion and Explicit Semantic Re-ranking for Zero-Shot Composed Image Retrieval](../../CVPR2026/multimodal_vlm/g_mixer_geodesic_mixup_based_implicit_semantic_expansion_for_zero_shot_cir.md)
- [\[AAAI 2026\] DEIG: Detail-Enhanced Instance Generation with Fine-Grained Semantic Control](deig_detail-enhanced_instance_generation_with_fine-grained_semantic_control.md)

</div>

<!-- RELATED:END -->
