---
title: >-
  [Paper Note] PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination
description: >-
  [ICCV 2025][Social Computing][visual grounding] This paper proposes PropVG, the first end-to-end proposal-based visual grounding framework that eliminates the need for pretrained detectors. It decomposes visual grounding into two stages — foreground proposal generation and contrastive learning-based referring scoring — and introduces a Multi-granularity Target Discrimination (MTD) module that integrates object-level and semantic-level information to determine target existence. PropVG achieves state-of-the-art performance on 10 datasets while running 4× faster than traditional proposal-based methods.
tags:
  - ICCV 2025
  - Social Computing
  - visual grounding
  - proposal-based
  - contrastive learning
  - referring expression
  - target existence discrimination
date: 2026-05-08
content_hash: 329141fe762f918a
---

# PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination

**Conference**: ICCV 2025
**arXiv**: [2509.04833](https://arxiv.org/abs/2509.04833)
**Code**: [GitHub](https://github.com/Dmmm1997/PropVG)
**Area**: Social Computing
**Keywords**: visual grounding, proposal-based, contrastive learning, referring expression, target existence discrimination

## TL;DR
This paper proposes PropVG, the first end-to-end proposal-based visual grounding framework that eliminates the need for pretrained detectors. It decomposes visual grounding into two stages — foreground proposal generation and contrastive learning-based referring scoring — and introduces a Multi-granularity Target Discrimination (MTD) module that integrates object-level and semantic-level information to determine target existence. PropVG achieves state-of-the-art performance on 10 datasets while running 4× faster than traditional proposal-based methods.

## Background & Motivation

**Evolution of Visual Grounding Tasks**:
   - **Classical VG**: REC (bounding box prediction) and RES (mask prediction), one expression per target
   - **Generalized VG (GVG)**: Extended to zero or multiple targets, requiring target existence judgment

**Tension Between Two Dominant Paradigms**:
   - **Traditional two-stage proposal methods** (e.g., MAttNet): Rely on pretrained detectors to generate candidate boxes before expression matching. Strengths: global foreground perception. Weaknesses: lower performance and slow inference (320 ms/frame)
   - **Direct referring methods** (e.g., TransVG, SimVG): End-to-end prediction of target location. Strengths: simple and efficient. Weaknesses: focus solely on the referred target while ignoring other foreground objects, weakening referential understanding and interpretability

**Two Key Open Problems**:
   - **(1)** How to preserve the global perception advantage of proposals while eliminating dependence on pretrained detectors for end-to-end training?
   - **(2)** In generalized settings, how to accurately determine target existence? Existing methods rely on global or single-granularity predictions, overlooking the complementarity of multi-granularity information

**Core Insight**: Visual grounding can be naturally decomposed into "which objects are foreground" (detection) and "which one is referred to" (matching). Joint end-to-end training of both steps enables simultaneous global perception and referential understanding.

## Method

### Overall Architecture
Input image $\mathcal{I}$ and text expression $\mathcal{T}$ → BEiT-3 multimodal encoder → SimFPN multi-scale features → Dual branches: (1) UNet decoder + SegHead for global segmentation $M_{seg}$; (2) multi-scale deformable decoder + DetHead for foreground proposal generation → CRS module for referring score computation → MTD module for target existence determination.

### Key Design 1: End-to-End Proposal-Based Framework

**Foreground Proposal Stage**:
- Initialize $N$ learnable queries $Q_{init}$
- Interact with SimFPN multi-scale features via a multi-scale deformable decoder to produce proposal queries $Q_{prop} \in \mathbb{R}^{N \times C}$
- DetHead outputs foreground bounding boxes $P_{bbox} \in \mathbb{R}^{N \times 4}$ and confidence scores $P_{score} \in \mathbb{R}^{N \times 2}$
- Hungarian matching assigns queries to **all foreground objects** (not only the referred target), providing global supervision

**Referring Scoring Stage**:
- Query Proj. maps $Q_{prop}$ to $Q_{prop}'$ for referring classification
- CRS module computes the referring score for each query

Key distinction from traditional two-stage methods: no external pretrained detector is required; proposals and referring are trained end-to-end within a unified framework.

### Key Design 2: Contrastive-based Refer Scoring (CRS)

The CRS module combines sentence-level and word-level contrastive learning to assess the relevance of each proposal to the text expression.

**Sentence-Level Contrastive Learning**: Computes the similarity matrix $S_{sent} \in \mathbb{R}^{N \times 1}$ between query features $Q_p$ and the global text feature $f_s$. The global text feature is obtained via valid mask pooling:

$$f_s^i = \max[f_t^i \times (\sim m)]$$

**Word-Level Contrastive Learning**: Computes the similarity matrix $S_{word} \in \mathbb{R}^{N \times N_t}$ between query features and word-level text features $f_t$.

**Adaptive Weighted Fusion**: A learnable weight $w_s$ (derived from $f_s$ via MLP + sigmoid) dynamically balances the contributions of the two levels:

$$S_{ref} = w_s \cdot S_{sent} + (1 - w_s) \cdot \text{MaxPool}(S_{word})$$

Similarity is computed using cosine similarity with a learnable temperature parameter $T$ (initialized to 0.07):

$$\text{Sim}(f_1, f_2) = \frac{f_1 \cdot f_2}{\|f_1\| \|f_2\|} / T$$

### Key Design 3: Multi-Granularity Target Discrimination (MTD)

MTD integrates object-level information (referring scores from the detection branch) and semantic-level information (mask predictions from the segmentation branch) to determine target existence.

**Score Prior Cross Attention (SPCA)**: Injects prior score information into standard attention:

$$O = \text{Softmax}(QK^T + \text{MLP}(S))V$$

where $S$ includes $S_{ref}$ (object-level prior) and $M_{seg}$ (semantic-level prior).

**Final Target Existence Score**: Fuses information from three granularities:

$$S_{exist} = \text{Max}(S_{ref}) \times \text{TAS}(M_{seg}) \times \varepsilon_{exist}$$

where TAS (TopK Average Score) computes the average score of the top-K pixels in the segmentation prediction, mitigating the influence of high-confidence outliers.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda_{det} \cdot \mathcal{L}_{det} + \lambda_{exist} \cdot \mathcal{L}_{exist} + \lambda_{ref} \cdot \mathcal{L}_{ref}$$

Default settings: $\lambda_{det}=0.1$, $\lambda_{exist}=0.2$, $\lambda_{ref}=1.0$.

## Key Experimental Results

### Evaluation Scope
10 datasets: RefCOCO/+/g (REC/RES), gRefCOCO (GREC/GRES), R-RefCOCO/+/g, Ref-ZOM

### Main Results (REC — RefCOCO)

| Method | Backbone | val | testA | testB |
|--------|----------|-----|-------|-------|
| MAttNet* (traditional proposal) | ResNet-101 | 76.65 | 81.14 | 69.99 |
| SimVG-DB | BEiT3-ViT-B | 91.47 | 93.65 | 87.94 |
| OneRef | BEiT3-ViT-B | 91.89 | 94.31 | 88.58 |
| **PropVG** | BEiT3-ViT-B | **92.70** | **95.07** | **89.58** |

PropVG comprehensively outperforms direct referring methods with the same backbone on RefCOCO, achieving a +14% performance gain over MAttNet at 4× faster inference.

### Generalized Visual Grounding (GRES — gRefCOCO)

| Method | val gIoU | testA gIoU | testB gIoU |
|--------|----------|------------|------------|
| HDC (Swin-B) | 68.28 | 72.52 | 63.85 |
| **PropVG** | **73.29** | **74.43** | **65.87** |

### GREC Detection Task

| Method | val F1 | val N-acc |
|--------|--------|-----------|
| SimVG | 62.1 | 54.7 |
| **PropVG** | **72.2** | **72.8** |

PropVG surpasses SimVG by +10.1% F1 on GREC and improves "no-target" discrimination accuracy by +18.1%.

### Ablation Study

| Component | F1score | N-acc. | gIoU |
|-----------|---------|--------|------|
| Basic Setting | 63.41 | 64.11 | 65.98 |
| + SimFPN | 65.14 | 68.02 | 66.86 |
| + UNet Decoder | 65.87 | 69.19 | 68.16 |
| + Multi-scale Deformable Decoder | 67.44 | 69.47 | 69.10 |
| + Channel Splitting | 67.98 | 70.44 | 69.59 |
| + Query Proj. (Baseline) | 68.81 | 70.39 | 69.85 |
| + CRS | 70.61 | 74.78 | 71.30 |
| + MTD | 72.20 | 72.83 | 73.29 |
| − Foreground Supervision | 66.83 | 61.06 | 66.37 |

### Key Findings
1. CRS yields +1.8 F1score and +4.4 N-acc.; the adaptive weighted fusion of word-level and sentence-level contrastive learning is critical
2. Removing foreground supervision causes an ~2% performance drop, confirming that the proposal stage's global object perception provides valuable priors for referring scoring
3. PropVG achieves an inference speed of only 76 ms/frame, significantly outperforming traditional proposal methods (MAttNet 320 ms, PolyFormer 150 ms, GroundingDINO 120 ms)
4. On R-RefCOCO/+/g, rIoU improves by 9.5–11.2%, with the largest gains in multi-target and target-absent scenarios

## Highlights & Insights
1. **Reviving the proposal-based paradigm**: By eliminating the speed and performance bottlenecks of traditional two-stage methods through end-to-end design, this work demonstrates that the proposal-based approach itself is not outdated — the issue lies in the implementation
2. **Reformulating VG as "detection + binary classification"**: The proposal stage generates candidates, and the referring stage performs a binary decision, reducing task complexity
3. **Elegant multi-granularity discrimination design**: MTD fuses detection scores, segmentation predictions, and learnable queries via a multiplicative structure that ensures low confidence in any single pathway suppresses the final score

## Limitations & Future Work
1. Proposal stage training requires foreground object annotations, increasing annotation cost
2. Target existence judgment in absent-target scenarios depends on a fixed MTD threshold (0.7), which may require tuning across datasets
3. A performance gap remains on certain metrics compared to MLLM-based methods (e.g., GSVA-13B), though PropVG uses far fewer parameters (0.2B vs. 13B+)
4. The framework has not been evaluated under open-vocabulary or zero-shot settings

## Related Work & Insights
- **REC/RES**: TransVG, LAVT, SeqTR, SimVG, OneRef
- **GVG**: ReLA, GREC, RefSegformer, HDC
- **Proposal-based**: MAttNet, NMTree, Ref-NMS
- **MLLM-based methods**: LISA, GSVA, GLaMM

## Rating
- Novelty: 4/5 (end-to-end proposal-based framework + multi-granularity target discrimination; architecturally innovative)
- Technical Depth: 4/5 (dual-level contrastive learning in CRS and SPCA mechanism in MTD are well-motivated)
- Experimental Thoroughness: 5/5 (10 datasets, extensive ablations, speed comparisons, parameter count comparison with MLLMs)
- Writing Quality: 4/5 (clear structure, comprehensive figures)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[ICLR 2026\] SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](../../ICLR2026/social_computing/sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)
- [\[AAAI 2026\] From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks](../../AAAI2026/social_computing/from_imitation_to_discrimination_toward_a_generalized_curriculum_advantage_mecha.md)
- [\[NeurIPS 2025\] DeepTraverse: A Depth-First Search Inspired Network for Algorithmic Visual Understanding](../../NeurIPS2025/social_computing/deeptraverse_a_depth-first_search_inspired_network_for_algorithmic_visual_unders.md)
- [\[NeurIPS 2025\] VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection](../../NeurIPS2025/social_computing/visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)

<!-- RELATED:END -->
