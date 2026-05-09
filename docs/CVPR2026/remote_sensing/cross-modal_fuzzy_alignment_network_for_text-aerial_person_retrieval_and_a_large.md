---
title: >-
  [Paper Note] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark
description: >-
  [CVPR 2026][Remote Sensing][Text-Aerial Person Retrieval] A cross-modal fuzzy alignment network (CFAN) that leverages fuzzy logic to quantify token-level reliability for fine-grained alignment and introduces ground-view bridging to alleviate the semantic gap between aerial images and text descriptions, along with a large-scale text-aerial person retrieval benchmark AERI-PEDES.
tags:
  - CVPR 2026
  - Remote Sensing
  - Text-Aerial Person Retrieval
  - Fuzzy Logic
  - Cross-Modal Alignment
  - UAV
  - Chain-of-Thought Annotation
date: 2026-05-08
content_hash: ffc578fab97245d4
---

# Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark

**Conference**: CVPR 2026  
**arXiv**: [2603.20721](https://arxiv.org/abs/2603.20721)  
**Code**: [https://github.com/Yifei-AHU/AERI-PEDES](https://github.com/Yifei-AHU/AERI-PEDES)  
**Area**: Remote Sensing / Person Retrieval  
**Keywords**: Text-Aerial Person Retrieval, Fuzzy Logic, Cross-Modal Alignment, UAV, Chain-of-Thought Annotation

## TL;DR

A cross-modal fuzzy alignment network (CFAN) that leverages fuzzy logic to quantify token-level reliability for fine-grained alignment and introduces ground-view bridging to alleviate the semantic gap between aerial images and text descriptions, along with a large-scale text-aerial person retrieval benchmark AERI-PEDES.

## Background & Motivation

**State of the field**: Text-image person retrieval (TIPR) has achieved significant progress, but all existing methods are based on fixed ground-level camera data. UAVs offer unique advantages of dynamic multi-angle surveillance, and extending TIPR to aerial scenarios holds substantial research value.

**Existing limitations**: (1) Aerial images suffer from nonlinear appearance distortions due to drastic variations in shooting angle and altitude; (2) visual cues of pedestrians in aerial views are sparse or partially missing (e.g., only the top of the head is visible), while text descriptions contain complete attributes that cannot fully correspond to aerial images; (3) during token-level fine-grained alignment, unobservable tokens introduce erroneous cross-modal alignments.

**Core tension**: Eyewitness descriptions are typically detailed and complete, but aerial images can only cover partial semantic regions—this visibility inconsistency generates substantial noisy matches during fine-grained alignment.

**Objective**: How to achieve robust text-aerial person cross-modal retrieval when visual cues in aerial images are incomplete?

**Approach**: (1) Fuzzy logic is used to quantify the reliability of each token, suppressing the influence of unobservable/noisy tokens; (2) ground-level views serve as intermediate bridges, with adaptive balancing between direct alignment and bridged alignment.

**Core idea**: Fuzzy membership modeling of token reliability + context-aware dynamic alignment = robust text-aerial alignment.

## Method

### Overall Architecture

CFAN comprises two core modules: Context-Aware Dynamic Alignment (CDA) for sample-level adaptive alignment, and Fuzzy Token Alignment (FTA) for token-level fine-grained alignment. A shared CLIP image encoder extracts aerial/ground features, and a CLIP text encoder extracts description features.

### Key Designs

1. **Context-Aware Dynamic Alignment (CDA)**:

    - **Function**: Computes the cosine similarity difference between text-aerial and text-ground pairs $\Delta_i = \text{sim}(T_i^C, A_i^C) - \text{sim}(T_i^C, G_i^C)$, maps it through sigmoid to a soft decision gate $\alpha_i$, and adaptively weights direct alignment and bridged alignment.
    - **Core formula**: $\alpha_i = \frac{1}{1 + \exp[-k \cdot \Delta_i]}$
    - **Loss**: $\mathcal{L}_{\text{CDA}} = \frac{1}{B} \sum_{i=1}^B [\alpha_i \cdot \mathcal{L}_{\text{direct}} + (1-\alpha_i) \cdot \mathcal{L}_{\text{bridge}}]$
    - **Design motivation**: Different aerial images have varying alignment difficulty—low-altitude close-ups can be directly aligned, while high-altitude distant shots require ground-level bridging. $\alpha_i$ automatically estimates "alignment difficulty" and assigns alignment strategies. Stop-gradient is applied to ground features in bridged alignment to prevent interference with ground representations.

2. **Fuzzy Token Alignment (FTA)**:

    - **Function**: Shared learnable queries $\mathbf{Q} \in \mathbb{R}^{K \times D}$ perform cross-attention with both modalities to obtain modality-aware query representations. A Gaussian fuzzy membership function then computes each token's reliability:
    - **Core formula**: $\mu_j^a = \exp(-\frac{(1-r_j)^2}{2\sigma^2})$, where $r_j$ is the cosine similarity between the query token and the global class token.
    - Joint membership from both modalities: $\mu_j^{\text{joint}} = \mu_j^a \cdot \mu_j^t$ (fuzzy AND operation)
    - Weighted similarity: $\text{sim}(Q_a, Q_t) = \frac{1}{K} \sum_{j=1}^K \mu_j^{\text{joint}} s_j$
    - **Design motivation**: Only tokens that are highly reliable in both modalities contribute to alignment—i.e., parts that are visible and semantically consistent in both modalities contribute most, while unobservable/noisy tokens are naturally suppressed. The Gaussian scale $\sigma$ is adaptively predicted from the global class token, enabling the model to adjust the reliability threshold based on image content.

3. **AERI-PEDES Dataset Construction**:

    - Chain-of-Thought decomposition for text generation: attribute parsing → initial annotation → review and refinement
    - Training set uses VLM-generated annotations; test set uses human annotations for evaluation reliability

### Loss Function / Training Strategy

- CDA direct alignment and bridged alignment both use SDM (Similarity Distribution Matching) loss
- FTA uses KL divergence for similarity distribution matching
- Total loss: $\mathcal{L} = \mathcal{L}_{\text{CDA}} + \mathcal{L}_{\text{FTA}}$
- Adam optimizer, initial learning rate $5 \times 10^{-6}$, cosine decay, 60 epochs

## Key Experimental Results

### Main Results

| Method | AERI-PEDES R1↑ | AERI-PEDES mAP↑ | TBAPR R1↑ | TBAPR mAP↑ |
|--------|---------------|-----------------|-----------|------------|
| IRRA (CVPR23) | 35.14 | 33.42 | 39.63 | 35.31 |
| HAM (CVPR25) | 44.58 | 42.45 | 47.81 | 41.86 |
| CFAN (no ground) | 45.06 | 43.27 | 49.15 | 42.89 |
| **CFAN (with ground)** | **47.16** | **44.79** | **49.47** | **43.96** |

### Ablation Studies

| Configuration | R1 | mAP | RSum | Description |
|--------------|-----|------|------|-------------|
| Baseline (bridge only) | 43.88 | 41.58 | 174.84 | Baseline |
| + CDA | 46.18 | 43.98 | 183.04 | RSum +8.2% |
| + FTA | 44.55 | 41.89 | 176.64 | R1 +0.67% |
| + CDA + FTA | **47.16** | **44.79** | **186.65** | Full combination |

Bridging modality comparison:

| Bridging Method | R1 | mAP | Description |
|----------------|-----|------|-------------|
| None (no bridging) | 45.06 | 43.27 | FTA only |
| Aerial (low-altitude bridging) | 46.08 | 44.20 | Effective but limited |
| Ground (ground-view bridging) | **47.16** | **44.79** | Optimal |

### Key Findings

- CDA contributes the most (RSum improvement of 8.2%), indicating that adaptive balancing of direct/bridged alignment is the core mechanism
- FTA provides complementary fine-grained alignment improvement
- Even without ground images, CFAN with FTA alone already outperforms all competitors

## Highlights & Insights

- **Integration of fuzzy logic with deep learning**: Using fuzzy membership functions to quantify token reliability is an elegant design with stronger theoretical grounding than simple attention weights
- **Large-scale dataset contribution**: AERI-PEDES (112K+ images) fills the data gap for text-aerial person retrieval
- **CoT annotation pipeline**: The structured reasoning-based annotation method is generalizable to other dataset construction tasks

## Limitations & Future Work

- Paired aerial and ground images of the same person are required, which may be difficult to obtain in practical deployment
- The fuzzy membership function uses a Gaussian form; more flexible parameterizations may yield better results
- Only CLIP encoders are used; larger and more powerful VLMs may bring further improvements

## Related Work & Inspiration

- Fuzzy deep learning has been applied in medical image analysis; this work extends it to cross-modal retrieval
- The dynamic alignment difficulty estimation in CDA can be generalized to other retrieval scenarios with incomplete visual information (e.g., foggy or occluded conditions)

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of fuzzy logic and dynamic bridged alignment is novel
- Experimental rigor: ⭐⭐⭐⭐⭐ Two datasets + complete ablations + parameter sensitivity analysis
- Writing quality: ⭐⭐⭐⭐ Clear formulas and well-structured presentation
- Impact: ⭐⭐⭐⭐ Both the dataset and the method provide tangible advancement for text-aerial person retrieval

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments](olbedo_an_albedo_and_shading_aerial_dataset_for_large-scale_outdoor_environments.md)
- [\[ICCV 2025\] CityNav: A Large-Scale Dataset for Real-World Aerial Navigation](../../ICCV2025/remote_sensing/citynav_a_large-scale_dataset_for_real-world_aerial_navigation.md)
- [\[CVPR 2026\] AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network](avion_aerial_visionlanguage_instruction_from_offli.md)
- [\[AAAI 2026\] Asymmetric Cross-Modal Knowledge Distillation: Bridging Modalities with Weak Semantic Consistency](../../AAAI2026/remote_sensing/asymmetric_cross-modal_knowledge_distillation_bridging_modalities_with_weak_sema.md)
- [\[CVPR 2026\] RHO: Robust Holistic OSM-Based Metric Cross-View Geo-Localization](rho_robust_holistic_osm-based_metric_cross-view_geo-localization.md)

</div>

<!-- RELATED:END -->
