---
title: >-
  [Paper Note] FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion
description: >-
  [AAAI 2026][Video Understanding][Fine-grained action recognition] This paper proposes FineTec, a framework that achieves robust fine-grained skeleton-based action recognition under temporal corruption via three modules:…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "Fine-grained action recognition"
  - "temporal corruption"
  - "skeleton decomposition"
  - "Lagrangian dynamics"
  - "sequence completion"
date: 2026-05-08
content_hash: 954766f1dfe9f723
---

# FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion

**Conference**: AAAI 2026
**arXiv**: [2512.25067](https://arxiv.org/abs/2512.25067)
**Code**: [Project Page](https://smartdianlab.github.io/projects-FineTec/)
**Area**: Video Understanding
**Keywords**: Fine-grained action recognition, temporal corruption, skeleton decomposition, Lagrangian dynamics, sequence completion

## TL;DR

This paper proposes FineTec, a framework that achieves robust fine-grained skeleton-based action recognition under temporal corruption via three modules: context-aware sequence completion, bio-prior-guided skeleton spatial decomposition, and physics-driven acceleration modeling.

## Background & Motivation

### State of the Field
Fine-grained action recognition (FAR) requires distinguishing actions with subtle temporal variations and fine semantic differences (e.g., "double-twisting double somersault in layout position"). Skeleton representations have emerged as an effective modality due to their compactness and explicit focus on motion cues.

### Limitations of Prior Work
In complex scenarios such as gymnastics, online pose estimation suffers from **severe frame dropping**—with missing frame rates reaching 69.6% during rapid motion. This results in **temporally corrupted** skeleton sequences, which critically undermines fine-grained action recognition that relies on continuous subtle motion cues.

Two major limitations exist in prior methods:

**Insufficient temporal recovery**: Most models are trained on clean, offline-annotated skeletons and lack mechanisms to handle artifacts introduced by online detection.

**Inadequate spatio-temporal modeling**: Intrinsic biological body structure is ignored; models focus primarily on pointwise positional features while neglecting continuous kinematic constraints—relying only on displacement information without physical interpretability.

### Core Idea
The problem is decomposed into three steps: (1) **completing** corrupted skeleton sequences to restore temporal continuity; (2) leveraging **biological priors** to decompose skeletons into dynamic/static regions for differentiated augmentation; (3) introducing **Lagrangian dynamics** to estimate joint acceleration as additional discriminative features. Position sequences and acceleration sequences are jointly fed into a GCN for classification.

## Method

### Overall Architecture

FineTec comprises three core modules that sequentially process a temporally corrupted skeleton sequence $S_{corrupt} \in \mathbb{R}^{T \times K \times 2}$:

1. **Context-aware Sequence Completion**: Recovers the base sequence $S_{base}$ from corrupted input.
2. **Skeleton-based Spatial Decomposition**: Decomposes $S_{base}$ into dynamic/static regions, generates augmented sequences $S_{dyna}$ and $S_{stat}$, and fuses them into $S_{pred}$.
3. **Physics-driven Acceleration Modeling**: Estimates acceleration features $\mathbf{a}$ using Lagrangian dynamics.

The final $S_{pred}$ and $\mathbf{a}_{pred}$ are jointly fed into a GCN classifier.

### Key Designs

#### 1. **Context-aware Sequence Completion**

- **Mechanism**: Adopts an **In-Context Learning (ICL)** paradigm to recover corrupted sequences.
- **Implementation**:
    - A skeleton library is constructed from Human3.6M; temporal averaging yields a prior sequence $S_{prior}$.
    - Sequences sampled from the skeleton library are corrupted using **five temporal masking strategies** (random, patterned, prefix/suffix/middle block), forming prompt pairs (original + masked).
    - The input $S_{corrupt}$ and $S_{prior}$ constitute the query pair.
    - Lightweight spatial and temporal MLPs are applied to complete the approximate recovery.
- **Design Motivation**: Five masking strategies cover diverse real-world missing data patterns; the ICL paradigm avoids overfitting to specific corruption patterns.
- **Loss**: $\mathcal{L}_{ICL} = \text{MSE}(S_{gt}, S_{base}) + \text{MSE}(S_{context}, S_{mask})$

#### 2. **Skeleton-based Spatial Decomposition**

- **Mechanism**: Based on biological structural priors of the human body, joints are partitioned into dynamic/static regions according to motion intensity for differentiated augmentation.
- **Procedure**:
    - $K=17$ joints are divided into 5 semantic regions: head ($G_0$), left arm ($G_1$), right arm ($G_2$), left leg ($G_3$), right leg ($G_4$).
    - The average inter-frame displacement per joint is computed: $D_{avg}^{(i)} = \frac{1}{T-1}\sum_{t=0}^{T-2}\|S_{base}^{t+1,i} - S_{base}^{t,i}\|_2$
    - Region-level motion intensity: $\bar{D}_j = \frac{1}{|G_j|}\sum_{i \in G_j} D_{avg}^{(i)}$
    - The **top-2 highest-motion regions → dynamic group**; the remaining 3 → static group.
- **Differentiated Augmentation**:
    - Dynamic regions: strong spatio-temporal perturbations (temporal cropping, random dropping, interpolation) → $S_{dyna}$
    - Static regions: weak spatial perturbations (random flipping) → $S_{stat}$
- **Fusion**: $S_{base}$, $S_{dyna}$, and $S_{stat}$ are fused into $S_{pred}$.
- **Design Motivation**: Subtle differences in dynamic regions are key to distinguishing similar actions; strong augmentation amplifies discriminative information, while stable static regions avoid introducing noise.

#### 3. **Physics-driven Acceleration Modeling**

- **Mechanism**: Joint acceleration is explicitly modeled using the Lagrangian dynamics equation.
- **Lagrangian Equation**: $M(S)\ddot{S} + C(S,\dot{S})\dot{S} + g(S) = \tau$
- **Acceleration Estimation**: $\ddot{S} = \{M(S)\}^{-1} \cdot \tau - \hat{C}(S,\dot{S})\dot{S} - \hat{g}(S)$
- **Implementation**:
    - Global and local position and velocity features are extracted.
    - A neural network $\mathbb{E}$ estimates each physical term (inertia matrix $M$, Coriolis force $C$, gravity $g$, generalized force $\tau$).
    - For symmetric matrices, only the upper triangular part is estimated and then symmetrized.
- **Pseudo-acceleration**: Computed via second-order finite differences: $\hat{a}_t = \frac{S_{t+1} - 2S_t + S_{t-1}}{(\Delta t)^2}$
- **Fusion**: $\mathbf{a}_t = \text{Fusion}(\hat{a}_t, \ddot{S}_t) \in \mathbb{R}^{K \times 2}$
- **Design Motivation**: Acceleration captures dynamic characteristics that displacement alone cannot express; physical constraints endow features with greater interpretability.

### Loss & Training

Two-stage training:
1. **Pre-training**: The sequence completion module is trained on skeleton datasets (Adam, lr $1 \times 10^{-5}$, 40K iterations).
2. **Fine-tuning**: The completion module is frozen; the full framework is trained (SGD + Nesterov momentum, lr 0.05–0.2, 150 epochs).

Overall loss: $\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{Ac}$, where $\mathcal{L}_{Ac} = \frac{1}{3}\sum_\alpha \text{MSE}(\hat{\mathbf{a}}_\alpha, \mathbf{a}_\alpha)$

## Key Experimental Results

### Main Results (Fine-grained Datasets)

| Method | Gym288-Sev. Top-1 | Gym288-Sev. Mean | Gym99-Sev. Top-1 | Gym99-Sev. Mean |
|------|-------------------|------------------|-------------------|-----------------|
| ST-GCN | 0.742 | 0.304 | 0.871 | 0.783 |
| PYSKL-J | 0.773 | 0.315 | 0.884 | 0.791 |
| CTR-GCN | 0.760 | 0.271 | 0.884 | 0.803 |
| Sparse(CVPR'25) | 0.683 | 0.237 | 0.808 | 0.725 |
| **FineTec** | **0.781** | **0.356** | **0.891** | **0.805** |

On the most challenging Gym288-Severe benchmark, FineTec achieves a mean class accuracy **13% higher** than the best baseline and **50% higher** than Sparse.

### Coarse-grained Datasets

| Method | NTU-60 Sev. | NTU-120 Sev. |
|------|-------------|--------------|
| ST-GCN | 0.879 | 0.781 |
| CTR-GCN | 0.879 | 0.793 |
| Sparse | 0.864 | 0.767 |
| **FineTec** | **0.892** | **0.813** |

Under severe corruption, FineTec improves by **1.3%** on NTU-60 and **1.7%** on NTU-120.

### Ablation Study

| Configuration | Minor | Moderate | Severe | Note |
|------|-------|----------|--------|------|
| w/o Completion | 0.812 | 0.785 | 0.751 | Completion module removed |
| w/o Skeleton Decomp. | 0.787 | 0.780 | 0.770 | Decomposition module removed |
| w/o Physics | 0.789 | 0.776 | 0.775 | Physics modeling removed |
| **Full FineTec** | **0.815** | **0.797** | **0.781** | Full model |

| $S_{dyna}$ | $S_{stat}$ | Moderate | Severe | Note |
|------------|------------|----------|--------|------|
| ✗ | ✓ | 0.790 | 0.774 | Static augmentation only |
| ✓ | ✗ | 0.786 | 0.764 | Dynamic augmentation only |
| ✓ | ✓ | **0.797** | **0.781** | Combined is optimal |

### Skeleton Recovery Quality

| Method | Gym99-Sev. MPJPE↓ | N-MPJPE↓ | MPJVE↓ |
|------|-------------------|----------|--------|
| SiC-Dyna | 0.192 | 0.174 | 0.321 |
| **FineTec** | **0.147** | **0.132** | **0.113** |

MPJPE is reduced by **23.4%**.

### Key Findings
1. Each of the three modules contributes independently and complementarily; removing any single module leads to a notable performance drop.
2. Sequence completion contributes most under severe corruption (improving from 0.751 to 0.781).
3. The combination of dynamic and static decomposition consistently outperforms either branch alone.
4. Cross-attention (CA) fusion outperforms MLP fusion (0.797 vs. 0.779).

## Highlights & Insights

1. **Physical interpretability**: Lagrangian dynamics are introduced to model acceleration, going beyond purely data-driven approaches and endowing motion features with physical meaning.
2. **Systematic pipeline**: The three modules correspond to a complete recovery→decomposition→augmentation pipeline, with each stage building upon the previous.
3. **New dataset contribution**: Gym288-skeleton (288 fine-grained action classes, 38K sequences) is introduced as a highly challenging benchmark with long-term value.
4. **ICL paradigm for skeleton completion**: The in-context learning paradigm from NLP is adapted for sequence recovery, representing a novel and transferable idea.

## Limitations & Future Work

1. The skeleton library relies on fixed Human3.6M data; more adaptive, data-driven approaches could be explored in future work.
2. Dynamic/static grouping is based on a manually defined rule (top-2 highest-motion regions); a learnable grouping strategy warrants investigation.
3. Validation is currently limited to 2D skeletons; extension to 3D skeletons and multimodal settings may yield further improvements.
4. Joint-level acceleration modeling could be extended to sub-group or limb-level dynamics.

## Related Work & Insights

- This work parallels InfoGCN++, which uses Neural-ODE for action recognition, but adopts the more classical Lagrangian mechanics formulation.
- The novel application of the ICL paradigm (originating from NLP) to skeleton recovery is generalizable to other sequential data completion tasks.
- The bio-prior-guided decomposition strategy is transferable to domains requiring fine-grained discrimination, such as gesture recognition and surgical action analysis.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Each module has precedents, but the integration scheme and physics-driven approach are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Fine-grained and coarse-grained datasets, three corruption severity levels, thorough ablations, and recovery quality evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with complete mathematical derivations.
- **Value**: ⭐⭐⭐⭐ — Temporal corruption scenarios have significant practical relevance; the new dataset offers long-term community value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)
- [\[AAAI 2026\] R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios](r-avst_empowering_video-llms_with_fine-grained_spatio-temporal_reasoning_in_comp.md)
- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [\[CVPR 2026\] Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding](../../CVPR2026/video_understanding/frame2freq_spectral_adapters_for_fine-grained_video_understanding.md)
- [\[AAAI 2026\] Task-Specific Distance Correlation Matching for Few-Shot Action Recognition](task-specific_distance_correlation_matching_for_few-shot_action_recognition.md)

</div>

<!-- RELATED:END -->
