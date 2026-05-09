---
title: >-
  [Paper Note] Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data
description: >-
  [AAAI 2026][Autonomous Driving][Multi-Task Learning] This paper proposes a task prototype-based knowledge retrieval framework that employs learnable Task Prototypes to encode task characteristics and quantify inter-task affinities, and a Knowledge Retrieval Transformer to adaptively refine feature representations based on task-affinity scores. The framework addresses multi-task learning from partially annotated data (MTPSL) without relying on predictions from unannotated tasks, achieving state-of-the-art performance on PASCAL-Context and NYUD-v2.
tags:
  - AAAI 2026
  - Autonomous Driving
  - Multi-Task Learning
  - Task Prototype
  - Partial Annotation
  - Knowledge Retrieval Transformer
  - Task Affinity
date: 2026-05-08
content_hash: eebcbe6b3bf13949
---

# Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data

**Conference**: AAAI 2026
**arXiv**: [2601.07474](https://arxiv.org/abs/2601.07474)
**Code**: Not available
**Area**: Multi-Task Learning / Dense Prediction / Partial Annotation
**Keywords**: Multi-Task Learning, Task Prototype, Partial Annotation, Knowledge Retrieval Transformer, Task Affinity

## TL;DR

This paper proposes a task prototype-based knowledge retrieval framework that employs learnable Task Prototypes to encode task characteristics and quantify inter-task affinities, and a Knowledge Retrieval Transformer to adaptively refine feature representations based on task-affinity scores. The framework addresses multi-task learning from partially annotated data (MTPSL) without relying on predictions from unannotated tasks, achieving state-of-the-art performance on PASCAL-Context and NYUD-v2.

## Background & Motivation

Applications such as autonomous driving and robotics require the simultaneous handling of multiple dense prediction tasks, including semantic segmentation, depth estimation, and surface normal estimation. Obtaining complete annotations for all tasks is prohibitively expensive, making multi-task learning from partially annotated data (MTPSL) a practical necessity.

Limitations of existing MTPSL methods:
- **MTPSL (Li et al., CVPR'22)**: Achieves cross-task regularization via joint task-space mapping, but relies on predictions from unannotated tasks, which introduces substantial noise.
- **DiffusionMTL (Ye & Xu, CVPR'24)**: Integrates cross-task information using diffusion models with multi-task conditioning, but the mixed use of prediction-based and feature-based inputs leads to inconsistent performance.
- **Shared limitation**: Both methods depend on predictions from unannotated tasks to establish task associations. Since such predictions are inherently noisy and incomplete, they induce negative transfer.

**Core Problem**: Under partial annotation, how can reliable task associations be established without relying on predictions from unannotated tasks, and how can task-specific feature representations be adaptively optimized?

## Method

### Overall Architecture

Two main modules are jointly trained end-to-end:
1. **Multi-Task Learning Module**: Backbone (ResNet-18) → encoded features $f^e$ → Vector Quantization-enhanced shared representation → per-task decoders → task features $f^t$
2. **Prototype Knowledge Retrieval Module**: $f^t$ → Task Prototype $\mathcal{V}$ generates task-affinity scores → Knowledge Retrieval Transformer adaptively refines features based on affinity → refined task features $f^{tr}$ → per-task prediction heads

### Key Designs

1. **Vector Quantization-Enhanced Representation**:

    - Maintains a learnable codebook $\mathcal{Z} = \{z_k\}_{k=1}^K$ ($K=4096$ slots, dimensionality $1024$).
    - Encoded features $f^e$ are mapped to codebook entries via nearest-neighbor quantization to obtain $f^q$, which is added element-wise to $f^e$ to produce $f^i$.
    - The codebook is trained via a **Task-Agnostic Enhancement (TAE) Loss**: $f^i$ is decoded by a convolutional decoder to reconstruct the input image using Smooth L1 loss.
    - Purpose: To expand the shared feature space under partial annotation so that cues from all tasks are preserved.

2. **Task Prototype $\mathcal{V}$**:

    - Contains $T$ learnable slots (where $T$ equals the number of tasks: $T=5$ for PASCAL-Context, $T=3$ for NYUD-v2), with each $v_\tau \in \mathbb{R}^{1 \times d}$ ($d=1024$).
    - Computes task-similarity: $S(\hat{f}^t, \mathcal{V}) = \text{cosine\_similarity}(\hat{f}^t, v_\tau)$ for each task $\tau$.
    - **Task Knowledge Embedding (TKE) Loss**: Applies cross-entropy to the softmax-normalized affinity scores, with one-hot vector $Y_t$ as the supervision target.
    - **Task Consistency (TC) Loss**: Aggregates features $\tilde{x}^t$ within a batch for the same task and applies triplet loss to encourage high intra-task similarity and low inter-task similarity.
    - Association Knowledge Generating (AKG) Loss: $\mathcal{L}_{akg} = \mathcal{L}_{tke} + \mathcal{L}_{tc}$

3. **Knowledge Retrieval Transformer**:

    - Input: task features $\hat{f}^t$ and task-affinity features $f^{ta} = \mathcal{A}(\hat{f}^t, \mathcal{V}) \cdot \mathcal{V}$
    - Multiple knowledge-retrieval blocks: Self-Attention → Cross-Attention (query $= f^t$, key/value $= f^{ta}$) → FFN
    - In cross-attention, $f^{ta}$ encodes information about "how much enhancement each task should retrieve from other tasks."
    - Output: refined task features $f^{tr}$, passed to per-task prediction heads.

### Loss & Training

Total loss:

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{MTL}} + \lambda_1 \cdot \sum \mathcal{L}_{tae} + \lambda_2 \cdot \mathcal{L}_{akg}$$

- $\mathcal{L}_{\text{MTL}}$: Supervised loss on annotated samples (cross-entropy for semantic segmentation/parsing/saliency/boundary; L1 for depth/normals).
- $\mathcal{L}_{tae}$: Codebook-enhanced image reconstruction loss (Smooth L1).
- $\mathcal{L}_{akg} = \mathcal{L}_{tke} + \mathcal{L}_{tc}$: Ensures prototypes correctly encode task-specific characteristics.

Training details: Adam optimizer, lr $= 2 \times 10^{-5}$; PASCAL-Context: 100 epochs, batch size 6; NYUD-v2: 200 epochs, batch size 4; single RTX A6000.

## Key Experimental Results

### Main Results: PASCAL-Context (5 Tasks)

| Method | Setting | Semseg mIoU↑ | Parsing mIoU↑ | Saliency maxF↑ | Normal mErr↓ | Boundary odsF↑ |
|------|------|:-:|:-:|:-:|:-:|:-:|
| Single-Task | one-label | 50.34 | 59.05 | 77.43 | 16.59 | 64.40 |
| MTL Baseline | one-label | 44.73 | 57.03 | 75.69 | 16.47 | 64.30 |
| MTPSL* (CVPR'22) | one-label | 55.08 | 56.72 | 77.06 | 16.93 | 63.70 |
| DiffusionMTL-Pred (CVPR'24) | one-label | 59.43 | 56.79 | 77.57 | 16.20 | 64.00 |
| DiffusionMTL-Feat (CVPR'24) | one-label | 57.78 | 58.98 | 77.82 | 16.11 | 64.50 |
| **Ours** | **one-label** | **59.78** | **59.08** | **78.62** | **15.63** | **65.10** |
| MTPSL* (CVPR'22) | random | 62.44 | 55.81 | 78.56 | 15.45 | 66.80 |
| DiffusionMTL-Feat (CVPR'24) | random | 62.55 | 56.84 | 80.44 | 14.85 | 67.10 |
| **Ours** | **random** | **64.30** | **56.87** | **80.51** | **14.48** | **67.30** |

### NYUD-v2 (3 Tasks)

| Method | Setting | Semseg mIoU↑ | Depth absErr↓ | Normal mErr↓ |
|------|------|:-:|:-:|:-:|
| Single-Task | one-label | 45.28 | 0.4802 | 25.93 |
| MTPSL* (CVPR'22) | one-label | 43.97 | 0.5140 | 26.30 |
| DiffusionMTL-Feat (CVPR'24) | one-label | 44.47 | 0.5059 | 25.84 |
| **Ours** | **one-label** | **45.95** | **0.4865** | **25.64** |
| **Ours** | **random** | **47.53** | **0.4621** | **24.67** |

### Ablation Study: Contribution of Each Loss (PASCAL-Context, one-label)

| $\mathcal{L}_{tae}$ | $\mathcal{L}_{tke}$ | $\mathcal{L}_{tc}$ | Semseg | Parsing | Saliency | Normal↓ | Boundary |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ✗ | ✗ | ✗ | 44.73 | 57.03 | 75.69 | 16.47 | 64.38 |
| ✓ | ✗ | ✗ | 44.83 | 57.13 | 76.13 | 16.22 | 64.50 |
| ✓ | ✓ | ✗ | 58.21 | 58.87 | 78.50 | 15.67 | 65.00 |
| ✓ | ✓ | ✓ | **59.78** | **59.08** | **78.62** | **15.63** | **65.10** |

### Prototype Dimensionality Ablation (NYUD-v2, one-label)

| Dimension | Parameters | Semseg↑ | Depth↓ | Normal↓ |
|------|:-:|:-:|:-:|:-:|
| No prototype | 146.5M | 42.77 | 0.5134 | 26.99 |
| 256 | 156.6M | 44.91 | 0.4931 | 25.67 |
| 512 | 157.5M | 45.65 | 0.4878 | 25.73 |
| **1024** | **159.4M** | **45.95** | **0.4865** | **25.64** |
| 2048 | 163.0M | 45.33 | 0.4867 | 25.77 |

### Key Findings

- $\mathcal{L}_{tke}$ is the largest contributor (Semseg improves ~13 points from 44.83 to 58.21), demonstrating the effectiveness of embedding task knowledge into prototypes.
- $\mathcal{L}_{tc}$ further improves consistency (58.21 → 59.78), ensuring stable task characteristics across different samples.
- A prototype dimensionality of 1024 is optimal — lower dimensions are insufficient to encode task characteristics, while higher dimensions impede effective information utilization.
- Compared to prompt-based methods (TaskPrompter, TSP-Transformer): the proposed explicit learning approach outperforms latent learning under partial annotation (50.08 vs. 48.68 Semseg on NYUD-v2).

## Highlights & Insights

- **Core Innovation**: Replaces reliance on predictions from unannotated tasks with explicit Task Prototypes — shifting the paradigm from "pseudo-label-driven" to "task-characteristic-driven."
- Task-affinity scores provide interpretable visualizations of cross-task associations, with different target tasks activating distinct prototype slots.
- The cross-attention mechanism in the Knowledge Retrieval Transformer enables "on-demand retrieval" — each task selectively acquires only relevant cross-task knowledge.
- The use of Vector Quantization to expand the shared feature space is inspired by VQ-VAE and is elegantly adapted to the partial annotation setting.

## Limitations & Future Work

- Validation is limited to the ResNet-18 backbone; larger models (e.g., ViT-L) are only compared with prompt-based methods in the discussion section.
- The number of prototype slots is fixed equal to the number of tasks; the effect of over-parameterization on generalization remains unexplored.
- The method requires at least partial annotation for all tasks (the one-label setting still requires each image to have a label for at least one task).
- The contribution of $\mathcal{L}_{tae}$ in the ablation is marginal (44.73 → 44.83), raising questions about the necessity of the codebook component.

## Related Work & Insights

- The Task Prototype concept is generalizable to multimodal learning, where different modalities can be treated as different "tasks."
- The AKG Loss design (TKE + TC) can serve as a general regularization scheme for prototype learning.
- Compared to DiffusionMTL: the effectiveness of diffusion models in MTL is sensitive to the type of input (prediction vs. feature), limiting robustness.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of Task Prototype and Knowledge Retrieval is novel)
- Technical Depth: ⭐⭐⭐⭐ (Loss design is hierarchically clear; VQ + Prototype + Transformer components work synergistically)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two datasets, extensive ablations, dimensionality analysis, comparison with prompt-based methods, and visualizations)
- Practical Value: ⭐⭐⭐⭐ (Partial annotation is a ubiquitous real-world scenario; the method exhibits good generalizability)
- Computational cost of affinity matrix grows with the number of tasks.
- Dynamic task weighting strategies could further improve performance.

## Related Work & Insights
- vs. pseudo-label methods: avoids negative transfer caused by noisy labels.
- vs. standard MTL: supports partially annotated settings.
- vs. task grouping methods: learns task associations adaptively via prototypes.

## Insights & Connections
The Task Prototype concept is broadly applicable to other scenarios requiring quantification of inter-task relationships. The AKG Loss design for maintaining prototype quality in a weakly supervised setting offers valuable reference for future work.

## Rating ⭐⭐⭐⭐ (4/5)
The problem is important and practically motivated, and the framework design is well-grounded. The method has direct applicability to multi-task perception in autonomous driving. Rating is conservative due to limited available information.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MAESTRO: Task-Relevant Optimization via Adaptive Feature Enhancement and Suppression for Multi-task 3D Perception](../../ICCV2025/autonomous_driving/maestro_task-relevant_optimization_via_adaptive_feature_enhancement_and_suppress.md)
- [\[AAAI 2026\] A Data-Driven Model Predictive Control Framework for Multi-Aircraft TMA Routing Under Travel Time Uncertainty](a_data-driven_model_predictive_control_framework_for_multi-aircraft_tma_routing_.md)
- [\[AAAI 2026\] Multimodal Data Fusion to Capture Dynamic Interactions between Built Environment and Vulnerable Older Adults](multimodal_data_fusion_to_capture_dynamic_interactions_between_built_environment.md)
- [\[ICCV 2025\] DuET: Dual Incremental Object Detection via Exemplar-Free Task Arithmetic](../../ICCV2025/autonomous_driving/duet_dual_incremental_object_detection_via_exemplar-free_task_arithmetic.md)
- [\[AAAI 2026\] RAST: A Retrieval Augmented Spatio-Temporal Framework for Traffic Prediction](rast_a_retrieval_augmented_spatio-temporal_framework_for_traffic_prediction.md)

</div>

<!-- RELATED:END -->
