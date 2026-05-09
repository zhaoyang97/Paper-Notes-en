---
title: >-
  [Paper Note] Evolving Prompt Adaptation for Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Prompt learning] This paper proposes EvoPrompt, a framework that treats prompt training as a progressive evolution from general semantic anchors to task-specific features. It introduces a Modal-shared Prompt Projector (MPP) for unified cross-layer and cross-modal prompt generation, an evolution trajectory-aware strategy (direction–magnitude decoupling with historical direction freezing) to prevent forgetting, and Feature Geometry Regularization (FGR) to prevent representation collapse. EvoPrompt achieves an average HM of 80.73% on base-to-novel generalization across 11 datasets, surpassing all existing prompt learning methods.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Prompt learning
  - vision-language models
  - catastrophic forgetting
  - low-rank adaptation
  - feature regularization
date: 2026-05-08
content_hash: a50a3eccbae51b91
---

# Evolving Prompt Adaptation for Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.09493](https://arxiv.org/abs/2603.09493)
**Code**: N/A
**Area**: Multimodal VLM
**Keywords**: Prompt learning, vision-language models, catastrophic forgetting, low-rank adaptation, feature regularization

## TL;DR

This paper proposes EvoPrompt, a framework that treats prompt training as a progressive evolution from general semantic anchors to task-specific features. It introduces a Modal-shared Prompt Projector (MPP) for unified cross-layer and cross-modal prompt generation, an evolution trajectory-aware strategy (direction–magnitude decoupling with historical direction freezing) to prevent forgetting, and Feature Geometry Regularization (FGR) to prevent representation collapse. EvoPrompt achieves an average HM of 80.73% on base-to-novel generalization across 11 datasets, surpassing all existing prompt learning methods.

## Background & Motivation

**State of the Field**: Large-scale VLMs such as CLIP acquire strong zero-shot capabilities through contrastive pre-training. Prompt learning (CoOp/CoCoOp/MaPLe) enables efficient downstream adaptation with minimal trainable parameters and has become the mainstream parameter-efficient fine-tuning paradigm.

**Limitations of Prior Work**: (1) **Layer-wise isolation** — methods such as MaPLe parameterize prompts independently at each layer, disrupting the semantic hierarchical flow in deeper encoder layers and preventing information learned at lower layers from propagating upward; (2) **Modality bias** — existing approaches are text-centric and fail to exploit complementary visual–language information; (3) **Catastrophic forgetting** — under few-shot adaptation, prompts rapidly deviate from pre-trained semantic anchors, leading to overfitting on downstream data and loss of zero-shot generalization.

**Root Cause**: Prompt learning must acquire task-specific features, yet unconstrained optimization overwrites pre-trained knowledge. This is fundamentally a trade-off between adaptation strength and knowledge retention.

**Paper Goals**: To explicitly guide the evolution trajectory of prompts during few-shot prompt learning, enabling simultaneous acquisition of task-specific features and retention of pre-trained knowledge.

**Starting Point**: Prompt training is recast as a progressive evolution from general semantic anchors to task-specific features. A key observation is that in low-rank adaptation, directions encode semantic knowledge (more critical), while magnitudes encode adaptation strength. Freezing learned directions and tuning only magnitudes enables continual learning without overwriting prior knowledge.

**Core Idea**: Direction–magnitude decoupling freezes historical semantic directions and tunes only magnitude coefficients; combined with a shared projector and feature regularization, this achieves controllable prompt evolution.

## Method

### Overall Architecture

The CLIP ViT-B/16 dual encoder is frozen. A unified learnable embedding space $E \in \mathbb{R}^{K \times d_r}$ ($K=5, d_r=512$) is initialized. MPP projects $E$ into per-layer, per-modal prompts via shared weights and layer-specific low-rank adapters. Prompts are injected into encoder layers $J=6$ through $L=12$. An evolution-aware training strategy progressively freezes historical directions epoch by epoch while tuning magnitudes. FGR constrains feature geometric structure. The full objective jointly optimizes InfoNCE, FGR, and a knowledge constancy loss.

### Key Designs

1. **Modal-shared Prompt Projector (MPP)**

   - **Function**: Generates cross-layer and cross-modal prompts from a unified embedding space, replacing the layer-wise independent parameterization of MaPLe.
   - **Mechanism**: For modality $m$ at layer $i$, the projection weight is $W_i^m = W_{\text{shared}}^m + A_i B_i$, where $W_{\text{shared}}^m$ is shared across layers to capture base semantics and $A_i B_i$ is a low-rank ($r \ll \min(d_r, d_m)$) layer-specific adapter. Parameter complexity is reduced from $\mathcal{O}((L-J+1) \cdot d_r d_m)$ to $\mathcal{O}(d_r d_m + (L-J+1) \cdot r(d_r + d_m))$, approximately 4.6× fewer parameters than MaPLe (0.764M vs. 3.555M).
   - **Design Motivation**: Cross-layer shared base semantics combined with low-rank adapters capturing layer-specific variations ensures inter-layer information flow while preserving expressiveness. The unified embedding space naturally supports cross-modal information interaction.

2. **Evolution Trajectory-Aware Learning Strategy**

   - **Function**: Decomposes the low-rank update at each epoch into direction and magnitude components, freezes historical directions, and tunes only magnitudes to achieve knowledge-preserving progressive adaptation.
   - **Mechanism**: At epoch $t$, $\Delta W_i^t$ is decomposed into magnitude $\alpha_i^t$ and normalized direction $\overline{A_i^t B_i^t}$ (Frobenius normalization). At epoch $T$, the accumulated weight is $W_i^T = W_{\text{shared}} + \sum_{t=1}^{T-1} \alpha_i^t \overline{A_i^t B_i^t} + \alpha_i^T \overline{A_i^T B_i^T}$. All historical directions $\{\overline{A_i^t B_i^t}\}_{t=1}^{T-1}$ are frozen to preserve geometric structure; only magnitude coefficients and the current new direction are trained. Adaptive rank reduction at predefined checkpoints $\mu, \nu$ further reduces late-stage overfitting.
   - **Design Motivation**: Directions encode semantic knowledge (prior work has established that directions are more critical than magnitudes); freezing directions preserves knowledge. Epoch-wise direction accumulation resembles knowledge accumulation in continual learning, while rank reduction acts as structured regularization.

3. **Feature Geometry Regularization (FGR)**

   - **Function**: Prevents feature dimension redundancy and representation collapse during InfoNCE training.
   - **Mechanism**: Derived from the Soft-HGR maximum correlation framework. $\mathcal{L}_{fgr} = \frac{1}{2} \text{tr}(\text{cov}(\mathcal{F}^v) \cdot \text{cov}(\mathcal{F}^t))$ minimizes the product of visual and textual feature covariance matrices to enforce feature decorrelation. The total loss is $\mathcal{L} = \mathcal{L}_{InfoNCE} + \gamma \mathcal{L}_{fgr} + \eta \mathcal{L}_{kcl}$, where $\mathcal{L}_{kcl}$ is the knowledge constancy loss constraining prompted features from deviating from original CLIP feature directions.
   - **Design Motivation**: InfoNCE focuses on instance-level alignment but ignores the geometric structure of the feature space; redundant dimensions under few-shot settings lead to overfitting.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{InfoNCE} + \gamma \mathcal{L}_{fgr} + \eta \mathcal{L}_{kcl}$ ($\gamma=25, \eta=0.5$); 16-shot/class; single NVIDIA A800 GPU; results averaged over 3 seeds.

## Key Experimental Results

### Main Results — Base-to-Novel Generalization (Average over 11 Datasets)

| Method | Base | Novel | HM |
|--------|------|-------|-----|
| CLIP | 69.34 | 74.22 | 71.70 |
| CoOp | 82.69 | 63.22 | 71.66 |
| MaPLe | 82.28 | 75.14 | 78.55 |
| PromptSRC | 84.26 | 76.10 | 79.97 |
| TCP | 84.13 | 75.36 | 79.51 |
| MMA | 83.20 | 76.80 | 79.87 |
| **EvoPrompt** | **84.28** | **77.76** | **80.73** |

### Ablation Study (ImageNet Base-to-Novel)

| Configuration | Base | Novel | HM |
|---------------|------|-------|-----|
| w/o MPP (layer-wise independent prompts) | 75.32 | 70.15 | 72.64 |
| w/o $W_{\text{shared}}$ | 75.80 | 71.42 | 73.54 |
| w/o AB (full-rank projection) | 76.15 | 70.90 | 73.43 |
| w/o evolution strategy | 77.42 | 70.25 | 73.66 |
| w/o $\mathcal{L}_{kcl}$ | 77.24 | 70.55 | 73.74 |
| w/o $\mathcal{L}_{fgr}$ | 76.70 | 70.52 | 73.48 |
| **EvoPrompt (Full)** | **76.98** | **71.80** | **74.29** |

### Key Findings

- MPP contributes most significantly: removing it drops HM from 74.29 to 72.64 (−1.65), confirming that the unified embedding space and shared projection are critical for cross-layer information flow.
- Removing the evolution strategy or $\mathcal{L}_{kcl}$ actually increases Base accuracy (77.42%/77.24%) but substantially decreases Novel accuracy (70.25%/70.55%), corroborating that their primary role is to prevent overfitting to base classes.
- Cross-dataset transfer (ImageNet → 10 target datasets) achieves an average of 66.82%, outperforming MMA (66.61%) and MaPLe (66.30%).
- Domain generalization (4 ImageNet variants) achieves state-of-the-art results, demonstrating effective preservation of CLIP's original OOD generalization capability.

## Highlights & Insights

1. **Direction–magnitude decoupling for forgetting prevention**: Decomposing low-rank updates into directions (encoding semantic knowledge) and magnitudes (encoding adaptation strength), then freezing historical directions while tuning magnitudes, is conceptually concise yet intuitively sound, and is transferable to any LoRA/adapter scenario requiring forgetting prevention.
2. **Theoretically grounded FGR**: FGR is derived from the Soft-HGR maximum correlation framework as a complementary term missing from InfoNCE, rather than being an ad hoc design.
3. **High parameter efficiency**: Only 0.764M trainable parameters (1/4.6 of MaPLe), yet HM improves by 2.18%.

## Limitations & Future Work

1. Experiments are conducted solely on ViT-B/16; performance on larger backbones (e.g., ViT-L/14) remains unverified.
2. The epoch-wise direction freezing and accumulation design introduces additional memory overhead, with historical matrices growing linearly as the number of epochs increases.
3. The rank reduction checkpoints $\mu, \nu$ are manually specified hyperparameters, lacking an adaptive selection strategy.
4. FGR computes covariance within a batch; estimation may be unreliable when batch size is small.

## Related Work & Insights

- **vs. MaPLe**: MaPLe employs layer-wise independent prompts with a text-centric design; EvoPrompt uses a shared projector for unified management with modality parity, achieving 4.6× fewer parameters and +2.18% HM.
- **vs. PromptSRC**: PromptSRC applies self-consistency regularization; EvoPrompt directly controls the evolution trajectory, achieving +0.76% HM.
- **vs. DePT/DualPrompt**: Feature-decoupling methods from the continual learning line; EvoPrompt's direction-freezing approach is more lightweight and requires no task identifiers.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐: The direction–magnitude decoupling approach to forgetting prevention is novel, and FGR is supported by theoretical derivation.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Four evaluation protocols with detailed ablations, though validation on larger backbones is absent.
- **Writing Quality** ⭐⭐⭐⭐: Motivation is clearly articulated and mathematical derivations are rigorous.
- **Value** ⭐⭐⭐⭐: Establishes a new paradigm for forgetting prevention in prompt learning; the direction-freezing idea is transferable to other PEFT methods.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] EvoPrompt: Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_vision-language_models.md)
- [\[CVPR 2026\] ReHARK: Refined Hybrid Adaptive RBF Kernels for Robust One-Shot Vision-Language Adaptation](rehark_refined_hybrid_adaptive_rbf_kernels_for_robust_one-shot_vision-language_a.md)
- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [\[CVPR 2026\] Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)
- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](understanding_task_transfer_in_vision-language_models.md)

<!-- RELATED:END -->
