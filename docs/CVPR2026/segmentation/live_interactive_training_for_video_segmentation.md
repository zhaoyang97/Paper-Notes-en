---
title: >-
  [Paper Note] Live Interactive Training for Video Segmentation
description: >-
  [CVPR 2026][Segmentation][Interactive video segmentation] LIT (Live Interactive Training) proposes a framework enabling interactive visual systems (e.g., SAM2) to learn online from user corrections during inference. Its lightweight implementation, LIT-LoRA, generalizes user feedback to subsequent frames by updating LoRA modules in real time, reducing user corrections by 18–34% on challenging VOS benchmarks with a training overhead of only ~0.5 seconds per correction.
tags:
  - CVPR 2026
  - Segmentation
  - Interactive video segmentation
  - online learning
  - LoRA adaptation
  - SAM2
  - user feedback-driven
date: 2026-05-08
content_hash: 575e52dc1e30cc61
---

# Live Interactive Training for Video Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.26929](https://arxiv.org/abs/2603.26929)
**Code**: [Project Page](https://youngxinyu1802.github.io/projects/LIT/)
**Area**: Segmentation / Video Object Segmentation
**Keywords**: Interactive video segmentation, online learning, LoRA adaptation, SAM2, user feedback-driven

## TL;DR

LIT (Live Interactive Training) proposes a framework enabling interactive visual systems (e.g., SAM2) to learn online from user corrections during inference. Its lightweight implementation, LIT-LoRA, generalizes user feedback to subsequent frames by updating LoRA modules in real time, reducing user corrections by 18–34% on challenging VOS benchmarks with a training overhead of only ~0.5 seconds per correction.

## Background & Motivation

- **Background**: Interactive video segmentation models such as SAM2 still require extensive user intervention in complex scenes involving occlusion, object splitting, and camouflage.
- **Limitations of Prior Work**: SAM2 treats user corrections only as immediate fix signals or stores them in a memory bank, while model parameters remain frozen. As a result, the model cannot genuinely learn from or generalize these interactions, trapping users in repetitive correction loops—e.g., segmenting a splitting card may require up to 14 corrections.
- **Key Challenge**: User-provided corrections contain rich domain adaptation information, yet existing models exploit them solely for immediate prediction rather than model improvement.
- **Core Idea**: Combine parameter-efficient fine-tuning (PEFT) with online learning to train lightweight LoRA modules in real time during inference, internalizing user feedback so that correction patterns generalize to subsequent frames within the same video. This constitutes a **user feedback-driven online learning paradigm**—conducted at inference time, supervised by human corrections rather than pseudo-labels.

## Method

### Overall Architecture

The LIT framework treats data as a stream $\{x_t\}_{t=1}^T$. The model produces predictions $\hat{y}_t = f_{\theta, \phi_t}(x_t)$, where $\theta$ denotes the frozen backbone parameters and $\phi_t$ the trainable lightweight adapter. Upon receiving a user correction $y_t^*$, the adapter is updated via gradient descent: $\phi_{t+1} \leftarrow \phi_t - \eta \nabla_{\phi_t} \mathcal{L}(f_{\theta,\phi_t}(x_t), y_t^*)$. The updated adapter is then used to improve predictions on subsequent frames. The adapter accumulates continuously within a video (streaming group) and is re-initialized when switching to a new video.

### Key Designs

1. **LIT-LoRA Online Adapter**:
    - **Function**: Learns user correction patterns in real time during inference with minimal overhead.
    - **Mechanism**: Injects low-rank residuals $W = W_0 + \Delta W$, $\Delta W = BA$ ($A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{d \times r}$, rank=4) into the Q/K/V projections of every attention layer in SAM2's mask decoder. Only 35K parameters are trainable; each correction requires ~0.5 seconds of training. Segmentation loss combines focal loss and dice loss at a weight ratio of 20:1.
    - **Design Motivation**: The low parameter count of LoRA enables fast convergence and low latency; restricting modifications to the mask decoder preserves the general-purpose features of the visual encoder.

2. **Correction Propagation and Validation Mechanism**:
    - **Function**: Automatically applies learned correction patterns to subsequent erroneous frames and validates prediction quality.
    - **Mechanism**: When an error occurs at frame $F_{t'}$, the updated LoRA generates a corrected prediction $M_{t'}^{\mathcal{A}}$. If the user accepts it (no new correction provided), the prediction is adopted and stored in the memory bank to enhance further propagation; if the user identifies a new error and provides a correction, the LoRA is incrementally updated. This forms a closed loop of "correct → learn → validate → accept/re-correct."
    - **Design Motivation**: Establishes a continuous human–machine collaborative adaptation cycle in which each correction strengthens model capability and progressively reduces repeated errors.

3. **Hybrid Correction Strategy**:
    - **Function**: Balances correction efficiency and segmentation quality.
    - **Mechanism**: Corrections are triggered when a frame's IoU falls below threshold $\tau_{\text{IoU}}$. Up to 3 clicks are first provided at the center of the erroneous region for local repair; if IoU remains insufficient, a complete ground-truth mask is supplied. Clicks enable rapid local adjustments, while complete masks handle complex failures.
    - **Design Motivation**: Simulates realistic user behavior in interactive annotation—attempting simple corrections first and providing detailed guidance only when necessary.

### Loss & Training

Online training loss: $\mathcal{L} = \lambda_{\text{focal}} \mathcal{L}_{\text{focal}} + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}$

LoRA configuration: rank=4, $\alpha=4$, dropout=0.1, learning rate $1 \times 10^{-4}$, 40 epochs per correction (with early stopping).

## Key Experimental Results

### Main Results

**Reduction in user corrections ($\tau_{\text{IoU}}=0.5$)**:

| Dataset | Baseline | LIT | Reduction |
|---------|----------|-----|-----------|
| VOST | 27.43 | 18.24 | ↓33.51% |
| LVOSv2 | 33.59 | 14.83 | ↓23.35% |
| MOSEv2 | 31.48 | 22.49 | ↓18.22% |
| SA-V Val | 20.66 | 12.90 | ↓18.16% |
| SA-V Test | 20.90 | 13.09 | ↓22.35% |

**Reduction in annotation time ($\tau_{\text{IoU}}=0.5$)**:

| Dataset | Baseline (min) | LIT (min) | Reduction |
|---------|----------------|-----------|-----------|
| VOST | 18.42 | 12.91 | ↓29.94% |
| LVOSv2 | 14.83 | 11.86 | ↓20.03% |

**Cross-model generalization (VOST)**:

| Model | Baseline | LIT | Reduction |
|-------|----------|-----|-----------|
| DAM4SAM | 34.60 | 22.46 | ↓35.09% |
| SAMURAI | 26.96 | 21.23 | ↓21.25% |

**Cross-task generalization (CLIP image classification)**:

| Dataset | Baseline | LIT | Reduction |
|---------|----------|-----|-----------|
| CUB-200-2011 | 13.04 | 8.53 | ↓34.55% |
| Stanford Cars | 13.38 | 7.57 | ↓43.40% |
| SUN397 | 13.92 | 8.95 | ↓35.70% |

### Ablation Study

| Configuration | Corrections | Parameters | Notes |
|---------------|-------------|------------|-------|
| Baseline (no adapter) | 27.43 | — | Baseline |
| Replace original decoder | 32.47 | 35K | Direct fine-tuning degrades performance |
| LoRA (no continual learning) | 21.43 | 35K | Training on first correction only is insufficient |
| LIT-FT (full decoder) | 17.46 | 3.3M | Better but 100× more parameters |
| LIT-LoRA (3 LoRAs) | 17.87 | 105K | Marginally better but increases user cognitive load |
| **LIT-LoRA (ours)** | **18.24** | **35K** | Best efficiency–effectiveness trade-off |

### Key Findings

- Correction counts follow a long-tail distribution: a small number of challenging videos consume the majority of the interaction budget, which is precisely the regime where LIT-LoRA yields the largest gains.
- Directly fine-tuning the original decoder leads to overfitting and corrupts stable representations (32.47 > 27.43), confirming the necessity of LoRA.
- Continual learning (updating at every correction) significantly outperforms training on the first correction only (18.24 vs. 21.43).
- A user study (6 participants × 7 videos) confirms a 41.92% reduction in corrections and a 23.04% reduction in time in real-world settings.

## Highlights & Insights

- The core insight is highly practical: existing interactive systems waste the generalization potential of user feedback by treating it only as an immediate fix.
- The framework is notably general—model-agnostic (SAM2/DAM4SAM/SAMURAI) and task-agnostic (VOS/image classification)—demonstrating the universality of online adaptation.
- The minimal overhead of 35K parameters and ~0.5 seconds per training step makes deployment in real annotation workflows genuinely feasible.
- An in-depth analysis of SAM2's predicted IoU reveals its unreliability as an automatic quality estimator, constituting a valuable finding in its own right.

## Limitations & Future Work

- The approach relies on user monitoring to detect errors; SAM2 lacks a reliable internal quality estimator, as its predicted IoU is poorly aligned with ground-truth IoU.
- Experiments primarily use simulated user corrections; real users may exhibit different correction strategies and behavioral patterns.
- LoRA adaptation converges rapidly only when the base model itself possesses strong generalization capability.
- In adversarial scenarios (e.g., severe object camouflage, extremely small targets, drastic deformation), the capacity of the LoRA adapter may be insufficient.

## Related Work & Insights

- **vs. SAM2 native interaction**: SAM2 stores user corrections in a memory bank without updating model parameters; LIT-LoRA achieves genuine learning through parameter updates.
- **vs. SAM2Long/SAMURAI**: These methods improve memory design or incorporate motion cues but do not exploit user feedback for adaptation. LIT-LoRA is orthogonal and can be stacked on top of them.
- **vs. OSVOS/OnAVOS**: OSVOS fine-tunes on the first frame (TTT paradigm); OnAVOS continuously adapts using pseudo-labels (CTTA paradigm). LIT-LoRA's continuous adaptation with real user feedback is more reliable.
- **Insight**: The paradigm of "learning from user feedback at inference time" is generalizable to all interactive AI systems, including interactive image editing and conversational AI.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of combining online learning with user interaction is clear and convincing, though the core techniques (LoRA + online fine-tuning) are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five VOS datasets, three classification datasets, three model backbones, a user study, and detailed ablations—comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem formulation is precise, motivation is clearly articulated, experimental design is rigorous, and supplementary material is thorough.
- **Value**: ⭐⭐⭐⭐ Direct practical value for annotation workflows (18–34% correction reduction translates to hours of saved annotation time); the framework's generality further amplifies its impact.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] INSID3: Training-Free In-Context Segmentation with DINOv3](insid3_training-free_in-context_segmentation_with_dinov3.md)
- [\[ICLR 2026\] VIRTUE: Visual-Interactive Text-Image Universal Embedder](../../ICLR2026/segmentation/virtue_visual-interactive_text-image_universal_embedder.md)
- [\[CVPR 2026\] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)
- [\[CVPR 2026\] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)
- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)

<!-- RELATED:END -->
