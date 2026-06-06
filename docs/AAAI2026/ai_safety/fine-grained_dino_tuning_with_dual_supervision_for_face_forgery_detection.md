---
title: >-
  [Paper Note] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection
description: >-
  [AAAI 2026][AI Safety][Deepfake Detection] This paper proposes DFF-Adapter (DeepFake Fine-Grained Adapter), a lightweight fine-tuning scheme for deepfake detection built upon DINOv2. A three-branch adapter (authenticity…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Deepfake Detection"
  - "DINOv2"
  - "Parameter-Efficient Fine-Tuning"
  - "Fine-Grained Classification"
  - "LoRA"
date: 2026-05-08
content_hash: 3607ae82d3bd15d5
---

# Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection

**Conference**: AAAI 2026
**arXiv**: [2511.12107](https://arxiv.org/abs/2511.12107)  
**Code**: None  
**Area**: AI Security
**Keywords**: Deepfake Detection, DINOv2, Parameter-Efficient Fine-Tuning, Fine-Grained Classification, LoRA

## TL;DR

This paper proposes DFF-Adapter (DeepFake Fine-Grained Adapter), a lightweight fine-tuning scheme for deepfake detection built upon DINOv2. A three-branch adapter (authenticity detection head, forgery type classification head, and shared head) is injected into each Transformer block. A Forgery-Aware Multi-Head Router enables subspace-level dynamic routing among LoRA experts. The auxiliary forgery type classification task enhances artifact sensitivity for the primary task. With only 3.5M trainable parameters, the method achieves state-of-the-art performance across multiple cross-dataset evaluations.

## Background & Motivation

### Challenges in Deepfake Detection

Deepfake technology is advancing rapidly, and synthetic content is becoming increasingly realistic, rendering traditional forensic methods inadequate. The core challenge is **generalization**—detectors trained on one dataset often fail to detect samples from other datasets or unseen forgery methods.

### Limitations of Prior Work

**Traditional methods**: Physiological/physical artifact detection, noise residual analysis, and feature consistency analysis exhibit strong dependence on specific forgery methods and offer limited generalizability.

**Methods based on pre-trained large models**:
- Insufficient fine-tuning scope: adapters are inserted only into the last one or a few Transformer blocks.
- Lack of task-specific design: deepfake detection is treated simply as a binary classification task, ignoring the **distinct artifact patterns** produced by different forgery methods.

### Why DINOv2

Unlike CLIP (which aligns image-text pairs and emphasizes semantics), DINOv2 adopts a purely visual self-supervised learning paradigm, better preserving fine-grained **local texture and geometric structure**, making it more sensitive to subtle forgery traces.

### Core Motivation

**Different forgery methods produce distinct artifact patterns**, and these method-specific cues are informative yet underexplored. If a model not only judges "real/fake" but also identifies "which forgery method," it necessarily becomes more sensitive to various artifacts. Therefore, **forgery type classification** can serve as an auxiliary task to enhance **authenticity discrimination**.

## Method

### Overall Architecture

DFF-Adapter is injected into each Transformer block of the frozen DINOv2 backbone. The training-phase three-branch structure consists of:
1. **Authenticity detection branch**: binary BCE loss
2. **Forgery type classification branch**: multi-class cross-entropy loss
3. **Shared branch**: transfers fine-grained forgery cues between the two tasks

At inference, only the authenticity detection branch fused with the shared branch is used.

### Key Designs

#### 1. **Forgery-Aware Multi-Head Router (DF-MHR)**

The core component, enabling subspace-level dynamic routing among LoRA experts.

**Channel splitting**: the hidden state $\mathbf{X} \in \mathbb{R}^{L \times d}$ is split along the channel dimension into $h$ head adapters.

**Shared LoRA expert pool**: $N$ low-rank LoRA experts $\{(\mathbf{A}_j, \mathbf{B}_j)\}$ shared across all head adapters.

**Task-specific routing**: for the $k$-th head adapter of task $t$, Top-3 routing with softmax is applied:
$$f_t^k = \beta \sum_{j \in S_{t,k}} \tilde{g}_{t,k}^{(j)} \mathbf{B}_j(\mathbf{A}_j \mathbf{X}^{(k)})$$
$$S_{t,k} = \text{Top3}(N, g_{t,k}), \quad g_{t,k} = \sigma(\mathbf{Z}_{task}[t,k])$$

**Shared head adapter**: uses global routing with soft weighting over all $N$ experts.

Design motivation: different channel subspaces encode different aspects of forgery features; after splitting, each subspace can independently route to the most suitable expert. Top-3 routing is sparser and more efficient than full soft weighting.

#### 2. **Shared-Enhanced Task Fusion (SETF)**

Task-specific and shared updates are propagated across all Transformer blocks via residual fusion:
$$f_{bin} = h_{cls} + \text{concat}(f_{share}, f_0^1, ..., f_0^n)$$
$$f_{ftc} = h_{cls} + \text{concat}(f_{share}, f_1^1, ..., f_1^n)$$

Design motivation: the shared branch acts as a bridge—when the forgery type classification task forces the shared branch to learn method-specific artifact features, those features automatically flow into the authenticity branch.

#### 3. **Dual-Task Decoupled Training**

Two forward passes are performed per mini-batch (task flag=0 for authenticity + task flag=1 for forgery type) to avoid gradient interference:
$$\mathcal{L} = \lambda_0 \mathcal{L}_{bce} + \lambda_1 \mathcal{L}_{ftc}$$

### Loss & Training

- Backbone frozen (DINOv2-Large-with-registers); only DFF-Adapter is trained
- Trainable parameters: **only 3.5M**
- Configuration: rank $r=16$, $\alpha=32$, 6 LoRA experts, 4 head adapters
- Adam optimizer, lr=$2 \times 10^{-4}$, 50 epochs
- Loss weights: $\lambda_0 = 10$, $\lambda_1 = 2$

## Key Experimental Results

### Main Results

**Cross-dataset evaluation (trained on FF++ c23, AUC %)**:

| Method | Conference | CDF-v2 | DFDC | CDF-v1 | DFDCP | Avg. |
|--------|-----------|--------|------|--------|-------|------|
| SBI | CVPR'22 | 93.82 | 74.47 | 93.44 | 90.95 | 88.17 |
| LVLM-DFD | ICML'25 | 94.71 | 79.12 | 97.62 | 91.81 | 90.82 |
| VB-StA | CVPR'25 | 94.7 | 84.3 | - | 90.9 | - |
| UDD | AAAI'25 | 93.13 | 81.21 | - | 88.11 | - |
| **DFF-Adapter** | - | **95.26** | **89.96** | 96.14 | **91.57** | **93.23** |

Average AUC of 93.23%, surpassing LVLM-DFD by 2.41 points. Gain on DFDC: +10.84.

**Cross-forgery-method evaluation (DF40)**:

| Method | FaceDancer | InSwapper | FSGAN | HyperReenact | Wav2Lip | DiT-XL/2 |
|--------|-----------|-----------|-------|-------------|---------|----------|
| LVLM-DFD | 82.97 | 87.64 | 93.75 | 81.56 | 78.60 | 86.61 |
| **DFF-Adapter** | **93.15** | **93.98** | **98.84** | **90.28** | **91.65** | **98.44** |

Best performance achieved across all forgery methods.

### Ablation Study

**Impact of core components (AUC %)**:

| Configuration | CDF-v2 | DFDC | CDF-v1 | DFDCP |
|--------------|--------|------|--------|-------|
| DINOv2 linear probing only | 61.93 | 52.83 | 68.46 | 55.54 |
| + FAMHR | 89.56 | 86.24 | 85.53 | 87.11 |
| **+ FAMHR + SETF** | **95.26** | **89.96** | **96.14** | **91.57** |

**Comparison with other fine-tuning strategies**:

| Fine-tuning Method | CDF-v2 | DFDC | DFDCP |
|-------------------|--------|------|-------|
| Linear Probing | 66.63 | 72.54 | 65.90 |
| LoRA | 85.14 | 79.95 | 81.49 |
| MoE-FFD | 80.40 | 76.52 | 87.60 |
| **DFF-Adapter** | **95.26** | **89.96** | **91.57** |

**Identity-constrained training (generalization under very few identities)**:

| # Training Identities | CDF-v2 | DFDC | CDF-v1 | DFDCP |
|----------------------|--------|------|--------|-------|
| 10 (~1%) | 81.97 | 82.63 | 81.62 | 82.22 |
| 50 (~5%) | 87.28 | 82.37 | 91.26 | 86.16 |

Competitive performance is maintained with as few as 10 identities, demonstrating exceptional data efficiency.

### Key Findings

1. **Injecting adapters into all Transformer blocks** is critical, allowing shallow layers to participate in task-specific learning.
2. **Forgery type auxiliary supervision** provides fine-grained artifact information beyond binary classification.
3. **DINOv2 is more suitable than CLIP for forgery detection**.
4. **t-SNE visualization** confirms that DFF-Adapter forms sub-clusters differentiated by forgery type.

## Highlights & Insights

1. **Elegant auxiliary task design**: forgery type classification serves not only as an auxiliary task but also as implicit data augmentation.
2. **Bridging role of the shared branch**: no additional inference overhead from the forgery type branch at test time.
3. **Extremely high parameter efficiency**: 3.5M parameters, trainable on a single RTX 4090.
4. **+10.84 gain on DFDC**: robustness demonstrated on the most challenging benchmark.

## Limitations & Future Work

1. Relies on FF++ for training; larger training sets remain unexplored.
2. Training requires forgery type labels, which may be unavailable in certain practical scenarios.
3. Dual forward passes increase training computational cost.
4. Only image-level detection is performed; pixel-level forgery region localization is not addressed.

## Related Work & Insights

- **DINOv2**: visual foundation model backbone; self-supervised training preserves rich local features.
- **LoRA series**: foundation for parameter-efficient fine-tuning.
- **MoE-FFD**: the most direct baseline for comparison, but lacks task-specific design.
- **LVLM-DFD**: a strong baseline leveraging large language models.
- **SBI**: achieves generalization through self-blended image training.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual-task supervision combined with shared-branch design is genuinely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-dimensional evaluation covering cross-dataset, cross-forgery-method, and identity-constrained settings.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear and ablations are thorough.
- **Value**: ⭐⭐⭐⭐⭐ — 3.5M parameters, single-GPU training, SOTA performance; directly deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement](../../ICCV2025/ai_safety/lora-fair_federated_lora_fine-tuning_with_aggregation_and_initialization_refinem.md)
- [\[ICML 2026\] VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection](../../ICML2026/ai_safety/vpd-100k_towards_generalizable_and_fine-grained_visual_privacy_protection.md)
- [\[AAAI 2026\] Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation](plug-and-play_parameter-efficient_tuning_of_embeddings_for_federated_recommendat.md)
- [\[ICML 2026\] OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL](../../ICML2026/ai_safety/omnivl-guard_towards_unified_vision-language_forgery_detection_and_grounding_via.md)
- [\[AAAI 2026\] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception](detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)

</div>

<!-- RELATED:END -->
