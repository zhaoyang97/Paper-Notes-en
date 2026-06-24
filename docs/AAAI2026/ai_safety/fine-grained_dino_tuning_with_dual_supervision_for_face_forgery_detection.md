---
title: >-
  [Paper Note] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection
description: >-
  [AAAI 2026][AI Safety][Deepfake Detection] Proposed DFF-Adapter (DeepFake Fine-Grained Adapter), a lightweight fine-tuning scheme for DINOv2 tailored for deepfake detection. By injecting a three-branch adapter (authenticity detection head, forgery type classification head, and shared head) into each Transformer block combined with a Forgery-Aware Multi-Head Router to enable subspace-level dynamic routing of LoRA experts, it leverages the auxiliary forgery type classification…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Deepfake Detection"
  - "DINOv2"
  - "Parameter-Efficient Fine-Tuning"
  - "Fine-Grained Classification"
  - "LoRA"
date: 2026-05-08
content_hash: e3e848cf97e73e9c
---

# Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection

**Conference**: AAAI 2026  
**arXiv**: [2511.12107](https://arxiv.org/abs/2511.12107)  
**Code**: None  
**Area**: AI Security  
**Keywords**: Deepfake Detection, DINOv2, Parameter-Efficient Fine-Tuning, Fine-Grained Classification, LoRA

## TL;DR

Proposed DFF-Adapter (DeepFake Fine-Grained Adapter), a lightweight fine-tuning scheme for DINOv2 tailored for deepfake detection. By injecting a three-branch adapter (authenticity detection head, forgery type classification head, and shared head) into each Transformer block combined with a Forgery-Aware Multi-Head Router to enable subspace-level dynamic routing of LoRA experts, it leverages the auxiliary forgery type classification task to enhance the artifact sensitivity of the primary task, achieving SOTA performance in multiple cross-dataset evaluations with only 3.5M trainable parameters.

## Background & Motivation

### Challenges in Deepfake Detection

With the rapid development of deepfake technologies, synthetic content has become increasingly realistic, making traditional forensic methods struggle to cope. The core challenge is **generalizability**—detectors trained on one dataset often fail to effectively detect samples from other datasets or unknown forgery methods.

### Limitations of Prior Work

**Traditional Methods**: Physiological/physical artifact detection, noise residual analysis, and feature consistency analysis strongly depend on specific forgery methods, leading to limited generalization.

**Methods Based on Pre-trained Foundation Models**:
   - Insufficient fine-tuning scope: Adapters are only inserted in the last or a few Transformer blocks.
   - Lack of task-specific design: Deepfake detection is simply treated as a binary classification task, ignoring the **distinct artifact patterns** produced by different forgery methods.

### Why DINOv2

Unlike CLIP (image-text alignment, semantically oriented), DINOv2 adopts a purely visual self-supervised learning paradigm, which better preserves fine-grained **local textures and geometric structures** and is more sensitive to subtle forgery traces.

### Core Motivation

**Different forgery methods produce distinct artifact patterns**, and these method-specific clues are informative but underestimated. If a model can not only determine "real/fake" but also identify "which forgery method", it will naturally be more sensitive to various artifacts. Therefore, **forgery type classification** can serve as an auxiliary task to enhance **authenticity detection**.

## Method

### Overall Architecture

Inject the DFF-Adapter into each Transformer block of the frozen DINOv2 backbone. The three-branch structure during the training phase includes:
1. **Authenticity Detection Branch**: Binary classification BCE loss
2. **Forgery Type Classification Branch**: Multi-class cross-entropy loss
3. **Shared Branch**: Transfers fine-grained forgery clues between the two tasks

Inference phase: Only the authenticity detection branch, fused with the shared branch, is used.

### Key Designs

#### 1. **Forgery-Aware Multi-Head Router（DF-MHR）**

A core component achieving subspace-level dynamic routing of LoRA experts.

**Channel Splitting**: The hidden state $\mathbf{X} \in \mathbb{R}^{L \times d}$ is split along the channel dimension into $h$ head adapters.

**Shared LoRA Expert Pool**: $N$ low-rank LoRA experts $\{(\mathbf{A}_j, \mathbf{B}_j)\}$ shared across all head adapters.

**Task-Specific Routing**: For the $k$-th head adapter of task $t$, routing is performed via softmax + Top-3:
$$f_t^k = \beta \sum_{j \in S_{t,k}} \tilde{g}_{t,k}^{(j)} \mathbf{B}_j(\mathbf{A}_j \mathbf{X}^{(k)})$$
$$S_{t,k} = \text{Top3}(N, g_{t,k}), \quad g_{t,k} = \sigma(\mathbf{Z}_{task}[t,k])$$

**Shared Head Adapter**: Employs global routing with soft-weighting across all $N$ experts.

Design Motivation: Different channel subspaces encode different aspects of forgery features; after splitting, each subspace can be routed independently to the most suitable experts. Top-3 routing is sparser and more efficient than fully weighted routing.

#### 2. **Shared-Enhanced Task Fusion**

Pass shared and task-specific updates across all Transformer blocks via residual fusion:
$$f_{bin} = h_{cls} + \text{concat}(f_{share}, f_0^1, ..., f_0^n)$$
$$f_{ftc} = h_{cls} + \text{concat}(f_{share}, f_1^1, ..., f_1^n)$$

Design Motivation: The shared branch acts as a bridge—when the forgery type classification task forces the shared branch to learn method-specific artifact features, these features automatically flow into the authenticity branch.

#### 3. **Dual-Task Decoupled Training**

Two forward passes (task flag=0 for authenticity + task flag=1 for forgery type) are performed for each mini-batch to avoid gradient interference:
$$\mathcal{L} = \lambda_0 \mathcal{L}_{bce} + \lambda_1 \mathcal{L}_{ftc}$$

### Loss & Training

- Backbone frozen (DINOv2-Large-with-registers), training only the DFF-Adapter
- Trainable Parameters: **Only 3.5M**
- Configuration: rank $r=16$, $\alpha=32$, 6 LoRA experts, 4 head adapters
- Adam optimizer, lr=$2 \times 10^{-4}$, 50 epochs
- Loss weights: $\lambda_0 = 10$, $\lambda_1 = 2$

## Key Experimental Results

### Main Results

**Cross-Dataset Evaluation (Trained on FF++ c23, AUC %)**:

| Method | Conference | CDF-v2 | DFDC | CDF-v1 | DFDCP | Mean |
|------|------|--------|------|--------|-------|------|
| SBI | CVPR'22 | 93.82 | 74.47 | 93.44 | 90.95 | 88.17 |
| LVLM-DFD | ICML'25 | 94.71 | 79.12 | 97.62 | 91.81 | 90.82 |
| VB-StA | CVPR'25 | 94.7 | 84.3 | - | 90.9 | - |
| UDD | AAAI'25 | 93.13 | 81.21 | - | 88.11 | - |
| **DFF-Adapter** | - | **95.26** | **89.96** | **96.14** | **91.57** | **93.23** |

Average AUC is 93.23%, outperforming LVLM-DFD by 2.41 percentage points. Achieves a +10.84 gain on DFDC.

**Cross-Forgery Method Evaluation (DF40)**:

| Method | FaceDancer | InSwapper | FSGAN | HyperReenact | Wav2Lip | DiT-XL/2 |
|------|-----------|-----------|-------|-------------|---------|----------|
| LVLM-DFD | 82.97 | 87.64 | 93.75 | 81.56 | 78.60 | 86.61 |
| **DFF-Adapter** | **93.15** | **93.98** | **98.84** | **90.28** | **91.65** | **98.44** |

Achieves optimal performance across all forgery methods.

### Ablation Study

**Impact of Core Components (AUC %)**:

| Configuration | CDF-v2 | DFDC | CDF-v1 | DFDCP |
|------|--------|------|--------|-------|
| DINOv2 Linear Probing Only | 61.93 | 52.83 | 68.46 | 55.54 |
| + FAMHR | 89.56 | 86.24 | 85.53 | 87.11 |
| **+ FAMHR + SETF** | **95.26** | **89.96** | **96.14** | **91.57** |

**Comparison with Other Fine-Tuning Strategies**:

| Fine-Tuning Strategy | CDF-v2 | DFDC | DFDCP |
|----------|--------|------|-------|
| Linear Probing | 66.63 | 72.54 | 65.90 |
| LoRA | 85.14 | 79.95 | 81.49 |
| MoE-FFD | 80.40 | 76.52 | 87.60 |
| **DFF-Adapter** | **95.26** | **89.96** | **91.57** |

**Identity-Restricted Training (Generalization under limited identities)**:

| No. of Training Identities | CDF-v2 | DFDC | CDF-v1 | DFDCP |
|-----------|--------|------|--------|-------|
| 10 (~1%) | 81.97 | 82.63 | 81.62 | 82.22 |
| 50 (~5%) | 87.28 | 82.37 | 91.26 | 86.16 |

Competitiveness is maintained with only 10 identities, demonstrating extremely high data efficiency.

### Key Findings

1. **Injecting adapters into all Transformer blocks** is crucial, allowing shallow layers to participate in task-specific learning.
2. **Auxiliary supervision of forgery types** provides fine-grained artifact information beyond binary classification.
3. **DINOv2 is more suitable for forgery detection than CLIP**.
4. **t-SNE visualization** confirms that DFF-Adapter forms sub-clusters distinguished by forgery types.

## Highlights & Insights

1. **Exquisite Auxiliary Task Design**: Forgery type classification is not only an auxiliary task but also acts as implicit data augmentation.
2. **Bridging Role of the Shared Branch**: Eliminates the extra overhead of the forgery type branch during inference.
3. **Extremely High Parameter Efficiency**: 3.5M parameters, trainable on a single RTX 4090 GPU.
4. **+10.84 Gain on DFDC**: Demonstrates robustness on the most challenging baseline.

## Limitations & Future Work

1. Remains reliant on training on FF++, without exploring larger datasets.
2. Requires forgery type labels during training, which may be unavailable in some real-world scenarios.
3. Dual forward passes increase training computational cost.
4. Limited to image-level detection, without performing pixel-level forgery region localization.

## Related Work & Insights

- **DINOv2**: Vision foundation model backbone, retaining rich local features via self-supervised learning.
- **LoRA Family**: The foundations of parameter-efficient fine-tuning.
- **MoE-FFD**: The most direct baseline of comparison, yet lacking task-specific designs.
- **LVLM-DFD**: A strong baseline utilizing large vision-language models.
- **SBI**: Self-blending images training for generalization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Novel dual-task supervision and shared branch design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-dimensional evaluations including cross-dataset, cross-forgery-method, and identity-restricted setups.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and detailed ablations.
- **Value**: ⭐⭐⭐⭐⭐ — 3.5M parameters, single-GPU training, SOTA performance; directly deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DiffusionFF: A Diffusion-based Framework for Joint Face Forgery Detection and Fine-Grained Artifact Localization](../../CVPR2026/ai_safety/diffusionff_a_diffusion-based_framework_for_joint_face_forgery_detection_and_fin.md)
- [\[ICML 2026\] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](../../ICML2026/ai_safety/towards_fine-grained_robustness_attention-guided_test-time_prompt_tuning_for_vis.md)
- [\[CVPR 2025\] Towards General Visual-Linguistic Face Forgery Detection](../../CVPR2025/ai_safety/towards_general_visual-linguistic_face_forgery_detection.md)
- [\[AAAI 2026\] Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation](plug-and-play_parameter-efficient_tuning_of_embeddings_for_federated_recommendat.md)
- [\[AAAI 2026\] Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)

</div>

<!-- RELATED:END -->
