---
title: >-
  [Paper Note] Efficient Multi-modal Large Language Models via Progressive Consistency Distillation
description: >-
  [NeurIPS 2025][Multimodal VLM][Multimodal LLM] This paper proposes EPIC, a framework that addresses the optimization difficulty caused by feature space perturbation during visual token compression training via progressive consistency distillation along two dimensions (Token and Layer), achieving efficient multimodal LLMs without modifying model architecture.
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Multimodal LLM"
  - "visual token compression"
  - "progressive distillation"
  - "consistency distillation"
  - "inference efficiency"
date: 2026-05-08
content_hash: e734c414d021bdbc
---

# Efficient Multi-modal Large Language Models via Progressive Consistency Distillation

**Conference**: NeurIPS 2025
**arXiv**: [2510.00515](https://arxiv.org/abs/2510.00515)  
**Code**: [Available](https://github.com/ZichenWen1/EPIC)  
**Area**: Multimodal VLM
**Keywords**: Multimodal LLM, visual token compression, progressive distillation, consistency distillation, inference efficiency

## TL;DR

This paper proposes EPIC, a framework that addresses the optimization difficulty caused by feature space perturbation during visual token compression training via progressive consistency distillation along two dimensions (Token and Layer), achieving efficient multimodal LLMs without modifying model architecture.

## Background & Motivation

Multimodal large language models (MLLMs) feed visual tokens extracted by a visual encoder into an LLM for understanding and reasoning. However, the large number of visual tokens (e.g., 576 tokens in LLaVA-v1.5) introduces substantial computational overhead:
- The quadratic complexity of attention mechanisms makes long token sequences a bottleneck at inference
- High-resolution images and multi-frame videos further exacerbate the problem

Existing visual token compression methods fall into two categories:

**Training-free methods** (FastV, SparseVLM, etc.): Prune tokens based on importance or redundancy, leading to noticeable performance degradation.

**Training-aware methods** (MQT-LLaVA, TokenPacker, etc.): Achieve flexible compression through architectural modifications.

**Core Problem**: Existing training-aware methods primarily rely on architectural improvements while **neglecting the training difficulties introduced by token compression**. As shown in Figure 1:
- Token compression alters the feature space distribution (introducing perturbations)
- Perturbations shift the optimal point in parameter space
- Higher compression ratios lead to larger shifts of the optimal point
- Direct training is prone to getting trapped in local optima

## Method

### Overall Architecture

EPIC is built on a standard MLLM architecture (CLIP + MLP projector + Vicuna LLM) and **does not modify any architectural components**. The core innovation lies in the training strategy: it decomposes feature space perturbation into token and layer dimensions, proposing Token Consistency Distillation (TCD) and Layer Consistency Distillation (LCD) respectively.

A single model with shared weights serves simultaneously as both teacher and student.

### Key Designs

**1. Token Consistency Distillation (TCD)**

Core Idea: **Progressively increasing the compression ratio** so that the optimal point shift at each step remains small, making optimization easier.

- **Student model**: At training iteration $t$, samples a compression ratio from the range $[R_{\min,t}^{\text{stu}}, R_{\max,t}^{\text{stu}}]$
- **Teacher model**: Uses a slightly lower compression ratio than the student (by a margin $\Delta_t$), providing better feature guidance
- **Progressive strategy**:
    - Early training: Both teacher and student use low compression ratios (easy tasks)
    - Late training: Compression ratios gradually increase, and the teacher–student gap $\Delta_t$ also increases progressively
- When the gap is too large, the student cannot receive effective guidance from the teacher; hence $\Delta_t$ also follows a progressive schedule

Any plug-and-play token compressor (FastV, DART, random pruning) can serve as the compression operator.

**2. Layer Consistency Distillation (LCD)**

Based on the observation that visual token attention drops significantly in the deeper layers of the LLM, compression at deeper layers has a smaller impact on the output.

- Normalized training progress is defined as $\beta_t = t/T$
- Compression layer position: $\ell_t = \text{Round}(L - \beta_t(L - \ell_{\min}))$
- **Progressive strategy**: Compression begins at the deepest layers early in training (minimal impact) and gradually moves to shallower layers (increasing impact)
- The teacher–student compression ratio gap is maintained throughout

### Loss & Training

**TCD/LCD Training Objective**:
$$\mathcal{L}_{\text{total}}(\theta) = (1-\lambda) \cdot \mathcal{L}_{\text{SFT}}(\theta) + \lambda \cdot \mathcal{L}_{\text{TCD/LCD}}(\theta)$$

where $\lambda = 0.7$ and $\mathcal{L}_{\text{SFT}}$ is the standard autoregressive cross-entropy loss.

The distillation loss is a temperature-scaled KL divergence:
$$\mathcal{L}_{\text{TCD}}(\theta) = \mathbb{E}_{I,P,t}\left[\text{KL}(p_{\text{tea}} \| p_{\text{stu}})\right]$$
$$p_{\text{tea}} = \text{Softmax}(h_{\text{tea}}/\tau), \quad p_{\text{stu}} = \text{Softmax}(h_{\text{stu}}/\tau)$$

Training requires only a single stage (visual instruction fine-tuning), completed in approximately 12 hours on 8×A100 GPUs.

## Key Experimental Results

### Main Results

**Table 1: Performance on 10 visual understanding benchmarks (selected results, compared against LLaVA-v1.5-7B)**

| Method | #Visual Tokens | VQAv2 | GQA | MME | MMB | Avg. (%) | vs LLaVA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LLaVA-v1.5 | 576 | 72.2 | 61.9 | 1785 | 64.1 | 61.4 | — |
| MQT-LLaVA | 64 | 65.6 | 58.7 | 1810 | 61.3 | 56.8 | -4.6 |
| MQT-LLaVA | 256 | 68.3 | 60.1 | 1740 | 61.7 | 57.8 | -3.6 |
| TokenPacker | 144 | 71.3 | 62.0 | 1716 | 63.9 | 59.9 | -1.5 |
| **TCD (Ours)** | 256 | 72.7 | 61.4 | 1807 | **66.1** | **61.7** | **+0.3** |
| **TCD (Ours)** | 128 | 69.7 | 59.9 | **1861** | 65.6 | 61.3 | -0.1 |
| **TCD (Ours)** | 64 | 66.1 | 57.1 | 1809 | 64.2 | 59.4 | -2.0 |
| **LCD (Ours)** | 256 | 72.6 | 62.0 | 1834 | 64.3 | **62.2** | **+0.8** |
| **LCD (Ours)** | 128 | 69.2 | 60.6 | 1832 | 64.1 | 61.3 | -0.1 |
| **LCD (Ours)** | 64 | 66.0 | 58.3 | 1794 | 62.1 | 59.4 | -2.0 |

Key findings:
- Retaining 128 tokens (77.8% compression) incurs virtually no performance loss (−0.1%)
- Retaining 256 tokens **surpasses** the original LLaVA-v1.5 (+0.3 ~ +0.8%)
- Even at 64 tokens, performance drops only 2%, far outperforming MQT-LLaVA, LLaVA-Mini, etc.

**Table 2: Inference efficiency analysis**

| Method | Visual Tokens | KV Cache (MB) ↓ | CUDA Time (s) ↓ | FLOPs (T) ↓ |
|:---:|:---:|:---:|:---:|:---:|
| LLaVA-v1.5 | 576 | 367.2 | 1103.5 | 9.3 |
| EPIC + DART | 64 | 40.9 (↓88.9%) | 744.3 (↓32.6%) | 1.5 (↓83.9%) |
| EPIC + Random | 64 | 40.9 (↓88.9%) | 697.3 (↓36.8%) | 1.5 (↓83.9%) |

KV cache reduced by **88.9%**, FLOPs reduced by **83.9%**, with a practical speedup of approximately **1.6×**.

### Ablation Study

**Table 3: TCD ablation**

| Method | VQAv2 | MME | MMB | Avg. (%) |
|:---:|:---:|:---:|:---:|:---:|
| TCD (128 tokens) | 69.7 | **1861** | **65.6** | **61.3** |
| w/o distillation loss | 67.2 | 1745 | 63.8 | 59.8 (−1.5) |
| w/o progressive compression ratio | 67.1 | 1788 | 63.8 | 59.1 (−2.2) |

**Table 4: LCD ablation**

| Method | VQAv2 | MME | MMB | Avg. (%) |
|:---:|:---:|:---:|:---:|:---:|
| LCD (128 tokens) | 69.2 | **1832** | 64.1 | **61.3** |
| w/o distillation loss | 67.1 | 1761 | 62.9 | 60.5 (−0.8) |
| w/o progressive compression layer | 68.7 | 1776 | 63.1 | 60.3 (−1.0) |

Key ablation conclusions:
- **Progressive strategy is indispensable**: Removing it causes TCD to drop 2.2% and LCD to drop 1.0% on average
- **Teacher guidance is effective**: Removing the distillation loss causes TCD to drop 1.5% and LCD to drop 0.8%
- TCD is more sensitive to the progressive strategy; LCD requires both components

### Key Findings

1. **Extreme compression yields diminishing returns**: FLOPs drop sharply from 576 to 128 tokens, but further reduction from 64 to 36→18 yields minimal FLOPs savings while causing severe performance degradation. The authors define a "high-ROI region" (≥64 tokens) and a "low-ROI region" (<64 tokens).
2. **Good generalization across compression strategies**: Models trained with DART also perform well at inference with FastV and Random pruning.
3. **Low training cost**: Only a single-stage fine-tuning is required (12 hours vs. 30–48 hours for architecture modification methods).
4. **Implies substantial visual token redundancy**: 128 tokens suffice to match the performance of 576 tokens.

## Highlights & Insights

1. **Deep problem insight**: This work is the first to explicitly identify that the core difficulty of token compression lies in feature space perturbation → shift of the optimal point in parameter space → optimization trapped in local optima.
2. **No architectural modification required**: The approach is a pure training strategy improvement, compatible with any plug-and-play token compressor.
3. **Elegant self-distillation design**: Teacher and student share weights, eliminating the need to maintain a separate model.
4. **Theoretically intuitive progressive learning**: The optimal point shift at each step is small, analogous to curriculum learning.

## Limitations & Future Work

1. Validation is currently limited to LLaVA-v1.5 (7B); applicability to larger models (13B+) and newer architectures (LLaVA-Next, etc.) remains to be verified.
2. TCD and LCD are currently used independently; the effect of joint application has not been explored.
3. The progressive schedule (linear growth) may not be optimal; adaptive scheduling strategies warrant further investigation.
4. Sensitivity analysis of the distillation hyperparameters $\lambda=0.7$ and temperature $\tau$ is insufficient.
5. Evaluation on video understanding scenarios (where token counts are even larger) has not been addressed.

## Related Work & Insights

- **LLaVA** [Liu et al., 2023/2024]: The base architecture of this work, using 576 visual tokens in full.
- **FastV** [Chen et al., 2024]: Training-free token pruning based on attention scores; one of the compression components in this work.
- **MQT-LLaVA** [Li et al., 2024]: Dynamic Q-former encoding for variable-length tokens; the primary training-aware baseline.
- **TokenPacker** [Li et al., 2024]: Coarse-to-fine visual projector that compresses tokens via architectural modification.
- **VoCo-LLaMA** [Ye et al., 2024]: Transfers visual information to a small number of VoCo tokens via attention modification.

## Rating

- **Novelty**: ★★★★☆ — The progressive distillation framework is novel; the combination of self-distillation and progressive learning is elegant.
- **Technical Depth**: ★★★★☆ — Design intuition is clear; the decomposition into token and layer dimensions is insightful.
- **Experimental Thoroughness**: ★★★★★ — 10 benchmarks, 3 compression strategies, comprehensive ablations, and efficiency analysis are all included.
- **Writing Quality**: ★★★★☆ — Loss landscape visualization (Figure 1) is intuitive and effective; tables are well-organized.
- **Practicality**: ★★★★★ — Code is open-sourced, no architectural modifications are required, training cost is low, and the method is directly deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[CVPR 2025\] A Two-Stage Progressive Pre-training using Multi-Modal Contrastive Masked Autoencoders](../../CVPR2025/multimodal_vlm/multi-modal_contrastive_masked_autoencoders_a_two-stage_progressive_pre-training.md)
- [\[NeurIPS 2025\] mmWalk: Towards Multi-modal Multi-view Walking Assistance](mmwalk_towards_multi-modal_multi-view_walking_assistance.md)
- [\[ACL 2025\] Jailbreak Large Vision-Language Models Through Multi-Modal Linkage](../../ACL2025/multimodal_vlm/jailbreak_large_vision-language_models_through_multi-modal_linkage.md)

</div>

<!-- RELATED:END -->
