---
title: >-
  [Paper Note] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models
description: >-
  [ICCV2025][Multimodal VLM][Knowledge Distillation] This paper proposes the LLaVA-KD framework, which efficiently transfers knowledge from large-scale MLLMs to small-scale MLLMs via two distillation strategies—Multimodal Distillation (MDist) and Relational Distillation (RDist)—combined with a three-stage training scheme (DPT-SFT-DFT), significantly improving small model performance without modifying the model architecture.
tags:
  - "ICCV2025"
  - "Multimodal VLM"
  - "Knowledge Distillation"
  - "Multimodal Large Language Models"
  - "Small Model Training"
  - "Vision-Language Alignment"
date: 2026-05-08
content_hash: d7b4493400a3ae5f
---

# LLaVA-KD: A Framework of Distilling Multimodal Large Language Models

**Conference**: ICCV2025
**arXiv**: [2410.16236](https://arxiv.org/abs/2410.16236)  
**Code**: [GitHub](https://github.com/Fantasyele/LLaVA-KD)  
**Area**: Multimodal VLM
**Keywords**: Knowledge Distillation, Multimodal Large Language Models, Small Model Training, Vision-Language Alignment

## TL;DR

This paper proposes the LLaVA-KD framework, which efficiently transfers knowledge from large-scale MLLMs to small-scale MLLMs via two distillation strategies—Multimodal Distillation (MDist) and Relational Distillation (RDist)—combined with a three-stage training scheme (DPT-SFT-DFT), significantly improving small model performance without modifying the model architecture.

## Background & Motivation

MLLMs have achieved remarkable success in unified vision-language understanding, but their continuously growing scale limits deployment in resource-constrained settings. Existing small-scale MLLMs (s-MLLMs) typically adopt lightweight LLM backbones to reduce computational cost, yet directly following the two-stage training paradigm (PT→SFT) of large models leads to significant performance degradation. For example, 4B TinyLLaVA achieves 65.0%, but performance drops sharply to 54.7% at 0.5B.

Prior work has attempted to address this through:
- **Architectural optimization**: MoE-LLaVA introduces a mixture-of-experts structure.
- **Training data optimization**: Bunny improves data quality via clustering and pruning.

However, these approaches either introduce additional parameters or incur higher data costs. The authors argue that **training paradigm optimization** is an underexplored yet highly promising direction. Existing LLM distillation methods focus solely on text-modality knowledge transfer, neglecting the critical role of visual representations in multimodal understanding, and directly incorporating distillation into the SFT stage yields only limited gains.

## Method

### Overall Architecture

LLaVA-KD consists of a large-scale teacher model (l-MLLM) and a small-scale student model (s-MLLM), both adopting the LLaVA-1.5 architecture (Visual Encoder + Projector + LLM). Teacher and student share the same visual encoder (SigLIP-B/14@384px), with a two-layer MLP projector mapping visual features $Z_v \in \mathbb{R}^{N_p \times C}$ to the text embedding space $H_v \in \mathbb{R}^{N_p \times D}$.

### Key Design 1: Multimodal Distillation (MDist)

MDist applies KL divergence distillation along two dimensions: **response tokens** and **visual tokens**.

**Response Distillation**: Aligns teacher and student output distributions over response tokens:

$$\mathcal{L}_{res} = \sum_{m=1}^{M} \text{KLD}(\phi_l(y_m | \mathbf{y}_{<m}), \phi_s(y_m | \mathbf{y}_{<m}))$$

**Visual Distillation**: Aligns teacher and student output distributions over visual tokens:

$$\mathcal{L}_{vis} = \sum_{k=1}^{K} \sum_{j=1}^{V} \phi_l(Y_j | \mathbf{y}_{<k}) \log \frac{\phi_l(Y_j | \mathbf{y}_{<k})}{\phi_s(Y_j | \mathbf{y}_{<k})}$$

where $K$ is the number of visual tokens and $V$ is the vocabulary size. Unlike conventional LLM distillation that targets only response tokens, MDist explicitly incorporates the visual modality into the distillation objective, ensuring comprehensive transfer of multimodal representations.

### Key Design 2: Relational Distillation (RDist)

RDist transfers the teacher model's ability to capture inter-visual-token relationships by constructing self-correlation matrices of visual tokens:

$$R_v^s = \mathbf{y}_v^s \otimes \mathbf{y}_v^s \in \mathbb{R}^{N_p \times N_p}, \quad R_v^t = \mathbf{y}_v^t \otimes \mathbf{y}_v^t \in \mathbb{R}^{N_p \times N_p}$$

The two matrices are then aligned by maximizing their cosine similarity:

$$\mathcal{L}_{rel} = 1 - \text{Cos}(R_v^s, R_v^t) = 1 - \frac{R_v^s \cdot R_v^t}{\|R_v^s\| \|R_v^t\|}$$

This design encodes spatial and semantic dependencies among visual tokens (e.g., object positions, interaction relationships), which are critical for understanding complex visual scenes.

### Three-Stage Training Scheme

1. **Distilled Pre-Training (DPT)**: The visual encoder and LLM are frozen; only the projector is trained. MDist and RDist are incorporated on top of the standard autoregressive loss: $\mathcal{L}_{DPT} = \mathcal{L}_{reg} + \alpha \mathcal{L}_{res} + \beta \mathcal{L}_{vis} + \gamma \mathcal{L}_{rel}$, enhancing vision-text alignment quality.

2. **Supervised Fine-Tuning (SFT)**: Standard SFT with the projector and LLM jointly optimized to establish foundational multimodal understanding capabilities.

3. **Distilled Fine-Tuning (DFT)**: Distillation is reintroduced after SFT to refine the student model's knowledge: $\mathcal{L}_{DFT} = \mathcal{L}_{reg} + \alpha' \mathcal{L}_{res} + \beta' \mathcal{L}_{vis} + \gamma' \mathcal{L}_{rel}$

All loss weights $\{\alpha, \beta, \gamma\}$ and $\{\alpha', \beta', \gamma'\}$ are set to 1.0, 1.0, and 0.5, respectively.

## Key Experimental Results

### Main Results: Comparison with State-of-the-Art

| Method | LLM | VQAv2 | GQA | SciQA | MME | MMB | POPE | Avg₁₀ |
|--------|-----|-------|-----|-------|-----|-----|------|-------|
| TinyLLaVA (Qwen1.5-0.5B) | 0.5B | 73.9 | 57.4 | 60.9 | 59.8 | 55.0 | 83.7 | 54.7 |
| LLaVA-MOD (Qwen1.5-0.5B) | 0.5B | - | 56.2 | 62.8 | 65.3 | 58.8 | - | 54.1 |
| **LLaVA-KD (Qwen1.5-0.5B)** | **0.5B** | **77.0** | **59.6** | **60.6** | **64.5** | **60.1** | **85.9** | **57.9** |
| TinyLLaVA (Qwen1.5-1.8B) | 1.8B | 73.1 | 55.5 | 65.3 | 61.2 | 57.1 | 83.4 | 56.8 |
| LLaVA-MOD (Qwen1.5-1.8B) | 1.8B | - | 58.7 | 68.0 | 66.7 | 66.3 | 87.0 | 59.9 |
| **LLaVA-KD (Qwen1.5-1.8B)** | **1.8B** | **79.0** | **62.3** | **64.7** | **69.1** | **64.0** | **86.3** | **62.1** |

LLaVA-KD outperforms baselines at both the 0.5B and 1.8B scales, achieving Avg₁₀ improvements of 3.2% and 5.3% respectively, using only 1.2M training samples (compared to 5M for LLaVA-MOD).

### Ablation Study: Effect of the Three-Stage Training Scheme

| Training Scheme | Avg₁₀ |
|----------------|-------|
| PT-SFT (baseline) | 54.7 |
| DPT-SFT | 55.6 (+0.9) |
| PT-DFT | 55.8 |
| DPT-DFT | 55.9 |
| PT-SFT-DFT | 56.6 |
| **DPT-SFT-DFT** | **57.9 (+3.2)** |
| DPT-DFT-DFT | 58.0 |

- DPT yields a +0.9% gain, confirming that distillation-based pre-training improves cross-modal alignment.
- DFT contributes the largest improvement (+2.3%), demonstrating effective teacher knowledge transfer.
- Skipping SFT (DPT-DFT) leads to performance degradation, confirming that SFT is indispensable for knowledge acquisition.
- DPT-DFT-DFT marginally outperforms but incurs greater computational overhead (120 GPU hours); DPT-SFT-DFT offers the best cost-performance trade-off.

### Ablation Study: Distillation Objectives

| Distillation Target | Response | Visual | Avg₁₀ |
|--------------------|----------|--------|-------|
| DPT: Response only | ✓ | ✗ | 54.9 |
| DPT: Response + Visual | ✓ | ✓ | 55.1 |
| DFT: Response only | ✓ | ✗ | 57.2 |
| DFT: Response + Visual | ✓ | ✓ | **57.7** |

Visual token distillation yields additional gains in both DPT and DFT stages, validating the importance of the visual distillation component in MDist.

## Highlights & Insights

1. **Novel multimodal distillation formulation**: This is the first work to extend distillation from response tokens to visual tokens within MLLMs, addressing the blind spot of prior LLM distillation methods that ignore the visual modality.
2. **Elegant relational distillation design**: Spatial and semantic inter-token relationships are captured via visual token self-correlation matrices, going beyond simple feature alignment.
3. **Well-motivated three-stage training**: Each stage has a clearly defined role—DPT for alignment, SFT for foundational capability, and DFT for knowledge refinement.
4. **Strong architecture-agnosticism**: The framework requires no architectural modifications and is directly applicable to various LLaVA-style MLLMs.

## Limitations & Future Work

- Teacher and student must share the same visual encoder, limiting distillation flexibility across architectures.
- Distillation increases training computation and memory overhead due to the need to run both teacher and student models simultaneously.
- Validation is limited to the LLaVA-1.5 architecture; applicability to more advanced architectures (e.g., dynamic resolution models) remains unknown.
- Loss weights ($\alpha, \beta, \gamma$) are fixed as empirical values without an adaptive adjustment mechanism.

## Related Work & Insights

- **Small-scale MLLMs**: TinyLLaVA, Bunny, MoE-LLaVA, MobileVLM, MiniCPM-V reduce costs via lightweight backbones or architectural optimization.
- **LLM Distillation**: MiniLLM (reverse KLD), DistiLLM (skewed KLD), and CoT distillation focus on the text modality.
- **MLLM Distillation**: LLaVA-MoD (output KLD + preference distillation + MoE); LLaVADI finds that most LLM distillation strategies provide no additional benefit for MLLMs.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Technical Quality | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Value | 4 |
| Overall | 4.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[CVPR 2025\] LLaVA-Critic: Learning to Evaluate Multimodal Models](../../CVPR2025/multimodal_vlm/llava-critic_learning_to_evaluate_multimodal_models.md)
- [\[ECCV 2024\] SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant](../../ECCV2024/multimodal_vlm/sq-llava_self-questioning_for_large_vision-language_assistant.md)
- [\[ACL 2025\] HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model](../../ACL2025/multimodal_vlm/hidellava_hierarchical_decoupling_for_continual_instruction.md)
- [\[ACL 2025\] AVG-LLaVA: An Efficient Large Multimodal Model with Adaptive Visual Granularity](../../ACL2025/multimodal_vlm/avg-llava_an_efficient_large_multimodal_model_with_adaptive_visual_granularity.md)

</div>

<!-- RELATED:END -->
