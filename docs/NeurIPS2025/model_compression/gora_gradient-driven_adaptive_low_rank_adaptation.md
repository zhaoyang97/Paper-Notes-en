---
title: >-
  [Paper Note] GoRA: Gradient-Driven Adaptive Low Rank Adaptation
description: >-
  [NeurIPS 2025][Model Compression][LoRA] GoRA is proposed to leverage **pre-computed gradient information** to simultaneously perform adaptive rank allocation and weight initialization prior to training — assigning per-layer ranks based on parameter sensitivity and initializing the $B$ matrix via the gradient pseudo-inverse so that the initial output approximates one step of gradient descent, thereby addressing both major bottlenecks of LoRA in a unified framework.
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "LoRA"
  - "adaptive rank allocation"
  - "gradient-driven initialization"
  - "parameter-efficient fine-tuning"
  - "LLM"
date: 2026-05-08
content_hash: daa4ed6f18f9c218
---

# GoRA: Gradient-Driven Adaptive Low Rank Adaptation

**Conference**: NeurIPS 2025
**arXiv**: [2502.12171](https://arxiv.org/abs/2502.12171)  
**Code**: [GitHub](https://github.com/hhnqqq/MyTransformers)  
**Area**: Model Compression / LLM Efficiency
**Keywords**: LoRA, adaptive rank allocation, gradient-driven initialization, parameter-efficient fine-tuning, LLM

## TL;DR

GoRA is proposed to leverage **pre-computed gradient information** to simultaneously perform adaptive rank allocation and weight initialization prior to training — assigning per-layer ranks based on parameter sensitivity and initializing the $B$ matrix via the gradient pseudo-inverse so that the initial output approximates one step of gradient descent, thereby addressing both major bottlenecks of LoRA in a unified framework.

## Background & Motivation

LoRA performance is constrained by two critical factors: **rank selection** and **weight initialization**.

**Rank allocation problem**:
- Higher rank consistently yields better performance, but directly increasing rank incurs large memory overhead
- AdaLoRA dynamically adjusts ranks during training via masking, but requires pre-allocating larger matrices (1.5× parameter count), limiting the rank upper bound
- Methods such as MeLoRA modify the LoRA structure at the cost of generality

**Initialization problem**:
- PiSSA/MiLoRA initialize via SVD decomposition of pretrained weights — task-agnostic and limited in generalization
- LoRA-GA initializes using singular features of gradients — but requires modifying the pretrained weights as $W_0 \leftarrow W_0 - A_0 B_0$, introducing a training–inference gap
- All non-zero initialization methods require saving the modified pretrained weights, sacrificing LoRA's storage advantage

**Core insight**: LoRA adapters are reinterpreted as **gradient compressors**. Experiments from LoRA-FA show that freezing random $A$ and training only $B$ can approximate full LoRA performance, where $\Delta W = \frac{\alpha}{r} A_0 \Delta B_t = -\eta \frac{\alpha}{r} \sum_t A_0 A_0^T \frac{\partial \mathcal{L}}{\partial W_0}$, i.e., $A_0$ compresses the gradient and $A_0$ decompresses it.

## Method

### Overall Architecture

GoRA proceeds in two steps, both completed before formal training begins:

1. **Gradient pre-computation**: Forward passes over $N$ batches to accumulate the average gradient of each layer's weights, $G = \frac{1}{N} \sum_i \frac{\partial \mathcal{L}_i}{\partial W_0}$
2. **Adaptive rank allocation**: Determine the per-layer rank based on an importance score derived from weight–gradient interaction
3. **Gradient-driven initialization**: Initialize $B_0$ via the gradient pseudo-inverse such that $A_0 B_0 \approx -G$

### Key Designs

**Adaptive rank allocation strategy**:

1. Compute the per-layer importance (parameter sensitivity metric):
$$I(W) = \text{avg}(|W \odot G|)$$
i.e., the element-wise absolute mean of the Hadamard product of weights and gradients — intuitively, layers with large weights and large gradients are more important.

2. Normalize to advantage scores: $a^i = I(W_0^i) / \sum_j I(W_0^j)$

3. Compute the total parameter budget based on a reference rank $r^{\text{ref}}$: $b = \sum_i \sqrt{m_i + n_i} \times r^{\text{ref}}$

4. Allocate the rank for each layer:
$$r^i = \text{clip}\left(\text{round}\left(\frac{b \cdot a^i}{\sqrt{m+n}}\right), r^{\min}, r^{\max}\right)$$

Design objectives: (1) completed once before training with no dynamic shape changes; (2) total parameter count comparable to standard LoRA (±10%); (3) structurally compatible with LoRA.

**Gradient-driven initialization**:

$A_0$ is initialized with Kaiming uniform distribution (consistent with the PEFT library); $B_0$ is initialized via the pseudo-inverse of the gradient:

$$B_0 = -(A_0^T A_0)^{-1} A_0^T G$$

This makes $A_0 B_0 = -A_0(A_0^T A_0)^{-1} A_0^T G$ the optimal low-rank approximation of $G$ in the column space of $A_0$ (minimizing $\|A_0 B_0 + G\|_F$).

**Scaling factor $\xi$**: To match the true gradient magnitude, a scaling factor is introduced:
$$\frac{\alpha}{\sqrt{r}} A_0(\xi B_0) \approx -\gamma G$$
where $\xi = \gamma \sqrt{m} / \alpha$ and $\gamma$ is a tunable step-size hyperparameter (recommended: 5e-2).

**No modification to pretrained weights**: Unlike PiSSA/LoRA-GA, GoRA does not require $W_0 \leftarrow W_0 - A_0 B_0$, because the initialization objective is to make $A_0 B_0 \approx -G$ (approximating one step of gradient descent) rather than decomposing $W_0$.

### Loss & Training

- Forward computation: $W_t = W_0 + \frac{\alpha}{\sqrt{r}} A_t B_t$ (adopting the $\sqrt{r}$ scaling from rsLoRA)
- Standard fine-tuning loss (next-token prediction or classification loss)
- The formal training procedure is identical to LoRA; GoRA's innovations are entirely within the initialization phase

## Key Experimental Results

### Main Results

**T5-Base on GLUE** ($r^{\text{ref}}=8$):

| Method | MNLI | SST-2 | CoLA | QNLI | MRPC | Avg. |
|--------|------|-------|------|------|------|------|
| Full FT | 86.33 | 94.75 | 80.70 | 93.19 | 84.56 | 87.91 |
| LoRA | 85.30 | 94.04 | 69.35 | 92.96 | 68.38 | 82.08 |
| LoRA-GA | 85.70 | 94.11 | **80.57** | 93.18 | 85.29 | 87.77 |
| AdaLoRA | 85.45 | 93.69 | 69.16 | 91.66 | 68.14 | 81.62 |
| **GoRA** | **85.91** | **94.68** | 79.86 | **93.27** | **86.10** | **87.96** |

**Llama-3.1-8B-Base on generation tasks**:

| Method | MTBench | GSM8k | HumanEval |
|--------|---------|-------|-----------|
| Full FT | 5.88 | 73.69 | 51.63 |
| LoRA | 6.15 | 67.78 | 43.09 |
| LoRA-GA | 5.99 | 71.39 | 43.29 |
| **GoRA** | 6.34 | **72.91** | **48.98** |
| GoRA ($r^{\text{ref}}=128$) | 5.82 | **75.74** | **52.03** |

GoRA ($r^{\text{ref}}=128$) **surpasses full fine-tuning** on GSM8k and HumanEval.

### Ablation Study

**Effect of rank allocation range** (Llama-3.1-8B, $\gamma=5e{-2}$):

| $r^{\min}$ | $r^{\max}$ | GSM8k | HumanEval |
|-----------|-----------|-------|-----------|
| 8 | 8 (fixed) | 72.10 | 44.75 |
| 6 | 15 | 72.25 | 45.85 |
| 4 | 32 | **72.88** | **48.98** |

**Effect of initialization scaling factor $\gamma$**:

| $\gamma$ | GSM8k | HumanEval |
|---------|-------|-----------|
| 0 (no initialization) | 72.45 | 46.34 |
| 3e-2 | 72.71 | 45.93 |
| 5e-2 | 72.88 | **48.98** |
| 8e-2 | **72.91** | 46.54 |

**Comparison of importance metrics**:

| Metric | GSM8k | HumanEval |
|--------|-------|-----------|
| $\text{avg}(\|W \odot G\|)$ (GoRA) | **72.88** | **48.98** |
| $\|G\|_*$ (nuclear norm) | 72.70 | 43.09 |
| $\|W \odot G\|_*$ | 72.65 | 45.12 |

### Key Findings

1. **Wider rank ranges are better**: $(r^{\min}=4, r^{\max}=32)$ significantly outperforms fixed rank=8; most rank budget is allocated to $W_v$ layers
2. **Initialization is critical**: $\gamma=0$ (no initialization) underperforms the optimal $\gamma$ by 2.64 points on HumanEval
3. **Parameter sensitivity metric is best**: outperforms gradient nuclear norm and the nuclear norm of the weight–gradient product
4. **Cross-modal consistency**: GoRA outperforms baselines on NLU (T5), NLG (Llama), and visual classification (CLIP-ViT)
5. **Surpasses full fine-tuning at high rank**: GoRA $r^{\text{ref}}=128$ outperforms full fine-tuning by 2.05 points on mathematical reasoning

## Highlights & Insights

- **LoRA as a gradient compressor**: This reinterpretation unifies the design logic for both rank allocation and initialization
- **One-time pre-training setup**: No runtime dynamic adjustment is required, making it fully compatible with distributed training (FSDP/ZeRO)
- **No training–inference gap**: Pretrained weights are not modified, preserving LoRA's storage advantage
- **Automatic hyperparameter tuning**: Adaptive gradient accumulation stopping and adaptive $\gamma$ search strategies are proposed, approaching the performance of manual tuning
- **Rank distribution pattern**: $W_v$ receives the most rank and $W_q$ the least, consistent with observations in the original LoRA paper

## Limitations & Future Work

1. Gradient pre-computation requires additional forward pass overhead ($N$ batches), which is non-negligible for very large models
2. The optimal value of $\gamma$ varies by task (GSM8k favors 8e-2, HumanEval favors 5e-2)
3. $r^{\min}$ and $r^{\max}$ still require manual selection, although the paper shows that wide ranges are generally preferable
4. The pseudo-inverse computation assumes $A_0$ is full rank, which may be numerically unstable at very low ranks
5. Combination with QLoRA (quantization) has not been explored

## Related Work & Insights

- **LoRA-GA**: Also gradient-driven initialization, but requires modifying pretrained weights; GoRA avoids this via the scaling factor
- **AdaLoRA**: Also an adaptive rank method, but through training-time masking, inflating parameter count by 1.5×; GoRA's pre-training allocation incurs no additional overhead
- **rsLoRA**: The $\alpha/\sqrt{r}$ scaling rule is adopted by GoRA to better leverage high ranks
- **Inspiration**: The gradient compressor perspective may generalize to other PEFT methods (Adapter, Prefix-tuning), warranting exploration of a gradient-driven unified framework

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The view of LoRA as a gradient compressor is original and unifies two previously separate problems
- **Technical Depth**: ⭐⭐⭐⭐ — Theoretical analysis is clear; the optimality of the pseudo-inverse is formally proven
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three modalities (NLU/NLG/vision), multiple models and benchmarks, comprehensive ablations
- **Practicality**: ⭐⭐⭐⭐⭐ — Plug-and-play, compatible with the LoRA ecosystem, code is open-sourced
- **Overall**: ⭐⭐⭐⭐

## Background & Motivation

## Core Problem

## Method

## Key Experimental Results

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Inspiration & Connections

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[ACL 2025\] CoLA: Collaborative Low-Rank Adaptation](../../ACL2025/model_compression/cola_collaborative_low-rank_adaptation.md)

</div>

<!-- RELATED:END -->
