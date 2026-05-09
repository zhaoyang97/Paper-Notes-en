---
title: >-
  [Paper Note] GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning
description: >-
  [NEURIPS2025][Image Generation][LoRA] This paper proposes GraLoRA, which partitions the LoRA weight update matrix into $k^2$ independent sub-blocks, each equipped with its own low-rank adapter pair. Without increasing parameter count or computational cost, GraLoRA elevates the effective rank from $r$ to $kr$, addressing the performance degradation caused by gradient entanglement in high-rank LoRA. On code generation, Pass@1 improves by up to +8.5%.
tags:
  - NEURIPS2025
  - Image Generation
  - LoRA
  - low-rank adaptation
  - parameter-efficient fine-tuning
  - gradient entanglement
  - block decomposition
  - high-rank expressiveness
date: 2026-05-08
content_hash: ed3e8e6caffcc890
---

# GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning

**Conference**: NEURIPS2025
**arXiv**: [2505.20355](https://arxiv.org/abs/2505.20355)
**Authors**: Yeonjoon Jung, Daehyun Ahn, Hyungjun Kim, Taesu Kim, Eunhyeok Park (SqueezeBits, POSTECH)
**Code**: Not released
**Area**: Parameter-Efficient Fine-Tuning (PEFT) / LoRA Improvements
**Keywords**: LoRA, low-rank adaptation, parameter-efficient fine-tuning, gradient entanglement, block decomposition, high-rank expressiveness

## TL;DR

This paper proposes GraLoRA, which partitions the LoRA weight update matrix into $k^2$ independent sub-blocks, each equipped with its own low-rank adapter pair. Without increasing parameter count or computational cost, GraLoRA elevates the effective rank from $r$ to $kr$, addressing the performance degradation caused by gradient entanglement in high-rank LoRA. On code generation, Pass@1 improves by up to +8.5%.

## Background & Motivation

- **Core Limitation**: LoRA performs best at rank 32–64; further increasing the rank leads to performance degradation, sometimes even underperforming lower-rank settings — contradicting the intuition that more parameters should yield better results.
- **Limitations of Prior Work**: OLoRA and PiSSA improve initialization; MoRA and RaSA alter the structure; however, none fundamentally addresses the root cause of high-rank degradation.
- **Key Observation**: The authors find that the Layer 1 down-projection in LLaMA3.1-8B exhibits severe channel activation imbalance (outlier channels), where the gradients of a few anomalous channels dominate the update direction of the entire low-rank adapter.

## Core Problem: Gradient Entanglement in LoRA

### Gradient Propagation: FFT vs. LoRA

In full fine-tuning (FFT), the influence of outlier channels is local — it only affects the column of weight matrix $W$ that directly interacts with the outlier channel. However, LoRA's low-rank constraint leads to:

$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial Y} X^\top A$$

The matrix $A$ mixes information from all input channels, so the large activations of outlier channels contaminate the entire $\partial L / \partial B$, inappropriately biasing the gradients even for channels unrelated to the outlier.

### High Rank Exacerbates the Problem

As rank increases, $A \in \mathbb{R}^{N \times r}$ mixes more channel information, leading to more severe gradient distortion. Experiments confirm that the gradient deviation between LoRA at rank 128 and FFT is significantly larger than at rank 32.

## Method

### 1. Block-wise Low-Rank Decomposition

The weight update matrix $R \in \mathbb{R}^{M \times N}$ is partitioned into a $k \times k$ grid, with each sub-block $(i,j)$ assigned an independent low-rank adapter pair:

$$R_{\text{GraLoRA}} = \begin{bmatrix} B_{1,1}A_{1,1}^\top & \cdots & B_{1,k}A_{1,k}^\top \\ \vdots & \ddots & \vdots \\ B_{k,1}A_{k,1}^\top & \cdots & B_{k,k}A_{k,k}^\top \end{bmatrix}$$

where $A_{i,j} \in \mathbb{R}^{N/k \times r/k}$ and $B_{i,j} \in \mathbb{R}^{M/k \times r/k}$.

### 2. Expressiveness Analysis

Rewriting GraLoRA as a sparse matrix product $R = B_{\text{GraLoRA}} A_{\text{GraLoRA}}^\top$ and applying the Sylvester rank inequality yields:

- LoRA effective rank $= r$
- GraLoRA effective rank $= kr$ (a $k$-fold increase)

This means GraLoRA can express higher-rank weight updates using the same number of parameters.

### 3. Gradient Localization

GraLoRA naturally confines the influence of outlier channels to the $k$ adapter blocks that interact with them, leaving the remaining $k^2 - k$ blocks unaffected. This gradient propagation pattern closely resembles FFT behavior, effectively preventing global gradient distortion.

### 4. Overhead Analysis

| Dimension | Compared to LoRA |
|-----------|-----------------|
| **Parameters** | Identical: $N \times r + M \times r$ |
| **FLOPs** | GraLoRA is actually lower: $\text{GraLoRA} = \text{LoRA} - (k-1)rT$ |
| **Training Memory** | Intermediate activations $A^\top X$ scale by $k$, but since $r \ll M, N$, the impact is negligible; virtually no difference with gradient checkpointing |
| **Inference** | Merges back into original weights with zero additional overhead |

### 5. Selection Strategy for $k$

Empirical rule: maintain minimum sub-block expressiveness $r/k^2 \approx 8$

- rank 16/32 → $k = 2$
- rank 64/128 → $k = 4$

### 6. Hybrid GraLoRA

In low-rank settings (rank ≤ 16), the per-block rank of pure GraLoRA becomes too small to be expressive. The proposed solution allocates a portion of the rank to standard LoRA, concatenated with the GraLoRA component. Empirically, assigning no more than 1/2 of the rank to the LoRA component yields the best results.

## Key Experimental Results

### Code Generation (HumanEval+, LLaMA3.1-8B)

| Rank | Method | Pass@1 | Pass@5 | Pass@10 |
|------|--------|--------|--------|---------|
| 64 | LoRA | 58.1% | 66.4% | 68.5% |
| 64 | **GraLoRA** | **60.5%** | **71.2%** | **72.6%** |
| 128 | LoRA | 55.8% | 64.8% | 68.6% |
| 128 | **GraLoRA** | **64.3%** | **71.7%** | **73.7%** |

At rank 128, LoRA suffers severe degradation (underperforming rank 32), while GraLoRA continues to improve, with a gap of **+8.5% Pass@1**.

### Commonsense Reasoning (Average over 8 Tasks)

| Model | LoRA | GraLoRA | Gain |
|-------|------|---------|------|
| Qwen2.5-1.5B | 78.7% | **79.8%** | +1.1% |
| Qwen2.5-7B | 85.6% | **86.4%** | +0.8% |
| LLaMA3.2-3B | 81.3% | **84.6%** | +3.3% |
| LLaMA3.1-70B | 91.3% | **92.4%** | +1.1% |

GraLoRA wins 26 out of 32 sub-tasks.

### Mathematical Reasoning (MATH, Qwen2.5-1.5B)

| Rank | LoRA | GraLoRA |
|------|------|---------|
| 64 | 23.6% | **25.7%** |
| 128 | 24.7% | **28.9%** (+4.2%) |

### GLUE (RoBERTa-base)

Best GraLoRA achieves an average of 86.0%, surpassing LoRA (84.2%), VeRA (85.2%), and FourierFT (85.0%).

### Image Generation (SDXL Fine-Tuning)

| Metric | LoRA | GraLoRA |
|--------|------|---------|
| CLIP Similarity | 91.4% | **91.9%** |
| DINOv2 Similarity | 79.2% | **81.3%** |

## Highlights & Insights

1. **Precise Problem Identification**: The paper offers a clear gradient-dynamics-based explanation for high-rank LoRA degradation — gradient entanglement caused by outlier channels — a perspective that has been largely overlooked.
2. **Elegant Simplicity**: The method merely partitions the matrix into blocks without changing parameter count, computation, or inference pipeline, yet achieves a $k$-fold increase in effective rank — a "one-line-of-code" improvement.
3. **Theory–Experiment Consistency**: The rank improvement is rigorously proven via the Sylvester rank inequality, and gradient distribution experiments visually demonstrate GraLoRA's gradient correction effect.
4. **Broad Coverage**: Experiments span 5 tasks (code generation / commonsense reasoning / math / NLU / image generation) × multiple model architectures (LLaMA / Qwen / RoBERTa / SDXL) × multiple ranks, constituting a thorough evaluation.
5. **Minimal Training Overhead**: Training time increases by only 3%–10% relative to LoRA, far less than MoRA's 40%+.

## Limitations & Future Work

1. **Uniform Partitioning Assumption**: The current design assumes equal-sized sub-blocks without accounting for varying channel importance; adaptive partitioning may yield further gains.
2. **Manual Selection of $k$**: Although an empirical rule ($r/k^2 \approx 8$) is provided, the optimal $k$ for different tasks and models still requires sweeping.
3. **Misclassification**: The paper is categorized under image generation, while its core contribution is a general-purpose PEFT method; the image generation experiment serves merely as one validation.
4. **Incomplete Comparison with Recent Methods**: Comparisons with DoRA and LoRA+ are available for commonsense reasoning but absent for code generation and mathematical reasoning.
5. **No Exploration of Adaptive Rank Allocation**: Different layers may benefit from different values of $k$; the paper uses a uniform $k$ across all layers.

## Related Work & Insights

| Method | Mechanism | vs. GraLoRA |
|--------|-----------|-------------|
| **LoRA** | Fixed low-rank decomposition | High-rank degradation due to gradient entanglement |
| **MoRA** | Replaces low-rank matrices with square matrices | Training time increases by 40%+; unstable performance |
| **RaSA** | Shares partial low-rank components across layers | Underperforms GraLoRA on code generation; similarly limited at high rank |
| **DoRA** | Decouples direction and magnitude updates | GraLoRA leads by 2.9% on commonsense reasoning |
| **VeRA** | Extreme parameter compression via shared random matrices | GraLoRA leads by 0.8% on GLUE |
| **PiSSA** | SVD-based LoRA initialization | Does not address the root cause of high-rank degradation |

### Broader Connections

1. **Connection to Quantization**: Outlier channel problems have been extensively studied in quantization (e.g., SmoothQuant, OWQ), but this paper is the first to systematically analyze their impact on LoRA training, suggesting that PEFT and quantization can share outlier handling strategies.
2. **Generality of Block-wise Thinking**: The paradigm of decomposing large matrices into independently processed sub-blocks is broadly applicable to other low-rank methods (e.g., low-rank attention, low-rank MoE) as a general expressiveness-enhancement strategy.
3. **Practical Implication**: Many practitioners observe that increasing the LoRA rank leads to performance drops without understanding the cause; GraLoRA provides both a theoretical explanation and a straightforward remedy.
4. **Merge-compatible Inference**: Zero additional inference overhead makes GraLoRA highly practical for deployment.

## Rating

- Novelty: ⭐⭐⭐⭐ (Novel problem analysis; the method itself is a precise application of existing ideas)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 tasks, multiple models, multiple ranks, comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear analysis, rich figures and tables)
- Value: ⭐⭐⭐⭐⭐ (Ready to use, minimal modification, clear benefit)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)
- [\[ICCV 2025\] Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models](../../ICCV2025/image_generation/transformed_low-rank_adaptation_via_tensor_decomposition_and_its_applications_to.md)
- [\[NeurIPS 2025\] DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)
- [\[NeurIPS 2025\] Towards Resilient Safety-Driven Unlearning for Diffusion Models Against Downstream Fine-tuning](towards_resilient_safety-driven_unlearning_for_diffusion_models_against_downstre.md)
- [\[ICCV 2025\] ShortFT: Diffusion Model Alignment via Shortcut-based Fine-Tuning](../../ICCV2025/image_generation/shortft_diffusion_model_alignment_via_shortcut-based_fine-tuning.md)

<!-- RELATED:END -->
