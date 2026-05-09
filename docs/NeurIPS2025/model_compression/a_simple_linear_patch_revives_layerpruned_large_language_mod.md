---
title: >-
  [Paper Note] A Simple Linear Patch Revives Layer-Pruned Large Language Models
description: >-
  [NeurIPS 2025][Model Compression][Layer Pruning] LinearPatch inserts a lightweight symmetric matrix — fusing a Hadamard transform with channel scaling — at the pruning interface to repair activation magnitude mismatches caused by layer pruning. On LLaMA-3-8B, it retains 94.15% of the original performance without any training, and reaches 95.16% after 30 minutes of distillation.
tags:
  - NeurIPS 2025
  - Model Compression
  - Layer Pruning
  - Activation Magnitude Alignment
  - Hadamard Transform
  - Channel Scaling
  - Knowledge Distillation
date: 2026-05-08
content_hash: 141c461f93caaece
---

# A Simple Linear Patch Revives Layer-Pruned Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.24680](https://arxiv.org/abs/2505.24680)
**Code**: [https://github.com/chenxinrui-tsinghua/LinearPatch](https://github.com/chenxinrui-tsinghua/LinearPatch)
**Area**: Model Compression
**Keywords**: Layer Pruning, Activation Magnitude Alignment, Hadamard Transform, Channel Scaling, Knowledge Distillation

## TL;DR

LinearPatch inserts a lightweight symmetric matrix — fusing a Hadamard transform with channel scaling — at the pruning interface to repair activation magnitude mismatches caused by layer pruning. On LLaMA-3-8B, it retains 94.15% of the original performance without any training, and reaches 95.16% after 30 minutes of distillation.

## Background & Motivation

**State of the Field**: Layer pruning is a straightforward approach to compressing large language models — it directly removes redundant Transformer layers without requiring specialized hardware support or low-level kernel modifications, making deployment easier than unstructured pruning or N:M sparsity. Methods such as ShortGPT, SLEB, and LLM-Streamline have proposed various layer selection strategies based on cosine similarity, perplexity, Taylor scores, and others.

**Limitations of Prior Work**: Despite its simplicity, layer pruning typically leads to severe performance degradation after removing even a few layers. Existing work focuses primarily on which layers can be safely removed, while overlooking a more fundamental question: what happens to the activation distributions between the remaining layers after pruning.

**Root Cause**: The authors find that performance degradation is caused not by information loss per se, but by **activation magnitude mismatch at the pruning interface**. Hidden states across different layers in LLMs exhibit substantially different per-channel magnitudes, and directly concatenating non-adjacent layers induces distribution shift. Compounding this, special tokens (e.g., [BOS], separators) carry outliers on the order of $10^3$, making it impossible for a simple channel scaling to accommodate all tokens simultaneously.

**Paper Goals**: (1) How to align channel magnitudes on both sides of the pruning interface? (2) How to handle the token-wise scaling inconsistency caused by large outliers in special tokens?

**Starting Point**: The authors observe that the Hadamard transform can redistribute outliers concentrated in a few tokens across all channels, enabling a single shared set of channel scaling factors. Fusing the Hadamard transform and channel scaling into a single matrix multiplication incurs negligible overhead.

**Core Idea**: Insert a symmetric matrix $P = HDH^\top$ at the pruning interface (where $H$ is the Hadamard matrix and $D$ is a diagonal scaling matrix), suppressing outliers and aligning channel magnitudes in a single GEMM operation.

## Method

### Overall Architecture

The input is an LLM for which the layers to be removed have already been identified via metrics such as cosine similarity. At the pruning interface — between the last layer before the removed block and the first layer after it — a matrix $P \in \mathbb{R}^{C \times C}$ is inserted, transforming the output $X^{(\ell^*)}$ of the preceding layer into $X^{(\ell^*)} P$ before feeding it into the subsequent layers. The method proceeds in two steps: a training-free initialization of $P$, followed by an optional distillation fine-tuning stage using 5K samples.

### Key Designs

1. **Channel Magnitude Alignment**:

    - Function: Eliminate per-channel magnitude discrepancies between hidden states before and after the pruning interface.
    - Mechanism: On a calibration set, compute the magnitude ratio for each channel $k$ as $d_k = \|X_{:,k}^{(\ell^*+n)}\|_1 / \|X_{:,k}^{(\ell^*)}\|_1$, yielding a scaling vector $d \in \mathbb{R}^C$ that is formed into a diagonal matrix for per-channel scaling.
    - Design Motivation: Experiments show that $\alpha = 1$ (i.e., using the exact statistical magnitude ratio) achieves optimal perplexity, and deviating from this value leads to substantial degradation.

2. **Token Magnitude Smoothing via Hadamard**:

    - Function: Address the problem that a single channel scaling cannot simultaneously accommodate all tokens due to the large outliers present in special tokens.
    - Mechanism: The orthogonality of the Walsh–Hadamard matrix is exploited to rotate activations via $XH$, redistributing outliers concentrated in a few tokens across all channels, thereby reducing the standard deviation of per-token scaling (from 2137.75 to 230.32).
    - Design Motivation: Performing channel scaling in the Hadamard-rotated space and then rotating back yields substantially better results than scaling directly in the original space.

3. **LinearPatch Fusion**:

    - Function: Fuse the Hadamard transform and channel scaling into a single matrix multiplication.
    - Mechanism: By the spectral theorem, $P = HDH^\top$ is a real symmetric matrix, reducing three GEMMs to one.
    - Design Motivation: Inference overhead is negligible, and the structured form of $P$ facilitates subsequent fine-tuning.

### Loss & Training

An optional knowledge distillation stage stores the teacher model's top-100 logit probability distributions offline (achieving 320× memory savings), freezes all model parameters, and fine-tunes only the $P$ matrix by minimizing KL divergence. Distillation for a 7B model requires only 5K samples and 30 minutes on a single V100.

## Key Experimental Results

### Main Results

| Model | Pruned / Total Layers | Method | QA Avg. | Retained Performance (RP) |
|---|---|---|---|---|
| LLaMA-2-7B | 7/32 | ShortGPT | 60.59 | 86.06% |
| LLaMA-2-7B | 7/32 | LLM-Streamline | 60.59 | 86.06% |
| LLaMA-2-7B | 7/32 | **LinearPatch** | **62.42** | **88.88%** |
| LLaMA-3-8B | 5/32 | LLM-Streamline | 63.06 | 90.84% |
| LLaMA-3-8B | 5/32 | **LinearPatch** | **65.42** | **94.15%** |
| LLaMA-3-8B | 5/32 | **LinearPatch + Distillation** | **66.13** | **95.16%** |

### Ablation Study

| Configuration | QA Avg. | Notes |
|---|---|---|
| No patch (direct pruning) | 56.52 | Baseline |
| Channel scaling only | 58.31 | +1.79; channel alignment is effective |
| Hadamard only | 57.68 | +1.16; outlier suppression is effective |
| LinearPatch (both fused) | 59.14 | +2.62; complementary effects are significant |
| LinearPatch + Distillation | 62.42 | Distillation yields further improvement |

### Key Findings
- Channel scaling and the Hadamard transform are complementary; neither alone is sufficient.
- LinearPatch generalizes across different pruning metrics (cosine similarity, Taylor score, perplexity).
- KL divergence loss outperforms MSE loss during distillation; the latter is prone to overfitting.
- 128 calibration samples suffice to initialize the scaling factors; the method is insensitive to calibration set size.

## Highlights & Insights
- **Minimal yet effective**: The entire method reduces to a single matrix multiplication with negligible inference overhead yet substantial performance gains. The idea of using a linear transformation to repair distribution shift is transferable to other pruning and quantization scenarios.
- **Applying Hadamard transforms to layer pruning**: Hadamard-based outlier handling has previously been confined to quantization methods (QuaRot, FlatQuant); this work is the first to introduce it into the layer pruning setting, demonstrating that the outlier problem is a common challenge across LLM compression paradigms.
- **Offline distillation strategy**: Storing only top-$K$ logits enables distillation without loading teacher and student simultaneously, making the approach memory-friendly.

## Limitations & Future Work
- Experiments are limited to models in the 7B–13B range; effectiveness on larger models (70B+) remains unverified.
- The pruning ratio is constrained to within 30%; whether the patch remains effective under more aggressive pruning (>50%) is unclear.
- Only contiguous layer pruning is considered; the placement of multiple patches for non-contiguous pruning (multiple pruning interfaces) is not discussed.
- Future work could explore combining LinearPatch with quantization — applying layer pruning with the patch first and then quantizing — potentially achieving higher overall compression ratios.

## Related Work & Insights
- **vs. ShortGPT / LLM-Streamline**: These methods focus on which layers to prune; LinearPatch focuses on how to repair the model after pruning. The two approaches are orthogonal and complementary.
- **vs. LoRA-based recovery**: Shortened LLaMA recovers performance via LoRA fine-tuning, requiring greater training resources; the training-free variant of LinearPatch already surpasses LoRA-based approaches.
- **vs. QuaRot / FlatQuant**: These methods apply the Hadamard transform to handle outliers in quantization; LinearPatch applies the same idea to address activation mismatch in layer pruning.

## Rating
- Novelty: ⭐⭐⭐⭐ Identifies and quantifies the previously overlooked activation magnitude mismatch problem; the proposed solution is elegant and concise.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple models, metrics, and ablations, though large-scale model experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear, visualizations are excellent, and the logical flow is coherent.
- Value: ⭐⭐⭐⭐ A plug-and-play practical technique that makes a substantive contribution to the layer pruning field.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Restoring Pruned Large Language Models via Lost Component Compensation](restoring_pruned_large_language_models_via_lost_component_compensation.md)
- [\[NeurIPS 2025\] The Structure of Relation Decoding Linear Operators in Large Language Models](the_structure_of_relation_decoding_linear_operators_in_large_language_models.md)
- [\[NeurIPS 2025\] LayerIF: Estimating Layer Quality for Large Language Models using Influence Functions](layerif_estimating_layer_quality_for_large_language_models_using_influence_funct.md)
- [\[NeurIPS 2025\] Correlation Dimension of Auto-Regressive Large Language Models](correlation_dimension_of_auto-regressive_large_language_models.md)
- [\[NeurIPS 2025\] PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](permllm_learnable_channel_permutation_for_nm_sparse_large_language_models.md)

<!-- RELATED:END -->
