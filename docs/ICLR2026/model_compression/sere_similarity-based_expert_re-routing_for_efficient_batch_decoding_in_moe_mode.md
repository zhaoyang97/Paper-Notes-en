---
title: >-
  [Paper Note] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models
description: >-
  [Model Compression] SERE is proposed to pre-compute an expert similarity matrix and dynamically re-route secondary experts to their most similar primary experts during batch decoding, achieving up to 2.0× speedup with negligible quality loss, accompanied by a plug-and-play vLLM CUDA kernel.
tags:
  - Model Compression
date: 2026-05-08
content_hash: 52119f37b812c2e8
---

# SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models

## Basic Information

- **Conference**: ICLR 2026
- **arXiv**: [2602.07616](https://arxiv.org/abs/2602.07616)
- **Code**: [GitHub](https://github.com/JL-Cheng/SERE)
- **Area**: Model Compression / Efficient Inference
- **Keywords**: Mixture-of-Experts, Batch Decoding, Expert Skipping, CUDA Kernel, vLLM

## TL;DR

SERE is proposed to pre-compute an expert similarity matrix and dynamically re-route secondary experts to their most similar primary experts during batch decoding, achieving up to 2.0× speedup with negligible quality loss, accompanied by a plug-and-play vLLM CUDA kernel.

## Background & Motivation

### State of the Field
Mixture-of-Experts (MoE) architectures achieve efficient inference through sparse activation, where each token activates only a small subset of experts (e.g., top-8 out of 128 in Qwen3-30B-A3B). In practice, however, **batch inference** causes different tokens within a batch to require different experts, resulting in the number of actually activated experts far exceeding the per-token budget.

### Root Cause
- **Sparsity vs. Batching**: As batch size increases, more experts are activated (Figure 1), significantly increasing memory bandwidth overhead during decoding;
- **Load balancing objectives exacerbate the issue**: Such objectives distribute tokens more evenly across experts, increasing expert diversity within a batch.

### Limitations of Prior Work
- **Static compression** (pruning/merging): High computational cost, task-dependent, reduces model capacity and generalization;
- **Dynamic skipping** (threshold/Top-p routing): Relies solely on routing scores, ignores intrinsic expert properties, requires additional training or threshold tuning, and is difficult to integrate into high-performance inference frameworks.

## Method

### Core Observations

1. **High functional similarity**: Many experts within an MoE layer exhibit high functional similarity and can serve as substitutes for one another;
2. **Primary–secondary distinction**: Top-ranked primary experts dominate the output (with substantially larger weights and contributions), while secondary experts contribute marginally;
3. **Critical experts exist**: A small number of experts have very low similarity to all others and are irreplaceable specialized units.

### 1. Expert Similarity Matrix Computation

A calibration dataset $\mathcal{D}_{\text{calib}}$ is used to pre-compute the expert similarity matrix for each layer:

$$
\mathbf{S}_{p,q}^{(l)} = \frac{1}{N} \sum_{i=1}^{N} \text{Sim}(\mathbf{A}_{i,p}^{(l)}, \mathbf{A}_{i,q}^{(l)})
$$

where $\mathbf{A}_{i,j}^{(l)} = \mathbf{E}_j^{(l)}(\mathbf{X}_i^{(l-1)})$ denotes the activation output of the $j$-th expert in layer $l$. The similarity function may be cosine similarity, Frobenius norm, or CKA.

**Key Findings** (Figure 4, Qwen3-30B-A3B):
- Layer-1 exhibits the highest average similarity (nearly all pairs > 0.9);
- Layer-6 exhibits the lowest average similarity (most pairs < 0.4);
- Each layer contains "critical experts" with very low similarity to all other experts.

### 2. Similarity-based Dynamic Re-routing

**Step 1 — Primary Expert Selection**:

$$
\mathcal{E}_p^{(l)} = \bigcup_{\mathcal{T}} \{\mathbf{E}_{r_k}^{(l)} \mid 1 \leq k \leq S\}
$$

The Top-$S$ experts across all tokens are retained as the primary expert set ($S$ is a hyperparameter controlling the speedup ratio).

**Step 2 — Secondary Expert Re-routing**:

For each secondary expert $\mathbf{E}_u^{(l)}$, the most similar primary expert is identified:

$$
v_u^* = \arg\max_{\mathbf{E}_v^{(l)} \in \mathcal{E}_p^{(l)}} \mathbf{S}_{u,v}^{(l)}
$$

If $\text{sim}_u^* \geq \rho$ (the similarity threshold), all tokens assigned to the secondary expert are re-routed to $\mathbf{E}_{v_u^*}^{(l)}$.
If $\text{sim}_u^* < \rho$, the expert is identified as a **critical expert** and is retained unchanged.

**Step 3 — Final Execution**:

$$
\mathcal{E}_{\text{final}}^{(l)} = \mathcal{E}_p^{(l)} \cup \{\mathbf{E}_u^{(l)} \mid \text{sim}_u^* < \rho\}
$$

### 3. High-Performance CUDA Kernel Implementation

- Model-agnostic and compatible with various MoE architectures;
- Seamlessly integrated into vLLM, requiring only a **single line of code** to enable;
- No modification to the vLLM core execution pipeline is needed.

## Experiments

### Main Results: Accuracy and Speedup Comparison (Qwen1.5-MoE-A2.7B)

| Method | Exam Avg | Math Avg | Code Avg | Overall Avg | TPOT (ms) ↓ |
|--------|---------|---------|---------|-------------|------------|
| Top-4 (Original) | 61.67 | 42.28 | 38.17 | 48.52 | 17.29 |
| Top-2 (Naive) | 58.27 | 36.19 | 29.71 | 42.85 | 13.53 |
| HC-SMoE (40 experts) | 49.69 | 24.86 | 3.34 | 28.79 | 14.20 |
| LYNX top-2 | 48.26 | 24.51 | 7.97 | 29.28 | 14.49 |
| **SERE top2; ρ=0.0** | **60.48** | **40.87** | **36.58** | **47.15** | **13.83** |
| **SERE top2; ρ=0.3** | **61.02** | **41.55** | **35.14** | **47.25** | **13.93** |

### Qwen3-30B-A3B Results

| Method | Exam Avg | Math Avg | Code Avg | Overall Avg | Speedup |
|--------|---------|---------|---------|-------------|---------|
| Top-8 (Original) | — | — | — | Baseline | 1.0× |
| Top-K Reduction | — | — | — | Significant drop | 1.3× |
| LYNX | — | — | — | Large drop | 1.4× |
| **SERE (K=2)** | — | — | — | **Nearly lossless** | **1.5×** |
| **SERE (K=1)** | — | — | — | **Outperforms all baselines** | **2.0×** |

### Ablation Study: Impact of Key Designs

| Ablation | Overall Avg Change | Notes |
|----------|--------------------|-------|
| Remove critical expert protection (ρ=∞) | −1.8% | Critical experts are irreplaceable |
| No primary–secondary distinction (random skipping) | −5.2% | Distinction is essential |
| Static vs. dynamic similarity | Marginal difference | Pre-computed similarity is sufficiently reliable |
| Different similarity functions | Frobenius norm is best | Outperforms cosine and CKA |

### Key Findings

1. **SERE achieves 2.0× speedup with negligible quality loss**: SERE (K=2) incurs virtually no degradation across all tasks; SERE (K=1) still outperforms all baselines;
2. **Substantially surpasses prior methods**: HC-SMoE static pruning causes a 20% absolute accuracy drop; LYNX drops by 19%; SERE drops by only 1–3%;
3. **Critical expert protection is effective**: The similarity threshold $\rho$ automatically preserves irreplaceable experts;
4. **Input-aware dynamic adaptation**: Different batches activate different expert subsets; SERE adaptively skips experts with higher redundancy;
5. **Plug-and-play deployment**: CUDA kernel integration into vLLM requires only a single line of code.

## Highlights & Insights

- Leverages intrinsic expert similarity properties rather than routing scores alone to guide skipping decisions;
- Dynamic, input-aware strategy — more skipping when redundancy is high, less skipping when diversity demand is large;
- Automatic critical expert protection prevents capability degradation;
- The similarity matrix is pre-computed once, requiring no retraining or task-specific tuning;
- A production-grade CUDA kernel is provided, enabling one-line activation in vLLM.

## Limitations & Future Work

- Re-routing changes the token-to-expert mapping without modifying routing weights, potentially introducing minor output drift;
- The choice of calibration dataset may affect the representativeness of the similarity matrix;
- Hyperparameters $S$ and $\rho$ require speed–quality trade-off tuning and may need to be adjusted for different models;
- Validation is primarily conducted on the decoding phase; applicability to the prefill phase remains unexplored;
- When tokens within a batch are highly diverse (e.g., mixed multi-domain requests), the number of skippable redundant experts may decrease.

## Related Work & Insights

- **Static expert compression**: MoE-I2 (Yang et al., 2024), HC-SMoE (Chen et al., 2025), EEP (Liu et al., 2024c)
- **Dynamic expert skipping**: Top-p routing (Huang et al., 2024), AdaMoE (Zhong et al., 2024), LYNX (Gupta et al., 2024)
- **MoE inference optimization**: vLLM (Kwon et al., 2023), DeepSeekV2-Lite (Liu et al., 2024b)
- **MoE architectures**: Qwen-MoE (Bai et al., 2023), Qwen3-30B-A3B (Yang et al., 2025a)

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The re-routing approach based on expert similarity is clear and effective
- **Technical Depth**: ⭐⭐⭐⭐ — A complete engineering pipeline from observation to method to CUDA kernel
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three MoE models × multiple benchmarks × latency measurements × comprehensive ablations
- **Practical Value**: ⭐⭐⭐⭐⭐ — Plug-and-play vLLM integration, directly applicable to production deployment

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[AAAI 2026\] StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion](../../AAAI2026/model_compression/stepfun-formalizer_unlocking_the_autoformalization_potential_of_llms_through_kno.md)

<!-- RELATED:END -->
