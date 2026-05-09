---
title: >-
  [Paper Note] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs
description: >-
  [NeurIPS 2025][Model Compression][Dual Sparsity] This paper proposes DuoGPT, a dual-sparse framework that reinterprets activation sparsity as dynamic structured weight sparsity and combines it with unstructured weight pruning. By extending the OBC framework with activation-aware calibration and a dense-model output residual correction term, DuoGPT achieves significant speedup and memory savings during the LLM decoding phase without any retraining.
tags:
  - NeurIPS 2025
  - Model Compression
  - Dual Sparsity
  - Activation Sparsity
  - Unstructured Pruning
  - OBC Framework
  - LLM Acceleration
date: 2026-05-08
content_hash: 7b0ef7a5da2e6720
---

# DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2506.20194](https://arxiv.org/abs/2506.20194)
**Code**: [GitHub](https://github.com/) (mentioned in the paper)
**Area**: LLM Inference Efficiency / Model Compression
**Keywords**: Dual Sparsity, Activation Sparsity, Unstructured Pruning, OBC Framework, LLM Acceleration

## TL;DR
This paper proposes DuoGPT, a dual-sparse framework that reinterprets activation sparsity as dynamic structured weight sparsity and combines it with unstructured weight pruning. By extending the OBC framework with activation-aware calibration and a dense-model output residual correction term, DuoGPT achieves significant speedup and memory savings during the LLM decoding phase without any retraining.

## Background & Motivation

1. **High Deployment Cost**: Decoding a single token with LLaMA-2-70B requires approximately 150 GB of GPU/CPU memory and 60 GFLOPs, posing severe storage and computational challenges for practical deployment.

2. **Inherent Trade-off between Structured and Unstructured Pruning**: Structured pruning yields good speedup but suffers from significant accuracy degradation (e.g., ShortGPT achieves only 51.35% accuracy at 31.25% sparsity), while unstructured pruning preserves accuracy but offers limited acceleration — making it historically difficult to achieve both simultaneously.

3. **Memory Bottleneck of Activation Sparsity**: Although activation sparsity is pervasive in LLMs (induced by activation functions such as ReLU/SwiGLU) and can skip weight rows corresponding to zero activations to accelerate computation, the activated rows cannot be predicted at runtime, necessitating storage of the full dense model in GPU HBM, which prevents actual memory reduction.

4. **Naive Combination Degrades Performance**: Directly applying activation sparsity on top of pruned weights leads to worst-case effective computation approaching $(1 - p^x)$ due to uneven weight sparsity distribution, and pruning calibration does not account for errors introduced by runtime activation sparsity.

5. **Limitations of Existing OBC-based Methods**: OBC-based methods such as SparseGPT calibrate only on dense activations, ignoring the fact that activations will be sparsified during inference, leading to a mismatch between the calibration objective and actual inference conditions.

6. **Computational Infeasibility of Naive Integration**: Directly incorporating activation sparsity into the OBC framework yields a computational complexity of $\mathcal{O}(nmk^2 + nk^3)$ (with $m = 262144$ and $k = 4096$ in LLaMA-2-7B), far exceeding the practical capacity of modern GPUs.

## Core Problem
How can both weight sparsity and activation sparsity be jointly exploited in one-shot LLM pruning, achieving computational acceleration and memory compression during the decoding phase while maintaining accuracy?

## Method

### Core Insight: Activation Sparsity = Dynamic Structured Weight Sparsity
During the decoding phase (GEMV operations), zero elements in the activation vector imply that the corresponding weight columns do not participate in computation, which is equivalent to dynamically applying structured pruning to the rows of the weight matrix. Combined with static unstructured weight pruning, this constitutes an spMspV (sparse matrix × sparse vector) workload.

### Activation-aware Pruning Calibration
The conventional OBC calibration objective $\|\Delta\mathbf{w}\mathbf{X}\|_F^2$ uses only dense inputs. DuoGPT introduces asymmetric calibration:
- Sparse activations $\hat{\mathbf{X}}$ (magnitude-pruned) are used as calibration inputs.
- The dense model output $\tilde{\mathbf{X}}$ is used as the calibration target, with residual $\mathbf{r} = \mathbf{w}(\tilde{\mathbf{X}} - \hat{\mathbf{X}})$.
- The calibration objective becomes $\|\Delta\mathbf{w}\hat{\mathbf{X}} - \mathbf{r}\|_F^2$, simultaneously adapting to activation sparsity and compensating for information loss.

### Efficient Implementation
Through three optimization steps, the complexity is reduced from $\mathcal{O}(nmk^2 + nk^3)$ to $\mathcal{O}(mk^2)$:
1. **Hessian Synchronization**: Once the pruning mask is fixed, all rows share the same Hessian, which is precomputed once via Cholesky decomposition.
2. **Pruning Score Precomputation**: The residual $\mathbf{R}$ is decomposed into a sum of column-wise outer products, yielding a vectorizable score formula $\mathbf{S}_{:,p} = \mathbf{W}_{:,p}^2(1/\mathbf{H}_{pp}^{-1} + \mathbf{a}_p - \mathbf{b}_p + 2\mathbf{c}_p)$.
3. **Intermediate Quantity Reuse**: The shared matrix $\mathbf{Q} = \Delta\mathbf{X}\hat{\mathbf{X}}^\top\mathbf{L}$ is reused for computing $\mathbf{b}$, $\mathbf{c}$, and the compensation term $\mathbf{D}$.

### Theoretical Guarantee
A theorem establishes that the improvement in loss achieved by DuoGPT over SparseGPT is lower-bounded and scales linearly with the activation sparsity $p^x$: $\Delta\mathcal{L} \geq \alpha p^x \sigma_r^2 C_\mathbf{w}^2 m / \lambda_{\max}(\mathbf{H})$.

## Key Experimental Results

### Table 1: Comparison with Unstructured Pruning Baselines (50% Dual Sparsity)

| Model | Method | Wiki2 PPL↓ | Avg. Accuracy↑ |
|------|------|-----------|------------|
| LLaMA-3-8B | Dense | 6.14 | 72.71% |
| LLaMA-3-8B | SparseGPT | 14.05 | 59.51% |
| LLaMA-3-8B | Wanda | 15.98 | 57.05% |
| LLaMA-3-8B | **DuoGPT** | **13.41** | **60.04%** |
| LLaMA-3-70B | Dense | 2.86 | 80.09% |
| LLaMA-3-70B | SparseGPT | 7.54 | 71.65% |
| LLaMA-3-70B | **DuoGPT** | **7.38** | **72.56%** |

### Table 2: Comparison with Structured Pruning Baselines (LLaMA-2-7B)

| Method | Model Size | Speedup | Avg. Accuracy↑ |
|------|---------|--------|------------|
| ShortGPT | 4.72B/6.74B | 1.44× | 51.35% |
| 2SSP | 4.72B/6.74B | 1.31× | 57.34% |
| SliceGPT | 5.29B/6.74B | 1.26× | 56.68% |
| **DuoGPT** | **3.50B/6.74B** | **1.39×** | **60.52%** |

At a comparable speedup ratio, DuoGPT surpasses ShortGPT, the best-performing structured pruning method, by **9.17%** in accuracy.

## Highlights & Insights

1. **Novel Perspective**: Reinterpreting activation sparsity as dynamic structured weight sparsity elegantly unifies the two forms of sparsity without performing actual structured pruning.
2. **Simultaneous Memory and Computation Reduction**: The dual-sparse scheme reduces both model storage (50% weight pruning) and runtime computation (skipping rows corresponding to zero activations), along with HBM→SRAM bandwidth consumption.
3. **Efficient Implementation**: Calibration for a 70B-parameter model requires only approximately 2.3 hours on a single A100 80 GB GPU.
4. **Theoretical Support**: A provable lower bound on the loss improvement of DuoGPT over SparseGPT is provided, with predictions consistent with empirical trends.

## Limitations & Future Work

1. **Decoding-only Focus**: The method primarily targets GEMV operations in single-batch decoding and is not optimized for the prefill phase (GEMM) or large-batch inference scenarios.
2. **Uniform Sparsity**: A uniform activation sparsity $p^x$ is applied across all Transformer layers; layer-adaptive sparsity strategies are not explored.
3. **Dependence on Natural Activation Sparsity**: The method relies on naturally occurring activation sparsity induced by activation functions in LLMs, and may be less effective for architectures with limited activation sparsity.
4. **Limited Hardware Adaptation**: Although the spMspV workload is discussed, no dedicated GPU kernel implementation or end-to-end latency measurements are provided.
5. **Significant Accuracy Degradation at High Sparsity**: At 65% dual sparsity, PPL increases from 5.47 to 77.3, limiting practical utility.

## Related Work & Insights

- **vs. SparseGPT/Wanda**: These methods perform weight pruning while ignoring runtime activation sparsity, resulting in a mismatch between calibration objectives and inference conditions. DuoGPT consistently achieves lower PPL and higher downstream accuracy across all models and scales through activation-aware calibration.
- **vs. TEAL/R-Sparse/CATS**: These methods leverage activation sparsity to accelerate inference but retain dense weights, precluding model storage reduction. DuoGPT reduces both storage and computation; additionally, R-Sparse's SVD branch actually increases memory by 1%.
- **vs. ShortGPT/SliceGPT/2SSP and other structured pruning methods**: At a comparable speedup ratio (~1.4×), DuoGPT achieves substantially higher accuracy (60.52% vs. 51.35%–57.34%) with a smaller model size.
- **vs. GPTQ-v2 Asymmetric Calibration**: DuoGPT draws inspiration from asymmetric calibration but applies it to pruning-based sparsification rather than quantization, representing the first application of this technique to dual-sparse pruning.
- **vs. STUN (Joint Structured + Unstructured Pruning)**: STUN physically performs both types of pruning, whereas DuoGPT *reinterprets* activation sparsity as structured sparsity and applies only unstructured pruning during calibration, yielding a more unified and elegant formulation.

## Rating
- Novelty: ⭐⭐⭐⭐ — The perspective of unifying activation sparsity as dynamic weight sparsity is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers the full LLaMA-2/3 model family with comprehensive ablation studies, though actual latency measurements are absent.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are rigorous and clear; the progression from theory to efficient implementation is well-structured.
- Value: ⭐⭐⭐⭐ — Practically significant for LLM deployment optimization; the dual-sparsity paradigm merits further exploration in subsequent work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning](twilight_adaptive_attention_sparsity_with_hierarchical_top-p_pruning.md)
- [\[NeurIPS 2025\] MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference](mustafar_promoting_unstructured_sparsity_for_kv_cache_pruning_in_llm_inference.md)
- [\[NeurIPS 2025\] Spark Transformer: Reactivating Sparsity in FFN and Attention](spark_transformer_reactivating_sparsity_in_ffn_and_attention.md)
- [\[ICCV 2025\] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](../../ICCV2025/model_compression/mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)
- [\[ACL 2026\] CadLLM: Improving the Throughput of Diffusion-based LLMs via Training-Free Confidence-Aware Calibration](../../ACL2026/model_compression/improving_the_throughput_of_diffusion-based_large_language_models_via_a_training.md)

</div>

<!-- RELATED:END -->
