---
title: >-
  [Paper Note] SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization
description: >-
  [ACL 2026][LLM Pretraining][Optimizer] This paper proposes the SAGE optimizer, which addresses the "embedding dilemma" where lightweight optimizers fail on embedding layers. By combining a Lion-style sign update directio…
tags:
  - "ACL 2026"
  - "LLM Pretraining"
  - "Optimizer"
  - "Memory Efficiency"
  - "Embedding Layer"
  - "Sign Optimization"
  - "Adaptive Scaling"
date: 2026-05-08
content_hash: b28c4d0e34d4d7be
---

# SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization

**Conference**: ACL 2026  
**arXiv**: [2604.07663](https://arxiv.org/abs/2604.07663)  
**Code**: [GitHub](https://github.com/naubull2/SAGE-optimizer)  
**Area**: LLM Pre-training  
**Keywords**: Optimizer, Memory Efficiency, Embedding Layer, Sign Optimization, Adaptive Scaling

## TL;DR

This paper proposes the SAGE optimizer, which addresses the "embedding dilemma" where lightweight optimizers fail on embedding layers. By combining a Lion-style sign update direction with an adaptive damping scale factor requiring only $O(d)$ memory overhead, SAGE achieves new SOTA perplexity on Llama models (up to 1.3B) with significantly lower optimizer memory.

## Background & Motivation

**Background**: AdamW is the standard optimizer for LLM pre-training, but its two full-sized momentum states ($O(Vd)$) consume memory equivalent to twice the model size, serving as a critical memory bottleneck. Lightweight optimizers like Lion (single momentum) and SinkGD (stateless normalization) have shown progress.

**Limitations of Prior Work**: Lightweight optimizers perform well on dense layers but fail on embedding layers. Embedding gradients exhibit sparsity and high variance due to Zipfian token distributions, which stateless methods cannot handle effectively. Consequently, methods like SinkGD adopt hybrid designs—falling back to AdamW for embedding layers—partially negating memory savings.

**Key Challenge**: The embedding layer is the largest consumer of optimizer state memory ($V > 100,000$), yet it is precisely where lightweight optimizers fail. To achieve true memory efficiency, the embedding layer must be conquered.

**Goal**: Design a lightweight optimizer that can successfully replace AdamW for embedding layers.

**Key Insight**: Lion's update magnitude is a static 1.0 (identical for every dimension), lacking control over high-variance dimensions. If a bounded adaptive scaling factor could be designed to selectively damp high-variance dimensions, stability could be gained while maintaining memory efficiency.

**Core Idea**: SAGE = Sign direction of Lion + a new $O(d)$ adaptive damping scale factor $\mathbf{H}_t$. This scaling factor is based on the EMA of absolute gradient values ($L_1$ norm), is theoretically bounded by $\|\mathbf{H}_t\|_\infty \leq 1.0$, applies stronger damping to high-variance dimensions, and degrades to Lion’s 1.0 for quiet dimensions.

## Method

### Overall Architecture

A hybrid optimizer structure is adopted: SAGE ($O(Vd) + O(d)$ states) for embedding layers and 1D parameters (bias/norm), and SinkGD ($O(1)$ state) for dense 2D weights. Compared to SinkGD+AdamW hybrids, this reduces optimizer state memory for the embedding layer by approximately 50%.

### Key Designs

1. **$O(d)$ Adaptive Damping Scale Factor $\mathbf{H}_t$**:

    - **Function**: Selectively damps the update magnitude of high-variance dimensions.
    - **Mechanism**: For the embedding layer, the mean of absolute gradient values per embedding dimension $j$ is first calculated: $(\mathbf{s}_t)_j = \frac{1}{V} \sum_{i=1}^V |g_{t,ij}|$ (an $O(d)$ vector). An EMA $\hat{\mathbf{S}}_t$ is computed, and the layer-wise RMS is used as a reference value $\sigma_{rms}$. The damping factor is $(\mathbf{H}_t)_j = \min(\sigma_{rms} / (\hat{\mathbf{S}}_t)_j, 1)$. "Quiet" dimensions ($\hat{S}_j < \sigma_{rms}$) with ratios $>1$ are clipped to 1 (degrading to Lion), while "noisy" dimensions ($\hat{S}_j > \sigma_{rms}$) are damped to $<1$.
    - **Design Motivation**: Dimension adaptation is achieved with $O(d)$ rather than $O(Vd)$ states, making memory overhead negligible. Boundedness ensures updates are never more aggressive than Lion, which is theoretically provable for convergence.

2. **Instantaneous Stability Constraint**:

    - **Function**: Prevents instability caused by sudden gradient spikes due to EMA lag.
    - **Mechanism**: In addition to EMA-based damping $\mathbf{D}_t^{ema}$, an instantaneous damping $\mathbf{D}_t^{inst}$ based on current batch statistics is calculated. The final factor is the minimum of both and 1.0: $(\mathbf{H}_t)_j = \min(\mathbf{D}_t^{ema}, \mathbf{D}_t^{inst}, 1)$.
    - **Design Motivation**: Similar to Adaptive Gradient Clipping (AGC), this provides immediate protection against catastrophic instability.

3. **Adaptive Generalization of Lion**:

    - **Function**: Generalizes Lion from static scaling to adaptive scaling.
    - **Mechanism**: Lion updates as $\hat{\mathbf{U}}_t^{Lion} = \mathbf{C}_t \odot \mathbf{1}$, whereas SAGE updates as $\hat{\mathbf{U}}_t^{SAGE} = \mathbf{C}_t \odot \mathbf{H}_t$. Lion is a special case of SAGE where $\mathbf{H}_t$ is fixed to $\mathbf{1}$. Since $\|\mathbf{H}_t\|_\infty \leq 1$, SAGE acts as a "safe generalization" of Lion.
    - **Design Motivation**: Safer updates allow for higher learning rates, leading to better convergence.

### Loss & Training

Decoupled weight decay (AdamW style) is utilized. SAGE maintains one $O(Vd)$ momentum state plus one $O(d)$ adaptive state, resulting in total memory that is only half of AdamW.

## Key Experimental Results

### Main Results (Test Perplexity)

| Method | 270M PPL | Memory | 1.3B PPL | Memory |
| :--- | :--- | :--- | :--- | :--- |
| AdamW | 37.35 | 2.1GB | 27.81 | 9.8GB |
| Lion | 30.24 | 1.0GB | 28.37 | 4.9GB |
| SinkGD-Hybrid | 34.30 | 0.9GB | 28.71 | 1.9GB |
| **SAGE-Hybrid** | **29.95** | **0.5GB** | **24.33** | **0.9GB** |

### Ablation Study

| Configuration | PPL (270M) | Description |
| :--- | :--- | :--- |
| SAGE-Hybrid | 29.95 | Full method |
| SinkGD-Pure | 192.7 | Stateless method fails on embedding layer |
| SAGE-Pure | 116.0 | SAGE alone for all layers is also suboptimal |
| Lion-Hybrid | 32.10 | Replacing AdamW for embedding with Lion |

### Key Findings
- SAGE-Hybrid achieves the lowest perplexity across all model sizes while using only ~10% of AdamW's memory.
- SinkGD-Pure confirms the existence of the embedding dilemma—pure stateless optimizers fail catastrophically on the embedding layer.
- The boundedness of SAGE allows for higher learning rates than Lion, which is key to performance improvements.
- The hybrid design (SAGE for embedding + SinkGD for dense) is the optimal combination.

## Highlights & Insights
- **Precise diagnosis of the "embedding dilemma"**: Identifying sparsity and high variance in embedding gradients as the root cause of failure for lightweight optimizers allows for a targeted solution.
- **Extremely memory-efficient $O(d)$ adaptive scaling**: Compressing $V \times d$ gradient information into a $d$-dimensional mean absolute value tracked by a $d$-dimensional EMA adds almost no memory overhead.
- **Elegant generalization from Lion to SAGE**: SAGE is a strictly safe generalization of Lion, offering both theoretical guarantees and intuitive explanation.

## Limitations & Future Work
- Experiments are limited to 1.3B parameters; effectiveness on larger models (7B+) has not been verified.
- SAGE was only tested on the Llama architecture; performance on other architectures (e.g., Mixture of Experts) is unknown.
- The number of pre-training tokens and datasets are relatively small (subsets of RedPajama); performance in large-scale pre-training remains to be verified.
- Systematic comparisons with other low-rank methods like GaLore or APOLLO were not performed (though APOLLO results were poor).

## Related Work & Insights
- **vs AdamW**: Significant improvement in memory efficiency (~10× smaller optimizer states) with superior perplexity.
- **vs Lion**: SAGE is an adaptive generalization of Lion, allowing higher learning rates through bounded damping.
- **vs SinkGD**: While SinkGD requires falling back to AdamW for embedding layers, SAGE replaces this fallback.

## Rating
- Novelty: ⭐⭐⭐⭐ The diagnosis of the embedding dilemma and the $O(d)$ adaptive scaling solution are highly novel.
- Experimental Thoroughness: ⭐⭐⭐ Model sizes are relatively small, and larger-scale validation is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivation of motivation and rigorous theoretical analysis.
- Value: ⭐⭐⭐⭐ Provides a practical optimizer solution for memory-constrained LLM training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](../../ICLR2026/llm_pretraining/scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)
- [\[NeurIPS 2025\] Vocabulary Customization for Efficient Domain-Specific LLM Deployment](../../NeurIPS2025/llm_pretraining/vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)
- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](../../NeurIPS2025/llm_pretraining/breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)
- [\[ICML 2026\] SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs](../../ICML2026/llm_pretraining/spare_stacked_parallelism_with_adaptive_reordering_for_fault-tolerant_llm_pretra.md)
- [\[ACL 2026\] Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity](working_memory_constraints_scaffold_learning_in_transformers_under_data_scarcity.md)

</div>

<!-- RELATED:END -->
