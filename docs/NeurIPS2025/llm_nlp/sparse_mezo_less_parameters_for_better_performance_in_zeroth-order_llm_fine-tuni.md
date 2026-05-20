---
title: >-
  [Paper Note] Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning
description: >-
  [NeurIPS 2025][LLM/NLP][zeroth-order optimization] This paper proposes Sparse MeZO (S-MeZO), motivated by the observation that zeroth-order gradient noise disproportionately affects parameters with large magnitudes. S-Me…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "zeroth-order optimization"
  - "sparse fine-tuning"
  - "memory efficiency"
  - "LLM fine-tuning"
  - "gradient noise"
date: 2026-05-08
content_hash: 194a79dfd6dab8df
---

# Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2402.15751](https://arxiv.org/abs/2402.15751)  
**Code**: [GitHub](https://github.com/NUS-HPC-AI-Lab/SparseMeZO)  
**Area**: LLM/NLP
**Keywords**: zeroth-order optimization, sparse fine-tuning, memory efficiency, LLM fine-tuning, gradient noise

## TL;DR

This paper proposes Sparse MeZO (S-MeZO), motivated by the observation that zeroth-order gradient noise disproportionately affects parameters with large magnitudes. S-MeZO selectively applies zeroth-order perturbation and updates only to small-magnitude parameters, achieving significant performance gains (+9% on RTE) and convergence acceleration (3.5×) without any additional memory overhead.

## Background & Motivation

LLM fine-tuning demands substantial memory due to activation caching during backpropagation. Parameter-efficient fine-tuning (PEFT) methods such as LoRA still require approximately 6× the memory of inference. Zeroth-order (ZO) optimization estimates gradients using only forward passes; MeZO applies this technique to LLM fine-tuning, reducing memory consumption to the inference level.

**Core limitations of MeZO**:

**High gradient estimation noise**: The ZO gradient $\delta = g(\theta) - g_z(\theta)$ deviates substantially from the true gradient.

**Extreme sensitivity to learning rate**: Increasing the learning rate from $1 \times 10^{-6}$ to $2 \times 10^{-6}$ causes divergence.

**Poor generalization**: While 90% of steps successfully reduce loss on the current batch, only 50% do so on a held-out batch.

**Key finding**: **Gradient noise has a far greater impact on large-magnitude parameters than on small-magnitude ones.** Resuming optimization exclusively on small-magnitude parameters from a performance degradation point recovers and even improves accuracy. This empirical observation forms the theoretical foundation of Sparse MeZO.

## Method

### Overall Architecture

S-MeZO augments standard MeZO with a sparse mask that restricts perturbation and parameter updates to a magnitude-selected subset of parameters.

### Key Design 1: Sparse Mask Mechanism

A sparse mask $\mathbf{m} \in \{0,1\}^d$ is defined to sparsify the random noise vector:

$$\hat{\mathbf{z}} = \mathbf{m} \odot \mathbf{z}$$

The sparse zeroth-order gradient estimate is then:

$$g_{\hat{z}}(\theta) = \frac{\mathcal{L}(\theta + \epsilon \mathbf{m} \odot \mathbf{z}) - \mathcal{L}(\theta - \epsilon \mathbf{m} \odot \mathbf{z})}{2\epsilon} \hat{\mathbf{z}}$$

The mask is dynamically generated based on parameter magnitudes: only the smallest-magnitude parameters within each layer (selected by a percentile threshold) are updated.

### Key Design 2: Dynamic Mask (No Extra Memory)

Two masking strategies are considered:

- **Constant mask**: Fixed before training, but requires storing a mask of the same size as the model.
- **Dynamic mask** (recommended): Regenerated at each iteration from current parameter values; no storage required.

### Key Design 3: Computing the Mask During the Forward Pass

A novel memory-efficient implementation computes and immediately discards the mask layer by layer during the forward pass, avoiding global mask storage:

$$y^{(i)} = (\theta_t^{(i)} + \epsilon \cdot m(\theta_t) \cdot z^{(i)}) x^{(i)} + b^{(i)}$$

After computing $m^{(i)}$ for each layer, its memory is released before processing the next layer. This ensures that S-MeZO incurs exactly the same memory footprint as the original MeZO (inference-level only).

### Loss & Training

S-MeZO adopts the same training setup as MeZO. The key difference is the use of a higher learning rate, which is feasible because updates restricted to small-magnitude parameters are more robust to noise. The sparsity ratio is typically set between 0.5 and 0.8, meaning only 20–50% of parameters are updated.

## Key Experimental Results

### Main Results

**LLaMA-7b on SuperGLUE**:

| Method | BoolQ | RTE | WIC | MultiRC | SST-2 | COPA | Avg. |
|--------|:-----:|:---:|:---:|:-------:|:-----:|:----:|:----:|
| Zero-Shot | 65.1 | 49.5 | 50.6 | 55.8 | 79.7 | 59.7 | 60.1 |
| LoRA | 84.5 | 82.3 | 67.6 | 78.3 | 95.0 | 86.0 | 82.3 |
| FT (Adam) | 84.5 | 83.6 | 68.4 | 80.2 | 95.7 | 85.0 | 82.9 |
| MeZO | 75.9 | 71.7 | 61.4 | 69.8 | 94.6 | 86.3 | 76.6 |
| R-MeZO | 76.9 | 75.2 | 62.1 | 68.1 | 94.6 | 84.3 | 76.9 |
| **S-MeZO** | **80.9** | **80.7** | **64.9** | **73.3** | **95.0** | **86.7** | **80.3** |

S-MeZO achieves **+9.0%** on RTE, **+5.0%** on BoolQ, and **+3.7%** on average over MeZO.

**LLaMA2-7b with additional baselines**:

| Method | BoolQ | RTE | WIC | SST-2 | Avg. |
|--------|:-----:|:---:|:---:|:-----:|:----:|
| MeZO | 78.8 | 70.2 | 62.2 | 94.0 | 76.3 |
| ZO-AdaMU | 78.2 | 76.5 | 63.0 | 93.6 | 77.8 |
| AdaZeta | 79.8 | 75.8 | 62.0 | 94.0 | 77.9 |
| **S-MeZO** | **82.2** | **77.6** | **65.3** | **94.8** | **80.0** |

### Ablation Study

**Convergence speed**: S-MeZO reaches 70% accuracy on RTE in approximately 5,000 steps, whereas MeZO requires 17,500 steps (**3.5× speedup**).

**Memory usage (LLaMA-7b)**:

| Method | Memory |
|--------|:------:|
| FT (Adam) | 128.2 GB |
| LoRA | 22.4 GB |
| MeZO | **14.6 GB** |
| S-MeZO (efficient impl.) | **14.6 GB** |

**Effect of sparsity ratio**: Sparsity ratios of 0.5–0.8 yield the best performance; at 0.8, RTE improves from 71.7% to 82.3%.

### Key Findings

1. **Large-magnitude parameters are more susceptible to ZO noise**: Restricting updates to small-magnitude parameters substantially recovers and improves performance.
2. **S-MeZO markedly narrows the gap between ZO and first-order methods**: The average gap shrinks from 6.3% to 2.6%.
3. **Efficient implementation introduces no additional memory overhead**: Computing the mask during the forward pass keeps memory at the inference level.
4. **Scalable to 30B models**: LLaMA-30b can be fine-tuned on a single A100 GPU.

## Highlights & Insights

1. **Observation-driven method design**: The approach originates from the empirical discovery that large-magnitude parameters are more sensitive to noise, which then motivates the targeted sparsification strategy.
2. **Extreme memory efficiency**: The forward-pass mask computation eliminates the need to store any global mask, making S-MeZO memory-equivalent to inference.
3. **Strong practical utility**: The ability to fine-tune a 30B model on a single A100 GPU makes the method accessible in resource-constrained settings.
4. **Consistent improvements**: Stable gains are observed across three model families—LLaMA, Mistral, and OPT.
5. **Simple yet effective**: The core idea—perturbing only small-magnitude parameters—is straightforward to implement but yields substantial improvements.

## Limitations & Future Work

1. **Remaining gap to first-order methods**: Despite significant progress, a 2.6% gap persists between S-MeZO (80.3) and full fine-tuning (82.9).
2. **Threshold selection**: The sparsity ratio must be set manually and may require task-specific tuning.
3. **Insufficient theoretical analysis**: The theoretical explanation for why small-magnitude parameters are more robust to noise remains shallow.
4. **Large number of training steps**: The method still requires a substantial number of steps to converge.
5. Hybrid strategies combining S-MeZO with PEFT methods such as LoRA warrant further exploration.

## Related Work & Insights

- **MeZO** (Malladi et al., 2023): The foundational work on memory-efficient zeroth-order optimization.
- **LoRA** (Hu et al., 2021): A seminal parameter-efficient fine-tuning method.
- **Lottery Ticket Hypothesis** (Frankle & Carlin, 2018): Theoretical grounding for sparse subnetworks.
- **DeepZero** (Chen et al., 2023): Zeroth-order training guided by model pruning.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: A simple but insightful core observation paired with an elegant memory-efficient implementation.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Comprehensive evaluation across multiple models and tasks.
- **Value** ⭐⭐⭐⭐⭐: Single-GPU fine-tuning of 30B models with minimal memory overhead.
- **Writing Quality** ⭐⭐⭐: Convergence proofs are provided but lack theoretical depth.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs](triplets_better_than_pairs_towards_stable_and_effective_self-play_fine-tuning_fo.md)
- [\[NeurIPS 2025\] Synergy over Discrepancy: A Partition-Based Approach to Multi-Domain LLM Fine-Tuning](synergy_over_discrepancy_a_partition-based_approach_to_multi-domain_llm_fine-tun.md)
- [\[NeurIPS 2025\] SPACE: Noise Contrastive Estimation Stabilizes Self-Play Fine-Tuning for Large Language Models](space_noise_contrastive_estimation_stabilizes_self-play_fine-tuning_for_large_la.md)
- [\[NeurIPS 2025\] MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery](moose-chem2_exploring_llm_limits_in_fine-grained_scientific_hypothesis_discovery.md)
- [\[NeurIPS 2025\] Spectral Conditioning of Attention Improves Transformer Performance](spectral_conditioning_of_attention_improves_transformer_performance.md)

</div>

<!-- RELATED:END -->
