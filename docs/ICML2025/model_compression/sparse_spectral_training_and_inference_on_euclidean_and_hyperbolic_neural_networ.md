---
title: >-
  [Paper Note] Sparse Spectral Training and Inference on Euclidean and Hyperbolic Neural Networks
description: >-
  [ICML2025][Model Compression][Sparse Spectral Training] Sparse Spectral Training (SST) is proposed to achieve a performance close to full-rank pre-training with memory overheads on par with LoRA. This is achieved by updating all singular values at each step in the spectral domain, selectively updating singular vectors via multinomial sampling according to their magnitudes, and periodically running re-SVD to maintain orthogonality.
tags:
  - "ICML2025"
  - "Model Compression"
  - "Sparse Spectral Training"
  - "SVD"
  - "Low-Rank Adaptation"
  - "Pre-training"
  - "Hyperbolic Neural Networks"
date: 2026-05-08
content_hash: 799afe0b484b3543
---

# Sparse Spectral Training and Inference on Euclidean and Hyperbolic Neural Networks

**Conference**: ICML2025  
**arXiv**: [2405.15481](https://arxiv.org/abs/2405.15481)  
**Code**: [GitHub](https://github.com/biomedical-cybernetics/sparse-spectral-training)  
**Area**: Sparse Training / Parameter-Efficient Pre-training  
**Keywords**: Sparse Spectral Training, SVD, Low-Rank Adaptation, Pre-training, Hyperbolic Neural Networks

## TL;DR

Sparse Spectral Training (SST) is proposed to achieve a performance close to full-rank pre-training with memory overheads on par with LoRA. This is achieved by updating all singular values at each step in the spectral domain, selectively updating singular vectors via multinomial sampling according to their magnitudes, and periodically running re-SVD to maintain orthogonality.

## Background & Motivation

Large-scale model pre-training requires substantial GPU memory. Existing low-rank training methods have distinct limitations:

- **LoRA**: Restricts the weight increment to a fixed low-rank subspace $\Delta W = BA$. According to the Eckart-Young-Mirsky theorem, the low-rank approximation error satisfies $\|\Delta W^* - \Delta W\|_F \ge \sqrt{\sigma_{r+1}^2 + \cdots + \sigma_m^2}$. In complex tasks such as pre-training, $\sigma_i (i > r)$ cannot be neglected, severely limiting performance.
- **ReLoRA / COLA / PLoRA**: Break the rank barrier by periodically merging low-rank matrices into the base weights. However, resetting $B$ to zero after each merge leads to $\frac{\partial \mathcal{L}}{\partial A} = \mathbf{0}^T \frac{\partial \mathcal{L}}{\partial W} = \mathbf{0}$, creating a **saddle point problem** that slows down convergence.
- **GaLore**: Project gradients onto a low-rank space, but the projection matrix is based only on the SVD of a single-batch gradient, which becomes unstable (e.g., causing loss spikes during OPT-350M training) when the low rank is small.

The motivation of SST is to balance **exploitation** of existing dominant directions and **exploration** of new directions in the spectral domain, thereby matching the learning dynamics of full-rank training.

## Method

### Core Framework: Sparse Spectral Layer

SST replaces the weight matrix $W \in \mathbb{R}^{m \times n}$ of each linear layer with its SVD formulation:

$$\mathbf{h} = W\mathbf{x} = U \Sigma V^T \mathbf{x}, \quad [U, \Sigma, V^T] = \text{SVD}(W)$$

where $U \in \mathbb{R}^{m \times m}$, $\Sigma \in \mathbb{R}^{m \times m}$, and $V^T \in \mathbb{R}^{m \times n}$. The key difference is that **SST retains the full-rank structure**, and the original $W$ is removed from the network.

### Update Strategy

**1. Step-wise updates for all singular values**: $\Sigma$ is simplified to an $m$-dimensional vector. It is updated at each step with minimal overhead:

$$\Sigma^{t+1} = \max(\Sigma^t - \eta \nabla \mathcal{L}_\Sigma, 0)$$

The $\max$ function ensures that singular values remain non-negative.

**2. Selective update of singular vectors via multinomial sampling**: $U$ and $V^T$ are selectively updated by sampling $r$ vectors at each iteration using multinomial sampling based on the magnitudes of their corresponding singular values:

$$S \subseteq \{1, \ldots, m\}, \quad S \sim \text{Multinomial}(r, \Sigma)$$

Selected vectors undergo gradient descent and are then normalized to maintain unit norm:

$$U_{\cdot i}^{t+1} = \frac{U_{\cdot i}^t - \eta \nabla \mathcal{L}_{U_{\cdot i}}}{|U_{\cdot i}^t - \eta \nabla \mathcal{L}_{U_{\cdot i}}|}, \quad i \in S$$

**3. Enhanced Gradient**: The default gradient $\nabla \mathcal{L}_{U_{\cdot i}} = \frac{\partial \mathcal{L}}{\partial W} V_{\cdot i} \Sigma_i$ couples the direction and magnitude. SST proposes an enhanced gradient that decouples $\Sigma_i$:

$$\tilde{\nabla} \mathcal{L}_{U_{\cdot i}} = \frac{\partial \mathcal{L}}{\partial W} V_{\cdot i}$$

This ensures that vectors corresponding to small singular values still receive adequate gradients.

**4. Periodic re-SVD**: As the orthogonality of $U$ and $V^T$ gradually degrades during training, a periodic re-SVD is performed to restore orthogonality:

$$[U^{t+1}, \Sigma^{t+1}, {V^{t+1}}^T] = \text{SVD}(U^t \Sigma^t {V^t}^T)$$

This prevents learning from degrading into a low-rank subspace.

### Memory-Efficient Implementation

$U$ and $V^T$ are split into an active partition ($m \times r$, which stores optimizer states) and a frozen partition. At each step, newly sampled vectors are swapped with the inactive (unselected) vectors in the active partition, similar to resource scheduling in a **time-sharing operating system**. The ratio of trainable parameters is:

$$\Gamma_{\text{SST},r} = \frac{r(m+n)+m}{mn}$$

This is slightly higher than LoRA with the same rank, but lower than LoRA with rank $r+1$.

## Key Experimental Results

### LLM Pre-training (OPT / LLaMA, OpenWebText)

| Model | r/d | Full | LoRA | ReLoRA* | **SST** |
|------|------|------|------|---------|---------|
| OPT-125M | 64/768 | 23.50 | 34.23 | 35.80 | **26.98** |
| OPT-350M | 64/1024 | 21.78 | 34.26 | 39.21 | **27.72** |
| OPT-1.3B | 64/2048 | 15.10 | 17.16 | 29.52 | **22.31** |
| LLaMA-130M | 64/768 | 20.04 | 29.71 | 31.33 | **23.35** |
| LLaMA-1.3B | 128/2048 | 14.54 | 16.50 | 17.32 | **14.59** |

- On LLaMA-1.3B, with only **18.7% trainable parameters** (rank being 6% of the embedding dimension), SST bridges the perplexity gap between low-rank methods and full-rank training by **97.4%**.

### Machine Translation (IWSLT'14, Euclidean & Hyperbolic Transformer)

| Dimension | r | Full (E) | SST (E) | Full (H) | SST (H) |
|------|---|----------|---------|----------|---------|
| 64 | 8 | 24.27 | 22.28 | 25.69 | 23.40 |
| 128 | 16 | 25.79 | **25.12** | 24.70 | **25.22*** |
| 256 | 32 | 23.92 | **23.97*** | 19.94 | **25.04*** |

- * indicates that SST **outperforms full-rank**. On the Hyperbolic Transformer, SST significantly outperforms full-rank training.
- On machine translation, SST reduces the BLEU gap by **66.7%** on average.
- Hyperbolic Graph Neural Networks: The gap is bridged by **73.7%** for node classification and **82.5%** for link prediction.

### Comparison of Gradient Dynamics

The correlation between SST's gradient norm and that of full-rank training is **0.85**, whereas ReLoRA* is only **0.58**. ReLoRA* exhibits periodic saddle points (where the gradient norm drops abruptly to 0) after each merge, an issue that does not occur in SST.

## Highlights & Insights

1. **Spectral Exploration-Exploitation Balance**: LoRA only exploits fixed top-$r$ directions (pure exploitation), while ReLoRA* discards learned directions upon restart (pure exploration). SST balances both via singular-value-weighted sampling.
2. **Saddle Point Avoidance**: SVD initialization and re-SVD ensure that low-rank matrices are always updated along the primary directions of the weight matrix, eliminating the saddle-point problem caused by ReLoRA*'s zero initialization.
3. **Decoupled Enhanced Gradient**: By separating direction learning from magnitude learning, directions corresponding to small singular values can still receive effective gradients.
4. **Strong Versatility**: This work introduces parameter-efficient pre-training to hyperbolic space for the first time, establishing its effectiveness across both Euclidean and hyperbolic neural networks.
5. **Near Full-Rank Performance on LLaMA-1.3B**: The perplexity gap is extremely narrow (14.59 vs 14.54, a difference of only 0.05), paving the way for practical deployment.

## Limitations & Future Work

1. **SVD Overhead**: Although SVD is only executed during initialization and periodic re-SVD, it may still become a bottleneck for ultra-large models (>10B parameters).
2. **Sensitivity to Rank Selection**: The optimal value of $r$ varies across different tasks and models, and theoretical guidance is currently lacking.
3. **Validation Limited to 1.3B**: The effectiveness has not been verified on larger models (e.g., 7B+ parameters).
4. **Insufficient Comparison with GaLore**: GaLore was omitted from the main experiments due to its instability, which limits the persuasiveness of the comparative analysis.
5. **Fixed Sampling Strategy**: The multinomial sampling weights are directly derived from the singular values, without exploring adaptive or task-specific sampling strategies.

## Related Work & Insights

- **LoRA / ReLoRA / PiSSA / GaLore**: SST unifies these methods under a single perspective—viewing them as weight updates under different strategies in the spectral domain.
- **Eckart-Young-Mirsky Theorem**: Provides the theoretical lower bound for low-rank approximation, serving as a critical tool to analyze the limitations of LoRA.
- **Hyperbolic Neural Networks (HyboNet)**: SST represents the first effort to apply parameter-efficient pre-training within hyperbolic spaces.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of spectral sampling, full singular value updates, and enhanced gradient is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers LLM pre-training, translation, graph networks, and hyperbolic spaces, with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical derivations and comprehensive comparative analysis with existing methods.
- Value: ⭐⭐⭐⭐ — Achieves near full-rank performance on LLaMA-1.3B, showing high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Cannistraci-Hebb Training on Ultra-Sparse Spiking Neural Networks](../../ICLR2026/model_compression/cannistraci-hebb_training_on_ultra-sparse_spiking_neural_networks.md)
- [\[ICML 2025\] An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks](an_efficient_matrix_multiplication_algorithm_for_accelerating_inference_in_binar.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](../../ICLR2026/model_compression/a_recovery_guarantee_for_sparse_neural_networks.md)
- [\[ICML 2025\] Eigenspectrum Analysis of Neural Networks without Aspect Ratio Bias](eigenspectrum_analysis_of_neural_networks_without_aspect_ratio_bias.md)
- [\[ICLR 2026\] Alignment-Enhanced Integration of Connectivity and Spectral Sparsity in Dynamic Sparse Training of LLM](../../ICLR2026/model_compression/alignment-enhanced_integration_of_connectivity_and_spectral_sparsity_in_dynamic_.md)

</div>

<!-- RELATED:END -->
