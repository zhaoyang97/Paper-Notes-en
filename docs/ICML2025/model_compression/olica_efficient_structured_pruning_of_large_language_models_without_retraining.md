---
title: >-
  [Paper Note] Olica: Efficient Structured Pruning of Large Language Models without Retraining
description: >-
  [ICML2025][Model Compression][Structured Pruning] This paper proposes Olica, a retraining-free framework for structured LLM pruning. By performing orthogonal decomposition (PCA/SVD) on the matrix products of MHA layers and linear calibration (ridge regression closed-form solution + low-rank approximation) on FFN layers, Olica prunes LLaMA-7B in just 7 minutes using 256 samples and 3GB VRAM, outperforming existing methods that require retraining.
tags:
  - "ICML2025"
  - "Model Compression"
  - "Structured Pruning"
  - "LLM Compression"
  - "Orthogonal Decomposition"
  - "Linear Calibration"
  - "Retraining-free"
  - "PCA"
  - "SVD"
date: 2026-05-08
content_hash: 24fc0127ca977ee3
---

# Olica: Efficient Structured Pruning of Large Language Models without Retraining

**Conference**: ICML2025  
**arXiv**: [2506.08436](https://arxiv.org/abs/2506.08436)  
**Code**: [BetterTMrR/LLM-Olica](https://github.com/BetterTMrR/LLM-Olica)  
**Area**: Model Compression  
**Keywords**: Structured Pruning, LLM Compression, Orthogonal Decomposition, Linear Calibration, Retraining-free, PCA, SVD

## TL;DR
This paper proposes Olica, a retraining-free framework for structured LLM pruning. By performing orthogonal decomposition (PCA/SVD) on the matrix products of MHA layers and linear calibration (ridge regression closed-form solution + low-rank approximation) on FFN layers, Olica prunes LLaMA-7B in just 7 minutes using 256 samples and 3GB VRAM, outperforming existing methods that require retraining.

## Background & Motivation

Existing structured pruning methods for LLMs (e.g., LLM-Pruner, SlimGPT, LoRAP) heavily rely on extensive data and computational resources for **LoRA retraining** to recover disrupted inter-layer relationships:

- DISP-LLM requires 4× A100 80GB GPUs to prune LLaMA-13B.
- LLM-Pruner / SlimGPT require 50K annotated samples (Alpaca) for retraining.
- Pruning in domain-specific scenarios incurs an extremely high cost for annotating large instruction datasets.

**Key Observation**: The computation of the MHA layer fundamentally relies on two categories of matrix products: $W_q W_k^\top$ and $W_v W_o^\top$. These products can be treated as unified entities, allowing direct PCA execution to extract the most critical information, thereby bypassing the need for retraining.

## Method

Olica consists of three core components: **Orthogonal Neuron Decomposition (OND)**, **Fast OND**, and **Linear Calibration (LC)**.

### 1. Orthogonal Neuron Decomposition (OND) — MHA Layer Compression

The MHA layer relies on matrix products $W_{qk} = W_q W_k^\top$ and $W_{vo} = W_v W_o^\top$. Treating these products as unified entities, SVD is applied to $W_{vo}$:

$$W_{vo} = U \Sigma V^\top$$

Let $\hat{W}_v \leftarrow U\Sigma$ and $\hat{W}_o \leftarrow V$, which yields $\hat{W}_v \hat{W}_o^\top = W_v W_o^\top$. Because the columns of $U$ and $V$ are orthogonal, the decomposed neurons carry non-redundant information, ensuring maximum information retention for a given dimension.

**Pruning Strategy**: Rather than performing simple pruning based on eigenvalue magnitudes, an importance score based on weight-activation magnitude is utilized:

$$\mathcal{I}(\text{neuron}_j) = \sum_i [\mathcal{I}(\hat{W}_{v_{ij}}) + \mathcal{I}(\hat{W}_{o_{ij}})]$$

where $\mathcal{I}(W_{ij}) = \|x^{(i)}\|_2 \cdot |W_{ij}|$ (Wanda score), pruning the least critical neurons.

### 2. Fast OND — Lowering SVD Complexity

Performing SVD on $W_{vo} \in \mathbb{R}^{d \times d}$ has a complexity of $O(d^3)$, totaling $O(h d^3)$ across $h$ attention heads, which takes approximately 1 hour for a 7B model.

**Key Discovery**: The singular value distributions of $W_v$ and $W_o$ are highly similar (as are those of $W_q$ and $W_k$). Consequently, performing SVD on only one of these matrices suffices to approximate the redundancy of the joint entity. For example, performing SVD on $W_v \in \mathbb{R}^{d \times d/h}$ reduces complexity to $O(d^3/h^2)$, totaling $O(d^3/h)$ across $h$ heads—effectively **reducing the computation by an $h^2$ factor**.

### 3. Linear Calibration (LC) — FFN Error Compensation

Pruning changes the outputs of FFN layers, which accumulates and amplifies errors across layers. Olica reconstructs the pruning residual utilizing a closed-form ridge regression:

$$\hat{W} = \arg\min_{W} \|E - XW\|_2^2 + \lambda \|W\|_F^2$$

where $E = f(X) - \hat{f}(X)$ is the output residual before and after pruning. The closed-form solution is $\hat{W} = (X^\top X + \lambda I)^{-1} X E$, which **requires no iterative training**.

After calibration, the forward pass becomes: $X_{l+1} = \hat{f}_l(X_l) + X_l \hat{W}_l$

**Low-Rank Approximation**: Applying SVD to $\hat{W} \in \mathbb{R}^{d \times d}$ keeps the top $r$ principal components ($r/d = 0.03$), reducing parameter size from $d^2$ to $2dr$, which constitutes only about 1% of the FFN layer parameters.

**Layer Selection Criteria**: FFN layers amenable to linear recovery are selected for calibration based on the Multiple Correlation Coefficient $R_{XE}$:

$$R_{XE} = \frac{1}{d} \sum_{i=1}^{d} R_i$$

where $R_i$ is the Pearson correlation coefficient between the predicted and true residuals. Experiments indicate that FFN residuals in shallower blocks are more suited for linear calibration.

### 4. RoPE Compatibility

Since LLaMA employs RoPE positional embeddings, there is no direct matrix multiplication between $W_q$ and $W_k$. The authors resolve this by performing Weighted SVD on $W_q$ and $W_k$ separately with the objective:

$$\arg\min_{W_1, W_2} \|(W - W_1 W_2) D\|_2^2$$

where $D = \text{diag}(\|x^{(1)}\|_2, \ldots, \|x^{(d)}\|_2)$, multiplying weights with input feature magnitudes.

## Key Experimental Results

### Resource Consumption Comparison (LLaMA-7B)

| Method | Data | Time | VRAM | PPL↓ (25%) | Acc↑ (25%) | PPL↓ (33%) | Acc↑ (33%) |
|------|--------|------|------|------------|------------|------------|------------|
| LLM-Pruner | 50K | 3h | 30GB | 20.57 | 58.67 | 24.50 | 55.39 |
| SlimGPT | 50K | 1h | 20GB | 18.45 | 62.45 | 22.43 | 61.41 |
| **Olica** | **256** | **7min** | **3GB** | **16.69** | **63.53** | **19.83** | **61.21** |

### Main Results for LLaMA-1 Series

| Model | Sparsity | Method | Retraining | PPL↓ | Avg. Acc (7 tasks)↑ |
|------|--------|------|--------|------|-----------|
| LLaMA-7B | 20% | Olica | ✗ | **15.35** | **64.54** |
| LLaMA-7B | 25% | Olica | ✗ | **16.69** | **63.54** |
| LLaMA-7B | 25% | SlimGPT | ✗ | 19.11 | 62.47 |
| LLaMA-7B | 25% | LoRAP | ✗ | 17.40 | 62.57 |
| LLaMA-7B | 33% | Olica | ✗ | **19.83** | **61.21** |
| LLaMA-7B | 33% | SlimGPT | ✗ | 24.55 | 60.37 |
| LLaMA-13B | 20% | Olica | ✗ | **13.68** | **67.67** |
| LLaMA-13B | 20% | LoRAP | ✗ | 13.84 | 66.84 |

### LLaMA-2 / Vicuna Results

| Model | Sparsity | Method | PPL↓ | Avg. Acc↑ |
|------|--------|------|------|-------|
| LLaMA-2-7B | 30% | Olica | **18.54** | **61.14** |
| LLaMA-2-7B | 30% | LoRAP | 19.42 | 58.89 |
| Vicuna-7B | 20% | Olica | **20.23** | **64.88** |
| Vicuna-7B | 20% | LoRAP | 20.74 | 64.39 |

### Inference Efficiency (LLaMA-7B, RTX 4090)

| Sparsity | Params | MACs | VRAM | Inference Latency |
|--------|--------|------|------|----------|
| 0% | 6.74B | 424.02G | 12884 MiB | 46.95s |
| 20% | 5.39B | 373.23G | 10464 MiB | 40.62s |
| 33% | 4.52B | 339.53G | 8718 MiB | 35.78s |

### Ablation Study (LLaMA-7B, 33% Sparsity)

| MHA Method | Linear Calibration | PPL↓ | Acc↑ |
|---------|---------|------|------|
| SVD | ✗ | 71.01 | 47.62 |
| Wanda | ✗ | 20.94 | 59.82 |
| Fast-OND | ✗ | 20.34 | 60.68 |
| Fast-OND | ✓ | **19.83** | **61.21** |

### Runtime Comparison (LLaMA-7B, 20%)

| Method | Runtime | PPL | Acc |
|------|----------|-----|-----|
| OND (Standard SVD) | 2910s | 15.17 | 64.32 |
| Fast-OND | **413s** | 15.35 | 64.54 |

Fast-OND delivers an approximate **7× speedup** with almost no loss in performance.

## Highlights & Insights

1. **Treating matrix products as unified entities** is the core insight—operating directly in the parameter space avoids the need for data-driven retraining.
2. The discovery of **singular value distribution symmetry** ($W_q \sim W_k$ and $W_v \sim W_o$) enables Fast-OND, reducing complexity by $h^2$ times.
3. The **closed-form solution for linear calibration** elegantly leverages ridge regression and SVD low-rank approximation, adding only ~1% extra parameters.
4. Requiring only **256 samples**, the framework is robust to both sample size and sequence length (with PPL fluctuating by only 2.4 as sequence length varies from 8 to 2048).
5. The extremely low resource requirement (3GB VRAM / 7 minutes) makes it viable to execute LLM pruning even on edge devices.

## Limitations & Future Work

1. **Severe performance drop at high sparsity**: At 50% sparsity, all methods experience a substantial degradation (Olica on LLaMA-7B yields only 50.68% accuracy), demonstrating that retraining-free methods have a lower performance ceiling in this regime.
2. **Evaluation limited to the LLaMA family**: The method has not been tested on non-LLaMA architectures like Mistral, Qwen, and Phi; thus, generalizability remains to be verified.
3. **RoPE handling is a workaround**: Because RoPE disrupts the direct product structure of $W_q W_k^\top$, $W_q$ and $W_k$ can only undergo separate weighted SVDs rather than a joint decomposition, which compromises the theoretical elegance of MHA compression.
4. **Limitations of the linear calibration assumption**: Reconstructing residuals solely with a linear model provides limited capacity to recover from non-linear error patterns.
5. **Lack of joint experiments with quantization**: Pruning and quantization joint compression is a mainstream choice in industry but is not addressed in this paper.
6. **Insufficient theoretical support for the layer selection criterion**: The MC² threshold must be manually chosen (from {6, 12, 16}) and lacks an adaptive selection mechanism.

## Related Work & Insights

- **LLM-Pruner (NeurIPS'23)**: Pioneering work introducing a two-stage process (pruning + LoRA retraining).
- **SliceGPT (ICLR'24)**: Reduces hidden representations using PCA, but operates in the activation space rather than the parameter space.
- **LoRAP (ICML'24)**: Performs low-rank approximation on QKV, but still requires retraining.
- **SlimGPT (NeurIPS'24)**: Extends OBS to structured pruning but requires extensive data.
- **Wanda (ICLR'24)**: Proposes weight-by-activation-magnitude importance scores, which Olica directly adopts.

The core inspiration of Olica: **Orthogonal decomposition in the parameter space can bypass the need for data-dependent retraining.** This approach could potentially be extended to other model compression scenarios (e.g., ViT, diffusion models).

## Rating
- Novelty: ⭐⭐⭐⭐ — The perspective of integrating matrix products as unified entities and applying PCA is highly novel, and the observation of singular value symmetry in Fast-OND is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablation studies and evaluations across multiple models and sparsities are provided, though evaluations on non-LLaMA architectures and joint quantization experiments are missing.
- Writing Quality: ⭐⭐⭐⭐ — The paper is clearly presented, mathematically rigorous, and features highly informative plots and tables.
- Value: ⭐⭐⭐⭐⭐ — Extremely practical. The remarkably low overhead of 3GB VRAM and 7 minutes runtime makes it a prime implementation for resource-constrained scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] SlimLLM: Accurate Structured Pruning for Large Language Models](slimllm_accurate_structured_pruning_for_large_language_models.md)
- [\[ICML 2025\] Instruction-Following Pruning for Large Language Models](instruction-following_pruning_for_large_language_models.md)
- [\[NeurIPS 2025\] Elastic ViTs from Pretrained Models without Retraining](../../NeurIPS2025/model_compression/elastic_vits_from_pretrained_models_without_retraining.md)
- [\[ACL 2025\] BlockPruner: Fine-grained Pruning for Large Language Models](../../ACL2025/model_compression/blockpruner_fine-grained_pruning_for_large_language_models.md)
- [\[ACL 2026\] GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](../../ACL2026/model_compression/grasprune_global_gating_for_budgeted_structured_pruning_of_large_language_models.md)

</div>

<!-- RELATED:END -->
