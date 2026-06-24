---
title: >-
  [Paper Note] Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models
description: >-
  [ICML2025][LLM Safety][Federated Learning] This paper proposes Ferret, the first federated full-parameter fine-tuning method that combines first-order optimization with shared randomness. By projecting local updates into low-dimensional spaces, Ferret achieves $10^6\times$ communication compression and $6\times$ computational acceleration while maintaining model accuracy comparable to FedAvg.
tags:
  - "ICML2025"
  - "LLM Safety"
  - "Federated Learning"
  - "Full-parameter Fine-tuning"
  - "Communication Compression"
  - "Shared Randomness"
  - "LLMs"
  - "Projection Reconstruction"
date: 2026-05-08
content_hash: 3bc09e39e3a9713c
---

# Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models

**Conference**: ICML2025  
**arXiv**: [2409.06277](https://arxiv.org/abs/2409.06277)  
**Code**: [allen4747/Ferret](https://github.com/allen4747/Ferret)  
**Area**: AI Safety  
**Keywords**: Federated Learning, Full-parameter Fine-tuning, Communication Compression, Shared Randomness, LLMs, Projection Reconstruction

## TL;DR

This paper proposes Ferret, the first federated full-parameter fine-tuning method that combines first-order optimization with shared randomness. By projecting local updates into low-dimensional spaces, Ferret achieves $10^6\times$ communication compression and $6\times$ computational acceleration while maintaining model accuracy comparable to FedAvg.

## Background & Motivation

- **Key Challenge**: When performing federated full-parameter fine-tuning on LLMs, federated learning (FL) must achieve a balance among data privacy, communication efficiency, and model accuracy.
- **Limitations of PEFT**: Although parameter-efficient fine-tuning (PEFT, such as LoRA) reduces communication overhead, it only updates a subset of parameters, failing to fully capture subtle differences in local data distributions and thus leading to accuracy degradation.
- **Limitations of Zeroth-Order Methods**: Zeroth-order optimization methods like FedKSeed reduce communication volume to $\mathcal{O}(K)$ by transmitting scalar gradients, but suffer from three major issues:
  1. High computational cost—estimating gradients requires $K$ forward passes per round.
  2. Slow convergence—requiring significantly more communication rounds.
  3. Biased gradient estimation—errors accumulate with local iterations $T$.
- **Limitations of FedAvg**: First-order methods are computationally efficient and converge fast, but their communication overhead is $\mathcal{O}(d)$ (where $d$ is the number of parameters, often in the billions), making them impractical for LLMs.

**Core Problem**: Can a method be designed to simultaneously incorporate the computational efficiency and fast convergence of first-order methods, and the low communication overhead of zeroth-order methods?

## Method

### Overall Architecture

Ferret repeats three steps in each communication round $r \in [R]$:

**Step ①: Global Aggregation**

Each client receives the random seeds $s^{(i)}$ and projection coordinates $\{\gamma_k^{(i)}\}_{k=1}^K$ from other clients, rejuvenates the random bases $\{\mathbf{v}_k^{(i)}\}_{k=1}^K$ using shared randomness, reconstructs local updates, and aggregates the global model:

$$\mathbf{w}_{r-1} \leftarrow \mathbf{w}_{r-2} - \frac{1}{N} \sum_{i \in [N]} \widetilde{\Delta}_{r-1}^{(i)}, \quad \widetilde{\Delta}_{r-1}^{(i)} \triangleq \sum_{k \in [K]} \gamma_k^{(i)} \mathbf{v}_k^{(i)}$$

**Step ②: Local Update (First-Order Optimization)**

Each client performs $T$ steps of local updates using standard gradient descent:

$$\mathbf{w}_{r,t}^{(j)} \leftarrow \mathbf{w}_{r,t-1}^{(j)} - \eta \nabla \ell(\mathbf{w}_{r,t-1}^{(j)}; \mathbf{x}_{t-1}^{(j)})$$

Unlike zeroth-order methods that require hundreds of steps, first-order methods leverage more precise gradient information, achieving equivalent local update effects with very few iteration steps (e.g., $T=10$).

**Step ③: Projection Update (Dimension-Reduced Transmission)**

The local update $\Delta_r^{(j)} = \mathbf{w}_{r-1}^{(j)} - \mathbf{w}_r^{(j)}$ is calculated and projected onto $K$-dimensional coordinates:

$$\boldsymbol{\gamma} \approx (\rho K)^{-1} \mathbf{V}^\top \Delta$$

where $\rho$ is the correction factor for the truncated normal distribution to ensure unbiased reconstruction. Only the seed $s^{(j)}$ and $K$ scalars are transmitted, reducing the communication volume from $\mathcal{O}(d)$ to $\mathcal{O}(K)$.

### Key Designs

**Choice of Random Bases**: Sampling from a truncated normal distribution $v \sim \mathcal{N}(0,1)$, $v \in [-1/\sqrt{d}, 1/\sqrt{d}]$ ensures $\|\mathbf{v}_k\| \leq 1$, enabling full-parameter updates while maintaining numerical stability.

**Inversion-Free Reconstruction**: Directly approximating $\mathbf{V}^\top\mathbf{V} \approx \mathbf{I}_K$ avoids the $\mathcal{O}(K^2d + K^3)$ computational overhead of matrix inversion, reducing it to $\mathcal{O}(Kd)$.

**Block-wise Reconstruction**: Splitting the $d$-dimensional parameters into $L$ blocks, where each block is independently projected and reconstructed, further reduces the computational complexity by $1/L$ and scales down the memory complexity to $\mathcal{O}(\max\{K_l, d_l\})$.

### Theoretical Guarantees

- **Unbiased Reconstruction** (Theorem 1): $\mathbb{E}[\widetilde{\Delta}] = \Delta$, avoiding the estimation bias of zeroth-order methods.
- **Reconstruction Error** (Theorem 2): The error rate is $\widetilde{\mathcal{O}}(d/K)$, which decreases linearly as $K$ increases and does not accumulate with local iteration steps $T$.
- **Convergence** (Theorem 4): The communication round complexity is $\mathcal{O}(1/\epsilon^2)$, which is asymptotically equivalent to standard SGD and independent of the parameter dimension $d$.

## Key Experimental Results

### Accuracy Comparison (Rouge-L %)

| Method | NI (DataJuicer-1.3B) | NI (LLaMA-3B) | Dolly (DataJuicer-1.3B) | Dolly (LLaMA-3B) |
|------|------|------|------|------|
| FedIT (PEFT) | 22.30 | 28.13 | 30.80 | 33.23 |
| FedZO | 21.74 | 29.46 | 26.99 | 31.67 |
| FedKSeed | 22.33 | 29.77 | **30.91** | **34.56** |
| FedAvg | **23.95** | **32.11** | 29.67 | 30.98 |
| **Ferret** | **24.99** | 30.03 | 30.63 | **34.57** |

### Experiments on Large Models (LLaMA2-7B / 13B)

| Method | CodeAlpaca (7B) | CodeAlpaca (13B) | GSM8K (7B) | GSM8K (13B) |
|------|------|------|------|------|
| FedKSeed | 8.33 | 10.70 | 28.26 | 33.67 |
| FedAvg | **15.41** | **14.68** | **38.30** | **39.82** |
| **Ferret** | 12.10 | 11.84 | 36.10 | 34.50 |

### Scalability Comparison (Overhead per Round on LLaMA-3B)

| Method | Local Update (s) | Global Aggregation (s) | Total (s) | Communication Volume (No. of Parameters) |
|------|------|------|------|------|
| FedKSeed | 56.9 | 123.8 | 180.7 | 8.2×10³ |
| FedAvg | 1.8 | 0.3 | 2.1 | 6.0×10⁹ |
| **Ferret** | **5.6** (10.2×↓ vs FedKSeed) | **24.7** (5.0×↓) | **30.3** (6.0×↓) | **7.8×10³** (10⁶×↓ vs FedAvg) |

### Overhead per Round on LLaMA2-7B

| Method | Total (s) | Communication Volume |
|------|------|------|
| FedKSeed | 627.0 | 8.2×10³ |
| FedAvg | 6.5 | 1.4×10¹⁰ |
| **Ferret** | **97.2** (6.5×↓ vs FedKSeed) | **6.4×10³** (10⁶×↓ vs FedAvg) |

Ferret converges in only 12 rounds on the NI dataset (compared to 40 rounds for FedKSeed), achieving a 3.3× reduction in the number of convergence rounds.

## Highlights & Insights

1. **Elegant Fusion of First-Order and Zeroth-Order Advantages**: Integrating first-order gradients to ensure computational efficiency and convergence speed while utilizing random projection and shared randomness for communication compression, yielding the best of both worlds.
2. **Theoretical Breakthrough in Unbiased Reconstruction**: Proving that the reconstruction remains unbiased under the approximation $\mathbf{V}^\top\mathbf{V} \approx \mathbf{I}_K$ and that the error does not accumulate over iterations—offering a fundamental advantage over the biased estimations of zeroth-order methods.
3. **Block-wise Reconstruction Strategy**: Reducing computational complexity by an additional $1/L$ factor, allowing the method to scale effectively to 7B and 13B models.
4. **High Practicality**: Fully compatible with arbitrary gradient optimizers (such as AdamW) and straightforward to integrate into existing LLM training pipelines.
5. **Enhanced Privacy**: Transmitting only the seeds and low-dimensional coordinates, providing superior privacy preservation compared to FedAvg, which transmits full gradients or parameters.

## Limitations & Future Work

1. **Accuracy Gap on Complex Tasks**: On complex tasks like CodeAlpaca and GSM8K, Ferret still underperforms compared to FedAvg (by a margin of 3-5%), suggesting that the information loss from projection reconstruction is more pronounced in challenging tasks.
2. **Higher Per-Round Computation than FedAvg**: Although significantly superior to FedKSeed, Ferret still requires 30s per round compared to 2.1s for FedAvg (approximately 14× slower), with the major overhead residing in the reconstruction step during global aggregation.
3. **Theoretical Analysis Limited to Homogeneous Settings**: The convergence analysis is only provided under the IID scenario where $\mathcal{L}^{(i)} = \mathcal{L}$, lacking rigorous guarantees in heterogeneous settings.
4. **Selection of Hyperparameter $K$**: A small $K$ leads to large reconstruction errors that degrade accuracy, while an excessively large $K$ diminishes the communication compression benefits, necessitating task-specific tuning.
5. **Unvalidated under Large-Scale Client Scenarios**: The experiments only evaluate a small number of active clients (5% sampling rate per round); performance under the scale of hundreds or thousands of clients remains unexplored.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to combine first-order optimization with shared randomness projection for federated full-parameter fine-tuning, featuring an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multiple models (1.3B to 13B) across various datasets, including scalability and ablation analyses, though large-scale client experiments are missing.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured paper with complementary theoretical formulation and empirical evaluations, along with intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Provides a scalable full-parameter framework for federated LLM fine-tuning, successfully balancing efficiency, communication, and accuracy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [\[CVPR 2025\] LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty](../../CVPR2025/llm_safety/lotus_large-scale_machine_unlearning_with_a_taste_of_uncertainty.md)
- [\[ICLR 2026\] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](../../ICLR2026/llm_safety/secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)
- [\[ICML 2025\] De-mark: Watermark Removal in Large Language Models](de-mark_watermark_removal_in_large_language_models.md)
- [\[ACL 2026\] Exploring Cross-Client Memorization of Training Data in Large Language Models for Federated Learning](../../ACL2026/llm_safety/exploring_cross-client_memorization_of_training_data_in_large_language_models_fo.md)

</div>

<!-- RELATED:END -->
