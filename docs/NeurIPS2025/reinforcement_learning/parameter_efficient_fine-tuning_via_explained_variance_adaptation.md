---
title: >-
  [Paper Note] Parameter Efficient Fine-tuning via Explained Variance Adaptation
description: >-
  [NeurIPS 2025][Reinforcement Learning][Parameter-efficient fine-tuning] This paper proposes Explained Variance Adaptation (EVA), which initializes LoRA matrices via incremental SVD on activation vectors from downstream data, provably maximizing the expected gradient signal. Combined with an adaptive rank allocation mechanism, EVA establishes a new accuracy–efficiency Pareto frontier across language generation/understanding, image classification, and reinforcement learning.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Parameter-efficient fine-tuning
  - LoRA
  - singular value decomposition
  - adaptive rank allocation
  - variance-optimal initialization
date: 2026-05-08
content_hash: c73bc0b62016a6f7
---

# Parameter Efficient Fine-tuning via Explained Variance Adaptation

**Conference**: NeurIPS 2025
**arXiv**: [2410.07170](https://arxiv.org/abs/2410.07170)
**Authors**: Fabian Paischer (JKU Linz), Lukas Hauzenberger (JKU Linz), Thomas Schmied, Benedikt Alkin, Marc Peter Deisenroth (UCL), Sepp Hochreiter (JKU Linz)
**Code**: Integrated into the HuggingFace PEFT library
**Area**: Reinforcement Learning
**Keywords**: Parameter-efficient fine-tuning, LoRA, singular value decomposition, adaptive rank allocation, variance-optimal initialization

## TL;DR

This paper proposes Explained Variance Adaptation (EVA), which initializes LoRA matrices via incremental SVD on activation vectors from downstream data, provably maximizing the expected gradient signal. Combined with an adaptive rank allocation mechanism, EVA establishes a new accuracy–efficiency Pareto frontier across language generation/understanding, image classification, and reinforcement learning.

## Background & Motivation

### State of the Field

Foundation models are typically pre-trained on large-scale data and then fine-tuned on specific downstream tasks. As model parameter counts grow, full fine-tuning (FFT) becomes prohibitively expensive. LoRA addresses this by introducing low-rank decomposition $\Delta W = BA$, and has become the dominant PEFT method.

### Limitations of Prior Work

- **Random initialization (vanilla LoRA)**: $A$ is randomly initialized and $B = 0$, ignoring all data and weight information, leading to slow convergence.
- **Weight-driven initialization (PiSSA/OLoRA/MiLoRA)**: Based on SVD of pre-trained weight matrices, these methods do not account for the activation distribution of the downstream task.
- **Data-driven initialization (LoRA-GA/CorDA)**: These leverage gradients or input–output correlations but cannot provably maximize the expected gradient signal; initialization overhead is also substantial (LoRA-GA requires 56.95 GB VRAM + 2.4% of training time; CorDA requires 55.64 GB + 4.5%).
- **Adaptive rank**: AdaLoRA dynamically adjusts ranks during training, increasing training complexity; no existing method unifies data-driven initialization with rank allocation.

### Root Cause

The paper seeks an LoRA initialization scheme that simultaneously **provably maximizes gradient signal** and **adaptively allocates rank budgets**, with negligible initialization overhead.

## Method

### Core Idea: Variance-Optimal Initialization

For a pre-trained weight matrix $W \in \mathbb{R}^{k \times d}$, EVA applies incremental SVD to downstream activation vectors $X \in \mathbb{R}^{b \times d}$, extracting the right singular vectors $V_{:r,:}$ that capture the maximum activation variance, and uses them to initialize the $A$ matrix.

**Theorem 3.1** (Variance Optimality): Given the SVD $X = U\Sigma V^\top$, the leading $r$ right singular vectors $V_{:r}$ solve the following optimization problem:

$$V_r = \arg\max_{V \in \mathbb{R}^{d \times r}, V^\top V = I} \text{Tr}(V^\top X^\top X V)$$

By the Eckart–Young theorem, these vectors simultaneously minimize the Frobenius norm reconstruction error, providing the optimal rank-$r$ basis for capturing maximum activation variance.

### Gradient Signal Amplification

**Theorem 3.2**: Let $\Sigma = \mathbb{E}[xx^\top]$ denote the activation covariance matrix. Initializing $A$ with the leading $r$ right singular vectors of the activation matrix maximizes the squared expected gradient norm:

$$\mathbb{E}\left[\left\|\frac{\partial \mathcal{L}}{\partial B}\right\|_F^2\right] \propto \text{Tr}(A\Sigma A^\top)$$

Initializing along high-variance directions thus amplifies the gradient signal and accelerates convergence.

### Incremental SVD Procedure

1. For each target weight matrix $W^i$, activation vector batches $X^i$ are collected during a forward pass.
2. Incremental truncated SVD is applied (based on the Sequential Karhunen–Loeve algorithm), updating right singular vectors batch by batch.
3. Convergence is monitored via cosine similarity: $\cos(v_{j,:}^{i,t-1}, v_{j,:}^{i,t}) \geq \tau, \forall 1 \leq j \leq r$.
4. Upon convergence, $A^i = V_{:r,:}^i$ and $B = 0$ are set.

The time and memory complexity of this procedure is independent of dataset size and depends only on the truncation rank, making it applicable at arbitrary scale.

### Adaptive Rank Allocation

Using the explained variance provided by singular values, EVA reallocates per-layer ranks under a global rank budget $l = Nr$:

1. Compute the explained variance ratio for each component of each weight matrix: $\xi_j^i = \frac{(\sigma_j^i)^2}{(M-1)\|\sigma^i\|_1}$.
2. Normalize across weight matrices to ensure comparability.
3. Globally sort all components across all weight matrices by $\xi_j^i$ and select the top-$l$ components.
4. Determine the rank of each weight matrix based on the number of selected components.

A hyperparameter $\rho \in [1, \infty)$ controls the heterogeneity of rank distribution: $\rho = 1$ degenerates to uniform rank (standard LoRA), while $\rho > 2$ yields a converged rank allocation. In practice, rank is typically redistributed from high-dimensional feed-forward layers toward lower-dimensional attention layers, reducing total trainable parameters.

### Theoretical Connection to NTK

At the fine-tuning initialization point, assuming weak correlation between activations and upstream gradients and approximately isotropic upstream gradients, EVA approximates the principal subspace of the Neural Tangent Kernel (NTK). Given the NTK generalization error $\varepsilon_{\text{gen}} \propto \sum_i (u_i^\top y)^2 / \lambda_i^2$, initializing along the principal NTK eigendirections minimizes the spectral tail of the generalization error.

## Key Experimental Results

### Experiment 1: Commonsense Reasoning + Mathematical Reasoning (Language Generation)

Five LLMs (Llama-2-7B, Llama-3.1-8B/70B, Gemma-2-9B/27B) are fine-tuned with $r=16$ on eight commonsense reasoning benchmarks:

| Method | Initialization Type | Adaptive Rank | Avg. Performance Trend | Parameter Count |
|--------|--------------------|--------------|-----------------------|----------------|
| LoRA | Random | ✗ | Baseline | 100% |
| PiSSA | Weight-driven | ✗ | ≈ LoRA | 100% |
| OLoRA | Weight-driven | ✗ | ≈ LoRA | 100% |
| LoRA-GA | Data-driven | ✗ | ≈ LoRA; fails to scale to 70B | 100% |
| CorDA | Data-driven | ✗ | Seed-sensitive; training collapse | 100% |
| **EVA** | **Data-driven** | **Yes** | **Highest avg. across all models** | **Reduced by 15M+** |

- On Llama-3.1-70B, EVA achieves an average score of 94.5 (highest), while reducing trainable parameters by over 15M.
- On math tasks (MetaMathQA → MATH/GSM8K): EVA achieves the highest score on Gemma-2-9B GSM8K and is on par with or ahead of baselines on other models.
- Convergence speed: EVA exhibits the largest gradient norm and fastest training loss decrease on Llama-3.1-8B.

### Experiment 2: Language Understanding — GLUE Benchmark

RoBERTa-Large performance on 8 GLUE tasks (mean ± std):

| Method | MNLI | QNLI | QQP | SST2 | CoLA | MRPC | RTE | STS-B | **Avg** |
|--------|------|------|-----|------|------|------|-----|-------|---------|
| FFT | 90.2 | 94.7 | **92.2** | **96.4** | 68.0 | 90.9 | 86.6 | 92.4 | 88.9 |
| LoRA | 90.7 | 94.8 | 92.0 | 96.2 | 69.1 | 91.1 | 88.1 | 92.3 | 89.3 |
| AdaLoRA | 90.5 | 94.8 | 90.6 | 96.1 | 68.2 | 90.7 | 84.4 | 91.8 | 88.4 |
| PiSSA | 90.1 | 94.7 | 91.0 | 96.1 | 68.7 | 90.4 | 87.6 | 92.5 | 88.9 |
| CorDA | 89.3 | 92.6 | 89.7 | 95.5 | 67.8 | 90.1 | 86.5 | 91.8 | 87.9 |
| **EVA** | **90.8** | **95.0** | 92.1 | 96.2 | **69.5** | **91.4** | **88.8** | **92.6** | **89.6** |

On DeBERTav3-Base, EVA likewise achieves the highest average of 89.9, with particularly notable advantages on low-resource tasks (RTE 89.4, MRPC 91.8, CoLA 72.5). Rank allocation analysis shows that more rank is assigned to Q/K/V projections in the upper attention layers.

### Experiment 3: Initialization Efficiency Comparison

Overhead of data-driven initialization methods for Llama-2-7B on a single A100:

| Method | Batch Size | Peak VRAM (GB) | Fraction of Training Time |
|--------|-----------|---------------|--------------------------|
| LoRA-GA | 8 | 56.95 | 2.4% |
| CorDA | 1 | 55.64 | 4.5% |
| **EVA** | **16** | **32.85** | **0.7%** |
| **EVA** | **8** | **29.39** | **0.3%** |
| **EVA** | **4** | **27.51** | **0.2%** |

At maximum batch size, EVA incurs only 0.7% overhead; reducing batch size to 4 brings this to 0.2%. Component cosine similarities across different batch sizes in incremental SVD are highly consistent, confirming robustness to both batch order and batch size.

### Supplementary Experiments

- **Image Classification (VTAB-1K)**: DINOv2-g/14 (1.1B) achieves the highest average accuracy under EVA across 19 tasks, with the most significant gains on natural image tasks.
- **Reinforcement Learning (Meta-World)**: A 12M Decision Transformer fine-tuned on the CW10 benchmark shows EVA substantially narrows the gap between LoRA and FFT; EVA+DoRA achieves the highest average success rate.
- **Rank Allocation Ablation**: Rank allocation converges for $\rho > 2$; selecting the lowest-variance components (reverse selection) leads to a substantial performance drop, validating the necessity of variance optimality.

## Highlights & Insights

- **Theoretical Rigor**: Provable initialization guarantees are established via Theorem 3.1 (variance optimality) and Theorem 3.2 (gradient signal amplification), further connected to generalization error through the NTK framework, forming a complete theory–experiment loop.
- **Pareto Dominance**: Across 51 tasks spanning language (8+8 tasks), vision (19 tasks), and RL (10 tasks), EVA achieves the highest average performance with fewer trainable parameters, and is the only method that simultaneously realizes data-driven initialization and adaptive rank allocation.
- **Minimal Overhead and Engineering Practicality**: Incremental SVD complexity is independent of dataset size; initialization requires only 0.2% of training time. Integration into the HuggingFace PEFT library enables plug-and-play deployment.

## Limitations & Future Work

- **Low-rank Preference**: EVA performs best in low-rank settings such as $r=16$; for $r \geq 128$, incremental SVD overhead increases substantially, making weight-driven methods such as PiSSA more lightweight.
- **Requires a Static Dataset**: Initialization depends on forward passes over downstream data to collect activations, making it inapplicable in streaming data scenarios or settings without a well-defined downstream dataset.
- **No Free Lunch**: Per-task rankings fluctuate — FFT outperforms on structured images and LoRA on specialized images; EVA's advantage manifests primarily in multi-task averages.
- **Gradient Information Not Utilized**: The current method exploits only activation variance and does not incorporate gradient direction information (explicitly noted by the authors as a future direction).
- **Fixed Rank Allocation Pattern**: Rank is consistently redistributed from high-dimensional FFN layers to lower-dimensional attention layers, without adaptive analysis of architectural differences across model families.

## Related Work & Insights

- **LoRA (Hu et al., 2022)**: Randomly initializes $A$ with fixed rank; EVA provides variance-optimal initialization and adaptive rank on top of this, comprehensively surpassing LoRA in average performance.
- **PiSSA / OLoRA**: SVD-based initialization from weight matrices, ignoring downstream data distribution; performance falls between LoRA and EVA; PiSSA occasionally remains competitive on CoLA.
- **LoRA-GA (Wang et al., 2024b)**: Gradient-based data-driven initialization with prohibitive overhead (56.95 GB / 2.4%); fails to scale to 70B+ models.
- **CorDA (Yang et al., 2024)**: Input–output correlation-based initialization; seed-sensitive (training collapse observed); highest overhead (55.64 GB / 4.5%); cannot scale to large models.
- **AdaLoRA (Zhang et al., 2023)**: Dynamically adjusts rank during training with random initialization; GLUE average 88.4 vs. EVA's 89.6; increases training complexity.
- **DoRA (Liu et al., 2024)**: Decomposes weight magnitude and direction; combinable with EVA — EVA+DoRA achieves the best results on RL tasks, though DoRA alone underperforms LoRA on high-resource tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The unified framework of variance-optimal initialization and adaptive rank allocation is original, though incremental SVD and PCA-based initialization are not entirely novel concepts.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 51 tasks across four domains (language, vision, RL); ablations are comprehensive (rank, learning rate, alpha, reverse variance); efficiency analysis is thorough.
- **Writing Quality**: ⭐⭐⭐⭐ — Theory–method–experiment structure is clear; notation is consistent; citations are complete and contextually appropriate.
- **Value**: ⭐⭐⭐⭐⭐ — Integrated into the HuggingFace PEFT library; establishes a new standard for low-rank fine-tuning scenarios with strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Continual Knowledge Adaptation for Reinforcement Learning](continual_knowledge_adaptation_for_reinforcement_learning.md)
- [\[NeurIPS 2025\] Prompt Tuning Decision Transformers with Structured and Scalable Bandits](prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)
- [\[NeurIPS 2025\] Parameter-Free Algorithms for the Stochastically Extended Adversarial Model](parameter-free_algorithms_for_the_stochastically_extended_adversarial_model.md)
- [\[NeurIPS 2025\] Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits](variance-aware_feel-good_thompson_sampling_for_contextual_bandits.md)
- [\[NeurIPS 2025\] EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)

<!-- RELATED:END -->
