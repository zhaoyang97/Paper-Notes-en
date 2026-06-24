---
title: >-
  [Paper Note] Revisiting LoRA through the Lens of Parameter Redundancy: Spectral Encoding Helps
description: >-
  [ACL 2025][Model Compression][LoRA] This paper systematically investigates the parameter redundancy issue in LoRA fine-tuning, discovering that reducing density redundancy does not compromise expressiveness (sparsity property). The authors propose SeLoRA, which reparameterizes LoRA matrices from a sparse spectral subspace using spectral transformations (Fourier/Wavelet) to achieve superior performance with fewer parameters, while offering plug-and-play integration with variou…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "LoRA"
  - "parameter-efficient fine-tuning"
  - "spectral encoding"
  - "sparse learning"
  - "low-rank adaptation"
date: 2026-05-08
content_hash: ba1b5e3fe6d678e0
---

# Revisiting LoRA through the Lens of Parameter Redundancy: Spectral Encoding Helps

**Conference**: ACL 2025  
**arXiv**: [2506.16787](https://arxiv.org/abs/2506.16787)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: LoRA, parameter-efficient fine-tuning, spectral encoding, sparse learning, low-rank adaptation

## TL;DR

This paper systematically investigates the parameter redundancy issue in LoRA fine-tuning, discovering that reducing density redundancy does not compromise expressiveness (sparsity property). The authors propose SeLoRA, which reparameterizes LoRA matrices from a sparse spectral subspace using spectral transformations (Fourier/Wavelet) to achieve superior performance with fewer parameters, while offering plug-and-play integration with various LoRA variants.

## Background & Motivation

### Background
Low-Rank Adaptation (LoRA) is currently the mainstream method for fine-tuning large language models, approximating weight updates via the product of two low-rank matrices. However, recent studies have discovered significant parameter redundancy in LoRA, which limits its capacity and efficiency.

### Key Findings: Sparsity Property of LoRA
The authors systematically investigate the impact of redundancy in LoRA from two perspectives:

**Rank Redundancy**: Directly reducing the LoRA rank $\rightarrow$ significant performance degradation.

**Density Redundancy**: Masking a portion of parameters to zero at a fixed rank $\rightarrow$ even with 60% of parameters masked, the performance remains comparable to full LoRA.

This finding indicates that LoRA parameters are underutilized and have room for optimization. Core Problem: **How to leverage the sparsity property of LoRA to unleash its potential?**

### Motivation
While traditional pruning methods require complex strategies for LoRA fine-tuning, spectral encoding of weight matrices can achieve highly expressive representation learning using sparse spectral components, offering a simpler and more effective alternative.

## Method

### Overall Architecture
The core idea of SeLoRA (Spectral-encoding Low-Rank Adaptation) is to reparameterize the low-rank matrices A and B of LoRA as spatial-domain equivalents of sparse spectral components:

$$\mathbf{W}' = \mathbf{W}_0 + \tilde{\mathbf{B}} \tilde{\mathbf{A}}, \quad \tilde{\mathbf{A}} = \mathcal{T}(\mathbf{F}_A), \quad \tilde{\mathbf{B}} = \mathcal{T}(\mathbf{F}_B)$$

where $\mathcal{T}(\cdot)$ represents the inverse spectral transformation, and $\mathbf{F}_A, \mathbf{F}_B$ are sparse spectral matrices.

### Key Designs

1. **Sparsity ratio $\eta$**: Controls the proportion of masked parameters; the number of learnable spectral components is $\lfloor(1-\eta) \cdot rd\rfloor$.
2. **Global Shared Index Set $\Omega$**: Randomly initialized, specifying the learnable spectral locations shared across all low-rank matrices.
3. **Two Spectral Transformations**:

    - **Fourier Encoding (SeLoRA_F)**: Employs discrete inverse 2D Fast Fourier Transform, taking the real part to simplify computation; excels at capturing high-fidelity information from sparse components.
    - **Wavelet Encoding (SeLoRA_W)**: Employs discrete inverse 2D Wavelet Transform (Haar wavelet by default); provides localized and hierarchical information reconstruction, striking a better balance between smoothness and detail preservation.

4. **Initialization Strategy**: Uses Kaiming initialization on spectral matrices and calibrates variance using an auxiliary matrix to ensure consistency in spatial domain variance.
5. **Plug-and-Play**: Seamlessly integrates into other LoRA variants such as DoRA, X-LoRA, and HiRA.

### Efficiency Advantages
- Minimal computational overhead during training using Fast Spectral Transforms (FFT/FWT).
- Zero inference overhead (spectral components can be pre-converted back to spatial-domain weights).
- Significant parameter reduction (typically reduced to 40%-60% of LoRA's parameters).

## Key Experimental Results

### Table 1: Commonsense Reasoning Tasks (Average Accuracy on 8 Benchmarks)

| Method | Model | Params (%) | Train Time | BoolQ | PIQA | HellaS. | ARC-c | OBQA | **Avg** |
|------|------|-----------|----------|-------|------|---------|-------|------|----------|
| LoRA | LLaMA2-7B | 0.83 | 7.4h | 71.4 | 81.4 | 87.8 | 67.5 | 81.5 | 79.4 |
| SeLoRA_F | LLaMA2-7B | 0.50 | 7.6h | 72.8 | 83.4 | 90.9 | 70.5 | 83.4 | **81.3** (+1.9) |
| SeLoRA_W | LLaMA2-7B | 0.50 | 7.5h | 72.9 | 83.3 | 92.1 | 71.9 | 83.2 | **81.6** (+2.2) |
| DoRA | LLaMA2-7B | 0.84 | 12.2h | 71.8 | 83.1 | 90.1 | 69.5 | 82.4 | 80.1 |
| SeDoRA_W | LLaMA2-7B | 0.51 | 12.4h | 73.7 | 83.8 | 92.0 | 71.6 | 83.0 | **81.9** (+1.8) |
| LoRA | LLaMA3-8B | 0.70 | 7.8h | 74.0 | 88.2 | 94.0 | 78.1 | 84.0 | 84.0 |
| SeLoRA_W | LLaMA3-8B | 0.28 | 8.0h | 76.0 | 89.3 | 95.9 | 81.4 | 86.6 | **85.9** (+1.9) |
| SeDoRA_W | LLaMA3-8B | 0.28 | 13.0h | 76.2 | 89.7 | 96.0 | 82.0 | 87.8 | **86.5** (+1.3) |

### Table 2: Mathematical Reasoning and Code Generation

| Method | Model | Params (%) | GSM8k | MATH | Math Avg | HumanEval | MBPP | Code Avg |
|------|------|-----------|-------|------|----------|-----------|------|----------|
| LoRA | LLaMA2-7B | 0.83 | 60.5 | 11.7 | 36.1 | 32.1 | 35.8 | 31.8 |
| SeLoRA_W | LLaMA2-7B | 0.66 | 62.4 | 13.7 | **38.1** (+2.0) | 35.2 | 40.1 | **34.8** (+3.0) |
| SeDoRA_W | LLaMA2-7B | 0.67 | 63.0 | 14.1 | **38.6** (+1.9) | 33.5 | 41.0 | **34.8** (+1.1) |
| LoRA | LLaMA3-8B | 0.70 | 77.2 | 28.2 | 52.7 | 57.9 | 64.8 | 57.7 |
| SeLoRA_W | LLaMA3-8B | 0.42 | 80.3 | 29.8 | **55.1** (+2.4) | 59.3 | 66.1 | **59.4** (+1.7) |
| SeDoRA_W | LLaMA3-8B | 0.42 | 80.4 | 30.3 | **55.4** (+2.0) | 63.4 | 63.5 | **59.6** (+1.6) |

### Table 3: Robustness on Different Wavelet Bases (LLaMA3-8B)

| Wavelet Base | Commonsense Reasoning | Math | Code | Avg |
|--------|----------|------|------|------|
| LoRA (Baseline) | 83.9 | 52.7 | 57.6 | 64.7 |
| Haar | 85.9 | 55.1 | 59.4 | 66.8 |
| Daubechies-4 | 85.9 | 55.4 | 59.1 | 66.8 |
| Biorthogonal | 85.9 | 54.8 | 59.5 | 66.7 |
| Coiflets | 86.2 | 55.2 | 59.8 | **67.0** |

## Highlights & Insights

1. **Profound Redundancy Analysis**: Systematically distinguishes the differing impacts of rank redundancy and density redundancy, revealing the "sparsity property" of LoRA (no performance drop when masking 60% of parameters), which provides a clear theoretical motivation for subsequent method designs.
2. **Ingenious Spectral Reparameterization Design**: Leverages the natural sparse representation capability of Fourier/Wavelet transforms, efficiently utilizing the redundant parameter space without requiring complex pruning strategies.
3. **Excellent Plug-and-Play Characteristics**: Seamlessly integrates into various variants such as LoRA, DoRA, and HiRA, with Wavelet encoding consistently outperforming Fourier encoding in almost all scenarios.
4. **Extremely Parameter-Efficient**: Achieves an average commonsense reasoning gain of +1.9 with only ~40% of the parameters (e.g., 0.28% vs 0.70% on LLaMA3-8B).
5. **Data Efficiency**: Surpasses the performance of LoRA trained on 100% of the data using only 25% of the training data.
6. **Subspace Analysis**: Proves via amplification factor (AF) and reverse amplification factor (RAF) that SeLoRA more efficiently amplifies task-relevant features and suppresses redundant amplification of already emphasized features.

## Limitations & Future Work

1. **Diminishing Returns in High-Rank Scenarios**: As the rank increases, the improvement of SeLoRA relative to LoRA gradually diminishes, eventually converging to a similar upper bound, limited by LoRA's inherent capacity constraints.
2. **Limited to Variants with the Same Update Paradigm**: Currently only compatible with variants sharing the same update mode as LoRA (BA decomposition); integration with alternative update strategies like SVD has not been explored.
3. **Missing Evaluation on Very Large Models**: Due to computational resource constraints, scalability on 70B-scale models has not been verified.
4. **Spectral Location Selection**: The index set $\Omega$ is randomly initialized instead of adaptively selected, which may not be the optimal strategy.

## Related Work & Insights

- **LoRA Series**: LoRA (Hu et al., 2021), DoRA (Liu et al., 2024), HiRA (Huang et al., 2025), X-LoRA (Buehler & Buehler, 2024)
- **Sparse Reparameterization**: LS-LoRA (He et al., 2022), LoRETTA (Yang et al., 2024), FourierFT (Gao et al., 2024), LoRA-XS (Balazy et al., 2024)
- **Redundancy Analysis**: Parameter pruning (Han et al., 2015), Lottery Ticket Hypothesis (Frankle & Carlin, 2018)
- **Spectral Encoding**: Spectral encoding of weight matrices (Koutnik et al., 2010; Van Steenkiste et al., 2016; Wolter et al., 2020)

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | Ingeniously combines spectral encoding with the sparsity property of LoRA. The idea is novel, although spectral encoding itself is not a brand-new concept. |
| Experimental Thoroughness | 5 | Highly comprehensive, covering three major task types, two model scales, six LoRA variants, and multidimensional ablation analyses. |
| Writing Quality | 4 | Clear logic, with a natural and fluent transition from redundancy analysis to methodology design. |
| Value | 5 | Plug-and-play, zero inference overhead, fewer parameters with better performance, which offers high practical deployment value. |
| **Total** | **4.5** | A solid PEFT advancement paper that balances theoretical insights with engineering utility. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Faster Parameter-Efficient Tuning with Token Redundancy Reduction (FPET)](../../CVPR2025/model_compression/faster_parameter-efficient_tuning_with_token_redundancy_reduction.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[ICLR 2026\] LoRA-Mixer: Coordinate Modular LoRA Experts Through Serial Attention Routing](../../ICLR2026/model_compression/lora-mixer_coordinate_modular_lora_experts_through_serial_attention_routing.md)
- [\[ACL 2025\] FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](fedex_lora_federated_exact_aggregation.md)
- [\[ACL 2025\] CoLA: Collaborative Low-Rank Adaptation](cola_collaborative_low-rank_adaptation.md)

</div>

<!-- RELATED:END -->
