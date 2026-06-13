---
title: >-
  [Paper Note] DMin: Scalable Training Data Influence Estimation for Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Diffusion Models] Proposes DMin, a scalable training data influence estimation framework for diffusion models. By using an efficient gradient compression pipeline…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Influence Estimation"
  - "Gradient Compression"
  - "KNN"
  - "Scalability"
date: 2026-05-08
content_hash: c1cd28729b16a7b7
---

# DMin: Scalable Training Data Influence Estimation for Diffusion Models

**Conference**: CVPR 2026 Findings  
**arXiv**: [2412.08637](https://arxiv.org/abs/2412.08637)  
**Code**: Available (Pytorch implementation with multi-process support will be open-sourced)  
**Area**: Image Generation / Model Interpretability  
**Keywords**: Diffusion Models, Influence Estimation, Gradient Compression, KNN, Scalability

## TL;DR

Proposes DMin, a scalable training data influence estimation framework for diffusion models. By using an efficient gradient compression pipeline, it reduces storage requirements from hundreds of terabytes down to MB/KB levels, enabling influence estimation for billion-parameter diffusion models for the first time and supporting sub-second top-k retrieval.

## Background & Motivation

Understanding "which training data points most influence a generated image" is crucial for model transparency, bias analysis, and copyright attribution. Existing influence estimation methods face three major bottlenecks:

**Model Scalability**: Second-order methods (DataInf, K-FAC) require Hessian inverse approximations, leading to exploding memory requirements for large models. For Stable Diffusion 3 Medium (2B parameters), caching gradients for a single sample over 10 steps would require 80 GB; 10,000 samples would require 800 TB.

**Excessive Projection Matrices**: First-order methods (D-TRAK, Journey-TRAK) use random projections for dimensionality reduction, but a projection matrix for 2B parameters $\times$ 32,768 dimensions requires 238 TB of storage.

**Gradient Instability**: Gradient values in deep models can be extremely large, causing inner product calculations to be dominated by outliers.

Consequently, existing methods can only be applied to LoRA fine-tuning or small diffusion models and cannot handle large models trained with full parameters.

## Method

### Overall Architecture

The core workflow of DMin (Fig. 2) is divided into two stages:
1. **Gradient Calculation and Compression**: For each training sample, gradients are calculated at sampled timesteps, then normalized, compressed, and cached.
2. **Influence Estimation**: Compressed gradients are calculated for the generated image, and influence scores are estimated via inner products or KNN retrieval.

The mathematical foundation for influence estimation is the first-order Taylor expansion approximation of loss change:

$$\mathcal{I}_\theta(X^s, X^i) = e\bar{\eta} \sum_{t=1}^{T} \nabla_\theta \mathcal{L}(f_\theta(z^i_p, z^i_t, t), \epsilon) \cdot \nabla_\theta \mathcal{L}(f_\theta(z^s_p, z^s_t, t), \epsilon)$$

This represents the sum of the inner products of loss gradients between training samples and generated samples across various timesteps.

### Key Designs

1. **Efficient Gradient Compression (Four-Step Compression Pipeline)**:

    - **Function**: Compresses the $2 \times 10^9$-dimensional gradient vector to $v$ dimensions ($v$ can be as low as $2^{12}=4096$).
    - **Mechanism**: (1) Padding to a multiple of $v$ → (2) Random permutation to break structure → (3) Element-wise multiplication by a random $\pm 1$ vector projection → (4) Dimensionality reduction via group summation.
    - **Design Motivation**: Random permutation + random projection ensures the Johnson-Lindenstrauss property, while group aggregation achieves extremely high compression ratios. Unlike traditional random projections, this does not require storing huge projection matrices—only a permutation vector (4 bytes/element) and a binary projection vector (1 bit/element).

2. **L2 Normalization**:

    - **Function**: Performs L2 normalization on gradient vectors before compression.
    - **Mechanism**: Eliminates the dominant effect of abnormally large gradient values.
    - **Design Motivation**: Experiments found that detection rates drop sharply without normalization (SD 1.4 LoRA Flowers Top-5: 0.887 → 0.133), confirming that gradient instability in deep models is a core obstacle for influence estimation.

3. **KNN Index Acceleration**:

    - **Function**: Builds an HNSW index after concatenating compressed gradients from different timesteps.
    - **Mechanism**: Replaces exhaustive inner product calculations with approximate nearest neighbor search.
    - **Design Motivation**: Enables sub-second top-k retrieval. Interestingly, KNN retrieval often outperformed exact inner product calculations in experiments, possibly due to an implicit regularization effect of the approximate search.

4. **Timestep Sampling**:

    - Sub-sampling the full 1000-step diffusion process (e.g., taking 5-10 steps) significantly reduces the computational and storage burden.
    - This is similar to step scheduling strategies used during diffusion model inference.

### Loss & Training

DMin itself does not train a model but performs post-analysis on pre-trained diffusion models. Key operations:
- Gradient Collection: For LoRA models, only gradients of adapter parameters are collected; for full-parameter models, gradients of all parameters are collected.
- Gradient collection for full-parameter SD3 Medium (2B parameters) costs approximately 330 GPU hours.
- KNN index construction takes only a few minutes.

## Key Experimental Results

### Main Results (Detection Rate for Conditional Diffusion Models)

On a mixed dataset (9,288 samples including Flowers/Lego/Magic Cards subsets), SD1.4 LoRA, SD3 Medium LoRA, and SD3 Medium Full were fine-tuned to retrieve the most relevant training samples for generated images:

| Method | Flowers Top-5 | Flowers Top-10 | Magic Cards Top-5 | Applicable Models |
|------|--------------|----------------|-------------------|---------|
| Random | 0.000 | 0.000 | 0.200 | Any |
| CLIP Similarity | 0.000 | 0.000 | 0.444 | Any |
| LiSSA | 0.514 | 0.457 | 0.967 | Small/LoRA |
| DataInf | 0.413 | 0.406 | 0.967 | Small/LoRA |
| DMin ($v=2^{16}$) | **0.862** | **0.823** | **0.978** | Any Scale |
| DMin (SD3 Full, $v=2^{16}$) | **0.959** | **0.931** | **0.996** | 2 Billion Params |

On SD3 Medium Full, LiSSA/DataInf/D-TRAK were **completely unable to run** due to requiring hundreds of TBs of cache, while DMin was the only feasible method.

### Storage and Speed Comparison

| Method | SD3 Full Storage per Sample | Total Dataset Storage | Gain (Compression) |
|------|-------------------|-------------|--------|
| Uncompressed Gradient | 37.42 GB | 339.39 TB | 100% |
| DMin ($v=2^{12}$) | 80 KB | 726 MB | 0.00017% |
| DMin ($v=2^{16}$) | 1.25 MB | 11.34 GB | 0.0028% |

| Method | SD3 LoRA Time/Test Sample | Gain (Speedup) |
|------|---------------------|--------|
| LiSSA | 2136.7s | 0.19x |
| DataInf (Hessian) | 932.8s | 0.44x |
| DMin ($v=2^{12}$, KNN top-5) | **0.004s** | **101,878x** |

### Key Findings

1. **Compression is Nearly Lossless**: The detection rate difference between $v=2^{16}$ compressed gradients and uncompressed gradients is less than 1%.
2. **Normalization is Crucial**: Performance plummeted without normalization, confirming that gradient instability in large models is a core problem.
3. **KNN Slightly Outperforms Exact Calculation**: Likely due to the regularization effect of approximate search.
4. Completed influence estimation on full-finetuned SD3 with 2B parameters for the first time; all other methods were infeasible.
5. On unconditional DDPM (MNIST), DMin achieved a Top-5 detection rate of 0.80, far exceeding Journey-TRAK (0.26) and D-TRAK (0.13).

## Highlights & Insights

- **Outstanding Engineering Contribution**: Compressed the storage requirement from 339 TB to 726 MB (a compression ratio of 0.00017%), making previously impossible tasks feasible.
- **Clever Design of the Four-Step Gradient Compression Pipeline**: The combination of permutation, random projection, and group summation avoids storing massive projection matrices while maintaining the distance-preserving properties of the JL Lemma.
- **Universal Significance of L2 Normalization**: Revealed the fundamental impact of gradient instability in large models on gradient-based analytical methods.
- The counter-intuitive conclusion that KNN retrieval outperforms exact calculation warrants further research.

## Limitations & Future Work

1. The gradient collection phase still incurs high costs for full-parameter models (330 GPU hours), though it is a one-time investment.
2. Influence estimation is based on a first-order approximation, ignoring second-order interactions across timesteps.
3. Currently validated on relatively small datasets (~9K samples); practical application on million-scale training sets needs exploration.
4. Uses fixed Gaussian noise to approximate actual noise during training, which is theoretically biased.
5. Has not yet been validated on even larger models like FLUX or SORA.

## Related Work & Insights

- **TRAK / D-TRAK / Journey-TRAK**: First-order random projection methods, but limited in scale by the size of the projection matrix.
- **DataInf / K-FAC**: Second-order Hessian approximation methods that require full gradient loading.
- **Vector Compression Literature**: Inspired the design of the DMin compression pipeline.
- The logic in this paper can be extended to training data provenance and data contamination detection in LLMs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The gradient compression pipeline is cleverly designed, though the core idea (influence via gradient inner products) is an engineering extension of existing frameworks.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across three model scales, multiple subsets, and storage/time/accuracy dimensions with thorough ablation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and complete derivations, though some excessively long tables hinder readability.
- **Value**: ⭐⭐⭐⭐⭐ — Extends influence estimation to billion-parameter diffusion models for the first time, with significant practical value for model auditing and data copyright.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](../../AAAI2026/image_generation/diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)
- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](../../ICLR2026/image_generation/learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](../../ICML2026/image_generation/guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[CVPR 2026\] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)
- [\[AAAI 2026\] Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](../../AAAI2026/image_generation/difficulty_controlled_diffusion_model_for_synthesizing_effec.md)

</div>

<!-- RELATED:END -->
