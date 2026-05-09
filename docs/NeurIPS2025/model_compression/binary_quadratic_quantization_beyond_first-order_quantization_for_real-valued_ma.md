---
title: >-
  [Paper Note] Binary Quadratic Quantization: Beyond First-Order Quantization for Real-Valued Matrix Compression
description: >-
  [NeurIPS 2025][Model Compression][Binary quantization] BQQ proposes quadratic binary quantization—representing weight matrices as products (rather than linear combinations) of binary matrices—thereby surpassing the expressive capacity of conventional first-order quantization. By extending AMFD (Annealed Mean-Field Descent) to PUBO problems for mixed-integer optimization, BQQ achieves a dramatic accuracy leap from 10.83% to 58.25% on 2-bit data-free ViT quantization.
tags:
  - NeurIPS 2025
  - Model Compression
  - Binary quantization
  - quadratic coding
  - mixed-integer programming
  - ViT compression
  - ultra-low bitwidth
date: 2026-05-08
content_hash: e158348ba8b42521
---

# Binary Quadratic Quantization: Beyond First-Order Quantization for Real-Valued Matrix Compression

**Conference**: NeurIPS 2025
**arXiv**: [2510.18650](https://arxiv.org/abs/2510.18650)
**Code**: To be confirmed
**Area**: Model Compression / Quantization
**Keywords**: Binary quantization, quadratic coding, mixed-integer programming, ViT compression, ultra-low bitwidth

## TL;DR
BQQ proposes quadratic binary quantization—representing weight matrices as products (rather than linear combinations) of binary matrices—thereby surpassing the expressive capacity of conventional first-order quantization. By extending AMFD (Annealed Mean-Field Descent) to PUBO problems for mixed-integer optimization, BQQ achieves a dramatic accuracy leap from 10.83% to 58.25% on 2-bit data-free ViT quantization.

## Background & Motivation

**Background**: Conventional quantization methods (uniform quantization, binary coding quantization BCQ) represent weights as linear combinations of binary matrices, $W \approx \sum_i \alpha_i B_i$, constituting first-order quantization. At ultra-low bitwidths (2-bit), the expressive capacity is severely limited.

**Limitations of Prior Work**: BCQ achieves only 10.83% accuracy (near random) on 2-bit data-free quantization of DeiT-S, since linear combinations can only cover a restricted subspace of the matrix space. Increasing the number of binary matrices yields only a linear improvement in expressiveness.

**Key Challenge**: Under extreme compression budgets, the growth of expressive power from linear combinations is too slow—$k$ binary matrices cover only $O(k)$ information. Stronger expressiveness is needed within the same storage budget.

**Goal**: Design a binary quantization scheme that transcends linear combinations, achieving higher reconstruction accuracy under identical storage budgets.

**Key Insight**: Replace linear combinations with products of binary matrices (quadratic terms)—the product $Y_i Z_i$ yields exponentially more combinatorial possibilities under the same storage cost.

**Core Idea**: Replace linear binary coding with a quadratic binary representation $W \approx \sum_i (r_i Y_i Z_i + s_i Y_i \mathbb{1} + t_i \mathbb{1} Z_i) + u\mathbb{1}$, solved via PUBO optimization.

## Method

### Overall Architecture
Input weight matrix $W$ → greedy term-by-term decomposition → solve a PUBO (Polynomial Unconstrained Binary Optimization) subproblem per term → continuous relaxation with temperature annealing via extended AMFD → step-function binarization → closed-form solution for scaling coefficients → residual propagation to the next term.

### Key Designs

1. **Quadratic Binary Representation (BQQ Representation)**:

    - Function: Represent weights via products of binary matrices.
    - Mechanism: $W \approx \sum_i (r_i Y_i Z_i + s_i Y_i \mathbf{1}_Z + t_i \mathbf{1}_Y Z_i) + u\mathbf{1}$, where $Y_i \in \{0,1\}^{m \times l}$ and $Z_i \in \{0,1\}^{l \times n}$.
    - Design Motivation: The product $Y_i Z_i$ generates an $m \times n$ matrix via an intermediate dimension $l$, yielding $2^{m \cdot l + l \cdot n}$ combinatorial possibilities—far exceeding the $2^{m \cdot n}$ of linear combinations for appropriately chosen $l$. The cross terms $Y_i \mathbf{1}$ and $\mathbf{1} Z_i$ handle row/column biases.

2. **PUBO Solving (Extended AMFD)**:

    - Function: Solve for the binary matrices $Y_i, Z_i$ in each subproblem.
    - Mechanism: Relax the discrete optimization to continuous optimization—$x_k \in [0,1]$ represents the probability of each binary variable. The objective becomes minimizing $KL(P_{MF} \| P_C)$. Temperature annealing decays from $T=0.2$ to $T=0.005$ over 50,000 steps, progressively approaching the discrete solution.
    - Design Motivation: AMFD originally addresses only QUBO (quadratic); this work extends it to PUBO (polynomial) to handle binary matrix product terms such as $Y_i Z_i$.

3. **Closed-Form Scaling Coefficients**:

    - Function: Optimally compute scaling coefficients $r_i, s_i, t_i, u$ given fixed binary matrices.
    - Mechanism: Construct and invert the Hessian matrix to obtain a closed-form solution (Algorithm 2, Eq. 10), with negligible computational cost.
    - Design Motivation: Decompose the non-convex mixed-integer problem into alternating optimization: binary variables solved via AMFD, and continuous coefficients solved in closed form.

### Loss & Training
- Objective: $\min \|W - \hat{W}\|_F^2$ (Frobenius norm reconstruction error).
- Greedy decomposition: each term operates on the residual $W_{res}^{(i)}$, progressively reducing reconstruction error.
- Data-free or requiring only a small calibration set.

## Key Experimental Results

### Main Results (ViT 2-bit Quantization, ImageNet Top-1)

| Model | Method | Data-Free 2-bit | Calibrated 2-bit |
|-------|--------|-----------------|------------------|
| DeiT-S | BCQ | 10.83% | 60.13% |
| DeiT-S | **BQQ** | **58.25%** | **69.41%** |
| DeiT-B | BCQ | 12.99% | — |
| DeiT-B | **BQQ** | **72.09%** | — |

### Ablation Study

| Quantization Scheme | MSE (Matrix Compression) | Notes |
|--------------------|--------------------------|-------|
| Linear BCQ | High | Insufficient first-order expressiveness |
| BQQ (quadratic) | **Significantly lower** | Quadratic terms enhance expressiveness |
| BQQ + joint optimization | Theoretically optimal | Upper bound of the current greedy approach |

### Key Findings
- BQQ improves data-free 2-bit DeiT-S accuracy from 10.83% to 58.25%—the first result to render 2-bit ViT quantization practically viable.
- The largest improvements are observed on matrices with skewed singular value spectra, which are common in ViTs.
- BQQ consistently outperforms BCQ at 3-bit and 4-bit, and generalizes across multiple ViT architectures (DeiT-S/B, Swin-T/S).
- State-of-the-art performance is maintained under group quantization (more compact setting).

## Highlights & Insights
- **A qualitative leap from linear to quadratic**: First-order quantization is a shared assumption of all existing methods; BQQ breaks this constraint and substantially increases expressiveness through matrix products rather than linear combinations.
- **Breakthrough improvement in data-free settings**: A 47-percentage-point accuracy gain demonstrates the overwhelming advantage of quadratic coding under extreme compression.
- **General value of extending AMFD to PUBO**: This binary optimization technique is applicable to other combinatorial optimization problems involving product constraints.

## Limitations & Future Work
- Greedy term-by-term decomposition is suboptimal; jointly optimizing all binary matrices could yield better results at greater computational cost.
- AMFD requires 50,000 optimization steps, which may be time-consuming for large matrices.
- Validation is limited to ViTs; applicability to LLM weight quantization remains untested.
- The intermediate dimension $l$ requires manual tuning.

## Related Work & Insights
- **vs. BCQ**: BCQ is the representative approach for linear binary coding; BQQ achieves greater expressiveness under identical storage via quadratic coding.
- **vs. COMQ/ERQ**: These methods improve quantization strategies but remain within the first-order framework; BQQ opens a new direction with quadratic coding.
- **vs. RepQ-ViT**: RepQ-ViT designs quantization strategies tailored to ViT characteristics, which is orthogonal to BQQ's coding scheme and potentially composable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Quadratic binary coding constitutes an entirely new quantization paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple ViT architectures, multiple bitwidths, and matrix compression benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clearly presented.
- Value: ⭐⭐⭐⭐⭐ A landmark breakthrough in ultra-low bitwidth quantization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)
- [\[NeurIPS 2025\] Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [\[NeurIPS 2025\] Tighter CMI-Based Generalization Bounds via Stochastic Projection and Quantization](tighter_cmi-based_generalization_bounds_via_stochastic_projection_and_quantizati.md)
- [\[NeurIPS 2025\] LittleBit: Ultra Low-Bit Quantization via Latent Factorization](littlebit_ultra_low-bit_quantization_via_latent_factorization.md)
- [\[ICLR 2026\] AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs](../../ICLR2026/model_compression/anybcq_hardware_efficient_flexible_binary-coded_quantization_for_multi-precision.md)

</div>

<!-- RELATED:END -->
