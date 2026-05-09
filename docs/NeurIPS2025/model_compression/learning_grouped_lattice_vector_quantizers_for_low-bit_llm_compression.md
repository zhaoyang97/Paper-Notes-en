---
title: >-
  [Paper Note] Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression
description: >-
  [NeurIPS 2025][Model Compression][lattice vector quantization] GLVQ learns a dedicated lattice codebook (defined by a learnable generator matrix) for each weight group of an LLM, combined with group-specific μ-law companding to handle heavy-tailed distributions. Under 2-bit quantization, it achieves a Wikitext-2 perplexity of 3.36 on Llama-2-70B, substantially outperforming QuIP# (3.91) and QTIP (3.78).
tags:
  - NeurIPS 2025
  - Model Compression
  - lattice vector quantization
  - low-bit compression
  - post-training quantization
  - learnable codebook
  - companding transform
date: 2026-05-08
content_hash: 1424a120c7d24395
---

# Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression

**Conference**: NeurIPS 2025
**arXiv**: [2510.20984](https://arxiv.org/abs/2510.20984)
**Code**: [GitHub](https://github.com/xzhang9308/GLVQ)
**Area**: Model Compression
**Keywords**: lattice vector quantization, low-bit compression, post-training quantization, learnable codebook, companding transform

## TL;DR
GLVQ learns a dedicated lattice codebook (defined by a learnable generator matrix) for each weight group of an LLM, combined with group-specific μ-law companding to handle heavy-tailed distributions. Under 2-bit quantization, it achieves a Wikitext-2 perplexity of 3.36 on Llama-2-70B, substantially outperforming QuIP# (3.91) and QTIP (3.78).

## Background & Motivation

**State of the Field**: Post-training quantization (PTQ) is the dominant compression paradigm for LLM deployment. Scalar quantization methods (e.g., GPTQ) perform adequately at 4-bit and above, but degrade severely below 3-bit. Vector quantization (VQ) methods (e.g., QuIP#, AQLM) improve quantization fidelity by exploiting structured codebooks in high-dimensional space.

**Limitations of Prior Work**: QuIP# applies a fixed $E_8$ lattice uniformly across all groups and layers, ignoring statistical heterogeneity among weight groups, which leads to quantization mismatch in certain groups. AQLM learns unconstrained VQ codebooks, offering flexibility at the cost of slow lookup-based decoding.

**Root Cause**: Fixed lattices (e.g., $E_8$) are highly structured but lack adaptability; free-form VQ is adaptive but computationally expensive—a fundamental tension between codebook flexibility and decoding efficiency.

**Paper Goals**: Design a quantization scheme that retains the efficient decoding of lattice quantization (simple matrix multiplication) while adapting to the varying statistical properties of different weight groups.

**Starting Point**: Each weight group learns an independent generator matrix $\mathbf{G}_g$ defining its lattice codebook, paired with a learnable companding transform $F_g$ to handle non-uniform distributions.

**Core Idea**: Group-wise learnable lattice generator matrices + group-wise μ-law companding = structured, efficient lattice decoding with adaptation to local weight distributions.

## Method

### Overall Architecture
Input: LLM weight matrix $\mathbf{W}$. Salience-Determined Bit Allocation first assigns a bit-width to each group → each group's weights undergo group-specific companding → Babai rounding quantizes to the learned lattice codebook → inverse companding reconstructs the weights. Decoding requires only the matrix-vector product $\hat{\mathbf{w}} = F_g^{-1}(\mathbf{G}_g \mathbf{z})$.

### Key Designs

1. **Salience-Determined Bit Allocation (SDBA)**:

    - Function: Assigns optimal bit-widths to each weight group under a global bit-budget constraint.
    - Mechanism: Minimizes the KL divergence of quantized outputs $D_{KL}(\mathbf{WX} \| \hat{\mathbf{W}}\mathbf{X})$, subject to $\frac{1}{G}\sum_g b_g = N$ with equal numbers of groups assigned one bit above and one bit below the target.
    - Search algorithm: Two-pointer method requiring only $\mathcal{O}(\log m)$ iterations.
    - Example at 2-bit target: high-salience groups use 3-bit, low-salience groups use 1-bit, and the remainder use 2-bit.

2. **Learnable Lattice Codebook (Lattice Codebook Learning)**:

    - Function: Learns a group-specific lattice structure for each weight group.
    - Core formulation: Weight group $\mathbf{W}_g \in \mathbb{R}^{m_g \times n_g}$ is reshaped into $d \times \ell_g$ and quantized via a generator matrix: $\hat{\mathbf{W}}_g = \mathbf{G}_g \mathbf{Z}_g$.
    - Optimization objective: $\mathcal{L}_g = \|\mathbf{W}_g \mathbf{X} - \mathbf{G}_g \mathbf{Z}_g \mathbf{X}\|_2^2 + \lambda \|\mathbf{G}_g - \mathbf{G}_g^{(0)}\|_2^2$
    - Alternating optimization: (i) Fix $\mathbf{G}_g$, update integer indices via Babai rounding $\mathbf{z}_i = \lfloor \mathbf{G}_g^{-1} \mathbf{w}_i \rceil$ (complexity $\mathcal{O}(d^3)$); (ii) Fix $\mathbf{Z}_g$, update $\mathbf{G}_g$ via gradient descent with gradient $\nabla_{\mathbf{G}_g} \mathcal{L}_g = -2(\mathbf{W}_g \mathbf{X} - \mathbf{G}_g \mathbf{Z}_g \mathbf{X})(\mathbf{Z}_g \mathbf{X})^\top$.
    - Initialization: $\mathbf{G}_g^{(0)}$ is obtained from the Cholesky decomposition of the group covariance matrix, aligning the initial lattice with the principal weight distribution.
    - Stabilization: Spectral normalization constrains the singular values of $\mathbf{G}_g$ to $[\sigma_{\min}, \sigma_{\max}]$.
    - **Key distinction**: QuIP# uses a fixed global $E_8$ lattice; AQLM learns free-form codebooks requiring lookup-based decoding; GLVQ learns group-specific lattices while preserving lattice structure, so decoding requires only matrix multiplication.

3. **Group-Specific μ-law Companding**:

    - Function: Compresses heavy-tailed weight distributions into a more uniform form prior to quantization, reducing quantization error in low-magnitude regions.
    - Transform: $F_g(x) = \text{sgn}(x) \frac{\ln(1 + \mu_g |x|)}{\ln(1 + \mu_g)}$, with inverse $F_g^{-1}(y) = \text{sgn}(y) \frac{(1+\mu_g)^{|y|}-1}{\mu_g}$
    - Learnable parameter: $\mu_g > 0$ controls compression strength and is jointly optimized with $\mathbf{G}_g$ via gradient descent.
    - Initialization: $\mu_g^{(0)} = 100 \tanh(\kappa_g / 10)$, where $\kappa_g$ is the sample kurtosis of the group—heavier-tailed groups are initialized with stronger compression.
    - Constraint: $\mu_g \in [10, 255]$ for numerical stability.
    - Full encode-decode pipeline: $\tilde{\mathbf{W}}_g = F_g(\mathbf{W}_g) \to \mathbf{Z}_g = \lfloor \mathbf{G}_g^{-1} \tilde{\mathbf{W}}_g \rceil \to \hat{\mathbf{W}}_g = F_g^{-1}(\mathbf{G}_g \mathbf{Z}_g)$

### Runtime Characteristics
- **Minimal storage overhead**: Each group requires only a $d \times d$ FP16 generator matrix plus one FP16 scalar $\mu_g$, adding approximately 2 MB for Llama 2-7B (0.2% of the total 1.1 GB).
- **Efficient decoding**: Each sub-block requires only $d^2 + d$ multiplications; end-to-end latency increases by only 2–3% compared to 4-bit uniform PTQ.
- **Streaming decoding**: Inference materializes only a small number of sub-blocks at a time, reducing peak memory by more than 10× compared to pre-decompressing entire layers.

## Key Experimental Results

### Main Results: Perplexity under 2-bit Quantization

| Method | Llama1-7B | Llama1-13B | Llama1-65B | Llama2-7B | Llama2-13B | Llama2-70B |
|--------|-----------|------------|------------|-----------|------------|------------|
| FP16 | 5.68 | 5.09 | 3.53 | 5.12 | 4.57 | 3.12 |
| OmniQuant | 15.5 | 13.2 | 7.58 | — | — | — |
| QuIP# | 6.86 | 5.97 | 4.36 | 6.19 | 5.35 | 3.91 |
| QTIP | 6.52 | 5.80 | 4.21 | 5.91 | 5.26 | 3.78 |
| GLVQ-8D | 6.28 | 5.64 | 4.01 | 5.69 | 5.02 | 3.62 |
| **GLVQ-32D** | **6.00** | **5.38** | **3.81** | **5.41** | **4.80** | **3.36** |

GLVQ-32D achieves the lowest perplexity at all model scales. On Llama2-70B at 2-bit, PPL = 3.36 vs. QuIP#'s 3.91 (a reduction of 0.55).

### Zero-Shot Accuracy (Llama-2-70B, 4-bit)

| Method | ARC-C | ARC-E | PIQA | WINO |
|--------|-------|-------|------|------|
| FP16 | 51.1 | 77.7 | 81.1 | 77.0 |
| QuIP# | 50.6 | 78.1 | 81.4 | 77.1 |
| QTIP | 50.0 | 77.6 | 81.5 | 77.0 |
| **GLVQ-8D** | **51.2** | 78.0 | **81.6** | **77.3** |

At 4-bit, GLVQ-8D surpasses QuIP# and QTIP on most tasks; the advantage is even more pronounced at 2-bit.

### GLVQ-8D vs. GLVQ-32D

| Dimension | GLVQ-32D | GLVQ-8D |
|-----------|----------|---------|
| PPL | Lower, better fidelity | Slightly higher than 32D |
| Encoding speed | Slower due to Babai rounding $\mathcal{O}(d^3)$ | Faster |
| Use case | Maximum compression quality | Balanced efficiency and quality |

### Key Findings
- The advantage of GLVQ grows with lower bit-widths (2-bit > 3-bit > 4-bit), indicating that group-adaptive lattices are most valuable under extreme compression.
- Larger lattice dimension $d$ yields better quantization fidelity at encoding complexity $\mathcal{O}(d^3)$.
- Companding contributes most significantly to groups with heavy-tailed weight distributions.
- Cholesky initialization converges faster and achieves lower final PPL than random initialization.
- Storage overhead is only 0.2% and latency increase is only 2–3%, making the approach highly practical.

## Highlights & Insights
- **Elegant balance between structure and flexibility**: GLVQ preserves the efficient decoding of lattice quantization (matrix multiplication) while achieving the adaptability of free-form codebooks through learned generator matrices—an approach of optimizing within a constrained space that is broadly instructive.
- **Elegant application of companding**: The classical μ-law transform from communications engineering is applied to weight quantization with per-group learnable $\mu_g$ and a kurtosis-based initialization strategy, addressing the heavy-tail problem via a simple yet principled mechanism.
- **Negligible storage overhead**: Each group requires only a small matrix and a single scalar as additional storage, making deployment straightforward.
- **Babai rounding as an approximation to nearest lattice point search**: Though approximate, it admits a formal error bound, and the differentiable training pipeline naturally compensates for the approximation error.

## Limitations & Future Work
- Evaluation is limited to the Llama 1/2 family; experiments on newer models such as Llama 3, Qwen, and Mistral are absent.
- The SDBA bit allocation strategy is adopted from Slim-LLM and is not an original contribution of this work.
- The lattice dimension $d$ is set manually without an adaptive selection mechanism.
- No comparison is made against PV-Tuning (which alternates optimization of continuous parameters and discrete assignments).
- Inference speed experiments are conducted only on an RTX 4090; actual speedups may vary across hardware platforms.
- Joint quantization of weights and activations is not evaluated.

## Related Work & Insights
- **vs. QuIP#**: Applies a fixed $E_8$ lattice with Hadamard rotation preprocessing, using the same codebook for all groups; GLVQ learns independent generator matrices per group, consistently achieving 0.4–0.5 lower PPL at 2-bit.
- **vs. AQLM**: Learns free-form vector codebooks that are flexible but require lookup-based decoding; GLVQ preserves lattice structure, enabling faster matrix-multiplication-based decoding.
- **vs. QTIP**: Achieves very high-dimensional VQ through stateful decoding, decoupling codebook size from bit-rate; GLVQ takes a different approach, optimizing lattice geometry within a fixed lattice dimension.
- **vs. GPTQ/AWQ**: Scalar quantization methods that are competitive at 4-bit and above but fall substantially behind VQ methods at 2–3 bits.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of group-wise learnable lattice codebooks and group-wise companding is a meaningful contribution, though each individual component (lattice quantization, μ-law, SDBA) has prior precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers the full Llama 1/2 series with both PPL and zero-shot evaluations and compares 8D/32D configurations, but lacks coverage of additional model families.
- Writing Quality: ⭐⭐⭐⭐⭐ Method derivations are rigorous, the pipeline diagram is clear, algorithm pseudocode is complete, and storage/latency analyses are thorough.
- Value: ⭐⭐⭐⭐⭐ Achieves a substantial advance in extremely low-bit quantization—a direction of high engineering importance—with open-source code, minimal latency overhead, and strong deployment practicality.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment](q-palette_fractional-bit_quantizers_toward_optimal_bit_allocation_for_efficient_.md)
- [\[NeurIPS 2025\] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](paretoq_improving_scaling_laws_in_extremely_low-bit_llm_quantization.md)
- [\[NeurIPS 2025\] LittleBit: Ultra Low-Bit Quantization via Latent Factorization](littlebit_ultra_low-bit_quantization_via_latent_factorization.md)
- [\[NeurIPS 2025\] FALQON: Accelerating LoRA Fine-tuning with Low-Bit Floating-Point Arithmetic](falqon_accelerating_lora_fine-tuning_with_low-bit_floating-point_arithmetic.md)
- [\[NeurIPS 2025\] A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings](a-thought_efficient_reasoning_via_bidirectional_compression_for_low-resource_set.md)

<!-- RELATED:END -->
