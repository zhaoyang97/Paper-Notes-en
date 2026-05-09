---
title: >-
  [Paper Note] Condition Errors Refinement in Autoregressive Image Generation with Diffusion Loss
description: >-
  [ICLR 2026][Image Generation][autoregressive generation] This paper theoretically analyzes the advantage of autoregressive diffusion loss models over conditional diffusion models in correcting condition errors (exponential decay of gradient norms), and proposes a condition refinement method based on optimal transport (Wasserstein Gradient Flow) to address the "condition inconsistency" problem in autoregressive generation, achieving FID 1.31 on ImageNet (based on MAR).
tags:
  - ICLR 2026
  - Image Generation
  - autoregressive generation
  - diffusion loss
  - condition refinement
  - optimal transport
  - Wasserstein gradient flow
date: 2026-05-08
content_hash: 5f3d48922739e325
---

# Condition Errors Refinement in Autoregressive Image Generation with Diffusion Loss

**Conference**: ICLR 2026
**arXiv**: [2602.07022](https://arxiv.org/abs/2602.07022)
**Code**: None
**Area**: Diffusion Models / Autoregressive Image Generation
**Keywords**: autoregressive generation, diffusion loss, condition refinement, optimal transport, Wasserstein gradient flow

## TL;DR
This paper theoretically analyzes the advantage of autoregressive diffusion loss models over conditional diffusion models in correcting condition errors (exponential decay of gradient norms), and proposes a condition refinement method based on optimal transport (Wasserstein Gradient Flow) to address the "condition inconsistency" problem in autoregressive generation, achieving FID 1.31 on ImageNet (based on MAR).

## Background & Motivation

**State of the Field**: Autoregressive image generation has advanced rapidly in recent years. Methods such as MAR replace VQ tokenization with diffusion loss, achieving image quality competitive with or surpassing diffusion models. However, the theoretical differences between autoregressive diffusion loss and standard conditional diffusion models remain underexplored.

**Limitations of Prior Work**: Although autoregressive conditional generation progressively builds context, the condition $c_i$ at each step accumulates redundant information from preceding patches that is irrelevant to the current patch—referred to as "condition inconsistency." This redundancy perturbs the conditional score $\nabla_{x_t} \log p(x_t|c_i)$ in the denoising process, degrading generation quality.

**Root Cause**: Autoregressive methods capture dependencies through context accumulation, but the context inevitably incorporates noisy information irrelevant to generating the current patch. The challenge is to retain useful dependencies while eliminating redundancy.

**Paper Goals**: (a) Theoretically characterize where autoregressive diffusion loss outperforms conditional diffusion; (b) analyze the mechanism behind condition inconsistency; (c) propose a theoretically grounded condition refinement method.

**Starting Point**: The paper begins with a theoretical analysis of conditional score matching, proving that the autoregressive process itself yields a condition refinement effect (exponential decay of gradient norms), and then applies optimal transport theory to further correct residual condition inconsistency.

**Core Idea**: Autoregressive conditional generation inherently exhibits condition error decay, yet condition inconsistency persists. Wasserstein Gradient Flow-based condition refinement provides convergence guarantees to the ideal conditional distribution.

## Method

### Overall Architecture

The method is built upon the autoregressive + diffusion loss framework (analogous to MAR). When generating each patch:
- **Input**: The autoregressive model predicts an initial condition $c_i$.
- **OT Refinement**: An optimal transport module refines the condition $c_i \to c_i^{(k)}$, driving it toward the ideal condition distribution $P_{c^*}$.
- **Denoise MLP**: The refined condition guides the denoising process to generate the latent representation of the current patch.
- **Output**: The generated patch is appended to the history, and the process continues to the next patch.

### Key Designs

1. **Theoretical Analysis of Condition Errors**:

    - **Function**: Proves that the conditional score matching loss upper-bounds the unconditional score matching loss (Theorem 1), and that the conditional gradient norm in the autoregressive process decays exponentially (Theorem 2).
    - **Mechanism**: Under standard Markov and Gaussian noise assumptions, a condition error term $\epsilon_c$ is defined to quantify the effect of conditioning on the score function. The key result is $\|\nabla_{x_t} \log p_t(x_t|c_i)\| \leq M\beta^i + m$, where $\beta \in (0,1)$, indicating that the influence of the condition decays exponentially to a steady-state value $m$ as autoregressive iteration proceeds.
    - **Design Motivation**: Provides theoretical support for autoregressive diffusion loss methods—patch-by-patch generation inherently performs condition refinement, which is the fundamental reason these methods outperform globally conditioned diffusion models.

2. **Analysis of Condition Inconsistency (Lemma 6)**:

    - **Function**: Formally defines the information redundancy problem in autoregressive conditional generation.
    - **Mechanism**: Each condition $c_i$ can be decomposed into the ideal condition $c_i^* = \pi_{\mathcal{I}_i^*}(c_i)$ (projected onto the minimal sufficient information subspace) and a redundant component $\eta_i = c_i - c_i^*$. The energy of the redundant component $\mathbb{E}[\|\eta_i\|_2^2]$ consists of two parts: propagation from preceding conditions and newly injected noise.
    - **Design Motivation**: Reveals that while autoregressive condition refinement is effective, it is imperfect—redundant information accumulates continuously and requires additional correction.

3. **Wasserstein Gradient Flow Condition Refinement (Proposition 2 + Theorem 3)**:

    - **Function**: Models condition refinement as a gradient flow optimization problem in Wasserstein space.
    - **Mechanism**: Minimizes the energy functional $\mathcal{F}(P_c) = W_2^2(P_c, P_{c^*}) + \lambda \mathbb{E}_{c \sim P_c}[\|c - \mathcal{T}^{-1}(x)\|^2]$ via JKO-scheme discretization. The first term pushes the condition toward the ideal distribution; the second term is an inverse-process regularizer that counteracts information accumulation.
    - **Design Motivation**: The OT metric measures the "transport cost" between distributions rather than their overlap (as in KL divergence), making it more suitable for distributions with differing supports. The theoretical guarantee $W_2(P_c^{(k)}, P_{c^*}) \leq \rho^k W_2(P_c^{(0)}, P_{c^*})$ ensures exponential convergence.

4. **Sinkhorn Algorithm Implementation**:

    - **Function**: Solves the regularized optimal transport problem in practice.
    - **Mechanism**: Solves $\inf_\gamma \mathbb{E}_{(c,c')}[\|c - c'\|^2] + \epsilon \text{KL}(\gamma|\pi)$ efficiently via Sinkhorn iterations.
    - **Design Motivation**: Directly solving the OT problem is NP-hard; entropic regularization via Sinkhorn enables $O(n^2)$ complexity.

### Loss & Training

- Base framework uses MAR's diffusion loss (cosine noise schedule, 1000 steps).
- Learning rate $1 \times 10^{-5}$, 400 epochs, batch size 2048.
- 100-epoch linear learning rate warmup.
- EMA momentum 0.9999.
- VAE: LDM's KL-16.

## Key Experimental Results

### Main Results

| Method | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|--------|-------|------|-------------|----------|
| MAR (943M) | 1.55 | 303.7 | 0.81 | 0.62 |
| De-MAR | 1.47 | 305.8 | 0.83 | 0.62 |
| RAR | 1.50 | 306.9 | 0.80 | 0.62 |
| **Ours (MAR)** | **1.31** | **324.2** | 0.81 | **0.63** |
| Ours (AR) | 1.52 | 317.6 | 0.82 | 0.60 |
| Baseline (CDM) | 3.26 | 259.6 | 0.81 | 0.58 |
| Baseline (AR) | 2.02 | 282.6 | 0.80 | 0.59 |

### Ablation Study (Scalability)

| Model Size | MAR FID | Ours FID | MAR IS | Ours IS |
|------------|---------|----------|--------|---------|
| 208M | 2.31 | **1.96** | 281.7 | **290.5** |
| 479M | 1.78 | **1.59** | 296.0 | **301.5** |
| 943M | 1.55 | **1.31** | 303.7 | **324.2** |

**ImageNet 512×512:**

| Method | FID ↓ | IS ↑ |
|--------|-------|------|
| MAR | 1.73 | 279.9 |
| **Ours** | **1.58** | **302.3** |

### Key Findings
- OT condition refinement consistently improves performance across all model sizes, with larger models showing more pronounced gains (208M: −0.35 FID → 943M: −0.24 FID, with superior absolute values).
- Analysis of the denoising process shows higher SNR and lower noise intensity in later denoising stages, confirming that condition refinement is effective.
- The autoregressive baseline (AR) substantially outperforms the conditional diffusion baseline (CDM), validating the theoretical analysis (3.26 → 2.02).
- The method remains effective at high resolution (512×512).

## Highlights & Insights
- **Strong integration of theory and practice**: Rather than designing methods heuristically, the paper first theoretically analyzes the advantages of autoregressive diffusion loss (exponential decay of conditional gradients), identifies the residual issue (condition inconsistency), and resolves it via OT theory. The analysis–discovery–solution chain is logically coherent.
- **Wasserstein Gradient Flow for condition refinement** is an interesting perspective that can be transferred to any scenario requiring "correction of a conditional distribution," such as guidance optimization in conditional generation or formalization of prompt engineering.
- The finding that **autoregressive generation inherently performs condition refinement** is independently valuable—it explains why methods such as MAR can generate high-quality images without global attention.

## Limitations & Future Work
- The OT refinement module introduces additional inference overhead (Sinkhorn iterations), but the paper does not report inference speed comparisons.
- The theoretical analysis relies on numerous simplifying assumptions (Gaussian distributions, small variance, bounded second-order derivatives, etc.); whether these hold strictly for deep networks is debatable.
- Validation is limited to ImageNet; experiments on more complex tasks such as text-to-image generation are absent.
- How the ideal condition distribution $P_{c^*}$ is obtained or approximated in practice is not sufficiently discussed.
- Comparisons with De-MAR and RAR are incomplete (e.g., parameter counts and training costs may not be controlled).

## Related Work & Insights
- **vs. MAR**: OT condition refinement is applied directly on top of MAR, reducing FID from 1.55 to 1.31 and increasing IS from 303.7 to 324.2, demonstrating that MAR's conditions have room for improvement.
- **vs. Conditional Diffusion Models (CDM)**: Both theoretical analysis and experiments show that autoregressive diffusion loss outperforms globally conditioned diffusion (FID 2.02 vs. 3.26).
- **vs. RAR/De-MAR**: These are all methods that improve autoregressive image generation; this paper approaches the problem from the perspective of condition refinement, making it complementary to the others.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The theoretical analysis perspective is novel and OT-based condition refinement is creative, though the method itself (Sinkhorn + JKO) combines existing tools.
- **Experimental Thoroughness**: ⭐⭐⭐ Main experiments are solid, but inference speed, training cost, and T2I experiments are missing.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are rigorous and clear, though the heavy notation raises the reading barrier.
- **Value**: ⭐⭐⭐⭐ Contributes to the theoretical understanding of autoregressive image generation; the OT refinement method is practical and transferable.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] From Prediction to Perfection: Introducing Refinement to Autoregressive Image Generation](from_prediction_to_perfection_introducing_refinement_to_autoregressive_image_gen.md)
- [\[ICLR 2026\] Autoregressive Image Generation with Randomized Parallel Decoding](autoregressive_image_generation_with_randomized_parallel_decoding.md)
- [\[ICLR 2026\] Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation](locality-aware_parallel_decoding_for_efficient_autoregressive_image_generation.md)
- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[ICLR 2026\] SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation](ssg_scaled_spatial_guidance_for_multi-scale_visual_autoregressive_generation.md)

<!-- RELATED:END -->
