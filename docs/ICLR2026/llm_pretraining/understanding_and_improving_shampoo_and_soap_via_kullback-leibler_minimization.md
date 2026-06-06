---
title: >-
  [Paper Note] Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization
description: >-
  [LLM Pretraining] This paper reinterprets the structured second-order moment estimation of Shampoo and SOAP through the lens of KL divergence minimization, reveals their inherent limitations…
tags:
  - "LLM Pretraining"
date: 2026-05-08
content_hash: d4c0fb649eb754c8
---

# Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization

## Paper Information
- **Conference**: ICLR 2026
- **arXiv**: [2509.03378](https://arxiv.org/abs/2509.03378)
- **Code**: [https://github.com/yorkerlin/KL-Methods](https://github.com/yorkerlin/KL-Methods)
- **Area**: LLM Pretraining
- **Keywords**: Shampoo, SOAP, KL divergence, Kronecker structure, second-order optimization, covariance estimation

## TL;DR
This paper reinterprets the structured second-order moment estimation of Shampoo and SOAP through the lens of KL divergence minimization, reveals their inherent limitations, and proposes two practical methods—KL-Shampoo and KL-SOAP—that match or surpass the original methods without requiring Adam grafting.

## Background & Motivation

### Core Problem
Shampoo and its efficient variant SOAP employ Kronecker-structured second-order moment estimates for preconditioned optimization. However:
1. Shampoo typically requires step-size grafting with Adam to remain competitive.
2. SOAP mitigates this by running Adam in the Shampoo eigenbasis, but incurs additional memory overhead.
3. Prior analyses are primarily based on the Frobenius norm, which ignores the SPD (symmetric positive definite) constraint.

### Why KL Divergence?
1. KL divergence naturally respects the SPD constraint, whereas the Frobenius norm does not.
2. In quasi-Newton methods (BFGS, DFP), KL provides a unified interpretive framework.
3. The entries of an SPD matrix do not play equivalent roles, yet the Frobenius norm treats them uniformly.
4. KL divergence extends naturally to the tensor-valued setting.

## Method

### KL Interpretation of Shampoo

**Claim 1**: The estimation rule of Shampoo ($p=1/2$) can be derived as the optimal solution to a KL minimization problem:

$$\min_{\boldsymbol{S}_a} \text{KL}(\mathbb{E}[\boldsymbol{g}\boldsymbol{g}^\top], \boldsymbol{S})$$

where $\boldsymbol{S} = (1/d_b \boldsymbol{S}_a) \otimes \boldsymbol{I}_b$, and the optimal solution is $\boldsymbol{S}_a^* = \mathbb{E}[\boldsymbol{G}\boldsymbol{G}^\top]$.

**Key Limitation**: The one-sided approach of Shampoo does not adequately address the KL problem of jointly learning both factors.

### KL-Shampoo: The Idealized Solution

**Claim 2**: The optimal solution to the joint KL minimization $\min_{\boldsymbol{S}_a, \boldsymbol{S}_b} \text{KL}(\mathbb{E}[\boldsymbol{g}\boldsymbol{g}^\top], \boldsymbol{S})$ satisfies:

$$\boldsymbol{S}_a^* = \frac{1}{d_b}\mathbb{E}[\boldsymbol{G}(\boldsymbol{S}_b^*)^{-1}\boldsymbol{G}^\top], \quad \boldsymbol{S}_b^* = \frac{1}{d_a}\mathbb{E}[\boldsymbol{G}^\top(\boldsymbol{S}_a^*)^{-1}\boldsymbol{G}]$$

**Statistical Equivalence**: KL-Shampoo $=$ maximum likelihood estimation of a zero-mean matrix Gaussian $=$ matrix Gaussian whitening.

### EMA-Based Implementation

An EMA update that approximates the above fixed-point conditions:
$$\boldsymbol{S}_a \leftarrow (1-\beta_2)\boldsymbol{S}_a + \frac{\beta_2}{d_b}\boldsymbol{G}\boldsymbol{S}_b^{-1}\boldsymbol{G}^\top$$

**Claim 3**: This EMA scheme is a stochastic proximal gradient step for KL minimization.

### Efficient Implementation: QR Decomposition + EMA Eigenvalues

Core technical contributions:
1. **QR decomposition in place of eigendecomposition**: achieves SOAP-level per-iteration runtime.
2. **EMA eigenvalue estimation**: a correction scheme for use with stale eigenbases.

$$\begin{pmatrix}\boldsymbol{\lambda}_a \\ \boldsymbol{\lambda}_b\end{pmatrix} \leftarrow (1-\beta_2)\begin{pmatrix}\boldsymbol{\lambda}_a \\ \boldsymbol{\lambda}_b\end{pmatrix} + \beta_2\begin{pmatrix}\text{diag}(\boldsymbol{Q}_a^\top \Delta_a \boldsymbol{Q}_a) \\ \text{diag}(\boldsymbol{Q}_b^\top \Delta_b \boldsymbol{Q}_b)\end{pmatrix}$$

### Unified Framework: Divergence-Projection Perspective

| Method | Divergence | Preconditioner Structure | Estimation Scheme |
|--------|------------|--------------------------|-------------------|
| KL-Shampoo | KL | Dense Kronecker | Maximum likelihood |
| Adafactor | von Neumann | Diagonal Kronecker | Matrix moment matching |
| F-Shampoo | Frobenius | Dense Kronecker | SVD-based |

### Memory Comparison

| Method | Kronecker | Eigenbasis | Eigenvalues | Adam 2nd moment | Extra overhead |
|--------|-----------|------------|-------------|-----------------|----------------|
| Shampoo | $d_a^2+d_b^2$ | $d_a^2+d_b^2$ | $d_a+d_b$ | **$d_a d_b$ (grafting)** | Yes |
| SOAP | $d_a^2+d_b^2$ | $d_a^2+d_b^2$ | N/A | **$d_a d_b$ (eigenbasis)** | Yes |
| **KL-Shampoo** | $d_a^2+d_b^2$ | $d_a^2+d_b^2$ | $d_a+d_b$ | **None** | **None** |

## Key Experimental Results

### Language Model Pretraining

Fair comparison using 150 random hyperparameter searches:

| Model | KL-Shampoo | SOAP | Shampoo+grafting | Shampoo (no grafting) |
|-------|-----------|------|-----------------|----------------------|
| NanoGPT (123M) | **Lowest loss** | 2nd | 3rd | Poor |
| NanoRWKV7 (162M) | **Lowest loss** | 2nd | Middle | Complete failure |
| Llama (134M) | **Lowest loss** | 2nd | — | — |
| NanoMoE (227M, 3D tensors) | **Lowest loss** | 2nd | — | — |

### Key Findings

1. **KL-Shampoo consistently outperforms SOAP**: across all 4 models—a surprising result.
2. **KL-Shampoo requires no grafting**: Shampoo ($p=1/2$) without grafting fails in all 150 runs on RWKV7.
3. **KL-Shampoo outperforms KL-SOAP**: the core reason is that at the optimal eigenbasis, gradients are already Kronecker-diagonalized, making the additional Adam correction redundant.
4. **EMA eigenvalue estimation is critical**: instantaneous estimation degrades severely when using a stale eigenbasis.
5. **VN-Shampoo (trace scaling) + EMA scheme also surpasses SOAP**.

## Highlights & Insights

1. **Deep theoretical insight**: the KL perspective provides a unified interpretation of Shampoo, SOAP, and Adafactor.
2. **Practical improvement**: eliminates Adam dependency, reduces memory, and maintains SOAP-level runtime.
3. **Explanation for KL-Shampoo > KL-SOAP**: at the optimal eigenbasis, matrix Gaussian whitening is already satisfied, and no further diagonal correction is needed.
4. **Natural extension to tensors**: the KL framework directly supports 3D+ weight tensors without reshaping.

## Limitations & Future Work

1. The EMA scheme of KL-Shampoo introduces additional matrix multiplications ($\boldsymbol{G}\boldsymbol{S}_b^{-1}\boldsymbol{G}^\top$).
2. The theoretical analysis assumes zero-mean Gaussian gradients, which may not hold in practice.
3. Experiments are primarily conducted on 100–200M scale models; billion-parameter regimes remain untested.
4. QR decomposition does not support half precision in PyTorch, requiring precision casting.

## Related Work & Insights
- **Shampoo**: Gupta et al. (2018) — original Kronecker preconditioner.
- **SOAP**: Vyas et al. (2025a) — runs Adam in the Shampoo eigenbasis.
- **Quasi-Newton methods**: BFGS/DFP — classical applications of KL divergence.
- **Second-order optimization**: K-FAC, EKFAC — Fisher information matrix approximations.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The KL perspective provides a deep and unified new understanding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Fair comparison with 150 random searches is highly convincing.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematically rigorous, though somewhat lengthy.
- **Value**: ⭐⭐⭐⭐⭐ — Significant practical improvements with reduced memory and better performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reducing Class-Wise Performance Disparity via Margin Regularization](reducing_class-wise_performance_disparity_via_margin_regularization.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[ICLR 2026\] Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors](understanding_the_emergence_of_seemingly_useless_features_in_next-token_predicto.md)
- [\[ICCV 2025\] ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training](../../ICCV2025/llm_pretraining/aceg_improving_generalization_of_scene_coordinate_regression.md)
- [\[ICML 2026\] Understanding Catastrophic Forgetting In LoRA via Mean-Field Attention Dynamics](../../ICML2026/llm_pretraining/understanding_catastrophic_forgetting_in_lora_via_mean-field_attention_dynamics.md)

</div>

<!-- RELATED:END -->
