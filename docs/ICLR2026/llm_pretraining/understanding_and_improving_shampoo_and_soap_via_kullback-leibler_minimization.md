---
title: >-
  [Paper Note] Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization
description: >-
  [ICLR 2026][Pretraining][Shampoo] This paper reinterprets the structured second-moment estimation of Shampoo and SOAP from the perspective of KL divergence minimization, revealing inherent limitations and proposing two practical solutions—KL-Shampoo and KL-SOAP—that match or exceed the original methods without requiring Adam grafting.
tags:
  - ICLR 2026
  - Pretraining
  - Shampoo
  - SOAP
date: 2026-05-08
content_hash: 89d69922c64352b0
---
# Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization

## Paper Information
- **Conference**: ICLR 2026
- **arXiv**: [2509.03378](https://arxiv.org/abs/2509.03378)
- **Code**: [https://github.com/yorkerlin/KL-Methods](https://github.com/yorkerlin/KL-Methods)
- **Area**: LLM Pre-training
- **Keywords**: Shampoo, SOAP, KL Divergence, Kronecker Structure, Second-order Optimization, Covariance Estimation

## TL;DR
This paper reinterprets the structured second-moment estimation of Shampoo and SOAP from the perspective of KL divergence minimization, revealing inherent limitations and proposing two practical solutions—KL-Shampoo and KL-SOAP—that match or exceed the original methods without requiring Adam grafting.

## Background & Motivation

### Core Problem
Shampoo and its efficient variant SOAP utilize Kronecker-structured second-moment estimates for preconditioned optimization. However:
1. Shampoo typically requires step-size grafting with Adam to achieve competitive performance.
2. SOAP mitigates this by running Adam within the Shampoo eigenbasis but introduces additional memory overhead.
3. Prior analyses were primarily based on the Frobenius norm, which ignores SPD (Symmetric Positive Definite) constraints.

### Key Insight: Why KL Divergence?
1. KL divergence naturally respects SPD constraints, whereas the Frobenius norm does not.
2. KL provides a unified explanatory framework for Quasi-Newton methods (BFGS, DFP).
3. The terms in an SPD matrix do not play equivalent roles; the Frobenius norm treats all entries equally, while KL does not.
4. KL divergence naturally extends to tensor-valued scenarios.

## Method

### Overall Architecture

The Kronecker factors $\boldsymbol{S}_a, \boldsymbol{S}_b$ of Shampoo are viewed as the covariances of a Matrix Gaussian. Consequently, "estimating the preconditioner" is reformulated as "approximating the true second moment of gradients $\mathbb{E}[\boldsymbol{g}\boldsymbol{g}^\top]$ using KL divergence." This perspective first explains why original Shampoo is merely a one-sided approximation and thus relies on Adam grafting. Subsequently, it derives optimal fixed-point conditions for the joint minimization of both factors, which are then converted into online-executable EMA updates. QR decomposition is utilized instead of eigendecomposition to keep costs at the SOAP scale. This trajectory ultimately yields KL-Shampoo without grafting, while unifying Shampoo, SOAP, and Adafactor into a single table of "Divergence + Preconditioning Structure + Estimation Scheme," making the memory advantages evident.

> This paper focuses on the analysis and improvement of optimizers. The methodology consists of a series of Matrix Gaussian/KL derivations (reinterpretation $\to$ joint optimal conditions $\to$ EMA $\to$ QR), rather than a multi-module data pipeline. Since flowcharts cannot meaningfully represent matrix operations, no architecture diagram is provided; the four key designs are detailed below in order of derivation.

### Key Designs

**1. KL Reinterpretation of Shampoo: Exposing the One-sided Approximation Gap**

The paper first proves (Claim 1) that when the exponent $p=1/2$, Shampoo’s estimation rule for a single factor is exactly the optimal solution for KL minimization $\min_{\boldsymbol{S}_a}\text{KL}(\mathbb{E}[\boldsymbol{g}\boldsymbol{g}^\top],\boldsymbol{S})$ under $\boldsymbol{S}=(1/d_b\,\boldsymbol{S}_a)\otimes\boldsymbol{I}_b$, yielding $\boldsymbol{S}_a^*=\mathbb{E}[\boldsymbol{G}\boldsymbol{G}^\top]$. KL divergence is chosen over the Frobenius norm because it naturally respects the SPD constraint of covariances and treats entries non-uniformly. The issue is that Shampoo estimates the two factors independently, effectively fixing the other side as an identity matrix. This one-sided treatment fails to solve the joint KL problem for both factors, leaving a gap that necessitates Adam step-size grafting for stable convergence. By decomposing the "Divergence + Preconditioning Structure + Estimation Scheme," the trade-offs between KL-Shampoo, Adafactor, and Frobenius-based Shampoo become clear:

| Method | Divergence | Preconditioning Structure | Estimation Scheme |
|------|------|-----------|---------|
| KL-Shampoo | KL | Dense Kronecker | MLE |
| Adafactor | von Neumann | Diagonal Kronecker | Matrix Moment Matching |
| F-Shampoo | Frobenius | Dense Kronecker | SVD-based |

**2. Joint Optimal Conditions for KL-Shampoo: One-step Matrix Gaussian Whitening**

Rather than independent estimation, the factors should be aware of each other. Claim 2 provides the optimal solution for joint minimization $\min_{\boldsymbol{S}_a,\boldsymbol{S}_b}\text{KL}(\mathbb{E}[\boldsymbol{g}\boldsymbol{g}^\top],\boldsymbol{S})$, which satisfies a pair of coupled fixed-point equations:

$$\boldsymbol{S}_a^* = \frac{1}{d_b}\mathbb{E}[\boldsymbol{G}(\boldsymbol{S}_b^*)^{-1}\boldsymbol{G}^\top], \quad \boldsymbol{S}_b^* = \frac{1}{d_a}\mathbb{E}[\boldsymbol{G}^\top(\boldsymbol{S}_a^*)^{-1}\boldsymbol{G}]$$

Each factor is estimated on gradients that have been "whitened" by the opposite side, thereby truly utilizing the coupled information from both sides of the Kronecker product. This solution has a clear statistical identity: it is exactly the maximum likelihood estimate (MLE) of a zero-mean Matrix Gaussian, equivalent to performing Matrix Gaussian whitening on the gradients. Because the gradients are already fully diagonalized by the Kronecker structure in the optimal eigenbasis, adding an Adam-style diagonal correction (as in SOAP) becomes redundant—this foreshadows why KL-Shampoo eventually outperforms KL-SOAP.

**3. EMA Updates: Converting Fixed Points to Online Stochastic Steps**

Since the fixed points cannot be calculated exactly as expectations, they are approximated online via Exponential Moving Average (EMA):

$$\boldsymbol{S}_a \leftarrow (1-\beta_2)\boldsymbol{S}_a + \frac{\beta_2}{d_b}\boldsymbol{G}\boldsymbol{S}_b^{-1}\boldsymbol{G}^\top$$

$\boldsymbol{S}_b$ is updated symmetrically. Crucially (Claim 3), this EMA is not a heuristic but is proven to be a stochastic proximal gradient step for the KL objective, inheriting theoretical guarantees for convergence toward the optimal whitening solution. The cost is an additional matrix multiplication of the form $\boldsymbol{G}\boldsymbol{S}_b^{-1}\boldsymbol{G}^\top$ to couple the two factors via the inverse of the opposite side.

**4. QR Decomposition + EMA Eigenvalues: Reducing Cost to SOAP Scale**

Repeatedly performing eigendecompositions on $\boldsymbol{S}_a, \boldsymbol{S}_b$ is too expensive. The paper instead uses QR decomposition to maintain the eigenbasis, keeping the per-iteration runtime comparable to SOAP. However, because the eigenbasis from QR is "stale," directly using instantaneous eigenvalue estimates leads to severe degradation. Thus, an EMA correction is applied to the eigenvalues itself:

$$\begin{pmatrix}\boldsymbol{\lambda}_a \\ \boldsymbol{\lambda}_b\end{pmatrix} \leftarrow (1-\beta_2)\begin{pmatrix}\boldsymbol{\lambda}_a \\ \boldsymbol{\lambda}_b\end{pmatrix} + \beta_2\begin{pmatrix}\text{diag}(\boldsymbol{Q}_a^\top \Delta_a \boldsymbol{Q}_a) \\ \text{diag}(\boldsymbol{Q}_b^\top \Delta_b \boldsymbol{Q}_b)\end{pmatrix}$$

New information $\Delta$ is projected onto the current eigenbasis $\boldsymbol{Q}$, and only the diagonal eigenvalues are updated via EMA. This EMA eigenvalue scheme is a critical engineering component for KL-Shampoo to outperform SOAP; it also directly supports 3D and higher-order tensors without reshaping. The most significant benefit is memory: Shampoo requires storing the $d_a d_b$ Adam second moment for grafting, and SOAP requires $d_a d_b$ to run Adam in the eigenbasis. In contrast, since the KL-Shampoo fixed point already provides a usable preconditioner, this extra overhead is reduced to zero.

| Method | Kronecker | Eigenbasis | Eigenvalues | Adam 2nd | Extra Overhead |
|------|-----------|--------|--------|----------|---------|
| Shampoo | $d_a^2+d_b^2$ | $d_a^2+d_b^2$ | $d_a+d_b$ | **$d_a d_b$ (grafting)** | Yes |
| SOAP | $d_a^2+d_b^2$ | $d_a^2+d_b^2$ | N/A | **$d_a d_b$ (eigenbasis)** | Yes |
| **KL-Shampoo** | $d_a^2+d_b^2$ | $d_a^2+d_b^2$ | $d_a+d_b$ | **None** | **None** |

## Key Experimental Results

### Main Results: Language Model Pre-training

Based on a fair comparison using 150 random searches:

| Model | KL-Shampoo | SOAP | Shampoo+grafting | Shampoo (No grafting) |
|------|-----------|------|-----------------|---------------------|
| NanoGPT (123M) | **Lowest loss** | Second | Third | Poor |
| NanoRWKV7 (162M) | **Lowest loss** | Second | Moderate | Failed |
| Llama (134M) | **Lowest loss** | Second | - | - |
| NanoMoE (227M, 3D tensors) | **Lowest loss** | Second | - | - |

### Key Findings

1. **KL-Shampoo consistently outperforms SOAP**: Observed across all four models.
2. **KL-Shampoo does not require grafting**: Shampoo ($p=1/2$) without grafting failed in all 150 runs on RWKV7.
3. **KL-Shampoo outperforms KL-SOAP**: In the optimal eigenbasis, gradients are already Kronecker-diagonalized, making additional Adam-style diagonal correction redundant.
4. **EMA Eigenvalue scheme is critical**: Instantaneous estimates degrade significantly when using stale eigenbases.
5. **VN-Shampoo (trace scaling) + EMA scheme also outperforms SOAP**.

## Highlights & Insights

1. **Deep Theoretical Insight**: The KL perspective provides a unified explanation for Shampoo, SOAP, and Adafactor.
2. **Practical Improvements**: Eliminates Adam dependency, reduces memory, and maintains SOAP-level runtime.
3. **Explanation for KL-Shampoo > KL-SOAP**: Matrix Gaussian whitening is sufficient in the optimal eigenbasis; further diagonal correction is unnecessary.
4. **Natural Extension to Tensors**: The KL framework directly supports 3D+ weights without reshaping.

## Limitations & Future Work

1. The EMA scheme in KL-Shampoo introduces an extra matrix multiplication ($\boldsymbol{G}\boldsymbol{S}_b^{-1}\boldsymbol{G}^\top$).
2. Theoretical analysis assumes a zero-mean Gaussian distribution, which actual gradients may not strictly follow.
3. Experiments were primarily validated on models at the 100-200M scale; billion-parameter models remain untested.
4. QR decomposition lacks half-precision support in PyTorch, necessitating precision conversion.

## Related Work & Insights
- **Shampoo**: Gupta et al. (2018) — Original Kronecker preconditioner.
- **SOAP**: Vyas et al. (2025a) — Running Adam on the Shampoo eigenbasis.
- **Quasi-Newton Methods**: BFGS/DFP — Classical applications of KL divergence.
- **Second-order Optimization**: K-FAC, EKFAC — Approximations of the Fisher Information Matrix.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The KL perspective offers profound and unified new understanding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Fair comparison with 150 random searches is very convincing.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematically rigorous, though extensive.
- **Value**: ⭐⭐⭐⭐⭐ — Significant practical improvements: less memory and better performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling](../../ACL2025/llm_pretraining/fr_spec_speculative_sampling.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[ICLR 2026\] StochasTok: Improving Fine-Grained Subword Understanding in LLMs](stochastok_improving_fine-grained_subword_understanding_in_llms.md)
- [\[ICLR 2026\] TNT: Improving Chunkwise Training for Test-Time Memorization](tnt_improving_chunkwise_training_for_test-time_memorization.md)
- [\[ICLR 2026\] Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors](understanding_the_emergence_of_seemingly_useless_features_in_next-token_predicto.md)

</div>

<!-- RELATED:END -->
