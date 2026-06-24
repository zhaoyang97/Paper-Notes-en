---
title: >-
  [Paper Note] System-Embedded Diffusion Bridge Models
description: >-
  [NeurIPS 2025][Image Generation][Diffusion bridge models] This paper proposes System-embedded Diffusion Bridge Models (SDB), which directly embed a known linear measurement system into the coefficients of a matrix-valued SDE, enabling decoupled control over denoising in the range space and information synthesis in the null space. SDB achieves consistent performance improvements across multiple inverse problems and demonstrates strong robustness to system mismatch.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Diffusion bridge models"
  - "inverse problems"
  - "matrix-valued SDE"
  - "measurement system embedding"
  - "pseudoinverse reconstruction"
date: 2026-05-08
content_hash: bf7ff7ba39a7ada1
---

# System-Embedded Diffusion Bridge Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.23726](https://arxiv.org/abs/2506.23726)  
**Code**: [https://github.com/sobieskibj/sdb](https://github.com/sobieskibj/sdb)  
**Area**: Image Generation / Inverse Problem Solving
**Keywords**: Diffusion bridge models, inverse problems, matrix-valued SDE, measurement system embedding, pseudoinverse reconstruction

## TL;DR

This paper proposes System-embedded Diffusion Bridge Models (SDB), which directly embed a known linear measurement system into the coefficients of a matrix-valued SDE, enabling decoupled control over denoising in the range space and information synthesis in the null space. SDB achieves consistent performance improvements across multiple inverse problems and demonstrates strong robustness to system mismatch.

## Background & Motivation

Inverse problems — recovering signals from incomplete or noisy measurements — are fundamental tasks in science and engineering. Diffusion-model-based solvers have converged on two major paradigms: **unsupervised methods** that leverage pretrained generative models guided by the measurement system, and **supervised bridge methods** that train stochastic processes on paired data to map from degraded observations to clean signals.

The root cause of the tension between these paradigms is that, while unsupervised methods typically assume a known measurement system and exploit its structure, bridge methods discard this prior knowledge and focus solely on generic mappings between arbitrary distributions. Yet in practical applications such as CT and MRI reconstruction, the linear measurement process is known and paired datasets are available. For instance, in image inpainting, existing bridge methods cannot distinguish between the known range-space component (unmasked regions) and the missing null-space component (masked regions), leading to unnecessary corruption in the range space.

The core idea of SDB is to **embed the measurement operator and noise covariance directly into the matrix-valued SDE coefficients of the diffusion process**, so that the generative process can separately handle denoising in the range space and information synthesis in the null space.

## Method

### Overall Architecture

SDB constructs a diffusion bridge from the pseudoinverse reconstruction (PR) to the clean signal. Given the linear measurement system $\mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\Sigma}^{1/2}\boldsymbol{\epsilon}$, SDB maps the measurements back to signal space via the pseudoinverse $\mathbf{A}^+$ as $\hat{\mathbf{x}} = \mathbf{A}^+\mathbf{y}$, and then learns a diffusion bridge from the PR to the clean signal.

### Key Designs

1. **Measurement System Embedding**: The central contribution of SDB is the design of specific mean matrix $\mathbf{H}_t$ and covariance matrix $\boldsymbol{\Sigma}_t$:

    $\mathbf{H}_t = \mathbf{A}^+\mathbf{A} + \alpha_t(\mathbf{I} - \mathbf{A}^+\mathbf{A})$
    $\boldsymbol{\Sigma}_t = \gamma_t\mathbf{A}^+\boldsymbol{\Sigma}\mathbf{A}^{+\top} + \beta_t(\mathbf{I} - \mathbf{A}^+\mathbf{A})$

   where $\alpha_t, \beta_t, \gamma_t$ govern null-space drift, null-space diffusion, and range-space diffusion, respectively. This design allows the range-space and null-space components of the intermediate state $\mathbf{x}_t$ to evolve independently: the range-space component directly models measurement noise (preserved perfectly when noiseless), while the null-space component undergoes information synthesis.

2. **SDE Perspective and Theoretical Guarantees**: Using the mapping from Theorem 1 (Tivnan et al., 2025), the corresponding drift and diffusion coefficients $\mathbf{F}_t, \mathbf{G}_t$ are derived from $\mathbf{H}_t, \boldsymbol{\Sigma}_t$. It is further shown (Theorem 2) that when $\alpha_t$ and $\beta_t$ take specific forms, the null-space dynamics reduce to an optimal transport ODE, establishing a connection to Schrödinger Bridges. Theorem 3 proves that under the conditions $\lim_{t\to 1}\gamma_t=1$ and $\lim_{t\to 1}\alpha_t^2/\beta_t=0$, SDB generates asymptotically exact samples from the Bayesian posterior.

3. **Three Variants**:

    - **SDB (SB)**: Based on the Schrödinger Bridge, with $\alpha_t = \bar{\sigma}_t^2 / (\bar{\sigma}_t^2 + \sigma_t^2)$, possessing optimal transport properties.
    - **SDB (VP)**: A reinterpretation of VP diffusion with $\alpha_t = 1-t$, performing convex interpolation in the null space.
    - **SDB (VE)**: A reinterpretation of VE diffusion with $\alpha_t = 1$, converging the null space to an isotropic Gaussian.

### Loss & Training

Training uses a denoising objective with an L1 reconstruction loss: $L_\theta = \|\mathcal{D}_\theta(\mathbf{x}_t, t) - \mathbf{x}\|_1$. The network $\mathcal{D}_\theta$ directly predicts the clean signal. During training, range-space noise $\boldsymbol{\epsilon} \in \mathbb{R}^m$ and null-space noise $\boldsymbol{\epsilon}' \in \mathbb{R}^d$ are sampled independently, reflecting the separate modeling of the two subspaces. Sampling uses the Euler-Maruyama solver with 100 discretization steps.

## Key Experimental Results

### Main Results

| Task-Dataset | Metric | SDB (SB) | Prev. SOTA Bridge | Gain |
|---|---|---|---|---|
| Inpainting-CelebA | FID↓ | **4.63** | 4.68 (IR-SDE) | −0.05 |
| Inpainting-CelebA | PSNR↑ | **30.40** | 29.92 (IR-SDE) | +0.48 |
| SuperRes-DIV2K | FID↓ | **81.56** | 83.73 (I2SB) | −2.17 |
| SuperRes-DIV2K | PSNR↑ | **26.10** | 25.79 (DDBM) | +0.31 |
| CT Recon.-RSNA | FID↓ | **15.02** | 18.88 (IR-SDE) | −3.86 |
| CT Recon.-RSNA | PSNR↑ | **46.672** | 44.415 (DDBM) | +2.26 |
| MRI Recon.-Br35H | FID↓ | **29.85** | 30.14 (IR-SDE) | −0.29 |
| MRI Recon.-Br35H | PSNR↑ | **29.812** | 28.971 (DDBM) | +0.84 |

### Ablation Study (System Mismatch Robustness)

| MRI Mismatch Setting | SDB (SB) PSNR | DDBM PSNR | Gap |
|---|---|---|---|
| $\lambda_1=16$ (training value) | 29.81 | 28.97 | +0.84 |
| $\lambda_1=14$ | Stably superior | Significant drop | Gap widens |
| $\lambda_1=12$ | Still leading | Further degradation | Gap larger |
| $\sigma^2$ increased to 2× | Robust | Notable drop | SDB advantage pronounced |

### Key Findings

- SDB (SB) consistently outperforms all baseline bridge methods across all four inverse problems, with the most stable performance ranking.
- On medical imaging tasks (CT/MRI), SDB achieves PSNR improvements exceeding 2 dB over baseline bridge methods.
- In system mismatch experiments, SDB's performance advantage **grows** as the degree of mismatch increases, demonstrating superior generalization capability.
- Unsupervised methods perform notably worse than bridge methods under a unified evaluation setting.

## Highlights & Insights

- The paper elegantly embeds the mathematical structure of the measurement system into the diffusion process, taking the principle of "injecting domain knowledge into generative models" to its logical conclusion.
- Through range-null space decomposition, independent modeling and independent noise control are achieved for the two subspaces.
- Three theorems provide rigorous theoretical support, including connections to optimal transport and asymptotic exactness of posterior sampling.
- The system mismatch robustness experiments carry important practical implications for real-world deployment.

## Limitations & Future Work

- Applicability is restricted to linear measurement systems; extension to nonlinear settings is only demonstrated via preliminary proof-of-concept experiments.
- The variance schedules adopt a simple linear design; the interaction between range-space and null-space schedules warrants further investigation.
- CT/MRI experiments are conducted on 2D slices; practical clinical deployment requires 3D reconstruction.
- The approach requires the pseudoinverse $\mathbf{A}^+$ to be known or computable.

## Related Work & Insights

- **vs. I2SB**: Shares the most similar stochastic process structure, but SDB achieves substantial performance gains by embedding measurement system information.
- **vs. IR-SDE/GOUB**: These methods employ scalar-valued SDE coefficients; SDB generalizes to matrix-valued coefficients for finer-grained control.
- **vs. DPS/ΠGD and other unsupervised methods**: Unsupervised methods exploit the known system without paired data; SDB leverages both simultaneously.
- **vs. DDBM**: DDBM symmetrizes the variance schedule, whereas SDB achieves a more principled design through system embedding.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The idea of embedding the measurement system into SDE coefficients is highly original and mathematically elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four inverse problems, three variants, and system mismatch analysis, all under a unified and fair comparison framework.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous and clear, motivation is well-articulated, and theory is corroborated by experiments.
- Value: ⭐⭐⭐⭐ Significant contribution to the inverse problems community, though applicability is limited to linear systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)
- [\[NeurIPS 2025\] Aligning Compound AI Systems via System-level DPO](aligning_compound_ai_systems_via_system-level_dpo.md)
- [\[NeurIPS 2025\] Dynamic Diffusion Schrödinger Bridge in Astrophysical Observational Inversions](dynamic_diffusion_schrödinger_bridge_in_astrophysical_observational_inversions.md)
- [\[AAAI 2026\] Rethinking Flow and Diffusion Bridge Models for Speech Enhancement](../../AAAI2026/image_generation/rethinking_flow_and_diffusion_bridge_models_for_speech_enhancement.md)
- [\[NeurIPS 2025\] Coupling Generative Modeling and an Autoencoder with the Causal Bridge](coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)

</div>

<!-- RELATED:END -->
