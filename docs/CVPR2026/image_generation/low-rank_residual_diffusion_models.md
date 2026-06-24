---
title: >-
  [Paper Note] Low-Rank Residual Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Residual Diffusion] LRDM identifies that in "near-domain image restoration" (tasks where source and target domains are already highly similar, such as deraining, deblurring, or deshadowing), degradation residuals are inherently low-rank. Consequently, it constrains the forward diffusion process within a low-rank residual subspace while maintaining the reverse process as full-rank. By adaptively adjusting the rank across time steps…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Residual Diffusion"
  - "Low-Rank Subspace"
  - "Image Deraining"
  - "Asymmetric Diffusion"
  - "Adaptive Rank"
date: 2026-05-08
content_hash: 0023bc3cc1353461
---

# Low-Rank Residual Diffusion Models

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_Low-Rank_Residual_Diffusion_Models_CVPR_2026_paper.html)  
**Code**: https://github.com/JF-Tan/LRDM  
**Area**: Diffusion Models / Image Restoration / Low-Rank Modeling  
**Keywords**: Residual Diffusion, Low-Rank Subspace, Image Deraining, Asymmetric Diffusion, Adaptive Rank

## TL;DR
LRDM identifies that in "near-domain image restoration" (tasks where source and target domains are already highly similar, such as deraining, deblurring, or deshadowing), degradation residuals are inherently low-rank. Consequently, it constrains the forward diffusion process within a low-rank residual subspace while maintaining the reverse process as full-rank. By adaptively adjusting the rank across time steps, the model theoretically tightens the variational lower bound and achieves superior restoration fidelity with fewer sampling steps.

## Background & Motivation
**Background**: Diffusion models have demonstrated significant power in image-to-image restoration tasks. Residual Diffusion Models (RDDM) represent a key approach—instead of directly generating a clean image, they learn the "residual" (pixel-level difference) between the degraded and clean domains. In the forward process, the residual $I_{res} = I_{in} - I_0$ is gradually injected with Gaussian noise according to a variance schedule $\{\alpha_t, \beta_t\}$, with closed-form sampling defined as $I_t = I_0 + \bar\alpha_t I_{res} + \bar\beta_t\epsilon$.

**Limitations of Prior Work**: Although RDDM, DiffUIR, and DeblurDiff differ in model design, they all execute diffusion in **full-rank** space, operating repeatedly across the complete pixel or latent space at every step. However, in many practical restoration scenarios, the source domain (degraded image) and target domain (clean image) are already highly similar in pixel space—a condition the authors call "near-domain restoration." In such cases, the residual is essentially sparse and occupies only a low-dimensional subspace. Using standard full-rank diffusion in such sparse residual spaces forces the generation process to explore a semantic space that is mostly empty, leading to inefficient and unstable sampling.

**Key Challenge**: While residuals actually reside in a low-dimensional subspace, full-rank diffusion introduces excessive degrees of freedom to the residual semantic space. This redundancy in expressive power slows down sampling and harms stable reconstruction.

**Goal**: To explicitly model the low-dimensional structure of residuals for near-domain restoration by situating the diffusion process within a low-rank residual subspace, aiming for both efficiency (fewer sampling steps) and fidelity (no loss of detail).

**Key Insight**: The authors performed Singular Value Analysis on residuals from deraining data (Raindrop) and found that the singular value spectrum decays sharply—the first 10 ranks account for approximately 90% of the energy. Reconstructing the image using truncated SVD with only a few low-rank components resulted in minimal difference from the original. This empirically confirms the strong low-rank property of residuals.

**Core Idea**: Construct the diffusion process within a low-rank residual subspace. The paper theoretically proves that "residuals falling into a low-rank subspace" tightens the Variational Lower Bound (VLB). In practice, this is implemented using an "asymmetric residual diffusion process" (low-rank forward, full-rank reverse) combined with adaptive rank adjustment over time steps.

## Method

### Overall Architecture
The starting point for LRDM is an empirical observation (low-rank residuals) and a theoretical guarantee (low-rank → tighter VLB). The methodology revolves around safely constraining diffusion into a low-rank subspace without compromising reverse reconstruction. It consist of three components: (1) **Low-Rank Residual Hypothesis**, providing the theoretical foundation to prove that restricting the generation process to a low-rank subspace yields a strictly tighter VLB; (2) **Asymmetric Residual Diffusion Process**, the core mechanism where the forward process only injects residual signals into the low-rank subspace $S_k$, while the reverse process is liberated to full-rank space for reconstruction to avoid model mismatch; (3) **Adaptive Rank Selection**, which acknowledges that residual complexity evolves with diffusion time steps and uses a rank scheduler $k(t)$ to dynamically adjust subspace dimensions. As the work focuses on theory and sampling mechanism improvements, the specific mathematical components are detailed below.

### Key Designs

**1. Low-Rank Residual Hypothesis: Proving that restricting diffusion to a low-rank subspace tightens the VLB**

This is the foundation of the paper. Let the residual be $I_{res}\in\mathbb{R}^D$ with an uncentered covariance $\Sigma = \mathbb{E}[I_{res}I_{res}^\top]$. Assuming near-domain restoration, the eigenvalues $\lambda_1\ge\cdots\ge\lambda_D\ge 0$ of $\Sigma$ decay rapidly. Thus, there exists a subspace $S_k$ of rank $k\ll D$ such that the projection error $\mathbb{E}[\|I_{res} - Q_k(I_{res})\|_2^2] = \sum_{i=k+1}^D \lambda_i \le \epsilon$ is bounded by an arbitrarily small threshold ($Q_k$ is the orthogonal projection onto $S_k$). Based on this, $I_t - I_0$ is projected into two orthogonal subspaces: inside $S_k$, $I_{t,k} = \bar\alpha_t I_{res} + \bar\beta_t Q_k(\epsilon)$ contains all residual information; in the complementary space $S_k^\perp$, $I_{t,k}^\perp = \bar\beta_t Q_k^\perp(\epsilon)$ is pure noise. Consequently, $q(I_t|I_0, I_{res})$ decomposes into "one RDDM process carrying all residuals + one pure DDPM noise process." The variational loss likewise decomposes as $L_{t-1} = L_{t-1,k}(\theta) + L_{t-1,k}^\perp(\theta)$. Since the KL divergence is non-negative ($L_{t-1,k}^\perp(\theta)\ge 0$), it follows that $L_{t-1}(\theta) \ge L_{t-1,k}(\theta)$. Because optimizing diffusion is equivalent to maximizing the VLB (minimizing $L_{t-1}$), and $L_{t-1,k}$ is the exact loss calculated only on the informative low-rank space $S_k$, this inequality provides the theoretical guarantee that restricting the generation process to the low-rank subspace inherently tightens the VLB—the pure noise chain in the complementary space is a wasted cost.

**2. Asymmetric Residual Diffusion Process: Low-rank constraints for forward signals, full-rank for reverse expressivity**

Directly moving the entire diffusion process into a low-rank space causes problems. If the residual is replaced by a low-rank version $I_{res}^{(k)}\triangleq Q_k(I_{res})$, the Gaussian transition mean of the forward process is constrained: $q(I_t|I_{t-1}) := \mathcal{N}(I_t; I_{t-1} + \alpha_t I_{res}^{(k)}, \beta_t^2 I)$, and the sampling at any $t$ is $I_t = I_0 + \bar\alpha_t Q_k(I_{res}) + \bar\beta_t\epsilon$. Importantly, even though the added signal is low-rank, $I_t$ itself remains full-rank due to the isotropic noise term $\bar\beta_t\epsilon$. Therefore, the true posterior $q(I_{t-1}|I_t, I_0, I_{res}^{(k)})$ is a full-rank isotropic Gaussian. Forcing the learned reverse kernel $p_\theta(I_{t-1}|I_t)$ into a low-rank subspace would create a fundamental model mismatch, as the low-rank parameterization cannot represent a full-rank posterior. LRDM's "asymmetry" addresses this: **The forward process only injects signals into $S_k$** (acting as a regularizer for the learned prior and preventing noise from contaminating background invariants), while **the reverse model $p_\theta$ is unrestricted, operating in the full-rank environment space $\mathbb{R}^D$** (preserving the capacity to approximate the full-rank posterior). This captures the benefits of a tighter VLB and inductive bias without sacrificing reconstruction fidelity.

**3. Adaptive Rank Selection: Dynamically changing subspace rank with diffusion time steps**

The asymmetric process defaults to a fixed rank throughout the forward pass, which is too rigid. Image structural statistics evolve over time—early stages are dominated by coarse, high-energy components, while later stages involve fine details and noise-like changes. LRDM decomposes residuals into a subspace with time-varying dimensions, introducing a rank scheduler $k(t)$ so that $I_t = I_0 + \bar\alpha_t Q_k^{(t)}(I_{res}) + \bar\beta_t\epsilon$. Four representative schedulers are presented: linear increasing $k_{LI}(t) = \lceil\frac{t}{T}R_m\rceil$, linear decreasing $k_{LD}(t) = \lceil(1-\frac{t}{T})R_m\rceil$, polynomial increasing $k_{PI}$, and polynomial decreasing $k_{PD}$ (using an envelope function $\text{envelope}(d;p) = 1 + a d^p + b d^{p+1} + c d^{p+2}$ where $d=t/T$). $R_m$ denotes the maximum rank. Experiments show that **Polynomial Decreasing (PD) is overall optimal**—it maintains low-rank regularization in early stages and provides sufficient capacity for detail restoration later, aligning perfectly with the inherent "coarse-to-fine" progression of diffusion.

### Loss & Training
The model follows RDDM by using two loss terms: residual prediction $\mathbb{E}[\|I_{res} - I_{res}^\theta(I_t, t, I_{in})\|^2]$ and noise prediction $\mathbb{E}[\|\epsilon - \epsilon_\theta(I_t, t, I_{in})\|^2]$. Training input $I_t$ is synthesized using the closed-form forward equation with $I_0$, $I_{res}$, and $\epsilon$. The key difference is that the forward residual is replaced by the projected low-rank residual $Q_k^{(t)}(I_{res})$, while the reverse network still predicts in full-rank space. The optimal configuration uses the PD scheduler with a low polynomial order $p$ (e.g., $p=1,2$).

## Key Experimental Results

### Main Results
Evaluations covered deraining (Raindrop, Rain1400), deblurring (GoPro, RealBlur-J/R), deshadowing (ISTD), and inpainting (CelebA-HQ), using PSNR/SSIM as metrics (both higher is better).

| Dataset | Metric | LRDM (Ours) | RDDM | Prev. SOTA |
|--------|------|--------------|------|--------------|
| Raindrop | PSNR / SSIM | 33.09 / **0.967** | 32.51 / 0.956 | Restormer 31.67 / 0.958 |
| Rain1400 | PSNR / SSIM | **34.39** / 0.954 | 32.21 / 0.952 | Restormer 33.68 / 0.939 |
| RealBlur-J | PSNR / SSIM | **30.21** / **0.933** | — | AdaRevD 30.12 / 0.894 |
| RealBlur-R | PSNR / SSIM | **37.92** / **0.976** | — | AdaRevD 36.53 / 0.957 |

On Raindrop, LRDM achieved the best SSIM and second-highest PSNR (the highest among all diffusion models). It reached the best overall performance on Rain1400. In both RealBlur benchmarks, LRDM secured the top scores among compared methods, with particularly significant gains in SSIM (e.g., 0.933 vs. 0.894 for RealBlur-J).

### Ablation Study
Ablations on fixed Rank (Fig. 6/7) and Adaptive Rank Scheduling (Tab. 3/4):

| Configuration | Raindrop PSNR/SSIM | Rain1400 PSNR/SSIM | Description |
|------|--------------------|--------------------|------|
| Fixed low-rank r=20/40 | — | — | Comparable to or better than full-rank baselines |
| lin increase | 32.95 / 0.9670 | 33.01 / 0.9535 | Linear Increasing |
| lin decrease | 32.92 / 0.9668 | 32.79 / 0.9534 | Linear Decreasing |
| poly increase | 32.69 / 0.9650 | 32.88 / 0.9516 | Polynomial Increasing |
| **poly decrease (PD)** | **33.03 / 0.9669** | **33.39 / 0.9540** | Overall optimal dynamic schedule |

Polynomial Order Ablation (PD): On Raindrop, $p=2$ achieved the best 33.09/0.9670; on Rain1400, $p=1$ achieved the best 34.39/0.9542. Generally, lower-order schedulers performed better.

### Key Findings
- **Existence of an Optimal Intermediate Rank**: Across all datasets, PSNR/SSIM rises quickly from very low ranks, peaks at a moderate rank (e.g., ~80 for Raindrop, ~100 for Rain1400), and then slightly decreases toward full-rank. This proves that moderate low-rank constraint is a beneficial inductive bias, whereas full-rank introduces redundant freedom.
- **Dynamic Rank is Superior, PD is Best**: Letting the rank decrease polynomially (starting high and ending low) best fits the coarse-to-fine diffusion process.
- **Low-Rank Residual Modeling is Universal**: It is effective across synthetic and real degradations (deraining, deblurring, deshadowing, inpainting) and achieves SOTA fidelity with fewer sampling steps.
- **Fixed Low-Rank is Sufficient; Dynamic Rank Provides Extra Gain**: Fixed r=20/40 is already competitive with full-rank baselines, indicating that most task-relevant residual structures are concentrated in a few principal components. Adding PD dynamic scheduling further enhances results.

## Highlights & Insights
- **Full-loop from "Observation" to "Theory" to "Implementation"**: The work observes low-rank residuals through spectral decay, proves it tightens the VLB, and turns it into a trainable mechanism with asymmetric diffusion.
- **"Asymmetry" is the Masterstroke**: Identifying that the posterior is full-rank despite the low-rank forward signal leads to the decision to only constrain the forward side. This insight on where to apply structural constraints without breaking distribution matching is transferable to other structural diffusion designs.
- **Low-Rank as a Free Inductive Bias**: Building the prior into the forward process rather than as an extra regularization term tightens the variational bound and potentially reduces sampling steps with almost zero extra inference cost.

## Limitations & Future Work
- **Reliance on the "Near-Domain" Assumption**: The low-rank residual hypothesis only holds when source and target domains are highly similar. It may fail in tasks with large domain gaps (e.g., extreme degradation, cross-modal translation).
- **Hand-crafted Rank Schedulers**: $k(t)$ is selected from four fixed types rather than being learned end-to-end. The optimal order $p$ needs to be tuned per dataset.
- **Sensitivity to Hyperparameters**: Results are sensitive to the maximum rank $R_m$ and thresholds, requiring specific selection for different tasks.
- **Idealized Theory**: The "ideal near-domain limit" where $\epsilon \to 0$ is only an approximation in real-world data.

## Related Work & Insights
- **vs RDDM [27]**: Both use residual diffusion, but RDDM operates in full-rank space. LRDM shows that near-domain residuals are inherently low-rank and constrains the forward pass accordingly, significantly outperforming RDDM.
- **vs DiffUIR / DeblurDiff**: These use selective hourglass structures or implicit blur kernels but retain full-rank generation. LRDM's subspace dimensionality approach is orthogonal to their architectural designs.
- **vs Traditional Low-Rank Methods (RPCA / Completion)**: Traditional methods use low-rank for denoising or alignment; LRDM is among the first to deeply integrate the low-rank observation of restoration residuals into a diffusion pipeline.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "near-domain low-rank residual + asymmetric diffusion + adaptive rank" perspective is novel and theoretically supported.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive multi-task coverage and ablation of ranks; however, some comparisons lack quantitative data on strictly unified sampling steps/speeds.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from observation to theory to mechanism; math-heavy sections require careful reading.
- Value: ⭐⭐⭐⭐ Provides an efficient and theoretically interpretable diffusion paradigm for near-domain restoration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Decoupled Residual Denoising Diffusion Models for Unified and Data Efficient Image-to-Image Translation](decoupled_residual_denoising_diffusion_models_for_unified_and_data_efficient_ima.md)
- [\[ICML 2025\] IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models](../../ICML2025/image_generation/intlora_integral_low-rank_adaptation_of_quantized_diffusion_models.md)
- [\[CVPR 2026\] Residual Diffusion Bridge Model for Image Restoration](residual_diffusion_bridge_model_for_image_restoration.md)
- [\[CVPR 2026\] ResCa: Residual Caching for Diffusion Transformers Acceleration](resca_residual_caching_for_diffusion_transformers_acceleration.md)
- [\[CVPR 2026\] Forecast the Principal, Stabilize the Residual: Subspace-Aware Feature Caching for Diffusion Transformers](forecast_the_principal_stabilize_the_residual_subspace-aware_feature_caching_for.md)

</div>

<!-- RELATED:END -->
