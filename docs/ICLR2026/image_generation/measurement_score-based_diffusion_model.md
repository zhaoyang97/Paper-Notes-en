---
title: >-
  [Paper Note] Measurement Score-based Diffusion Model (MSM)
description: >-
  [ICLR 2026][Image Generation][Diffusion Model] Instead of attempting to learn the "score of clean images," this method directly learns a "local measurement score" for subsampled, noisy data in the **measurement domain**. By aggregating these via random masks, the full measurement is reconstructed, allowing diffusion models to be trained entirely on degraded observa
tags:
  - ICLR 2026
  - Image Generation
  - Diffusion Model
date: 2026-05-08
content_hash: 33dbb25f9463721d
---
# Measurement Score-based Diffusion Model (MSM)

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=pFByPVh6bd](https://openreview.net/forum?id=pFByPVh6bd)  
**Code**: [https://github.com/wustl-cig/MSM](https://github.com/wustl-cig/MSM)  
**Area**: Image Generation / Diffusion Models / Inverse Problems  
**Keywords**: Diffusion Models, Self-supervised, Measurement Domain, Training without Clean Data, MRI Reconstruction, Posterior Sampling  

## TL;DR
Instead of attempting to learn the "score of clean images," this method directly learns a "local measurement score" for subsampled, noisy data in the **measurement domain**. By aggregating these via random masks, the full measurement is reconstructed, allowing diffusion models to be trained entirely on degraded observations for both unconditional generation and linear inverse problems.

## Background & Motivation
**Background**: Score-based diffusion models sample from high-dimensional distributions by learning the score function (gradient of the log-density). They have achieved SOTA performance in natural and medical image generation and can be adapted for conditional sampling to solve inverse problems. However, training typically requires a vast amount of **clean ground-truth images**.

**Limitations of Prior Work**: In many scenarios, clean data is unavailable—high-resolution images are limited by hardware, and fully sampled MRI scans are time-consuming and uncomfortable for patients. Existing works (Ambient diffusion / SURE-score / GSURE diffusion) attempt to **directly approximate the score of clean images** from degraded data.

**Key Challenge**: Recovering the "full-image score" from degraded data is an **unnecessarily difficult** goal. Measurements naturally lie in a structured subspace; forcing the recovery of the full-image score is both unnatural and difficult. Worse, a corrupted image is not uniquely determined by its measurements—infinitely many images can map to the same degraded observation, creating ambiguity in the supervision signal. GSURE diffusion also has two major flaws: it was only validated on single-coil MRI (scaling to multi-coil requires computationally infeasible SVD on the full measurement operator) and requires the minimum diffusion noise $\sigma_0$ to match the measurement noise $\rho$, leading to severe degradation in sampling quality when $\rho$ is large.

**Goal**: To completely bypass "full-image score reconstruction" and train diffusion models **using only degraded measurements**, while supporting both unconditional generation and inverse problem solving.

**Core Idea (Learning in Measurement Domain)**: This work translates the successful patch-based learning concepts from the image domain to the **measurement domain**. Instead of learning the full score, it learns a "local measurement score restricted to the observed regions." A key advantage is that each subsampled measurement is **uniquely determined** by the acquisition operator (unlike corrupted images, which are ambiguous), meaning the model learns a physically well-defined denoising input.

## Method

### Overall Architecture
MSM consists of three parts: During training, it learns the **local measurement score** only on subsampled measurements (derived from a denoiser via Tweedie’s formula). For unconditional sampling, multiple local scores under random masks are **aggregated** in an expectation sense into an "MSM score" to iteratively reconstruct the full measurement $z$, which is then mapped back to the image. For inverse problems, a data fidelity gradient term is inserted into the sampling loop to transform unconditional sampling into posterior sampling.

```mermaid
flowchart TB
    subgraph Training
    A[Degraded measurement s=Sz<br/>Subsampling + Optional Noise] --> B[Add Diffusion Noise s_t=s+σ_t n]
    B --> C[Denoiser D_θ predicts ŝ_θ]
    C --> D[Tweedie formula yields<br/>Local measurement score S_θ]
    end
    subgraph Sampling
    E[Full measurement iteration z_t] --> F[Randomly draw w masks S^i<br/>to get local measurements s_t^i]
    F --> G[Individual local denoising ŝ_θ^i]
    G --> H[Weighted aggregation W·ΣS^iᵀŝ_θ^i<br/>→ MMSE estimate ẑ_θ]
    H --> I[Reverse diffusion step z_{t-1}]
    I -.Optional Data Fidelity.-> H
    end
```

### Key Designs

**1. Local measurement score: Restricting score learning to the observable subspace.** Assume the full measurement $z \in \mathbb{R}^n$ is uniquely determined by the image (for natural images $z=x$; for MRI, $z=FCx$, where $F$ is the Fourier transform and $C$ represents coil sensitivities; more generally, $z=Tx$ where $T$ is invertible). A mask $S \in \{0,1\}^{m \times n}$ ($m < n$) is drawn from a distribution $p(S)$ to obtain a subsampled measurement $s=Sz$. Diffusion noise is added to $s$ as $s_t = s + \sigma_t n$, and a denoiser outputs $\hat{s}_\theta(s_t; \sigma_t) = D_\theta(s_t; \sigma_t)$, trained with MSE loss. Once trained, the local score is obtained via Tweedie’s formula: $S_\theta(s_t; \sigma_t, S) = \frac{1}{\sigma_t^2}(\hat{s}_\theta(s_t; \sigma_t) - s_t)$, which is **explicitly conditioned** on the mask $S$ that generated $s_t$. The entire process never touches the full measurement $z$ or the clean image $x$, which is the source of the self-supervision.

**2. MSM score: Reconstructing the full score via expectation over random masks.** The target is the score on the full measurement $\nabla \log p_{\sigma_t}(z_t)$, but only local scores are learned. MSM defines this as the expectation over all masks: $\nabla \log q_{\sigma_t}(z_t) := W \mathbb{E}_{S \sim p(S)} [S^\top \nabla \log p_{\sigma_t}(s_t \mid S)]_{s_t=Sz_t}$, where the transpose $S^\top$ maps the local score back to the full measurement space. The weight vector $W = [\max(\mathbb{E}_S[\mathrm{diag}(S^\top S)], 1)]^{-1}$ compensates for overlapping contributions based on the inverse of the expected coverage frequency per coordinate, with the `max` operation preventing division by zero in unobserved regions. This aggregation can be interpreted as a **product-of-experts** model—each mask is an expert, and the MSM score is the score of their product model.

**3. Monte Carlo approximation: Unbiased estimation with $w$ masks.** Since the expectation is intractable, $w$ masks $S^{(i)}$ are randomly sampled at each step to compute an unbiased estimate $\nabla \log \hat{q}_{\sigma_t}(z_t) := W [\frac{1}{w} \sum_i S^{(i)\top} \nabla \log p_{\sigma_t}(s_t^{(i)} \mid S^{(i)})]$. During sampling (Algorithm 1), for each mask: perform local denoising $\rightarrow$ sample noise to estimate $s_t^{(i)} \sim p(s_t^{(i)} \mid \hat{s}_\theta^{(i)})$ $\rightarrow$ **replacement update** $z_t \leftarrow S^{(i)\top} s_t^{(i)} + (I - S^{(i)\top} S^{(i)}) z_t$. This ensures subsequent masks work on iterations that have already absorbed prior information. The $w$ random loops complementarily refine different regions. The final result is aggregated into an MMSE estimate $\hat{z}_\theta = W \sum_i S^{(i)\top} \hat{s}_\theta^{(i)} + \mathbf{1}_{C=0} \cdot \hat{z}_\theta$ (uncovered coordinates retain old values), which serves as the clean prediction for standard reverse diffusion. Theoretically, $D_{KL}(q \| \hat{q}) \le \frac{v^2}{w} C$; a larger number of random iterations $w$ yields a closer approximation to the ideal distribution.

**4. Posterior sampling: Solving inverse problems via data fidelity gradients.** For linear inverse problems $y = Hz + e$ ($A = HT$, where $H$ represents subsampling/blurring/inpainting/random projection), the posterior score is split into prior + likelihood: $\nabla \log p_{\sigma_t}(z_t \mid y) \approx \nabla \log \hat{q}_{\sigma_t}(z_t) + \gamma_t \nabla \|y - H \hat{z}_\theta\|_2^2$. Implementation-wise, one simply inserts a step $\hat{z}_\theta \leftarrow \hat{z}_\theta - \gamma_t \nabla_{\hat{z}_\theta} \|y - H \hat{z}_\theta\|_2^2$ after the aggregation in Algorithm 1. This allows the pre-trained MSM prior to be used for inpainting, super-resolution, and CS-MRI without retraining. Note that $H$ can differ from the random subsampling operator $S$ used during training.

**5. Training with noise + subsampling: Two regimes based on noise levels.** When observations are noisy $s = Sz + \nu, \nu \sim \mathcal{N}(0, \rho I)$, the diffusion noise $\sigma_t$ is compared with the measurement noise $\rho$. Case 1: If $\sigma_t > \rho$ (most steps in practice), residual noise is added $s_t \leftarrow s + \sqrt{\sigma_t^2 - \rho^2} n$, and the loss uses a "denoising more-noisy input with less-noisy reference" term plus a SURE loss $L_{\text{SURE}}$ (Stein’s Unbiased Risk Estimate) to teach the model to remove measurement noise. Case 2: If $\sigma_t \le \rho$, the model first denoises $s$ to get a pseudo-clean reference $\hat{s}_\theta(s; \rho)$, then adds diffusion noise and constrains consistency in non-subsampled regions. Since Case 1 is sampled more frequently, the pseudo-clean reference naturally improves during training, ensuring stability.

## Key Experimental Results

Setup: Standard Dhariwal & Nichol diffusion architecture, trained from scratch for 1M steps on a single A100. Data: 69k FFHQ faces ($128 \times 128$ RGB) and 2k fastMRI T2 slices ($256 \times 256$ complex multi-coil). 100 test images for inverse problems.

### Main Results: Unconditional Generation FID

| Data/Degradation | Method | Face FID↓ | MRI FID↓ |
|---|---|---|---|
| No Degradation (Upper Bound) | Oracle diffusion (Clean training) | 10.21 | 28.41 |
| Subsampling only ($\rho=0$) | **MSM** | **29.14** | **64.37** |
| Subsampling only ($\rho=0$) | Ambient diffusion | 55.90 | 70.07 |
| Subsampling + Noise ($\rho=0.1$) | **MSM** | **37.14** | **82.17** |
| Subsampling + Noise ($\rho=0.1$) | GSURE diffusion | 89.71 | (Infeasible for multi-coil) |

MSM achieves significantly lower FID than Ambient / GSURE across all "no clean data" settings. For the MRI multi-coil scenario, GSURE is absent due to the computational infeasibility of its SVD requirements.

### Inverse Problems: Natural Images + CS-MRI

| Task | Metric | Input | A-DPS | SSDU | **MSM** |
|---|---|---|---|---|---|
| Inpainting | PSNR↑ | 18.26 | 20.14 | — | **24.71** |
| Inpainting | LPIPS↓ | 0.304 | 0.305 | — | **0.076** |
| SR ×4 | PSNR↑ | 23.21 | 22.61 | — | **28.11** |
| SR ×4 | LPIPS↓ | 0.459 | 0.277 | — | **0.117** |
| CS-MRI ×4 | PSNR↑ | 22.75 | 27.28 | 29.65 | **30.71** |
| CS-MRI ×4 | LPIPS↓ | 0.306 | 0.173 | 0.160 | **0.145** |
| CS-MRI ×6 | PSNR↑ | 21.94 | 26.29 | 28.02 | **28.86** |
| CS-MRI ×6 | LPIPS↓ | 0.342 | 0.201 | 0.186 | **0.168** |

MSM outperforms A-DPS across the board in inpainting and SR (A-DPS sometimes performs worse than the input image in PSNR/SSIM because the Ambient prior struggles with details under box masks). In CS-MRI, MSM exceeds both the diffusion baseline A-DPS and the specialized self-supervised reconstruction method SSDU.

### Key Findings
- **Efficiency**: While A-DPS requires 1000 steps, MSM achieves better results in only 200 steps ($w=3$ for inverse problems, $w=1$ for generation).
- **Role of $w$**: Larger random iteration counts $w$ improve sampling quality (consistent with the KL bound), but $w=1$ is sufficient for generation.
- **Mask Generalization**: Inverse problems involving $H$ (e.g., box inpainting, bicubic SR) that differ from the training subsampling masks can still utilize the pre-trained MSM prior directly without retraining.

## Highlights & Insights
- **Changing the Problem is Smarter than Changing the Trick**: While others try to patch the loss to approximate the full-image score from degraded data, MSM switches to "learning local measurement scores and aggregating," turning a binary ill-posed target into a well-defined denoising problem.
- **Disambiguation in Measurement Domain**: Subsampled measurements are uniquely determined by the acquisition operator, whereas corrupted images are not. Training in the measurement domain naturally removes supervisory ambiguity—a key observation overlooked by prior work.
- **Unified Framework**: The same MSM score supports both unconditional generation and posterior sampling; inverse problems are handled by simply adding a data fidelity gradient, making it engineering-wise elegant.
- **Product-of-experts Perspective**: Interpreting random mask aggregation as the score of a composite likelihood model provides an elegant explanation for why local scores can approximate the full measurement score, supported by a KL convergence bound.

## Limitations & Future Work
- **Sampling Cost**: $w$ random iterations per reverse diffusion step increases compute time for large $w$, leading to a quality-time trade-off (though 200 steps is already more efficient than A-DPS's 1000).
- **Requirement for Invertible $z=Tx$**: The method relies on an invertible mapping between full measurements and images (e.g., $FC$ in MRI). How to generalize to non-linear or highly underdetermined acquisition operators remains unclear.
- **Ad-hoc Training Regimes**: The switch between Case 1 and Case 2 based on $\sigma_t$ vs $\rho$ and the reliance on SURE might lead to unstable pseudo-clean references during early training.
- **Limited scale validation**: Experiments are focused on FFHQ and fastMRI; validation on larger, more diverse datasets and more acquisition modalities is needed.

## Related Work & Insights
- **Diffusion Training without Clean Data**: Ambient diffusion (subsampling only), SURE-score / Daras 2024b (noisy only), and GSURE diffusion (noisy + subsampling)—MSM's fundamental difference is that it does not attempt to learn the full-image score.
- **Self-supervised Reconstruction**: SSDU / Robust SSDU use measurement subsets to supervise the training of end-to-end networks; MSM introduces similar self-supervised logic into a diffusion prior.
- **Solving Diffusion Inverse Problems**: DPS / A-DPS (posterior sampling using diffusion priors trained on degraded data); MSM's posterior sampling can be seen as applying the DPS data fidelity term to measurement-domain aggregated estimates.
- **Insight**: The idea of patch-based learning to improve scalability also holds in the measurement domain. "Restricting the learning objective to physically uniquely defined subspaces" is a transferable design principle that could inspire other generation or reconstruction tasks involving degraded observations.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Shifting score learning from the image domain to the measurement domain and using expectation aggregation to reconstruct the full score is a fundamental redefinition of the "training without clean data" problem, offering both novelty and theoretical grounding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers generation (FID) and inverse problems (inpainting/SR/CS-MRI) across natural images and multi-coil MRI. Baselines are highly relevant, though dataset scale and modal diversity are somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation progresses logically from the critique of learning full-image scores to disambiguation in the measurement domain. The algorithms and formulas are complete, though the dual-case noise training section is dense.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses the critical pain point of missing clean data in fields like medical imaging. The framework is unified, requires minimal engineering changes, and outperforms specialized self-supervised methods in MRI, indicating high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection](sample-efficient_evidence_estimation_of_score_based_priors_for_model_selection.md)
- [\[ICLR 2026\] Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](large_scale_diffusion_distillation_via_score-regularized_continuous-time_consist.md)
- [\[CVPR 2025\] Traversing Distortion-Perception Tradeoff Using a Single Score-Based Generative Model](../../CVPR2025/image_generation/traversing_distortion-perception_tradeoff_using_a_single_score-based_generative_.md)
- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](../../ICML2026/image_generation/alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICLR 2026\] Latent Diffusion Model without Variational Autoencoder](latent_diffusion_model_without_variational_autoencoder.md)

</div>

<!-- RELATED:END -->
