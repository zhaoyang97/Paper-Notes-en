---
title: >-
  [Paper Note] Residual Diffusion Bridge Model for Image Restoration
description: >-
  [CVPR 2026][Image Generation][Diffusion Bridge] This paper re-derives diffusion bridges as stochastic interpolations unified by a "mean-reverting OU process + Doob h-transform." It uses the **residual $\boldsymbol{\pi}=\mathbf{x}_0-\boldsymbol{\mu}$ of paired images to modulate noise injection and removal**, ensuring that the model only applies perturbations to degraded regions while protecting clean areas from iterative reconstruction. This approach achieves an average gain…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion Bridge"
  - "Residual Modulation"
  - "Universal Image Restoration"
  - "Doob h-transform"
  - "Adaptive Noise Perturbation"
date: 2026-05-08
content_hash: a475208c155e7248
---

# Residual Diffusion Bridge Model for Image Restoration

**Conference**: CVPR 2026  
**arXiv**: [2510.23116](https://arxiv.org/abs/2510.23116)  
**Code**: https://github.com/MiliLab/RDBM (Available)  
**Area**: Diffusion Models / Image Restoration  
**Keywords**: Diffusion Bridge, Residual Modulation, Universal Image Restoration, Doob h-transform, Adaptive Noise Perturbation

## TL;DR
This paper re-derives diffusion bridges as stochastic interpolations unified by a "mean-reverting OU process + Doob h-transform." It uses the **residual $\boldsymbol{\pi}=\mathbf{x}_0-\boldsymbol{\mu}$ of paired images to modulate noise injection and removal**, ensuring that the model only applies perturbations to degraded regions while protecting clean areas from iterative reconstruction. This approach achieves an average gain of 1.55 dB PSNR across five universal restoration tasks (deraining, low-light enhancement, desnowing, dehazing, and deblurring) while proving existing bridge models to be special cases of this framework.

## Background & Motivation

**Background**: Universal image restoration aims to handle multiple degradations such as denoising, deraining, dehazing, and super-resolution within a unified model. Diffusion models are currently the primary approach, following three main routes: (a) standard diffusion, which maps high-quality images to Gaussian noise and reverses from pure noise; (b) mean-reverting diffusion (e.g., IRSDE), where the forward endpoint converges near the degraded observation to preserve task cues; and (c) diffusion bridges (DDBM/BBDM/GOUB), which establish point-to-point probabilistic paths between known endpoints "degraded distribution ↔ clean distribution," offering stronger associations and higher fidelity.

**Limitations of Prior Work**: These three categories rely on **global noise perturbation** to construct probabilistic trajectories—noise is added to the entire image forward and removed holistically backward. This leads to two major drawbacks: (i) it **does not distinguish between regions with different levels of degradation**, forcing noise injection and reconstruction even on originally clean areas, where the reverse process inevitably accumulates errors; (ii) existing bridge models use differing formulas, lacking a unified analytical perspective to explain their relationships.

**Key Challenge**: There is a fundamental mismatch between the "global noise injection" of diffusion bridges and the "spatially uneven degradation" in image restoration. In restoration tasks, degradation is often local (raingrops only on some pixels, haze density varies spatially), yet bridge models process all pixels uniformly with a global noise intensity.

**Goal**: (1) Provide a unified SDE analytical framework for diffusion bridges with closed-form forward and reverse processes; (2) Make noise perturbations **spatially adaptive**—applying more noise and reconstruction to heavily degraded areas while leaving clean areas mostly untouched.

**Key Insight**: The authors observe that the diffusion coefficient $\boldsymbol{\pi}$ in bridge SDEs is traditionally set to a constant 1 (global uniform noise). By replacing it with the **pixel-wise residual** $\mathbf{x}_0-\boldsymbol{\mu}$ (clean image minus degraded image), pixels with large residuals (heavy degradation) receive strong noise, while pixels with zero residuals (already clean) receive no noise, naturally achieving spatial adaptivity.

**Core Idea**: Use the "residual between given distributions" to modulate noise injection and removal in diffusion bridges. This collapses the general global bridge SDE into a special case where the residual-to-noise ratio (RNR) is pixel-wise consistent and decays smoothly over time, achieving adaptive restoration of degraded regions and fidelity preservation for clean areas.

## Method

### Overall Architecture
The input to RDBM is a pair of images: a clean image $\mathbf{x}_0\sim p_{HQ}$ and a degraded image $\boldsymbol{\mu}\sim p_{LQ}$ (only $\boldsymbol{\mu}$ is available during inference); the output is the restored $\mathbf{x}_0$. Instead of a multi-module network, the method focuses on **rewriting the Stochastic Differential Equations (SDEs) underlying diffusion bridges**. The authors first generalize the diffusion term of a standard Ornstein–Uhlenbeck (OU) process with a coefficient $\boldsymbol{\pi}$ (Eq. 9), then apply the Doob h-transform to fix the forward endpoint at $\mathbf{x}_T=\boldsymbol{\mu}$ (removing the stationary noise $\lambda\epsilon$). This yields a generalized diffusion bridge unified by three parameters: $\lambda$, $\theta_t$, and $\boldsymbol{\pi}$ (Eq. 10), with a closed-form solution for $\mathbf{x}_t$ at any time (Eqs. 11–13). Crucially, setting $\boldsymbol{\pi} = \mathbf{x}_0-\boldsymbol{\mu}$ enables pixel-adaptive noise. The reverse process uses Bayes' theorem to derive a deterministic sampling formula (Eqs. 18–19), where a U-Net predicts the product of "residual × noise" $\boldsymbol{\pi}\epsilon$ in one go. Finally, Flow Matching, VE/VP bridges, Brownian bridges, and OU bridges are proven to be special cases under specific $(\theta_t,\lambda,\boldsymbol{\pi})$ configurations.

The process follows a standard "forward noise addition – reverse denoising" cycle (using closed-form $\mathbf{x}_t$ for training and U-Net iteration for reversal). Since the innovation lies in the SDE derivation rather than a complex pipeline, the designs are detailed via equations below.

### Key Designs

**1. Generalized Forward Process: Unified Bridge SDE via OU + Doob h-transform**

To address the lack of a unified perspective, the authors start from an OU process with an adjustable diffusion coefficient:
$$d\mathbf{x}_t=\theta_t(\boldsymbol{\mu}-\mathbf{x}_t)dt+\boldsymbol{\pi}\sigma_t d\omega_t,$$
where mean-reversion ensures the trajectory converges toward $\boldsymbol{\mu}$. Applying the Doob h-transform (where h is the log-transition kernel gradient from $t$ to $T$) anchors the endpoint at $\mathbf{x}_T=\boldsymbol{\mu}$. With a fixed drift-to-diffusion ratio $\lambda=\sigma_t^2/(2\theta_t)$, the generalized bridge SDE is:
$$d\mathbf{x}_t=\theta_t\coth(\overline{\theta}_{t:T})(\boldsymbol{\mu}-\mathbf{x}_t)dt+\sqrt{2\boldsymbol{\pi}^2\lambda\theta_t}\,d\omega_t,$$
where $\overline{\theta}_{s:t}=\int_s^t\theta_z dz$. This has an analytical solution where any state follows a Gaussian distribution with mean $\mathbb{E}[\mathbf{x}_t]=\boldsymbol{\mu}+(\mathbf{x}_0-\boldsymbol{\mu})\Theta_t$ and variance $\mathrm{Var}[\mathbf{x}_t]=\boldsymbol{\pi}^2\Sigma_t^2$, where $\Theta_t=\sinh(\overline{\theta}_{t:T})/\sinh(\overline{\theta}_{0:T})$ and $\Sigma_t^2=2\lambda\sinh(\overline{\theta}_{0:t})\sinh(\overline{\theta}_{t:T})/\sinh(\overline{\theta}_{0:T})$. The closed-form $\mathbf{x}_t$ allows for single-step sampling during training.

**2. Residual Modulated Noise: Pixel-Adaptive Perturbation via $\boldsymbol{\pi}=\mathbf{x}_0-\boldsymbol{\mu}$**

This is the core contribution. Equation 11 reveals the trajectory is a weighted mixture of a "residual term" and a "Gaussian noise term." To characterize its dynamics, the authors define the **residual-to-noise ratio (RNR)**—the ratio of residual energy to noise energy for pixel $(i,j)$ at time $t$:
$$R(i,j,t)=\frac{[x_0(i,j)-\boldsymbol{\mu}(i,j)]^2}{2[\boldsymbol{\pi}(i,j)]^2\lambda}\cdot\frac{\sinh(\overline{\theta}_{t:T})}{\sinh(\overline{\theta}_{0:t})\sinh(\overline{\theta}_{0:T})}.$$
While $\boldsymbol{\pi}=1$ in prior work causes uneven RNR across pixels and forces reconstruction on clean areas, setting **$\boldsymbol{\pi}=\mathbf{x}_0-\boldsymbol{\mu}$** allows the residual terms in the numerator and denominator to **cancel out**. This results in an RNR that is pixel-independent and decays smoothly over time $R(t)\propto\sinh(\overline{\theta}_{t:T})/[\sinh(\overline{\theta}_{0:t})\sinh(\overline{\theta}_{0:T})]$. Effectively, pixels with high residuals receive high variance $\boldsymbol{\pi}^2\Sigma_t^2$ (more reconstruction), while pixels with zero residuals have zero variance (protected from perturbation).

**3. Residual Bridge Score Matching: Deterministic Sampling + Single-Step Product Prediction**

The reverse process estimates $\mathbf{x}_{t-1}$ from $\mathbf{x}_t$ using a deterministic formula derived from Bayes' theorem:
$$\mathbf{x}_{t-1}=\boldsymbol{\mu}+\frac{\Theta_{t-1}}{\Theta_t}(\mathbf{x}_t-\boldsymbol{\mu})-\Big(\frac{\Theta_{t-1}}{\Theta_t}\Sigma_t-\Sigma_{t-1}\Big)\boldsymbol{\pi}\epsilon_t,$$
where $\boldsymbol{\pi}$ and $\epsilon_t$ appear as a product. A U-Net $\boldsymbol{\pi}_\epsilon^{\dot\theta}(\mathbf{x}_t,t,\boldsymbol{\mu})$ is trained to **predict the product $\boldsymbol{\pi}\epsilon$ directly**. Sampling starts from the degraded image $\boldsymbol{\mu}$ and iteratively applies Equation 19. Optimal performance is reached in only 10 steps (NFE=10).

**4. Unifying Existing Bridge Models**

The authors prove that standard diffusion processes are special cases of RDBM: $\boldsymbol{\pi}=0$ corresponds to Flow Matching; taking limits of $\theta_t\to0$ and $\lambda$ yields VE and VP bridges; $\boldsymbol{\pi}=1$ corresponds to Brownian/OU bridges (e.g., GOUB). This unification shows that the proposed residual coefficient $\boldsymbol{\pi}=\mathbf{x}_0-\boldsymbol{\mu}$ (31.04 dB) outperforms the standard $\boldsymbol{\pi}=1$ (30.15 dB) within the same framework.

### Loss & Training
Training simplifies KL divergence between distributions into a mean-matching $L_1$ loss:
$$\nabla_\theta\big\|\boldsymbol{\pi}\epsilon-\boldsymbol{\pi}_\epsilon^{\dot\theta}(\mathbf{x}_t,t,\boldsymbol{\mu})\big\|_1.$$
Training was conducted on 8×A800 GPUs using PyTorch for 500k iterations with a total batch size of 20 (distributed across tasks). Adam optimizer with a learning rate of 1e-4 was used on 256×256 random patches. The backbone is a U-Net scaled into T/S/B/L sizes (0.45M–7.73M parameters). The noise schedule uses cosine with stationary variance $\lambda=10/255$. Full-resolution inference uses 10 steps.

## Key Experimental Results

### Main Results
Performance comparison across five restoration tasks (deraining, low-light enhancement, desnowing, dehazing, and deblurring) on mixed datasets:

| Method | Year | Avg PSNR↑ | Avg SSIM↑ | Params(M) | FLOPs(G) |
|------|------|-----------|-----------|-----------|----------|
| GOUB | 2024 | 27.60 | 0.895 | 137.13 | 379.34 |
| ConvIR | 2024 | 29.49 | 0.903 | 14.82 | 128.93 |
| MaIR | 2025 | 29.51 | 0.904 | 20.71 | 110.44 |
| **RDBM-B** | - | **30.24** | **0.904** | **3.65** | **23.97** |
| **RDBM-L** | - | **31.04** | **0.917** | 7.73 | 32.93 |

RDBM-L significantly outperforms previous SOTA (MaIR) by **1.55 dB PSNR / 0.013 SSIM**. Notably, **RDBM-B achieves 30.24 dB with only 3.65M parameters**, outperforming the 20.71M MaIR with approximately 1/6 the parameters and 1/5 the FLOPs, highlighting the efficiency of residual modulation.

### Ablation Study

| Dimension | Optimal Value | Avg PSNR / SSIM | Note |
|---------|---------|-----------------|------|
| Noise Schedule | Cosine | 31.04 / 0.917 | Better than Linear (30.99) and Sigmoid (30.84) |
| Stationary Var $\lambda$ | 10/255 | 31.04 / 0.917 | 1/255 yields 30.36; 100/255 drops to 29.08 |
| NFE (Steps) | 10 | 31.04 / 0.917 | 2 steps: 22.81; 20/50/100 steps show slight decay |
| Coeff $\boldsymbol{\pi}$ | $\mathbf{x}_0-\boldsymbol{\mu}$ | 31.04 / 0.917 | $\boldsymbol{\pi}{=}0$ (FM): 28.21; $\boldsymbol{\pi}{=}1$ (Existing Bridge): 30.15 |

### Key Findings
- **Residual Modulation is the Primary Factor**: Changing $\boldsymbol{\pi}$ from 0 (Flow Matching) to 1 (Existing Bridges) to residuals increases average PSNR from 28.21 to 30.15 to 31.04. The residual version gains nearly 0.9 dB over $\boldsymbol{\pi}=1$, validating that adaptive injection is superior to global injection.
- **NFE=10 is the Sweet Spot**: Performance peaks at 10 steps and slightly decreases beyond that. In universal models, the output may deviate from a single reference image when over-sampled—a unique phenomenon in multi-task evaluation.
- **Excessive $\lambda$ Causes Degradation**: When $\lambda=100/255$, dehazing drops from 33.45 to 27.03, suggesting that excessive global noise overwhelms the adaptive signal of the residual modulation.

## Highlights & Insights
- **Elegant "Coefficient-Based" Innovation**: Without changing network architectures or adding modules, simply replacing the diffusion coefficient $\boldsymbol{\pi}$ (traditionally 1) with the residual $\mathbf{x}_0-\boldsymbol{\mu}$ converts the RNR from pixel-wise jumpy to globally smooth, naturally enabling region adaptivity.
- **Unification as a Theoretical and Empirical Strength**: Proving Flow Matching and various bridges as special cases not only organizes the diffusion bridge family but also gives the "residual vs. $\boldsymbol{\pi}=1$" ablation the weight of defeating all predecessors within a unified framework.
- **Transferable Efficiency Dividend**: RDBM-B's ability to exceed SOTA with 1/6 the parameters suggests that explicitly injecting degradation priors (residuals) into the diffusion coefficient relieves the network of learning where to modify and where to preserve.

## Limitations & Future Work
- **Dependency on Paired Residuals**: $\boldsymbol{\pi}=\mathbf{x}_0-\boldsymbol{\mu}$ requires strictly paired, pixel-aligned images during training. Constructing residuals for unpaired or weakly aligned real-world data remains unexplored.
- **"Biased" Performance on Multiple Degradations**: The authors noted that in samples with multiple degradations, the model tends to remove the primary degradation first, causing it to deviate from single-reference GTs and leading to lower metrics at higher NFEs.
- **Gaussian/Linear SDE Assumption**: The closed-form solutions are built on OU processes and Gaussian kernels, which may not hold for highly non-linear degradations like severe compression or complex motion blur.

## Related Work & Insights
- **vs. IRSDE**: IRSDE converges to a degraded image with stationary noise and uses global perturbations; RDBM removes the stationary noise at the endpoint and uses residual modulation for spatial adaptivity.
- **vs. GOUB / DDBM / BBDM**: These are special cases where $\boldsymbol{\pi}=1$. RDBM outperforms them by a wide margin (GOUB 27.60 vs. RDBM-L 31.04) while using fewer parameters.
- **vs. Flow Matching**: FM corresponds to $\boldsymbol{\pi}=0$ (no stochastic noise). Ablations show that some residual modulation is more effective for restoration than purely deterministic paths.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elegant innovation in diffusion coefficients that unifies the bridge model family.
- Experimental Thoroughness: ⭐⭐⭐⭐ Excellent multi-task/scale coverage, though unpaired settings are not addressed.
- Writing Quality: ⭐⭐⭐⭐ Rigorous derivation and clear special-case mapping, though highly mathematical.
- Value: ⭐⭐⭐⭐⭐ Strong methodological and practical value for generative image restoration with low parameter counts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EMR-Diff: Edge-aware Multimodal Residual Diffusion Model for Hyperspectral Image Super-resolution](emr-diff_edge-aware_multimodal_residual_diffusion_model_for_hyperspectral_image_.md)
- [\[CVPR 2026\] Low-Rank Residual Diffusion Models](low-rank_residual_diffusion_models.md)
- [\[CVPR 2026\] ResCa: Residual Caching for Diffusion Transformers Acceleration](resca_residual_caching_for_diffusion_transformers_acceleration.md)
- [\[CVPR 2026\] Decoupled Residual Denoising Diffusion Models for Unified and Data Efficient Image-to-Image Translation](decoupled_residual_denoising_diffusion_models_for_unified_and_data_efficient_ima.md)
- [\[CVPR 2026\] CARD: Correlation Aware Restoration with Diffusion](card_correlation_aware_restoration_with_diffusion.md)

</div>

<!-- RELATED:END -->
