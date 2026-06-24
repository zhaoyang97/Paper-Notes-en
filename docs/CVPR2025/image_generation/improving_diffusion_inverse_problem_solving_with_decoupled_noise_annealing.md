---
title: >-
  [Paper Note] Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing
description: >-
  [CVPR 2025][Image Generation][Inverse Problem Solving] This paper proposes Decoupled Annealed Posterior Sampling (DAPS). By decoupling the sample dependencies between adjacent steps during the diffusion sampling process, it allows large-scale non-local jumps to correct early sampling errors, substantially outperforming existing methods on non-linear inverse problems (e.g., phase retrieval).
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Inverse Problem Solving"
  - "Posterior Sampling"
  - "Noise Annealing"
  - "Diffusion Models"
  - "Image Restoration"
date: 2026-05-08
content_hash: df8dc2ce0d99e763
---

# Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing

**Conference**: CVPR 2025  
**arXiv**: [2407.01521](https://arxiv.org/abs/2407.01521)  
**Code**: [https://github.com/zhangbingliang2019/DAPS](https://github.com/zhangbingliang2019/DAPS)  
**Area**: Diffusion Models  
**Keywords**: Inverse Problem Solving, Posterior Sampling, Noise Annealing, Diffusion Models, Image Restoration

## TL;DR

This paper proposes Decoupled Annealed Posterior Sampling (DAPS). By decoupling the sample dependencies between adjacent steps during the diffusion sampling process, it allows large-scale non-local jumps to correct early sampling errors, substantially outperforming existing methods on non-linear inverse problems (e.g., phase retrieval).

## Background & Motivation

**Background**: Bayesian inverse problem solving based on diffusion models has become a mainstream approach. Existing methods (e.g., DPS, DDRM, DDNM, DiffPIR) modify the reverse SDE in each diffusion sampling step to incorporate measurement constraints into the denoising process, thereby sampling from the posterior distribution $p(\mathbf{x}_0|\mathbf{y})$.

**Limitations of Prior Work**: Existing methods sample from $p(\mathbf{x}_t|\mathbf{x}_{t+\Delta t}, \mathbf{y})$ at each denoising step. Because the step size $\Delta t$ of the reverse SDE is very small, $\mathbf{x}_t$ and $\mathbf{x}_{t+\Delta t}$ are forced to be close. This means only local corrections can be made at each step—allowing the correction of minor errors from the previous step, but failing to rectify global errors that require substantial updates.

**Key Challenge**: The strong coupling between sampling steps restricts the exploration range of the solution space. In non-linear inverse problems (such as phase retrieval), the measurement function is highly non-linear, which makes early steps prone to getting trapped in incorrect modes. Subsequent steps cannot escape these local minima, ultimately converging to erroneous solutions that are consistent with the measurements but have very low probability.

**Goal**: How can we allow sufficiently large changes between successive sampling steps to correct global errors while maintaining sampling quality?

**Key Insight**: The authors recognize that the key is that solving continuous reverse SDEs is unnecessary; instead, one can sample directly from the marginal temporal distribution $p(\mathbf{x}_t|\mathbf{y})$. By introducing an intermediate variable $\mathbf{x}_0$, $\mathbf{x}_t$ and $\mathbf{x}_{t+\Delta t}$ are decoupled, becoming conditionally independent given $\mathbf{x}_0$.

**Core Idea**: Replace the reverse SDE solver with a decoupled noise annealing process, recursively sampling from the marginal temporal distribution through a three-step cycle: "inverse diffusion $\to$ Langevin sampling $\to$ forward noise addition".

## Method

### Overall Architecture

DAPS does not solve reverse SDEs; instead, it recursively samples from the marginal temporal distribution $p(\mathbf{x}_t|\mathbf{y})$, with the noise level gradually annealing to zero. Given a noise schedule and time discretization $\{t_i\}$, starting from $\mathbf{x}_T \sim \mathcal{N}(0, \sigma_T^2 I)$, the method iteratively samples $\mathbf{x}_{t_i}$ by: (1) solving the ODE from $\mathbf{x}_{t_{i+1}}$ to obtain $\hat{\mathbf{x}}_0$; (2) sampling $\mathbf{x}_{0|y}$ from $p(\mathbf{x}_0|\mathbf{x}_t, \mathbf{y})$ using Langevin dynamics; (3) adding noise to $\mathbf{x}_{0|y}$ up to $\sigma_{t_i}$ to obtain $\mathbf{x}_{t_i}$. Finally, $\sigma_t \to 0$ yields the posterior sample.

### Key Designs

1. **Decoupled Noise Annealing Process**:

    - **Function**: Allows large-scale variations between consecutive sampling steps, breaking the local constraints of traditional methods.
    - **Mechanism**: The core proposition proves that if $\mathbf{x}_{t_1}$ follows the marginal temporal distribution $p(\mathbf{x}_{t_1}|\mathbf{y})$, then $\mathbf{x}_{t_2} \sim \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0|\mathbf{x}_{t_1}, \mathbf{y})}[\mathcal{N}(\mathbf{x}_0, \sigma_{t_2}^2 I)]$ also follows $p(\mathbf{x}_{t_2}|\mathbf{y})$. This makes it possible to jump from any noise level to another, where $\mathbf{x}_t$ and $\mathbf{x}_{t+\Delta t}$ are decoupled and conditionally independent given $\mathbf{x}_0$.
    - **Design Motivation**: The coupled sampling of traditional methods ($\mathbf{x}_t$ closely following $\mathbf{x}_{t+\Delta t}$) clearly demonstrates the problem of converging to wrong solutions in 2D synthetic experiments, whereas DAPS allows points on the trajectory to vary significantly, enabling escape from incorrect modes.

2. **Langevin Dynamics Sampling of $p(\mathbf{x}_0|\mathbf{x}_t, \mathbf{y})$**:

    - **Function**: Samples clean signals given the current noisy sample and measurement.
    - **Mechanism**: Utilizes Bayes' rule to decompose $p(\mathbf{x}_0|\mathbf{x}_t, \mathbf{y}) \propto p(\mathbf{x}_0|\mathbf{x}_t) p(\mathbf{y}|\mathbf{x}_0)$. Here, $p(\mathbf{x}_0|\mathbf{x}_t)$ is approximated by a Gaussian $\mathcal{N}(\hat{\mathbf{x}}_0(\mathbf{x}_t), r_t^2 I)$, where $\hat{\mathbf{x}}_0$ is estimated by solving the probability flow ODE. The Langevin update incorporates two gradient terms: a prior gradient (pulling towards $\hat{\mathbf{x}}_0$) and a likelihood gradient (pulling towards the measurement $\mathbf{y}$ consistency), plus a stochastic noise term.
    - **Design Motivation**: Although the Gaussian approximation is less precise than diffusion score estimation, experiments show that both achieve comparable performance, and the Gaussian approximation is substantially more computationally efficient. The computational overhead of Langevin dynamics mainly stems from evaluating the measurement function $\mathcal{A}$, which is much faster than evaluating diffusion models for image restoration tasks.

3. **LatentDAPS Latent Space Extension**:

    - **Function**: Extends DAPS to pretrained Latent Diffusion Models (LDMs) to support high-resolution image restoration.
    - **Mechanism**: Decomposes according to the probabilistic graphical model of latent diffusion, where Langevin updates are performed in the latent space: the prior gradient pulls towards $\hat{\mathbf{z}}_0$, and the likelihood gradient propagates measurement constraints back to the latent space via the decoder $\mathcal{D}$, i.e., $\nabla_{\mathbf{z}_0} \log p(\mathbf{y}|\mathcal{D}(\mathbf{z}_0))$.
    - **Design Motivation**: LDMs can leverage large-scale pretrained models (e.g., Stable Diffusion) to scale to high-resolution tasks. LatentDAPS is a natural extension of DAPS and displays better performance than existing latent space methods (such as PSLD and ReSample).

### Loss & Training

DAPS is an inference-time method that requires no training and directly utilizes pretrained diffusion models. DAPS-1K (for linear tasks) uses 44 ODE steps + 250 annealing steps, while DAPS-4K (for non-linear tasks) uses 10 ODE steps + 400 annealing steps. Langevin steps: 100 steps for DAPS, 50 steps for LatentDAPS. Learning rates are tuned individually for different tasks.

## Key Experimental Results

### Main Results

| Task | Dataset | Metric | DAPS | Prev. SOTA | Gain |
|------|--------|------|------|----------|------|
| Phase Retrieval | FFHQ 256 | PSNR↑ | 30.72 | 28.74 (DPS) | +1.98 |
| HDR Reconstruction | FFHQ 256 | PSNR↑ | 27.12 | 22.73 (DPS) | +4.39 |
| Super-Resolution 4× | FFHQ 256 | PSNR↑ | 29.07 | 28.66 (DCDP) | +0.41 |
| Gaussian Deblurring | FFHQ 256 | PSNR↑ | 29.19 | 28.63 (DCDP) | +0.56 |
| Random Inpainting | FFHQ 256 | PSNR↑ | 31.12 | 30.69 (DCDP) | +0.43 |
| CS-MRI | - | PSNR↑ | 31.49 | 28.79 (Prev. SOTA) | +2.70 |

DAPS achieves state-of-the-art results across almost all linear and non-linear tasks, with an particularly massive advantage in non-linear tasks.

### Ablation Study

| Configuration | PSNR (Phase Retrieval) | Description |
|------|----------------|------|
| DAPS (Gaussian Approximation) | 30.72 | Default configuration |
| DAPS (Diffusion Score Estimation) | ~30.7 | Comparable performance but computationally much heavier |
| DPS baseline | 28.74 | Coupled sampling limits correction capability |
| DAPS-1K (NFE=1000) | Suitable for linear tasks | Balance of efficiency and quality |
| DAPS-4K (NFE=4000) | Suitable for non-linear tasks | More thorough annealing |

### Key Findings

- Non-linear inverse problems represent the most advantageous scenario for DAPS, leading DPS by 1.98 dB and 4.39 dB in phase retrieval and HDR, respectively.
- The Gaussian approximation yields comparable performance to diffusion score estimation but with much higher efficiency, validating the practicality of the approximation strategy.
- DAPS achieves good results with around 100 neural network function evaluations (NFEs), exhibiting an efficiency-quality trade-off superior to competing methods.
- 2D synthetic experiments intuitively demonstrate that DAPS's trajectory variations are much larger than those of DPS, enabling a better approximation of the true posterior.

## Highlights & Insights

- **Highly elegant decoupling concept**: By introducing the intermediate variable $\mathbf{x}_0$, the sampling steps are decoupled. This is mathematically clean (demonstrated clearly with a single proposition) and highly effective in practice. This concept of "breaking the chain of dependency + annealing" can be generalized to other sequential sampling problems.
- **Connection to optimization methods**: Under specific parameter settings, DAPS degenerates into MAP estimation of optimization methods like ReSample. However, DAPS is fundamentally posterior sampling rather than point estimation, which is theoretically more complete.
- **Minimal overhead introduced by MCMC**: The computation of Langevin steps primarily consists of evaluating the measurement function rather than the diffusion model. This introduces almost zero additional overhead for image restoration tasks.

## Limitations & Future Work

- For each task, the Langevin learning rate and the prior/likelihood variance parameters $r_t$ and $\beta_y$ need to be manually tuned.
- Non-linear tasks require 4000 NFEs. Although this is fewer than intuitive expectations, there is still room for optimization.
- The Gaussian approximation may not be precise enough at high noise levels, though this did not become a bottleneck in the experiments.
- The authors conducted a preliminary exploration on discrete diffusion models (categorical data), which could be further extended to other data modalities.

## Related Work & Insights

- **vs DPS**: DPS adds a likelihood gradient constraint to the reverse SDE, performing local corrections at each step. DAPS is fully decoupled, allowing global jumps. DPS works reasonably well on linear tasks but collapses on non-linear tasks.
- **vs ReSample/DiffPIR**: These methods also alternate between denoising, optimization, and resampling, but are essentially MAP estimation. DAPS performs posterior sampling, from which these approaches can be derived as a degenerate case when $\beta_y \to 0$.
- **vs DDRM/DDNM**: These methods operate in the frequency domain via SVD, limiting them to linear inverse problems and rendering them unable to handle non-linear measurements.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The concept of decoupled noise annealing is elegant and profound, backed by solid theoretical support, representing a breakthrough in the inverse problem-solving paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 types of inverse problem tasks, pixel + latent spaces, two datasets, and comprehensive comparisons with numerous baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is clearly written, the visualization of the 2D synthetic experiment is highly intuitive, and the theoretical derivations are complete.
- Value: ⭐⭐⭐⭐⭐ Highly advances the field of diffusion inverse problems, opening up promising new possibilities particularly for non-linear inverse problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)
- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](../../NeurIPS2025/image_generation/a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)
- [\[ICML 2025\] Integrating Intermediate Layer Optimization and Projected Gradient Descent for Solving Inverse Problems with Diffusion Models](../../ICML2025/image_generation/integrating_intermediate_layer_optimization_and_projected_gradient_descent_for_s.md)
- [\[AAAI 2026\] Constrained Particle Seeking: Solving Diffusion Inverse Problems with Just Forward Passes](../../AAAI2026/image_generation/constrained_particle_seeking_solving_diffusion_inverse_problems_with_just_forwar.md)
- [\[CVPR 2025\] Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification](divide_and_conquer_heterogeneous_noise_integration_for_diffusion-based_adversari.md)

</div>

<!-- RELATED:END -->
