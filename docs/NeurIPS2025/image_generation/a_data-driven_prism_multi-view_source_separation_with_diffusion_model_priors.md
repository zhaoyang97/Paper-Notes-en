---
title: >-
  [Paper Note] A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors
description: >-
  [NeurIPS 2025][Image Generation][source separation] This paper proposes DDPRISM, a method that exploits structural differences among linear transformations across multi-view observations. Within an EM framework, it learns an independent diffusion model prior for each unknown source without requiring any isolated source samples, enabling source separation and posterior sampling. DDPRISM outperforms existing methods on both synthetic benchmarks and real galaxy observations.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "source separation"
  - "diffusion model"
  - "multi-view"
  - "expectation-maximization"
  - "Bayesian inverse problem"
date: 2026-05-08
content_hash: 3b0a8ecf18666ee7
---

# A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors

**Conference**: NeurIPS 2025
**arXiv**: [2510.05205](https://arxiv.org/abs/2510.05205)  
**Code**: [GitHub](https://github.com/swagnercarena/DDPRISM)  
**Area**: Diffusion Models / Scientific Computing / Signal Separation
**Keywords**: source separation, diffusion model, multi-view, expectation-maximization, Bayesian inverse problem

## TL;DR
This paper proposes DDPRISM, a method that exploits structural differences among linear transformations across multi-view observations. Within an EM framework, it learns an independent diffusion model prior for each unknown source without requiring any isolated source samples, enabling source separation and posterior sampling. DDPRISM outperforms existing methods on both synthetic benchmarks and real galaxy observations.

## Background & Motivation

**Background**: In the natural sciences, many observations are mixtures of multiple unknown sources—overlapping celestial objects in galaxy images, superimposed neural activities in EEG signals, or seismic signals mixed with background noise. Traditional source separation methods (ICA, NMF, template fitting) require strong prior assumptions or training samples from individual sources, creating a chicken-and-egg dilemma: separating sources requires priors, yet obtaining priors requires separated sources.

**Limitations of Prior Work**: (a) Contrastive learning methods (CPCA, CLVM, CVAE) assume the existence of "background views" containing only background sources—yet in many settings every view contains all sources; (b) existing methods either have limited expressive capacity (linear models) or cannot handle incomplete data (CVAE); (c) deep learning approaches (VAEs, etc.) require clean source samples for training.

**Key Challenge**: How can one learn the prior distribution of each source without any isolated source samples, when observations are noisy, incomplete, and acquired at different resolutions?

**Goal**
   - Learn the prior distribution of each source from mixed observations in an unsupervised manner.
   - Perform posterior sampling (i.e., source separation) given new observations.

**Key Insight**: The paper exploits the multi-view structure—different observation sets contain distinct linear transformations of the same sources (different mixing matrices), providing constraints for source separation. By combining the expressive power of diffusion models with iterative optimization under an EM framework, source priors can be learned progressively without any isolated source samples.

**Core Idea**: Under an EM framework, the E-step performs joint posterior sampling (source separation) using the current diffusion models, while the M-step trains improved diffusion models on the separated source samples. Iterating this procedure converges to the correct source priors.

## Method

### Overall Architecture
DDPRISM is an iterative framework alternating between an E-step and an M-step. The input consists of mixed observations from multiple views, $\mathbf{y}^\alpha = \sum_\beta \mathbf{A}^{\alpha\beta} \mathbf{x}^\beta + \eta$, with known mixing matrices $\mathbf{A}^{\alpha\beta}$ and noise covariances $\Sigma^\alpha$. The output is an independent diffusion model $d_{\theta^\beta}$ for each source $\beta$, which supports both prior sampling and posterior inference.

### Key Designs

1. **Problem Formulation (Multi-View Linear Source Separation)**

    - **Function**: Provides a unified formulation for diverse scientific signal separation problems.
    - **Mechanism**: Observations are modeled as $\mathbf{y}^\alpha_{i_\alpha} = \sum_{\beta=1}^{N_s} \mathbf{A}^{\alpha\beta}_{i_\alpha} \mathbf{x}^\beta_{i_\alpha} + \eta^\alpha_{i_\alpha}$, where different views $\alpha$ may have different observation dimensionalities $d_\alpha$. Mixing matrices may be rank-deficient (yielding incomplete data), and sources are assumed mutually independent. Key assumptions include known mixing matrices, source independence, and problem identifiability.
    - **Design Motivation**: Compared to contrastive learning methods that are limited to a two-source (background + target) setting, this formulation is substantially more general—supporting an arbitrary number of sources and views, allowing every view to contain all sources, and accommodating incomplete observations.

2. **M-Step: Independent Training of Source Diffusion Models**

    - **Function**: Updates each diffusion model using separated source samples.
    - **Mechanism**: By source independence, the overall optimization objective decomposes into independent per-source objectives: $\Theta_{k+1} = \arg\max_\Theta \sum_\beta \mathbb{E}[\log q_{\theta^\beta}(\mathbf{x}_0^\beta)]$. Each diffusion model is trained independently using the standard denoising score matching objective with denoiser parameterization.
    - **Design Motivation**: Source independence naturally decomposes the optimization, so each diffusion model attends only to its own source, making training computationally efficient.

3. **E-Step: Joint Posterior Sampling**

    - **Function**: Samples from the joint posterior $q_{\Theta_k}(\{\mathbf{x}^\beta\} | \mathbf{y}^\alpha, \{\mathbf{A}^{\alpha\beta}\})$ given the current diffusion models and observations.
    - **Mechanism**: The joint posterior score decomposes into a prior score (sum of individual diffusion model scores, by source independence) plus a likelihood score. The prior score is provided directly by each diffusion model. The likelihood score is approximated via Moment Matching Posterior Sampling (MMPS): the Tweedie formula is used to estimate $\mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t]$ and $\mathbb{V}[\mathbf{x}_0 | \mathbf{x}_t]$, enabling analytic evaluation of the likelihood integral under a Gaussian approximation. The key contribution is extending MMPS from single-source to multi-source joint sampling—the covariance in the likelihood becomes a weighted sum of all source variances: $\Sigma^\alpha + \sum_\beta \mathbf{A}^{\alpha\beta} \mathbb{V}[\mathbf{x}_0^\beta | \mathbf{x}_t^\beta] (\mathbf{A}^{\alpha\beta})^\top$.
    - **Design Motivation**: Jointly sampling all sources simultaneously (DDPRISM-Joint) is both more efficient and more effective than alternating Gibbs sampling (DDPRISM-Gibbs), as joint sampling exploits inter-source constraints.

4. **Multi-View Joint Utilization**

    - **Function**: Leverages constraints from all views for source separation.
    - **Mechanism**: During the E-step, likelihood scores are computed independently for each view and averaged across views—equivalent to joint maximum likelihood over all views. During the M-step, posterior source samples drawn from different views all contribute to training the corresponding diffusion models.
    - **Design Motivation**: Multiple views provide additional constraints. Even if a single view's mixing matrix is non-invertible (incomplete data), combining multiple views may render the problem identifiable.

### Loss & Training
- Diffusion model: variance exploding SDE with denoiser parameterization; predictor-corrector (PC) sampling.
- EM iteration: typically converges within 5–10 rounds.
- Likelihood score computation: MMPS approximation with conjugate gradient solvers for matrix inversion (avoiding explicit Jacobian computation).

## Key Experimental Results

### Main Results: 1D Manifold (Synthetic)

| Method | Posterior PSNR↑ | Posterior SD↓ | Prior SD↓ |
|--------|----------------|--------------|----------|
| PCPCA | 9.35 | 7.69 | 7.91 |
| CLVM-Linear | 9.58 | 5.80 | 5.86 |
| CLVM-VAE | 17.15 | 1.81 | 2.91 |
| DDPRISM-Gibbs | 12.66 | 3.96 | 3.92 |
| **DDPRISM-Joint** | **38.27** | **0.35** | **0.37** |

DDPRISM-Joint achieves a posterior PSNR of 38.27 (vs. 17.15 for CLVM-VAE) and a prior Sinkhorn distance of only 0.37.

### Ablation Study: 3-Source / Non-Contrastive Settings

| Setting | DDPRISM-Joint Posterior PSNR | Best Baseline PSNR |
|---------|-----------------------------|--------------------|
| Contrastive 2-source | 38.27 | 17.15 (CLVM-VAE) |
| Contrastive 3-source | 19.78 | 13.09 (CLVM-VAE) |
| Mixed ($f_\text{mix}=0.1$) | 24.15 | 17.69 (DDPRISM-Gibbs) |

Conventional methods support only two-source contrastive settings; DDPRISM handles three sources and non-contrastive mixed settings where all views contain all sources.

### Real Data: Galaxy Deblending
On real galaxy observations (GMNIST), DDPRISM-Joint comprehensively outperforms all baselines in both posterior and prior FID.

### Key Findings
- **DDPRISM-Joint substantially outperforms DDPRISM-Gibbs**: Joint sampling is far superior to Gibbs sampling, demonstrating the importance of exploiting inter-source constraints.
- **Non-contrastive settings are feasible**: The method remains effective even without "clean" views containing a single source—a critical advantage over all baselines.
- **Different observation dimensionalities are supported**: Observation dimensions may differ from source dimensions, and mixing matrices may be rank-deficient.
- **Stable EM convergence**: The method typically converges within 5–10 EM iterations.

## Highlights & Insights
- **Combining EM with diffusion-based posterior sampling for source separation** is a natural and elegant pairing: EM provides the iterative framework to resolve the chicken-and-egg problem, diffusion models provide expressive prior representations, and MMPS enables efficient posterior sampling.
- **The design of multi-source joint posterior sampling** (DDPRISM-Joint) is notably elegant: source independence decomposes the prior score, and the linear observation model makes the likelihood score analytically tractable, keeping overall complexity manageable.
- **Eliminating the requirement for contrastive views** breaks the core assumption of existing methods and substantially broadens applicability—a critical requirement in many scientific settings.
- **Strong practical utility**: The codebase is open-source and directly applicable to real-world problems in astronomy (galaxy deblending) and neuroscience (spike sorting).

## Limitations & Future Work
- **Known mixing matrices are required**: This is not blind source separation; the linear transformation of each source in each observation must be known, which is not always available in practice.
- **Linear mixing assumption**: Only linear mixtures $\mathbf{y} = \sum \mathbf{A}\mathbf{x} + \eta$ are supported; nonlinear mixing scenarios cannot be handled.
- **Identifiability assumption**: Not all mixing matrix configurations yield uniquely separable sources; the authors assume identifiability without providing sufficient conditions.
- **Local optima in EM**: Monte Carlo EM lacks a monotonic increase guarantee and may converge to local optima.
- **Computational cost**: Each EM round requires posterior sampling over all observations (requiring full diffusion reverse SDEs) followed by retraining the diffusion models; multiple iterations incur substantial computational overhead.

## Related Work & Insights
- **vs. CLVM-VAE**: CLVM uses a VAE as the source prior; DDPRISM uses a diffusion model—providing greater expressive capacity and supporting non-contrastive settings.
- **vs. PCPCA**: A linear method that can only learn target sources within a linear subspace, unable to capture complex nonlinear source distributions.
- **vs. DDPRISM-Gibbs**: The authors' own Gibbs variant alternately fixes one source while sampling the other; DDPRISM-Joint's simultaneous joint sampling yields significantly better results.
- **vs. Ambient Diffusion**: Ambient Diffusion trains diffusion models on incomplete data; DDPRISM extends this paradigm to source separation settings.
- **Transferable ideas**: The EM + diffusion posterior sampling framework may be applicable to other latent variable problems, such as mixture distribution learning and missing data imputation.

## Rating
- Novelty: ⭐⭐⭐⭐ Embedding diffusion-based posterior sampling into an EM framework for source separation is a novel combination, though each individual component draws on established techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic 1D manifolds, GMNIST, and real galaxy observations, but lacks large-scale high-resolution experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear, derivations are rigorous, and the logical chain from motivation to method is well articulated.
- Value: ⭐⭐⭐⭐ Has practical significance for scientific data analysis (astronomy, neuroscience, etc.) with strong methodological generality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data](geneman_generalizable_single-image_3d_human_reconstruction_from_multi-source_hum.md)
- [\[NeurIPS 2025\] Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation](diffusion-driven_progressive_target_manipulation_for_source-free_domain_adaptati.md)
- [\[CVPR 2025\] SIR-DIFF: Sparse Image Sets Restoration with Multi-View Diffusion Model](../../CVPR2025/image_generation/sir-diff_sparse_image_sets_restoration_with_multi-view_diffusion_model.md)
- [\[NeurIPS 2025\] A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)
- [\[CVPR 2026\] Cinematic Audio Source Separation Using Visual Cues](../../CVPR2026/image_generation/cinematic_audio_source_separation_using_visual_cues.md)

</div>

<!-- RELATED:END -->
