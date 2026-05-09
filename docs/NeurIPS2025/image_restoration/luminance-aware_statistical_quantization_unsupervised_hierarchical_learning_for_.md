---
title: >-
  [Paper Note] Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement
description: >-
  [NeurIPS 2025][Image Restoration][Low-light image enhancement] This paper proposes the LASQ framework, which reformulates low-light image enhancement (LLIE) as a statistical sampling process over hierarchical luminance distributions. By exploiting the power-law distribution inherent in natural luminance transitions, LASQ employs MCMC sampling to generate hierarchical luminance adaptation operators (LAOs) that are embedded into the forward process of a diffusion model, enabling fully unsupervised enhancement without requiring any normal-light reference images.
tags:
  - NeurIPS 2025
  - Image Restoration
  - Low-light image enhancement
  - diffusion model
  - power-law distribution
  - MCMC sampling
  - unsupervised learning
date: 2026-05-08
content_hash: 21971895b66b7500
---

# Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement

**Conference**: NeurIPS 2025
**arXiv**: [2511.01510](https://arxiv.org/abs/2511.01510)
**Code**: [GitHub](https://github.com/XYLGroup/LASQ)
**Area**: Low-Light Image Enhancement / Image Restoration
**Keywords**: Low-light image enhancement, diffusion model, power-law distribution, MCMC sampling, unsupervised learning

## TL;DR

This paper proposes the LASQ framework, which reformulates low-light image enhancement (LLIE) as a statistical sampling process over hierarchical luminance distributions. By exploiting the power-law distribution inherent in natural luminance transitions, LASQ employs MCMC sampling to generate hierarchical luminance adaptation operators (LAOs) that are embedded into the forward process of a diffusion model, enabling fully unsupervised enhancement without requiring any normal-light reference images.

## Background & Motivation

**Background**: LLIE methods are broadly divided into supervised (requiring paired data) and unsupervised approaches; recent integration of diffusion models has improved flexibility.

**Limitations of Prior Work**:
- Supervised methods overfit to pixel-level correspondences, neglecting the continuous physical process underlying luminance transitions.
- Unsupervised methods rely on pseudo-references (e.g., empirical gamma correction), inheriting their prior biases.
- Both paradigms oversimplify the fundamentally continuous and context-dependent luminance dynamics, leading to limited generalization.

**Key Challenge**: A tension exists between reconstruction fidelity and cross-scene generalization — optimizing for in-domain accuracy degrades generalization, while prioritizing generalization weakens in-domain performance.

**Goal**: To establish a statistical model for LLIE grounded in the physical laws of natural illumination, without requiring paired data.

**Key Insight**: An empirical observation that natural luminance transitions follow a power-law density distribution, which can be approximated by hierarchical power functions.

**Core Idea**: Reformulate LLIE from deterministic pixel mapping to a statistical sampling process over hierarchical luminance distributions.

## Method

### Overall Architecture

Three core components: (1) **Hierarchical Luminance Modeling** — constructing a luminance variation coordinate system and designing hierarchical LAOs; (2) **MCMC Sampling** — generating a coarse-to-fine collection of LAOs; (3) **Diffusion Model** — embedding the hierarchical samples into the forward process for unsupervised learning.

### Key Designs

1. **Luminance Variation Coordinate System**:

   - **Function**: Establishes a geometric framework for the relationship between low-light and normal-light luminance.
   - **Design Motivation**: To mathematically formalize the physical laws governing luminance transitions.
   - **Mechanism**: For each pixel $i$, the coordinate point $(I_L^{(i)}, I_N^{(i)})$ is observed to follow a power-law distribution $y = ax^\kappa$. Different values of $\kappa$ correspond to distinct adaptation strategies ($\kappa < 0.5$: dark-region recovery; $0.5 < \kappa < 1$: midtone enhancement; $\kappa \to 1$: highlight preservation).

2. **Hierarchical Luminance Adaptation Operator (LAO)**:

   - **Function**: Constructs multi-scale luminance correction operators ranging from global to local.
   - **Mechanism**: For a region $\mathcal{P}$, a luminance scalar $G_\mathcal{P}$ and the corresponding LAO are computed as:
     $$\gamma_\mathcal{P} = (\alpha + G_\mathcal{P})^{\beta_\mathcal{P}}, \quad \beta_\mathcal{P} = 2G_\mathcal{P} - 1 + \eta\frac{\sigma_{G_\mathcal{P}}^2}{\sigma_{G_\mathcal{P}}^2 + \delta}$$
   - **Distribution Modeling**: LAOs follow a truncated Gaussian distribution $\gamma \sim \mathcal{N}_{\text{trunc}}(\mu=\gamma_0, \sigma^2; \gamma_{\min}, \gamma_{\max})$.
   - **Physical Interpretation**: High-probability operators correspond to physically plausible global adaptations, while low-probability operators capture local fine-grained adjustments.

3. **MCMC Hierarchical Sampling**:

   - **Function**: Progressively samples from the LAO distribution space to generate a coarse-to-fine set of enhanced images.
   - **Mechanism**: The $n$-th iteration produces $2^{n-1}$ LAO configurations:
     $$p(\mathcal{I}_H^{(n)}) \approx \sum_{z=1}^{2^{n-1}} p(\mathcal{I}_H^{(n)}|\gamma_{\mathcal{P},z}^{(n)}) p(\gamma_{\mathcal{P},z}^{(n)})$$
     The transition kernel is a truncated Gaussian: $q(\gamma_z^{(n)}|\gamma_{z-1}^{(n)}) = \mathcal{N}_{\text{trunc}}(\gamma_z^{(n)}|\gamma_{z-1}^{(n)}, \lambda^2)$.
   - **Grid Strategy**: At iteration $n$, the image is partitioned into $m_n \times w_n$ non-overlapping patches (where $m_n = 2^{\lceil(n-1)/2\rceil}$), realizing a coarse-to-fine spatial progression.

4. **Hierarchically-Guided Diffusion**:

   - **Function**: Embeds the MCMC-sampled hierarchical enhancements into the diffusion forward process.
   - **Mechanism**: A time mapping $\psi(t) = \lfloor t \cdot N/T \rfloor$ aligns the $T$-step diffusion process with the $N$-level hierarchy. Within each time interval $T_n$, the corresponding $\mathcal{F}_H^{(\psi(t))}$ serves as the illumination-normalized reference.
   - **Training**: The objective combines a noise prediction loss $\mathcal{L}_d$ and a global label weak guidance loss $\mathcal{L}_g$.
   - **LASQ++ Extension**: An adversarial discriminator conditioned on unpaired normal-light references can optionally be incorporated:
     $$\mathcal{L}_{\text{total}} = \lambda_d\mathcal{L}_d + \lambda_g\mathcal{L}_g + \lambda_{\text{GAN}}\mathbb{E}[-\log\mathcal{D}_\phi(G_\theta(\mathcal{I}_L))]$$

### Loss & Training

- Noise prediction loss $\mathcal{L}_d$ (weight 0.9) + global guidance loss $\mathcal{L}_g$ (weight 0.005).
- Optional GAN loss (weight 0.7, LASQ++ mode).
- Adam optimizer, learning rate $2 \times 10^{-5}$, U-Net backbone, $T=1000$ diffusion steps.

## Key Experimental Results

### Main Results

Comparison on paired datasets (LOLv1 / LSRW):

| Type | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|--------|-------|-------|--------|
| SL | PyDiff | 23.275 | 0.859 | 0.108 |
| SL | SMG | 23.814 | 0.809 | 0.144 |
| UL | LightenDiffusion | 20.453 | 0.803 | 0.192 |
| UL | NeRCo | 19.738 | 0.740 | 0.239 |
| **UL** | **LASQ** | **20.375** | **0.814** | **0.191** |
| UL+ | LASQ++ | 20.481 | 0.807 | 0.205 |

No-reference datasets (DICM/NPE/VV) — the true strength of LASQ:

| Method | DICM NIQE↓ | NPE NIQE↓ | VV NIQE↓ |
|--------|-----------|----------|---------|
| LightenDiffusion | 3.724 | 3.618 | 2.941 |
| NeRCo | 4.107 | 3.902 | 3.765 |
| **LASQ** | **3.715** | **3.571** | **2.777** |

LASQ comprehensively outperforms all methods — including supervised ones — on no-reference datasets, demonstrating strong cross-scene generalization.

### Ablation Study

| Method | LOLv1 PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------------|-------|--------|
| Fixed Luminance Adj. | 16.741 | 0.715 | 0.273 |
| Limited Hierarchy (2 levels) | 19.139 | 0.792 | 0.243 |
| **LASQ (Full)** | **20.375** | **0.814** | **0.191** |

### Computational Efficiency

| Method | FLOPs (G) | Params (M) | Inference Time (ms) |
|--------|----------|-----------|---------------------|
| SCI | 0.13 | — | 50.14 |
| LightenDiffusion | 367.99 | 27.83 | 257.94 |
| **LASQ** | **219.75** | **24.08** | **213.89** |

LASQ preserves the performance advantages of diffusion models while achieving inference efficiency approaching non-diffusion methods.

### Key Findings

- Adaptive MCMC sampling substantially outperforms fixed luminance adjustment (PSNR gap of 3.6 dB).
- Intermediate hierarchy levels are indispensable — the two-level simplified variant improves over fixed adjustment but falls short of the full LASQ (PSNR gap of 1.2 dB).
- LASQ surpasses all supervised methods in no-reference scenarios, confirming the generalization advantage of physics-driven modeling.
- Incorporating normal-light references (LASQ++) improves in-domain color fidelity but may slightly reduce generalization.
- Low sensitivity to hyperparameters: PSNR variation remains below 0.3 dB across the tested ranges of $\alpha$, $\eta$, $\lambda_d$, and $\lambda_g$.

## Highlights & Insights

- **Physics-driven paradigm shift**: The first work to reformulate LLIE from deterministic pixel mapping to a statistical process grounded in the physical laws of natural luminance.
- **No paired data required**: MCMC sampling embedded in the diffusion forward process enables fully unsupervised training, fundamentally eliminating the dependence on paired data.
- **Superior generalization**: LASQ outperforms even supervised methods on no-reference datasets, demonstrating that physics-based priors generalize more effectively than data-driven mappings.
- **Dual-mode compatibility**: Seamlessly supports both settings — with and without normal-light references.
- **Power-law distribution discovery**: The empirical finding that natural luminance transitions follow a power-law distribution is itself a valuable contribution.

## Limitations & Future Work

- MCMC sampling increases training time, though it is not used at inference.
- The power-law assumption may not hold in extreme regions (e.g., pure black or pure white areas).
- The current static power-law parameterization has not been validated for time-varying scenarios such as video.
- The U-Net backbone could be replaced with more advanced denoising networks (e.g., DiT) for further performance gains.
- Hardware–software co-design for sensor-specific noise characteristics remains unexplored.

## Related Work & Insights

- Related to but fundamentally distinct from the "curve estimation" paradigm of Zero-DCE — LASQ performs statistical sampling rather than fitting a single curve.
- LightenDiffusion integrates Retinex theory into diffusion steps, whereas LASQ establishes a more general physical framework based on power-law distributions.
- The hierarchical MCMC sampling concept may generalize to other image degradation restoration tasks, such as dehazing and deraining.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Reformulates LLIE as a statistical sampling problem with a unique and empirically grounded theoretical perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage of paired/no-reference benchmarks, ablation studies, computational efficiency, and hyperparameter sensitivity.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous theoretical derivations and clear physical intuition, though mathematical notation is somewhat dense.
- **Value**: ⭐⭐⭐⭐⭐ Unsupervised, physics-driven, and highly generalizable — of significant practical value for real-world deployment without paired data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Outlier-Aware Post-Training Quantization for Image Super-Resolution](../../ICCV2025/image_restoration/outlier-aware_post-training_quantization_for_image_super-resolution.md)
- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](../../ICCV2025/image_restoration/low-light_image_enhancement_using_event-based_illumination_estimation.md)
- [\[AAAI 2026\] ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement](../../AAAI2026/image_restoration/iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i.md)
- [\[ICLR 2026\] ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting](../../ICLR2026/image_restoration/protots_learning_hierarchical_prototypes_for_explainable_time_series_forecasting.md)
- [\[ICCV 2025\] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](../../ICCV2025/image_restoration/lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)

</div>

<!-- RELATED:END -->
