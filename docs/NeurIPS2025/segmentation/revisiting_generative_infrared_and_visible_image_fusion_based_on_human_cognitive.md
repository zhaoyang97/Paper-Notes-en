---
title: >-
  [Paper Note] HCLFuse: Revisiting Generative Infrared and Visible Image Fusion Based on Human Cognitive Laws
description: >-
  [NeurIPS 2025][Segmentation][Infrared-visible fusion] HCLFuse performs modality alignment via the information bottleneck principle and optimal transport theory…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Infrared-visible fusion"
  - "variational information bottleneck"
  - "optimal transport"
  - "physics-guided diffusion"
  - "Wasserstein distance"
date: 2026-05-08
content_hash: d79f9154b3ca5e3d
---

# HCLFuse: Revisiting Generative Infrared and Visible Image Fusion Based on Human Cognitive Laws

**Conference**: NeurIPS 2025
**arXiv**: [2510.26268](https://arxiv.org/abs/2510.26268)  
**Code**: [https://github.com/lxq-jnu/HCLFuse](https://github.com/lxq-jnu/HCLFuse)  
**Area**: Image Fusion / Generative Models
**Keywords**: Infrared-visible fusion, variational information bottleneck, optimal transport, physics-guided diffusion, Wasserstein distance

## TL;DR
HCLFuse performs modality alignment via the information bottleneck principle and optimal transport theory, combining a Variational Bottleneck Encoder (VBE) with a physics-guided conditional diffusion model. Three physical constraints—heat conduction, structure preservation, and physical consistency—are injected into the diffusion process. On the MSRS dataset, the gradient metric AG improves by 69.87% and spatial frequency SF improves by 39.41%.

## Background & Motivation

**Background**: Infrared-visible image fusion combines thermal information with texture details for scene understanding under low-light or occluded conditions. Generative methods based on GANs and diffusion models have become mainstream in recent years.

**Limitations of Prior Work**: Existing generative fusion methods lack an interpretable mechanism for modal information selection—specifically, how to balance infrared thermal information against visible-light texture. These methods are highly data-dependent and noise-sensitive, and the absence of physical constraints in the generation process leads to artifacts.

**Key Challenge**: Fusion must simultaneously preserve critical information from both modalities while suppressing redundancy, yet existing methods perform information selection implicitly and without controllability.

**Goal**: To provide a theoretical foundation—information bottleneck combined with optimal transport—to guide information selection during fusion, and to employ physical constraints to regularize the generation process.

**Key Insight**: Information bottleneck theory offers a theoretical framework for fusion (retaining sufficient information while compressing redundancy); optimal transport aligns modality distributions; and physical laws (heat conduction, structure preservation) constrain the diffusion process.

**Core Idea**: Information bottleneck + optimal transport for modality-aligned encoding → physics-guided (heat conduction + structure preservation + physical consistency) conditional diffusion model for fused image generation.

## Method

### Overall Architecture
Infrared image $X$ + visible image $Y$ → **Optimal Transport Alignment**: Sinkhorn divergence minimization to obtain mapping $T^*(X) = P^* \cdot X_{flat}$ → **VBE Encoding**: multi-scale masked feature extraction → Gaussian modeling $q(Z|F_m) \sim \mathcal{N}(\mu, \sigma^2)$ → latent representation $Z = \mu + R$ → **Physics-Guided Diffusion**: injection of heat conduction, structure preservation, and physical consistency constraints into the denoising process → fused image.

### Key Designs

1. **Variational Bottleneck Encoder (VBE) + Optimal Transport**:

    - **Function**: Aligns the two modalities and encodes them into a compact latent representation.
    - **Mechanism**: The optimal transport map $T^*$ aligns the infrared distribution to the visible distribution via Sinkhorn divergence minimization. The VBE loss is $\mathcal{L}_{VBE} = -\mathbb{E}[\log p(Y|Z)] - \alpha\mathbb{E}[\log p(X'|Z)] + \beta D_{KL}[q(Z|X',Y) \| p(Z)]$, combining dual-modality reconstruction with KL regularization. Multi-scale learnable masks $M_s = \sigma(w_s)$ control information selection at each scale.
    - **Design Motivation**: Theorem 1 establishes that the mutual information lower bound is related to the Wasserstein distance—optimal transport alignment tightens this bound, enabling the encoder to retain more task-relevant information.

2. **Physics-Guided Conditional Diffusion Model**:

    - **Function**: Injects three physical constraints into the diffusion denoising process.
    - **Mechanism**: (a) Heat conduction: $\Phi_{heat} = \hat{z}_0 + \lambda_{heat}(t) \nabla^2 \hat{z}_0$—Laplacian smoothing of the thermal distribution; (b) Structure preservation: $\Phi_{stru} = \hat{z}_0^{heat} + \lambda_{stru}(t)(G_{max} - G_{\hat{z}_0}) M_{stru}$—gradient enhancement to protect visible-light edges; (c) Physical consistency: $\Phi_{con} = \hat{z}_0^{stru} + \lambda_{con}(t)(w_{ir} X M_{heat} + w_{vis} Y M_{stru})$—incorporates original modality information.
    - **Design Motivation**: The time-varying guidance coefficient $\lambda_i(t) = \lambda_i^0 e^{-\gamma t}$ provides strong guidance early (coarse structure) and weak guidance late (fine details generated autonomously by the model). Physical constraints replace non-interpretable loss weight tuning.

3. **Multi-Scale Learnable Masks**:

    - **Function**: Adaptively selects modality information at different scales.
    - **Mechanism**: $F_m = \sigma(\theta_s \cdot (M_s \odot F_s))$, where $M_s = \sigma(w_s)$ are learnable parameters.
    - **Design Motivation**: Different regions require information at different scales—thermal target regions favor coarse-scale infrared features, while texture regions favor fine-scale visible-light features.

### Loss & Training
- VBE loss = dual-modality reconstruction + KL regularization
- Modified physics-aware diffusion denoising: $p_\theta^{phys}(z_{t-1}|z_t) \approx \mathcal{N}(\mu_\theta + \Delta\mu_{phys}, \Sigma_\theta)$
- DDIM sampling for acceleration

## Key Experimental Results

### Main Results (MSRS Dataset)

| Metric | Second-Best | HCLFuse | Gain |
|--------|------------|---------|------|
| AG (Gradient) | 3.78 | **6.44** | +69.87% |
| SF (Spatial Frequency) | 12.84 | **17.90** | +39.41% |
| DF (Discrete Frequency) | 4.61 | **7.64** | +65.56% |
| QSF (Quaternion) | 0.47 | **0.54** | +14.89% |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| W/O physics guidance (W/O TPG) | Unstable generation |
| W/O VBE | Visual artifacts (buildings/sky) |
| W/O optimal transport | Sharp decline across all metrics |
| W/O DDIM | Quality degradation |
| Full model | Best performance |

### Key Findings
- Optimal transport alignment is the most critical component—its removal causes a sharp decline across all metrics.
- Physics guidance stabilizes the generation process—without it, the diffusion model readily produces artifacts.
- Consistent improvements are observed on TNO and FMB datasets, demonstrating strong generalizability.
- Downstream semantic segmentation also benefits, indicating that fusion quality directly impacts high-level tasks.

## Highlights & Insights
- **Solid Theoretical Foundation**: The combination of information bottleneck and optimal transport is not a simple aggregation; Theorem 1 rigorously demonstrates how OT alignment tightens the mutual information bound, providing theoretical guidance for information selection in fusion.
- **Physical Constraints Replace Hyperparameter Tuning**: Traditional methods require manual adjustment of infrared/visible-light weights, whereas physics-guided constraints (heat conduction/structure preservation) offer a more interpretable and automated alternative.
- **Well-Motivated Time-Varying Guidance Coefficients**: The exponential decay naturally aligns with the coarse-to-fine generation process of diffusion models, preserving global structure early and fine details late.

## Limitations & Future Work
- The diffusion process incurs significant computational overhead, making real-time application challenging.
- The method requires spatially aligned infrared-visible image pairs.
- The physical constraints assume a specific heat conduction model, which may not generalize to all scenarios.
- Performance on fully unaligned or corrupted image pairs has not been validated.

## Related Work & Insights
- **vs. DiffFuse**: Diffusion-based fusion without physical constraints, prone to unnatural artifacts.
- **vs. CDDFuse**: Fusion based on decorrelation strategies; information selection is less principled than the information bottleneck formulation.
- **vs. TarDAL**: Detection-driven fusion; HCLFuse's physics-guided approach is more general-purpose.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Triple innovation combining information bottleneck + optimal transport + physics-guided diffusion.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset evaluation, ablation studies, and downstream task assessment.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous theoretical derivations.
- **Value**: ⭐⭐⭐⭐ Establishes a theoretically interpretable new paradigm for generative image fusion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CtrlFuse: Mask-Prompt Guided Controllable Infrared and Visible Image Fusion](../../AAAI2026/segmentation/ctrlfuse_mask-prompt_guided_controllable_infrared_and_visible_image_fusion.md)
- [\[NeurIPS 2025\] HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance](haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)
- [\[NeurIPS 2025\] HumanCrafter: Synergizing Generalizable Human Reconstruction and Semantic 3D Segmentation](humancrafter_synergizing_generalizable_human_reconstruction_and_semantic_3d_segm.md)
- [\[NeurIPS 2025\] Panoptic Captioning: An Equivalence Bridge for Image and Text](panoptic_captioning_an_equivalence_bridge_for_image_and_text.md)
- [\[NeurIPS 2025\] HopaDIFF: Holistic-Partial Aware Fourier Conditioned Diffusion for Referring Human Action Segmentation in Multi-Person Scenarios](hopadiff_holistic-partial_aware_fourier_conditioned_diffusion_for_referring_huma.md)

</div>

<!-- RELATED:END -->
