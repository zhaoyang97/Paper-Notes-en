---
title: >-
  [Paper Note] Angle Domain Guidance: Latent Diffusion Requires Rotation Rather Than Extrapolation
description: >-
  [ICML 2025][Image Generation][Classifier-Free Guidance] It is discovered that the root cause of color distortion in Classifier-Free Guidance (CFG) is the amplification of sample norms in the latent space. To address this, the Angle Domain Guidance (ADG) algorithm is proposed, which enhances guidance in the angle domain rather than the amplitude domain. By constraining norm variation while optimizing angular alignment, ADG eliminates abnormal color saturation under high guidan…
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Classifier-Free Guidance"
  - "Diffusion Models"
  - "Color Distortion"
  - "Angle Domain"
  - "Latent Space"
date: 2026-05-08
content_hash: 2eafe148a92a18a9
---

# Angle Domain Guidance: Latent Diffusion Requires Rotation Rather Than Extrapolation

**Conference**: ICML 2025  
**arXiv**: [2506.11039](https://arxiv.org/abs/2506.11039)  
**Code**: [https://github.com/jinc7461/ADG](https://github.com/jinc7461/ADG)  
**Area**: Image Generation  
**Keywords**: Classifier-Free Guidance, Diffusion Models, Color Distortion, Angle Domain, Latent Space

## TL;DR
It is discovered that the root cause of color distortion in Classifier-Free Guidance (CFG) is the amplification of sample norms in the latent space. To address this, the Angle Domain Guidance (ADG) algorithm is proposed, which enhances guidance in the angle domain rather than the amplitude domain. By constraining norm variation while optimizing angular alignment, ADG eliminates abnormal color saturation under high guidance weights while maintaining or even improving text-image alignment.

## Background & Motivation

**Background**: Classifier-Free Guidance (CFG) is a core technology in text-to-image diffusion models (such as Stable Diffusion) that enhances text-image alignment by linearly extrapolating conditional and unconditional score functions. CFG has become a standard technique for practical deployment.

**Limitations of Prior Work**: Under high guidance weights, although CFG significantly enhances text alignment, it simultaneously leads to severe color distortion, causing over-saturated and unnatural colors in images. Existing remedy methods (such as dynamic weighting schemes and auxiliary Langevin sampling) are still based on the linear combination framework, treating the symptoms rather than the root cause.

**Key Challenge**: The linear extrapolation of CFG inevitably amplifies sample norms. When the guidance weight $w > 1$, the linear extrapolation of conditional scores deviates the denoising path from the true distribution, manifested as an abnormal increase in sample norms in the latent space. Sample norm amplification $\rightarrow$ decoded pixel values pushed to extremes $\rightarrow$ color over-saturation.

**Goal**: To eliminate color distortion while maintaining high text alignment.

**Key Insight**: The authors observe that the latent space of VAE assumes samples approximately follow a high-dimensional Gaussian distribution. In high-dimensional Gaussians, samples almost entirely distribute along a spherical shell of a fixed radius (the concentration phenomenon). Therefore, the norm should not vary significantly; the genuinely meaningful information lies in the direction (angle).

**Core Idea**: To shift guidance from "amplitude domain extrapolation" (where linear extrapolation changes the norm) to "angle domain rotation" (which modifies direction only, preserving the norm). Specifically, the score function is transformed into an expectation in the angle domain $\rightarrow$ the conditional distribution is enhanced in the angle domain $\rightarrow$ and then mapped back to the score function for diffusion.

## Method

### Overall Architecture
ADG modifies the guidance mechanism during every denoising step of the diffusion model:
1. Compute the conditional score $\epsilon_\theta(x_t, c)$ and unconditional score $\epsilon_\theta(x_t, \varnothing)$.
2. Instead of performing linear extrapolation like CFG: $\tilde{\epsilon} = (1+w)\epsilon_c - w\epsilon_\varnothing$.
3. Operations are executed in the angle domain:
    - Decompose the predicted $x_0$ into its norm and direction.
    - Enhance the conditional signal along the direction (angle).
    - Prevent the norm from changing.
4. Reconstruct the guided score from the enhanced direction and preserved norm.

### Key Designs

1. **Theoretical Analysis of Norm Amplification**:

    - Function: Prove that the linear extrapolation in CFG inevitably leads to norm growth.
    - Core Argument: Let the conditional prediction be $\hat{x}_{0,c}$ and the unconditional prediction be $\hat{x}_{0,\varnothing}$. CFG yields $\hat{x}_0^{\text{CFG}} = (1+w)\hat{x}_{0,c} - w\hat{x}_{0,\varnothing}$. For $w > 0$, as long as the two predictions are not perfectly collinear, the norm of the extrapolated output is strictly greater than that of the conditional prediction: $\|\hat{x}_0^{\text{CFG}}\| > \|\hat{x}_{0,c}\|$.
    - Design Motivation: Theoretically explain the root of color distortion under high $w$—norm amplification forces VAE decoder outputs beyond normal operating ranges.
    - Abnormal Diffusion Phenomenon: Norm amplification accumulates progressively through the denoising process, causing the trajectory to deviate from the true manifold of data distribution.

2. **Angle Domain Guidance (ADG) Algorithm**:

    - Function: Perform direction rotation in the latent space without altering the norm.
    - Mechanism: Based on the high-dimensional Gaussian assumption of the VAE latent space—samples concentrate on a spherical shell of a fixed radius, with meaningful information encoded in the direction.
    - Concrete Steps:
      1. Derive the prediction of $\hat{x}_0$ from the current $x_t$ and the score estimate.
      2. Decompose it into direction $\hat{d} = \hat{x}_0 / \|\hat{x}_0\|$ and norm $r = \|\hat{x}_0\|$.
      3. Perform conditional enhancement within the directional space (on the sphere).
      4. Keeping $r$ constant, reconstruct $\hat{x}_0$ from the enhanced direction and the original norm.
    - Design Motivation: Angular components contain semantic details (content, structure), while the norm affects global traits (brightness, saturation). The goal is to enhance the former and preserve the latter.

3. **Theoretical Framework—Spherical Decomposition of the Score Function**:

    - Function: Convert the score function into an expectation over the joint distribution of norm and direction.
    - Mechanism: $\nabla \log p_t(x_t|c)$ can be decomposed into a radial component (controlling the norm) and a tangential component (controlling the direction). ADG enhances only the tangential component, which guarantees vector norm preservation while optimizing directional alignment.
    - Design Motivation: A rigorous mathematical framework validates the rationality of ADG.

### Loss & Training
- ADG is an inference-time method and requires no training.
- It directly replaces the guidance step of CFG and is compatible with any diffusion sampler.
- The computational overhead is identical to CFG (both requiring two score evaluations).
- Hyperparameter: guidance strength $w$ (carrying the same semantic meaning as in CFG).

## Key Experimental Results

### Main Results
Text-to-image generation using the SDXL model on the COCO dataset:

| Guidance Method | FID ↓ | CLIP Score ↑ | Saturation Deviation ↓ | HPSv2 ↑ |
|---------|-------|------------|-----------|---------|
| CFG (w=3) | 24.8 | 0.31 | 0.08 | 0.267 |
| CFG (w=7) | 28.5 | 0.33 | 0.25 | 0.251 |
| CFG (w=15) | 42.3 | 0.35 | 0.52 | 0.223 |
| Rescaled CFG (w=7) | 26.1 | 0.32 | 0.12 | 0.263 |
| **ADG (w=7)** | **22.3** | **0.34** | **0.05** | **0.278** |
| **ADG (w=15)** | **23.1** | **0.35** | **0.06** | **0.275** |

### Ablation Study

| Configuration | FID | CLIP | Saturation Deviation | Description |
|------|-----|------|----------|------|
| CFG w=7 (Baseline) | 28.5 | 0.33 | 0.25 | Standard CFG |
| Norm clipping only | 25.8 | 0.32 | 0.09 | Corrects color but decreases text alignment |
| Angle enhancement only (no norm constraint) | 27.2 | 0.34 | 0.18 | Partial improvement |
| **ADG (Angle enhancement + Norm preservation)** | **22.3** | **0.34** | **0.05** | Optimal on both aspects |
| ADG + DDIM sampling | 22.1 | 0.34 | 0.05 | Compatible with deterministic sampling |
| ADG + DDPM sampling | 22.5 | 0.34 | 0.05 | Compatible with stochastic sampling |

### Qualitative Comparison
- Low $w$ ($w=3$): ADG successfully aligns with the text context, whereas CFG still exhibits under-alignment.
- High $w$ ($w=15$): ADG retains stable color and quality, while CFG suffers from severe over-saturation distortion.
- At the same $w$, ADG achieves a higher effective guidance strength because no energy is wasted on norm amplification.

### Key Findings
- ADG maintains an FID of 23.1 at $w=15$, whereas CFG already deteriorates to 28.5 at $w=7$.
- The saturation deviation drops from 0.25 (CFG $w=7$) to 0.05 (ADG $w=7$)—almost completely eliminating color distortion.
- Under high $w$, ADG significantly outperforms CFG in HPSv2 (Human Preference Score v2), confirming human visual preferences.
- ADG not only rectifies CFG's flaws but also notably improves FID, demonstrating that eliminating norm amplification avoids distortion and elevates the overall quality of the generated distribution.
- Consistent effectiveness is demonstrated across multiple models, including SD 1.5, SDXL, and SD 3.0.

## Highlights & Insights
- The proposed concept of **"rotation rather than extrapolation"** is highly vivid and intuitive—the core contribution can be understood in a single sentence.
- Utilizing the concentration phenomenon of high-dimensional Gaussians to explain the latent space structure is particularly elegant—since VAE is designed to align latent distributions with Gaussians, the spherical shell assumption is highly reasonable.
- Perfect correspondence between theory and intuition: norm = brightness/saturation (global attributes), direction = content/structure (semantic information).
- Substantial yield improvement with identical computational cost compared to CFG—a genuine "free lunch."
- Stability at high $w$ means users can aggressively strive for higher text alignment without worrying about adverse side effects.

## Limitations & Future Work
- The high-dimensional spherical shell assumption might not hold if the VAE is poorly trained.
- The applicability to diffusion models without a VAE latent space (e.g., pixel-space diffusion) has not been analyzed.
- There is still room to explore specific forms of angle-domain enhancement (Ours employs a relatively simple rotation strategy).
- Mitigating effects on color flickering issues caused by CFG in video diffusion models have not been discussed.
- The combined effects with other CFG enhancement methods (such as Autoguidance) remain to be explored.

## Related Work & Insights
- **vs Rescaled CFG**: Scaled norm post-hoc, but also alters semantic information; ADG prevents norm growth at the source.
- **vs Dynamic CFG**: Dynamic weighting schemes remain within the linear extrapolation framework, whereas ADG bypasses linear extrapolation entirely.
- **vs Perp-Neg**: Performs orthogonal decomposition in the noise space, which is complementary to ADG performing spherical decomposition in the $x_0$ prediction space.
- **Insights**: The idea of angle-domain operation can be extended to other conditional generation tasks utilizing CFG (such as 3D generation, video generation, etc.).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The idea of angle domain guidance is simple, profound, and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple models (SD1.5/SDXL/SD3), diverse metrics, and thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Elegant combination of theory, intuition, and experiments.
- Value: ⭐⭐⭐⭐⭐ Offers universal value for all diffusion models that utilize CFG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Latent Diffusion Inversion Requires Understanding the Latent Space](../../CVPR2026/image_generation/latent_diffusion_inversion_requires_understanding_the_latent_space.md)
- [\[ICML 2025\] Visual Generation Without Guidance](visual_generation_without_guidance.md)
- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](../../ICCV2025/image_generation/whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)
- [\[ICML 2025\] GaussMarker: Robust Dual-Domain Watermark for Diffusion Models](gaussmarker_robust_dual-domain_watermark_for_diffusion_models.md)
- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](../../NeurIPS2025/image_generation/entropy_rectifying_guidance_for_diffusion_and_flow_models.md)

</div>

<!-- RELATED:END -->
