---
title: >-
  [Paper Note] Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis
description: >-
  [ICCV 2025][Image Generation][diffusion models] Omegance proposes scaling the noise prediction in each denoising step of a diffusion model by a single parameter $\omega$, enabling training-free global, spatial, and temporal control over the detail granularity of generated images and videos. The method is architecture-agnostic and compatible with SDXL, SD3, FLUX, and other models.
tags:
  - ICCV 2025
  - Image Generation
  - diffusion models
  - granularity control
  - noise scaling
  - training-free
  - detail enhancement/suppression
date: 2026-05-08
content_hash: 1f87e133478d6f07
---

# Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis

**Conference**: ICCV 2025
**arXiv**: [2411.17769](https://arxiv.org/abs/2411.17769)
**Code**: [https://github.com/itsmag11/Omegance](https://github.com/itsmag11/Omegance)
**Area**: Diffusion Models / Image Generation
**Keywords**: diffusion models, granularity control, noise scaling, training-free, detail enhancement/suppression

## TL;DR
Omegance proposes scaling the noise prediction in each denoising step of a diffusion model by a single parameter $\omega$, enabling training-free global, spatial, and temporal control over the detail granularity of generated images and videos. The method is architecture-agnostic and compatible with SDXL, SD3, FLUX, and other models.

## Background & Motivation

**Background**: Diffusion models have become the dominant paradigm for high-quality image synthesis, yet they lack direct, fine-grained control over the detail granularity of generated outputs. Users frequently need to selectively adjust the level of detail across different regions during creative workflows.

**Limitations of Prior Work**: (a) Text prompts cannot precisely express desired detail levels (e.g., "reduce background detail while preserving high detail in the subject" is difficult to convey through prompting); (b) existing quality enhancement methods (e.g., FreeU, SAG/PAG) only support global enhancement and lack spatially fine-grained control; (c) RLHF-based fine-tuning approaches are costly and inflexible; (d) FreeU is tightly coupled to the U-Net architecture and is not applicable to newer architectures such as DiT.

**Key Challenge**: The uniform denoising process in diffusion models does not permit imposing different levels of detail control on different regions within the same image, and the SNR schedule is fixed throughout generation.

**Goal**: To achieve, in the simplest possible manner: (a) global detail enhancement/suppression; (b) spatially region-specified granularity control; and (c) temporally stage-dependent granularity control.

**Key Insight**: Noise scaling is a fundamental operation in diffusion models, yet it has never been systematically explored as a means of granularity control. The authors find that simply scaling the noise prediction effectively modifies the SNR, thereby controlling the retention of high-frequency versus low-frequency information.

**Core Idea**: Multiply the noise prediction by $\omega$ at each denoising step — $\omega < 1$ retains more high-frequency information to produce richer detail, while $\omega > 1$ removes more high-frequency content to yield smoother outputs.

## Method

### Overall Architecture
Omegance applies a scaling factor $\omega$ to the noise prediction at every step of the diffusion model's reverse denoising process. The standard denoising step is $z_{t-1} = \delta_t \cdot z_t + \zeta_t \cdot \epsilon_\theta(z_t, t)$, which Omegance modifies to $z'_{t-1} = \delta_t \cdot z_t + \zeta_t \cdot \epsilon_\theta(z_t, t) \cdot \omega$. The method requires no architectural modifications, no retraining, and introduces negligible computational overhead.

### Key Designs

1. **Global Omega Control**:

    - Function: Uniformly enhances or suppresses detail across the entire image.
    - Mechanism: Using DDIM as an example, the modified SNR is:
      $$\text{SNR}(t-1)' = \frac{\alpha_{t-1}}{[\frac{\sqrt{\alpha_{t-1}}\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}} + \omega(\frac{\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}} - \sqrt{\alpha_{t-1}}\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}})]^2}$$
      Since $\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}} - \sqrt{\alpha_{t-1}}\sqrt{1-\alpha_t}$ is always negative, $\omega < 1$ yields $\text{SNR}' < \text{SNR}$ (retaining more high-frequency content), while $\omega > 1$ yields $\text{SNR}' > \text{SNR}$ (removing more high-frequency content).
    - Design Motivation: Controlling high-frequency information via effective SNR modification provides a physically interpretable mechanism.

2. **Omega Mask (Spatial Control)**:

    - Function: Assigns different $\omega$ values to different spatial regions of the image.
    - Mechanism: $\omega_{i,j} = \mathcal{M}(i,j)$, where $\mathcal{M} \in \mathbb{R}^{H' \times W'}$ is a mask of the same spatial dimensions as the denoising latent. Owing to the locality of the denoising process, adjusting $\omega$ in one region does not affect the SNR of neighboring regions.
    - Mask Sources: User-drawn strokes, segmentation masks, ControlNet signals (e.g., pose skeletons → subject masks, depth maps → foreground/background masks), or continuous depth values.
    - Design Motivation: Practical applications frequently require high subject detail with low background detail, or vice versa. The spatial mask provides an intuitive and flexible interface.

3. **Omega Schedule (Temporal Control)**:

    - Function: Applies different $\omega$ values at different denoising stages.
    - Mechanism: $\omega_t = \mathcal{S}(t)$, exploiting the denoising dynamics of diffusion models — early stages ($t \in [T, \tau]$) form the layout and structure, while later stages ($t \in [\tau, 0]$) refine details and textures. A lower $\omega$ in early stages increases layout complexity; a lower $\omega$ in later stages enhances textural detail. The layout formation phase spans only approximately the first ~10 steps out of 50 total.
    - Design Motivation: Layout and texture correspond to distinct denoising stages, and temporal control allows independent adjustment of the complexity of each.

4. **Adaptation to Different Schedulers**:

    - DDIM/Euler: Directly multiply $\epsilon_\theta$ by $\omega$.
    - Flow matching (e.g., FLUX): A mean-preserving operation is required to avoid color shift: $z'_{t-dt} = z_t + [(dt \cdot v_\theta(z_t,t) - m) \cdot \omega + m]$, where $m = \mathbb{E}[dt \cdot v_\theta(z_t,t)]$.

### Practical Notes
In practice, $\omega$ is reparameterized via $\omega = \mathcal{R}(\varpi)$, centering the user-facing input $\varpi \in (-\infty, \infty)$ at zero for more intuitive adjustment.

## Key Experimental Results

### Main Results (SDXL T2I Quantitative Evaluation)

| Method | FID↓ | IS↑ | CLIP↑ | Q-Align↑ | PickScore↑ |
|--------|------|-----|-------|----------|-----------|
| SDXL (baseline) | 162.18 | 13.23 | 32.88 | 4.68 | 0.1468 |
| + FreeU | 167.22 | 12.25 | 31.76 | 4.64 | 0.0967 |
| + Cosine Sch. | 182.06 | 11.38 | 30.78 | 2.88 | 0.0376 |
| + Rescaled Sch. | 163.29 | 10.88 | 28.80 | 3.25 | 0.0295 |
| + **Omegance** $\varpi(6.0)$ | **157.47** | **13.82** | 32.70 | 4.64 | 0.1149 |
| + **Omegance** $\varpi(-6.0)$ | 170.52 | 13.01 | 32.81 | 4.67 | **0.1601** |
| + EXP1 schedule | 173.49 | 12.67 | 32.70 | 4.64 | 0.1578 |
| + COS1 schedule | 159.87 | 13.25 | 32.64 | 4.60 | 0.0962 |

Omegance outperforms FreeU and Cosine/Rescaled schedulers across all key metrics.

### Frequency-Domain Analysis (Detail Change Quantification)

| Setting | SSIM↑ | HFE Change |
|---------|-------|-----------|
| SDXL baseline | 1.0 | 0 |
| $\varpi(6.0)$ detail suppression | 0.8124 | −204.2 |
| $\varpi(-6.0)$ detail enhancement | 0.7940 | +520.7 |
| EXP1 (layout+detail enhanced) | 0.7087 | +1113.1 |
| EXP2 (layout enhanced+detail suppressed) | 0.6926 | +205.8 |
| COS1 (layout+detail suppressed) | 0.8183 | −154.7 |
| COS2 (layout suppressed+detail enhanced) | 0.7311 | −546.6 |

High-frequency energy (HFE) changes are fully consistent with the design intent of $\omega$, validating the controllability of the method.

### User Study

| Evaluation Dimension | Omegance Score |
|----------------------|---------------|
| Granularity control accuracy | 93.94% |
| Output quality preference | 81.38% (67.62% better + 13.76% equivalent) |

A study with 101 participants demonstrates that Omegance accurately controls granularity without degrading the quality of the base model.

### Key Findings
- **Detail suppression improves image quality**: $\varpi=6.0$ achieves the lowest FID (157.47) and highest IS (13.82), indicating that moderate smoothing can reduce artifacts from lower-quality models.
- **Detail enhancement improves aesthetics**: $\varpi=-6.0$ achieves the highest PickScore (0.1601), suggesting that richer detail better aligns with human aesthetic preferences.
- **High SSIM values** confirm that Omegance does not alter the overall layout structure.
- **Complementary effect on FLUX**: FLUX-generated images tend to be over-smoothed; Omegance's detail enhancement mode recovers fine-grained textures and improves perceptual realism.
- The method generalizes successfully to video generation (Mochi, Hunyuan) while maintaining temporal consistency.

## Highlights & Insights
- **Extreme simplicity** is the primary highlight: a single parameter, no architectural changes, no training, and virtually zero additional computation. While the modification appears minimal, it is supported by a rigorous theoretical analysis via SNR. Methods that are surprisingly simple often deliver the greatest practical value.
- **Combined spatial and temporal control** is highly practical: the ability to independently regulate layout and texture complexity can be directly integrated into post-processing pipelines involving ControlNet or IP-Adapter.
- **Architecture agnosticism** makes it a truly plug-and-play tool: from U-Net to DiT to Flow Matching, integration requires only a few lines of code.
- **Frequency-domain analysis** provides a valuable perspective for understanding the denoising process in diffusion models.

## Limitations & Future Work
- Omegance cannot fundamentally improve the generative capacity of the base model; it can only adjust granularity within the model's existing capability.
- The optimal value of $\omega$ depends on the specific model and use case, requiring manual tuning.
- Flow matching models require an additional mean-preserving operation, making the adaptation less elegant than for DDIM.
- The paper does not discuss boundary behavior of the $\omega$ range — it remains unclear how generation degrades when $\omega$ is excessively large or small.

## Related Work & Insights
- **vs. FreeU**: FreeU improves quality by adjusting scaling factors for U-Net backbone features and skip connections; it is tightly coupled to U-Net and requires two parameters. Omegance uses a single parameter and is architecture-agnostic, making it more general and concise.
- **vs. SAG/PAG**: These methods replace the null-text prediction in CFG to globally enhance quality, but lack spatially fine-grained control. Omegance achieves true region-level control via the omega mask.
- **vs. Noise Scheduling**: Conventional schedulers (cosine/linear/rescaled) require retraining and do not support local control, whereas Omegance enables flexible adjustment at inference time.

## Rating
- Novelty: ⭐⭐⭐⭐ — Conceptually minimal yet previously unexplored in a systematic manner; theoretical analysis is rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers T2I/I2I/T2V across multiple models with quantitative evaluation, user studies, and frequency-domain analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, complete derivations, and high-quality figures.
- Value: ⭐⭐⭐⭐ — Highly practical plug-and-play tool, though not a methodological breakthrough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering](ouroboros_single-step_diffusion_models_for_cycle-consistent_forward_and_inverse_.md)
- [\[ICCV 2025\] Disrupting Model Merging: A Parameter-Level Defense Without Sacrificing Accuracy](disrupting_model_merging_a_parameter-level_defense_without_sacrificing_accuracy.md)
- [\[ICCV 2025\] FaceCraft4D: Animated 3D Facial Avatar Generation from a Single Image](facecraft4d_animated_3d_facial_avatar_generation_from_a_single_image.md)
- [\[ICCV 2025\] InfGen: A Resolution-Agnostic Paradigm for Scalable Image Synthesis](infgen_a_resolution-agnostic_paradigm_for_scalable_image_synthesis.md)
- [\[ICCV 2025\] MaskControl: Spatio-Temporal Control for Masked Motion Synthesis](maskcontrol_spatio-temporal_control_for_masked_motion_synthesis.md)

</div>

<!-- RELATED:END -->
