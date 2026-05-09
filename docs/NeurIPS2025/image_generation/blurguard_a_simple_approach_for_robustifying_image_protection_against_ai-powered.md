---
title: >-
  [Paper Note] BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Edit
description: >-
  [NeurIPS 2025][Image Generation][Image protection] This paper proposes BlurGuard—a method that applies mild blurring to an image prior to adversarial perturbation generation, causing the perturbation to couple with low-frequency structures and thereby resist post-processing operations such as JPEG compression and Gaussian noise. This approach more effectively prevents AI editing tools such as Stable Diffusion from tampering with protected images, achieving over 20% improvement in protection success rate compared to the non-blurred baseline.
tags:
  - NeurIPS 2025
  - Image Generation
  - Image protection
  - adversarial perturbation
  - AI editing defense
  - blur preprocessing
  - diffusion models
date: 2026-05-08
content_hash: f0ea57a7e3e673e0
---

# BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Edit

**Conference**: NeurIPS 2025
**arXiv**: [2511.00143](https://arxiv.org/abs/2511.00143)
**Code**: Available
**Area**: Image Security / Adversarial Perturbation
**Keywords**: Image protection, adversarial perturbation, AI editing defense, blur preprocessing, diffusion models

## TL;DR
This paper proposes BlurGuard—a method that applies mild blurring to an image prior to adversarial perturbation generation, causing the perturbation to couple with low-frequency structures and thereby resist post-processing operations such as JPEG compression and Gaussian noise. This approach more effectively prevents AI editing tools such as Stable Diffusion from tampering with protected images, achieving over 20% improvement in protection success rate compared to the non-blurred baseline.

## Background & Motivation

**Background**: AI image editing tools (e.g., Stable Diffusion inpainting, Instruct-Pix2Pix) allow anyone to easily manipulate others' photos, raising concerns about portrait rights and misinformation. Adversarial perturbation is the primary defense mechanism—adding imperceptible noise to images to cause AI editing to fail.

**Limitations of Prior Work**:
- Adversarial perturbations are inherently high-frequency signals and are easily eliminated by simple post-processing operations (JPEG compression, Gaussian filtering, denoising).
- Social media platforms routinely compress uploaded images, rendering perturbations ineffective after upload.
- Existing robustness-enhancing methods (e.g., DiffPure, adversarial training) incur high computational costs or require modifications to the defense framework.

**Key Challenge**: Adversarial perturbations must remain "imperceptible" (small magnitude, high frequency), yet it is precisely these properties—small magnitude and high frequency—that make them susceptible to compression and filtering. Increasing perturbation magnitude improves robustness but degrades image quality.

**Goal**: To make adversarial perturbations more resistant to post-processing without increasing perturbation magnitude.

**Key Insight**: If an image is mildly blurred before perturbation generation, the perturbation is optimized to couple with the low-frequency structure of the blurred image, making it harder for high-frequency filtering to eliminate.

**Core Idea**: Apply blur first (shifting the image into the low-frequency domain) → generate perturbation (coupling it with low-frequency structure) → the perturbation becomes more resistant to post-processing.

## Method

### Overall Architecture
Original image $x$ → mild Gaussian blur / denoising $x_{blur} = \text{Blur}(x)$ → generate adversarial perturbation $\delta$ on $x_{blur}$ → output protected image $x_{blur} + \delta$ → upload to social media (subject to JPEG / compression) → AI editing fails.

### Key Designs

1. **Pre-blur Strategy**:

    - **Function**: Shifts the image into the low-frequency domain prior to perturbation generation.
    - **Mechanism**: Applies mild Gaussian blur ($\sigma \approx 0.5$–$1.0$) or bilateral filtering to the image. Blur intensity must be balanced—too weak is ineffective; too strong degrades image quality.
    - **Design Motivation**: High-frequency details (textures, edges) in the original image and adversarial perturbations are both high-frequency signals and are removed together during post-processing. Pre-blurring eliminates the image's own high-frequency components, forcing the perturbation generation algorithm to embed the perturbation into the remaining low-frequency structures, which are precisely what post-processing cannot easily remove.

2. **Perturbation Generation Compatibility**:

    - **Function**: Ensures BlurGuard is compatible with any existing perturbation method.
    - **Mechanism**: BlurGuard operates as a preprocessing step and does not modify the perturbation generation algorithm itself (PGD, C&W, AdvDM, etc. are all applicable).
    - **Design Motivation**: The plug-and-play design allows users to freely select the most suitable perturbation method.

3. **Visual Quality Control**:

    - **Function**: Ensures that blurring does not excessively degrade image quality.
    - **Mechanism**: Blur parameters ($\sigma$, kernel size) are determined via grid search to achieve the optimal balance between protection success rate and PSNR/SSIM.
    - **Design Motivation**: Protected images with poor visual quality will not be adopted by users in practice.

### Loss & Training
- Standard adversarial perturbation optimization: $\min_\delta \mathcal{L}_{edit}(x_{blur} + \delta) + \lambda \|\delta\|_\infty$
- No training required; operates entirely at inference time.
- Perturbation magnitude is constrained by $\|\delta\|_\infty \leq \epsilon$, typically $\epsilon = 8/255$.

## Key Experimental Results

### Main Results
Protection success rate on Stable Diffusion Inpainting (greater degradation in editing quality indicates more successful protection):

| Method | No Post-processing | JPEG q=75 | Gaussian Noise | Combined Attack |
|--------|--------------------|-----------|----------------|-----------------|
| PGD (no blur) | High | Large drop | Large drop | Near failure |
| **PGD + BlurGuard** | **High** | **Slight drop** | **Slight drop** | **Still effective** |
| AdvDM (no blur) | High | Drop | Drop | Drop |
| **AdvDM + BlurGuard** | **High** | **Maintained** | **Maintained** | **Maintained** |

### Ablation Study: Effect of Blur Intensity

| Blur $\sigma$ | Protection Success Rate | Image PSNR | Notes |
|---------------|------------------------|------------|-------|
| 0 (no blur) | Baseline | Highest | No resistance to post-processing |
| 0.5 | Moderate improvement | High | Mildly effective |
| 1.0 | Significant improvement | Medium-high | Optimal trade-off |
| 2.0 | Highest | Lower | Excessive image quality loss |

### Key Findings
- **Remarkably simple yet significantly effective**: Adding a single Gaussian blur step raises the post-JPEG survival rate of adversarial perturbations from approximately 30% to over 50%.
- **BlurGuard is compatible with all perturbation methods**: PGD, C&W, and AdvDM all achieve substantial robustness gains when combined with BlurGuard.
- **Blur intensity $\sigma \approx 1.0$ is the optimal operating point**.
- **Effective against adaptive attacks**: Even when an adversary is aware that BlurGuard is in use, the low-frequency coupling property makes perturbations harder to eliminate.

## Highlights & Insights
- The principle of **"removing high frequencies before generating perturbations"** is concise yet insightful—intuitively counterintuitive (does blurring not harm the image?), yet entirely sound from a frequency-domain perspective.
- **Zero-cost plug-and-play**: A single-line Gaussian blur call requiring no training whatsoever.
- Broadly applicable to any scenario requiring robust adversarial perturbations, not limited to image editing protection.

## Limitations & Future Work
- Blurring itself constitutes a loss in visual quality and may be unacceptable for high-resolution, detail-rich images.
- The optimal $\sigma$ may vary depending on image content and the type of post-processing; manual tuning is currently required.
- The effectiveness against nonlinear post-processing (e.g., AI-based denoisers) warrants further investigation.
- End-to-end evaluation on real social media platforms has not been conducted.

## Related Work & Insights
- **vs. PhotoGuard (Salman et al.)**: A standard adversarial perturbation method that does not address post-processing robustness. BlurGuard can be directly stacked on top of it.
- **vs. DiffPure**: Uses the diffusion model itself as a purification defense; computationally expensive. BlurGuard incurs zero cost.
- **vs. Adversarial Training**: Requires prior knowledge of the post-processing type and a training phase. BlurGuard requires no such prior information.

## Rating
- Novelty: ⭐⭐⭐ The idea is simple but effective; technical innovation is limited.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple perturbation methods, multiple post-processing operations, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear and intuitive.
- Value: ⭐⭐⭐⭐ The plug-and-play robustness improvement has practical significance for image privacy protection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Creating Blank Canvas Against AI-Enabled Image Forgery](../../AAAI2026/image_generation/creating_blank_canvas_against_ai-enabled_image_forgery.md)
- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [\[NeurIPS 2025\] Towards Resilient Safety-Driven Unlearning for Diffusion Models Against Downstream Fine-tuning](towards_resilient_safety-driven_unlearning_for_diffusion_models_against_downstre.md)
- [\[NeurIPS 2025\] Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models](perturb_a_model_not_an_image_towards_robust_privacy_protection_via_anti-personal.md)
- [\[ICCV 2025\] Anti-Tamper Protection for Unauthorized Individual Image Generation](../../ICCV2025/image_generation/anti-tamper_protection_for_unauthorized_individual_image_generation.md)

</div>

<!-- RELATED:END -->
