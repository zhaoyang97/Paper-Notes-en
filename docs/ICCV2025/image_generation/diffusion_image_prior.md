---
title: >-
  [Paper Note] DIIP: Diffusion Image Prior
description: >-
  [ICCV 2025][Image Generation][diffusion model] This paper discovers that pretrained diffusion models exhibit an implicit bias analogous to Deep Image Prior (DIP) when reconstructing degraded images—the iterative optimiza…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "diffusion model"
  - "Blind Image Restoration"
  - "Deep Image Prior"
  - "Early Stopping"
  - "Zero-shot"
date: 2026-05-08
content_hash: fa9939c92f95f490
---

# DIIP: Diffusion Image Prior

**Conference**: ICCV 2025
**arXiv**: [2503.21410](https://arxiv.org/abs/2503.21410)
**Code**: None
**Area**: Image Generation / Image Restoration
**Keywords**: diffusion model, Blind Image Restoration, Deep Image Prior, Early Stopping, Zero-shot

## TL;DR

This paper discovers that pretrained diffusion models exhibit an implicit bias analogous to Deep Image Prior (DIP) when reconstructing degraded images—the iterative optimization first produces a clean image before overfitting to the degraded input—and that this bias generalizes to a broader range of degradation types than DIP. Based on this finding, the authors propose DIIP, a fully blind (degradation-model-free) image restoration method.

## Background & Motivation

Image restoration (IR) aims to recover a clean image $x$ from a degraded observation $y$. Existing methods can be categorized by the strength of their degradation model assumptions:

**Non-blind methods** (DDRM, DDNM, DPS): require a fully known degradation model (e.g., known blur kernel)

**Partially blind methods** (BlindDPS, BIRD): require knowledge of the parametric form of the degradation

**Fully blind methods** (DIP, DreamClean): require no knowledge of the degradation model

Deep Image Prior (DIP) is a classical fully blind approach that exploits the implicit prior of CNNs by optimizing network parameters to fit a degraded image, producing clean outputs at intermediate iterations. However, its core limitation is that **this "clean-before-overfitting" behavior holds only for high-frequency degradations (e.g., noise) and fails for low-frequency degradations (e.g., blur)**.

The authors pose the key research question: do pretrained diffusion models exhibit a similar bias, and if so, does it generalize beyond the scope of DIP?

## Method

### Overall Architecture

DIIP employs a frozen pretrained diffusion model $g$ (via DDIM deterministic mapping) and **optimizes the input noise** $z$ rather than the model parameters:

$$z^* = \arg\min_z \|g(z) - y\|^2, \quad x^* = g(z^*)$$

Gradient descent is used to iteratively optimize $z$, and early stopping is applied at the appropriate iteration to obtain the clean restored image.

### Key Designs

1. **Discovery and Verification of the Diffusion Model's Implicit Prior**:

    - Gaussian noise and Gaussian blur are respectively applied to FFHQ images, and the optimization above is run to convergence (1500 iterations).
    - Two key findings emerge:
        - **(a) Two-phase behavior**: Regardless of the degradation type (noise or blur), the optimization exhibits two phases—(I) an intermediate phase in which a clean, realistic image is produced; and (II) a late phase in which the model begins to overfit to the degraded input. In contrast, DIP fails to produce sharp images at intermediate iterations under blur degradation.
        - **(b) High-frequency inertia**: Similar to DIP, diffusion models exhibit strong resistance to high-frequency artifacts and begin overfitting to noise only at very late iterations.
    - Design Motivation: This finding extends the applicable scope of DIP-like behavior from "high-frequency degradations only" to "a wide range of degradations including low-frequency ones."

2. **Stopping Criterion for Low-Frequency Degradations (Laplacian Variance)**:

    - The Laplacian variance (LV) of the reconstructed image is monitored as a sharpness indicator.
    - During phase (I), the generated image is sharp and LV is high.
    - When $k > k_{min}$ and $\sigma^2[k+1] < \sigma^2[k]$, the image begins to blur, triggering early stopping.
    - The reconstruction corresponding to the last LV peak is returned.
    - Design Motivation: Absolute sharpness is difficult to measure, but the relative trend of sharpness change is reliable.

3. **Stopping Criterion for High-Frequency Degradations (Normalized Loss Slope)**:

    - The normalized slope $\Delta_k = \frac{E(z^k;y) - E(z^{k-1};y)}{E(z^{k-1};y)}$ is computed.
    - Experiments show that the minimum of $\Delta_k$ corresponds precisely to the maximum PSNR.
    - Optimization stops when $\Delta_k < \epsilon$ (default 0.001).
    - Design Motivation: The inflection point in the loss curve marks the transition from "learning the clean signal" to "fitting the noise."

### Loss & Training

- Optimization uses the Adam optimizer with a learning rate of 0.0015.
- Hyperparameters: $k_{min} = 100$, $\epsilon = 0.001$.
- The fast diffusion inversion method proposed in BIRD is adopted to accelerate the computation of $g(z)$.
- The pretrained model is an unconditional diffusion model with a UNet backbone.
- No training dataset is required; the method is purely test-time optimization.

## Key Experimental Results

### Main Results

**CelebA Structured Degradations (Denoising, Super-Resolution)**:

| Method | Denoising PSNR↑ | Denoising SSIM↑ | SR×4 PSNR↑ | SR×8 PSNR↑ |
|---|---|---|---|---|
| DIP | 25.81 | 0.606 | 21.33 | 20.34 |
| BIRD | 27.92 | 0.821 | 25.26 | 22.63 |
| DreamClean | 27.05 | 0.771 | 23.44 | 21.33 |
| **DIIP** | **28.37** | **0.842** | 25.14 | **22.86** |

**Unstructured Degradations (Fully Blind Setting)**:

| Method | JPEG PSNR↑ | Raindrop Removal PSNR↑ | Non-uniform Deformation PSNR↑ |
|---|---|---|---|
| DIP | 20.43 | 20.37 | 18.83 |
| DreamClean | 23.92 | 22.94 | 22.16 |
| **DIIP** | **25.29** | **23.78** | **23.45** |

Note: Partially blind methods such as BIRD and BlindDPS **cannot be applied** to unstructured degradation scenarios.

### Ablation Study

**Effect of $k_{min}$ (PSNR in dB)**:

| $k_{min}$ | Non-uniform Deformation | Raindrop Removal |
|---|---|---|
| 50 | 22.18 | 22.48 |
| **100** | **23.45** | **23.78** |
| 150 | 23.52 | 23.82 |

**Effect of $\epsilon$ (PSNR in dB)**:

| $\epsilon$ | Denoising | JPEG Artifact Removal |
|---|---|---|
| 0.005 | 27.25 | 22.38 |
| **0.001** | **28.37** | **25.29** |
| 0.0005 | 28.14 | 25.02 |

**Gap from Oracle Stopping**: DIIP trails the oracle by approximately 0.3 dB (denoising: 28.37 vs. 28.63), demonstrating that the self-supervised stopping criteria are near-optimal.

### Key Findings

- DIIP achieves state-of-the-art performance on all fully blind restoration tasks, and even surpasses some partially blind methods that require explicit degradation models on certain tasks.
- Runtime is 138 seconds per image with 1.2 GB memory, comparable to DreamClean.
- The pretrained diffusion model has never been exposed to degraded data, yet consistently reconstructs clean images first—a manifestation of a purely inductive bias.
- The two stopping criteria are complementary and apply to different degradation types; the algorithm automatically selects whichever criterion triggers first, without requiring prior knowledge of the degradation type.

## Highlights & Insights

- **The core finding is highly valuable**: the implicit prior of diffusion models is stronger and more broadly applicable than that of DIP, opening a new direction for fully blind image restoration.
- The method is remarkably simple and elegant: frozen model + noise optimization + early stopping, with no additional training or degradation modeling required.
- The two self-supervised stopping criteria are complementary and require no knowledge of the degradation type.
- In experiments, DreamClean tends to alter image identity (e.g., facial features), whereas DIIP better preserves the original content.

## Limitations & Future Work

- Computational overhead is substantial (~138 seconds per image), making the method unsuitable for real-time applications.
- Each image requires independent optimization, precluding batch processing.
- The stopping criteria exhibit some sensitivity to hyperparameters ($k_{min}$, $\epsilon$).
- Validation is limited to 256×256 resolution; high-resolution scenarios would require more efficient diffusion inversion schemes.
- The interaction between the two stopping criteria under composite degradations involving both high-frequency and low-frequency components has not been thoroughly analyzed.

## Related Work & Insights

- DIP is the direct source of inspiration; DIIP can be viewed as an upgrade of DIP instantiated on diffusion models.
- DreamClean is the closest competitor (also fully blind) but adopts a different diffusion inference strategy.
- The fast diffusion inversion technique from BIRD is borrowed to accelerate optimization.
- The core finding of this paper may transfer to other generative models—whether Flow Matching, Consistency Models, and related frameworks possess analogous implicit priors warrants future investigation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The discovery of the diffusion model's implicit prior is highly insightful, and the method is concise and original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — A diverse range of degradation types is covered with thorough ablations; however, the evaluation datasets are relatively small in scale.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated, and the logical chain from discovery to method is highly natural.
- **Value**: ⭐⭐⭐⭐ — Offers a new perspective for fully blind image restoration, though computational cost limits practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model](learning_deblurring_texture_prior_from_unpaired_data_with_diffusion_model.md)
- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [\[NeurIPS 2025\] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](../../NeurIPS2025/image_generation/diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](balanced_image_stylization_with_style_matching_score.md)
- [\[ICCV 2025\] Addressing Text Embedding Leakage in Diffusion-based Image Editing](addressing_text_embedding_leakage_in_diffusion_based_image_editing.md)

</div>

<!-- RELATED:END -->
