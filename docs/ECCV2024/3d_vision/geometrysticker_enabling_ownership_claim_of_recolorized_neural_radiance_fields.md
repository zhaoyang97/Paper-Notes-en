---
title: >-
  [Paper Note] GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields
description: >-
  [ECCV 2024][3D Vision][NeRF ownership protection] Proposes GeometrySticker, which "sticks" binary copyright information onto the **geometric components** (instead of the color components) of NeRF. This allows original creators to extract watermarks from rendered images to claim ownership, even if the NeRF is recolorized.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "NeRF ownership protection"
  - "digital watermarking"
  - "geometric embedding"
  - "recolorization robustness"
  - "implicit representation security"
date: 2026-05-08
content_hash: c8d61928b360aa65
---

# GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields

**Conference**: ECCV 2024  
**arXiv**: [2407.13390](https://arxiv.org/abs/2407.13390)  
**Code**: [https://kevinhuangxf.github.io/GeometrySticker](https://kevinhuangxf.github.io/GeometrySticker) (Available, project page)  
**Area**: 3D Vision  
**Keywords**: NeRF ownership protection, digital watermarking, geometric embedding, recolorization robustness, implicit representation security

## TL;DR

Proposes GeometrySticker, which "sticks" binary copyright information onto the **geometric components** (instead of the color components) of NeRF. This allows original creators to extract watermarks from rendered images to claim ownership, even if the NeRF is recolorized.

## Background & Motivation

**Background**: NeRF recolorization techniques (CLIP-NeRF, PaletteNeRF, RecolorNeRF, etc.) are highly mature, making it easy to modify the color attributes of NeRF models. Meanwhile, NeRF is gaining popularity as sharing digital assets (e.g., NeRFStudio platform).

**Limitations of Prior Work**: Malicious users can recolor others' NeRF models and falsely claim ownership. Existing watermarking methods like CopyRNeRF embed information into color representations; once the colors are modified, the watermark fails (bit accuracy drops to ~50% random guessing level). StegaNeRF relies on complete geometry + color representations for steganography, which also fails after recolorization.

**Key Challenge**: Recolorization modifies color representations, whereas prior watermarking methods heavily rely on color representations to embed information. A "color-agnostic" watermarking scheme is needed.

**Goal**: How to embed a watermark in a NeRF model such that it can still be extracted after various recolorization operations?

**Key Insight**: NeRF models color and geometry separately using a color MLP ($\Theta_c$) and a geometry MLP ($\Theta_\sigma$). Since recolorization only modifies the color part, embedding the watermark into the geometry part naturally provides robustness against recolorization.

**Core Idea**: Attach (rather than modify) learnable watermark information to high-density 3D points (geometric components) on the object surface in NeRF, similar to a sticker.

## Method

### Overall Architecture

GeometrySticker consists of three core components: **(1)** Cover media selection based on learnable Laplace CDF; **(2)** MLP-based Message Sticker; **(3)** VGG16-based message extractor. Overall workflow: Select 3D points on the object surface $\rightarrow$ Convert the binary message to a geometrically compatible format using an MLP $\rightarrow$ Attach to the density values of selected points via addition $\rightarrow$ Extract the watermark from rendered images for verification.

### Key Designs

1. **Cover Media Selection**: Instead of embedding watermarks into all geometric components (which causes visible artifacts), only **high-density 3D points near the object surface** are selected. The Laplace CDF is used to compute the importance value of each sampled point:

$$\psi = \frac{1}{2} + \frac{1}{2} \cdot \text{sign}(\sigma - \mu) \cdot \left(1 - \exp\left(-\frac{|\sigma - \mu|}{\beta}\right)\right)$$

where $\mu$ is the mean of the geometry field, $\beta$ is a **learnable parameter** (instead of a fixed threshold), and $\psi \in [0,1]$ indicates the cumulative probability of the density value. Points with $\psi$ close to 1 are selected as the cover—these are high-density points on the object surface, constituting only a small fraction of the 3D points, which ensures the **Subtlety** of the embedding. A sparsity loss is used to constrain $\psi$ to tend towards a binary distribution of 0 or 1:

$$\mathcal{L}_{sparse} = \frac{1}{|N_p|} \sum_{\psi_i} [\log(\psi_i) + \log(1 - \psi_i)]$$

The key innovation lies in the learnability of $\beta$—adapting the threshold dynamically to avoid the "excessive coverage leading to distortion" or "insufficient coverage leading to unextractability" trade-offs of fixed thresholds.

2. **Message Sticker**: A lightweight MLP $\Theta_\mathbf{m}$ maps positional encodings and binary messages into a 1D message embedding:

$$m = \Theta_\mathbf{m}(\gamma_x(\mathbf{x}), \mathbf{M})$$

Then, the message is attached to the density value via simple **addition**:

$$\tilde{\sigma} = \sigma + \psi \cdot m$$

This design guarantees **Scalability**: it does not modify the original NeRF structure and is independent of specific architectures, making it applicable to various variants like vanilla NeRF, InstantNGP, and TensoRF. The watermarked density values are processed through standard volume rendering:

$$\tilde{C} = \sum_{i=1}^{N} \exp\left(-\sum_{j=1}^{i-1} \tilde{\sigma}_j \delta_j\right)(1 - \exp(-\tilde{\sigma}_i \delta_i)) \mathbf{c}_i$$

This operation is repeated across every view during training to ensure the watermark is accessible from any angle, achieving **Ubiquity**.

3. **Message Extraction & Verification**: A CNN message extractor $D_\chi$, using VGG16 as the backbone, extracts the watermark $\hat{\mathbf{M}} = D_\chi(\mathbf{I}_w)$ from the rendered image $\mathbf{I}_w$. Additionally, a CNN classifier $C_\phi$ is trained to predict whether an image contains a watermark.

### Loss & Training

The total loss consists of four parts:

$$\mathcal{L}_{total} = \mathcal{L}_{cont} + \mathcal{L}_{msg} + \mathcal{L}_{cls} + \mathcal{L}_{sparse}$$

- **Content Loss**: $\mathcal{L}_{cont} = \|\mathbf{I}_w - \mathbf{I}_o\|_2^2$, ensuring that rendering quality is not affected by watermarking.
- **Message Loss**: $\mathcal{L}_{msg} = \text{BCE}(D_\chi(\mathbf{I}_w), \mathbf{M})$, supervising the accuracy of watermark extraction.
- **Classification Loss**: $\mathcal{L}_{cls} = \text{BCE}(C_\phi(\mathbf{I}_w), C_\phi(\mathbf{I}_u))$, training the watermark detection classifier.
- **Sparsity Loss**: $\mathcal{L}_{sparse}$, constraining the importance values to a binary distribution.

During training, disturbances such as Gaussian noise, random rotation, random cropping, and Gaussian blur are applied to the rendered images to enhance robustness. The training patch size is $400 \times 400$, and training takes only 5,000 steps, completed in 45 minutes on a single V100 GPU. The embedded message length is 48 bits.

### Supported Recolorization Schemes

- **CLIP-based Recolorization**: Modifies colors using text prompts (e.g., "red") via CLIP feature matching loss.
- **Palette-based Recolorization**: Decomposes colors into linear combinations of palette bases and modifies the palette colors.
- **Image-level Color Jittering**: Directly modifies the hue of rendered images.

## Key Experimental Results

### Main Results

**Watermark extraction accuracy after recolorization (Blender/LLFF datasets, core table)**:

| Method | PSNR/SSIM↑ | LPIPS↓ | Color-jitter Accuracy | CLIP Accuracy | Palette Accuracy |
|------|------------|--------|-------------------|-----------|--------------|
| HiDDeN+NeRF | 30.80/0.9999 | 0.0167 | 50.13% | 51.08% | 50.91% |
| CopyRNeRF | 29.99/0.9999 | 0.0171 | 51.32% | 49.96% | N.A. |
| StegaNeRF | 31.48/0.9999 | 0.0149 | 54.18% | 52.48% | N.A. |
| **Ours** | **32.13/0.9999** | **0.0136** | **99.33%** | **99.50%** | **99.40%** |

**Standard scenes (no recolorization) + various image perturbations (Blender)**:

| Method | No Perturbation | Noise | Rotation | Cropping | Blur |
|------|--------|------|------|------|------|
| HiDDeN+NeRF | 50.19% | 49.84% | 50.12% | 50.09% | 50.16% |
| CopyRNeRF | 66.80% | 65.92% | 64.52% | 63.44% | 66.22% |
| StegaNeRF | 100% | 90.21% | 57.17% | 60.30% | 92.88% |
| **Ours** | **100%** | **99.25%** | **98.87%** | **98.75%** | **99.88%** |

### Ablation Study

**Ablation of cover media selection strategies**:

| Strategy | Effect | Description |
|------|------|------|
| All geometry points embedded | Obvious distortion | Modified low-density spatial points lead to visible artifacts |
| Fixed threshold Laplace CDF | Still has perceptible distortion | Fixed threshold lacks flexibility |
| **Learnable Laplace CDF** | **Almost imperceptible** | Adaptively finds the optimal threshold |

**Cross-architecture scalability**:

| NeRF Architecture | PSNR/SSIM↑ | Bit Accuracy |
|----------|------------|----------|
| NeRF | 27.44/0.8759 | 100% |
| InstantNGP | 28.59/0.8868 | 100% |
| TensoRF | 29.18/0.8907 | 100% |

### Key Findings

- Other methods show watermark accuracies dropping to ~50% (equivalent to random guessing) after recolorization, while GeometrySticker maintains >96%.
- GeometrySticker does not affect recolorization quality (recolorization results with vs. without watermark achieve PSNR >32dB).
- The learnable threshold $\beta$ is critical; fixed threshold schemes result in noticeable quality loss.
- Adversarial attack analysis: PGD attacks can degrade accuracy if the message extractor is leaked, but model purification attacks struggle to remove the watermark while preserving rendering quality.

## Highlights & Insights

- **Novel problem definition**: This is the first work to address copyright protection under NeRF recolorization scenarios, defining a practical and critical security threat model.
- **Elegant "sticker-style" design**: Without modifying the original NeRF structure, it appends information purely via addition, achieving true plug-and-play capability.
- **Three clear design principles**: Scalability (cross-architecture adaptation), Subtlety (inconspicuousness), and Ubiquity (all-view accessibility), each supported by corresponding technical solutions.
- **Learnable CDF**: Upgrading cover selection from a fixed threshold to learnable parameters is both clever and practical.

## Limitations & Future Work

- Adversarial attacks can still degrade accuracy if the message extractor is leaked; the extractor must remain private.
- Verified only on NeRF families, not yet extended to 3D Gaussian Splatting (noted as future work by the authors).
- Robustness against geometric editing (e.g., cage-based deformation, motion transfer) has not been explored.
- Watermark embedding requires an additional training step (though only 45 minutes), failing to achieve zero-shot embedding.
- Only verified 48-bit message length; the feasibility of larger capacities remains unexplored.

## Related Work & Insights

- **CopyRNeRF**: Embeds watermarks in color representations, making it vulnerable to recolorization; GeometrySticker's direct improvement target.
- **StegaNeRF**: A NeRF steganography method, which is also fragile against recolorization.
- **HiDDeN**: A classic deep-learning-based image watermarking method, which fails completely in NeRF scenarios.
- **PaletteNeRF/RecolorNeRF**: Representative recolorization methods, validating the compatibility of GeometrySticker.
- **Insight**: For implicit representation security, embedding information using "components unaffected by attacking operations" is a generalizable strategy. This can be extended to copyright protection for other 3D representations in the future.

## Rating

- Novelty: ⭐⭐⭐⭐ The problem definition is novel; the intuition of "geometric embedding for recolorization resistance" is clear and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering multi-architectures $\times$ multi-recolorization schemes $\times$ multi-perturbation types $\times$ adversarial attack/model purification analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear paper structure, with the three design principles consistently emphasized throughout.
- Value: ⭐⭐⭐⭐ Solves a practical security issue, though currently constrained by the real-world application scope of the NeRF ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] G2fR: Frequency Regularization in Grid-Based Feature Encoding Neural Radiance Fields](g2fr_frequency_regularization_in_grid-based_feature_encoding_neural_radiance_fie.md)
- [\[ECCV 2024\] BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream](benerf_neural_radiance_fields_from_a_single_blurry_image_and_event_stream.md)
- [\[ECCV 2024\] Omni-Recon: Harnessing Image-Based Rendering for General-Purpose Neural Radiance Fields](omni-recon_harnessing_image-based_rendering_for_general-purpose_neural_radiance_.md)
- [\[ECCV 2024\] LaRa: Efficient Large-Baseline Radiance Fields](lara_efficient_large-baseline_radiance_fields.md)
- [\[CVPR 2026\] Evidential Neural Radiance Fields](../../CVPR2026/3d_vision/evidential_neural_radiance_fields.md)

</div>

<!-- RELATED:END -->
