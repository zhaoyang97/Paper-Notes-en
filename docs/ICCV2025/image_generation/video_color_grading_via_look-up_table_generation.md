---
title: >-
  [Paper Note] Video Color Grading via Look-Up Table Generation
description: >-
  [ICCV 2025][Image Generation][video color grading] This paper proposes a video color grading framework that explicitly generates Look-Up Tables (LUTs) via a diffusion model. A GS-Extractor captures high-level style features from a reference scene, and an L-Diffuser generates a color LUT that can be applied losslessly to all video frames in a single forward pass. Text prompts are further supported for fine-grained adjustments such as brightness and contrast.
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "video color grading"
  - "LUT generation"
  - "diffusion model"
  - "color grading"
  - "reference image"
  - "style transfer"
date: 2026-05-08
content_hash: 44d2f37d412a5d48
---

# Video Color Grading via Look-Up Table Generation

**Conference**: ICCV 2025
**arXiv**: [2508.00548](https://arxiv.org/abs/2508.00548)  
**Code**: [GitHub](https://github.com/seunghyuns98/VideoColorGrading)  
**Area**: Image Generation / Video Editing
**Keywords**: video color grading, LUT generation, diffusion model, color grading, reference image, style transfer

## TL;DR

This paper proposes a video color grading framework that explicitly generates Look-Up Tables (LUTs) via a diffusion model. A GS-Extractor captures high-level style features from a reference scene, and an L-Diffuser generates a color LUT that can be applied losslessly to all video frames in a single forward pass. Text prompts are further supported for fine-grained adjustments such as brightness and contrast.

## Background & Motivation

- **Core Problem**: Color grading is a central component of film post-production, aimed at adjusting color and contrast to convey specific artistic intent and emotional atmosphere (e.g., the warm/cool tone contrast in *Parasite* as a metaphor for class disparity). The process demands deep color theory expertise and artistic intuition, and has long been the exclusive domain of professional colorists.
- **Task Distinction**: Color grading differs from color transfer and style transfer:
    - Color transfer: matches color distributions (low-level features)
    - Style transfer: transfers texture + color (may destroy content structure)
    - **Color grading**: conveys **high-level subjective qualities** such as mood, atmosphere, and tone, while strictly preserving content detail
- **Limitations of Prior Work**:
    - Color transfer methods (statistical alignment, histogram matching) operate on low-level features and cannot convey "atmosphere"
    - Style transfer methods (WCT2, CAP-VSTNet, etc.) operate in latent space, inevitably losing high-frequency detail
    - Per-frame processing leads to temporal inconsistency (flickering)
    - Existing LUT prediction methods (NLUT, AdaCM) generalize poorly to unseen styles and produce color distortion
- **Key Insight**: The proposed framework mirrors the professional colorist's workflow—select keyframes, match style, generate a LUT, apply to all frames, and refine—using an explicit LUT to guarantee temporal consistency and zero content loss.

## Method

### Overall Architecture (Three-Stage Pipeline)

1. **Keyframe Selection**: A CLIP image encoder identifies the semantically most similar frame pair between the input and reference videos.
2. **LUT Generation**: The GS-Extractor extracts high-level style features from the reference; the L-Diffuser generates a LUT that is then applied to all frames.
3. **User Preference Refinement**: Text prompts enable low-level adjustments such as contrast, brightness, and hue.

### Keyframe Selection

Given input frame sequence $I_{1:M}$ and reference frame sequence $I'_{1:N}$, the frame pair maximizing CLIP cosine similarity is selected:

$$(\hat{m}, \hat{n}) = \arg\max_{m,n} \frac{f_m \cdot f'_n}{\|f_m\| \|f'_n\|}$$

Frames are sampled at 1 fps, as semantics remain stable over short intervals.

### GS-Extractor (Grading Style Extractor)

- Adopts the ReferenceNet architecture: two structurally identical U-Nets
    - **GS-Extractor U-Net**: extracts high-level subjective features (atmosphere, emotion, tone) from the reference frame
    - **Image Editor U-Net**: conditioned on the extracted features, edits the input frame to match the target style via spatial attention
- Training data construction: frame pairs are sampled from the same film; a LUT is applied to one frame to introduce a style difference, training the network to extract and restore this difference
- Denoising objective:

$$\ell = \mathbb{E}_{z_k, z_0^{I_{\hat{m}}}, G(z_0^{I'_{\hat{n}}}), \epsilon, k} \left[\|\epsilon - \epsilon_\theta(z_k^{I'_{\hat{m}}}, z_0^{I_{\hat{m}}}, G(z_0^{I'_{\hat{n}}}), k)\|_2^2\right]$$

### L-Diffuser (LUT Diffusion Generator)

**Core Idea**: Rather than performing style transfer in pixel or latent space, the model **directly generates a LUT**.

- The GS-Extractor is frozen; L-Diffuser is trained to generate a LUT from random noise.
- The model generates the residual $\Delta L \in \mathbb{R}^{16 \times 16 \times 16 \times 3}$ relative to the identity LUT, reshaped to $64 \times 64 \times 3$ to match the image latent dimensions.
- **Conditioning vector**: the difference between the GS-Extractor features of the reference and input frames, serving as a "style gradient" from input to reference:

$$C = G(z_0^{I'_{\hat{n}}}) - G(z_0^{I_{\hat{m}}})$$

- Diffusion denoising objective:

$$\ell = \mathbb{E}_{\Delta L'_k, L^I, C, \epsilon, k} \left[\|\epsilon - \epsilon_\theta(\Delta L'_k, L^I, C, k)\|_2^2\right]$$

- Final LUT = identity LUT + $\Delta L$ output by L-Diffuser

### Why LUT?

- **Temporal consistency**: a single LUT applied uniformly to all frames produces zero flickering
- **Zero content loss**: LUT is a deterministic color mapping that does not alter spatial structure
- **Fast inference**: LUT lookup is nearly instantaneous
- **Interpretability and editability**: the generated LUT can be further manually adjusted by professional colorists

### Loss & Training

- GS-Extractor training: diffusion denoising loss (Eq. 3)
- L-Diffuser training: diffusion denoising loss on the LUT residual (Eq. 5)
- Implicit constraint: high-level feature alignment in the GS-Extractor ensures that grading captures atmosphere rather than simple color distribution matching

## Key Experimental Results

### Main Results: Condensed Movie Dataset

| Method | PSNR | SSIM | LPIPS | Inference Time (s) |
|--------|:---:|:---:|:---:|:---:|
| WCT2 | 19.97 | 0.749 | 0.303 | 130.08 |
| PhotoNAS | 17.08 | 0.652 | 0.394 | 54.48 |
| Deep Preset | 20.70 | 0.743 | 0.322 | 25.64 |
| HistoGAN | 18.36 | 0.689 | 0.360 | 50.47 |
| CCPL | 17.13 | 0.620 | 0.395 | 38.55 |
| CAP-VSTNet | 20.55 | 0.764 | 0.303 | 46.97 |
| NLUT | 21.41 | 0.740 | 0.303 | 21.24 |
| **Ours** | **24.55** | **0.845** | **0.146** | **12.10** |

The proposed method achieves comprehensive superiority across all metrics: PSNR exceeds the second-best baseline by 3.14 dB, LPIPS is reduced by more than half, and inference speed is the fastest (12.1 s vs. 21.2 s for the next fastest baseline).

### Adobe5K Dataset

| Method | PSNR | SSIM | LPIPS |
|--------|:---:|:---:|:---:|
| CAP-VSTNet | 16.33 | 0.786 | 0.327 |
| NLUT | 16.76 | 0.730 | 0.392 |
| **Ours** | **18.78** | **0.797** | - |

Cross-dataset generalization is similarly strong.

### User Study

- A large-scale user study validates subjective aesthetic quality that quantitative metrics struggle to capture.
- The proposed method achieves the highest user preference scores across all reference style conditions.

### Key Findings

- Directly generating a LUT is more effective than performing color transformation in latent space, as the LUT is the native representation of color mapping.
- Using the feature difference from the GS-Extractor as the conditioning vector accurately captures the transformation direction "from current style to target style."
- BRISQUE (no-reference image quality metric) and blur metrics confirm that the proposed method best preserves the quality of the original video.

## Highlights & Insights

1. **Precise task definition**: The paper clearly distinguishes "color grading" from "color transfer/style transfer"—grading conveys high-level aesthetic intent rather than simply matching color distributions.
2. **LUT as the generation target**: Using a diffusion model to directly generate a LUT is an elegant design choice—reformulating the continuous pixel-level transformation problem as discrete color mapping learning, which naturally resolves both temporal consistency and content preservation.
3. **"Style gradient" conditioning design**: Using the feature difference between reference and input as the conditioning vector is conceptually clean and physically interpretable.
4. **Alignment with industry workflow**: The framework mirrors the professional colorist's workflow (keyframe selection, style matching, adjustment, refinement), offering strong practical utility.
5. **Text-based refinement interface**: Post-grading text prompt control over contrast, brightness, and other parameters extends the dimensionality of user control.

## Limitations & Future Work

- The LUT resolution of $16^3$ may produce banding artifacts under extreme color transformations.
- Reliance on CLIP-based keyframe selection may fail when the input and reference videos are semantically dissimilar.
- Training data construction depends on existing LUTs and film data; generalization to non-cinematic domains (e.g., drone footage, medical imaging) remains unvalidated.
- The GS-Extractor is built on the SD U-Net architecture, resulting in a non-trivial model size.
- A single LUT is applied uniformly to all frames, limiting adaptability for scenarios requiring frame-level fine-grained grading (e.g., temporally varying lighting conditions).

## Related Work & Insights

- **Reference-based color adjustment**: WCT2, PhotoNAS, HistoGAN, CAP-VSTNet, etc.
- **Video color consistency**: optical flow constraints, contrastive learning for temporal consistency (CCPL), joint training in UniST
- **Deterministic color mapping**: bilateral grids, MLP transformations, LUT prediction (NLUT, AdaCM)
- **LUT learning**: 3D LUT prediction, LUT compression, and adaptive weighting
- **ReferenceNet architecture**: dual U-Net design for reference image feature extraction

## Rating

| Dimension | Score (1–5) |
|-----------|:---:|
| Novelty | 4 |
| Theoretical Depth | 3.5 |
| Experimental Thoroughness | 4.5 |
| Writing Quality | 4 |
| Value | 4.5 |
| **Overall** | **4.1** |

Applying a diffusion model to explicitly generate LUTs is the most compelling design choice in this paper, elegantly leveraging deterministic mapping to address the two core challenges of video editing: temporal consistency and content preservation. The experiments are comprehensive, the user study is thorough, and the work holds high industrial application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SA-LUT: Spatial Adaptive 4D Look-Up Table for Photorealistic Style Transfer](sa-lut_spatial_adaptive_4d_look-up_table_for_photorealistic_style_transfer.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](balanced_image_stylization_with_style_matching_score.md)
- [\[ICCV 2025\] Video Motion Graphs](video_motion_graphs.md)
- [\[ICCV 2025\] SMGDiff: Soccer Motion Generation using Diffusion Probabilistic Models](smgdiff_soccer_motion_generation_using_diffusion_probabilistic_models.md)
- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](summdiff_generative_modeling_of_video_summarization_with_diffusion.md)

</div>

<!-- RELATED:END -->
