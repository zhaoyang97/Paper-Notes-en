---
title: >-
  [Paper Note] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video
description: >-
  [ICCV 2025][Image Generation][video disentanglement] This paper proposes BCD (Bitrate-Controlled Diffusion), a general self-supervised video disentanglement framework that separates per-frame motion features from global content features via a low-bitrate vector quantization information bottleneck, and reconstructs video using a conditional diffusion model. The approach demonstrates high-quality motion transfer and autoregressive video generation on talking-head and pixel-art cartoon datasets.
tags:
  - ICCV 2025
  - Image Generation
  - video disentanglement
  - motion-content separation
  - information bottleneck
  - vector quantization
  - diffusion models
date: 2026-05-08
content_hash: 59727055043105db
---

# Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video

**Conference**: ICCV 2025
**arXiv**: [2509.08376](https://arxiv.org/abs/2509.08376)
**Code**: None
**Area**: Image Generation
**Keywords**: video disentanglement, motion-content separation, information bottleneck, vector quantization, diffusion models

## TL;DR

This paper proposes BCD (Bitrate-Controlled Diffusion), a general self-supervised video disentanglement framework that separates per-frame motion features from global content features via a low-bitrate vector quantization information bottleneck, and reconstructs video using a conditional diffusion model. The approach demonstrates high-quality motion transfer and autoregressive video generation on talking-head and pixel-art cartoon datasets.

## Background & Motivation

Decomposing video into static content and dynamic motion is a core problem in video understanding. However, disentanglement learning is inherently ill-posed (a given video admits multiple valid decompositions), and the high dimensionality of video data further complicates the problem. Limitations of existing methods:
- **VAE-based methods** impose strong assumptions: low-dimensional motion features, explicit independence/dependence between motion and content, or the assumption that a single frame fully characterizes content.
- **Talking-head-specific methods** rely on domain-specific priors: optical flow (FOMM), keypoints (LivePortrait), 3D parametric face models (HyperReenact), or linear motion bases (LIA), limiting their applicability.
- These methods either over-constrain feature expressiveness or are tied to specific tasks, limiting generalization.

The core insight of BCD is that **a low-bitrate constraint itself serves as a general disentanglement prior**. Information bottleneck theory states that optimal compression retains the most relevant information while discarding irrelevant details — a goal perfectly aligned with disentangled representations (compact yet informative).

## Method

### Overall Architecture

Input video is first encoded into a per-frame latent sequence $z = \{z_t | t \in [1,T]\}$ by a pretrained image VAE. A Transformer encoder then extracts per-frame motion features $m$ and a global content feature $c$. The motion features are quantized through a low-bitrate VQ bottleneck, after which both motion and content features are used as conditions for a DiT diffusion decoder to reconstruct the original latent codes, which are finally decoded into video frames by the image VAE decoder.

### Key Designs

1. **Content and Motion Extraction (Transformer Encoder)**:

    - Uses a T5 Transformer (12 layers, hidden=512, 8-head attention, relative positional encoding).
    - $K$ learnable query tokens are prepended to the frame sequence; after passing through the Transformer, the prefix outputs form the content feature $c \in \mathbb{R}^{K \times C_c}$ and the remaining outputs form the motion feature $m \in \mathbb{R}^{T \times C_m}$.
    - The learnable queries are optimized over the full training set, enabling robust aggregation of content information across multiple frames (superior to single-frame references or simple pooling).
    - Supports flexible input frame counts, and handles videos with large variations (e.g., different viewpoints or details appearing only in specific frames) more robustly.

2. **Low-Bitrate Vector Quantization Information Bottleneck**:

    - Applies Group VQ by splitting motion features into $N=64$ groups, each independently quantized using a codebook of $K=32$ codewords.
    - Uses distance-based Gumbel-Softmax for differentiable quantization sampling:
    $\mu_t^i = \text{GumbelSoftmax}(-\alpha \cdot d_t^i)$
    - **Bitrate control**: Via Shannon's source coding theorem, the entropy $\mathcal{H}_{model}$ of the quantized motion features is estimated from the average sampling histogram, and an MSE constraint is imposed against a target bitrate $\mathcal{H}_{target}$.
    - The target bitrate is set to 4 kbps (160 bits per frame at 25 fps), slightly below the ~5 kbps average bitrate for talking-head videos reported by existing video codecs.
    - A low bitrate prevents content leakage into motion (otherwise motion bitrate would exceed the target); a non-zero bitrate prevents insufficient motion information (avoiding the information preference problem).

3. **Conditional Diffusion Decoder**:

    - Based on the DiT-B/4 architecture, with temporal attention layers inserted between spatial blocks to ensure temporal smoothness.
    - Motion conditioning is injected by adding to the diffusion timestep embedding.
    - Content conditioning is injected by concatenation with the noisy input.
    - Training and sampling follow the EDM framework.

### Loss & Training

Rate-distortion optimization objective:
$$\mathcal{L} = \mathcal{L}_d + \lambda \mathcal{L}_{VQ} = \text{MSE}(z, \tilde{z}) + 0.04 \cdot \text{MSE}(\mathcal{H}, \hat{\mathcal{H}})$$

**Cross-driving strategy**: During training, each video clip is evenly split into two temporal segments (semantically similar but with different motion). The content feature of the first segment and the motion feature of the second segment are used to reconstruct the second segment, preventing trivial entangled representations.

Training details: batch=32, 50 frames per clip; trained for 30 epochs (without temporal layers), followed by 15 epochs (with temporal layers). Training takes 4–5 days on 8×A100 GPUs.

## Key Experimental Results

### Main Results (LRS3 Talking-Head Motion Transfer)

| Method | FID↓ | CSIM↑ | Identity Err.↓ | Motion Err.↓ | Cross Err.↓ |
|--------|------|-------|----------------|--------------|-------------|
| FOMM | 98.5 | 0.76 | 0.75 | 24.3×10⁻² | 24.1×10⁻² |
| MCNET | 98.6 | 0.76 | 0.85 | 23.9×10⁻² | 23.6×10⁻² |
| HyperReenact | 106.8 | 0.58 | 0.57 | 3.94×10⁻² | 4.68×10⁻² |
| LIA | 104.4 | 0.71 | 0.57 | 36.1×10⁻² | 34.1×10⁻² |
| LivePortrait | 100.3 | 0.69 | 0.66 | 24.6×10⁻² | 23.7×10⁻² |
| **BCD (Ours)** | **86.0** | 0.69 | **0.41** | **3.13×10⁻²** | **3.67×10⁻²** |

### Ablation Study (Effect of Target Bitrate)

| Target Bitrate (kbps) | FID↓ | CSIM↑ | Identity Err.↓ | Motion Err.↓ | Cross Err.↓ |
|-----------------------|------|-------|----------------|--------------|-------------|
| 2.0 | 88.5 | 0.71 | 0.34 | 5.26×10⁻² | 5.68×10⁻² |
| 6.0 | 87.6 | 0.68 | 0.56 | 3.23×10⁻² | 4.13×10⁻² |
| 8.0 | 89.3 | 0.66 | 0.49 | 3.04×10⁻² | 3.74×10⁻² |
| **4.0** | **86.0** | 0.69 | 0.41 | 3.13×10⁻² | **3.67×10⁻²** |
| 4.0 (single ref. frame) | 87.9 | 0.69 | 0.47 | 3.13×10⁻² | 3.81×10⁻² |
| 4.0 (w/o cross-driving) | 120.1 | 0.64 | 0.58 | 41.5×10⁻² | 40.4×10⁻² |

### User Study

| Method | Identity Preservation↑ | Motion Consistency↑ | Visual Quality↑ |
|--------|------------------------|---------------------|-----------------|
| FOMM | 3.34 | 2.63 | 2.84 |
| HyperReenact | 2.97 | 4.01 | 3.72 |
| LIA | 3.66 | 2.53 | 3.53 |
| **BCD (Ours)** | **4.10** | **4.30** | **4.00** |

### Key Findings

- BCD uses no face-specific priors yet achieves the best identity error, motion error, and FID across all methods.
- 4 kbps is the optimal bitrate — too low (2 kbps) yields insufficient motion information and increases error, while too high (8 kbps) provides an insufficiently tight bottleneck, leading to poor disentanglement.
- Removing the cross-driving strategy causes FID to surge from 86 to 120 and motion error to increase by 13×, demonstrating its critical role in preventing disentanglement collapse.
- Keypoint-based methods (FOMM, LivePortrait) preserve identity well but exhibit large motion errors due to optical flow artifacts and first-frame pose assumptions.
- LIA's constrained motion space (linear basis combinations) results in the largest motion error.
- The method generalizes directly to previously unseen data types (Sprites cartoon dataset).

## Highlights & Insights

- Introducing rate-distortion theory into disentanglement learning is highly natural — optimal compression corresponds to optimal factor separation.
- The combination of implicit features (no inductive bias) and an information bottleneck balances flexibility with constraint.
- The learned motion space exhibits structure, directly supporting GPT-2-based autoregressive motion generation, demonstrating that the motion codebook captures the complete motion distribution.
- The Group VQ + Gumbel-Softmax differentiable quantization scheme enables fully end-to-end training.

## Limitations & Future Work

- Large amounts of training data are required to achieve prior-free disentanglement.
- The optimal bitrate may vary across datasets and requires manual tuning.
- Mild video flickering persists even after temporal fine-tuning.
- Since training data typically contains more dynamic than static variation, static modeling capability may degrade on out-of-distribution video inputs.

## Related Work & Insights

- Low-bitrate VQ is transferred from audio codec research (separating speech content from speaker identity) to video disentanglement, providing cross-domain inspiration.
- Using a diffusion model as the decoder (rather than a conventional VAE decoder) maintains higher reconstruction fidelity under information bottleneck constraints.
- Future work may extend the framework to general video content beyond talking heads and cartoons.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The idea of bitrate control as a general disentanglement prior is highly original and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation with quantitative metrics, user study, two datasets, bitrate ablation, and generative validation.
- Writing Quality: ⭐⭐⭐⭐ Motivation and theoretical foundations are clearly articulated, with well-explained connections to coding theory.
- Value: ⭐⭐⭐⭐ Provides a general disentanglement paradigm with broad implications for video editing and generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Video Motion Graphs](video_motion_graphs.md)
- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](summdiff_generative_modeling_of_video_summarization_with_diffusion.md)
- [\[ICCV 2025\] Diffusion-based 3D Hand Motion Recovery with Intuitive Physics](diffusion-based_3d_hand_motion_recovery_with_intuitive_physics.md)
- [\[ICCV 2025\] AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction](aid_adapting_image2video_diffusion_models_for_instruction-guided_video_predictio.md)

</div>

<!-- RELATED:END -->
