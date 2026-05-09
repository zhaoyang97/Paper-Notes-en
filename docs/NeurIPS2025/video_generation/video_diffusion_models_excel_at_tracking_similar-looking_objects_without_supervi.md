---
title: >-
  [Paper Note] Video Diffusion Models Excel at Tracking Similar-Looking Objects Without Supervision
description: >-
  [NeurIPS 2025][Video Generation][video diffusion models] This paper reveals that pretrained video diffusion models naturally learn motion representations suitable for tracking during high-noise denoising stages, and proposes the TED framework that fuses motion and appearance features, achieving up to 10 percentage points improvement over existing self-supervised methods on tracking similar-looking objects.
tags:
  - NeurIPS 2025
  - Video Generation
  - video diffusion models
  - self-supervised tracking
  - motion representation
  - similar-looking objects
  - label propagation
date: 2026-05-08
content_hash: 52d70abe228af40c
---

# Video Diffusion Models Excel at Tracking Similar-Looking Objects Without Supervision

**Conference**: NeurIPS 2025
**arXiv**: [2512.02339](https://arxiv.org/abs/2512.02339)
**Code**: None
**Area**: Video Generation
**Keywords**: video diffusion models, self-supervised tracking, motion representation, similar-looking objects, label propagation

## TL;DR

This paper reveals that pretrained video diffusion models naturally learn motion representations suitable for tracking during high-noise denoising stages, and proposes the TED framework that fuses motion and appearance features, achieving up to 10 percentage points improvement over existing self-supervised methods on tracking similar-looking objects.

## Background & Motivation

1. **State of the Field**: Video label propagation — transferring first-frame annotations to subsequent frames — is a core task in video understanding. Supervised methods (e.g., SAM2) perform well but require extensive annotations. Self-supervised methods learn frame representations for pixel-level matching.

2. **Limitations of Prior Work**: Existing self-supervised tracking methods over-rely on appearance features — tracking fails when multiple objects look similar (e.g., two deer, two balls of the same color). Even methods trained with temporal signals (e.g., cycle-consistency) still use 2D image encoders to process each frame independently at inference, discarding inter-frame motion information.

3. **Root Cause**: Appearance features serve as a shortcut for distinguishing objects of different categories, but become an obstacle when distinguishing similar-looking objects of the same category. Motion is the only reliable discriminative cue, yet existing methods fail to exploit it effectively.

4. **Paper Goals**: Achieve robust tracking of similar-looking objects without any tracking annotations.

5. **Starting Point**: Pretrained video diffusion models must implicitly model inter-frame dynamics to generate coherent videos — their internal features can be directly leveraged as motion-aware representations without any tracking-specific training.

6. **Core Idea**: Video diffusion models naturally encode motion information during high-noise denoising stages (since appearance is no longer visible), and these features can directly distinguish objects that look identical but move differently.

## Method

### Overall Architecture

The TED (Temporal Enhanced Diffusion) framework: (1) extract motion features $\mathbf{R}_m$ from a video diffusion model → (2) extract appearance features $\mathbf{R}_a$ from an image diffusion model → (3) fuse as $\mathbf{R}_f = \text{concat}(\lambda \cdot \mathbf{R}_m, (1-\lambda) \cdot \mathbf{R}_a)$ → (4) label propagation.

### Key Designs

**1. Motion-Aware Representation Extraction**

- **Function**: Obtain features encoding inter-frame motion from a pretrained video diffusion model.
- **Mechanism**: Given a video $X$, high-intensity noise is added to obtain $\mathbf{X}^\tau$ (larger $\tau$ denotes more noise), which is fed into the 3D UNet of a video diffusion model for a single forward pass; features from the third block are extracted as $\mathbf{R}_m$. Crucially, the entire video sequence is processed as a whole (exploiting temporal attention and 3D convolutions) rather than frame-by-frame. A sliding window (window size $L$ with overlapping frames for motion consistency) is applied to long videos.
- **Design Motivation**: At high noise levels, appearance information is nearly destroyed, forcing the model to denoise from global motion patterns; thus the features encode rich motion information. Experiments confirm that $\tau = 600$–$900$ yields the best motion feature tracking performance — at this range, appearance features become completely ineffective.

**2. Motion + Appearance Feature Fusion**

- **Function**: Combine the complementary information from motion and appearance.
- **Mechanism**: Inspired by Two-Stream ConvNets, features are L2-normalized and concatenated: $\mathbf{R}_f = \text{concat}(\lambda \cdot \frac{\mathbf{R}_m}{\|\mathbf{R}_m\|_2}, (1-\lambda) \cdot \frac{\mathbf{R}_a}{\|\mathbf{R}_a\|_2})$. Appearance features are computed frame-independently using an image diffusion model (ADM). $\lambda$ controls the weighting: $\lambda = 1.0$ is optimal for completely identical objects, while $\lambda \approx 0.5$ is optimal for real-world videos.
- **Design Motivation**: Pure motion features suffice for objects with identical appearances, but in real-world scenarios the combination of motion and appearance yields better results.

**3. Label Propagation Tracking**

- **Function**: Propagate first-frame labels to subsequent frames based on feature similarity.
- **Mechanism**: For each pixel $i$ in the target frame, feature dot-product similarity $A_{tr}(i,j) = \mathbf{R}_f^t(i) \cdot \mathbf{R}_f^r(j)$ is computed within a spatial neighborhood $\mathcal{S}(i)$ in the reference frame; top-K values are retained and labels are aggregated by weighted summation. Recursive propagation is adopted using the first frame and the preceding $m$ frames as reference frames.
- **Design Motivation**: This follows the standard video label propagation protocol, enabling fair comparison with existing methods.

### Loss & Training

- **No training**: TED entirely exploits features from pretrained models without any tracking-specific training.
- I2VGen-XL is used as the default video diffusion model; ADM serves as the image diffusion model.
- The noise timestep $\tau$ and feature layer $n_v$ are selected empirically.

## Key Experimental Results

### Main Results

| Method | Temporal Training | DAVIS $\mathcal{J\&F}_m$ | Youtube-Similar $\mathcal{J\&F}_m$ | Kubric-Similar $\mathcal{J}_m$ |
|--------|------------------|--------------------------|-------------------------------------|-------------------------------|
| MoCo | ✕ | 65.9 | 48.0 | 51.6 |
| SFC | ✕ | 71.2 | 55.5 | 43.1 |
| DIFT | ✕ | 75.7 | 60.7 | 52.7 |
| CRW | ✓ | 67.6 | 52.0 | 49.7 |
| Spa-then-Temp | ✓ | 74.1 | 59.6 | 44.0 |
| SMTC | ✓ | 73.0 | 57.5 | 64.7 |
| **TED (Ours)** | **✕** | **77.6** | **66.0** | **87.2** |

### Ablation Study

| Ablation | DAVIS | Youtube-Similar | Kubric-Similar |
|----------|-------|-----------------|---------------|
| Appearance only $\mathbf{R}_a$ ($\lambda$=0) | 75.7 | 60.7 | ~50 |
| Motion only $\mathbf{R}_m$ ($\lambda$=1) | ~73 | ~63 | **87.2** |
| **Fused $\mathbf{R}_f$ ($\lambda$=0.5)** | **77.6** | **66.0** | ~85 |
| No overlap frames ($l$=0) | - | ~62 | - |
| Overlap frames ($l$=2) | - | **~66** | - |

**Tracking performance under different noise levels $\tau$**:

| $\tau$ | $\mathbf{R}_a$ Performance | $\mathbf{R}_m$ Youtube-Similar | Interpretation |
|--------|---------------------------|-------------------------------|----------------|
| 200 | High | Lower | Appearance information rich; motion information masked |
| 400 | Moderate | Increasing | Model begins to emphasize motion denoising |
| **600** | Very low | **Peak** | Appearance nearly unavailable; **motion dominates** |
| 900 | ~0 | High (Kubric peak) | Near-pure noise; only motion learnable |

### Key Findings

- On Kubric-Similar (two completely identical balls), most methods achieve $\mathcal{J}_m \approx 50\%$ (equivalent to random guessing), while TED reaches **87.2%** — a qualitative leap.
- TED surpasses all self-supervised methods on the standard DAVIS benchmark by more than 6 points.
- **Core Finding**: Video diffusion models learn motion at high noise levels $\tau$ and appearance at low noise levels $\tau$ — revealing a hierarchical structure in the diffusion denoising process.
- A small number of overlapping frames ($l=2$) is sufficient to ensure motion consistency.
- The method is model-agnostic: substituting Stable Video Diffusion yields equally effective results.

## Highlights & Insights

- **Deep insight**: The paper reveals a hierarchical division of labor in the video diffusion denoising process — high noise encodes motion, low noise encodes appearance.
- Zero tracking annotations, zero additional training — purely reusing knowledge from pretrained generative models.
- PCA visualizations intuitively demonstrate the ability of motion features to discriminate similar-looking objects (Figure 4).
- The controlled experimental design is elegant: the two completely identical balls in Kubric-Similar perfectly isolate the contributions of motion versus appearance features.

## Limitations & Future Work

- Constrained by the window length of the pretrained video diffusion model (16 frames for I2VGen-XL); long videos require sliding window processing.
- Inference speed is relatively slow due to the need to run a full 3D UNet forward pass.
- $\lambda$ requires adjustment depending on the scene type (identical objects vs. real-world scenarios).
- No direct comparison with supervised methods (e.g., SAM2) is provided.

## Related Work & Insights

- DIFT extracts appearance features from image diffusion models for tracking; TED extends this to motion features from video diffusion models.
- Unlike Track4Gen, which requires tracking annotations to train video diffusion models, TED directly leverages pretrained models without additional training.
- Insight: The internal representations of generative models encode rich understanding of the physical world, awaiting further exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The discovery that high-noise stages of video diffusion models encode motion is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Benchmarks spanning three difficulty levels (standard / similar / identical), noise-level ablations, and model ablations are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, controlled experimental design is elegant, and visualizations are convincing.
- **Value**: ⭐⭐⭐⭐⭐ Offers both deep insight and practical utility, opening a new direction for representation reuse from generative models.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Target-Aware Video Diffusion Models](../../ICLR2026/video_generation/target-aware_video_diffusion_models.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](../../ICCV2025/video_generation/vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](../../ICCV2025/video_generation/leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)
- [\[NeurIPS 2025\] VMDT: Decoding the Trustworthiness of Video Foundation Models](vmdt_decoding_the_trustworthiness_of_video_foundation_models.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](../../ICCV2025/video_generation/efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)

<!-- RELATED:END -->
