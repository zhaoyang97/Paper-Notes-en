---
title: >-
  [Paper Note] SelfHVD: Self-Supervised Handheld Video Deblurring
description: >-
  [CVPR 2026][Image Restoration][Video Deblurring] SelfHVD exploits naturally occurring sharp frames in handheld videos as supervisory signals. Through Self-Enhanced Video Deblurring (SEVD)…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "Video Deblurring"
  - "Self-Supervised Learning"
  - "Handheld Devices"
  - "Optical Image Stabilization"
  - "Self-Enhancement Training"
date: 2026-05-08
content_hash: 956f2105445f2bc7
---

# SelfHVD: Self-Supervised Handheld Video Deblurring

**Conference**: CVPR 2026
**arXiv**: [2508.08605](https://arxiv.org/abs/2508.08605)  
**Code**: [https://cshonglei.github.io/SelfHVD](https://cshonglei.github.io/SelfHVD)  
**Area**: Image Restoration
**Keywords**: Video Deblurring, Self-Supervised Learning, Handheld Devices, Optical Image Stabilization, Self-Enhancement Training

## TL;DR
SelfHVD exploits naturally occurring sharp frames in handheld videos as supervisory signals. Through Self-Enhanced Video Deblurring (SEVD), it constructs high-quality training pairs that surpass the quality ceiling of sharp frames, while Self-Constrained Spatial Consistency Maintenance (SCSCM) prevents spatial displacement drift, enabling handheld video deblurring without paired training data.

## Background & Motivation
1. **Background**: Learning-based video deblurring methods have achieved substantial progress in network design, yet their pretrained models are typically effective only on blur distributions similar to their training data.
2. **Limitations of Prior Work**: Blur in handheld videos is influenced not only by camera shake but also by OIS correction, resulting in a blur distribution that differs significantly from existing training datasets (e.g., GoPro, BSD), causing poor generalization of existing models.
3. **Key Challenge**: Collecting paired handheld video deblurring datasets is costly and complex, while using synthetic blur data introduces a domain gap.
4. **Goal**: Leverage naturally occurring sharp frames in handheld videos to learn a deblurring model in a self-supervised manner, eliminating the need for paired data.
5. **Key Insight**: When the camera motion trajectory is simple (e.g., linear) and slow, OIS functions properly and produces sharp frames. These sharp frames can provide deblurring cues and supervision for neighboring blurry frames.
6. **Core Idea**: Sharp frames → aligned supervision → SEVD self-enhancement beyond the sharp-frame ceiling → SCSCM to prevent spatial drift.

## Method

### Overall Architecture
Given a handheld blurry video, sharp frames are first selected via Laplacian variance combined with Otsu thresholding, then aligned and used as supervisory signals for the deblurring model. SEVD is subsequently applied to construct higher-quality training pairs to improve the model, while SCSCM prevents spatial displacement between input and output.

### Key Designs

1. **Sharp Frame Selection and Alignment**:
    - **Function**: Automatically identify sharp frames from handheld videos and align them with blurry frames.
    - **Mechanism**: Image sharpness is measured by the variance of the Laplacian $v_l(\mathbf{I})$, with thresholds determined automatically via Otsu's method. The video is segmented (20 frames per segment) to ensure uniform distribution of sharp frames. The SEA-RAFT optical flow model is used to align sharp frames with blurry frames, while an uncertainty mask $\mathbf{M}_{uncer}$ and an occlusion mask $\mathbf{M}_{occ}$ are designed to exclude misaligned and occluded regions.
    - **Design Motivation**: Sharp frames are a natural byproduct of handheld video capture; the selection accuracy reaches 96.77% on GoProShake and 91.88% on HVD.

2. **Self-Enhanced Video Deblurring (SEVD)**:
    - **Function**: Leverage the model's own deblurring capability to construct higher-quality training data.
    - **Mechanism**: (1) Random Sharp Cue Removal (RSCR): sharp frames in the input video are randomly replaced with neighboring blurry frames to produce a video $\tilde{\mathbf{B}}$ with fewer cues; (2) Supervision Information Selection (SIS): the better signal between the aligned sharp frame $\mathbf{S}_{j \to i}$ and the deblurring result $\mathcal{D}(\mathbf{B})_k$ from the original video is selected as supervision for $\tilde{\mathbf{B}}$. The sharp frame is used when it is not excessively distorted and is visually sharper; otherwise, the deblurring result is used (with stop gradient).
    - **Design Motivation**: When training with sharp frames as direct supervision, the quality ceiling is bounded by those sharp frames. SEVD enables the model to surpass the quality of the sharpest frames in the input and to handle object motion blur.

3. **Self-Constrained Spatial Consistency Maintenance (SCSCM)**:
    - **Function**: Prevent spatial displacement between input and output during training.
    - **Mechanism**: Based on an information-bottleneck-theoretic observation — the model maintains spatial consistency in early training but develops displacement in later stages. The output of a historical model (with parameters $\Theta_{\mathcal{D}_e}$ from iteration $e$) is used as auxiliary supervision: $\mathcal{L}_{scscm} = \|\tilde{\mathbf{R}}_i - sg(\mathbf{R}_k^e)\|_1$, constraining the current output to remain spatially consistent with the historical result.
    - **Design Motivation**: Optical flow alignment is inherently imperfect, and small alignment errors accumulate over training to cause spatial displacement. The natural spatial consistency of early-stage models is exploited as a regularization signal.

### Loss & Training
Total loss = reconstruction loss $\mathcal{L}_{rec}$ (masked L1) + SEVD loss $\mathcal{L}_{sevd}$ (conditionally selected L1) + SCSCM loss $\mathcal{L}_{scscm}$ (historical-model-constrained L1).

## Key Experimental Results

### Main Results

| Dataset | Metric | SelfHVD | Ren et al. | DaDeblur | Gain |
|--------|------|---------|-----------|---------|------|
| GoProShake | PSNR | Best | 2nd Best | — | Significant |
| HVD (Real) | Visual Quality | Best | — | 2nd Best | Clearly Sharper |

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| Full SelfHVD | Best | Complete model |
| Sharp frame supervision only | Baseline | Ceiling limited by sharp frame quality |
| + SEVD | Significant gain | Self-enhancement surpasses ceiling |
| + SCSCM | Further gain | Prevents spatial drift |
| Uncertainty + occlusion masks | Better than w/o masks | Excludes misaligned regions |

### Key Findings
- SEVD enables the model to surpass the quality of the sharpest frames in the input video and represents the most critical contribution.
- SCSCM is particularly important in later training stages; without it, the model gradually develops spatial displacement.
- The proposed method also partially restores object motion blur, as SEVD exploits sharp information across frames.

## Highlights & Insights
- The **closed-loop self-supervised design** is elegant: sharp frames → model → better supervision → better model.
- **Practical application of information bottleneck theory**: SCSCM is motivated by the observation that spatial consistency is preserved in early training, translating theoretical insight into practical design.
- The method is agnostic to the deblurring network architecture and can be adapted to multiple backbones.

## Limitations & Future Work
- The method relies on the presence of sufficiently many sharp frames in the video and is not applicable to videos that are severely blurred throughout.
- The accuracy of the optical flow model remains a bottleneck; alignment may be unreliable in scenes with complex motion.
- Future work could explore integration with diffusion-model-based deblurring methods.

## Related Work & Insights
- **vs. Ren et al.**: Randomly generated blur kernels are applied to sharp frames to synthesize training pairs, but a gap between synthetic and real blur persists. The proposed method directly uses real sharp frames combined with a self-enhancement strategy, yielding a closer match to the real distribution.
- **vs. DaDeblur**: A diffusion model is used to blur sharp images, yet the resulting blur is still not authentic.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Both SEVD self-enhancement training and SCSCM spatial consistency maintenance are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on synthetic and real datasets with complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Method motivation is clearly articulated with a coherent logical chain.
- Value: ⭐⭐⭐⭐ Addresses a practical pain point in handheld video deblurring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](tm-bsn_triangular-masked_blind-spot_network_for_real-world_self-supervised_image.md)
- [\[NeurIPS 2025\] MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes](../../NeurIPS2025/image_restoration/moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)
- [\[CVPR 2026\] BluRef: Unsupervised Image Deblurring with Dense-Matching References](bluref_unsupervised_image_deblurring_with_dense-matching_references.md)
- [\[ICCV 2025\] Blind2Sound: Self-Supervised Image Denoising without Residual Noise](../../ICCV2025/image_restoration/blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [\[CVPR 2026\] MAD-Avatar: Motion-Aware Animatable Gaussian Avatars Deblurring](motionaware_animatable_gaussian_avatars_deblurring.md)

</div>

<!-- RELATED:END -->
