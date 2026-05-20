---
title: >-
  [Paper Note] Seen-to-Scene: Keep the Seen, Generate the Unseen for Video Outpainting
description: >-
  [CVPR 2026][Video Understanding][video outpainting] This paper proposes Seen-to-Scene, a unified video outpainting framework that integrates propagation-based and generation-based paradigms. By combining reference-frame-…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "video outpainting"
  - "propagation"
  - "diffusion model"
  - "optical flow"
  - "temporal coherence"
date: 2026-05-08
content_hash: 9ed28f06cade6fe0
---

# Seen-to-Scene: Keep the Seen, Generate the Unseen for Video Outpainting

**Conference**: CVPR 2026
**arXiv**: [2604.14648](https://arxiv.org/abs/2604.14648)  
**Code**: [github.com/InSeokJeon/Seen_to_Scene](https://github.com/InSeokJeon/Seen_to_Scene)  
**Area**: Video Understanding / Generation
**Keywords**: video outpainting, propagation, diffusion model, optical flow, temporal coherence

## TL;DR

This paper proposes Seen-to-Scene, a unified video outpainting framework that integrates propagation-based and generation-based paradigms. By combining reference-frame-guided latent-space propagation with a video diffusion model, it achieves spatiotemporal consistency and visual fidelity in zero-shot inference that surpasses prior methods requiring input-specific adaptation.

## Background & Motivation

Video outpainting requires extending content beyond frame boundaries while preserving spatial fidelity and temporal consistency. Existing approaches are polarized: propagation-based methods rely on optical flow to propagate known content but incur high computational cost and cannot synthesize invisible regions; generation-based methods leverage the powerful generative capacity of diffusion models but suffer from inter-frame inconsistency due to implicit temporal modeling and hallucinated content caused by limited spatial cues. Unifying the complementary strengths of both paradigms within a single framework remains an open challenge.

## Method

### Overall Architecture

Given an input video, reference frames are selected and optical flow is estimated; a flow completion network then completes the flow field in the outpainting region. Input frames are encoded into latent space, after which reference-frame-guided latent propagation is performed using the completed flow. A lightweight refinement module mitigates propagation artifacts, and the propagated latent codes are then fed as conditions into a 3D U-Net diffusion denoising process. A VAE decoder reconstructs the final frames.

### Key Designs

1. **Reference-Frame-Guided Latent Propagation**: Content-rich reference frames are selected within a sliding window based on inter-frame structural correlation (the structural component of SSIM). Propagation is performed in latent space rather than pixel space, avoiding the high computational overhead of dense per-frame propagation. Long-range direct propagation is achieved by accumulating flow fields across reference frames.

2. **Domain Adaptation for the Flow Completion Network**: This work presents the first analysis of the domain gap in flow completion for video outpainting—flow completion networks trained on video inpainting perform poorly over large outpainting regions. The pretrained flow completion network is jointly fine-tuned within an end-to-end pipeline to adapt it to the outpainting domain.

3. **Latent Code Refinement Module**: A lightweight module selectively adjusts latent codes in uncertain regions by predicting residual sampling offsets and adaptive modulation weights. Bidirectional alignment fusion reduces over-reliance on the flow field and alleviates local misalignment.

### Loss & Training

A pretrained video diffusion model (AnimateDiff) is fine-tuned with convolutional and spatial attention layers frozen to preserve spatial priors; only the temporal Transformer blocks are trained. Propagated latent codes and ground-truth latent codes are concatenated along the channel dimension as input to the denoising process. Training uses only 100K video samples from YouTube-VOS.

## Key Experimental Results

### Main Results

| Method | Adaptation | DAVIS PSNR↑ | DAVIS FVD↓ | YT-VOS LPIPS↓ |
|--------|-----------|------------|-----------|--------------|
| M3DDM | zero-shot | low | high | high |
| MOTIA | one-shot | mid | mid | mid |
| Follow-Canvas | zero-shot | mid | mid | mid |
| **Seen-to-Scene** | **zero-shot** | **21.95** | **218.8** | **best** |

Seen-to-Scene outperforms all prior methods on all metrics across DAVIS and YouTube-VOS, including one-shot methods that require input-specific adaptation.

### Ablation Study

- Domain adaptation of the flow completion network is critical for flow field quality in the outpainting region.
- The proposed reference frame selection strategy outperforms fixed-interval sampling.
- The latent code refinement module substantially reduces propagation artifacts.

### Key Findings

- Spatial cues provided by propagated latent codes significantly reduce content hallucination in the diffusion process.
- Strong zero-shot generalization is achievable with only 100K publicly available training samples.
- Latent-space propagation is orders of magnitude more efficient than pixel-space propagation.

## Highlights & Insights

- The unified propagation-plus-generation framework naturally combines the advantages of both paradigms.
- Zero-shot performance surpassing one-shot methods eliminates the deployment barrier of input-specific adaptation.
- The first systematic analysis of the flow completion domain gap fills a notable gap in the literature.

## Limitations & Future Work

- The approach depends on the accuracy of optical flow estimation and may degrade under extreme motion.
- Inference incurs additional overhead from computing the reference frame chain and flow fields.
- Quality challenges remain at very large expansion ratios (e.g., mask ratio of 0.66).

## Related Work & Insights

- The unified propagation-generation paradigm is applicable to related tasks such as video inpainting and video extrapolation.
- The reference frame selection strategy offers general value for video processing pipelines.
- The idea of domain-adaptive flow completion can be extended to other cross-domain transfer scenarios.

## Rating

7/10 — The unified framework design is elegant, the experiments are comprehensive, and surpassing one-shot methods in a zero-shot setting is a convincing contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion](savax_egotoexo_imitation_error_detection_via_scene.md)
- [\[NeurIPS 2025\] Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](../../NeurIPS2025/video_understanding/seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[CVPR 2026\] VidTAG: Temporally Aligned Video to GPS Geolocalization](vidtag_video_gps_geolocalization.md)
- [\[CVPR 2026\] AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding](adaspark_adaptive_sparsity_for_efficient_long_video_understanding.md)

</div>

<!-- RELATED:END -->
