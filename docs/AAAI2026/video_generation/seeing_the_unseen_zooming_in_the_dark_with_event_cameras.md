---
title: >-
  [Paper Note] Seeing the Unseen: Zooming in the Dark with Event Cameras
description: >-
  [AAAI 2026][Video Generation][Low-light video super-resolution] This paper proposes RetinexEVSR, the first event-driven low-light video super-resolution (LVSR) framework. Through a Retinex-inspired bidirectional fusion strategy (RBF)—which first uses illumination maps to guide event feature denoising (IEE), then leverages enhanced event features to recover reflectance details (ERE)—the method achieves a 2.95 dB gain on the SDSD benchmark while reducing runtime by 65%.
tags:
  - "AAAI 2026"
  - "Video Generation"
  - "Low-light video super-resolution"
  - "event camera"
  - "Retinex decomposition"
  - "cross-modal fusion"
  - "bidirectional enhancement"
date: 2026-05-08
content_hash: b7f9b49d18676823
---

# Seeing the Unseen: Zooming in the Dark with Event Cameras

**Conference**: AAAI 2026
**arXiv**: [2601.02206](https://arxiv.org/abs/2601.02206)  
**Code**: [RetinexEVSR](https://github.com/DachunKai/RetinexEVSR)  
**Area**: Video Generation
**Keywords**: Low-light video super-resolution, event camera, Retinex decomposition, cross-modal fusion, bidirectional enhancement

## TL;DR

This paper proposes RetinexEVSR, the first event-driven low-light video super-resolution (LVSR) framework. Through a Retinex-inspired bidirectional fusion strategy (RBF)—which first uses illumination maps to guide event feature denoising (IEE), then leverages enhanced event features to recover reflectance details (ERE)—the method achieves a 2.95 dB gain on the SDSD benchmark while reducing runtime by 65%.

## Background & Motivation

**Background**: Video super-resolution (VSR) has achieved strong results under normal lighting conditions. However, VSR under low-light conditions (e.g., nighttime surveillance, night photography) faces severe challenges, including high noise, low contrast, and significant loss of texture detail in input frames.

**Limitations of Prior Work**:
(1) **Cascaded strategies fail**—enhance-then-super-resolve propagates and amplifies pixel errors; super-resolve-then-enhance amplifies noise and weakens textures;
(2) **Single-stage methods are insufficient**—directly learning a mapping from low-light LR to normal-light HR (e.g., DP3DF) still produces severe artifacts, structural distortion, and inaccurate illumination;
(3) **Event signal fusion is difficult**—under low light, both RGB frames and event data are severely degraded (noise, temporal trailing, spatially non-stationary distributions), and direct fusion introduces artifacts.

**Key Challenge**: Event cameras provide high dynamic range (120 dB) and high-frequency edge information, making them an ideal complementary source under low-light conditions. However, effectively extracting and fusing useful information when both modalities are degraded remains a core challenge.

**Key Insight**: By leveraging Retinex decomposition (illumination + reflectance), the method uses the smooth, low-noise illumination map to guide event denoising, and then uses the enhanced event features to recover reflectance details, forming a mutually beneficial bidirectional fusion strategy.

## Method

### Overall Architecture

RetinexEVSR takes a low-resolution image sequence $\{X_t^{LR}\}_{t=1}^T$ and corresponding event data $\{\mathcal{E}_t^{LR}\}_{t=1}^T$ as input, and outputs a super-resolved normal-light sequence $\{Y_t^{SR}\}_{t=1}^T$. Each frame first undergoes Retinex decomposition to obtain illumination and reflectance maps. Optical flow estimated from the reflectance is used for temporal alignment. The IEE and ERE modules then perform bidirectional enhancement, followed by joint upsampling reconstruction using the illumination map, enhanced reflectance, and refined event features.

### Key Designs

1. **Illumination-guided Event Enhancement Module (IEE)**

    - **Function**: Progressively removes low-light artifacts in event features using the illumination map obtained from Retinex decomposition.
    - **Mechanism**: The illumination map provides smooth, low-noise global brightness information. Multi-scale fusion is used to guide event feature refinement in a coarse-to-fine manner. At each time step, illumination features and event features interact across multiple spatial scales to progressively eliminate noise and trailing effects.
    - **Design Motivation**: When both degraded modalities are unreliable, the illumination component—by virtue of its inherent smoothness—contains the least noise and can serve as a reliable guidance signal. Purifying event features prior to fusion avoids noise propagation, which would occur with direct fusion of degraded RGB and event data.

2. **Event-guided Reflectance Enhancement Module (ERE)**

    - **Function**: Recovers missing high-frequency details in the reflectance map using event features enhanced by IEE.
    - **Mechanism**: A dynamic attention mechanism injects the high-frequency edge information from events into the reflectance feature stream. Multi-scale fusion enables fine-grained texture recovery.
    - **Design Motivation**: The reflectance component preserves the intrinsic scene content but lacks detail at low resolution. The motion-edge information unique to event cameras precisely compensates for this deficiency. Event features refined by IEE are of higher quality and can inject information more effectively.

3. **Retinex-inspired Bidirectional Fusion Strategy (RBF)**

    - **Function**: Unifies the fusion pipeline of IEE and ERE, forming a mutually beneficial chain: illumination → event → reflectance.
    - **Mechanism**: Unlike prior methods that directly fuse two degraded modalities, RBF introduces Retinex decomposition as an intermediate bridge: illumination guides event refinement, and the refined events subsequently guide reflectance recovery.
    - **Design Motivation**: This cascaded enhancement avoids direct conflict between degraded signals, ensuring that each fusion step is guided by a high-quality signal.

### Loss & Training

End-to-end training uses a combination of Charbonnier Loss and perceptual loss. The method is trained and evaluated on three datasets—SDSD, SDE, and RELED—covering both synthetic and real-world data.

## Key Experimental Results

### Main Results — SDSD Dataset, 4× LVSR

| Method | Type | PSNR↑(in) | PSNR↑(out) | SSIM↑(in) | LPIPS↓(in) |
|------|------|----------|-----------|----------|-----------|
| Retinexformer+IART | Enhance→SR | 27.09 | 19.84 | 0.8331 | 0.3938 |
| IART+Retinexformer | SR→Enhance | 25.30 | 24.07 | 0.8500 | 0.3541 |
| EvTexture | Event SR | 27.33 | 24.20 | 0.8776 | 0.3286 |
| FMA-Net | Joint method | 27.53 | 23.93 | 0.8680 | 0.3300 |
| **RetinexEVSR** | **Ours** | **30.28** | **25.15** | **0.8932** | **0.3149** |

### RELED Real-World Dataset Comparison

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Params(M) | FLOPs(G) | Runtime(ms) |
|------|-------|-------|--------|-----------|----------|-------------|
| EvTexture | 28.07 | 0.8604 | 0.4837 | 8.90 | 1141.1 | 126.9 |
| FMA-Net | 27.61 | 0.8611 | 0.4633 | 9.62 | 1941.3 | 596.3 |
| **RetinexEVSR** | **28.92** | **0.8707** | **0.4612** | **8.07** | **159.1** | **44.5** |

### Key Findings

- RetinexEVSR surpasses the second-best method by 2.95 dB on SDSD-in, with a runtime of only 44.5 ms (65% faster than EvTexture and 93% faster than FMA-Net).
- The model achieves the lowest parameter count (8.07 M) and FLOPs (159.1 G) among all compared methods, demonstrating significant efficiency advantages.
- The method recovers legible text and textures under extremely dark conditions (−6.7 EV) and severe motion blur.
- Cascaded strategies (Types I–IV) consistently underperform end-to-end methods (Types V–VI), validating the cascade error propagation issue.

## Highlights & Insights

1. **Elegant RBF strategy design**: The illumination component from Retinex decomposition serves as a bridge, resolving the challenge of directly fusing two degraded modalities.
2. **Pareto-optimal efficiency–performance trade-off**: The method achieves the smallest parameter count and FLOPs, best performance, and shortest runtime simultaneously.
3. **Systematic baseline comparison**: Six strategy types are covered (enhance→SR, SR→enhance, event enhance→SR, event SR→enhance, SR only, joint method), providing a comprehensive evaluation.
4. **Real-world validation**: The method also achieves state-of-the-art results on the RELED real-world low-light event dataset, demonstrating practical applicability.

## Limitations & Future Work

1. Reliance on event camera hardware limits the generalizability of the approach to broader application scenarios.
2. The accuracy of Retinex decomposition affects subsequent modules; decomposition quality may be unstable under extreme degradation.
3. Only 4× super-resolution is evaluated; performance at higher magnification factors (8×, 16×) remains unknown.
4. Temporal alignment relies on optical flow estimation, which may fail in large-displacement scenes.

## Related Work & Insights

- Using Retinex decomposition as a bridge for inter-modality fusion is a generalizable design paradigm, applicable to other degraded signal fusion tasks (e.g., rainy-weather super-resolution, underwater enhancement).
- The synergy between event cameras' high dynamic range properties and low-light vision offers further exploration opportunities (e.g., low-light semantic segmentation, low-light object detection).
- The "purify auxiliary modality before fusion" strategy holds broad reference value for the multi-sensor fusion community.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐: First event-driven LVSR solution; the RBF bidirectional fusion strategy is genuinely novel.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Three datasets, six comparison strategy types, and both synthetic and real-world data validation.
- **Writing Quality** ⭐⭐⭐⭐: Clear motivation; strategy comparison diagrams are intuitive and easy to understand.
- **Value** ⭐⭐⭐⭐: Provides a new efficient and high-performance paradigm for low-light video restoration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](../../CVPR2026/video_generation/seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)
- [\[NeurIPS 2025\] Seeing the Wind from a Falling Leaf](../../NeurIPS2025/video_generation/seeing_the_wind_from_a_falling_leaf.md)
- [\[CVPR 2026\] Video-as-Answer: Predict and Generate Next Video Event with Joint-GRPO](../../CVPR2026/video_generation/video-as-answer_predict_and_generate_next_video_event_with_joint-grpo.md)
- [\[CVPR 2026\] SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](../../CVPR2026/video_generation/switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [\[CVPR 2026\] Chain of Event-Centric Causal Thought for Physically Plausible Video Generation](../../CVPR2026/video_generation/chain_of_event-centric_causal_thought_for_physically_plausible_video_generation.md)

</div>

<!-- RELATED:END -->
