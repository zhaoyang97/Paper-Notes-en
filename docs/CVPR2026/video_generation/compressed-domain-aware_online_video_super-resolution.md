---
title: >-
  [Paper Note] Compressed-Domain-Aware Online Video Super-Resolution
description: >-
  [CVPR 2026][Video Generation][Online video super-resolution] CDA-VSR leverages compressed-domain information (motion vectors, residual maps, and frame types) to guide three key stages of online video super-resolution: motion-vector-guided deformable alignment for efficient and accurate registration, residual-map-gated fusion to suppress misalignment artifacts, and frame-type-aware reconstruction to adaptively allocate computation. The method achieves state-of-the-art PSNR on REDS4 at 93 FPS—more than twice the speed of prior SOTA.
tags:
  - CVPR 2026
  - Video Generation
  - Online video super-resolution
  - compressed-domain information
  - motion vectors
  - deformable alignment
  - frame-type-aware processing
date: 2026-05-08
content_hash: 7eaa8d9f726218b9
---

# Compressed-Domain-Aware Online Video Super-Resolution

**Conference**: CVPR 2026
**arXiv**: [2603.07694](https://arxiv.org/abs/2603.07694)
**Code**: [https://github.com/sspBIT/CDA-VSR](https://github.com/sspBIT/CDA-VSR)
**Area**: Video Generation
**Keywords**: Online video super-resolution, compressed-domain information, motion vectors, deformable alignment, frame-type-aware processing

## TL;DR
CDA-VSR leverages compressed-domain information (motion vectors, residual maps, and frame types) to guide three key stages of online video super-resolution: motion-vector-guided deformable alignment for efficient and accurate registration, residual-map-gated fusion to suppress misalignment artifacts, and frame-type-aware reconstruction to adaptively allocate computation. The method achieves state-of-the-art PSNR on REDS4 at 93 FPS—more than twice the speed of prior SOTA.

## Background & Motivation

1. **Background**: Online video super-resolution (Online VSR) requires real-time reconstruction of the current frame during playback, using only past and current frames. Recent methods (e.g., TMP, DAP, MMVSR) have improved performance through better alignment and fusion modules, yet still struggle to meet real-time requirements at higher resolutions such as 2K.

2. **Limitations of Prior Work**: (1) **Computationally intensive motion estimation**: Optical-flow-based alignment methods (e.g., BasicVSR) are accurate but computationally expensive; implicit alignment methods (e.g., RRN) are efficient but degrade under large motions. (2) **Uniform treatment of redundant frames**: Existing methods apply the same computational budget to all frames, resulting in unnecessary redundant computation for the frequently occurring P-frames. (3) **Wasted information**: Compressed-domain information obtained during decoding (motion vectors, residual maps, frame types) is discarded rather than exploited.

3. **Key Challenge**: In bandwidth-constrained online video streaming, video is downsampled and transmitted in compressed form. Rich compressed-domain priors are available at the decoder at virtually no cost, yet existing methods rely solely on decoded low-resolution frames and ignore these valuable auxiliary signals.

4. **Goal**: To design dedicated modules tailored to the distinct characteristics of three types of compressed-domain information—motion vectors, residual maps, and frame types—so as to simultaneously improve super-resolution quality and substantially accelerate inference.

5. **Key Insight**: Within a video bitstream, motion vectors describe block-level inter-frame motion (serving as a coarse substitute for optical flow), residual maps reflect regions where motion compensation fails (naturally marking unreliable areas), and frame types determine inter-frame reference relationships (I-frames require high-quality reconstruction; P-frames can be processed more lightly). Each type of information offers distinct utility.

6. **Core Idea**: Treat the three categories of compressed-domain information—motion vectors for coarse alignment, residual maps for quality gating, and frame types for computation allocation—as natural priors for online VSR, allowing "free" information to yield simultaneous gains in quality and speed.

## Method

### Overall Architecture
CDA-VSR adopts a recurrent structure that takes decoded low-resolution frames together with compressed-domain information (MVs, residual maps, and frame types) as input and produces high-resolution frames. The pipeline proceeds as follows: (1) a shallow feature extraction network maps each frame to latent features; (2) the **MVGDA module** uses motion vectors to guide deformable convolutions for inter-frame alignment; (3) the **RMGF module** generates spatial weights from the residual map for selective feature fusion; (4) the **FTAR module** selects reconstruction branches of different depths based on frame type. The entire pipeline enforces the causal constraint (using only past and current frames) while meeting real-time processing requirements.

### Key Designs

1. **Motion-Vector-Guided Deformable Alignment (MVGDA)**:

    - **Function**: Efficiently and accurately aligns preceding-frame features with the current frame.
    - **Mechanism**: A two-step approach is employed. First, the motion vector is used to perform coarse warping of the previous frame features: $\bar{h}_{t-1} = \mathcal{W}(h_{t-1}; MV_{t-1 \to t})$, efficiently compensating for large-scale inter-frame motion. The MV is then used to initialize the offset $o_{MV}$ of a deformable convolutional network (DCN), while a lightweight convolutional network predicts a residual offset $\Delta o$ and a modulation mask $m$. The final aligned feature is: $\hat{h}_{t-1} = \mathcal{D}(h_{t-1}; o_{MV} + \Delta o, m)$. In practice, two complementary features are used: encoder coarse features $h^L$ (structural prior) and reconstruction-module fine features $h^H$ (textural detail), both guided by the same MV-based alignment.
    - **Design Motivation**: MVs are "free"—they are available at decode time and provide block-level displacement priors. However, their block-level nature means all pixels within a block share the same vector, leading to inaccuracies at object boundaries and under complex motion. By initializing the DCN offsets with MVs, the network only needs to learn **local residual offsets** rather than estimating full motion from scratch, greatly simplifying offset learning. Ablation results show that MV-only (OnlyMV) outperforms DCN-only (OnlyDCN) by 0.24 dB, and combining both (MVGDA) yields a further 0.17 dB gain.

2. **Residual-Map-Gated Fusion (RMGF)**:

    - **Function**: Selectively leverages reliable information from previous frames while suppressing interference from misaligned regions.
    - **Mechanism**: The residual map $Res_t$ represents pixel-level differences between the current frame and its motion-compensated prediction—large values indicate regions where motion compensation fails (occlusions, complex motion). A lightweight network converts the residual map into a spatial gating map: $M_t = \sigma(\mathcal{F}_{res}(Res_t))$. This gating weight is then used to suppress unreliable regions in the aligned previous-frame features: $h_t^f = \mathcal{C}^f([M_t \odot \hat{h}_{t-1}^L, M_t \odot \hat{h}_{t-1}^H, h_t^L])$.
    - **Design Motivation**: Naively concatenating inter-frame features propagates errors from misaligned regions. The residual map serves as a natural "reliability indicator"—large residual values directly identify regions where motion compensation has failed. Visualization of the gating heatmap confirms that stable regions (e.g., vehicle bodies) receive high weights, while dynamic regions (e.g., rotating wheels) are suppressed. Ablation results show that removing the gate (NoGate) reduces PSNR by 0.13 dB compared to RMGF.

3. **Frame-Type-Aware Reconstruction (FTAR)**:

    - **Function**: Adaptively allocates computational resources according to the differing importance of I-frames and P-frames.
    - **Mechanism**: I-frames contain complete spatial information and serve as critical references for subsequent frames; they are processed by a **high-capacity reconstruction branch** $\mathcal{R}_I$ (24 residual blocks) operating on encoder features $h_t^L$. P-frames primarily store incremental updates and appear far more frequently; they are processed by a **lightweight reconstruction branch** $\mathcal{R}_P$ (12 residual blocks) operating on fused features $h_t^f$. During inference, only the branch corresponding to the current frame type is activated.
    - **Design Motivation**: Applying identical computational budgets to all frames is inefficient—P-frames are over-computed, wasting resources, while under-computing I-frames degrades overall sequence quality. Ablation results show that I=P=12 (uniform lightweight) underperforms FTAR by 0.16 dB at nearly identical speed (10.7 ms vs. 10.8 ms), while I=P=24 (uniform heavy) gains only 0.04 dB over FTAR at 57% higher latency (16.8 ms). FTAR thus recovers most of the quality benefit at a negligible latency cost.

### Loss & Training
Training employs the Charbonnier loss: $\mathcal{L} = \frac{1}{T}\sum_{t=1}^T \sqrt{(I_t^{SR} - I_t^{GT})^2 + \epsilon^2}$. Inputs are H.264-encoded low-resolution video frames at CRF 18/23/28, with a 4× upsampling factor. The model is trained for 300K iterations with a batch size of 8, using 15-frame clips and 64×64 random crops. The Adam optimizer is used with an initial learning rate of $2 \times 10^{-4}$ and cosine annealing scheduling, on a single RTX 3090 GPU.

## Key Experimental Results

### Main Results

| Dataset / Method | PSNR (CRF18) | PSNR (CRF28) | FPS | MACs (G) | Real-time |
|------------------|--------------|--------------|-----|----------|-----------|
| CDA-VSR | **27.76** | **25.30** | **93** | **78** | Gaming real-time ✓ |
| TMP | 27.68 | 25.17 | 45 | 176 | Cinema real-time ✓ |
| BasicVSR* | 27.63 | 25.13 | 29 | 254 | Cinema real-time ✓ |
| KSNet-uni | 27.58 | 25.12 | 34 | 148 | Cinema real-time ✓ |
| RRN | 27.10 | 24.96 | 59 | 193 | Cinema real-time ✓ |

On Inter4K at 2K resolution: CDA-VSR achieves 29.98 dB at 25.1 FPS—the only method to exceed 24 FPS—compared to TMP at 29.76 dB / 11.4 FPS.

### Ablation Study

| Configuration | PSNR (CRF18) | Runtime (ms) | Notes |
|---------------|--------------|--------------|-------|
| OnlyMV | 27.59 | 10.2 | Coarse alignment via MV only |
| OnlyDCN | 27.35 | 10.6 | Deformable convolution only |
| OnlyGL (optical flow) | 27.73 | 15.5 | Optical flow alignment; 1.4× latency |
| **MVGDA** | **27.76** | **10.8** | Best quality with high efficiency |
| NoGate | 27.63 | 10.8 | Without residual-map gating |
| **RMGF** | **27.76** | **10.8** | Gated fusion gains 0.13 dB |
| I=12, P=12 | 27.60 | 10.7 | Uniform lightweight reconstruction |
| I=24, P=24 | 27.80 | 16.8 | Uniform heavy reconstruction |
| **I=24, P=12 (FTAR)** | **27.76** | **10.8** | Adaptive allocation |

### Key Findings
- **MV guidance substantially outperforms DCN-only**: OnlyMV exceeds OnlyDCN by 0.24 dB, demonstrating that compressed-domain motion vectors provide a strong motion prior, particularly for large-motion scenarios. MVGDA further improves upon OnlyMV by 0.17 dB, confirming that residual offset learning effectively corrects the block-level inaccuracies of MVs.
- **Residual maps serve as natural reliability indicators**: RMGF consistently outperforms NoGate by 0.08–0.13 dB across all CRF levels with negligible additional overhead (only 0.02M additional parameters).
- **FTAR is the key to efficiency**: The I=24, P=12 FTAR configuration recovers approximately 80% of the quality gain from the uniform heavy configuration at virtually zero latency cost (+0.1 ms), confirming that redundant computation on P-frames can be safely eliminated.
- **Efficiency advantage amplifies at higher resolutions**: CDA-VSR is the only method to achieve cinema real-time (>24 FPS) on Inter4K 2K (25.1 vs. TMP's 11.4 FPS), with the efficiency advantage growing with resolution.
- **Robustness across compression levels**: CDA-VSR maintains top performance across all CRF levels (18/23/28), with larger absolute gains at higher compression (CRF28: +0.13 dB over TMP), indicating that compressed-domain information is more valuable under higher compression rates.

## Highlights & Insights
- **"Free lunch" design philosophy**: Motion vectors, residual maps, and frame types are all byproducts of bitstream decoding, obtainable at zero additional computational cost. Repurposing these signals rather than discarding them reflects an elegant systems-level perspective that could be extended to other compressed-video tasks such as video editing and video analysis.
- **Complementary MV + DCN design**: MVs handle large-scale global motion (coarse alignment), while DCN is responsible only for local residual correction—a division of labor that simplifies and stabilizes offset learning. Heatmap visualizations clearly demonstrate that MVGDA produces the cleanest alignment results.
- **Differentiated processing via frame-type awareness**: Assigning different computational budgets to I- and P-frames is a simple yet effective strategy. Routing 97% of frames (P-frames) through the lightweight branch yields substantial overall acceleration, while the heavy branch reserved for the 3% I-frames preserves reference quality.

## Limitations & Future Work
- **H.264-only validation**: The method is evaluated solely on H.264-encoded video; the impact of motion vector quality differences in modern codecs (H.265/VVC/AV1) remains unexplored.
- **Fixed GOP structure**: The method assumes a standard I-P frame structure and does not address B-frame handling (though B-frames are not required in online streaming scenarios).
- **Dependence on MV quality**: MV accuracy degrades at low bitrates, potentially affecting alignment quality; the paper does not analyze extremely low-bitrate scenarios.
- **Unused quantization parameter (QP) information**: QP maps and other signals present in the bitstream are not exploited and could serve as additional priors for compression quality.
- **Dual-branch parameter overhead**: Although only one branch is active during inference, the total parameter count (3.3M) is slightly higher than some competing methods.

## Related Work & Insights
- **vs. TMP**: TMP propagates offsets by exploiting inter-frame motion continuity but still estimates motion purely from LR frames. CDA-VSR directly uses bitstream MVs as coarse motion priors, reducing the computational burden of motion estimation while consistently outperforming TMP across all CRF levels.
- **vs. CDVSR/CIAF**: Earlier compressed-domain VSR methods also utilize MVs and residual maps, but they are not designed for online scenarios and do not meet real-time requirements. CDA-VSR specifically addresses online constraints by introducing a frame-type-aware differentiated processing strategy.
- **vs. BasicVSR\***: BasicVSR\* is a causally constrained variant of BasicVSR with the backward propagation branch removed; it satisfies online constraints but remains slow (29 FPS). CDA-VSR is more than three times faster while achieving 0.13 dB higher PSNR.

## Rating
- **Novelty**: ⭐⭐⭐ Leveraging compressed-domain information is not entirely new, but the tailored module designs for three distinct types of information represent meaningful engineering innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparisons across multiple CRF levels, resolutions, and baselines, with thorough ablation studies and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with clear correspondence between motivation and methodology.
- **Value**: ⭐⭐⭐⭐ Offers direct practical value for real-world online video streaming super-resolution; 2K real-time processing represents a significant milestone.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution](../../ICCV2025/video_generation/vsrm_a_robust_mamba-based_framework_for_video_super-resolution.md)
- [\[CVPR 2026\] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)
- [\[CVPR 2026\] TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models](tear_temporal-aware_automated_red-teaming_for_text-to-video_models.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](../../ICCV2025/video_generation/vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICLR 2026\] Target-Aware Video Diffusion Models](../../ICLR2026/video_generation/target-aware_video_diffusion_models.md)

<!-- RELATED:END -->
