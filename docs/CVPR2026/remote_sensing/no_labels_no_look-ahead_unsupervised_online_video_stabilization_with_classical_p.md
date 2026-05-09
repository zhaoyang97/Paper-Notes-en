---
title: >-
  [Paper Note] No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors
description: >-
  [CVPR 2026][Remote Sensing][Video stabilization] This paper proposes LightStab, an unsupervised online video stabilization framework built upon the classical three-stage pipeline (motion estimation → motion propagation → motion compensation) augmented with multi-threaded asynchronous buffering. LightStab is the first online method to comprehensively match offline SOTA across 5 benchmark datasets, and introduces UAV-Test, the first multimodal UAV aerial stabilization benchmark covering both visible-light and infrared imagery.
tags:
  - CVPR 2026
  - Remote Sensing
  - Video stabilization
  - unsupervised
  - online processing
  - optical flow estimation
  - UAV
date: 2026-05-08
content_hash: 9afe53c7915799a6
---

# No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors

**Conference**: CVPR 2026
**arXiv**: [2602.23141](https://arxiv.org/abs/2602.23141)
**Code**: [GitHub](https://github.com/liutao23/LightStab.git)
**Area**: Remote Sensing
**Keywords**: Video stabilization, unsupervised, online processing, optical flow estimation, UAV

## TL;DR

This paper proposes LightStab, an unsupervised online video stabilization framework built upon the classical three-stage pipeline (motion estimation → motion propagation → motion compensation) augmented with multi-threaded asynchronous buffering. LightStab is the first online method to comprehensively match offline SOTA across 5 benchmark datasets, and introduces UAV-Test, the first multimodal UAV aerial stabilization benchmark covering both visible-light and infrared imagery.

## Background & Motivation

**Background**: Video stabilization aims to suppress camera shake and improve visual quality. Classical approaches follow a three-stage pipeline of motion estimation → motion smoothing → frame compensation, and are categorized by motion model dimensionality into 2D (affine/homography/optical flow), 2.5D (limited 3D cues), and 3D (depth + point cloud) methods. Deep learning methods (DUT, NNDVS, RStab, etc.) generate stabilized frames directly via end-to-end learning.

**Limitations of Prior Work**:
- **Perception limitations**: Classical methods rely on hand-crafted feature detectors (SIFT, ORB, etc.), which are not robust under weak texture, occlusion, and large motion, and non-uniform keypoint distributions introduce bias in motion estimation.
- **Smoothing limitations**: Fixed smoothing strategies fail to generalize, resulting in residual jitter; learned smoothing lacks geometric interpretability and may over-smooth or introduce distortion.
- **Online processing limitations**: Most high-quality stabilizers (classical and learned) rely on offline batch processing or future frames, introducing latency. Learned methods also require large amounts of paired labeled data and substantial computational resources.

**Key Challenge**: It is inherently difficult to simultaneously achieve unsupervised training, online (causal) processing, and high stabilization quality. The best existing online method (NNDVS) still shows notable gaps in certain scenarios, and existing benchmarks primarily consist of handheld visible-light videos, failing to cover practical settings such as nighttime UAV remote sensing.

**Goal**: Design a fully unsupervised, strictly causal (no future frames) online video stabilization framework that achieves quality comparable to or exceeding offline SOTA, and generalizes to multimodal UAV scenarios.

**Key Insight**: Rather than pursuing an end-to-end approach, the authors return to the classical three-stage pipeline while equipping each stage with modern components—replacing single hand-crafted detectors with multi-detector collaboration and optical flow, replacing fixed filtering with a lightweight self-supervised network, and eliminating serial latency bottlenecks via multi-threaded parallelism.

**Core Idea**: Classical three-stage pipeline + modern components (multi-detector collaboration, causal optical flow fusion, self-supervised motion propagation network, dynamic-kernel online smoothing) + system-level multi-threading optimization = unsupervised, online, high-quality stabilization.

## Method

### Overall Architecture

A strictly causal three-stage pipeline in which all operations depend only on past frames:

1. **Motion Estimation**: Multi-detector collaboration for keypoint detection → SSC-based spatial uniformization → MemFlow causal optical flow estimation → sparse keypoint-guided flow field fusion → output motion feature vector $\mathbf{m}_t = [x_{kp}; y_{kp}; u; v]$
2. **Motion Propagation**: EfficientMotionPro network propagates sparse keypoint motion to a dense grid motion field $\Delta g_t$, based on multi-homography priors and residual learning.
3. **Motion Compensation**: OnlineSmoother network applies learnable causal kernels to smooth grid trajectories, generating a compensation displacement field and rendering the stabilized frame.

The three stages are executed in parallel via a multi-threaded asynchronous pipeline (TME/TMP/TMC) communicating through FIFO shared queues. Throughput is determined by the slowest stage, with theoretical speedup $S = (t_{est} + t_{prop} + t_{smooth}) / \max\{t_{est}, t_{prop}, t_{smooth}\}$.

### Key Designs

1. **Multi-Detector Collaboration + Keypoint Uniformization**:

    - **Function**: Fuses keypoints from multiple heterogeneous feature detectors and ensures uniform spatial distribution via spatially selective clustering.
    - **Mechanism**: A detector ensemble $\mathcal{D} = \{D_m^{trad}\} \cup \{D_n^{deep}\}$ extracts keypoints independently; after confidence normalization, they are merged via NMS: $\tilde{K}_t = \text{NMS}(\bigcup_j w_j \cdot \tilde{K}_t^{(j)})$. SSC then partitions the image into a $G_x \times G_y$ grid, selecting the top-$k$ highest-confidence points per cell while enforcing a minimum spacing $\tau$.
    - **Design Motivation**: Single detectors tend to cluster in texture-rich regions (e.g., SIFT, SuperPoint), biasing motion estimation toward local areas. Visualizations confirm that multi-detector collaboration achieves more uniform spatial coverage.

2. **EfficientMotionPro Self-Supervised Motion Propagation Network**:

    - **Function**: Propagates sparse keypoint motion into a dense grid motion field.
    - **Mechanism**: A multi-homography prior (K-means clustering + RANSAC estimation of $K_{homo}$ homographies, soft-weight blending) establishes a base displacement $\Delta g_{base,t}$; a lightweight Ghost+ECA backbone then predicts a residual $\Delta g_{res,t}$. The total loss comprises: keypoint consistency loss $\mathcal{L}_{kp}$ (Charbonnier penalty + adaptive confidence weighting), homography projection consistency loss $\mathcal{L}_{proj}$, and grid structure preservation loss $\mathcal{L}_{struct}$ (orthogonality constraint to prevent shear distortion). The model has only ~22.9K parameters, with computational cost linear in the number of keypoints.
    - **Design Motivation**: Decomposing motion propagation into "multi-homography prior + non-rigid residual" constrains the network to learn only deviations from the rigid model, reducing learning difficulty. Multi-homography modeling handles complex scenes containing dynamic objects.

3. **OnlineSmoother Dynamic-Kernel Online Smoothing**:

    - **Function**: Smooths grid trajectories online to suppress high-frequency jitter while preserving intentional motion.
    - **Mechanism**: A Lite LS-3D encoder extracts spatiotemporal features, and a Star-gated decoder predicts 3-tap causal kernels (3 kernel coefficients each for $x$ and $y$ directions). The smoothing formula is: $S_t^x = \frac{\lambda \sum_r k_{t,r}^x S_{t-r}^x + O_t^x}{1 + \lambda \sum_r |k_{t,r}^x|}$ ($\lambda=100$), with an effective temporal window of $L=7$ frames. Training losses include: temporally adaptive second-order penalty $\mathcal{L}_{time}$ (with motion-magnitude adaptive decay $\beta$), frequency-domain high-frequency suppression $\mathcal{L}_{freq}$ (DFT frequency weighting), spatial distortion constraint $\mathcal{L}_{spatial}$ (triangular mesh edge ratio + angle preservation), and keypoint projection consistency $\mathcal{L}_{proj}$.
    - **Design Motivation**: Fixed filters (Gaussian, mean) cannot adapt to varying motion, leading to over-smoothing or insufficient suppression. Learnable causal kernels dynamically adjust smoothing strength based on current motion, while the frequency-domain loss explicitly suppresses high-frequency oscillations.

### Loss & Training

**EfficientMotionPro**: $\mathcal{L} = 10\mathcal{L}_{kp} + 40\mathcal{L}_{proj} + 40\mathcal{L}_{struct}$, Adam optimizer, OneCycleLR (peak lr $2\times10^{-4}$), 100 epochs, batch=64, ~12h on a single RTX 4090.

**OnlineSmoother**: $\mathcal{L}_{total} = \mathcal{L}_{temp} + 10\mathcal{L}_{spatial} + 5\mathcal{L}_{proj}$, where $\mathcal{L}_{temp} = 20\mathcal{L}_{time} + \mathcal{L}_{freq}$. batch=1 to preserve causality, gradient clipping threshold 5.0, ~2.5h.

Frame-boundary black borders are filled via ProPainter outpainting as a post-processing step.

## Key Experimental Results

### Main Results

Comparison across 5 datasets using Cropping Ratio (C), Distortion Value (D), and Stability Score (S), all higher is better:

| Method | Type | NUS (C/D/S) | DeepStab (C/D/S) | Selfie (C/D/S) | GyRo (C/D/S) | UAV-Test (C/D/S) |
|--------|------|------------|-----------------|---------------|-------------|-----------------|
| DUT | Offline | 0.98/0.88/0.85 | 0.99/0.95/0.95 | 0.99/0.98/0.93 | 0.99/0.98/0.89 | 0.95/0.89/0.94 |
| RStab | Offline | 1.00/0.99/0.94 | 1.00/0.98/0.96 | 1.00/0.92/0.95 | 1.00/0.95/0.92 | 1.00/0.96/0.94 |
| NNDVS | Online | 0.92/0.98/0.87 | 0.93/0.91/0.84 | 0.97/0.92/0.91 | 0.99/0.93/0.88 | 0.89/0.87/0.84 |
| Liu et al. | Online | 0.72/0.89/0.89 | 0.89/0.88/0.85 | 0.79/0.89/0.85 | 0.99/0.94/0.89 | 0.82/0.89/0.85 |
| **Ours** | **Online** | **0.95/0.98/0.90** | **0.94/0.91/0.85** | **0.98/0.93/0.91** | **0.99/0.96/0.93** | **0.94/0.90/0.89** |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| w/o MP (A1) | Removing motion propagation degrades D and PSNR, weakening global motion modeling. |
| w/o TS (A2) | Removing trajectory smoothing reduces structural stability and worsens D. |
| w/o MP&TS (A3) | Removing both causes the largest performance drop, demonstrating their complementarity. |
| w/o Loss_kp (A4) | Removing keypoint consistency loss weakens motion supervision. |
| w/o Homo (A6) | Replacing multi-homography with a single homography introduces jitter and local distortion. |
| w/o KPC (A7) | Disabling collaborative detection yields non-uniform keypoints and lower D. |
| Window L=5/7/9 | L=5 improves stability but reduces fidelity; L=9 increases cost without consistent gain; L=7 is optimal. |
| Full model (A10) | All modules + L=7 achieves the highest overall score. |

### Key Findings

- **Online first matches offline**: On the GyRo dataset (C=0.99, D=0.96, S=0.93), the proposed online method is competitive with the strongest offline method Gavs (C=1.00, D=0.99, S=0.93).
- **Clear advantage on UAV-Test**: The method comprehensively outperforms existing online approaches on the new UAV dataset (vs. NNDVS: +0.05 C, +0.03 D, +0.05 S).
- **Embedded platform feasibility**: Achieves ~13 FPS (78.94 ms/frame) on Jetson AGX Orin, more than 4× faster than NNDVS (2.94 FPS).
- **High complementarity of motion propagation and trajectory smoothing**: Removing either component individually yields limited degradation, but removing both (A3) causes the largest performance drop.

## Highlights & Insights

- **Hybrid strategy of classical pipeline + modern components**: Rather than adopting an end-to-end black-box approach, the method preserves the interpretability and controllability of the three-stage pipeline while replacing the weakest link in each stage with learned components. This principled engineering hybrid is more suitable for real-world deployment than pure end-to-end methods.
- **Self-supervised training eliminates data dependency**: Both core networks (EfficientMotionPro and OnlineSmoother) are trained with self-supervised objectives, completely avoiding the need for paired stable/unstable video data, which is critical for practical applicability.
- **Elegant engineering of the multi-threaded asynchronous pipeline**: Serial latency $t_1+t_2+t_3$ is reduced to $\max(t_1,t_2,t_3)$ through a FIFO queue back-pressure mechanism that ensures resource safety.

## Limitations & Future Work

- **Dependence on an external optical flow estimator**: MemFlow is used for causal optical flow; its accuracy may be insufficient in complex scenes. Exploring more accurate and efficient optical flow models is a promising direction.
- **Frame outpainting is not online**: Black-border inpainting via ProPainter is a computationally expensive post-processing step that is not integrated into the online pipeline. Developing more lightweight, online-friendly outpainting techniques is needed.
- **Lambertian camera model limitation**: The simple 2D motion model may fail under large parallax and significant 3D structural variation.
- **UAV-Test is limited in scale**: With only 92 sequences and limited scene diversity, it serves as a starting point for a larger-scale UAV stabilization benchmark.

## Related Work & Insights

- **vs. DUT**: DUT also combines a classical pipeline with neural networks but operates offline and relies on a global smoothing strategy. The key distinction of this work is its online causal design (no access to future frames), with motion propagation and smoothing trained independently.
- **vs. NNDVS**: NNDVS achieves online stabilization using existing motion estimation frameworks but lacks open-source motion estimators and exhibits insufficient robustness in complex scenes. This work significantly improves motion perception robustness through multi-detector collaboration and keypoint uniformization.
- **vs. RStab**: RStab is the strongest offline method, achieving very high quality via neural rendering and adaptive modules, but requires future frames. The proposed method achieves comparable performance under online constraints while being substantially faster at inference.

## Rating

- Novelty: ⭐⭐⭐⭐ — The component-level designs (multi-detector collaboration, multi-homography prior, causal dynamic kernels) reflect systematic innovation, though the core concept remains a modernization of the classical pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five datasets, full ablation, user study, embedded platform evaluation, and extensive visualizations; exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, complete derivations, and highly detailed supplementary material.
- Value: ⭐⭐⭐⭐ — First online method to match offline performance, unsupervised training, new benchmark dataset; strong practical impact.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels](lumosaic_hyperspectral_video_via_active_illumination_and_coded-exposure_pixels.md)
- [\[CVPR 2026\] Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction](exploring_spatiotemporal_feature_propagation_for_video-level_compressive_spectra.md)
- [\[AAAI 2026\] UniABG: Unified Adversarial View Bridging and Graph Correspondence for Unsupervised Cross-View Geo-Localization](../../AAAI2026/remote_sensing/uniabg_unified_adversarial_view_bridging_and_graph_correspondence_for_unsupervis.md)
- [\[ICLR 2026\] Spectral Gaps and Spatial Priors: Studying Hyperspectral Downstream Adaptation Using TerraMind](../../ICLR2026/remote_sensing/spectral_gaps_and_spatial_priors_studying_hyperspectral_downstream_adaptation_us.md)
- [\[CVPR 2026\] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)

<!-- RELATED:END -->
