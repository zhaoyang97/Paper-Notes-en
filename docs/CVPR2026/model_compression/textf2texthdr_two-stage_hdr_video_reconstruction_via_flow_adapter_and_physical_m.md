---
title: >-
  [Paper Note] F²HDR: Two-Stage HDR Video Reconstruction via Flow Adapter and Physical Motion Modeling
description: >-
  [CVPR 2026][Model Compression][HDR video reconstruction] This paper proposes F²HDR, a two-stage HDR video reconstruction framework that adapts general-purpose pre-trained optical flow to alternating-exposure scenes via a Flow Adapter for robust alignment, and employs physical motion modeling to extract continuous motion masks from optical flow to guide artifact removal in the second stage, achieving state-of-the-art performance on real-world HDR video benchmarks.
tags:
  - CVPR 2026
  - Model Compression
  - HDR video reconstruction
  - flow adapter
  - physical motion modeling
  - alternating exposure
  - two-stage reconstruction
date: 2026-05-08
content_hash: 97cb76925394c740
---

# F²HDR: Two-Stage HDR Video Reconstruction via Flow Adapter and Physical Motion Modeling

**Conference**: CVPR 2026
**arXiv**: [2603.14920](https://arxiv.org/abs/2603.14920)
**Code**: [https://github.com/wei1895/F2HDR](https://github.com/wei1895/F2HDR)
**Area**: Model Compression
**Keywords**: HDR video reconstruction, flow adapter, physical motion modeling, alternating exposure, two-stage reconstruction

## TL;DR
This paper proposes F²HDR, a two-stage HDR video reconstruction framework that adapts general-purpose pre-trained optical flow to alternating-exposure scenes via a Flow Adapter for robust alignment, and employs physical motion modeling to extract continuous motion masks from optical flow to guide artifact removal in the second stage, achieving state-of-the-art performance on real-world HDR video benchmarks.

## Background & Motivation
Reconstructing HDR video from LDR frame sequences captured under alternating exposures is a cost-effective HDR acquisition approach. Long-exposure frames capture shadow detail but suffer from highlight saturation, while short-exposure frames preserve highlight information at the cost of increased noise in dark regions; their complementary fusion enables reconstruction of high dynamic range content.

**Key Challenge**: Cross-exposure frame alignment in dynamic scenes is extremely challenging — frames at different exposures exhibit large luminance discrepancies, and optical flow estimation is unreliable in regions with moving objects and occlusions, leading to ghosting artifacts and detail loss in the fused output.

**Two core limitations of existing methods**:

**Insufficient alignment accuracy in Stage I**:
   - Pre-trained general-purpose optical flow models perform poorly between frames of different exposures (matching fails in over/under-exposed regions)
   - Task-specific flow models retrained for HDR handle exposure variations but produce blurry flow boundaries, since they are trained with HDR reconstruction losses rather than optical flow prediction losses
   - Root cause: no optical flow dataset suitable for HDR reconstruction exists

**Lack of guidance for artifact removal in Stage II**:
   - Existing methods naively fuse same-modality inputs without explicit modeling of regions prone to motion artifacts
   - Shu et al. introduce exposure masks, but these reflect only highlight regions rather than genuinely difficult motion areas

**Core Idea**:
- Stage I: leverage sharp boundaries from pre-trained optical flow + learn a residual flow to adapt to HDR scenes → **best of both worlds**
- Stage II: extract motion masks via physical decomposition of optical flow (translation, divergence, curl, shear) → precisely localize artifact-prone regions → guide fusion

## Method

### Overall Architecture
Input: alternating-exposure LDR frame sequence $\{L_{t-1}, L_t, L_{t+1}\}$, with the middle frame $L_t$ as the reference. Output: HDR frame $H_t$.

**Stage I (Coarse Fusion)**: optical flow estimation → alignment → weighted fusion → motion modeling
**Stage II (Refinement)**: feature extraction → motion-guided modulation → feature enhancement → fusion decoding

### Key Designs

1. **Flow Adapter**:

    - Function: predicts residual optical flow on top of pre-trained flow to adapt to alternating-exposure scenes
    - Design Motivation: pre-trained optical flow yields sharp boundaries but handles over/under-exposed regions poorly; retrained flow handles exposure variation but produces blurry boundaries
    - Mechanism:
        - First apply exposure normalization to the reference frame to align adjacent frames in exposure: $g_{t\to t-1}(L_t) = \text{clip}(((L_t^\gamma / e_t) e_{t-1})^{1/\gamma})$
        - Feed the normalized frame pair into pre-trained flow network SEA-RAFT to obtain initial flow $f_{t\to t\pm1}$
        - **Flow Adapter** (a shallow residual CNN with varying dilation rates) receives concatenated input $\mathbf{x}_t = [L_{t-1}, L_t, L_{t+1}, f/\lambda, f/\lambda]$ and predicts residuals $[\Delta f_{t\to t-1}, \Delta f_{t\to t+1}] = \mathcal{A}(\mathbf{x}_t)$
        - Refined flow: $\tilde{f} = f + \lambda \Delta f$, where $\lambda=20$ is a fixed normalization factor
    - Key advantage: the adapter is jointly trained with the HDR reconstruction network, adapting the pre-trained flow to HDR scenes while preserving sharp motion boundaries

2. **Physical Motion Modeling**:

    - Function: extracts continuous physical motion masks from the optical flow field to identify regions prone to artifacts
    - Design Motivation: existing binary masks (occlusion masks, SAM masks) fail to capture the continuous physical characteristics of motion
    - Mechanism: compute first-order spatial derivatives of flow $f=[u,v]$ and decompose into four physical motion components:
        - **Translation**: $\|f\|_2 = \sqrt{u^2 + v^2}$ (global displacement magnitude)
        - **Divergence**: $\nabla \cdot f = u_x + v_y$ (scaling motion, e.g., approaching/receding)
        - **Curl**: $\nabla \times f = v_x - u_y$ (rotational motion)
        - **Shear**: $S = \frac{1}{2}(u_y + v_x)$ (local non-rigid deformation)
    - Unified motion energy: $E_m = \frac{w_t \odot \|f\|_2 + w_d \odot |\nabla \cdot f| + w_c \odot |\nabla \times f| + w_s \odot |S|}{w_t + w_d + w_c + w_s + \epsilon}$, with weights adaptively learned from optical flow via convolutional blocks
    - Multi-scale contrast enhancement: $E_s = E_m \odot (1 + 2 S_{multi})$, computing center-surround motion contrast at scales $\{1,2,4\}$ to sharpen boundary regions
    - Adaptive Otsu thresholding + sigmoid softening: $M = \frac{1}{2}[1 + \tanh(8(E_s - \tau))]$
    - The resulting mask $M \in [0,1]$ is continuous and differentiable, supporting end-to-end training

3. **Motion-Guided Feature Modulation (Stage II)**:

    - Function: modulate aligned LDR features using motion masks to focus the network on artifact-prone regions
    - Mechanism:
        - Extract features from each frame $L_t$ and HDR domain $I_t$ separately, then fuse via 1×1 convolution: $F_{LI_t} = \text{Conv}([F_{L_t}, F_{I_t}])$
        - Warp neighboring frame features $\tilde{F}_{LI_{t\pm1}}$ using optical flow
        - Encode motion information: $G_{t\pm1} = \sigma(\text{Conv}([\|\tilde{f}\|_2/\tilde{f}_{max}, M_{t\pm1}, \mathbf{1}]))$
        - Gated modulation: $\bar{F}_{LI_{t\pm1}} = G_{t\pm1} \odot \tilde{F}_{LI_{t\pm1}}$
    - Effect: regions with significant motion are emphasized (high gate values) while static regions are suppressed
    - Subsequently enhanced via three-branch dilated convolutions, fused with coarse HDR features, and decoded to the final output

### Loss & Training
- $\mu$-law tone mapping: $\mathcal{T}(H) = \frac{\log(1+\mu H)}{\log(1+\mu)}$, $\mu=5000$
- $\ell_1$ reconstruction loss: $\mathcal{L} = \|\mathcal{T}(H_{final}) - \mathcal{T}(H_{gt})\|_1$
- Training set: Vimeo-90K with simulated alternating exposures
- Adam optimizer, initial LR $1 \times 10^{-4}$, halved every 10 epochs, 50 epochs total, batch size 16
- Trained on a single RTX 3090 GPU

## Key Experimental Results

### Main Results — DeepHDRVideo Dataset

| Method | PSNR_T↑ | SSIM_T↑ | HDR-VDP-2↑ |
|--------|---------|---------|------------|
| Kalantari19 | 39.91 | 0.9329 | — |
| Chen21 | 43.32 | 0.9551 | 78.37 |
| HDRFlow | 43.03 | 0.9518 | 77.58 |
| NECHDR | 43.44 | 0.9558 | 77.28 |
| HDR-V-Diff | 42.07 | 0.9604 | — |
| **F²HDR** | **43.87** | **0.9573** | **78.88** |

### Main Results — Real-HDRV Dataset

| Method | PSNR_T↑ | SSIM_T↑ | HDR-VDP-2↑ | PSNR_L↑ |
|--------|---------|---------|------------|---------|
| Chen21 | 40.79 | 0.9510 | 76.50 | 50.30 |
| HDRFlow | 40.34 | 0.9481 | 75.82 | 49.72 |
| NECHDR | 40.88 | 0.9518 | 76.77 | 50.41 |
| **F²HDR** | **41.01** | **0.9538** | **76.93** | **50.51** |

### Ablation Study

| Configuration | DeepHDRVideo PSNR_T | Real-HDRV PSNR_T |
|---------------|---------------------|------------------|
| Stage I w/o Flow Adapter | 43.06 | 40.29 |
| Stage I w/o FA + Stage II | 43.21 | 40.45 |
| Stage I (with FA) | 43.39 | 40.61 |
| Stage I w/o Mask + Stage II | 43.68 | 40.87 |
| Stage I w/ CNN Mask + Stage II | 43.76 | 40.93 |
| **Full (Stage I + Stage II)** | **43.87** | **41.01** |

### Optical Flow Quality Evaluation (Real-HDRV)

| Method | EPE↓ | LDR Warping PSNR↑ | LDR Warping SSIM↑ |
|--------|------|-------------------|-------------------|
| HDRFlow | 3.53 | 29.93 | 0.8763 |
| NECHDR | 3.30 | 30.31 | 0.8901 |
| Ours w/o FA | 1.81 | 30.39 | 0.8897 |
| **Ours** | **2.08** | **30.84** | **0.8922** |

### Key Findings
- The Flow Adapter is the most critical component: removing it causes a 0.81 dB drop in PSNR on DeepHDRVideo (43.87→43.06)
- Physical motion masks contribute 0.19 dB, outperforming CNN-learned masks (0.11 dB), validating the value of physical priors
- Noteworthy finding: adding the Flow Adapter increases EPE (2.08 > 1.81) while improving warping quality — because the adapter generates non-zero flow offsets in occluded regions where pseudo-labels are zero
- The two-stage scheme substantially outperforms the single-stage baseline
- Inference speed (0.29s @ 1080p) lies between the fastest method HDRFlow (0.076s) and the slowest LAN-HDR (0.905s)

## Highlights & Insights
- **Elegant Flow Adapter design**: rather than retraining optical flow from scratch, the method learns a residual, combining the sharp boundaries of pre-trained models with task-specific adaptability
- **Physical motion decomposition** has theoretical elegance: decomposing flow derivatives into four physically interpretable components (translation, divergence, curl, shear); the continuous differentiable mask outperforms binary alternatives
- The second-stage optimization back-propagates gradients into the first-stage optical flow, enabling mutual reinforcement between stages
- The exposure normalization trick (aligning to neighboring frame exposure before feeding into pre-trained flow) is simple yet effective

## Limitations & Future Work
- The method takes three frames as input ($t-1, t, t+1$); a longer temporal window may further improve temporal stability
- The Flow Adapter is a shallow CNN, which may lack capacity for extremely large displacements
- Training data relies on simulated exposures from Vimeo-90K; domain gap may affect generalization to real-world scenes
- Physical motion decomposition is based on first-order derivatives; higher-order motion features (e.g., acceleration) are not considered
- Inference speed leaves room for optimization (4× slower than HDRFlow)
- Evaluation lacks temporal consistency metrics (e.g., tOF, tLP)

## Related Work & Insights
- HDRFlow's lightweight task-specific flow vs. the adapter-based approach in this paper represent two complementary paradigms, each with distinct trade-offs
- Physical motion decomposition is transferable to other motion-aware video processing tasks (deblurring, video stabilization, etc.)
- The Flow Adapter concept parallels adapter tuning in NLP, reflecting a broadly applicable paradigm for adapting pre-trained models to downstream tasks

## Rating
- Novelty: ⭐⭐⭐⭐ Both the Flow Adapter (residual flow adaptation of pre-trained models) and physical motion modeling represent clear technical contributions
- Experimental Thoroughness: ⭐⭐⭐⭐ Two real-world datasets + optical flow quality evaluation + comprehensive ablations + inference speed comparison
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; the correspondence between two core problems and two proposed solutions is well-structured
- Value: ⭐⭐⭐⭐ A practical framework for HDR video reconstruction; the Flow Adapter concept has broader applicability

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](../../ICLR2026/model_compression/s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[CVPR 2026\] UniComp: Rethinking Video Compression Through Informational Uniqueness](unicomp_rethinking_video_compression_through_informational_uniqueness.md)
- [\[CVPR 2026\] Generative Video Compression with One-Dimensional Latent Representation](generative_video_compression_with_one-dimensional_latent_representation.md)
- [\[ICCV 2025\] Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP](../../ICCV2025/model_compression/gain-mlp_improving_hdr_gain_map_encoding_via_a_lightweight_mlp.md)
- [\[CVPR 2026\] PriVi: Towards a General-Purpose Video Model for Primate Behavior in the Wild](privi_towards_a_general-purpose_video_model_for_primate_behavior_in_the_wild.md)

<!-- RELATED:END -->
