---
title: >-
  [Paper Note] Learning to Robustly Reconstruct Dynamic Scenes from Low-Light Spike Streams
description: >-
  [ECCV 2024][Image Restoration][Spike Camera] To address the reconstruction difficulties caused by sparse information of spike cameras in low-light environments, this paper proposes a bidirectional recurrent reconstruction framework. Its core is a light-robust representation (LR-Rep) that aggregates temporal information through global spike interval (GISI), combined with a feature fusion module to extract temporal features. The paper also constructs a dedicated low-light high-…
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Spike Camera"
  - "Low-Light Reconstruction"
  - "Bidirectional Recurrent Network"
  - "Light-Robust Representation"
  - "High-Speed Scenes"
date: 2026-05-08
content_hash: 750db7dbf693793e
---

# Learning to Robustly Reconstruct Dynamic Scenes from Low-Light Spike Streams

**Conference**: ECCV 2024  
**arXiv**: [2401.10461](https://arxiv.org/abs/2401.10461)  
**Code**: [GitHub](https://github.com/Acnext/Learning-to-Robustly-Reconstruct-Dynamic-Scenes-from-Low-light-Spike-Streams/)  
**Area**: Image Reconstruction / Neuromorphic Vision  
**Keywords**: Spike Camera, Low-Light Reconstruction, Bidirectional Recurrent Network, Light-Robust Representation, High-Speed Scenes  

## TL;DR

To address the reconstruction difficulties caused by sparse information of spike cameras in low-light environments, this paper proposes a bidirectional recurrent reconstruction framework. Its core is a light-robust representation (LR-Rep) that aggregates temporal information through global spike interval (GISI), combined with a feature fusion module to extract temporal features. The paper also constructs a dedicated low-light high-speed dataset, substantially outperforming existing methods on both synthetic and real-world data.

## Background & Motivation

**Background**: Spike Camera is a neuromorphic sensor with ultra-high temporal resolution (40,000 Hz) that records pixel-wise light intensity by accumulating photons and emitting continuous binary spike streams. Compared to traditional cameras and event cameras, spike cameras can directly capture light intensity information, showing great potential in tasks such as high-speed scene reconstruction, optical flow estimation, and depth estimation. Recently, deep learning methods (e.g., S2I, WGSE) have significantly improved the reconstruction quality of spike cameras.

**Limitations of Prior Work**: Existing methods perform well under normal lighting, but their performance drops sharply in low-light environments. The root cause lies in the working principle of the spike camera—under low light, pixels take longer to accumulate photons to reach the firing threshold, leading to extremely sparse spike streams and a massive reduction in effective information. The representations relied upon by existing methods (e.g., TFI, LISI) fail to extract sufficient temporal information from sparse spike streams. Moreover, there is a lack of specialized datasets for low-light high-speed scenes to evaluate different methods.

**Key Challenge**: The sparseness of spikes under low-light conditions leads to insufficient information, yet the core value of spike cameras precisely lies in the temporal information of high-speed scenes. How to still fully exploit temporal information under insufficient information conditions becomes the key challenge.

**Goal**: (1) How to design a spike stream representation method robust to low light to extract more information from sparse spikes? (2) How to construct a reliable low-light high-speed scene dataset?

**Key Insight**: The authors observe that while spikes within a single time window are sparse, spikes in adjacent forward and backward time windows can provide complementary information. The key is to utilize the spike firing times across time windows to construct global interval information (rather than using only local intervals) and accumulate temporal features through a bidirectional recurrent structure.

**Core Idea**: Use forward and backward spike firing times to construct a Global Inter-Spike Interval (GISI) to replace the Local Inter-Spike Interval (LISI), and fully exploit the temporal information of low-light sparse spike streams through a bidirectional recurrent framework.

## Method

### Overall Architecture

The input consists of $K$ continuous spike streams $\{S_{t_i}\}$, each of size $H \times W \times (2\Delta t + 1)$ (time window of 41 frames). For each time step $t_i$, the light-robust representation $\text{Rep}_{t_i}$ is first calculated via LR-Rep, and then deep features $F_{t_i}$ are extracted using ResNet. Then, the forward and backward fusion modules merge adjacent temporal features with the current features to generate forward temporal features $F_{t_i}^f$ and backward temporal features $F_{t_i}^b$, respectively. Finally, the bidirectional features are concatenated and passed through a convolutional layer to reconstruct the image $\hat{Y}_{t_i}$ at the current time step.

### Key Designs

1. **Global Inter-Spike Interval Transform (GISI Transform)**:

    - **Function**: Extract richer temporal information from sparse spike streams than local intervals.
    - **Mechanism**: Traditional Local Inter-Spike Interval (LISI) only computes the time intervals between adjacent spikes within the current time window. GISI leverages the "release time" propagation mechanism of forward/backward spikes to extend the information coverage to the entire sequence. Specifically, taking the backward direction as an example, this is done in three steps: (a) compute the LISI of the current window; (b) update the GISI using the backward spike release time $Spike_{t_{i+1}}^b$ passed back from the next time window—if a pixel in current window has no spikes, the propagated release time is used to fill the interval information; (c) maintain and propagate the backward release time $Spike_{t_i}^b$ of the current window to the previous window. The key to GISI is to "borrow" information from other time windows through cross-window propagation of release times when most pixels have no spikes in low light.
    - **Design Motivation**: Under low light, the interval values of many pixels in LISI are zero or saturated (due to lack of spikes in the window). Through the global propagation mechanism, GISI enables even pixels without spikes in the current window to obtain effective interval estimations. Experimental visualization shows that the distribution of GISI information is more even than that of LISI. The extra overhead is only two $400 \times 250$ matrices to store the release times, which does not affect network parameters or efficiency.

2. **Light-Robust Representation (LR-Rep)**:

    - **Function**: Fuse GISI and original spike streams into a feature representation robust to illumination changes.
    - **Mechanism**: Shallow features $F_G$ and $F_S$ are extracted from GISI and input spike streams respectively using convolutional blocks, and then adaptively fused via an attention module. The attention module (3 convolutional layers + activation function) predicts two channel weights $\beta$ and $\alpha$, resulting in the final representation $\text{Rep}_{t_i} = \beta_{t_i} F_G + \alpha_{t_i} F_S$. The attention mechanism allows the network to automatically decide whether to rely more on GISI features or the original spike features based on current lighting conditions.
    - **Design Motivation**: GISI captures global temporal information but may lose instantaneous details, while the original spike stream preserves instantaneous details but is too sparse under low light. Adaptive fusion lets the model dynamically adjust information sources under different lighting conditions: relying more on the global information provided by GISI under low light, and utilizing more of the instantaneous details of original spikes under normal light.

3. **Bidirectional Fusion with Alignment**:

    - **Function**: Fuse adjacent temporal features in a recurrent structure while handling motion alignment.
    - **Mechanism**: The forward fusion merges $F_{t_{i-1}}^f$ and $F_{t_i}$ into $F_{t_i}^f$, and the backward fusion merges $F_{t_{i+1}}^b$ and $F_{t_i}$ into $F_{t_i}^b$. Before fusion, a Pyramid, Cascaded, and Deformable convolution (PCD) module is used to align features from different timestamps, avoiding misalignment caused by motion. The aligned features are concatenated with original features and then go through a feature extraction module. The final reconstruction uses 3 convolutional layers: $\hat{Y}_{t_i} = c([F_{t_i}^b, F_{t_i}^f])$.
    - **Design Motivation**: Bidirectional recurrence allows utilizing both past and future information at each moment, which is extremely crucial when low-light information is insufficient. The PCD alignment module resolves the spatial misalignment of features across different times in high-speed motion scenes; directly concatenating unaligned features would lead to motion blur.

### Loss & Training

L1 loss is used: $\mathcal{L} = \sum_{i=1}^{K} \|\hat{Y}_{t_i} - Y_{t_i}\|_1$. The training set is the self-built RLLR (100 random low-light high-speed scenes), with spatial cropping to $64 \times 64$, time window of 41, keeping 21 consecutive spike streams. Adam optimizer ($\beta_1=0.9, \beta_2=0.99$) is used, with an initial learning rate of 1e-4, decaying by 10 times after 70 epochs, trained for 100 epochs in total on a single A100 GPU.

## Key Experimental Results

### Main Results

| Method | Source | PSNR(↑) | SSIM(↑) | Gain(PSNR) |
|------|------|---------|---------|-----------|
| TFI | ICME'19 | 31.41 | 0.723 | baseline |
| STP | CVPR'21 | 24.88 | 0.555 | -6.53 |
| S2I | CVPR'21 | 40.88 | 0.959 | +9.47 |
| SSML | IJCAI'22 | 38.43 | 0.899 | +7.02 |
| RSIR | MM'23 | 34.12 | 0.883 | +2.71 |
| WGSE | AAAI'23 | 42.96 | 0.971 | +11.55 |
| **Ours** | **Ours** | **45.08** | **0.987** | **+13.67** |

### Ablation Study

| Configuration | PSNR | SSIM | Explanation |
|------|------|------|------|
| (A) Basic baseline | 42.74 | 0.974 | No LR-Rep, No temporal fusion |
| (B) + ADF (temporal fusion) | 44.15 | 0.985 | Bidirectional temporal features +1.41 |
| (C) + LR-Rep | 44.74 | 0.986 | Light-robust representation +2.00 |
| (D) + ADF + LR-Rep | 44.96 | 0.987 | Combination +2.22 |
| (E) + ADF + LR-Rep + AIF | **45.08** | **0.987** | Plus alignment +2.34 |
| (F) Replace GISI with LISI | 45.00 | 0.987 | GISI is slightly better than LISI |

### Key Findings

- LR-Rep contributes the most (+2.00 PSNR), proving that light-robust representation is crucial for low-light reconstruction.
- Bidirectional temporal fusion is the second most important (+1.41 PSNR), validating the necessity of accumulating information across time windows.
- GISI yields a stable, small improvement compared to LISI, with almost zero extra computational overhead.
- Excellent performance is also shown on real low-light spike data—while other methods introduce massive motion blur or dark background artifacts, this method can recover sharp textures.
- When replacing the representation with those of other methods (such as TFI, AST, AMIM, etc.), performance is consistently lower than LR-Rep, showing that LR-Rep is the best fit for this framework.
- User study rankings on both datasets are consistently first.

## Highlights & Insights

- **Elegant Design of GISI**: Propagating spike release times across windows to compute global intervals is essentially an information borrowing mechanism. Under low light when the current window lacks spikes, it utilizes spikes from other windows to complement information. This idea of "borrowing info across time" can be generalized to any scene processing sparse temporal signals.
- **Rigorous Dataset Design** is worth learning from—the light source types and powers of the LLR dataset match the real world, and the motion comes from real-world scenes. This "realistic synthesis" ensures that the performance on synthetic data can generalize to real data.
- The adaptive attention fusion of LR-Rep allows the network to automatically adjust strategies under different lighting conditions, without manual mode switching.

## Limitations & Future Work

- The dataset scale is relatively small (RLLR has only 100 scenes), which may limit the generalization ability of the network.
- Only L1 loss is used, and perceptual loss or GAN loss has not been explored to improve visual quality.
- The bidirectional recurrent structure needs to wait for all inputs before processing, making it unsuitable for strict real-time applications.
- No comparison with the latest Transformer-based video restoration methods.
- The improvement of GISI over LISI is modest (only 0.08 PSNR), suggesting there might be better ways for global information extraction.

## Related Work & Insights

- **vs WGSE (AAAI'23)**: WGSE suppresses noise via wavelet transform but remains restricted by sparse inputs in low light. This paper obtains more information globally via GISI, exceeding its PSNR by 2.12 dB.
- **vs RSIR (MM'23)**: RSIR's AST representation compresses the spike stream into a spike count map, losing dynamic information and leading to severe motion blur in high-speed scenes. LR-Rep preserves temporal dynamic information.
- **vs EDVR (Video Restoration)**: This paper draws inspiration from EDVR's PCD alignment module, proving that video restoration techniques can be migrated to spike camera reconstruction.

## Rating

- Novelty: ⭐⭐⭐⭐ The concepts of GISI global interval and LR-Rep adaptive fusion design are novel, but the overall pipeline is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Synthetic data + real data + user study, very detailed ablation studies including comparison of representations and the impact of data volume.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly explained with rich and illustrative figures.
- Value: ⭐⭐⭐⭐ Provides significant technical support for the practical application of spike cameras in low-light scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Unrolled Decomposed Unpaired Learning for Controllable Low-Light Video Enhancement](unrolled_decomposed_unpaired_learning_for_controllable_low-light_video_enhanceme.md)
- [\[ECCV 2024\] Towards Real-world Event-guided Low-light Video Enhancement and Deblurring](towards_real-world_event-guided_low-light_video_enhancement_and_deblurring.md)
- [\[ECCV 2024\] Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators](efficient_diffusion_transformer_with_step-wise_dynamic_attention_mediators.md)
- [\[CVPR 2026\] BiEvLight: Bi-level Learning of Task-Aware Event Refinement for Low-Light Image Enhancement](../../CVPR2026/image_restoration/bievlight_bi-level_learning_of_task-aware_event_refinement_for_low-light_image_e.md)
- [\[CVPR 2026\] DeSpike: Defocus Deblurring and Image Reconstruction for Spike Camera](../../CVPR2026/image_restoration/seeing_through_blur_tackling_defocus_in_spike-based_imaging.md)

</div>

<!-- RELATED:END -->
