---
title: >-
  [Paper Note] Towards Real-world Event-guided Low-light Video Enhancement and Deblurring
description: >-
  [ECCV 2024][Image Restoration][Event Camera] This paper introduces the joint task of event-guided low-light video enhancement and deblurring for the first time. It constructs a beam-splitter-based real-world dataset, RELED, and designs an end-to-end framework consisting of two core modules: Event-Guided Deformable Temporal Feature Alignment (ED-TFA) and Spectrum Frequency-based Cross-Modal Feature Enhancement (SFCM-FE), outperforming previous state-of-the-art methods by over…
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Event Camera"
  - "Low-Light Enhancement"
  - "Motion Deblurring"
  - "Cross-Modal Fusion"
  - "Temporal Alignment"
date: 2026-05-08
content_hash: 1fe7a55b01b32b8b
---

# Towards Real-world Event-guided Low-light Video Enhancement and Deblurring

**Conference**: ECCV 2024  
**arXiv**: [2408.14916](https://arxiv.org/abs/2408.14916)  
**Code**: [https://github.com/intelpro/ELEDNet](https://github.com/intelpro/ELEDNet)  
**Area**: Image Restoration / Low-Light Video Enhancement and Motion Deblurring  
**Keywords**: Event Camera, Low-Light Enhancement, Motion Deblurring, Cross-Modal Fusion, Temporal Alignment

## TL;DR

This paper introduces the joint task of event-guided low-light video enhancement and deblurring for the first time. It constructs a beam-splitter-based real-world dataset, RELED, and designs an end-to-end framework consisting of two core modules: Event-Guided Deformable Temporal Feature Alignment (ED-TFA) and Spectrum Frequency-based Cross-Modal Feature Enhancement (SFCM-FE), outperforming previous state-of-the-art methods by over 1.2 dB in PSNR.

## Background & Motivation

In low-light environments, frame cameras require prolonged exposure times to gather sufficient light, which introduces two co-occurring degradations: **low visibility** and **motion blur**. These two issues are typically processed separately in existing studies, with dedicated methods and datasets for low-light enhancement and motion deblurring, respectively. However, cascading these two tasks often leads to suboptimal results.

**Key Challenge**: Solving the joint problem of low light and blur simultaneously is highly challenging:
1. Relying solely on frame information, there is almost no usable motion and structural detail in the frames when motion blur is severe or illumination is extremely low;
2. Existing joint low-light and deblurring methods (e.g., LEDNet) rely on synthetic data and fail to generalize to real-world scenarios;
3. There is a lack of synchronized datasets containing real-world low-light blurry images, normal-light sharp images, and event streams.

**Advantages of Event Cameras** provide the possibility to jointly solve these two problems: their **high dynamic range** (HDR) property allows them to capture scene details even under low light, and their **high temporal resolution** enables them to accurately record motion information during long exposures.

However, the **limitations of the event camera itself** cannot be ignored: under extremely low illumination, events also generate substantial noise. Therefore, a cross-modal fusion method is required to effectively leverage the advantages of events while suppressing noise, which is the exact design motivation of the SFCM-FE module.

## Method

### Overall Architecture

The overall pipeline is as follows: given three consecutive low-light blurry video frames $\{B^{t-1}, B^t, B^{t+1}\}$ and their corresponding event voxel grids $\{E^{t-1}, E^t, E^{t+1}\}$ as inputs, frame features and event features are extracted via convolutional layers, respectively, and then processed as follows:
1. **ED-TFA Module**: Performs event-guided multi-scale deformable temporal alignment to generate aligned features;
2. **SFCM-FE Module**: Utilizes a low-pass filter in the frequency domain to enhance the primary structural information of cross-modal features while suppressing noise;
3. **UNet Decoder**: Generates the final normal-light sharp output from the enhanced multi-scale feature pyramid.

### Key Designs

1. **RELED Dataset and Tri-axis Beam-Splitter Camera System**:

    - **Function**: Constructs the first synchronized tri-modal dataset containing real-world low-light blurry images, normal-light sharp images, and event streams.
    - **Mechanism**: Uses two beam splitters to split the incident light into three paths: the first camera captures normal-light sharp images with a short exposure (2ms); the second camera simulates real low-light blur with a 1/32 ND filter and long exposure (16ms); the third path connects to an event camera. Hardware-level synchronization is achieved via a microcontroller, and minor offsets among the multiple cameras are calibrated using homography matrices.
    - **Design Motivation**: Existing datasets either use gamma correction/ZeroDCE to synthesize low-light (showing a significant gap from real-world scenes) or use synthetic blur (such as frame averaging in GoPro), lacking synchronized real-world event streams. The RELED dataset has a resolution of 1024×768, covers 42 urban scenes, and provides far superior quality compared to previous synthetic datasets and low-resolution DAVIS data.

2. **Event-Guided Deformable Temporal Feature Alignment (ED-TFA)**:

    - **Function**: Leverages event information to guide temporal alignment across multiple frames, extracting useful motion information from neighboring frames.
    - **Mechanism**: First, a transposed attention Transformer encoder extracts the frame feature pyramid $\{\mathcal{F}(B^k)_s\}$ and the event feature pyramid $\{\mathcal{F}(E^k)_s\}$ respectively. Then, at multiple scales, the frame and event features are concatenated to generate template features, which are aligned using deformable convolution (DCN):
    $\mathcal{F(T)}_s^{t+1 \to t} = \mathcal{D}(\mathcal{F(T)}_s^{t+1 \to t}, \mathcal{O}_s^{t+1 \to t}, \mathcal{M}_s^{t+1 \to t})$
      A coarse-to-fine strategy is adopted: coarse offsets are estimated at low resolutions and progressively upsampled and passed to higher resolutions for fine-grained adjustment. Alignment is performed in both forward and backward directions before merging.
    - **Design Motivation**: Under low-light blurry conditions, finding inter-frame correspondences using frame information alone is highly ill-posed, as frames are filled with noise and lack clear structures. The high dynamic range and high temporal resolution of event cameras compensate for this limitation. The multi-scale coarse-to-fine design ensures robustness—since sub-pixel offsets are small at lower resolutions and easy to align, and higher resolutions can be refined incrementally based on previous offset estimates.

3. **Spectrum Frequency-based Cross-Modal Feature Enhancement (SFCM-FE)**:

    - **Function**: Enhances the structural information of cross-modal features through frequency-domain low-pass filtering while suppressing low-light noise.
    - **Mechanism**: The aligned frame features, event features, and the output from the previous scale are concatenated and split into two branches:
        - **(a) Low-pass filtering branch**: Applies FFT to transform features into the frequency domain, applies a Gaussian low-pass filter $\mathcal{P}(x,y,\sigma) = \exp\left(-\frac{(x-x_c)^2 + (y-y_c)^2}{2\sigma^2}\right)$ to extract low-frequency structural information, further performs frequency selection via an FFC (Fast Fourier Convolution) block, and finally transforms back to the spatial domain via IFFT. The low-frequency filtered features then pass through a **pixel-wise spatial dynamic filter** to enhance spatially-varying major structures.
        - **(b) Identity branch**: Keeps original features without frequency-domain filtering.
        - Finally, the two branches are fused dynamically using **spatial attention**: $\mathcal{G}(X)^{(c)} = \mathcal{G}(\bar{X})_L^{(a)} \odot \sigma(\text{Conv}(\cdot)) + \mathcal{G}(\tilde{X})^{(b)} \odot \sigma(\text{Conv}(\cdot))$.
    - **Design Motivation**: In low-light scenes, both frames and events suffer from severe noise, causing direct cross-modal feature fusion to perform poorly (in ablation studies, EFNet-style fusion even dropped the performance by 0.23 dB). Noise is mainly concentrated in high frequencies, whereas primary scene structures reside in low frequencies. Low-pass filtering naturally suppresses high-frequency noise and preserves structural information. This is supplemented by pixel-wise dynamic filters to adapt to spatially-varying structural differences, and a residual connection is used to retain raw details.

### Loss & Training

- Employs supervised training with multi-scale outputs $\{S_s^t\}, s \in \{0, 1, 2\}$.
- All methods are trained for 200 epochs on the RELED dataset.
- Offers a lightweight version Ours-s (5.3MB) and a standard version Ours (12.8MB).

## Key Experimental Results

### Main Results (RELED Dataset)

| Method Category | Method | PSNR | SSIM | Params (MB) |
|---------|------|------|------|------------|
| Frame - Low-Light | LLFormer | 26.62 | 0.862 | 13.15 |
| Frame - Deblurring | DSTNet | 29.59 | 0.903 | 7.53 |
| Frame - Joint | LEDNet | 26.47 | 0.856 | 7.41 |
| Event - Deblurring | REFID | 30.10 | 0.913 | 15.9 |
| Event - Deblurring | UEVD | 29.93 | 0.905 | 27.88 |
| **Ours-s** | **ELEDNet-s** | **30.98** | **0.919** | **5.3** |
| **Ours** | **ELEDNet** | **31.30** | **0.925** | **12.8** |

Key comparisons:
- Outperforms the best event-guided method REFID by **+1.20 dB PSNR**
- Outperforms the only joint method LEDNet by **+4.83 dB PSNR**
- The lightweight version Ours-s with only 5.3MB of parameters surpasses all other methods

### Ablation Study

| Configuration | PSNR | Contribution |
|------|------|------|
| Baseline (w/o ED-TFA, w/o SFCM-FE) | 29.59 | - |
| + ED-TFA | 30.78 | +1.19 dB |
| + SFCM-FE | 30.40 | +0.81 dB |
| + ED-TFA + SFCM-FE (Full) | **31.30** | +1.71 dB |

Ablation inside SFCM-FE module:

| Components | PSNR | Description |
|------|------|------|
| CABs only | 30.81 | +0.03 dB, barely effective |
| CABs + SA | 30.79 | +0.01 dB, naive stacking was ineffective |
| SA + LPF branch | 31.22 | +0.44 dB, low-pass filtering is the core component |
| CABs + SA + LPF | **31.30** | +0.52 dB |

Comparison of cross-modal fusion methods:

| Fusion Method | PSNR | Change |
|----------|------|------|
| No fusion | 30.78 | - |
| EFNet fusion | 30.55 | -0.23 dB (performance degraded instead) |
| REFID fusion | 30.86 | +0.08 dB |
| SFCM-FE (Ours) | **31.30** | +0.52 dB |

### Key Findings

- Event-guided methods generally outperform frame-only methods by a large margin, validating the core advantages of event cameras under low-light degradation.
- The cross-modal fusion of EFNet degrades performance in low-light scenarios, indicating that naive fusion cannot handle scenarios where both modalities are highly noisy.
- The low-pass filtering branch is the primary source of performance gain in SFCM-FE (+0.44 dB), verifying the effectiveness of the frequency-domain noise suppression strategy.
- The contributions of ED-TFA and SFCM-FE to the final performance are complementary.

## Highlights & Insights

- **Pioneering Task Definition**: Unifies event-guided low-light enhancement and deblurring into a joint task for the first time, filling a research gap.
- **Real-World Dataset**: RELED utilizes a beam splitter to capture truly synchronized tri-modal data, with quality and scale far exceeding synthetic approaches.
- **Elegant Design of Frequency-Domain Noise Suppression**: Since both frames and events suffer from severe noise under low-light conditions, extracting structural information via low-pass filtering in SFCM-FE is a highly targeted solution.
- **Strong Performance in Lightweight Version**: Ours-s with only 5.3MB surpasses all baseline methods, demonstrating the intrinsic effectiveness of the network design rather than merely stacking parameters.

## Limitations & Future Work

- The RELED dataset is limited in scale (42 scenes, 29 for training and 13 for testing), and its generalization capability requires further verification.
- The beam splitter system is costly and bulky, which restricts the convenience of data collection.
- Only 3 consecutive frames are processed as input; temporal modeling of longer sequences could potentially improve quality.
- The standard deviation $\sigma$ of the Gaussian low-pass filter is fixed; adaptive frequency selection might yield better results.
- The impact of different event representations (such as event frames or time surfaces, in addition to voxel grids) remains unexplored.

## Related Work & Insights

- Distinct from event-guided deblurring methods like EFNet and REFID, this work addresses the more complex joint low-light enhancement and deblurring problem.
- The frequency-domain filtering concept in SFCM-FE can be generalized to other cross-modal fusion scenarios, especially when multiple modalities contain severe noise.
- The design concept of the beam splitter data acquisition system provides a valuable reference for constructing other multimodal paired datasets.
- The event-guided coarse-to-fine deformable alignment strategy also provides insights for general video restoration tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ Defines a joint task for the first time; the real-world dataset and frequency-domain cross-modal fusion are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive and in-depth ablation studies; however, the dataset scale and scene diversity are limited.
- Writing Quality: ⭐⭐⭐⭐ Understood and clearly defined problem, with a complete and logical description of the method.
- Value: ⭐⭐⭐⭐ Pioneering task and real-world dataset, providing a significant boost to the event vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Event-Illumination Collaborative Low-light Image Enhancement with a High-resolution Real-world Dataset](../../CVPR2026/image_restoration/event-illumination_collaborative_low-light_image_enhancement_with_a_high-resolut.md)
- [\[ECCV 2024\] Unrolled Decomposed Unpaired Learning for Controllable Low-Light Video Enhancement](unrolled_decomposed_unpaired_learning_for_controllable_low-light_video_enhanceme.md)
- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](../../ICCV2025/image_restoration/low-light_image_enhancement_using_event-based_illumination_estimation.md)
- [\[ECCV 2024\] Domain-Adaptive Video Deblurring via Test-Time Blurring](domain-adaptive_video_deblurring_via_test-time_blurring.md)
- [\[CVPR 2026\] TextOVSR: Text-Guided Real-World Opera Video Super-Resolution](../../CVPR2026/image_restoration/textovsr_text-guided_real-world_opera_video_super-resolution.md)

</div>

<!-- RELATED:END -->
