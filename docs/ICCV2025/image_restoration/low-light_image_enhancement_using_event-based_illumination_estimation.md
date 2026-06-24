---
title: >-
  [Paper Note] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)
description: >-
  [ICCV 2025][Image Restoration][Low-light enhancement] RetinEV proposes exploiting *temporal-mapping events* (triggered by transmittance modulation) rather than conventional motion events for illumination estimation. Combined with Retinex theory, it decomposes low-light images into illumination and reflectance components, and employs an Illumination-guided Reflectance Enhancement (IRE) module to achieve high-quality low-light image enhancement…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "Low-light enhancement"
  - "event camera"
  - "Retinex theory"
  - "temporal-mapping events"
  - "illumination estimation"
  - "reflectance enhancement"
date: 2026-05-08
content_hash: 13f6126a068362f7
---

# Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)

**Conference**: ICCV 2025
**arXiv**: [2504.09379](https://arxiv.org/abs/2504.09379)  
**Code**: Not released  
**Area**: Low-Light Image Enhancement / Event Camera
**Keywords**: Low-light enhancement, event camera, Retinex theory, temporal-mapping events, illumination estimation, reflectance enhancement

## TL;DR

RetinEV proposes exploiting *temporal-mapping events* (triggered by transmittance modulation) rather than conventional motion events for illumination estimation. Combined with Retinex theory, it decomposes low-light images into illumination and reflectance components, and employs an Illumination-guided Reflectance Enhancement (IRE) module to achieve high-quality low-light image enhancement, reaching real-time inference at 35.6 FPS on 640×480 images.

## Background & Motivation

**Background**: Low-light image enhancement (LLIE) is a fundamental task in computer vision. Traditional approaches fall into two categories: histogram equalization and Retinex-based methods. Deep learning methods (RetinexNet, SCI, Diff-Retinex, etc.) have achieved notable progress. Event cameras have attracted attention due to their high dynamic range and superior low-light response.

**Limitations of Prior Work**:
   - Existing event-guided LLIE methods (e.g., eSL-Net, EvLight) rely on **motion events** (triggered by object or camera motion), which suffer from three key problems:
     - (a) Low-light conditions require long exposure, during which motion events compound motion blur degradation.
     - (b) Motion events provide only edge information, failing to cover flat regions for illumination estimation.
     - (c) They are prone to generating artifacts.
   - Image-only methods lack effective illumination priors under extreme low-light conditions.

**Key Challenge**: Event cameras possess excellent HDR and low-light response capabilities, yet existing methods exploit only motion-triggered events, leaving the potential of event cameras along the illumination dimension largely untapped.

**Key Insight**: A paradigm shift—rather than relying on scene motion, the paper actively generates *temporal-mapping events* by varying the optical transmittance (opening a mechanical shutter). These events directly encode scene brightness changes, providing denser and more accurate illumination information than motion events. Converting event timestamps to brightness values yields fine-grained illumination estimation.

## Method

### Overall Architecture

RetinEV is built on Retinex theory: observed image = reflectance × illumination ($I = R \cdot S$). The pipeline consists of:

1. **Illumination Estimation**: Estimate the illumination component $S$ from temporal-mapping events.
2. **Image Decomposition**: A decomposition network $\mathcal{F}_{Decom}$ decomposes both low-light and normal-light images into reflectance $R$ and illumination $S$.
3. **Reflectance Enhancement**: The IRE module uses estimated illumination as a prior to enhance detail in the reflectance component.
4. **Brightness Control**: A coefficient $\beta$ enables arbitrary control of output brightness.

During training, normal-light images serve as supervision, and reflectance consistency constraints ensure that low-light and normal-light images share the same reflectance.

### Key Designs

1. **Temporal-Mapping Event-Driven Illumination Estimation**:

    - **Function**: Opening the mechanical shutter (taking ~2 ms) transitions the optical transmittance from 0 to 1, triggering dense event signals on the event camera.
    - **Mechanism**: For pixel $(x,y)$, the event timestamp $t(x,y)$ is inversely proportional to pixel brightness $I(x,y)$—brighter pixels trigger events earlier due to faster brightness change, while darker pixels trigger later. Converting event timestamps to brightness values thus yields precise illumination estimates.
    - **Design Motivation**: Motion events have limited information density (triggered only at motion edges), whereas temporal-mapping events provide dense illumination information across the **entire image**, independent of scene motion.

2. **Illumination-guided Reflectance Enhancement (IRE) Module**:

    - **Function**: Uses the estimated illumination component as a prior and enhances the reflectance component via cross-modal attention.
    - **Mechanism**: Conventional Retinex methods focus solely on enhancing the illumination component, but under low light the reflectance is also severely affected by noise and color bias. IRE leverages **high-quality illumination priors** to guide reflectance denoising and detail recovery.
    - **Design Motivation**: The illumination component encodes the scene's overall brightness structure, which helps identify noise patterns and textural details within the reflectance.

3. **Degradation Model for Temporal-Mapping Events under Low Light**:

    - **Function**: Models the effect of low-light conditions on event camera behavior, specifically changes in the event triggering threshold.
    - **Mechanism**: In low-light scenes, the SNR of event cameras decreases, and events in dark regions may be missing or noisy. The degradation model brings synthetic training data closer to real low-light conditions, improving generalization.

4. **Brightness Controllability**: A coefficient $\beta$ multiplied by the illumination component $S$ allows users to freely adjust the brightness of the output image.

### Hardware & Dataset

- **Beam-splitter System**: A coaxial optical system combining a DVS event camera and an RGB image sensor, ensuring precise alignment between the event stream and the low-light image.
- **EvLowLight Dataset**: 60 extremely low-illumination and high-contrast scenes, containing aligned images, temporal-mapping events, and motion events.
- The mechanical shutter opening process (~2 ms) simultaneously accomplishes image exposure and temporal-mapping event acquisition.

### Loss & Training

- Decomposition loss: enforces reflectance consistency between low-light and normal-light images.
- Reconstruction loss: constrains the quality of enhanced images.
- Event degradation augmentation: accounts for low-light event camera characteristics on synthetic data.
- Training data: 5 synthetic datasets + the EvLowLight real-world dataset.

## Key Experimental Results

### Main Results

Comprehensive evaluation on 5 synthetic datasets and the EvLowLight real-world dataset:

- **vs. motion-event methods**: RetinEV surpasses the previous state-of-the-art event-based method (EvLight) by up to **6.62 dB** PSNR.
- **vs. image-only methods**: Consistently outperforms traditional LLIE methods on both reference-based and no-reference objective metrics, as well as subjective user studies.
- **Inference speed**: Achieves real-time processing at **35.6 FPS** on 640×480 images.

### Core Advantages over Motion-Event Methods

| Dimension | Motion-Event Methods | RetinEV (Temporal-Mapping Events) |
|-----------|---------------------|-----------------------------------|
| Event source | Scene/camera motion | Transmittance modulation (shutter opening) |
| Information density | Edge regions only | Full image coverage |
| Illumination information | Indirect (requires inference) | Directly encodes brightness |
| Motion blur | Severe under low light | Independent of motion; not an issue |
| Artifacts | Prone to generation | Significantly reduced |

### Key Findings

- **Temporal-mapping events are a superior event signal for LLIE**: From an information-theoretic perspective, temporal-mapping events directly encode illumination information, making them more suitable for illumination estimation than the indirect edge information provided by motion events.
- **Effectiveness of the IRE module**: Jointly enhancing both reflectance and illumination outperforms traditional Retinex methods that enhance only the illumination component.
- **Importance of event degradation modeling**: Modeling the impact of low light on event cameras leads to significantly improved generalization from synthetic training data to real-world data.
- **Lightweight and efficient architecture**: Compared to generative model approaches (GAN/diffusion models), RetinEV is more lightweight, achieving real-time inference at 35.6 FPS.

## Highlights & Insights

- **Value of the paradigm shift**: The transition from "motion events" to "temporal-mapping events" is a remarkably clever paradigm shift. By actively generating events (modulating transmittance) rather than passively waiting for them (scene motion), the HDR and low-light capabilities of event cameras are transformed from "potentially useful" to "directly applicable for illumination estimation."
- **Deep exploitation of physical priors**: The conversion from event timestamps to brightness values has a clear physical basis (the correspondence between transmittance change and brightness), making this physics-prior-based design more interpretable and generalizable than purely data-driven approaches.
- **Elegant cross-modal fusion design**: The IRE module injects illumination priors into reflectance enhancement, representing an insightful cross-modal attention design. It recognizes that illumination and reflectance in Retinex decomposition are not independent—high-quality illumination estimates can in turn assist reflectance denoising.
- **Complete system solution**: From hardware (beam-splitter system) to dataset (EvLowLight) to algorithm (RetinEV), the paper provides a comprehensive end-to-end solution that facilitates follow-up research.

## Limitations & Future Work

- **Hardware dependency**: Requires a dedicated beam-splitter system integrating both DVS and RGB sensors, as well as a precisely controlled mechanical shutter, limiting deployment convenience.
- **Acquisition constraints for temporal-mapping events**: Events generated by shutter opening must be captured at the start of exposure, imposing special requirements on the capture workflow and making the approach unsuitable for continuous video enhancement.
- **Limited scale of EvLowLight dataset**: With only 60 scenes, the dataset may not cover a sufficiently diverse range of low-light degradation patterns.
- **Extreme low-light scenarios near the event noise floor**: Event cameras may fail to trigger events effectively in very dark regions, leading to incomplete illumination estimates.
- **Incomplete comparison with latest diffusion-based LLIE methods**: Methods such as DiffLL may offer advantages in perceptual quality, though their inference speed is far below RetinEV's real-time performance.

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Event-Illumination Collaborative Low-light Image Enhancement with a High-resolution Real-world Dataset](../../CVPR2026/image_restoration/event-illumination_collaborative_low-light_image_enhancement_with_a_high-resolut.md)
- [\[ECCV 2024\] Towards Real-world Event-guided Low-light Video Enhancement and Deblurring](../../ECCV2024/image_restoration/towards_real-world_event-guided_low-light_video_enhancement_and_deblurring.md)
- [\[ICCV 2025\] CWNet: Causal Wavelet Network for Low-Light Image Enhancement](cwnet_causal_wavelet_network_for_low-light_image_enhancement.md)
- [\[CVPR 2026\] BiEvLight: Bi-level Learning of Task-Aware Event Refinement for Low-Light Image Enhancement](../../CVPR2026/image_restoration/bievlight_bi-level_learning_of_task-aware_event_refinement_for_low-light_image_e.md)
- [\[CVPR 2025\] HVI: A New Color Space for Low-light Image Enhancement](../../CVPR2025/image_restoration/hvi_a_new_color_space_for_low-light_image_enhancement.md)

</div>

<!-- RELATED:END -->
