---
title: >-
  [Paper Note] Event Fields: Capturing Light Fields at High Speed, Resolution, and Dynamic Range
description: >-
  [CVPR 2025][3D Vision][Event Camera] This paper proposes Event Fields—a new paradigm for capturing high-speed, high-resolution, high-dynamic-range light fields using event cameras. It designs two complementary optical schemes: a kaleidoscope (spatial multiplexing, capturing temporal derivatives) and a galvanometer (temporal multiplexing, capturing angular derivatives), achieving unprecedented capabilities such as 250fps megapixel dynamic scene refocusing and 100Hz real-time d…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Event Camera"
  - "Light Field"
  - "High-Speed Imaging"
  - "Spatial Multiplexing"
  - "Temporal Multiplexing"
date: 2026-05-08
content_hash: b85395cbeece6253
---

# Event Fields: Capturing Light Fields at High Speed, Resolution, and Dynamic Range

**Conference**: CVPR 2025  
**arXiv**: [2412.06191](https://arxiv.org/abs/2412.06191)  
**Code**: Yes (open-source simulator, data, and processing code)  
**Area**: 3D Vision  
**Keywords**: Event Camera, Light Field, High-Speed Imaging, Spatial Multiplexing, Temporal Multiplexing

## TL;DR
This paper proposes Event Fields—a new paradigm for capturing high-speed, high-resolution, high-dynamic-range light fields using event cameras. It designs two complementary optical schemes: a kaleidoscope (spatial multiplexing, capturing temporal derivatives) and a galvanometer (temporal multiplexing, capturing angular derivatives), achieving unprecedented capabilities such as 250fps megapixel dynamic scene refocusing and 100Hz real-time depth estimation.

## Background & Motivation

1. **Background**: Event cameras are widely used in high-speed applications, including optical flow, object tracking, and HDR imaging, due to their high temporal resolution (microsecond level), high dynamic range (140dB), low bandwidth, and asynchronous operating mode. On the other hand, light field imaging enables post-processing refocusing and depth estimation by capturing angular radiance information, but traditional light field cameras are limited by frame rates and dynamic range.

2. **Limitations of Prior Work**: Event cameras only capture intensity changes in the temporal dimension, losing angular information. Traditional light field cameras (Lytro, Raytrix) have low frame rates and limited dynamic range, rendering them unable to handle high-speed scenes. Existing attempts combining event cameras and light fields are either limited to static scenes (LCOS scheme, 22ms response time) or sacrifice spatial resolution (microlens array scheme).

3. **Key Challenge**: Event cameras excel at high-speed temporal perception but lack the angular dimension, while light fields excel at angular perception but lack high-speed capability. The challenge lies in how to capture both temporal and angular information simultaneously on a single event camera.

4. **Goal**: To encode angular information into the existing spatial or temporal dimensions of an event camera, thereby capturing a complete light field—dubbed "Event Field"—using a single event camera.

5. **Key Insight**: The light field is typically smooth (redundant) in the angular dimension, rendering the angular derivative $\partial L / \partial \omega$ sparse—this perfectly aligns with the natural characteristic of event cameras to capture sparse changes.

6. **Core Idea**: By using optical designs to multiplex angular information into either the spatial dimension (kaleidoscope) or temporal dimension (galvanometer), the event camera is extended from capturing "temporal derivatives" to capturing "spatiotemporal-angular derivatives," enabling high-speed light fields.

## Method

### Overall Architecture
The mathematical framework of Event Fields is established on the derivatives of the plenoptic function $L(\vec{x}, \vec{\omega}, \lambda, t)$. Traditional event cameras only measure the temporal derivative $\partial B / \partial t$. In this work, the angular dimension is encoded using two multiplexing strategies: spatial multiplexing to measure the temporal derivatives of multiple angles $\partial B(\mathbf{x_s}, \omega, t_k) / \partial t$, and temporal multiplexing to directly measure the angular derivative $\partial B(\mathbf{x}, \omega) / \partial \omega$. Built upon this, high-speed light field videos are reconstructed using existing event-processing algorithms (E2VID, TimeLens), enabling subsequent applications such as refocusing and depth estimation.

### Key Designs

1. **Kaleidoscope for Spatial Multiplexing**:

    - **Function**: Encodes multi-angle light field information into the spatial dimension, capturing temporal derivatives of each angle.
    - **Mechanism**: A rectangular kaleidoscope (four-sided mirror) is placed behind the main lens. The main lens forms an image at the entrance of the kaleidoscope, which undergoes multiple reflections to create $3\times3$ flipped images on the event camera sensor, each corresponding to a different virtual camera position. The mapping function is $\mathbf{x_s} = \mathbf{x} \bmod (\mathbf{r/n})$, where $\mathbf{n}$ is the number of angular views and $\mathbf{r}$ is the sensor resolution. The event camera measures $\partial B / \partial t (\mathbf{x_s}, \omega, t_k) \approx p_k C / \Delta t_k$. Complementing this setup with a beamsplitter and an RGB camera enables color high-speed light fields (sensor fusion).
    - **Design Motivation**: The kaleidoscope has no moving parts, offering a simple and reliable design. It trades spatial resolution for angular resolution (reduced by a factor of $n$), but is naturally suited for dynamic scenes (unaffected by motion blur, since each angle accumulates events independently). The disadvantage is the inability to capture static scenes (no events are triggered when there is no temporal change).

2. **Galvanometer for Temporal Multiplexing**:

    - **Function**: Encodes angular information into the temporal dimension, directly capturing angular derivatives.
    - **Mechanism**: A dual-axis galvanometer is placed in the optical path of the event camera, scanning at a high speed of 250Hz along a periodic curve $\mathcal{C}(t)$ (such as a circular Lissajous curve) to virtually rotate the camera. For static scenes, the event camera measures the angular derivative $\partial B / \partial \omega \cdot \partial \mathcal{C}(t) / \partial t |_{t=t_k} \approx p_k C / \Delta t_k$. This implies that events directly encode the angular gradient information of the light field. Reconstructing video at 10000fps using E2VID yields a 250fps $\times$ 40-view light field under 250Hz scanning.
    - **Design Motivation**: It does not sacrifice spatial resolution, allows capturing static scenes (as angular variations themselves trigger events), and directly obtaining angular derivatives provides a more efficient representation of light fields. The cost is that moving objects suffer from motion blur (object motion superimposing with the galvanometer motion), and the bandwidth requirement is higher.

3. **Physical Simulator (Blender Plugin)**:

    - **Function**: Provides a controllable, fair comparison and systematic evaluation of the two schemes.
    - **Mechanism**: A plugin for rendering Event Fields is developed based on Blender. It simulates both spatial and temporal multiplexing designs through a pipeline of rendering the plenoptic function $\rightarrow$ optical propagation calculation for intensity $\rightarrow$ pixel-wise tracking of intensity changes $\rightarrow$ threshold-triggered events. GPU acceleration and GUI interaction are supported.
    - **Design Motivation**: In real systems, the optical parameters (focal length, etc.) of the two schemes differ, making it difficult to conduct a fair comparison under identical conditions. The simulator allows precise control over scene and design parameters to systematically evaluate the strengths and weaknesses of both schemes.

### Loss & Training
This work presents a computational imaging system and does not involve training neural networks. The core algorithms include: (1) reconstructing high-speed videos from event streams using pre-trained E2VID; (2) performing event-RGB fused frame interpolation using TimeLens; (3) post-processing refocusing based on light field theory (light field shearing and integration); (4) depth estimation based on depth-from-focus.

## Key Experimental Results

### Main Results

| Property | Kaleidoscope (Spatial Multiplexing) | Galvanometer (Temporal Multiplexing) |
|------|------------------|----------------|
| Captured Derivative Type | Temporal derivative | Angular + temporal derivative |
| Static Scene Capture | ✗ | ✓ |
| Dynamic Scene Refocusing | ✓ (No motion blur) | ✓ (Affected by motion speed) |
| Color Support | ✓ (Beamsplitter + RGB) | ✗ |
| Spatial Resolution | Reduced by $n \times$ | Full resolution |
| Bandwidth Requirement | Low | High |
| Refocusing Quality | Discretized stepping effect | Smooth and continuous |

### Application Demos

| Application | Scheme | Key Performance |
|------|------|---------|
| SloMoRF Slow-Motion Refocusing | Kaleidoscope + RGB Fusion | 720fps color light field, three-focal-plane dynamic refocusing |
| High-Speed Refocusing | Galvanometer | 250fps $\times$ 40 views, megapixel resolution |
| Instant Real Depth Estimation | Galvanometer | 100Hz real-time depth measurement |
| HDR Light Field | Kaleidoscope | Captures 140dB dynamic range (vs. RGB 60dB) |

### Key Findings
- The kaleidoscope scheme is more robust to dynamic scenes (unaffected by motion blur) but cannot handle static scenes (the absence of events), and its spatial resolution is reduced by 9 times ($3\times3$ views).
- The galvanometer scheme can capture static scenes without sacrificing spatial resolution, but the refocusing quality degrades as the speed of moving objects increases (a clear blur appears at $8\times$ speed in the experiments).
- When the galvanometer scanning frequency exceeds 250Hz, the event camera reaches its readout bandwidth limit and begins to randomly drop events. Counterintuitively, however, characters on a rotating fan appear clearer (due to reduced motion blur) under high scanning frequencies.
- In HDR experiments, the event light field successfully captured both bright LEDs and dark-region textures simultaneously, while the RGB camera failed to accommodate both regardless of long or short exposures.

## Highlights & Insights
- **The "Event Fields" concept itself** is the most significant contribution—redefining the event camera from a "temporal derivative sensor" to a "spatiotemporal-angular derivative sensor," which pioneers a completely new computational imaging paradigm. The angular sparsity of light fields perfectly matches the sparse sensing nature of event cameras, which is an elegant observation.
- **The systematic comparison of two complementary schemes** is highly valuable—the kaleidoscope is suitable for high-speed dynamic scenes, while the galvanometer is suited for cases requiring high spatial resolution or static scenes, offering a clear selection guide for practical applications.
- **The SloMoRF (Slow-motion Refocusing)** demo is impressive—tracking a thrown wooden block splashing into water, while simultaneously achieving 720fps and dynamic refocusing, which is utterly impossible with traditional cameras.
- The open-sourcing of the simulator holds long-term value for the community, which can accelerate algorithm research in the Event Fields domain.

## Limitations & Future Work
- Currently, pre-trained E2VID and TimeLens are utilized. These models are optimized for ordinary event streams and may not be optimal for the specific data distribution of Event Fields. Training reconstruction models specifically tailored for Event Fields would significantly improve the quality.
- The motion blur issue in the galvanometer scheme can be mitigated by adaptive scanning (fast scanning in active motion areas, slow scanning in static areas). Using a MEMS array could enable spatially-varying scanning speeds.
- The experiments only demonstrate macroscopic scenes. Microscopic scenes (digital holography, light-field microscopy) represent potentially more valuable application directions (already preliminarily explored by EventLFM).
- Both schemes have their limitations. Hybrid designs could be considered to simultaneously employ spatial and temporal multiplexing for capturing more complete Event Fields.

## Related Work & Insights
- **vs. Habuchi et al. (LCOS + Events)**: They utilized an LCOS to encode angles at the aperture plane, but the 22ms response time of LCOS limited its use in dynamic scenes. The galvanometer scheme possesses a response speed several orders of magnitude faster.
- **vs. Guo et al. (Microlens Array + Events)**: The MLA scheme is functionally equivalent to the spatial multiplexing of the kaleidoscope, but the kaleidoscope is simpler to implement and more suitable for macroscopic scenes.
- **vs. He et al. (Rotating Wedge Prism + Events)**: They used rotating optics to achieve omnidirectional edge detection, sharing a similar concept to the galvanometer scheme, but Event Fields extends this for the first time to a complete light field capture framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "Event Fields" concept pioneeringly integrates event cameras with light fields, and the systematic design of two complementary schemes is highly comprehensive.
- Experimental Thoroughness: ⭐⭐⭐⭐ The simulation comparison is systematic and fair, and the hardware prototype demonstrations are rich and diverse, though quantitative comparisons are somewhat lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ The mathematical framework is derived clearly and elegantly, the figures are meticulously designed, and simulation and real-world captures complementarily validate each other.
- Value: ⭐⭐⭐⭐⭐ It opens up a new paradigm in computational imaging, with broad application prospects for photography, robotics, and AR/VR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HDR-NSFF: High Dynamic Range Neural Scene Flow Fields](../../ICLR2026/3d_vision/hdr-nsff_high_dynamic_range_neural_scene_flow_fields.md)
- [\[CVPR 2025\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [\[CVPR 2025\] RelationField: Relate Anything in Radiance Fields](relationfield_relate_anything_in_radiance_fields.md)
- [\[CVPR 2025\] Exploiting Deblurring Networks for Radiance Fields](exploiting_deblurring_networks_for_radiance_fields.md)
- [\[CVPR 2025\] Preconditioners for the Stochastic Training of Neural Fields](preconditioners_for_the_stochastic_training_of_neural_fields.md)

</div>

<!-- RELATED:END -->
