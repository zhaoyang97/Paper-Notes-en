---
title: >-
  [Paper Note] Coded-E2LF: Coded Aperture Light Field Imaging from Events
description: >-
  [CVPR2026][light field imaging] This paper provides the first demonstration that an event camera alone (without conventional intensity images) can reconstruct a 4D light field at pixel-level accuracy. The proposed Coded-…
tags:
  - "CVPR2026"
  - "light field imaging"
  - "event camera"
  - "coded aperture"
  - "deep optics"
  - "end-to-end optimization"
  - "black-first coding sequence"
date: 2026-05-08
content_hash: 3c3681b68b4527bc
---

# Coded-E2LF: Coded Aperture Light Field Imaging from Events

**Conference**: CVPR2026
**arXiv**: [2602.22620](https://arxiv.org/abs/2602.22620)
**Code**: To be confirmed
**Area**: Others (Computational Photography / Event Camera)
**Keywords**: light field imaging, event camera, coded aperture, deep optics, end-to-end optimization, black-first coding sequence

## TL;DR

This paper provides the first demonstration that an event camera alone (without conventional intensity images) can reconstruct a 4D light field at pixel-level accuracy. The proposed Coded-E2LF system triggers events via a coded aperture pattern sequence and accumulates them into event images. By introducing an all-black pattern, a mathematical equivalence between event-based and intensity-based coded aperture imaging is established. Combined with end-to-end deep optics training, the system achieves 8×8 sub-aperture light field reconstruction.

## Background & Motivation

**Value and limitations of light field imaging**: 4D light fields capture both spatial and angular information of scene radiance, enabling applications such as digital refocusing, depth estimation, and view synthesis. Conventional light field cameras (e.g., Lytro) employ microlens arrays, which introduce an inherent resolution trade-off between spatial and angular dimensions.

**Advances in coded aperture approaches**: Coded aperture methods encode angular information into a single 2D image by placing known binary patterns on the lens aperture, followed by computational reconstruction. This avoids the spatial resolution penalty of microlens arrays, but reconstruction quality depends critically on the coding design and decoding algorithm.

**Limitations of conventional coded aperture**: Intensity camera-based coded aperture imaging requires multiple exposures (one per pattern), constrained by camera readout speed and scene dynamics—object motion between exposures introduces reconstruction artifacts.

**Unique advantages of event cameras**: Event cameras asynchronously detect per-pixel brightness changes with microsecond temporal resolution, high dynamic range (120+ dB), and low power consumption. When a coded aperture pattern is switched, the pattern change itself triggers events even in a completely static scene.

**An unexplored combination**: The combination of event cameras and coded apertures has no prior precedent. Event cameras are naturally suited to detecting brightness changes caused by pattern switching, theoretically enabling multi-pattern acquisition at extremely high speed. However, the nonlinear logarithmic response of event data prevents direct application of conventional coded aperture theory.

## Core Problem

How to exploit the high temporal resolution of event cameras to reconstruct a complete 4D light field from event data alone via a coded aperture pattern sequence, resolve the nonlinearity in event-to-intensity conversion, and realize a practically deployable hardware system.

## Method

### System Overview

The Coded-E2LF system comprises three components: **(1)** hardware layer—programmable aperture + event camera; **(2)** coding theory—black-pattern equivalence theorem + black-first (BF) coding sequence; **(3)** network layer—AcqNet (learning coding patterns) + RecNet (light field reconstruction), trained end-to-end.

### Coded Aperture + Event Camera Imaging Model

- **Coding process**: $N$ coding patterns $\{a^{(n)}\}_{n=1}^{N}$ are applied to the aperture sequentially, where $a^{(n)} \in \{0, 1\}^{u \times v}$ ($u \times v$ denotes angular resolution, e.g., $8 \times 8$) controls the corresponding sub-aperture elements.
- **Static scene assumption**: The scene remains static during the pattern sequence (approximately 20 ms); pattern switching is the sole source of brightness changes that trigger events.
- **Event accumulation**: Events triggered by switching from pattern $a^{(n-1)}$ to $a^{(n)}$ can be accumulated into an event image:
  $$E^{(n-1,n)}(x) = \log I^{(n)}(x) - \log I^{(n-1)}(x)$$
  where $I^{(n)}(x) = \sum_{s,t} a^{(n)}(s,t) \cdot L(x, s, t)$ is the intensity image under pattern $a^{(n)}$, and $L(x,s,t)$ is the light field to be reconstructed.

### Key Theory: Role of the Black Pattern

- **Core theorem (Eq. 8)**: If the coding sequence contains an all-black pattern $a^{(n_B)} = \mathbf{0}$ (i.e., the aperture is fully closed), then:
  $$E^{(n_B, n)}(x) = \log I^{(n)}(x) - \log I^{(n_B)}(x) = \log I^{(n)}(x) + C$$
  Because $I^{(n_B)} = 0$ requires special treatment—in practice, event cameras have a dark-current baseline $I_{\text{dark}}$, rendering $\log I^{(n_B)}$ a global constant $C$.
- **Equivalence**: The above expression shows that an event image involving the black pattern differs from the corresponding intensity-based coded aperture measurement by only a global constant, making conventional coded aperture decoding methods directly applicable.
- **Approximate permutation invariance**: With the black pattern included, event images generated under different pattern orderings are approximately equivalent (since the black pattern provides a unified reference baseline), simplifying coding design.

### Black-First Coding Sequence (BF)

- **Design**: The black pattern is fixed as the first element of the sequence ($a^{(1)} = \mathbf{0}$), followed by $N-1$ coded patterns applied successively.
- **Advantages**:
  - Event images $\{E^{(1,n)}\}_{n=2}^{N}$ from the initial black pattern to each subsequent pattern directly correspond to intensity-based measurements.
  - Substantially reduces event count—compared to arbitrary sequences, BF avoids redundant events between adjacent non-zero patterns.
  - $N-1$ event images suffice to reconstruct the complete $u \times v$ view light field.
- **Measured efficiency**: The entire coding sequence can be acquired in approximately 20 ms.

### Reference-Aware Event Generation (RA)

- **Motivation**: The logarithmic response and threshold mechanism of event cameras introduce errors in naive event accumulation.
- **Method**: The reference intensity $I_{\text{ref}}$ is explicitly tracked to accurately simulate the event generation process:
  $$e_k = \begin{cases} +1 & \text{if } \log I(x_k, t_k) - \log I_{\text{ref}}(x_k) \geq C_{\text{pos}} \\ -1 & \text{if } \log I(x_k, t_k) - \log I_{\text{ref}}(x_k) \leq -C_{\text{neg}} \end{cases}$$
  Each time an event fires, $I_{\text{ref}}$ is updated accordingly.
- **During training**: RA serves as a differentiable event generation simulator, enabling accurate gradient backpropagation through the coding pattern optimization.

### End-to-End Deep Optics Training

- **AcqNet (learning coding patterns)**: Takes randomly initialized continuous patterns $\tilde{a}^{(n)} \in [0,1]^{u \times v}$ as input; after training convergence, patterns are binarized to $a^{(n)} \in \{0,1\}^{u \times v}$.
- **RecNet (light field reconstruction)**: Receives $N-1$ event images and outputs the complete light field $\hat{L} \in \mathbb{R}^{H \times W \times u \times v}$.
  - Architecture: CNN-based encoder-decoder that processes spatial and angular dimensions separately before fusion.
- **Loss function**: $\mathcal{L} = \mathcal{L}_{\text{recon}}(\hat{L}, L_{\text{GT}}) + \gamma \cdot \mathcal{L}_{\text{binary}}$
  - $\mathcal{L}_{\text{recon}}$: L1 + SSIM loss for light field reconstruction.
  - $\mathcal{L}_{\text{binary}}$: Regularization encouraging patterns toward binary values.
- **Training pipeline**: Forward pass—AcqNet generates patterns → RA simulates events → RecNet reconstructs light field; Backward pass—gradients propagate through the entire pipeline for joint optimization of coding and decoding.

## Experiments

### Experimental Setup

- **Synthetic data**: Based on the HCI light field dataset and custom synthetic scenes; $8 \times 8$ views, spatial resolution $512 \times 512$.
- **Real hardware**: Prophesee EVK4 event camera (resolution $1280 \times 720$) + programmable LCD aperture (covering the lens aperture plane).
- **Evaluation metrics**: PSNR, SSIM, LPIPS.

### Synthetic Data Results

| Method | #Patterns | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|-----------|--------|--------|---------|
| Intensity-based coded aperture | 9 | 34.2 | 0.952 | 0.041 |
| Naive event accumulation | 9 | 28.7 | 0.891 | 0.098 |
| Coded-E2LF (random patterns) | 9 | 33.5 | 0.945 | 0.048 |
| **Coded-E2LF (learned, BF)** | **9** | **35.1** | **0.961** | **0.035** |

- The learned BF coding sequence outperforms conventional intensity-based methods, validating the effectiveness of end-to-end optimization.
- Naive event accumulation (without black pattern or RA) degrades substantially, confirming the necessity of the theoretical analysis.

### Real Hardware Validation

- A physical prototype is assembled using Prophesee EVK4 + LCD aperture, with 9 patterns (including 1 black pattern) and a total acquisition time of approximately 20 ms.
- Successful reconstruction of $8 \times 8$ real-scene light fields is demonstrated, enabling digital refocusing and view switching.
- Compared to intensity-based methods, the event-based approach performs superiorly in high dynamic range scenes (coexistence of bright highlights and dark regions).

### Ablation Study

| Configuration | PSNR |
|---------------|------|
| No black pattern ($N$ arbitrary non-zero patterns) | 29.4 |
| Black pattern at random position | 33.8 |
| Black pattern + BF (fixed as first) | **35.1** |
| BF without RA | 33.2 |
| BF + RA (full) | **35.1** |

- The black pattern is the key driver of performance improvement (+4.4 dB).
- The BF sequence further improves over randomly placing the black pattern by 1.3 dB.
- The RA module contributes 1.9 dB; accurate event generation modeling is indispensable.

## Highlights & Insights

- **Pioneering contribution**: This work provides the first demonstration that an event camera can independently reconstruct a 4D light field without any conventional intensity image assistance.
- **Black pattern equivalence theorem**: Elegantly resolves the core challenge of logarithmic nonlinearity in event data—by introducing an all-black reference pattern, event-based imaging is transformed into an equivalent intensity-based problem.
- **BF coding sequence design**: The simple "black-pattern-first" strategy simultaneously reduces event count and improves reconstruction quality, offering high practical value.
- **End-to-end deep optics**: Joint optimization of AcqNet and RecNet for coding and decoding surpasses the performance ceiling of manually designed codes.
- **Real hardware validation**: Beyond theoretical contributions, physical experiments on the Prophesee EVK4 demonstrate engineering feasibility.
- **Extremely fast acquisition**: The complete pattern sequence is acquired in 20 ms, one to two orders of magnitude faster than conventional multi-exposure approaches.

## Limitations & Future Work

- The static scene assumption restricts applicability—scene motion within 20 ms still introduces artifacts; dynamic scenes require additional motion compensation.
- The switching speed of the LCD aperture (approximately 2 ms per pattern) is the acquisition bottleneck; replacing it with a DMD (microsecond-level switching) could further accelerate the system.
- The current $8 \times 8$ angular resolution requires 9 pattern switches; higher angular resolutions will linearly increase acquisition time.
- Dark current and noise of event cameras may degrade event image quality in low-light conditions.
- The scalability of the CNN-based RecNet to very high spatial resolutions (e.g., 4K) has yet to be verified.
- Validation is limited to static indoor scenes; outdoor, long-range, and large-baseline scenarios remain unexplored.

## Related Work & Insights

- **Conventional light field cameras**: Lytro, RayTrix (microlens arrays)—suffer from severe spatial/angular resolution trade-offs.
- **Coded aperture light field**: Veeraraghavan et al. (2007), Marwah et al. (2013)—intensity camera-based coded aperture with multiple exposures, constrained by dynamic scenes.
- **Deep optics**: Sitzmann et al. (2018), Chang & Wetzstein (2019)—end-to-end optimization of optical coding and computational decoding, but all based on intensity cameras.
- **Event camera 3D reconstruction**: E2VID, ESIM, EventNeRF—events used for depth estimation or NeRF, without complete light field reconstruction.
- **Event-based HDR**: Han et al. (2020), Rebecq et al. (2019)—exploit the high dynamic range advantage of event cameras; complementary to this work.
- **Compressive light field**: Kamal et al. (2016)—compressive sensing framework for light field reconstruction; the proposed end-to-end method achieves superior performance.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to introduce event cameras into coded aperture light field imaging; the black pattern equivalence theorem is theoretically original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive synthetic + real hardware validation + ablation study, though diversity of real-world scenes is limited.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear; the logic from physical modeling to system design is coherent.
- Value: ⭐⭐⭐⭐⭐ — Opens a new direction in event-based computational light field imaging, with both theoretical contributions and practical engineering validation.
- Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Cameras](../../AAAI2026/others/spike_imaging_velocimetry_dense_motion_estimation_of_fluids_using_spike_cameras.md)
- [\[CVPR 2026\] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](rooftop_wind_field_reconstruction_using_sparse_sen.md)
- [\[CVPR 2026\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_widefield_and_highdynamic_range.md)
- [\[ICLR 2026\] Neural Force Field: Few-shot Learning of Generalized Physical Reasoning](../../ICLR2026/others/neural_force_field_few-shot_learning_of_generalized_physical_reasoning.md)
- [\[AAAI 2026\] Towards Temporal Fusion Beyond the Field of View for Camera-based Semantic Scene Completion](../../AAAI2026/others/towards_temporal_fusion_beyond_the_field_of_view_for_camera-based_semantic_scene.md)

</div>

<!-- RELATED:END -->
