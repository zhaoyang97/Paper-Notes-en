---
title: >-
  [Paper Note] One-Step Event-Driven High-Speed Autofocus
description: >-
  [CVPR 2025][Image Restoration][Event Camera] An Event Laplacian Product (ELP) focus detection function is proposed, which combines event data and intensity Laplacian information to reformulate the focus search as a detection task, achieving event-driven one-step autofocus for the first time, reducing focus time by 2/3 and decreasing autofocus errors by 22-24 times.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Event Camera"
  - "Auto Focus"
  - "One-step Focusing"
  - "Laplacian Operator"
  - "High-speed Autofocus"
date: 2026-05-08
content_hash: 39303ea8e0b175ef
---

# One-Step Event-Driven High-Speed Autofocus

**Conference**: CVPR 2025  
**arXiv**: [2503.01214](https://arxiv.org/abs/2503.01214)  
**Code**: To be released  
**Area**: Image Restoration / Computational Photography  
**Keywords**: Event Camera, Auto Focus, One-step Focusing, Laplacian Operator, High-speed Autofocus

## TL;DR

An Event Laplacian Product (ELP) focus detection function is proposed, which combines event data and intensity Laplacian information to reformulate the focus search as a detection task, achieving event-driven one-step autofocus for the first time, reducing focus time by 2/3 and decreasing autofocus errors by 22-24 times.

## Background & Motivation

High-speed autofocus remains a major challenge in extreme scenarios (such as low-light and motion blur). Traditional contrast-based AF requires repetitive sampling around the focal point, leading to "focus hunting." Although PDAF achieves one-step focusing, it is limited by the complexity of dual-pixel designs and poor low-light performance.

Existing event-driven AF methods (e.g., EGS, PBF), while leveraging the microsecond-level temporal resolution and high dynamic range benefits of event cameras, still require capturing the full focal stack (from out-of-focus → in-focus → out-of-focus) to search for the focal position and then driving the motor back—essentially still requiring a complete "focus hunting" sweep.

Key Insight: If the focal position can be **detected in real-time** during the focusing process and the motor stopped immediately, capturing the full stack becomes unnecessary, enabling true "one-step autofocus."

## Method

### Overall Architecture

The ELP method is based on the intrinsic relationship between the spatial second-order derivative and the temporal first-order derivative of images during the focusing process. The system monitors the sign of the ELP value: positive values indicate approaching the focus, negative values indicate moving away from the focus, and an abrupt sign change indicates that the focus position has been reached. For event-only cameras, intensity images are first retrieved via EvTemMap, and the ELP is then computed in combination with the event stream.

### Key Designs

#### Key Design 1: Event Laplacian Product (ELP) Focus Detection Function

- **Function**: To detect the focal position in real-time and determine whether the lens is moving towards or away from the focus.
- **Mechanism**: The function is defined as $\text{ELP}(t) = -\sum(\nabla^2 I(t) \cdot E(t))$, where $\nabla^2 I(t)$ is the Laplacian of the intensity image and $E(t)$ is the event frame. Theoretical derivation demonstrates that $S(t) = -\int \frac{\partial G}{\partial t} \cdot \frac{\partial^2 G}{\partial x^2} dx = -\alpha[\int(F * \frac{\partial^2 h}{\partial x^2})^2 dx]$, whose sign is solely determined by $\alpha$ (the rate of change of the Gaussian blur variance).
- **Design Motivation**: Traditional "peak-type" focus evaluation functions require searching for the maximum value, whereas the "abrupt sign change" property of ELP makes it a detection function—the focal point can be determined simply by detecting a transition from positive to negative, without needing to traverse the entire stack.

#### Key Design 2: ELP Adaptive Filter

- **Function**: To suppress local fluctuations in ELP values while preserving sharp transitions at the focal point.
- **Mechanism**: Calculates the mean $\overline{ELP}$ of the past $W$ ELP values; if the deviation of the current value is less than the threshold $ELP_{\text{thd}}$, rolling average smoothing (with factor $S$) is applied; otherwise, the original value is preserved. This conditional judgment ensures that fluctuations are smoothed when far from focus, while preserving the sharpness of the sudden change near the focus.
- **Design Motivation**: A short time interval $\Delta t$ (<1ms) improves sensitivity but introduces noise fluctuations. Simple low-pass filtering would blur the sign transition point. The adaptive strategy balances noise resistance with sensitivity.

#### Key Design 3: Event-Only One-Step AF Pipeline

- **Function**: To enable complete one-step AF for event-only cameras (e.g., Prophesee EVK4).
- **Mechanism**: A two-phase pipeline—(1) Aperture opening phase: Uses EvTemMap to map event timestamps into an HDR intensity image (20ms); (2) Focusing phase: The user selects an ROI, and the ELP value is continuously computed during motor movement, stopping immediately upon detecting a sign transition (~100ms).
- **Design Motivation**: Event-only cameras cannot directly acquire intensity images. EvTemMap provides a method to obtain intensity information via active transmittance modulation, allowing ELP to scale to event-only cameras.

### Loss & Training

ELP is a physics-based derivation of a detection function and does not require training.

## Key Experimental Results

### Synthetic Dataset MAE Comparison (μm)

| Method | Static | Medium Motion | Violent Motion |
|------|------|---------|---------|
| EGS | 33.31 | 29.33 | 21.78 |
| PBF | 4.93 | 3.99 | 2.69 |
| ELP (1FPS) | 2.00 | 3.66 | 3.66 |
| ELP (20FPS) | 2.00 | 2.40 | 2.97 |
| **ELP (50FPS)** | **2.00** | **1.51** | **2.26** |

### Real-world Dataset Focusing Error Comparison

| Dataset | EGS Error | PBF Error | ELP Error | ELP Reduction Factor |
|--------|---------|---------|---------|-----------|
| DAVIS346 | Large | Medium | Smallest | 24× |
| EVK4 | Large | Medium | Smallest | 22× |

### Key Findings

- ELP one-step AF reduces focusing time by 2/3 (by eliminating the return travel).
- EGS fails to locate the focus in 28.6% of synthetic scenes, and has errors exceeding the depth of focus in 68.3% of cases.
- ELP achieves optimal performance in static scenes even when utilizing only a single frame intensity image (1FPS).
- The adaptive filter effectively suppresses ELP fluctuations caused by violent vibrations.

## Highlights & Insights

1. **Paradigm Shift from Optimization to Detection**: Reformulating autofocusing from an optimization problem that searches for local maxima into a detection problem that identifies abrupt sign transitions is the core theoretical contribution.
2. **Complementarity of Events and Intensity**: Events provide luminance change information with high temporal resolution, while intensity images provide spatial structure information (Laplacian); their product neatly generates a highly discriminative signal at the focal point.
3. **Physics-Driven Design**: The design is strictly derived from the theoretical principles of focusing optics, rather than being purely data-driven.

## Limitations & Future Work

- Requires intensity images to provide Laplacian information, which in event-only schemes necessitates an auxiliary EvTemMap phase.
- The theoretical derivation assumes that scene changes are much slower than the focusing process, which may limit performance in extremely dynamic scenarios.
- Currently, operations are restricted to a single focal-plane ROI; expanding this framework to full-frame, multi-zone autofocusing is worth exploring.

## Related Work & Insights

- The spatial-temporal derivative relationship of ELP might inspire other applications that require real-time detection of optical state changes.
- Event cameras possess immense potential in computational photography; this work demonstrates the unique advantages of purely physical methods.

## Rating

⭐⭐⭐⭐⭐ — Rigorous theory, clear physical intuition, and the "detection vs. search" paradigm shift represents a genuine technological breakthrough. The experimental results, achieving a 22-24 fold reduction in focusing error, are highly impressive. The combination of event cameras with traditional optics elegantly demonstrates the beauty of computational photography.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks](vision-language_gradient_descent-driven_all-in-one_deep_unfolding_networks.md)
- [\[CVPR 2026\] Self-Attention Driven Tensor Representation for High-Order Data Recovery](../../CVPR2026/image_restoration/self-attention_driven_tensor_representation_for_high-order_data_recovery.md)
- [\[CVPR 2025\] INFP: Audio-Driven Interactive Head Generation in Dyadic Conversations](infp_audio-driven_interactive_head_generation_in_dyadic_conversations.md)
- [\[ICCV 2025\] PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining](../../ICCV2025/image_restoration/pre-mamba_a_4d_state_space_model_for_ultra-high-frequent_event_camera_deraining.md)
- [\[CVPR 2026\] Event-Illumination Collaborative Low-light Image Enhancement with a High-resolution Real-world Dataset](../../CVPR2026/image_restoration/event-illumination_collaborative_low-light_image_enhancement_with_a_high-resolut.md)

</div>

<!-- RELATED:END -->
