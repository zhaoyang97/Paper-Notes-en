---
title: >-
  [Paper Note] ChangeBridge: Spatiotemporal Image Generation with Multimodal Controls for Remote Sensing
description: >-
  [CVPR 2026][Image Generation][Remote sensing change generation] Proposes ChangeBridge, which achieves conditional spatiotemporal image generation from pre-event to post-event in remote sensing scenes via a drift-asynchro…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Remote sensing change generation"
  - "diffusion bridge model"
  - "spatiotemporal image generation"
  - "multimodal conditions"
  - "change detection data engine"
date: 2026-05-08
content_hash: 8b670ba44517615b
---

# ChangeBridge: Spatiotemporal Image Generation with Multimodal Controls for Remote Sensing

**Conference**: CVPR 2026  
**arXiv**: [2507.04678](https://arxiv.org/abs/2507.04678)  
**Code**: [GitHub](https://github.com/zhenghuizhao/ChangeBridge)  
**Area**: Image Generation  
**Keywords**: Remote sensing change generation, diffusion bridge model, spatiotemporal image generation, multimodal conditions, change detection data engine

## TL;DR

Proposes ChangeBridge, which achieves conditional spatiotemporal image generation from pre-event to post-event in remote sensing scenes via a drift-asynchronous diffusion bridge. It supports multimodal controls including coordinate-text, semantic masks, and instance layouts, serving as a data generation engine for change detection tasks.

## Background & Motivation

**Background**: Remote sensing generation methods have progressed in sectors like layout-to-image and modality conversion, but conditional spatiotemporal generation (synthesizing future scenes based on historical observations and multimodal conditions) remains largely unexplored.

**Limitations of Prior Work**: Existing change generation methods start from pure noise and can only handle event-driven changes (e.g., new buildings). They fail to model cross-temporal dynamics (e.g., seasonal lighting changes, vegetation growth) and lack a direct correlation between pre- and post-temporal phases.

**Key Challenge**: Spatiotemporal generation must simultaneously handle heterogeneous evolution—drastic foreground event changes + subtle background temporal dynamics—where the evolution speed and magnitude differ significantly.

**Goal**: Design a generative model capable of discriminatively processing foreground event changes and background temporal evolution.

**Key Insight**: Diffusion bridge models to replace pure noise initialization + pixel-level drift magnitude maps to achieve asynchronous evolution.

**Core Idea**: Drift-asynchronous diffusion bridge—starting from a composite state of the pre-event, using different drift magnitudes to control differentiated generation of foreground and background.

## Method

### Overall Architecture

Three core modules: (a) Composite bridge initialization—a composite image of pre-event background + conditional foreground serves as the diffusion starting point; (b) Asynchronous drift diffusion—a pixel-level drift map assigns different evolution magnitudes to foreground/background; (c) Drift-aware denoising—the denoising network is conditioned on the drift map. Supports both UNet and DiT backbones.

### Key Designs

1. **Composite Bridge Initialization**: Given multimodal conditions $\mathbf{x}_c$, the foreground mask $\mathbf{M}_{fg}$ is extracted to construct $\mathbf{x}_a = \mathbf{M}_{fg} \odot \mathbf{x}_c + (1-\mathbf{M}_{fg}) \odot \mathbf{x}_0$, serving as the starting point of the diffusion bridge rather than noise. Design Motivation: Starting from a composite state maintains spatial consistency and temporal continuity better than starting from noise.

2. **Asynchronous Drift Diffusion**: Defines $\mathbf{d}_{map} = \mathbf{M}_{fg} \cdot \gamma^{fg} + (1-\mathbf{M}_{fg}) \cdot \gamma^{bg}$ ($\gamma^{fg}=1.0, \gamma^{bg}=0.8$), modifying the drift coefficient $\tilde{m}_t(i,j) = m_t \cdot \mathbf{z}_d(i,j)$. Design Motivation: Foreground requires large-scale generation while background only needs slight evolution; uniform drift would lead to imbalance.

3. **Drift-Aware Denoising**: The denoising network is conditioned on $\mathbf{z}_d$ (latent representation of the drift map) and $\mathbf{z}_c$ (pre-event context). Loss:
    $$\mathcal{L}_{asy} = \mathbb{E}\left[\|\tilde{m}_t(\mathbf{z}_a - \mathbf{z}_b) + \sqrt{\delta_t}\epsilon - \epsilon_\theta(\mathbf{z}_t, t, \mathbf{z}_a, \mathbf{z}_c, \mathbf{z}_d)\|^2\right]$$

### Loss & Training

- UNet (SD1.5) 60 epochs, DiT (DiT-XL/2) 100 epochs, Adam 1e-4, batch 64, 2×A100.
- VQGAN encoder + SkyCLIP text encoder.

## Key Experimental Results

### Main Results

| Condition | Method | FID↓ | IS↑ | Consistency |
|-----|------|------|-----|--------|
| Coord-Text | Instruct-Imagen | 48.17 | 3.70 | CosSim 0.81 |
| Coord-Text | **Ours-T** | **31.45** | **5.14** | **0.85** |
| Layout (WHU) | Changen2 | 48.85 | 5.64 | IoU 74.33 |
| Layout (WHU) | **Ours-T** | **40.12** | **6.77** | **78.13** |
| Semantic (SECOND) | Changen2 | 69.43 | 6.18 | mIoU 73.20 |
| Semantic (SECOND) | **Ours-T** | **59.33** | **6.41** | **74.26** |

### Ablation Study

| CB | AD | DD | FID↓ | IoU↑ | Description |
|----|----|----|------|------|------|
| ✗ | ✗ | ✗ | 76.81 | 65.29 | SD1.5 Baseline |
| ✓ | ✗ | ✗ | 56.24 (-20.57) | 71.87 | Bridge initialization contributes most |
| ✓ | ✓ | ✓ | 45.47 (-11.59) | 75.30 | Full effect of three components |

### Key Findings

1. Composite bridge initialization contributes the most (FID decreased by 20.57), verifying that "starting from a state" is superior to "starting from noise."
2. As a data engine: 2× synthetic data augmentation can improve change detection tasks by BCD +2.26 IoU and CC +10.97 CIDEr.
3. The DiT variant overall outperforms the UNet variant (FID 31.45 vs 38.36).

## Highlights & Insights

- First to propose the task of conditional spatiotemporal image generation for remote sensing, filling the gap where change generation could not model temporal dynamics.
- The drift-asynchronous diffusion bridge is the core innovation, introducing spatial adaptive drift within the diffusion bridge.
- Huge potential as a data engine: remote sensing change detection faces severe scarcity of paired data.
- Backbone-agnostic design (applicable to both UNet/DiT), showing high generalizability.

## Limitations & Future Work

- Drift magnitudes $\gamma^{fg}, \gamma^{bg}$ require manual setting; only supports 256×256 resolution.
- Drift modeling for transition areas (e.g., transition zones between old and new buildings) has not been explored.
- Diminishing returns after exceeding 2× synthetic data.

## Related Work & Insights

- BBDM provides the theoretical basis for "state-to-state" generation; ChangeBridge extends this with asynchronous drift.
- Direct value for applications like urban planning and disaster assessment.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ New technical contribution with drift-asynchronous diffusion bridge; task definition is also pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 datasets × 6 baselines × 3 conditions + downstream validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Complete mathematical derivations and exquisite illustrations.
- Value: ⭐⭐⭐⭐⭐ Trinity of task definition + method innovation + data engine.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OpenDPR: Open-Vocabulary Change Detection via Vision-Centric Diffusion-Guided Prototype Retrieval for Remote Sensing Imagery](opendpr_open-vocabulary_change_detection_via_vision-centric_diffusion-guided_pro.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[CVPR 2026\] ConsistCompose: Unified Multimodal Layout Control for Image Composition](consistcompose_multimodal_layout_control.md)
- [\[CVPR 2026\] Mixture of States: Routing Token-Level Dynamics for Multimodal Generation](mixture_of_states_routing_token-level_dynamics_for_multimodal_generation.md)
- [\[CVPR 2026\] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](enhancing_image_aesthetics_with_dualconditioned_di.md)

</div>

<!-- RELATED:END -->
