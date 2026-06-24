---
title: >-
  [Paper Note] EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision
description: >-
  [ICCV 2025][Object Detection][event camera] This paper proposes I2EvDet, a framework that adapts mainstream image detectors to event-based video detection by inserting lightweight RNN temporal modules into the frozen latent space of RT-DETR, achieving state-of-the-art results of +2.3 and +1.4 mAP on the Gen1 and 1Mpx benchmarks, respectively, with minimal architectural modifications.
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "event camera"
  - "RT-DETR"
  - "latent space adaptation"
  - "temporal modeling"
date: 2026-05-08
content_hash: 45c1b1fe0066404b
---

# EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision

**Conference**: ICCV 2025
**arXiv**: [2412.02890](https://arxiv.org/abs/2412.02890)  
**Code**: [https://github.com/realtime-intelligence/evrt-detr](https://github.com/realtime-intelligence/evrt-detr)  
**Area**: Object Detection / Event-based Vision
**Keywords**: event camera, object detection, RT-DETR, latent space adaptation, temporal modeling

## TL;DR

This paper proposes I2EvDet, a framework that adapts mainstream image detectors to event-based video detection by inserting lightweight RNN temporal modules into the frozen latent space of RT-DETR, achieving state-of-the-art results of +2.3 and +1.4 mAP on the Gen1 and 1Mpx benchmarks, respectively, with minimal architectural modifications.

## Background & Motivation

Event-based cameras (EBCs) are bio-inspired visual sensors that offer significant advantages over conventional frame-based cameras, including low power consumption (~10 mW), high temporal resolution (μs-level), and high dynamic range (>100 dB), making them attractive for applications in autonomous driving and robotics. However, the sparse and asynchronous nature of event data poses fundamental challenges to conventional vision methods.

Existing EBC object detection methods have primarily followed two lines: (1) designing sophisticated event data representations (e.g., Time Surface, ERGO-12); and (2) constructing dedicated architectures to handle the temporal characteristics of event data (e.g., RVT, S5-ViT). Both lines assume a fundamental incompatibility between EBC data and conventional vision, causing EBC research to diverge increasingly from mainstream computer vision.

This paper proposes a fundamentally different perspective: leveraging the powerful feature extraction capabilities of mainstream object detectors and bridging the gap between the two domains through targeted adaptation rather than designing new architectures from scratch.

## Method

### Overall Architecture

I2EvDet is a general two-stage adaptation framework. In the first stage, an image detector is trained directly on simple event frame representations. In the second stage, the detector parameters are frozen, and lightweight RNN modules are inserted into the encoder's latent space to capture temporal dynamics. The framework is applicable to any detection model with a clear separation between feature extraction and detection (e.g., YOLO series, DETR series).

### Key Designs

1. **Stacked 2D Histogram Event Representation**: The event stream is divided into frames using a fixed time window of $T_{\text{frame}}=50$ ms, and each frame is further subdivided into 10 intervals of $T_{\text{bin}}=5$ ms. An intermediate stacked histogram $S(p, t_i, y, x)$ of shape $(2, 10, H, W)$ is constructed; after merging the polarity and temporal dimensions, a $(20, H, W)$ image-like representation is obtained. Although simple, this representation provides a directly compatible input format for mainstream detectors.

2. **Latent Space Temporal Adaptation Module (Core of I2EvDet)**: Three parallel ConvLSTM temporal memory modules $\{\mathbf{R}_3, \mathbf{R}_4, \mathbf{R}_5\}$ are inserted into the multi-scale features $\{\mathcal{E}_3, \mathcal{E}_4, \mathcal{E}_5\}$ output by the frozen RT-DETR encoder. Each module interacts with the corresponding feature map via a residual connection:

    $\mathcal{E}_i^{t,proj} = W_i^{down} \cdot \mathcal{E}_i^t$
    $(\mathcal{O}_i^{t,proj}, \mathcal{M}_i^t) = \mathbf{R}_i(\mathcal{E}_i^{t,proj}, \mathcal{M}_i^{t-1})$
    $\tilde{\mathcal{E}}_i^t = \mathcal{E}_i^t + \alpha_i \cdot W_i^{up} \cdot \mathcal{O}_i^{t,proj}$

   where $W_i^{down}$ and $W_i^{up}$ are projection matrices and $\alpha_i$ is a learnable scaling factor (inspired by the ReZero technique). This design incorporates temporal context while preserving the integrity of the original spatial representations.

3. **Theoretical Justification for RNN over Transformer**: Event cameras face a unique challenge — stationary objects produce no events and are thus "invisible" to the camera, requiring persistent memory of historical events for detection. The fixed context window of Transformers limits temporal memory length, whereas RNNs theoretically possess unbounded memory capacity through their recurrent states, making them particularly well-suited for maintaining object presence in event streams.

### Loss & Training

A two-stage training strategy is employed. In the first stage, RT-DETR is trained on single-frame event images using the Adam optimizer with EMA weight averaging. In the second stage, all first-stage parameters are frozen, and only the newly inserted RNN modules are trained. Temporal training uses a mix of random and sequential clips; RNN state continuity is maintained for sequential clips, while states are reset for random clips. Clip lengths of 21 frames (Gen1) and 10 frames (1Mpx) are used.

## Key Experimental Results

### Main Results

| Model | Gen1 mAP(%) | 1Mpx mAP(%) | Params (M) | Inference (ms) |
|------|------------|-------------|----------|-------------|
| RVT-B | 47.2 | 47.4 | 18.5 | 10.2/11.9 |
| ERGO-12 | 50.4 | 40.6 | 59.6 | 69.9/100.0 |
| S5-ViT-B | 47.4 | 47.2 | 17.5 | 8.2/9.6 |
| SAST-CB | 48.2 | 48.7 | 18.9 | -/57.5 |
| RT-DETR-B (Stage1) | 47.6 | 45.2 | 42.8 | 10.5/14.9 |
| **EvRT-DETR-B** | **52.7** | **50.1** | 57.1 | 12.7/18.8 |
| **EvRT-DETR-T** | 52.3 | 49.9 | 34.4 | 8.4/12.5 |

### Ablation Study

| Temporal Module Config $(x,y,z)$ | mAP(%) | mAP₅₀(%) | Note |
|----------------------|--------|----------|------|
| $(0,1,1)$ w/o $\mathcal{E}_3$ | 51.0 | 80.7 | Largest drop from missing low-level features |
| $(1,0,1)$ | 52.2 | 81.3 | - |
| $(1,1,0)$ | 52.4 | 81.7 | - |
| $(1,1,1)$ Full | **52.7** | **82.0** | Multi-scale temporal adaptation is optimal |

| Hidden Dim M | mAP(%) | Trainable Params (M) | Note |
|----------|--------|-------------|------|
| 64 (4× compression) | 52.1 | 2.3 | Only 5.4% param increase, already surpasses all prior SOTA |
| 128 | 52.5 | 5.4 | - |
| 256 (default) | 52.7 | 14.4 | - |
| 512 | 52.9 | 42.9 | Diminishing returns |

| Detector | Original mAP(%) | After I2EvDet mAP(%) | Gain |
|--------|----------|-------------------|------|
| RT-DETR-T | 46.0 | 52.3 | +6.3 |
| RT-DETR-B | 47.6 | 52.7 | +5.1 |
| YOLOX-T | 36.0 | 42.4 | +6.4 |
| YOLOX-X | 43.4 | 47.8 | +4.4 |

### Key Findings

- Training RT-DETR with a simple stacked histogram representation alone already matches the performance of dedicated EBC methods, challenging the assumption that event data requires specialized architectures.
- Low-level features ($\mathcal{E}_3$) benefit most from temporal adaptation; removing the corresponding module causes the largest performance drop.
- Temporal adaptation is critical for detecting stationary objects: both RT-DETR and EvRT-DETR perform well when objects are in motion, but only EvRT-DETR maintains detection when objects are stationary.
- The I2EvDet framework consistently improves performance by 4.4–6.4 mAP across different architectures (RT-DETR, YOLOX), validating its generality.

## Highlights & Insights

- The research approach is highly elegant: rather than designing new architectures for event cameras, the paper adapts already well-optimized mainstream detectors.
- LoRA-like parameter-efficient adaptation (only 2.3M additional parameters at $M=64$) already surpasses all prior methods, suggesting that the event camera field may have been consistently over-engineering solutions.
- The theoretical motivation for choosing RNN over Transformer as the temporal module is well-grounded — the "stationary equals invisible" property of event cameras demands unbounded memory.

## Limitations & Future Work

- Inference latency increases with the temporal modules, which may become a bottleneck in extreme real-time scenarios.
- Validation is limited to two autonomous driving datasets; other event camera application scenarios (e.g., high-speed motion capture) remain unexplored.
- The choice of ConvLSTM is relatively conservative; more advanced state space models (e.g., Mamba) could be explored as alternatives.

## Related Work & Insights

- The latent space adaptation paradigm is analogous to the success of LoRA in NLP, validating the cross-domain effectiveness of the "freeze + lightweight plugin" approach.
- The I2EvDet framework can be generalized to other temporal vision tasks (e.g., video object detection, optical flow estimation).
- The paper establishes a new research paradigm of "standing on the shoulders of giants" for the event camera community.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of adapting mainstream detectors to event cameras is novel, and the latent space RNN insertion scheme is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ State-of-the-art results on two standard benchmarks, cross-architecture validation, and thorough parameter efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, rigorous method description, and information-dense figures and tables.
- Value: ⭐⭐⭐⭐⭐ Opens a new research direction for event-based vision based on adapting mainstream architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Visual Modality Prompt for Adapting Vision-Language Object Detectors](visual_modality_prompt_for_adapting_vision-language_object_detectors.md)
- [\[ICCV 2025\] Sim-DETR: Unlock DETR for Temporal Sentence Grounding](sim-detr_unlock_detr_for_temporal_sentence_grounding.md)
- [\[ICCV 2025\] DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion](distil_data-free_inversion_of_suspicious_trojan_inputs_via_latent_diffusion.md)
- [\[ICCV 2025\] UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement](upre_zero-shot_domain_adaptation_for_object_detection_via_unified_prompt_and_rep.md)
- [\[CVPR 2026\] AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](../../CVPR2026/object_detection/anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)

</div>

<!-- RELATED:END -->
