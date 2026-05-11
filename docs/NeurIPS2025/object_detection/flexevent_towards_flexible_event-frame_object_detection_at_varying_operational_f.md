---
title: >-
  [Paper Note] FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies
description: >-
  [NeurIPS 2025][Object Detection][event camera] This paper proposes FlexEvent, a framework that achieves flexible object detection with event cameras across varying operational frequencies through an adaptive event-frame…
tags:
  - "NeurIPS 2025"
  - "Object Detection"
  - "event camera"
  - "event-frame fusion"
  - "frequency adaptation"
  - "self-training"
date: 2026-05-08
content_hash: 5ddb704bd1ed127e
---

# FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies

**Conference**: NeurIPS 2025  
**arXiv**: [2412.06708](https://arxiv.org/abs/2412.06708)  
**Authors**: Dongyue Lu, Lingdong Kong, Gim Hee Lee, Camille Simon Chane, Wei Tsang Ooi (NUS, CNRS, CY Cergy Paris University)  
**Code**: [flexevent.github.io](https://flexevent.github.io)  
**Area**: Object Detection / Event Camera / Multimodal Fusion  
**Keywords**: event camera, object detection, event-frame fusion, frequency adaptation, self-training

## TL;DR

This paper proposes FlexEvent, a framework that achieves flexible object detection with event cameras across varying operational frequencies through an adaptive event-frame fusion module (FlexFuse) and a frequency-adaptive fine-tuning mechanism (FlexTune). The framework maintains robust performance in the range of 20Hz to 180Hz, significantly outperforming existing methods.

## Background & Motivation

Event cameras offer unique advantages in dynamic environments due to their microsecond-level temporal resolution and asynchronous operation. However, existing event-based detectors suffer from two core limitations:

**Fixed-frequency paradigm**: Most methods align event data with low-frequency frame rates and process event streams at fixed time intervals, ignoring rich temporal details in high-frequency event streams. Performance degrades sharply when high-frequency detection is required in dynamic environments.

**Insufficient semantic information**: Pure event-based methods lack the spatial and semantic information provided by RGB frames; while existing event-frame fusion methods offer improvements, they remain inadequate in adapting to varying operational frequencies.

The central challenge is that annotation of high-frequency event data is extremely costly (requiring substantial manual effort), and existing fusion methods cannot effectively balance the contributions of different modalities at different frequencies. For instance, the classic RVT detector exhibits significant performance degradation when the operational frequency is increased beyond 20Hz.

## Method

### Overall Architecture

FlexEvent consists of two key components:

- **FlexFuse**: An adaptive event-frame fusion module that dynamically integrates high-frequency event data with the rich semantic information of RGB frames.
- **FlexTune**: A frequency-adaptive fine-tuning mechanism that achieves cross-frequency generalization by generating frequency-adjusted labels.

### Event Data Representation

An event camera generates an event $e=(x,y,t,p)$ at pixel $(x,y)$ when the change in log luminance exceeds a threshold $C$, where $p \in \{-1,1\}$ denotes polarity. The event stream is preprocessed into a 4D tensor $E(p,\tau,x,y)$ of dimensions $[2, T, H, W]$, mapping continuous events into $T$ temporal bins via temporal discretization for subsequent convolutional processing.

### FlexFuse: Adaptive Event-Frame Fusion

**Dynamic event aggregation**: Given annotated data at frequency $a$ and corresponding frame data, the time interval $\Delta t^a$ is divided into $b/a$ sub-intervals (where $b > a$), from which a high-frequency event set $\mathbf{E}^b$ is randomly sampled. This strategy introduces millisecond-level temporal jitter during training as implicit temporal augmentation, improving robustness to real-world synchronization noise.

**Feature extraction**: A dual-branch architecture is employed:
- Event branch $\phi_E(\cdot)$: extracts event features based on RVT
- Frame branch $\phi_F(\cdot)$: extracts RGB features based on ResNet-50

Both branches adopt a four-stage structure, extracting event features ${}^{(i)}\mathbf{h}_E^a, {}^{(i)}\mathbf{h}_E^b$ and frame features ${}^{(i)}\mathbf{h}_F$ at each scale $i$.

**Adaptive gated fusion**: At each scale $i$, event and frame features are first concatenated as ${}^{(i)}\mathbf{h}_{\text{shared}}^a = [{}^{(i)}\mathbf{h}_E^a,\ {}^{(i)}\mathbf{h}_F]$, and adaptive soft weights are computed via a noisy gating function:

$$[\alpha, \beta] = \text{Softmax}((\mathbf{h}_{\text{shared}} \cdot \mathbf{W}) + \sigma \cdot \epsilon)$$

where $\mathbf{W}$ is a trainable weight matrix, $\sigma$ is a learned standard deviation controlling the magnitude of noise perturbation, and $\epsilon \sim \mathcal{N}(0,1)$ is Gaussian noise. The fused features are obtained via element-wise weighting:

$$\mathbf{h}_{\text{fuse}}^a = \alpha \odot \mathbf{h}_E^a + \beta \odot \mathbf{h}_F$$

The fusion features from different frequencies are summed as $\mathbf{h}_{\text{fuse}} = \mathbf{h}_{\text{fuse}}^a + \mathbf{h}_{\text{fuse}}^b$, and multi-scale features are concatenated before being fed into the detection head.

**Regularization**: A coefficient of variation penalty term is introduced to prevent the model from collapsing onto a single modality:

$$\mathcal{L}_{\text{fuse}} = \mathcal{L}_{\text{det}} + \lambda \left(\frac{\text{Var}(\alpha)}{(\mathbb{E}[\alpha])^2} + \frac{\text{Var}(\beta)}{(\mathbb{E}[\beta])^2}\right)$$

### FlexTune: Frequency-Adaptive Fine-Tuning

FlexTune consists of two main stages:

**Stage 1 — Low-frequency sparse training**: Training is performed at high frequency $b$, but only the last event (corresponding to the annotation timestamp) is used, enabling the model to capture high-frequency temporal information while leveraging low-frequency labels.

**Stage 2 — Cross-frequency propagation**: This stage comprises three steps:

1. **High-Frequency Bootstrapping**: The pre-trained model generates pseudo-labels $\tilde{\mathbf{y}}$ on the complete high-frequency event set.

2. **Temporal Consistency Calibration**:
    - Bidirectional event augmentation: the event stream is processed in both forward and reverse directions to enhance recall.
    - Confidence-aware filtering: NMS and a low-confidence threshold $\tau$ are applied to eliminate duplicates and retain high-potential detections.
    - Trajectory pruning: IoU-based tracking associates cross-frame detections, and short trajectories are pruned to suppress transient noise.

3. **Cyclic Self-Training**: Iterative training with the total loss function:

$$\mathcal{L}_{\text{tune}} = \mathcal{L}_{\text{GT}} + \beta \sum \mathcal{L}_{\text{det}}(\tilde{\mathbf{y}}, \hat{\mathbf{y}})$$

## Key Experimental Results

### Experimental Setup

Validation is conducted on three large-scale datasets:
- **DSEC-Det**: 78,344 frames, 60 sequences, 8 categories (primary benchmark)
- **DSEC-Detection**: 52,727 frames, 41 sequences, 3 categories
- **DSEC-MOD**: 13,314 frames, 16 sequences, 1 category

Training runs for 100K iterations with batch size=8, sequence length=11, learning rate 1e-4, completed in approximately one day on two A5000 GPUs.

### Main Results

| Dataset | Metric | Prev. SOTA | Ours | Gain |
|--------|------|--------|-----------|------|
| DSEC-Det | mAP | 41.9 (DAGr-50) | **57.4** | +15.5% |
| DSEC-Detection | Avg mAP | 38.0 (CAFR) | **47.4** | +9.4% |
| DSEC-MOD | Avg mAP | 29.0 (RENet) | **36.9** | +7.9% |

Full metrics on DSEC-Det: mAP 57.4, AP50 78.2, AP75 66.6, APS 51.7, APM 64.9, APL 83.7, surpassing all baseline methods across the board.

### High-Frequency Generalization

FlexEvent demonstrates exceptional robustness under frequency variation:

| Frequency | 20Hz | 36Hz | 45Hz | 60Hz | 90Hz | 180Hz | Avg |
|------|------|------|------|------|------|-------|------|
| w/o FlexFuse/FlexTune | 53.2 | 52.0 | 49.4 | 45.9 | 38.8 | 22.9 | 43.7 |
| Full FlexEvent | **57.4** | **60.1** | **59.5** | **58.8** | **56.5** | **50.9** | **57.2** |

- Only ~1.5% performance drop from 20Hz to 90Hz (retaining 96.2%)
- Still achieves 50.9% mAP at the extreme condition of 180Hz (baseline: only 22.9%)

### Inference Efficiency

| Method | Params | 20Hz | 90Hz | 180Hz |
|------|--------|------|------|-------|
| RVT | 18.5M | 9.20ms | 7.19ms | 6.77ms |
| DAGr-50 | 34.6M | 73.35ms | 45.29ms | 43.89ms |
| FlexEvent | 45.4M | 14.27ms | 12.47ms | 12.37ms |

Despite the larger parameter count, inference speed is comparable to SAST and far faster than DAGr. FlexTune operates offline and introduces no runtime overhead.

### Ablation Study

- **FlexFuse contribution**: Adding frame information alone improves average mAP from 43.7% to 56.4%, with more pronounced gains at high frequencies.
- **FlexTune contribution**: At 180Hz, mAP improves from 22.9% to 30.4% (without FlexFuse); jointly with FlexFuse, it improves from 49.2% to 50.9%.
- **Fusion strategy comparison**: Adaptive gating outperforms simple Add, Concat, and Vanilla Attention.
- **Interpolated labels vs. FlexTune**: Linear interpolation of labels performs poorly for rapidly appearing/disappearing objects; FlexTune generates more accurate pseudo-labels via temporal consistency calibration.

## Highlights & Insights

- **Frequency flexibility**: This work is the first to explicitly address event camera detection across varying operational frequencies, maintaining high accuracy from 20Hz to 180Hz—a practically significant property that eliminates the need to train separate models for different scenarios.
- **Elegant fusion design**: The noisy adaptive gating mechanism is concise yet effective, dynamically balancing the contributions of event and frame modalities via learned soft weights; coefficient of variation regularization prevents modality collapse.
- **Pseudo-label quality assurance**: Temporal consistency calibration in FlexTune (bidirectional augmentation + trajectory pruning) ensures the reliability of high-frequency pseudo-labels and avoids the noise accumulation commonly seen in self-training pipelines.
- **Strong practicality**: FlexTune is an offline step that incurs no inference overhead; the overall framework completes training on two A5000 GPUs in one day, offering good reproducibility.

## Limitations & Future Work

- **RGB frame dependency**: In extreme lighting conditions (e.g., complete darkness), RGB frame quality degrades and fusion may introduce noise; a fallback mechanism for pure event mode warrants exploration.
- **Pseudo-label ceiling**: The quality of high-frequency pseudo-labels in FlexTune is bounded by the teacher model's initial performance at low frequency, imposing a natural performance ceiling.
- **Limited category scope**: DSEC datasets primarily cover driving scenarios (vehicles, pedestrians); generalization to more diverse object categories and scenes (indoor, industrial) remains unverified.
- **Computational overhead**: Deployment challenges of the 45.4M-parameter model on embedded platforms are not discussed; lighter variants are needed for practical autonomous driving applications.
- **Temporal consistency assumption**: Trajectory pruning assumes relatively smooth object motion and may fail under extreme conditions such as sudden occlusion or hard braking.

## Related Work & Insights

- **RVT (CVPR'23)**: A Transformer-based event detector serving as the backbone of the event branch, but its fixed-frequency paradigm limits high-frequency performance.
- **DAGr (Nature'24)**: The latest event-frame fusion method based on graph attention networks; this work surpasses it by +15.5% mAP on DSEC-Det.
- **CAFR (ECCV'24)**: A cross-attention fusion method; this work surpasses it by +9.4% on DSEC-Detection.
- **LEOD (CVPR'24)**: A pioneer in label-efficient event detection, but does not address high-frequency generalization.
- **SSM (CVPR'24)**: A frequency-adaptive method based on state space models, but the pure event mode struggles to detect static objects at high frequencies.

## Rating ⭐

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision](../../ICCV2025/object_detection/evrt-detr_latent_space_adaptation_of_image_detectors_for_event-based_vision.md)
- [\[NeurIPS 2025\] Test-Time Adaptive Object Detection with Foundation Model](test-time_adaptive_object_detection_with_foundation_model.md)
- [\[NeurIPS 2025\] ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection](recon_region-controllable_data_augmentation_with_rectification_and_alignment_for.md)
- [\[NeurIPS 2025\] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)
- [\[NeurIPS 2025\] CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection](cq-dino_mitigating_gradient_dilution_via_category_queries_for_vast_vocabulary_ob.md)

</div>

<!-- RELATED:END -->
