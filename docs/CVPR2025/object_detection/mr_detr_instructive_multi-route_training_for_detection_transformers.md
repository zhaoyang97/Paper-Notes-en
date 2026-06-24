---
title: >-
  [Paper Note] Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE
description: >-
  [CVPR 2025][Object Detection][DETR] This work systematically investigates the roles of various components in the DETR decoder within a joint one-to-one/one-to-many multi-task framework, and reveals that transitioning any single component to be independent can effectively coordinate the two objectives. Based on this, instructive multi-route training is proposed (Instructive Self-Attention + Independent FFN + Route-Aware MoE), which discards auxiliary routes during inference…
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "DETR"
  - "multi-route training"
  - "instructive self-attention"
  - "mixture-of-experts"
  - "one-to-many assignment"
  - "detection transformer"
date: 2026-05-08
content_hash: fe117bda2bbcd41b
---

# Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE

**Conference**: CVPR 2025  
**arXiv**: [2412.10028](https://arxiv.org/abs/2412.10028)  
**Code**: [Project Page](https://visual-ai.github.io/mrdetr/)  
**Area**: Object Detection  
**Keywords**: DETR, multi-route training, instructive self-attention, mixture-of-experts, one-to-many assignment, detection transformer

## TL;DR

This work systematically investigates the roles of various components in the DETR decoder within a joint one-to-one/one-to-many multi-task framework, and reveals that transitioning any single component to be independent can effectively coordinate the two objectives. Based on this, instructive multi-route training is proposed (Instructive Self-Attention + Independent FFN + Route-Aware MoE), which discards auxiliary routes during inference, incurring zero extra cost.

## Background & Motivation

**Background**: The DETR family achieves end-to-end detection without NMS through one-to-one matching, but the sparse supervision of one-to-one matching leads to slow convergence. Existing methods (H-DETR, DN-DETR, DINO, DAC-DETR, MS-DETR) introduce auxiliary one-to-many matching to accelerate training.

**Limitations of Prior Work**: (1) Simply sharing all components for both one-to-one and one-to-many predictions severely degrades one-to-one performance (-6.0 AP); (2) Existing methods only investigate the functions of decoder components under single-task settings, lacking a systematic analysis within a multi-task framework; (3) The designs of DAC-DETR and MS-DETR are based on heuristic observations rather than systematic experiments.

**Key Challenge**: A conflict arises when the two training objectives, one-to-one and one-to-many, pass through the same decoder—the same predicted box may be a positive sample in one-to-many but a negative sample in one-to-one.

**Goal**: Systematically understand the roles of individual decoder components (self-attention, cross-attention, FFN) under a multi-task framework, and identify the optimal component-sharing/independent strategy.

**Key Insight**: View the joint one-to-one and one-to-many auxiliary training as multi-task learning, and exhaustively validate the performance of different combinations of shared/independent components.

## Method

### Overall Architecture

Three-route decoder structure:
- **Route-2 (Main Route)**: Identical to the baseline model, performs one-to-one prediction, and is preserved during inference.
- **Route-1 (Auxiliary-FFN)**: Uses an independent FFN to perform one-to-many prediction, and is discarded during inference.
- **Route-3 (Auxiliary-InstructSA)**: Uses Instructive Self-Attention to perform one-to-many prediction, and is discarded during inference.

All routes share the object queries and detection heads.

### Key Designs

#### 1. Empirical Foundation of the Multi-Route Training Mechanism

Exhaustive validation of 12 shared/independent component combinations (Table I):
- Sharing all components: -6.0 AP (severe task conflict).
- Independent Self-Attention: +2.1 AP; Independent Cross-Attention: +1.6 AP; Independent FFN: +2.0 AP.
- **Having any individual component independent is sufficient to effectively coordinate the two objectives.**
- Optimal combination: The two-route combination of independent SA + independent FFN achieves +2.4 AP (independent CA degrades performance because CA converges slowly).

#### 2. Instructive Self-Attention

To replace independent Self-Attention and reduce parameters, an instructive mechanism is designed:
- Construct $m$ learnable instruction tokens $\mathbf{Q}^{ins}$.
- Concatenate them with the object queries and perform Self-Attention with shared parameters: $\hat{Q}^{ins} = \{q_0^{ins}, ..., q_{m-1}^{ins}, q_0, ..., q_{n-1}\}$.
- Discard the outputs of the instruction tokens after Self-Attention.
- **No extra parameters** (shares SA weights), guiding the queries from one-to-one to one-to-many prediction mode solely through the instruction tokens.
- Uses 10 instruction tokens by default.

#### 3. Route-Aware Mixture-of-Experts (Mr. DETR++)

Replaces the two independent FFNs with MoE to achieve knowledge sharing:
- $t$ shared experts, with sparse top-$k$ activation.
- Route-2 and Route-3 share the gating function $G(\cdot)$.
- Route-1 uses an independent gating function $G'(\cdot)$—preventing gradients from the one-to-many route from interfering with the one-to-one route.
- **Scale-aware MoE in Encoder**: Applies MoE to low-resolution features, while using only a shared FFN for high-resolution features, to balance computational overhead.

### Loss & Training

- **One-to-one Matching Route (Route-2)**: Standard Hungarian matching + classification loss + box loss.
- **One-to-many Matching Route (Route-1, Route-3)**: Top-$K$ matching based on $M_{ij} = \alpha \cdot s_i + (1-\alpha) \cdot \text{IoU}(b_i, \bar{b_j})$.
- **Localization-aware Score Calibration**: VFL Loss learns a class-aware IoU score, making $s_{calib} = s_{cls}^\phi \cdot s_{iou}^{1-\phi}$ during inference.
- Parameters: $K=6$, $\alpha=0.3$, $\tau=0.4$, with $\phi$ used to balance classification and localization confidence.

## Key Experimental Results

### Main Results (Table II, ResNet-50)

| Baseline | Queries | Epochs | Baseline AP | +Mr.DETR | +Mr.DETR++ |
|------|---------|--------|---------|----------|------------|
| Deformable-DETR++ | 300 | 12 | 47.0 | 49.5 (+2.5) | 51.0 (+4.0) |
| Deformable-DETR++ | 900 | 12 | 47.6 | 50.7 (+3.1) | 51.8 (+4.2) |
| DINO | 900 | 12 | 49.0 | 50.9 (+1.9) | 52.2 (+3.2) |
| Align-DETR | 900 | 12 | 50.2 | 51.4 (+1.2) | 52.2 (+2.0) |

### Swin-L Backbone (Table III, 12 epochs, 900 queries)

| Method | AP | AP50 | AP75 |
|------|-----|------|------|
| DINO | 56.8 | 75.4 | 62.3 |
| Rank-DETR | 57.6 | 76.0 | 63.4 |
| Stable-DINO | 57.7 | 75.7 | 63.4 |
| **Mr. DETR** | **58.4** | **76.3** | **63.9** |
| **Mr. DETR++** | **58.7** | **76.5** | **64.0** |

### Large-Scale Dataset Scaling (ResNet-50, 900 queries, 12 epochs)

| Dataset | Baseline AP | +Mr.DETR | +Mr.DETR++ |
|--------|---------|----------|------------|
| Objects365 | 30.4 | 32.7 (+2.3) | 34.9 (+4.5) |
| NuImages | 48.5 | 51.2 (+2.7) | 52.3 (+3.8) |

### Ablation Study (Table I)

| Combination | No. Routes | AP(o2o) | vs Baseline |
|------|--------|---------|------------|
| Fully Shared | 1 | 41.6 | -6.0 |
| Independent SA | 2 | 49.7 | +2.1 |
| Independent FFN | 2 | 49.6 | +2.0 |
| Independent SA + Independent FFN | 3 | **50.0** | **+2.4** |
| Independent CA + Independent FFN | 3 | 49.0 | +1.4 |

### Key Findings

1. **Any Single Independent Component Suffices**: This is the most core empirical finding—breaking the preconceived notion that only SA can differentiate between o2o/o2m.
2. **Independent CA is Detrimental**: The independent CA combination yields the worst performance (+1.4 vs +2.4) due to the slow convergence of independent CA.
3. **Zero Inference Overhead for Mr. DETR++**: Auxiliary routes are completely discarded during inference, without affecting the architecture and speed.
4. **Consistent Improvements across Baselines**: Consistently improves 1-4 AP across Deformable-DETR++, DINO, and Align-DETR.
5. **Consistent across Datasets**: Significant improvements on Objects365 (+4.5) and NuImages (+3.8).

## Highlights & Insights

1. **Systematic Empirical Analysis**: Exhaustive experiments of 12 combinations in Table I clearly reveal the roles of decoder components under a multi-task framework, providing reliable design guidelines for future works.
2. **Ingenious Instruction Token Design**: Simply prepending learnable tokens switches the self-attention behavior mode ($o2o \to o2m$) without introducing any new parameters—conceptually simple yet highly effective.
3. **Mitigating Conflict via Route-Aware MoE**: The design of shared experts + independent gating strikes an effective balance between knowledge sharing and task conflicts.
4. **Plug-and-Play**: Does not alter the baseline model architecture, introduces no inference cost, and is adaptable to various DETR variants—exhibiting high practical value.

## Limitations & Future Work

1. Increases training computational overhead by around 20-30% (requiring gradient computations for three routes).
2. MoE introduces additional expert parameters, which increases training memory despite being reducible during inference.
3. The number of instruction tokens (10) is a hyperparameter, which, although shown to be insensitive in experiments, might have varying optimal values depending on the model.
4. Validation is primarily conducted on COCO, Objects365, and NuImages; more downstream tasks (e.g., open-vocabulary detection) remain to be explored.

## Related Work & Insights

- **DAC-DETR**: Achieves o2m by removing SA—this paper proves that discarding SA is unnecessary, and using instruction tokens is more elegant.
- **MS-DETR**: Uses CA output for o2m and SA output for o2o—this paper proves that this is not the only effective combination.
- **DINO**: Uses multiple groups of denoising queries for auxiliary training—the multi-route strategy in this paper is orthogonal to denoising queries and can be combined.
- **Inspiration**: The methodology of component-wise analysis under a multi-task framework can be extended to other multi-head/multi-task architectures (e.g., multimodal fusion modules in VLMs).

## Rating

⭐⭐⭐⭐ (8/10)

- **Novelty**: ⭐⭐⭐⭐ — Valuable systematic empirical analysis; the Instruction Token design is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Convincing with multiple baselines, datasets, tasks, and exhaustive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, with a complete logical flow from experimental observations to method design.
- **Value**: ⭐⭐⭐⭐⭐ — High engineering value owing to zero inference overhead, plug-and-play capability, and generalizability across baselines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)
- [\[CVPR 2025\] Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset](object_detection_using_event_camera_a_moe_heat_conduction_based_detector_and_a_n.md)
- [\[CVPR 2026\] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](../../CVPR2026/object_detection/unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)
- [\[ICLR 2026\] RF-DETR: Neural Architecture Search for Real-Time Detection Transformers](../../ICLR2026/object_detection/rf-detr_neural_architecture_search_for_real-time_detection_transformers.md)
- [\[CVPR 2026\] YOLO-Master: MOE-Accelerated with Specialized Transformers for Enhanced Real-time Detection](../../CVPR2026/object_detection/yolo-master_moe-accelerated_with_specialized_transformers_for_enhanced_real-time.md)

</div>

<!-- RELATED:END -->
