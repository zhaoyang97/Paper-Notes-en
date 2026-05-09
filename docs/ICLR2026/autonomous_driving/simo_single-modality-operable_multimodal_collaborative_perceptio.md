---
title: >-
  [Paper Note] SiMO: Single-Modality-Operable Multimodal Collaborative Perception
description: >-
  [ICLR 2026][Autonomous Driving][collaborative perception] This paper proposes SiMO, a framework that introduces the LAMMA fusion module and PAFR training strategy to achieve, for the first time in multi-agent collaborative perception, a multimodal perception system that remains operational under arbitrary modality absence—particularly when LiDAR fails and only cameras are available. The design is analogous to a parallel circuit: the system functions as long as at least one pathway is active.
tags:
  - ICLR 2026
  - Autonomous Driving
  - collaborative perception
  - multimodal fusion
  - modality failure
  - BEV
  - 3D detection
date: 2026-05-08
content_hash: 2566235c82e64a18
---

# SiMO: Single-Modality-Operable Multimodal Collaborative Perception

**Conference**: ICLR 2026
**arXiv**: [2603.08240](https://arxiv.org/abs/2603.08240)
**Code**: [dempsey-wen/SiMO](https://github.com/dempsey-wen/SiMO)
**Area**: Collaborative Perception / Multimodal Fusion / Autonomous Driving
**Keywords**: collaborative perception, multimodal fusion, modality failure, BEV, 3D detection

## TL;DR

This paper proposes SiMO, a framework that introduces the LAMMA fusion module and PAFR training strategy to achieve, for the first time in multi-agent collaborative perception, a multimodal perception system that remains operational under arbitrary modality absence—particularly when LiDAR fails and only cameras are available. The design is analogous to a parallel circuit: the system functions as long as at least one pathway is active.

## Background & Motivation

**Multi-Agent Collaborative Perception (MACP)** extends the perceptual range and mitigates occlusion by sharing features across vehicles. However, existing multimodal methods behave like series circuits: the failure of any single sensor—especially LiDAR—causes complete system breakdown.

**Root Cause of Modality Failure**: Existing fusion methods (concat / CNN / Transformer) create a mismatch between the pre- and post-fusion feature spaces. When a modality is absent, the unfused unimodal features become incompatible with the downstream task heads designed for fused representations, causing system collapse.

**Increased Complexity in Collaborative Settings**: Unlike single-agent scenarios requiring only local alignment, MACP demands that features transmitted across agents (e.g., an ego vehicle using LiDAR while a neighbor uses only camera) reside in a unified semantic space for effective cross-agent interaction. Existing single-agent robustness methods cannot guarantee this cross-agent semantic consistency.

**Overlooked Modality Competition**: During joint multimodal training, modalities with higher information density for the target task (e.g., LiDAR for 3D detection) converge faster and dominate the optimization process, suppressing the adequate training of weaker modality branches (camera), which consequently cannot operate independently.

**Limitations of Prior Work**: Methods such as BM2CP, BEVFusion, and CoBEVFusion focus solely on improving accuracy through multimodal fusion while ignoring the standalone usability of the camera branch under LiDAR failure. MetaBEV and UniBEV explore modality robustness only in single-agent settings and do not generalize to multi-agent scenarios.

**This work is the first to systematically address dynamic, heterogeneous modality absence in collaborative perception.**

## Method

### Overall Architecture

The core idea of SiMO is **align before fusing**, ensuring that pre- and post-fusion features reside in the same semantic space, thereby guaranteeing compatibility between unimodal and multimodal fused features with respect to downstream task heads.

Overall pipeline:
1. **Feature Extraction**: PointPillar extracts LiDAR BEV features; LSS (Lift-Splat-Shoot) extracts Camera BEV features.
2. **Feature Alignment**: Two independent 3-layer ConvNeXt aligners $g_L$ and $g_C$ map heterogeneous features into a unified semantic space.
3. **LAMMA Multimodal Fusion**: Length-Adaptive Multi-Modal Attention produces multimodal BEV features.
4. **Multi-Agent Fusion**: AttFusion or Pyramid Fusion aggregates features across vehicles.
5. **Task Head**: cls/reg/dir heads output 3D detection results.

### Key Designs 1: LAMMA (Length-Adaptive Multi-Modal Fusion)

The core contribution—a plug-and-play fusion module that adaptively handles inputs from a variable number of modalities:

- **Shared Weights**: Linear projections $W_Q, W_K, W_V$ for Q/K/V are shared across all modalities, ensuring consistent semantic processing.
- **Parallel Concatenated Attention**: Queries from both modalities are concatenated as $Q = [Z_A; Z_B]$, while Keys and Values remain separate; each modality undergoes one multi-head attention operation that encompasses both self-attention and cross-attention.
- **Additive Fusion for Spatial Consistency**: Attention outputs are split and summed to obtain modality-enhanced representations $Z_{fused\_m}$; the two modalities are then combined via element-wise addition to produce $Z_{mm}$, avoiding feature space shift.
- **Graceful Degradation**: When a modality is absent (e.g., $Z_A = 0$), the corresponding portion of the Query becomes zero, and LAMMA naturally degrades to a self-attention module without requiring explicit modality-detection logic, preserving semantic consistency by design.

### Key Designs 2: PAFR Training Strategy (Pretrain-Align-Fuse-RD)

A four-stage training procedure that fundamentally circumvents modality competition:

| Stage | Operation | Frozen Parameters |
|-------|-----------|-------------------|
| Step 1: Pretrain | Load pretrained feature extractors for each modality | All extractors frozen |
| Step 2: Align | Train LiDAR aligner to convergence and freeze; then train Camera aligner to convergence and freeze | Extractors + previously trained aligners |
| Step 3: Fuse | Train LAMMA fusion module using multimodal inputs | Extractors + aligners + task heads |
| Step 4: RD | Randomly drop one modality's features with 50% probability; fine-tune LAMMA for modality absence | All other modules |

Key insight: Modality competition stems from fundamental differences in task-relevant information density (LiDAR directly provides 3D geometry vs. Camera requiring 2D-to-3D inference) and is unavoidable in end-to-end joint training. The PAFR strategy bypasses competition by isolating the training of each branch, rather than attempting to balance them.

### Loss & Training

$$L(\hat{Y}, Y) = L_{Focal}(\hat{Y}_{cls}, Y_{cls}) + L_{SmoothL1}(\hat{Y}_{reg}, Y_{reg})$$

## Key Experimental Results

### Main Results: OPV2V-H 3D Detection (AP%)

| Method | Modality | AP@30 | AP@50 | AP@70 |
|--------|----------|-------|-------|-------|
| BM2CP | L+C | 91.69 | 91.45 | 86.87 |
| BM2CP | L only | 91.55 | 91.31 | 86.80 |
| BM2CP | **C only** | **0** | **0** | **0** |
| BEVFusion+RD | L+C | 95.18 | 94.21 | 81.09 |
| BEVFusion+RD | **C only** | **0** | **0** | **0** |
| UniBEV+RD | L+C | 93.33 | 91.71 | 70.75 |
| UniBEV+RD | **C only** | **1.93** | **0** | **0** |
| HEAL (Pyramid) | L | 98.22 | 98.00 | 96.16 |
| HEAL (Pyramid) | C | 68.45 | 60.48 | 39.07 |
| **SiMO-PF+RD** | **L+C** | **98.30** | **97.94** | **94.64** |
| **SiMO-PF+RD** | **L only** | **97.32** | **97.07** | **94.06** |
| **SiMO-PF+RD** | **C only** | **80.81** | **69.63** | **44.82** |

**Key Finding**: BM2CP, BEVFusion, and UniBEV completely fail upon LiDAR absence (Camera-only AP ≈ 0). SiMO-PF achieves AP@30 = 80.81% under camera-only conditions, outperforming HEAL's camera-only baseline by 12.36 points.

### Heterogeneous Modality Failure Experiment

| Mode | HEAL AP@50 | SiMO-PF AP@50 |
|------|-----------|---------------|
| L only | 0.98 | 0.97 |
| C only | 0.60 | 0.70 |
| C-ego (heterogeneous) | 0.82 | 0.85 |
| L-ego (heterogeneous) | 0.96 | 0.97 |

SiMO adapts to heterogeneous modality failure scenarios without requiring additional fine-tuning.

### Ablation Study

| Learning Strategy | RD | LAMMA | AP@70 (L+C / L / C) | Modality-Failure Adaptive? |
|------------------|----|-------|---------------------|---------------------------|
| ✗ | ✗ | ✗ | 0.94 / 0.01 / 0 | ✗ |
| ✗ | ✔ | ✗ | 0.11 / 0 / 0 | ✗ |
| ✔ | ✗ | ✔ | 0.95 / 0.26 / 0 | ✗ |
| ✗ | ✔ | ✔ | 0.81 / 0.72 / 0 | ✗ |
| **✔** | **✔** | **✔** | **0.95 / 0.94 / 0.45** | **✔** |

All three components are indispensable: without PAFR, RD is detrimental; without RD, modality absence cannot be handled; without LAMMA, BEVFusion+RD still fails under camera-only conditions.

### Procrustes Analysis for Feature Alignment Verification

| Comparison | BEVFusion | Before LAMMA | After LAMMA |
|------------|-----------|--------------|-------------|
| cam vs lidar | 0.8645 | 0.6747 | **0.0472** |
| cam vs fused | 0.7297 | 0.3886 | **0.0215** |
| lidar vs fused | 0.5747 | 0.2773 | **0.0064** |

After LAMMA, inter-modal feature discrepancy is reduced from 0.67 to 0.05, empirically validating the high degree of feature space unification.

## Highlights & Insights

1. **Apt Parallel Circuit Analogy**: Designing the multimodal system as a parallel rather than series circuit—functional as long as one pathway is active—is a concise and practically motivated concept.
2. **Novel Understanding of Modality Competition**: Attributing modality competition to differences in task-relevant information density and circumventing it through isolated branch training—rather than attempting to balance gradients—offers greater determinism than existing gradient-control methods.
3. **Elegant Graceful Degradation in LAMMA**: Natural reduction to self-attention upon modality absence, with no auxiliary detection logic required, yields a structurally clean design.
4. **Plug-and-Play Compatibility**: LAMMA can be integrated into different collaborative perception frameworks (AttFusion / Pyramid Fusion) without modifying the original methods.
5. **Substantial Camera Branch Enhancement**: SiMO-PF outperforms HEAL's camera-only baseline by 12.36 / 9.15 / 5.75 (AP@30/50/70), indicating that prior frameworks significantly underutilize camera features.

## Limitations & Future Work

1. **Unimodal Performance Bounded by Extractor Capability**: In single-view camera scenarios (e.g., DAIR-V2X), limited depth estimation due to the absence of multi-view parallax prevents SiMO from overcoming the inherent physical information bottleneck.
2. **Multi-Stage Training Pipeline**: The four-stage PAFR procedure inevitably increases total training time.
3. **Lack of Smoothing in Additive Fusion**: Compared to the implicit smoothing afforded by CNN-based fusion, additive fusion is more sensitive to high-intensity sensor noise.
4. **Limited Experimental Datasets**: Main experiments are conducted on the simulated OPV2V-H dataset; real-world datasets (DAIR-V2X / V2XReal) are only briefly validated in the appendix.

## Related Work & Insights

- **Multimodal Collaborative Perception**: HM-ViT (pioneer in heterogeneous modality collaboration), HEAL (modality + model heterogeneity), BM2CP (bimodal fusion), CoBEVFusion
- **Single-Agent Modality Robustness**: CMT (first single-modality-operable design), MetaBEV (position dependency from CNN+Concat), UniBEV (unified architecture alignment)
- **Multimodal Balanced Learning**: Gradient Blending, OGM (gradient modulation), PMR, UMT
- **Foundational Components**: PointPillar (LiDAR BEV), LSS (Camera BEV), BEVFusion, ConvNeXt, Pyramid Fusion

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The problem formulation is clear and practically motivated (modality failure is inevitable in real-world driving). LAMMA exhibits an elegant design (shared weights + additive fusion + natural degradation), and the PAFR strategy reflects a deep understanding of modality competition. The ablation study thoroughly demonstrates that all three components are necessary. Points are deducted for the primary experiments remaining on a simulated dataset and for the increased engineering complexity introduced by the multi-stage training procedure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](../../CVPR2026/autonomous_driving/colc_communication-efficient_collaborative_perception_with_lidar_completion.md)
- [\[ICLR 2026\] x²-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)
- [\[NeurIPS 2025\] Layer-wise Modality Decomposition for Interpretable Multimodal Sensor Fusion](../../NeurIPS2025/autonomous_driving/layer-wise_modality_decomposition_for_interpretable_multimodal_sensor_fusion.md)
- [\[ICCV 2025\] INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception](../../ICCV2025/autonomous_driving/instinct_instance-level_interaction_architecture_for_query-based_collaborative_p.md)
- [\[CVPR 2026\] Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception](../../CVPR2026/autonomous_driving/learning_mutual_view_information_graph_for_adaptive_adversarial_collaborative_pe.md)

</div>

<!-- RELATED:END -->
