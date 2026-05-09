---
title: >-
  [Paper Note] DeDelayed: Deleting Remote Inference Delay via On-Device Correction
description: >-
  [CVPR 2026][Segmentation][Collaborative inference] DeDelayed is an edge-cloud collaborative inference framework that combines a lightweight on-device image model with a latency-aware cloud-side temporal prediction video model. By training the network with temporally predictive objectives to compensate for communication delay, DeDelayed achieves gains of 6.4 mIoU over local-only inference and 9.8 mIoU over remote-only inference under 100 ms latency.
tags:
  - CVPR 2026
  - Segmentation
  - Collaborative inference
  - real-time video segmentation
  - latency compensation
  - temporal prediction
  - edge-cloud collaboration
date: 2026-05-08
content_hash: 0d64ea7878037cb0
---

# DeDelayed: Deleting Remote Inference Delay via On-Device Correction

**Conference**: CVPR 2026
**arXiv**: [2510.13714](https://arxiv.org/abs/2510.13714)
**Code**: [github.com/InterDigitalInc/dedelayed](https://github.com/InterDigitalInc/dedelayed)
**Area**: Image Segmentation
**Keywords**: Collaborative inference, real-time video segmentation, latency compensation, temporal prediction, edge-cloud collaboration

## TL;DR
DeDelayed is an edge-cloud collaborative inference framework that combines a lightweight on-device image model with a latency-aware cloud-side temporal prediction video model. By training the network with temporally predictive objectives to compensate for communication delay, DeDelayed achieves gains of 6.4 mIoU over local-only inference and 9.8 mIoU over remote-only inference under 100 ms latency.

## Background & Motivation
**State of the Field**: The most powerful video understanding models are computationally prohibitive for resource-constrained edge devices, while offloading inference to the cloud introduces communication latency that renders predictions stale.

**Limitations of Prior Work**: (1) Existing segmentation offloading methods dedicate all local compute to a single inference pipeline, providing no fallback when the cloud is unavailable; (2) the impact of latency on prediction accuracy is not addressed; (3) computational cost is controlled by reducing spatiotemporal resolution.

**Root Cause**: Cloud models offer high accuracy but incur latency, whereas local models run in real time but at lower accuracy — how can the advantages of both be combined?

**Paper Goals**: Design a real-time inference system that jointly exploits delayed high-quality remote features and instantaneous low-resolution local features.

**Starting Point**: Train the remote model to predict features for **future frames**, so that delayed remote outputs remain useful upon arrival.

**Core Idea**: $\hat{y}_t = f_{\text{local}}(x_t, z_{t-\tau})$, where the local model processes the current frame and the remote model predicts future-frame features; the two are fused via element-wise addition.

## Method

### Overall Architecture
- **Remote model**: A 2D ViT (EfficientViT-L1) extracts per-frame features → temporal concatenation of $K=4$ frames → delay embedding injection → 3D ViT encoder → adaptive pooling + channel bottleneck (DR-AE) → downlink transmission.
- **Local model**: CNN2D + CoAt2D processes low-resolution current frames; segmentation is performed after fusing remote features.
- **Fusion**: Remote features are spatially pooled to align with local resolution and then **element-wise added** to the CNN2D output.

### Key Designs
1. **Temporally Predictive Training**: During training, an artificial delay of $D$ frames (uniformly sampled from 0–5) is applied to the remote model's input, while supervision is provided by the labels of the corresponding future frame. A learnable *delay embedding* (analogous to positional encoding) is introduced so that the model's behavior adapts to the actual delay at inference time. **Design Motivation**: Since network latency is unavoidable, the model is trained to predict future states and thereby proactively compensate for delay.

2. **Full Integration with Local Fallback**: Remote features are fused into an intermediate layer of the local model via element-wise addition — if the remote output is absent, the local model operates independently. **Design Motivation**: Hard real-time applications require a complete local fallback. The element-wise addition ensures that when the remote signal is zero, the system behaves identically to pure local inference.

3. **Mixed-Resolution Inference**: The local model processes low-resolution frames (704×480), while the remote model processes high-resolution frames (720p). **Design Motivation**: Running any model at capture resolution on an edge device is impractical, but cloud GPUs can handle high-resolution video. The remote branch provides semantic understanding, while the local branch provides spatial localization.

### Loss & Training
- Multi-stage training: remote and local models are pretrained separately on ImageNet → Cityscapes → BDD100K, followed by joint fine-tuning.
- Joint training uses per-pixel cross-entropy loss, the Adan optimizer, and a warmup-stable-decay learning rate schedule.
- Training delay $\tau$ is uniformly sampled from 0–5 frames (0–167 ms at 30 fps).

## Key Experimental Results

### Main Results (BDD100K Semantic Segmentation mIoU)

| Inference Configuration | 0 ms | 33 ms | 67 ms | 100 ms | 167 ms |
|------------------------|------|-------|-------|--------|--------|
| Local only | 0.601 | 0.601 | 0.601 | 0.601 | 0.601 |
| Remote image | 0.655 | 0.616 | 0.567 | 0.530 | 0.525 |
| Remote predictive | 0.655 | 0.649 | 0.644 | 0.637 | 0.624 |
| **DeDelayed** | **0.670** | **0.668** | **0.666** | **0.665** | **0.668** |

### Ablation Study

| Configuration | mIoU @167 ms | Notes |
|---------------|-------------|-------|
| Local only | 0.601 | Unaffected by latency but low accuracy |
| Remote image | 0.525 | Latency severely degrades accuracy |
| Remote predictive | 0.624 | Temporal prediction substantially mitigates degradation |
| **DeDelayed (full)** | **0.668** | Latency effect nearly eliminated |

### Key Findings
- Conventional remote inference degrades below local-only performance when latency exceeds 67 ms.
- DeDelayed surpasses local-only inference by 6.7 mIoU at 167 ms latency, equivalent to using a model that is 10× larger.
- Activation map visualizations reveal that the remote branch provides accurate object classification while the local branch provides precise spatial localization.
- The delay embedding enables a single model to adapt dynamically to latencies ranging from 0 to 167 ms.

## Highlights & Insights
- **Fallback-first design**: Remote information serves as an auxiliary signal rather than a required dependency, ensuring hard real-time safety.
- **Delay embedding ≈ learnable motion compensation**: By conditioning on the delay magnitude, the model learns motion prediction of varying degrees.
- **Complementarity of mixed resolution**: The high-resolution remote branch recognizes small distant objects (e.g., far-away pedestrians), while the low-resolution local branch provides precise spatial calibration.
- Element-wise addition is simple yet well-defined in behavior, degrading gracefully when the remote signal is absent.

## Limitations & Future Work
- Validation is limited to segmentation; detection and other dense prediction tasks are not evaluated.
- Pseudo-labels are used for training due to the lack of per-frame annotations in BDD100K; performance with ground-truth labels may be higher.
- Distortion introduced by uplink video compression is not explicitly modeled.
- High-latency scenarios beyond 167 ms are not evaluated.
- Multiple remote models and hierarchical feature fusion are not explored.
- The feasibility of the local model on ultra-low-power devices (<5 W) remains to be verified.
- The compression efficiency of DR-AE under downlink bandwidth constraints warrants further optimization.
- Heterogeneous sensor fusion scenarios (e.g., LiDAR + camera) are not covered.

## Related Work & Insights
- **Distinction from split computing**: FCM dedicates all local compute to the uplink pipeline with no fallback capability.
- **Comparison with Knowledge Boosting**: The latter requires training a separate model for each fixed latency value.
- The delay embedding design is generalizable to other asynchronous information fusion scenarios.
- Adaptive Model Streaming updates model weights in a streaming fashion, which is orthogonal to DeDelayed's feature-level fusion.

## Technical Details
- **Remote model**: EfficientViT-L1 (2D ViT, patch 8×8) → temporal concatenation of $K=4$ frames → 3D ViT + delay embedding.
- **Local model**: CNN2D + CoAt2D, maximum resolution 704×480.
- **DR-AE**: Adaptive spatial pooling + channel bottleneck to match local resolution and compress downlink bandwidth.
- **Uplink compression**: 720p at 30 fps transmitted at 1–10 Mbps (5G cellular network).
- **Target latency**: 33 ms for both local and remote branches (single frame at 30 fps).
- **Pseudo-labels**: DepthAnything for the validation set; EoMT for the training set.
- **Optimizer**: Adan + warmup-stable-decay + gradient clipping + LLRD.
- **Key insight**: Remote model activation maps show accurate object discrimination and classification (e.g., distant pedestrians); the local model provides precise spatial correction.
- **Dataset**: BDD100K contains 70K training videos of urban driving scenes at 30 fps.
- **Evaluation**: Cityscapes 19-class semantic segmentation protocol.
- **5G suitability**: Design parameters are matched to 5G cellular network uplink capacity (1–10 Mbps).

## Rating
- Novelty: ⭐⭐⭐⭐ — The edge-cloud collaborative framework combining temporal prediction with delay embeddings is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparison across multiple configurations, though evaluated on only one dataset and one task.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem formulation is clear and system diagrams are intuitive.
- Value: ⭐⭐⭐⭐⭐ — Directly targets practical deployment scenarios with high engineering value.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] A²LC: Active and Automated Label Correction for Semantic Segmentation](../../AAAI2026/segmentation/a2lc_active_and_automated_label_correction_for_semantic_segm.md)
- [\[CVPR 2026\] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semanticguided_modalityaware_segmentation_for.md)
- [\[CVPR 2026\] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation](task-oriented_data_synthesis_and_control-rectify_sampling_for_remote_sensing_sem.md)
- [\[CVPR 2026\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportionaware_dynamic_adaptive_sali.md)
- [\[NeurIPS 2025\] Self-supervised Synthetic Pretraining for Inference of Stellar Mass Embedded in Dense Gas](../../NeurIPS2025/segmentation/self-supervised_synthetic_pretraining_for_inference_of_stellar_mass_embedded_in_.md)

<!-- RELATED:END -->
