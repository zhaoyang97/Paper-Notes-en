---
title: >-
  [Paper Note] SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection
description: >-
  [AAAI2026][Object Detection][Multi-modal remote sensing] SM3Det introduces the M2Det task for remote sensing (multi-modal datasets + multi-task object detection), employing a grid-level sparse MoE backbone and a Dynamic Sub-module Optimization (DSO) mechanism to handle SAR/optical/infrared modalities with both horizontal and oriented bounding box detection in a single unified model, substantially outperforming three independently trained modality-specific models combined.
tags:
  - AAAI2026
  - Object Detection
  - Multi-modal remote sensing
  - sparse MoE
  - dynamic learning rate optimization
  - unified model
date: 2026-05-08
content_hash: 6bf9d8efa2aba38a
---

# SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection

**Conference**: AAAI2026  
**arXiv**: [2412.20665](https://arxiv.org/abs/2412.20665)  
**Code**: [github.com/zcablii/SM3Det](https://github.com/zcablii/SM3Det)  
**Area**: Object Detection  
**Keywords**: Multi-modal remote sensing, object detection, sparse MoE, dynamic learning rate optimization, unified model  

## TL;DR

SM3Det introduces the M2Det task for remote sensing (multi-modal datasets + multi-task object detection), employing a grid-level sparse MoE backbone and a Dynamic Sub-module Optimization (DSO) mechanism to handle SAR/optical/infrared modalities with both horizontal and oriented bounding box detection in a single unified model, substantially outperforming three independently trained modality-specific models combined.

## Background & Motivation

1. **Growing abundance of multi-modal data**: Remote sensing platforms (UAVs/satellites) are typically equipped with multiple sensors (SAR/optical/infrared), yet detection models are conventionally trained on a single modality and a single dataset.
2. **Cross-modal knowledge neglected**: Independent per-modality training fails to exploit shared knowledge across modalities, such as common object shape and scale characteristics.
3. **Prior multi-source detection requires alignment**: Existing multi-source detection methods heavily rely on scarce and inflexible spatially aligned image pairs and registration algorithms, limiting practical utility.
4. **Crowded representation space**: Fitting a dense model with a single shared parameter set to multi-modal, multi-task data leads to a crowded representation space where one parameter set struggles to capture the divergent distributions of different modalities.
5. **Optimization inconsistency**: Varying learning difficulties across modalities and tasks cause asynchronous convergence rates and conflicting gradient directions, mutually interfering with loss convergence.
6. **Low-altitude economy applications**: Scenarios involving flying cars, UAVs, and satellites urgently require a unified detection capability that handles multiple modalities simultaneously, reducing on-device model maintenance costs.

## Method

### Task Definition: M2Det

The authors formally define the **M2Det** task for remote sensing — using a unified model to detect objects in images from arbitrary sensor modalities while simultaneously handling both horizontal bounding box (HBB) and oriented bounding box (OBB) detection formats.

### Overall Architecture

The framework follows a classic multi-task learning design: a shared backbone with lightweight task-specific detection heads. The SA dataset uses a GFL head (HBB), while DOTA and DroneVehicle use an O-RCNN head (OBB).

### Grid-Level Sparse MoE Backbone

**Core innovation**: A plug-and-play grid-level sparse MoE architecture is introduced into the backbone.

- Unlike prior **image-level routing** (routing an entire image to a single expert), SM3Det performs expert selection at the spatial grid granularity of feature maps.
- For CNN backbones (e.g., ConvNeXt), the $1 \times 1$ convolutional layers are replaced with MoE; for Transformer backbones, MoE is integrated into the FFN.
- Gating function: $G(x_{ij}) = \text{TOP}_k(\text{Softmax}(\frac{E^T W x_{ij}}{\tau \|Wx_{ij}\| \|E\|}))$
- Final output is the weighted sum of top-$k$ experts: $f_{MoE}(x_{ij}) = \sum_{n=1}^N G_n(x_{ij}) \cdot Conv_n^{1\times 1}(x_{ij})$
- **Initialization strategy**: All experts are initialized by copying the pretrained $1 \times 1$ convolution weights, ensuring uniform expert selection at the start of training.

**Advantage**: Shared experts learn cross-modal common knowledge (e.g., object shape and scale), while specialized experts capture modality-specific features (e.g., SAR scattering characteristics).

### Dynamic Sub-module Optimization (DSO)

DSO consists of two components that independently regulate the learning rates of the detection heads and the backbone:

**Detection head learning rate adjustment** (balancing convergence rates across tasks):
- Maintains an EMA history $his\_L_i^t$ of each task's loss.
- Computes a convergence rate inverse metric: $w_i^t = his\_L_i^t / cur\_L_i^t$
- Re-weights head learning rates via temperature-scaled Softmax: $\lambda_i^t = T \cdot e^{w_i^t/\theta} / \sum_k e^{w_i^k/\theta}$
- Effect: faster-converging tasks receive reduced learning rates while slower ones are boosted, maintaining synchronous convergence.

**Backbone learning rate adjustment** (ensuring optimization direction consistency):
- Computes the KL divergence between the current and historical loss distributions: $C = 1 - D_{KL}(P(cur\_L) \| P(his\_L))$
- A high consistency score $C$ indicates a stable current batch, permitting larger update steps; a low $C$ signals imbalanced learning difficulty across tasks, calling for cautious updates.
- Modulated via Sigmoid: $\gamma_i = 2 \cdot \text{Sigmoid}((C-b) \cdot \tau)$

### Benchmark Dataset: SOI-Det

SARDet-100K (SAR/HBB) + DOTA-v1.0 (optical/OBB) + DroneVehicle (infrared/OBB) are merged with a sampling ratio of 2:1:1.

## Key Experimental Results

### Table 1: Main Results on SOI-Det (ConvNeXt-T Backbone)

| Method | FLOPs | Params | Overall mAP | @50 | @75 |
|--------|-------|--------|-------------|-----|-----|
| 3 independent models | 403G | 126M | 48.23 | 79.39 | 51.26 |
| Naive joint training | 403G | 66M | 47.05 | 77.56 | 50.11 |
| DA + ConvNeXt-T | 403G | 66M | 48.37 | 79.76 | 51.66 |
| UniDet (Partitioned) | 403G | 66M | 48.47 | 79.55 | 52.01 |
| Uncertainty loss | 403G | 66M | 48.79 | 79.99 | 52.50 |
| SM3Det (DSO only) | 403G | 66M | 49.40 | 80.19 | 52.93 |
| **SM3Det (Full)** | 487G | 178M | **50.20** | **80.68** | **53.79** |

- The full SM3Det surpasses the three independent models by **+1.97 mAP**.
- The lightweight DSO-only variant (no MoE, same parameter count) already outperforms all prior SOTA methods.

### Table 5: Parameter Efficiency Comparison (Varying Backbone Scale)

| Configuration | Params | mAP |
|---------------|--------|-----|
| 3 models (Small) | 192M | 49.17 |
| SM3Det (Tiny) | 178M | **50.20** |
| 3 models (Base) | 309M | 50.18 |
| SM3Det (Small) | 275M | **50.28** |
| 3 models (Large) | 636M | 50.50 |
| SM3Det (Base) | 459M | **51.33** |
| SM3Det (Large) | 770M | **52.16** |

- SM3Det-Tiny (178M) outperforms 3 models-Small (192M) while using 7.3% fewer parameters.
- SM3Det-Base (459M) outperforms 3 models-Large (636M) with 27.8% fewer parameters.

### Ablation Study

- **Number of experts and top-k**: 8 experts with top-2 selection is the optimal configuration, balancing performance and computational efficiency.
- **Grid-level vs. image-level MoE**: Grid-level routing (50.20) significantly outperforms image-level routing (48.25), confirming that spatially fine-grained routing is critical for detection tasks.
- **MoE layer placement**: Inserting MoE into even-numbered layers of the last three stages yields the best results (49.53); applying it to all layers leads to a slight drop (49.47).
- **Necessity of DSO**: Removing DSO reduces mAP from 50.20 to 49.47.
- **DSO hyperparameter robustness**: Performance is largely insensitive to variations in the bias parameter $b$, demonstrating method robustness.

## Highlights & Insights

1. **New task definition**: The paper is the first to systematically define the M2Det task for remote sensing, filling a research gap in unified multi-modal detection.
2. **Elegant grid-level MoE design**: Unlike coarse image-level routing, grid-level experts can perceive spatially local patterns and simultaneously learn shared and modality-specific representations.
3. **Distinctive DSO mechanism**: Unlike GradNorm and similar methods that modify loss weights or gradients, DSO directly regulates sub-module learning rates, enabling finer-grained and more efficient control.
4. **Better performance with fewer parameters**: A single SM3Det model surpasses multiple independent models combined while using fewer parameters.
5. **Strong generalizability**: The method is effective across diverse backbones (ConvNeXt, VAN, LSKNet, PVT-v2) and both single-stage and two-stage detectors.
6. **Insightful expert activation visualization**: SAR employs a distinct set of experts, while RGB and infrared share more experts — consistent with the known characteristics of these modalities.

## Limitations & Future Work

1. **Absence of multispectral modality**: Multispectral imaging experiments are not included due to the scarcity of large-scale multispectral detection datasets.
2. **Restricted to remote sensing**: Although the method is transferable to multi-modal scenarios such as medical imaging and autonomous driving, this has not been empirically validated.
3. **Additional parameters from MoE**: The full SM3Det (178M) has nearly three times the parameters of the baseline (66M); while still fewer than the three independent models (126M), deployment trade-offs must be considered.
4. **Training resource requirements**: Training requires 8 RTX 3090 GPUs, representing a non-trivial computational cost.
5. **Fixed dataset sampling strategy**: The 2:1:1 sampling ratio is not ablated in depth and may not be optimal.

## Related Work & Insights

### vs. UniDet (Unified Label Space Multi-Dataset Detection)
UniDet employs partitioned detection heads and a unified label space for multi-dataset training, which is effective for general object detection (optical concept datasets). However, when facing multi-modal remote sensing imagery (where SAR/optical/infrared exhibit fundamentally different pattern concepts), UniDet (48.47 mAP) offers only marginal improvement over naive joint training (47.05). SM3Det (50.20) addresses the crowded representation space via MoE and the optimization inconsistency via DSO, yielding a clear advantage (+1.73 mAP).

### vs. DA Networks (Domain-Specific SE Attention)
DA networks use SE layers as domain-specific attention mechanisms and represent a prior approach to multi-dataset detection. However, DA's domain-specific mechanism is hard-coded image-level routing, incapable of flexible expert selection at the spatial feature level. On SOI-Det, DA (48.37) even underperforms uncertainty loss (48.79) and falls well short of SM3Det (50.20).

### vs. GradNorm (Gradient-Balanced Multi-Task Learning)
GradNorm balances learning by adjusting gradient magnitudes across tasks. SM3Det's DSO instead directly regulates sub-module learning rates — using loss ratios to balance convergence rates for detection heads and KL divergence consistency scores to modulate backbone update magnitudes. This two-level strategy ensures synchronous convergence across tasks while preventing shared weights from being overly biased by hard samples from any single task.

## Rating
- Novelty: ⭐⭐⭐⭐ — Triple innovation: M2Det task definition + grid-level MoE + DSO
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-backbone, multi-detector, comprehensive ablations, and visualization analyses
- Writing Quality: ⭐⭐⭐⭐ — Clear problem definition, well-motivated methodology, logically organized experiments
- Value: ⭐⭐⭐⭐ — Pioneering contribution to unified multi-modal remote sensing detection, with transferable methodology to other multi-modal scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](../../CVPR2026/object_detection/fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)
- [\[ICCV 2025\] OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images](../../ICCV2025/object_detection/openrsd_towards_open-prompts_for_object_detection_in_remote_sensing_images.md)
- [\[CVPR 2026\] Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](../../CVPR2026/object_detection/learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)
- [\[CVPR 2026\] Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](../../CVPR2026/object_detection/small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)
- [\[AAAI 2026\] AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios](aerialmind_towards_referring_multi-object_tracking_in_uav_sc.md)

</div>

<!-- RELATED:END -->
