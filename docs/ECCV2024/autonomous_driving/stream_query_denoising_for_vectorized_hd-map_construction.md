---
title: >-
  [Paper Note] Stream Query Denoising for Vectorized HD-Map Construction
description: >-
  [ECCV 2024][Autonomous Driving][HD map construction] This paper proposes the Stream Query Denoising (SQD) strategy. By adding noise to the ground truth (GT) of the previous frame and training the network to reconstruct the current frame's GT, temporal consistency modeling in streaming HD map construction is enhanced. This approach consistently outperforms StreamMapNet on nuScenes and Argoverse2.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "HD map construction"
  - "Temporal modeling"
  - "Query denoising"
  - "Streaming inference"
  - "Vectorized map"
date: 2026-05-08
content_hash: 703b06faf1c54023
---

# Stream Query Denoising for Vectorized HD-Map Construction

**Conference**: ECCV 2024  
**arXiv**: [2401.09112](https://arxiv.org/abs/2401.09112)  
**Code**: The paper mentions code will be available soon  
**Area**: Image Restoration  
**Keywords**: HD map construction, Temporal modeling, Query denoising, Streaming inference, Vectorized map

## TL;DR

This paper proposes the Stream Query Denoising (SQD) strategy. By adding noise to the ground truth (GT) of the previous frame and training the network to reconstruct the current frame's GT, temporal consistency modeling in streaming HD map construction is enhanced. This approach consistently outperforms StreamMapNet on nuScenes and Argoverse2.

## Background & Motivation

Vectorized HD map construction is evolving from single-frame detection to temporal streaming inference, with StreamMapNet being a representative work. However, directly transferring the streaming query propagation paradigm from object detection to map construction faces unique challenges:

**Curves differ from boxes**: In object detection, objects mainly undergo rigid motion (translation and rotation) between frames, making temporal box prediction relatively simple. However, road lines can grow or be truncated across frames, requiring each point on the line to learn a different offset. Such complex variations are difficult to model explicitly.

**Difficulties in streaming training**: Experiments show that directly incorporating streaming strategies actually leads to a 0.7 mAP performance drop—the network struggles to learn to handle continuous variations of curves across different frames.

**Query Denoising is not directly applicable to curves**: DN-DETR is designed for bounding boxes, while noise strategies for curves have not yet been explored.

**Core Observation**: Adding noise to the previous frame's GT to simulate the prediction behavior of stream queries, and training the network to reconstruct the current frame's GT from this noise, can effectively facilitate temporal consistency learning.

## Method

### Overall Architecture

SQD-MapNet = StreamMapNet + SQD strategy. The overall pipeline is as follows:

1. Multi-view images $\rightarrow$ backbone + FPN $\rightarrow$ BEVFormer to obtain BEV features
2. Polyline Decoder uses learnable queries to extract map elements
3. Temporal query propagation: Top-$k$ high-scoring queries are propagated to the next frame after coordinate transformation
4. **SQD (Training only)**: Noise is added to the previous frame's GT to generate denoising queries, which are sent into the decoder alongside normal queries, supervised by the current frame's GT on the denoising output.

SQD is removed during inference, incurring no extra inference overhead.

### Key Designs

1. **Normal Query Denoising — Curve Noise Strategy**: For the first time, curves are unified into a minimum bounding rectangle representation $(x, y, w, h)$, based on which three types of noise are designed:

    - **Box Shifting (Line Translation)**: Shifting the center of the rectangle by $(\Delta x, \Delta y)$, constrained by $|\Delta x| < \frac{\lambda_1 w}{2}$, while keeping the relative positions of points on the curve within the rectangle unchanged.
    - **Box Scaling (Rotation and Scale Transformation)**: Scaling the width and height of the rectangle, $h' \in [(1-\lambda_3)h, (1+\lambda_3)h]$, which simultaneously achieves rotation and length variations.
    - Noise query generation: Class $\rightarrow$ learnable embedding $C_q$; point coordinates $\rightarrow$ positional encoding $\rightarrow$ MLP to fuse into positional embedding $Pos_q$; final $Q_{\text{denoise}} = \text{MLP}^{(\text{fuse})}(\text{Concat}(C_q, Pos_q))$.

2. **Adaptive Temporal Matching (ATM)**: Establishes a one-to-one correspondence between the previous frame's GT and the current frame's GT. This is achieved by transforming the previous frame's GT to the current frame's coordinate system using the ego-motion matrix, and then computing the bidirectional Chamfer Distance (CD). An adaptive threshold is set based on the scale of each curve itself:
    $$\delta = \alpha \frac{w + h}{2}$$
    A match is established only if the CD is smaller than the threshold. This avoids the limitation of a fixed threshold that ignores the scale differences among curves.

3. **Dynamic Query Noising**: The matched curves already possess a natural deviation due to ego-motion, making it unreasonable to add an equal amount of random noise. Therefore, the noise is dynamically decayed based on Chamfer Distance:
    $$R_{decay} = 1 - \frac{D}{\gamma \cdot \frac{\delta}{\alpha}}$$
    A larger natural deviation ($D$ is larger) leads to smaller additional random noise. The final noisy instance is formulated as: $B_{ins} = \{x,y,w,h\} + \eta \cdot R_{decay}$.

### Loss & Training

Total training loss = Map loss + Denoising loss:

- Map loss: $\mathcal{L}_{map} = \lambda_1 \mathcal{L}_{Focal} + \lambda_2 \mathcal{L}_{line} + \lambda_3 \mathcal{L}_{trans}$
- Denoising loss: $\mathcal{L}_{denoise} = \lambda_4 \mathcal{L}_{Focal}^{DN} + \lambda_5 \mathcal{L}_{line}^{DN}$

Training Strategy:
- Normal Query Denoising is used in the single-frame training phase, and Stream Query Denoising is used in the streaming training phase.
- nuScenes is trained for 24 epochs, and Argoverse2 is trained for 30 epochs.
- The default backbone is ResNet-50, and a single-layer BEVFormer encoder is used for BEV feature extraction.
- 8$\times$ V100 GPUs, with a batch size of 32.

## Key Experimental Results

### Main Results (nuScenes val)

| Method | Backbone | Perception Range | AP_ped | AP_div | AP_bound | mAP |
|------|----------|---------|--------|--------|----------|-----|
| MapTR | R50 | 60$\times$30m | 46.3 | 51.5 | 53.1 | 50.3 |
| StreamMapNet | R50 | 60$\times$30m | 60.4 | 61.9 | 58.9 | 60.4 |
| **SQD-MapNet** | **R50** | **60$\times$30m** | **63.0** | **62.5** | **63.3** | **63.9** |
| StreamMapNet | R50 | 100$\times$50m | 62.9 | 63.1 | 55.8 | 60.6 |
| **SQD-MapNet** | **R50** | **100$\times$50m** | **67.0** | **65.5** | **59.5** | **64.0** |
| **SQD-MapNet** | **V2-99** | **60$\times$30m** | **74.2** | **72.3** | **75.6** | **74.0** |

### Ablation Study

| Configuration | mAP | Description |
|------|-----|------|
| Single-frame baseline | 59.9 | Without temporal modeling |
| + Temporal Stream | 59.2 (-0.7) | Direct addition of streaming leads to performance drop |
| + Dynamic Query Noising | 62.9 (+3.7) | Core contribution of SQD |
| + Adaptive Temporal Matching | 63.9 (+1.0) | ATM brings further improvement |

| Denoising Strategy | mAP |
|----------|-----|
| No Denoising | 59.2 |
| Normal Query Denoising | 62.7 |
| Stream Query Denoising | 63.9 |

### Key Findings

- Directly using streaming temporal propagation degrades performance (-0.7), validating that temporal modeling for curves is indeed challenging.
- The SQD strategy contributes +4.7 mAP (from 59.2 to 63.9), serving as the core of the performance boost.
- The adaptive matching threshold achieves optimal performance (63.5) when $\alpha=0.1$, whereas the fixed threshold only yields a maximum of 62.8.
- The optimal balance is achieved with a noise decay rate of $\gamma=0.2$ (63.9).
- Normal DN and Stream DN exhibit complementary effects: Stream DN partially incorporates the effect of Normal DN.

## Highlights & Insights

1. **First to introduce Query Denoising to HD map construction**: Identifying the fundamental differences between curves and bounding boxes (partial growth/truncation), a unified minimum bounding rectangle + box shifting/scaling noise scheme is proposed.
2. **Stream Query Denoising kills two birds with one stone**: By adding noise to the previous frame's GT, it simulates the behavior of stream queries, simultaneously learning temporal query modeling and general query denoising.
3. **Adaptive design throughout**: Instead of a one-size-fits-all approach, each curve determines its matching threshold (ATM) and noise decay rate (Dynamic QN) according to its own scale.
4. **Training-time enhancement, zero inference overhead**: SQD is used only during training and does not affect inference efficiency.

## Limitations & Future Work

1. Validated only on a single streaming baseline (StreamMapNet), lacking verification on more streaming methods.
2. Whether the noise strategies should differ for various map elements (pedestrian crossings, lane lines, road boundaries) has not been explored in depth.
3. The matching from the previous frame's GT to the current frame is calculated offline; the overhead of real-time computation has not been analyzed yet.
4. Future work can explore extending SQD to streaming methods for 3D object detection (e.g., StreamPETR).

## Related Work & Insights

- The query denoising concepts in DN-DETR/DINO inspired this work, but the differences between curves and boxes necessitate a completely new design.
- The temporal query propagation mechanism of StreamMapNet serves as the foundation of this paper.
- The adaptive matching philosophy of ATM can be extended to other tasks requiring cross-frame association (e.g., video segmentation).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Extending query denoising from boxes to curves, with an ingenious design for stream query denoising.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated on two datasets (nuScenes + Argoverse2), with ablation studies covering all key components and hyperparameter sensitivity.
- **Writing Quality**: ⭐⭐⭐⭐ — The motivation analysis is clear, and Figure 2 (temporal curve variation) illustrates the problem very well.
- **Value**: ⭐⭐⭐⭐ — The SQD strategy is plug-and-play, effectively advancing the field of streaming HD map construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MapDistill: Boosting Efficient Camera-based HD Map Construction via Camera-LiDAR Fusion Model Distillation](mapdistill_boosting_efficient_camera-based_hd_map_construction_via_camera-lidar_.md)
- [\[CVPR 2025\] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](../../CVPR2025/autonomous_driving/mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [\[ICML 2025\] SafeMap: Robust HD Map Construction from Incomplete Observations](../../ICML2025/autonomous_driving/safemap_robust_hd_map_construction_from_incomplete_observations.md)
- [\[CVPR 2025\] Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction](../../CVPR2025/autonomous_driving/uncertainty-instructed_structure_injection_for_generalizable_hd_map_construction.md)
- [\[ECCV 2024\] Enhancing Vectorized Map Perception with Historical Rasterized Maps](enhancing_vectorized_map_perception_with_historical_rasterized_maps.md)

</div>

<!-- RELATED:END -->
