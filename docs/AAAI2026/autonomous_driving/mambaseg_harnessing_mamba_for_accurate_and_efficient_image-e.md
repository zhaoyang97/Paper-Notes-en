---
title: >-
  [Paper Note] MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation
description: >-
  [AAAI 2026][Autonomous Driving][Event Camera] MambaSeg is proposed, employing a dual-branch parallel Mamba encoder to process RGB images and event streams respectively, with a Dual-Dimension Interaction Module (DDIM) for fine-grained cross-modal fusion along both spatial and temporal dimensions. It achieves state-of-the-art performance of 77.56%/75.10% mIoU on DDD17 and DSEC with only 25.44M parameters, offering substantially better efficiency than Transformer-based approaches.
tags:
  - AAAI 2026
  - Autonomous Driving
  - Event Camera
  - Multimodal Fusion
  - Mamba/SSM
  - Semantic Segmentation
  - Spatiotemporal Interaction
date: 2026-05-08
content_hash: 5aff73593f2ca0dd
---

# MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation

**Conference**: AAAI 2026
**arXiv**: [2512.24243](https://arxiv.org/abs/2512.24243)
**Code**: [https://github.com/CQU-UISC/MambaSeg](https://github.com/CQU-UISC/MambaSeg)
**Area**: Autonomous Driving
**Keywords**: Event Camera, Multimodal Fusion, Mamba/SSM, Semantic Segmentation, Spatiotemporal Interaction

## TL;DR
MambaSeg is proposed, employing a dual-branch parallel Mamba encoder to process RGB images and event streams respectively, with a Dual-Dimension Interaction Module (DDIM) for fine-grained cross-modal fusion along both spatial and temporal dimensions. It achieves state-of-the-art performance of 77.56%/75.10% mIoU on DDD17 and DSEC with only 25.44M parameters, offering substantially better efficiency than Transformer-based approaches.

## Background & Motivation
- **Limitations of frame cameras**: Conventional RGB semantic segmentation degrades severely under high-speed motion, low-light, and high dynamic range conditions due to motion blur and latency.
- **Complementarity of event cameras**: Event cameras offer microsecond-level temporal resolution and high dynamic range, but lack color and texture information, making them insufficient for dense prediction when used alone.
- **Issues with existing fusion methods**: (1) Transformer-based methods (CMX, EISNet) are effective but computationally expensive due to the quadratic complexity of self-attention; (2) Most methods perform only spatial fusion, neglecting the inherent temporal dynamics of event streams, leading to insufficient cross-modal alignment and semantic inconsistency.

## Core Problem
How to perform RGB-Event cross-modal fusion simultaneously along both spatial and temporal dimensions while maintaining low computational overhead, so as to reduce cross-modal ambiguity?

## Method

### Overall Architecture
A dual-branch architecture is employed: two parallel VMamba-T encoders (pretrained on ImageNet-1K) process the image and the Voxel Grid-formatted event stream respectively. VSS Blocks at four scales extract multi-scale features, and a DDIM module is embedded at each scale for cross-modal interaction. The fused features are fed back into their respective encoders for the next stage. A SegFormer MLP decoder is used to produce the final segmentation output.

Event stream preprocessing: Raw asynchronous events $(x_i, y_i, t_i, p_i)$ are accumulated into a Voxel Grid $E \in \mathbb{R}^{T \times H \times W}$ by partitioning into $T=10$ temporal bins.

### Key Designs
1. **CSIM (Cross-Spatial Interaction Module)**:

    - **Cross-modal spatial attention**: AvgPool and MaxPool are applied to event, image, and shallow fused features separately (6 spatial maps in total); the concatenated result is passed through two convolutional layers with sigmoid activation to generate three sets of spatial attention weights, which are cross-applied across modalities (event features are modulated by image attention weights and vice versa).
    - **SS2D spatial refinement**: Concatenated features are unfolded into four directional sequences, each processed by an independent S6 Block to capture multi-directional long-range dependencies before being reassembled into 2D feature maps.
    - **Modality-aware residual update**: Features are separated back into each modality, followed by spatial attention and residual connections to preserve modality-specific information.

2. **CTIM (Cross-Temporal Interaction Module)**:

    - **Cross-modal temporal attention**: Event and image features are interleaved along the temporal dimension to form a $2T \times H \times W$ temporal sequence; global MaxPool/AvgPool followed by $1 \times 1$ convolution generates temporal attention weights $W_F^T \in \mathbb{R}^{T \times 1 \times 1}$, which jointly modulate both modalities.
    - **Bidirectional Temporal Selective Scan (BTSS)**: Concatenated features are flattened into a temporal sequence and processed by forward and backward S6 Blocks respectively; the outputs are summed and reshaped to aggregate past and future temporal context.
    - **Modality-aware residual update**: Same as CSIM — separation, temporal attention, and residual connection.

3. **DDIM = CSIM + CTIM**: The two modules are applied in series at each encoder scale, performing dual-dimension fusion along both spatial and temporal axes.

### Loss & Training
- Loss function: Standard cross-entropy loss
- Optimizer: AdamW, trained for 60 epochs
- DDD17: lr=2e-4, batch_size=12; DSEC: lr=6e-5, batch_size=4
- Data augmentation: Random cropping, horizontal flipping, random scaling
- Training hardware: Single RTX 4090D GPU

## Key Experimental Results

### Main Results (Table 1)

| Method | Type | Backbone | DDD17 mIoU | DSEC mIoU |
|--------|------|----------|-----------|-----------|
| SegFormer | Image-only | Transformer | 71.05% | 71.99% |
| EV-SegNet | Event-only | CNN | 54.81% | 51.76% |
| CMX | Fusion | Transformer | 71.88% | 72.42% |
| CMNeXt | Fusion | Transformer | 72.67% | 72.54% |
| EISNet | Fusion | Transformer | 75.03% | 73.07% |
| **MambaSeg** | **Fusion** | **Mamba** | **77.56%** | **75.10%** |

Compared to the previous SOTA EISNet: +2.53% on DDD17, +2.03% on DSEC.

### Efficiency Comparison (Table 2, DDD17)

| Method | Params (M) | MACs (G) | mIoU |
|--------|-----------|----------|------|
| CMX | 66.56 | 16.29 | 71.88% |
| EISNet | 34.39 | 17.30 | 75.03% |
| **MambaSeg** | **25.44** | **15.59** | **77.56%** |

MambaSeg uses only 74% of EISNet's parameters and fewer MACs, while achieving 2.53% higher mIoU.

### Ablation Study
- **CSIM vs. CTIM** (Table 4): baseline 74.38% → +CTIM 76.20% → +CSIM 76.32% → both combined 77.56%, demonstrating complementarity between spatial and temporal fusion.
- **DDIM vs. other fusion methods** (Table 3): Element-wise Add 74.38%, FFM 76.44%, MRFM 76.19%, CSF 76.65%, **DDIM 77.56%**.
- **CSIM sub-modules** (Table 5): All three components — CSA, SS2D, and SA — are indispensable; the complete CSIM yields the best performance.
- **CTIM sub-modules** (Table 6): CTA, BTSS, and TA are similarly complementary; the complete CTIM achieves the best result.

## Highlights & Insights
1. **First application of Mamba to RGB-Event multimodal fusion segmentation**, replacing the quadratic complexity of Transformers with the linear complexity of SSMs for significant efficiency gains.
2. **The dual spatial-temporal fusion design is principled and novel**: CSIM exploits the edge advantages of events and the texture advantages of images for spatial complementarity; CTIM leverages Mamba's strength in sequence modeling for temporal alignment; the two modules are mutually complementary.
3. **Qualitative results** demonstrate that MambaSeg substantially outperforms EISNet on small objects (pedestrians, traffic signs) and under challenging lighting conditions.
4. The ablation study is systematic, providing detailed analysis from the module level down to sub-component level.

## Limitations & Future Work
1. **Limited dataset diversity**: Validation is conducted only on DDD17 (6 classes) and DSEC (11 classes) in autonomous driving scenarios, with few categories and limited scene diversity; generalization ability remains unknown.
2. **Limitations of the event representation**: The fixed-window Voxel Grid may discard fine-grained temporal information in event streams; adaptive temporal segmentation or direct processing of asynchronous events could be explored.
3. **Encoders not jointly pretrained**: Both branches are initialized with the same pretrained VMamba-T; whether ImageNet pretraining is optimal for the event branch remains an open question.
4. **Only cross-entropy loss is used**: Segmentation-friendly alternatives such as Dice Loss and Lovász Loss are not explored.
5. **No validation in larger-scale or more diverse settings**, such as urban or indoor scenes, or with additional modalities (depth, LiDAR).

## Related Work & Insights
- **vs. CMX/CMNeXt**: Both perform multimodal fusion via Transformer-based cross-attention, incurring high computational cost (66M/58M parameters); MambaSeg replaces this with Mamba, reducing parameter count to 25M while improving accuracy.
- **vs. EISNet**: EISNet employs gated attention and progressive recalibration for adaptive spatial alignment; MambaSeg additionally incorporates temporal fusion and achieves better efficiency.
- **vs. Hybrid-Seg**: The CNN+SNN hybrid architecture offers good parameter efficiency but lags significantly in accuracy (67.31% vs. 77.56% on DDD17).
- **vs. VM-UNet and other medical Mamba segmentation methods**: VM-UNet is a single-modality Mamba segmentation model, whereas MambaSeg features dual-branch multimodal input with cross-modal interaction.

The effectiveness of Mamba in RGB-Event fusion validated here suggests potential applicability to other multimodal combinations such as RGB-Depth, RGB-Thermal, and RGB-LiDAR. The bidirectional temporal selective scan in CTIM is also applicable to temporally-aware tasks such as video segmentation. More broadly, the dual spatial-temporal fusion paradigm of DDIM generalizes to scenarios requiring the fusion of heterogeneous multi-source data.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introducing Mamba to RGB-Event fusion is novel, and the dual spatial-temporal design of DDIM is innovative; however, individual sub-modules (pooling-based attention, SS2D) are relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation studies are highly systematic (module-level, sub-component-level, fusion method comparison, and efficiency analysis), though validation is limited to two driving datasets without other scene types.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear, figures and tables are well-organized, and mathematical derivations are complete; however, the Related Work section is somewhat brief.
- **Value**: ⭐⭐⭐⭐ The work represents a clear advancement in RGB-Event segmentation with notable efficiency advantages, though its impact is constrained by the limited real-world deployment of event cameras.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision](../../CVPR2026/autonomous_driving/flashcap_millisecond-accurate_human_motion_capture_via_flashing_leds_and_event-b.md)
- [\[NeurIPS 2025\] StreamForest: Efficient Online Video Understanding with Persistent Event Memory](../../NeurIPS2025/autonomous_driving/streamforest_efficient_online_video_understanding_with_persistent_event_memory.md)
- [\[AAAI 2026\] Hierarchical Prompt Learning for Image- and Text-Based Person Re-Identification](hierarchical_prompt_learning_for_image-_and_text-based_person_re-identification.md)
- [\[ICLR 2026\] x²-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](../../ICLR2026/autonomous_driving/x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)
- [\[ICLR 2026\] Adaptive Augmentation-Aware Latent Learning for Robust LiDAR Semantic Segmentation](../../ICLR2026/autonomous_driving/adaptive_augmentation-aware_latent_learning_for_robust_lidar_semantic_segmentati.md)

<!-- RELATED:END -->
