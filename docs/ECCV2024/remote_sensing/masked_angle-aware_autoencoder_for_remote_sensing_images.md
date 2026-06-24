---
title: >-
  [Paper Note] Masked Angle-Aware Autoencoder for Remote Sensing Images
description: >-
  [ECCV 2024][Remote Sensing][Self-supervised learning] The authors propose MA3E, which explicitly introduces angle variations into MAE pre-training (by constructing rotational crops via scaling center crop) and automatically assigns reconstruction targets using an optimal transport loss. This allows the model to perceive the diverse angles of remote sensing objects and learn rotation-invariant representations.
tags:
  - "ECCV 2024"
  - "Remote Sensing"
  - "Self-supervised learning"
  - "Masked image modeling"
  - "Rotational invariance"
  - "Optimal transport"
  - "Remote sensing images"
date: 2026-05-08
content_hash: 136f50b950f641b1
---

# Masked Angle-Aware Autoencoder for Remote Sensing Images

**Conference**: ECCV 2024  
**arXiv**: [2408.01946](https://arxiv.org/abs/2408.01946)  
**Code**: [GitHub](https://github.com/benesakitam/MA3E)  
**Area**: Remote Sensing  
**Keywords**: Self-supervised learning, Masked image modeling, Rotational invariance, Optimal transport, Remote sensing images

## TL;DR

The authors propose MA3E, which explicitly introduces angle variations into MAE pre-training (by constructing rotational crops via scaling center crop) and automatically assigns reconstruction targets using an optimal transport loss. This allows the model to perceive the diverse angles of remote sensing objects and learn rotation-invariant representations.

## Background & Motivation

There is a significant domain gap between remote sensing (RS) images and natural images. Objects in natural images usually have fixed orientations due to gravity, whereas remote sensing images are captured from a bird's-eye view, and the objects exhibit **various angles**—the same object can present completely different shapes and appearances under different angles.

Existing self-supervised methods in remote sensing (such as SatMAE, ScaleMAE, and RingMo), while considering factors like multi-resolution, multi-scale, and multi-spectral data, **neglect the angular diversity of remote sensing objects**. These methods focus only on pixel-value reconstruction, and the learning of angular information is only implicitly associated with the reconstruction process.

The authors intuitively demonstrate this problem through experiments (Fig.1): after pre-training with standard MAE and performing rotated object detection, the model only performs well on objects oriented near horizontal directions (0° or 90°), while its performance drops significantly for objects with large tilt angles (10°-80°). This indicates that existing MIM methods fail to effectively learn angle-aware representations.

**Core Problem**: How can the model be made to explicitly perceive and learn the angle information of remote sensing targets during the pre-training stage, thereby obtaining rotation-invariant visual representations?

## Method

### Overall Architecture

MA3E follows the asymmetric encoder-decoder architecture of MAE. The core improvement is the construction of a **rotated crop** with a random direction on the original image, which is embedded back into the original image to form a synthetic image as input. The training objective is to **reconstruct the original image** (i.e., simultaneously completing pixel reconstruction and angle restoration).

Pipeline: Original image → scaling center crop to create rotated crop → replace scene at original position → add angle embedding → separate masking → encoding-decoding → background reconstruction with MSE loss + rotated region reconstruction with OT loss.

### Key Designs

1. **Scaling Center Crop**: The core operation for constructing the rotated crop. For a square region with side length $h$ in the image, after rotating it at a random angle, its **maximum inscribed square within the maximum inscribed circle** is taken as the rotated crop, with side length $a = \frac{\sqrt{2}}{2}h$. The motivation is: direct random rotation leads to three issues—(i) introduction of meaningless zero-value backgrounds, (ii) scene loss, and (iii) changes in scene scale. In contrast, the scaling center crop introduces arbitrary angle variations while preserving the main scene. The rotated crop replaces the original scene to form a synthetic image, providing explicit angle variation signals for the model.

2. **Angle Embedding**: A learnable angle embedding vector is added to each patch within the rotated crop (shared within the same crop). This embedding acts as an **implicit hint** for the model to perceive the angular variation of the rotated crop while distinguishing the rotated region from the background. This is a lightweight but effective design—it does not require explicit angle labels, and allows the model to automatically learn angle-aware details through the extra embedding vector.

3. **Separate Random Masking**: The $N_r$ patches of the rotated crop and the $N_b$ patches of the background are independently masked at a rate of 75%. Motivation: Global random masking in standard MAE might cause the patches in the rotated crop to be excessively or even completely masked (since the rotated region is relatively small), failing to learn angle information. Separate masking ensures both regions have sufficient visible patches.

4. **Optimal Transport Loss (OT Loss)**: After rotation, the patches in the rotated crop experience a scene shift relative to the patches at the same position in the original image. Directly using MSE for reconstruction would introduce significant bias. MA3E formulates this as an optimal transport problem: the $N_r$ patches of the original image are treated as suppliers, and the $N_r$ predicted patches as demanders, with the transport cost defined as the L2 distance:

$$c_{ij} = \|r_i - \hat{r}_j\|_2^2$$

Using the Sinkhorn-Knopp fast iteration algorithm to solve for the transport plan $\Omega$, the OT loss automatically assigns similar original patches as reconstruction targets for each predicted patch:

$$\mathcal{L}_{OT}(r, \hat{r}) = \sum_{i=1}^{N_r}\sum_{j=1}^{N_r} \|r_i - \hat{r}_j\|_2^2 \omega_{ij}$$

### Loss & Training

The total loss consists of the background MSE loss and the rotated region OT loss:

$$\mathcal{L}_{rec} = \mathcal{L}_{MSE}(b^m, \hat{b}^m) + \mathcal{L}_{OT}(r, \hat{r})$$

- Background region: MSE loss is computed only for masked patches (consistent with standard MAE).
- Rotated crop region: OT loss is computed for all patches (both visible and masked).

Pre-training is conducted on the MillionAID dataset (approx. 990k RS images), with input size $224 \times 224$, patch size=16, rotated crop side length $a=96$, rotation range $[-45°, +45°]$, encoder ViT-B, and decoder with 8-layer ViT blocks (512-D).

## Key Experimental Results

### Main Results

**Scene Classification (Fine-tuning)**:

| Dataset | Metric | MA3E (300ep) | MA3E (1600ep) | MAE (1600ep) | MAE+RVSA (1600ep) |
|--------|------|-------------|--------------|-------------|-------------------|
| NWPU-RESISC45 | Top-1 Acc | 95.77 | **96.23** | 95.40 | 95.49 |
| AID | Top-1 Acc | 98.44 | **99.04** | 98.36 | 98.33 |
| UC Merced | Top-1 Acc | 99.05 | **99.81** | 99.44 | 99.70 |

**Rotated Object Detection & Semantic Segmentation**:

| Dataset | Metric | MA3E (1600ep) | MAE+RVSA (1600ep) | MAE+ViTAE+RVSA (1600ep) |
|--------|------|--------------|-------------------|------------------------|
| DOTA1.0 | mAP | **79.47** | 78.75 | 78.96 |
| DIOR-R | mAP | **71.82** | 70.67 | 70.95 |
| iSAID | mIoU | **64.06** | 63.76 | 63.48 |
| Potsdam | mF1 | **91.50** | 90.60 | 91.22 |

### Ablation Study

**Ablation of Components (300 epochs, ViT-B)**:

| Configuration | NU45 (ft) | DOTA1.0 (det) | iSAID (seg) | Description |
|------|-----------|--------------|-------------|------|
| MAE baseline | 95.31 | 75.85 | 60.96 | Standard MAE |
| + SCC | 95.43 | 76.12 | 61.24 | Add scaling center crop |
| + SCC + AE | 95.47 | 76.41 | 61.86 | Add angle embedding |
| + SCC + OT | 95.36 | 76.46 | 61.88 | Add OT loss |
| + SCC + Mask. | 95.06 | 77.23 | 62.17 | Add separate masking |
| + SCC + AE + Mask. | 95.53 | 76.70 | 61.93 | Combination of three components |
| **MA3E (All)** | **95.77** | **77.93** | **62.74** | All components |

**Ablation on Rotation Range**:

| Rotation Range | NU45 (ft) | DOTA1.0 (det) | iSAID (seg) |
|---------|-----------|--------------|-------------|
| [-30°, +30°] | 95.78 | 77.68 | 62.49 |
| **[-45°, +45°]** | **95.77** | **77.93** | **62.74** |
| [-60°, +60°] | 95.32 | 77.22 | 62.55 |
| [-90°, +90°] | 94.89 | 76.45 | 61.90 |

### Key Findings

- MA3E significantly improves the detection AP50 for objects with large tilt angles of 10°-80° (Fig. 1), verifying the effectiveness of angle awareness.
- Rotated crop side length $a=96$ (36 patches) performs best; being too large or using multiple crops degrades performance instead.
- Scaling center crop improves performance over simple random rotation on the three tasks by 1.95, 1.79, and 1.51, respectively.
- The $\pm 45°$ rotation range is optimal; a wider range makes angle restoration excessively difficult.

## Highlights & Insights

- **Precise Problem Definition**: Angular diversity in remote sensing objects is an overlooked but crucial issue, directly impacting tasks like rotated object detection.
- **Elegant OT Loss Design**: Formulating the reconstruction after scene shift as an optimal transport problem avoids the bias of rigid one-to-one matching.
- **Low Computational Cost**: Compared to MAE, training time increases by only about 0.2 hours/epoch, with minimal extra parameters.
- Outperforms methods using more complex architectures (such as ViTAE+RVSA) using only a simple ViT-B backbone.

## Limitations & Future Work

- Angle awareness is more valuable for **man-made objects** (vehicles, buildings, etc.) and offers limited gains for large-scale natural terrain (woodlands, water bodies).
- **Scale factor** is not considered—the multi-scale nature of remote sensing images is as important as the angular variety, and joint modeling of both is worth exploring.
- The selection of the rotated crop location is relatively random. Although selective search brings minor improvements, its cost is high. More efficient target region selection strategies remain to be investigated.
- Validated only on ViT-B; the scalability to larger models (ViT-L/H) has not been explored.

## Related Work & Insights

- Contrasting with the approach of MixMAE (which reconstructs mixed inputs from multiple images), MA3E creates a synthetic input **within a single image**, which is more suitable for remote sensing contexts.
- The application of OT in pre-training loss can be extended to other scenarios with spatial shifts (e.g., deformation, perspective transformation).
- Methodological takeaway: Designing customized pre-training strategies tailored to domain-specific physical properties (such as the bird's-eye view in remote sensing or multi-modalities in medical imaging) is highly beneficial.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing angle awareness into MIM pre-training and handling rotation reconstruction with OT loss are innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 7 datasets, 3 downstream tasks, and extensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic, strong motivation, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Practically advances remote sensing pre-training with a simple and effective design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images](../../ICCV2025/remote_sensing/smarties_spectrum-aware_multi-sensor_auto-encoder_for_remote_sensing_images.md)
- [\[CVPR 2026\] NeighborMAE: Exploiting Spatial Dependencies between Neighboring Earth Observation Images in Masked Autoencoders Pretraining](../../CVPR2026/remote_sensing/neighbormae_exploiting_spatial_dependencies_between_neighboring_earth_observatio.md)
- [\[ECCV 2024\] Learning Representations of Satellite Images From Metadata Supervision](learning_representations_of_satellite_images_from_metadata_supervision.md)
- [\[NeurIPS 2025\] ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](../../NeurIPS2025/remote_sensing/chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)
- [\[CVPR 2026\] SegEarth-R2: Towards Comprehensive Language-guided Segmentation for Remote Sensing Images](../../CVPR2026/remote_sensing/segearth-r2_towards_comprehensive_language-guided_segmentation_for_remote_sensin.md)

</div>

<!-- RELATED:END -->
