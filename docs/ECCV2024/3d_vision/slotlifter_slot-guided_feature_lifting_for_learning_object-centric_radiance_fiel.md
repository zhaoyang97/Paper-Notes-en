---
title: >-
  [Paper Note] SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields
description: >-
  [ECCV 2024][3D Vision][Object-centric learning] SlotLifter is proposed, which combines 2D-to-3D feature lifting with Slot Attention through a slot-guided feature lifting design. It achieves state-of-the-art performance in both scene decomposition and novel view synthesis, while accelerating training efficiency by approximately 5 times.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Object-centric learning"
  - "Radiance fields"
  - "Slot Attention"
  - "Scene decomposition"
  - "Novel view synthesis"
date: 2026-05-08
content_hash: f30c3f1275fab6d1
---

# SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields

**Conference**: ECCV 2024  
**arXiv**: [2408.06697](https://arxiv.org/abs/2408.06697)  
**Code**: [Project Page](https://slotlifter.github.io)  
**Area**: 3D Vision  
**Keywords**: Object-centric learning, Radiance fields, Slot Attention, Scene decomposition, Novel view synthesis

## TL;DR

SlotLifter is proposed, which combines 2D-to-3D feature lifting with Slot Attention through a slot-guided feature lifting design. It achieves state-of-the-art performance in both scene decomposition and novel view synthesis, while accelerating training efficiency by approximately 5 times.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Object-centric learning aims to extract object-level abstract representations from visual scenes in an unsupervised manner. Existing 3D object-centric methods (such as uORF, COLF) suffer from several limitations: (1) requiring additional auxiliary losses (adversarial loss, LPIPS loss); (2) difficulty in generalizing to real-world scenes; (3) high computational cost (such as OSRT requiring 64 TPUv2s for 7 days of training). The key challenge lies in effectively aligning multi-view information and reconstructing 3D scenes from compressed slot representations.

## Method

### Overall Architecture

SlotLifter consists of three stages: (1) Scene encoding—extracting slots via Slot Attention and lifting 2D features to 3D; (2) Point-slot mapping—assigning slots to 3D points via cross-attention; (3) Rendering—generating images and segmentation masks based on volume rendering.

### Key Designs

**Slot-guided Feature Lifting**: The 2D feature maps from input viewpoints are projected onto 3D point coordinates to obtain the lifted point feature $\mathbf{F}_{lift}$. After computing the mean and variance across multi-view features, the 3D point feature $\mathbf{F}_p$ is obtained through an MLP, with positional encodings added to preserve spatial information.

**Point-slot Mapping**: Cross-attention is employed to let 3D point features query slot representations, with an additional empty slot introduced to handle empty regions. Finally, the slot assignment for each 3D point and the slot-aggregated point feature $\mathbf{F}_s$ are obtained through the attention weights $\mathbf{W}_p$.

**Slot-based Density Prediction**: The attention weights from the mapping module are directly utilized to predict the density $\sigma_i = \text{sum}(\mathbf{W}_p^i \odot \text{ReLU}(\mathbf{A}_p^i))$, where ReLU is used to suppress the contributions of irrelevant slots.

**Random Masking Strategy**: The lifted features are randomly masked (using a cosine annealing schedule from 0.99 to 0) to prevent the model from degenerating into relying solely on lifted features while ignoring slot information.

### Loss & Training

Only the MSE reconstruction loss is used: $\mathcal{L}_{recon} = \|\mathbf{C}(r) - \hat{\mathbf{C}}(r)\|^2$, with no auxiliary losses required.

## Key Experimental Results

### Synthetic Scene Decomposition


### Main Results

| Method | CLEVR-567 NV-ARI↑ | Room-Chair NV-ARI↑ | Room-Diverse NV-ARI↑ | Room-Texture NV-ARI↑ |
|------|--------------------|--------------------|----------------------|----------------------|
| uORF | 83.8 | 74.3 | 56.9 | 57.8 |
| BO-uORF | 78.4 | 80.9 | 62.5 | 60.4 |
| COLF | 55.8 | 80.7 | 52.5 | 1.1 |
| uOCF-P | - | - | - | 70.4 |
| **SlotLifter** | **87.0** | **89.7** | **77.5** | **79.3** |

### Novel View Synthesis

Quantitative comparison on the Room-Texture dataset:


### Ablation Study

| Method | LPIPS↓ | SSIM↑ | PSNR↑ | NV-ARI↑ | FG-ARI↑ |
|------|--------|-------|-------|---------|---------|
| uORF | 0.254 | 0.711 | 24.23 | 57.8 | 9.3 |
| BO-uORF | 0.215 | 0.739 | 25.26 | 60.4 | 35.4 |
| uOCF-P | 0.136 | 0.798 | 28.85 | 70.4 | 56.3 |
| **SlotLifter** | **0.131** | **0.858** | **30.68** | **79.3** | **70.7** |

### Key Findings

- ARI on Room-Diverse increases by +15 (77.5 vs 62.5), demonstrating that feature lifting is highly effective for complex scenes.
- Training requires only 1024 sampled rays and the MSE loss, making it approximately 5 times faster than uORF.
- Competitive novel view synthesis performance is also demonstrated on real-world datasets ScanNet and DTU.

## Highlights & Insights

1. **Simple and Elegant Design**: Fully SOTA performance is achieved using only reconstruction loss, without auxiliary losses or extra decoders.
2. Feature lifting provides more explicit 3D guidance for slot learning, avoiding the information bottleneck of pure slot decoders.
3. Only 1 radiance field (instead of K) is required, significantly reducing computational overhead.

## Limitations & Future Work

- Requires known camera parameters as input.
- Background segmentation remains challenging in real-world scenes.
- The number of slots must be preset, which limits adaptability to unseen scenes.

## Related Work & Insights

Introducing the concepts of image-based rendering (e.g., PixelNeRF, IBRNet) into object-centric learning is a novel intersection. The random masking strategy's effectiveness in preventing feature degeneration is worth referencing in other multi-signal learning scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐
- Practicality: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] G2fR: Frequency Regularization in Grid-Based Feature Encoding Neural Radiance Fields](g2fr_frequency_regularization_in_grid-based_feature_encoding_neural_radiance_fie.md)
- [\[ECCV 2024\] BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream](benerf_neural_radiance_fields_from_a_single_blurry_image_and_event_stream.md)
- [\[ECCV 2024\] LaRa: Efficient Large-Baseline Radiance Fields](lara_efficient_large-baseline_radiance_fields.md)
- [\[ECCV 2024\] Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal](learning_3d_geometry_and_feature_consistent_gaussian_splatting_for_object_remova.md)
- [\[ECCV 2024\] GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields](geometrysticker_enabling_ownership_claim_of_recolorized_neural_radiance_fields.md)

</div>

<!-- RELATED:END -->
