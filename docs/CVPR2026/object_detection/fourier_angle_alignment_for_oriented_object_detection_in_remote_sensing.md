---
title: >-
  [Paper Note] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing
description: >-
  [CVPR 2026][Object Detection][Oriented object detection] By exploiting Fourier rotational equivariance to estimate the principal orientation of objects in the frequency domain and align features accordingly…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "Oriented object detection"
  - "Fourier rotational equivariance"
  - "frequency-domain orientation estimation"
  - "feature fusion"
  - "remote sensing"
date: 2026-05-08
content_hash: 9890bac22ab0ad18
---

# Fourier Angle Alignment for Oriented Object Detection in Remote Sensing

**Conference**: CVPR 2026
**arXiv**: [2602.23790](https://arxiv.org/abs/2602.23790)  
**Code**: [https://github.com/gcy0423/Fourier-Angle-Alignment](https://github.com/gcy0423/Fourier-Angle-Alignment)  
**Area**: Object Detection
**Keywords**: Oriented object detection, Fourier rotational equivariance, frequency-domain orientation estimation, feature fusion, remote sensing

## TL;DR

By exploiting Fourier rotational equivariance to estimate the principal orientation of objects in the frequency domain and align features accordingly, this paper proposes two plug-and-play modules—FAAFusion and FAA Head—to address cross-scale directional incoherence in FPN and the classification–regression task conflict in detection heads, respectively, achieving new state-of-the-art results on DOTA-v1.0/v1.5 and HRSC2016.

## Background & Motivation

**Core challenge in oriented object detection**: Objects such as ships, aircraft, and vehicles in remote sensing images exhibit arbitrary orientations, requiring detectors to predict a rotation angle $\theta$ in addition to the standard HBB representation $(x,y,w,h,c)$. Existing methods focus on rotation-sensitive convolutions (ARC, GRA), novel backbones (ReDet, LSKNet, PKINet, Strip R-CNN), or angular regression loss optimization (GWD, KLD), yet two structural bottlenecks remain overlooked.

**Bottleneck 1 — Directional Incoherence in the Neck**: In FPN, high-level features carry rich semantics but lose directional precision after repeated downsampling (low-frequency), enabling only coarse horizontal/vertical orientation perception; low-level features preserve abundant edges and textures, retaining accurate directional cues (high-frequency). Conventional FPN fuses these directionally inconsistent features via element-wise addition, introducing directional noise that degrades angular prediction accuracy.

**Bottleneck 2 — Task Conflict in the Detection Head**: The same RoI feature must simultaneously serve classification and angular regression—classification demands rotation-invariant features (an aircraft is an aircraft regardless of orientation), while regression demands rotation-sensitive features (different orientations should yield different angle predictions). A single shared feature is forced into a compromise, being neither fully invariant nor fully sensitive, limiting the accuracy of both tasks.

**Core insight — Fourier Rotational Equivariance**: When a spatial-domain signal is rotated by $\phi$, its spectrum rotates by exactly $\phi$ (i.e., $\mathbf{F}_\phi(\boldsymbol{\omega}) = \mathcal{F}\{\mathbf{I}(\mathbf{R}_{-\phi}\mathbf{x})\}$). Moreover, the principal direction of a rectangular target's power spectrum is perpendicular to its long axis (when $a > b$, the main lobe of $\operatorname{sinc}(2au)$ is narrower, concentrating high-frequency energy along the $v$-axis). This enables reliable estimation of the principal orientation from the frequency domain and explicit feature alignment—an essential complement to purely spatial-domain approaches.

## Method

### Overall Architecture

Fourier Angle Alignment (FAA) comprises two plug-and-play modules: **FAAFusion**, embedded in the FPN neck to replace element-wise addition and resolve cross-scale directional incoherence; and **FAA Head**, replacing the original detection head to resolve the classification–regression task conflict. Both share a core Fourier Angle Estimation (FAE) pipeline.

### Key Designs

1. **Fourier Angle Estimation (FAE)**

    - **Function**: Estimates the principal orientation angle of a feature map in the frequency domain.
    - **Mechanism**: Given a square feature map $\mathbf{X} \in \mathbb{R}^{H \times H}$, a 2D DFT is applied to obtain the spectrum $\mathbf{F} = \mathcal{F}(\mathbf{X})$; the DC component is shifted to the center (via multiplication by $(-1)^{u+v}$); Cartesian coordinates $(u,v)$ are converted to polar coordinates $(\rho, \theta)$, and the energy spectrum $E(\rho, \theta) = |\mathbf{F}_c(u(\rho,\theta), v(\rho,\theta))|^2$ is computed; a radially weighted sum yields the one-dimensional angular energy distribution $E_\theta(\theta) = \sum_\rho \rho \cdot E(\rho, \theta)$; the peak direction is taken as $\hat{\theta} = \arg\max_\theta E_\theta(\theta)$, constrained to $[0, \pi)$.
    - **Design Motivation**: Grounded in the mathematical property that the principal spectral direction of a rectangular target is perpendicular to its long axis; radial weighting enhances sensitivity to high-frequency directional components.

2. **FAAFusion (Direction-Consistent Feature Fusion)**

    - **Function**: Embedded in FPN to replace element-wise addition, aligning the orientations of high-level and low-level features before fusion.
    - **Mechanism**: The high-level feature $\mathbf{Y}^{l+1}$ is upsampled to the resolution of the low-level feature; both are projected to $C_{mid}$ via $1 \times 1$ convolutions and unfolded into local patches $\{\mathbf{p}_i^h\}$, $\{\mathbf{p}_i^l\}$; for each location $i$, FAE estimates the principal direction $\theta_i^l$ of the low-level patch; the corresponding high-level patch is rotated toward $\theta_i^l$ to obtain $\mathbf{p}_i^{rh} = \text{FAA}(\mathbf{p}_i^h; \theta_i^l)$; the aligned high-level feature $\mathbf{Y}_{recon}^{l+1}$ is reconstructed via folding and restored to the original channel dimension via $1 \times 1$ convolution; the final output is the three-way sum $\mathbf{Y}^l = \mathbf{X}^l + \mathbf{Y}_u^{l+1} + \mathbf{Y}_{recon}^{l+1}$.
    - **Design Motivation**: Low-level features have precise orientations (high-frequency edges) and serve as a reference to align the ambiguous orientations of high-level features, eliminating directional signal conflicts introduced by direct addition; retaining the original upsampled feature addition preserves semantic information.

3. **FAA Head (Orientation-Aware Detection Head)**

    - **Function**: Replaces the standard detection head by pre-aligning RoI features to a canonical orientation to decouple classification and regression.
    - **Mechanism**: The RoI-aligned feature $\mathbf{F}_{roi}$ is processed by FAA to align its principal direction to $0°$, yielding a rotation-invariant feature $\mathbf{F}_{inv} = \text{FAA}(\mathbf{F}_{roi}; 0°)$; a residual addition gives $\mathbf{F}_{final} = \mathbf{F}_{inv} + \mathbf{F}_{roi}$; after flattening, the feature passes through two shared FC layers (first layer output dimension $1024 + 256 = 1280$), followed by separate classification and regression branches.
    - **Design Motivation**: $\mathbf{F}_{inv}$ eliminates orientation variation, producing approximately consistent representations for the same category, which benefits classification; $\mathbf{F}_{roi}$ retains orientation-sensitive information, which benefits angular regression; the residual connection ensures that both tasks receive the signals they require.

### Loss & Training

Standard Oriented R-CNN losses are adopted (RPN classification + regression, Head classification + regression) with no additional loss terms. Optimizer: AdamW (weight decay 0.05). Initial learning rates: 0.0001 for DOTA (16 epochs) and 0.0004 for HRSC2016 (36 epochs). Batch size: 2; hardware: single RTX 3090. FAAFusion is deployed at the P3–P2 fusion stage of the FPN.

## Key Experimental Results

### Main Results — DOTA-v1.0 (Single-Scale Training and Testing)

| Method | Backbone | mAP |
|--------|----------|-----|
| O-RCNN | ResNet50 | 75.87% |
| **O-RCNN + Ours** | ResNet50 | **76.55%** (+0.68) |
| LSKNet | LSKNet-S | 77.49% |
| **LSKNet + Ours** | LSKNet-S | **78.49%** (+1.00) |
| PKINet | PKINet-S | 78.39% |
| S-RCNN | StripNet-S | 78.09% |
| **S-RCNN + Ours** | StripNet-S | **78.72%** (+0.63, new SOTA) |

### Main Results — DOTA-v1.5 (Single-Scale Training and Testing)

| Method | Backbone | mAP |
|--------|----------|-----|
| O-RCNN | ResNet50 | 66.77% |
| **O-RCNN + Ours** | ResNet50 | **67.14%** (+0.37) |
| S-RCNN | StripNet-S | 69.84% |
| **S-RCNN + Ours** | StripNet-S | **71.57%** (+1.73) |
| PKINet | PKINet-S | 71.47% |
| LSKNet | LSKNet-S | 70.26% |
| **LSKNet + Ours** | LSKNet-S | **72.28%** (+2.02, new SOTA) |

### Main Results — HRSC2016

| Method | Params | FLOPs | AP50 (VOC07) | AP75 | mAP |
|--------|--------|-------|--------------|------|-----|
| O-RCNN | 41.13M | 134.46G | 89.7 | 79.5 | 64.77 |
| **O-RCNN + Ours** | 63.27M | 140.70G | **89.8** | **80.0** | **66.94** (+2.17) |
| LSKNet | 30.96M | 111.42G | 90.2 | 87.9 | 68.78 |
| **LSKNet + Ours** | 48.34M | 114.89G | **90.6** | **89.8** | **70.74** (+1.96) |
| S-RCNN | 45.12M | 157.19G | 89.5 | 78.8 | 69.18 |
| **S-RCNN + Ours** | 49.05M | 115.91G | **90.0** | 78.6 | **70.41** (+1.23) |

### Ablation Study (DOTA-v1.0, LSKNet-S Backbone)

| FAAFusion | FAA Head | Params | GFLOPs | mAP |
|-----------|----------|--------|--------|-----|
| ✘ | ✘ | 30.98M | 173.68G | 77.49% |
| ✘ | ✔ | 48.35M | 177.15G | 78.27% (+0.78) |
| ✔ | ✘ | 32.18M | 175.59G | 77.91% (+0.42) |
| ✔ | ✔ | 49.56M | 179.06G | **78.49%** (+1.00) |

### Detection Head Comparison (DOTA-v1.0)

| Backbone | Head | Params | GFLOPs | mAP |
|----------|------|--------|--------|-----|
| ResNet50 | Original Head | 41.14M | 211.43G | 75.81% |
| ResNet50 | Strip Head | 55.82M | 258.35G | 76.11% |
| ResNet50 | **FAA Head** | 58.51M | 214.90G | **76.18%** |
| LSKNet-S | Original Head | 30.98M | 173.68G | 77.49% |
| LSKNet-S | Strip Head | 45.65M | 220.60G | 78.04% |
| LSKNet-S | **FAA Head** | 48.35M | 177.15G | **78.27%** |
| StripNet-S | Original Head | 30.46M | 171.79G | 77.03% |
| StripNet-S | Strip Head | 45.14M | 218.71G | 78.09% |
| StripNet-S | **FAA Head** | 47.83M | 175.26G | **78.52%** |

### Key Findings

- FAAFusion and FAA Head are complementary: individually they yield +0.42% and +0.78% gains, respectively, while their combination achieves +1.00%.
- Consistent improvements across three different backbones demonstrate the plug-and-play generality of the proposed modules.
- The largest gains are observed on DOTA-v1.5 (LSKNet +2.02%), a dataset containing numerous extremely small objects (<10 pixels), indicating that orientation alignment is particularly beneficial for small targets.
- HRSC2016 ship detection yields the largest absolute improvement (O-RCNN +2.17%), highlighting the advantage of frequency-domain orientation estimation for high aspect-ratio objects.
- FAA Head achieves higher accuracy than Strip Head with a similar parameter count but over 40 GFLOPs fewer—suggesting that frequency-domain alignment is more computationally efficient than spatial strip convolutions.
- Analysis at high IoU thresholds (0.70–0.90) shows that the performance advantage grows with increasing threshold, confirming that orientation alignment enhances precise localization.

## Highlights & Insights

- **Novel frequency-domain perspective**: This work is the first to systematically apply Fourier rotational equivariance to oriented object detection, with direction angle estimation in the frequency domain supported by rigorous mathematical derivation (sinc main-lobe analysis for rectangular targets) and strong physical interpretability.
- **Precise problem diagnosis**: Two independent problems—directional incoherence in the neck and task conflict in the head—are clearly identified and addressed by FAAFusion and FAA Head, respectively.
- **FAAFusion's low-level reference strategy** is intuitively sound: low-level features have clear edges and reliable orientations, making them a natural reference for correcting the ambiguous orientation of high-level features.
- **FAA Head's residual design** is elegantly effective: $\mathbf{F}_{inv} + \mathbf{F}_{roi}$ achieves implicit decoupling of classification and regression in a single step, without requiring a complex dual-branch architecture.
- Sustained advantages at high IoU thresholds confirm that orientation alignment genuinely improves fine-grained localization.

## Limitations & Future Work

- **Notable parameter overhead**: O-RCNN grows from 41M to 63M parameters (+54%); the unfold/fold operations in FAAFusion introduce additional computational cost; lighter frequency-domain processing could be explored.
- **Rectangular target assumption**: Frequency-domain orientation estimation relies on the prior that objects are approximately rectangular; accuracy may degrade for irregularly shaped targets (e.g., circular storage tanks).
- **Framework dependency**: Validation is limited to the Oriented R-CNN two-stage framework; applicability to one-stage detectors (e.g., S2A-Net) or anchor-free methods has not been tested.
- **FAAFusion deployed at only one feature level**: Only the P3–P2 addition is replaced; the trade-off between gains and overhead from full-level deployment merits exploration.
- Performance has not been verified on larger-scale datasets (e.g., DIOR-R) or under multi-scale training and testing conditions.

## Related Work & Insights

- **FreqFusion** decomposes features into high- and low-frequency components for separate processing, whereas FAA directly exploits rotational equivariance for orientation estimation—both operate in the frequency domain but with different objectives.
- **ReDet** models directional information via a rotation-equivariant backbone (ReResNet); FAA achieves similar goals in a lighter, plug-and-play manner at the neck and head levels.
- **Strip R-CNN** models the geometric characteristics of high aspect-ratio objects with strip convolutions; FAA Head achieves higher accuracy with fewer FLOPs, suggesting that frequency-domain alignment may be more efficient than explicit geometric convolutions.
- The proposed frequency-domain orientation estimation approach has potential for extension to tasks requiring directional modeling, such as instance segmentation, remote sensing change detection, and pose estimation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Frequency-domain rotational equivariance applied to oriented object detection; theoretically rigorous, conceptually fresh, with well-motivated designs for both FAAFusion and FAA Head.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three datasets, three backbones, complete ablations, and convincing head comparisons; lacks multi-scale and broader framework validation.
- Writing Quality: ⭐⭐⭐⭐ — Detailed theoretical derivations, clear formulation and motivation sections, good integration of figures and text.
- Value: ⭐⭐⭐⭐⭐ — Plug-and-play integration, consistent and stable improvements, strong physical interpretability, and open-source code make this highly practical.

---

## Related Papers

- [\[CVPR 2026\] Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)
- [\[ECCV 2024\] MutDet: Mutually Optimizing Pre-training for Remote Sensing Object Detection](../../ECCV2024/object_detection/mutdet_mutually_optimizing_pre-training_for_remote_sensing_object_detection.md)
- [\[AAAI 2026\] SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](../../AAAI2026/object_detection/sm3det_a_unified_model_for_multi-modal_remote_sensing_object_detection.md)
- [\[ICLR 2026\] SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](../../ICLR2026/object_detection/spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)
- [\[ICCV 2025\] OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images](../../ICCV2025/object_detection/openrsd_towards_open-prompts_for_object_detection_in_remote_sensing_images.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)
- [\[ICLR 2026\] SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](../../ICLR2026/object_detection/spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)
- [\[AAAI 2026\] SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](../../AAAI2026/object_detection/sm3det_a_unified_model_for_multi-modal_remote_sensing_object_detection.md)
- [\[ICCV 2025\] OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images](../../ICCV2025/object_detection/openrsd_towards_open-prompts_for_object_detection_in_remote_sensing_images.md)
- [\[CVPR 2026\] SPAN: Spatial-Projection Alignment for Monocular 3D Object Detection](span_spatial-projection_alignment_for_monocular_3d_object_detection.md)

</div>

<!-- RELATED:END -->
