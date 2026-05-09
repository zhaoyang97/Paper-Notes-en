---
title: >-
  [Paper Note] MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator
description: >-
  [CVPR2026][Segmentation][video matting] This paper proposes a learned Matting Quality Evaluator (MQE) that assesses alpha quality at the pixel level without ground-truth supervision. MQE serves dual roles as an online training guide and an offline data filter, enabling the construction of VMReal — a real-world video matting dataset comprising 28K clips / 2.4M frames. Combined with a reference-frame training strategy, the proposed method significantly outperforms all existing approaches.
tags:
  - CVPR2026
  - Segmentation
  - video matting
  - quality evaluator
  - alpha matte
  - dataset curation
  - reference-frame strategy
date: 2026-05-08
content_hash: ff7c0ad4f4947acd
---

# MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator

**Conference**: CVPR2026
**arXiv**: [2512.11782](https://arxiv.org/abs/2512.11782)
**Code**: [Project Page](https://pq-yang.github.io/projects/MatAnyone2/)
**Area**: Semantic Segmentation / Video Matting
**Keywords**: video matting, quality evaluator, alpha matte, dataset curation, reference-frame strategy

## TL;DR

This paper proposes a learned Matting Quality Evaluator (MQE) that assesses alpha quality at the pixel level without ground-truth supervision. MQE serves dual roles as an online training guide and an offline data filter, enabling the construction of VMReal — a real-world video matting dataset comprising 28K clips / 2.4M frames. Combined with a reference-frame training strategy, the proposed method significantly outperforms all existing approaches.

## Background & Motivation

1. **Scarcity of video matting data**: The largest video matting dataset, VM800, contains only 826 sequences — approximately 1/60 the scale of the VOS dataset used by SAM 2 — severely limiting model training.
2. **Domain gap in synthetic data**: Conventional compositing of RGBA foregrounds onto random backgrounds introduces lighting inconsistencies and unnatural boundaries, degrading generalization to real-world scenes.
3. **Degradation after segmentation pretraining**: Fine-tuning segmentation-pretrained models on matting data causes the learned segmentation capability to deteriorate, due to the scarcity of high-quality matting annotations.
4. **Weak boundary supervision in joint training**: Methods such as MatAnyone apply segmentation labels in non-boundary regions and unsupervised losses at boundaries; the overly strong assumptions of the latter cause predicted alpha mattes to degenerate into segmentation masks.
5. **Trade-off between boundary detail and semantic accuracy**: Existing methods cannot simultaneously improve matting precision and segmentation accuracy.
6. **Large appearance variation in long videos**: Propagation-based methods with limited training windows fail to model drastic appearance changes (e.g., newly appearing clothing or body parts) over long video sequences.

## Method

### Overall Architecture

The core of MatAnyone 2 is the **Matting Quality Evaluator (MQE)**, which takes a triplet $\langle I_{rgb}, \hat{\alpha}, M^{seg} \rangle$ (RGB frame, predicted alpha, segmentation mask) as input and produces a pixel-wise binary evaluation map $M^{eval} \in \{0,1\}^{H \times W}$ (1 = reliable, 0 = erroneous). MQE drives scalable matting training in two modes:

- **Online Guidance**: Evaluates alpha quality in real time during training to provide dynamic supervision for both boundary and interior regions.
- **Offline Selection**: Acts as a quality arbiter for data curation, fusing the complementary strengths of video and image matting models.

### MQE Model Design

- **Encoder**: A pretrained DINOv3 backbone for high-quality feature extraction.
- **Decoder**: A DPT decoder producing the evaluation map.
- **Training data construction**: Based on the P3M-10k image matting dataset; local patch-level discrepancies $\mathcal{D}(\cdot)$ between predicted alpha $\hat{\alpha}$ and ground-truth $\alpha_{gt}$ are computed using MAD and Grad metrics, then thresholded to generate binary labels.
- **Loss function**: Focal Loss is adopted to address severe class imbalance between reliable and erroneous pixels.

### Online Guidance Loss

$$\mathcal{L}_{eval} = \|P^{(0)}_{eval}\|_1$$

where $P^{(0)}_{eval}$ is the pixel-wise error probability map produced by MQE. This loss encourages the network to reduce per-pixel error probability, providing a more effective and stable learning signal at boundary regions than unsupervised losses.

### Dual-Branch Annotation Pipeline → VMReal Dataset

| Branch | Model | Strength | Weakness |
|--------|-------|----------|----------|
| $B_V$ (Video Branch) | MatAnyone | Temporal stability, semantic consistency | Insufficient boundary detail |
| $B_I$ (Image Branch) | MattePro + SAM 2 | Sharp boundaries, rich detail | Temporal inconsistency |

MQE evaluates the alpha from each branch independently, yielding evaluation maps $M_V^{eval}$ and $M_I^{eval}$. The fusion mask $M^{fuse} = M_I^{eval} \odot (1 - M_V^{eval})$ is smoothed via Gaussian blur and used for blending:

$$\alpha = \alpha_V \odot (1 - M^{fuse}) + \alpha_I \odot M^{fuse}$$

The resulting **VMReal** dataset contains approximately 28K clips / 2.4M frames, including 4.5K high-quality 1080p clips with rich hair detail, and the remaining human-subject subset sourced from SA-V at 720p.

### Reference-Frame Training Strategy

Distant reference frames outside the training window (8 frames) are introduced into the memory bank to simulate large appearance variations in long videos. Random dropout augmentation (randomly masking local patches of RGB and alpha) further reduces over-reliance on historical memory.

## Key Experimental Results

### Synthetic Benchmark: VideoMatte (1920×1080)

| Method | MAD↓ | MSE↓ | Grad↓ | dtSSD↓ |
|--------|------|------|-------|--------|
| MatAnyone | 4.24 | 0.33 | 4.00 | 1.19 |
| GVM (diffusion prior) | 6.33 | 2.08 | 8.04 | 1.59 |
| MaGGIe (per-frame mask) | 4.42 | 0.40 | 4.03 | 1.31 |
| **MatAnyone 2** | **4.10** | **0.28** | **3.45** | **1.15** |

### Real-World Benchmark: CRGNN (manually annotated)

| Method | MAD↓ | MSE↓ | Grad↓ | dtSSD↓ |
|--------|------|------|-------|--------|
| MatAnyone | 5.76 | 3.04 | 15.55 | 5.44 |
| GVM | 5.03 | 2.15 | 14.28 | 4.86 |
| **MatAnyone 2** | **4.24** | **2.00** | **11.74** | **4.54** |

### Ablation Study (YoutubeMatte 1920×1080)

| Configuration | MAD↓ | MSE↓ | Grad↓ | dtSSD↓ |
|---------------|------|------|-------|--------|
| (a) Baseline MatAnyone | 1.99 | 0.71 | 8.91 | 1.65 |
| (b) + Online guidance $\mathcal{L}_{eval}$ | 1.90 | 0.62 | 8.20 | 1.63 |
| (c) + VMReal | 1.76 | 0.61 | 7.65 | 1.54 |
| (d) + Reference-frame strategy | **1.61** | **0.50** | **7.13** | **1.53** |

Each component contributes consistent improvements; relative to the baseline, MAD decreases by 19.1% and Grad by 20.0%.

## Highlights & Insights

- **MQE as a dual-purpose tool**: The same evaluator provides online training signals and serves as an offline data filter — an elegant unified design.
- **GT-free quality assessment**: MQE requires only a segmentation mask to perform pixel-level alpha quality judgment, circumventing the annotation bottleneck in video matting.
- **First large-scale real-world video matting dataset**: VMReal contains 28K clips / 2.4M frames, approximately 35× larger than VM800.
- **Pure CNN outperforms diffusion-based methods**: Without relying on video diffusion priors and using only a first-frame mask, the method surpasses diffusion-based approaches such as GVM.
- **Reference-frame strategy with zero additional memory cost**: Long-term appearance changes are modeled by introducing distant frames rather than extending training sequence length.

## Limitations & Future Work

- MQE training relies on the static image matting dataset P3M-10k, which may limit generalization to extreme scenarios such as transparent materials or smoke.
- The quality ceiling of the dual-branch annotation pipeline is bounded by MatAnyone and MattePro; MQE cannot recover from failures of the underlying models.
- VMReal focuses exclusively on human matting and does not cover non-human subjects such as animals or objects.
- The paper does not discuss inference speed or real-time performance; the efficiency advantage of the pure CNN architecture is not quantified.
- The sensitivity of performance to hyperparameters such as the dropout ratio in the reference-frame strategy is not thoroughly analyzed.

## Related Work & Insights

| Dimension | MatAnyone | GVM | MaGGIe | MatAnyone 2 |
|-----------|-----------|-----|--------|-------------|
| Backbone | CNN (SAM 2-based) | Video diffusion model | CNN | CNN (SAM 2-based) |
| Input guidance | First-frame mask | None | Per-frame instance mask | First-frame mask |
| Boundary supervision | Unsupervised loss | Diffusion prior | Segmentation labels | MQE online guidance |
| Training data | VM800 + segmentation data | VM800 + 4K rendered | VM800 | VMReal (28K clips) |
| Long-video handling | Local window memory | None | None | Reference-frame strategy |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual online/offline usage of MQE and the automated annotation pipeline are original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Full coverage of synthetic and real-world benchmarks with clear, component-wise ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured presentation with intuitive figures and sufficiently motivated problem formulation.
- **Value**: ⭐⭐⭐⭐⭐ — The VMReal dataset and the MQE methodology represent significant contributions to the video matting community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels](towards_high-quality_image_segmentation_improving_topology_accuracy_by_penalizin.md)
- [\[ICCV 2025\] ZIM: Zero-Shot Image Matting for Anything](../../ICCV2025/segmentation/zim_zero-shot_image_matting_for_anything.md)
- [\[CVPR 2026\] 3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion](3m-ti_high-quality_mobile_thermal_imaging_via_calibration-free_multi-camera_cros.md)
- [\[NeurIPS 2025\] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing](../../NeurIPS2025/segmentation/roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)
- [\[CVPR 2026\] VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](videomt_your_vit_is_secretly_also_a_video_segmentation_model.md)

</div>

<!-- RELATED:END -->
