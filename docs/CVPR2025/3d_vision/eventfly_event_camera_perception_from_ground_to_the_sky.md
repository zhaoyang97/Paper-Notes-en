---
title: >-
  [Paper Note] EventFly: Event Camera Perception from Ground to the Sky
description: >-
  [CVPR 2025][3D Vision][Event camera] EventFly proposes the first cross-platform domain adaptation framework for event cameras. By identifying high-activation regions using the Event Activation Prior (EAP), blending source/target domain event data with EventBlend, and aligning feature distributions with EventMatch dual discriminators, it achieves an average improvement of 23.8% in accuracy and 77.1% in mIoU compared to source-only training on semantic segmentation tasks across…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Event camera"
  - "cross-platform domain adaptation"
  - "semantic segmentation"
  - "data mixing"
  - "adversarial training"
date: 2026-05-08
content_hash: 9740d12b544742f8
---

# EventFly: Event Camera Perception from Ground to the Sky

**Conference**: CVPR 2025  
**arXiv**: [2503.19916](https://arxiv.org/abs/2503.19916)  
**Code**: Yes ([https://event-fly.github.io](https://event-fly.github.io))  
**Area**: 3D Vision  
**Keywords**: Event camera, cross-platform domain adaptation, semantic segmentation, data mixing, adversarial training

## TL;DR
EventFly proposes the first cross-platform domain adaptation framework for event cameras. By identifying high-activation regions using the Event Activation Prior (EAP), blending source/target domain event data with EventBlend, and aligning feature distributions with EventMatch dual discriminators, it achieves an average improvement of 23.8% in accuracy and 77.1% in mIoU compared to source-only training on semantic segmentation tasks across three platforms: vehicle $\to$ UAV $\to$ quadruped robot.

## Background & Motivation

1. **Background**: Event cameras, with their characteristics of asynchronous operation, high temporal resolution, and high dynamic range, have demonstrated great advantages in high-speed and dynamic environments such as autonomous driving, aerial navigation, and robot perception. However, existing event camera perception datasets and methods are almost entirely focused on vehicle-mounted platforms (12 out of 15 public datasets are from ground vehicles), while data for platforms like UAVs and quadruped robots is extremely limited.

2. **Limitations of Prior Work**: Different robotic platforms exhibit distinct motion patterns, viewpoints, and environmental interactions: vehicles capture roads and obstacles, UAVs look down on ground features, and quadruped robots have unstable, close-to-ground viewpoints. These differences lead to completely different spatial-temporal activation patterns in event data. Traditional frame-based domain adaptation methods (such as AdaptSegNet, DACS, MIC, etc.) cannot handle the unique spatiotemporal characteristics of event data.

3. **Key Challenge**: Perception models for event cameras are typically trained on a single platform (primarily vehicles). When deploying to new platforms like UAVs or quadruped robots, the discrepancy in data distribution causes severe performance degradation. Meanwhile, annotating large amounts of data for every new platform is prohibitively expensive.

4. **Goal** To design a cross-platform domain adaptation framework tailored specifically to the characteristics of event camera data, enabling a perception model trained on one platform to be effectively transferred to other platforms.

5. **Key Insight**: Although the overall distribution of event data from different platforms varies significantly, there are overlaps in activation patterns within specific spatial regions. These shared high-activation regions can be leveraged as a bridge for domain adaptation.

6. **Core Idea**: By identifying cross-platform shared high-activation regions through the Event Activation Prior, constructing an intermediate domain using data mixing based on activation similarity, and then aligning source-intermediate-target domain features with dual discriminators to achieve progressive cross-platform adaptation.

## Method

### Overall Architecture
The inputs to EventFly are labeled source domain event voxel grids (e.g., vehicles) and unlabeled target domain event voxel grids (e.g., UAVs). The overall workflow consists of: (1) EAP computes an aggregated density map of the target domain to identify high-activation regions; (2) EventBlend generates a blended event voxel grid and corresponding labels based on the similarity between source samples and the target's aggregated density map; (3) Three data streams—source, target, and blended—pass through a shared feature extractor, and the two discriminators of EventMatch align the source-blended and blended-target features respectively.

### Key Designs

1. **Event Activation Prior EAP**:

    - **Function**: Identify high-activation regions in the target domain and guide the model to produce high-confidence predictions in these regions.
    - **Mechanism**: For each sub-region $\mathbf{S}$ in the target domain, EAP encourages minimizing the conditional entropy $H(y_\mathbf{S} | \mathbf{V}_\mathbf{S}, \mathbf{S})$. Through the principle of maximum entropy, the constraint $\mathbb{E}_\theta[H(\mathbf{V_S}, y_\mathbf{S} | \mathbf{S})] \leq c$ is transformed into a parameter prior $P(\theta) \propto \exp(-\lambda H(y_\mathbf{S} | \mathbf{V_S}, \mathbf{S}))$, which is integrated into the MAP training objective $C(\theta) = \mathcal{L}(\theta) - \lambda H_{\text{emp}}(y|\mathbf{V}, \mathbf{S})$. The intuition is: event cameras across different platforms exhibit persistent high-activation patterns in specific areas (e.g., lower part for vehicles = road, upper part for UAVs = sky). These regions possess highly consistent semantic patterns, making them suitable for low-entropy regularization.
    - **Design Motivation**: Traditional entropy minimization treats all regions equally, but a large portion of regions in event data have no activation (no information). Enforcing low entropy in these uninformative regions instead introduces noise. EAP focuses on event-active, information-rich regions, making the regularization more targeted.

2. **EventBlend Cross-Platform Data Mixing**:

    - **Function**: Construct intermediate domain data that merges features of the source and target domains.
    - **Mechanism**: First, compute the density map for each source sample $\mathbf{D}_i^\mathbf{v}(\mu,\nu) = \sum_{t=1}^T |\mathbf{V}_i^\mathbf{v}(t,\mu,\nu)|$ and the aggregated density map of the target domain $\tilde{\mathbf{D}}^\mathbf{d}$. Second, compute pixel-wise similarity $\mathbf{SIM}_i(\mu,\nu) = 1 - |\mathbf{D}_i^\mathbf{v}(\mu,\nu) - \tilde{\mathbf{D}}^\mathbf{d}(\mu,\nu)|$, calculated only at coordinates where at least one domain is active. Third, generate a binary mask $\mathcal{M}_i$ using a threshold $\tau$; regions with high similarity retain the source domain (labeled), while regions with low similarity are replaced with the target domain (requiring adaptation). Fourth, blend the labels using ground-truth source labels and target domain pseudo-labels (via a mean teacher).
    - **Design Motivation**: Unlike general-purpose mixing strategies such as CutMix or ClassMix, EventBlend's blending is driven by the physical properties of event activations—preserving reliable source domain annotations in areas where event patterns across the two platforms are similar, and introducing target domain data in highly discrepant areas to promote adaptation.

3. **EventMatch Dual-Discriminator Alignment**:

    - **Function**: Align the distributions of the source, target, and blended domains at the feature level.
    - **Mechanism**: Utilize two fully convolutional discriminators $\sigma_1$ and $\sigma_2$: $\sigma_1$ aligns source and blended domain features (maintaining source reliability), while $\sigma_2$ softly aligns blended domain features to the target domain (enhancing target adaptation by focusing on high-activation regions identified by EAP). This hierarchical design allows blended domain features to act as an intermediate bridge between the source and target domains—preserving the reliability of source domain supervision while gradually shifting towards the target distribution.
    - **Design Motivation**: Directly aligning source and target domains with a single discriminator yields poor results on event data due to the massive discrepancy in activation patterns across the two platforms. Using the blended domain as an intermediate stepping stone allows for a two-step progressive alignment, where the distribution gap at each step is much smaller, leading to more stable adversarial training.

### Loss & Training
The total loss includes: supervised cross-entropy loss $\mathcal{L}(\theta)$ on the source domain; EAP entropy regularization $-\lambda H_{\text{emp}}$ (restricted to high-activation areas); cross-entropy loss on the blended domain (utilizing blended labels); adversarial loss between source-blended for $\sigma_1$; and adversarial loss between blended-target for $\sigma_2$. Target domain pseudo-labels are generated online using a mean teacher or pre-computed offline.

## Key Experimental Results

### Main Results (Vehicle $\to$ UAV Adaptation)

| Method | Acc | mAcc | mIoU | fIoU |
|------|-----|------|------|------|
| Source-Only | 43.69 | 33.81 | 15.04 | 11.81 |
| AdaptSegNet | 49.14 | 35.38 | 21.16 | 12.15 |
| DACS | 59.81 | 42.01 | 27.07 | 16.14 |
| MIC | 63.11 | 45.60 | 28.87 | 17.46 |
| PLSR | 64.61 | 45.93 | 29.69 | 17.99 |
| **EventFly** | **69.17** | **48.20** | **32.67** | **20.01** |
| Target (Upper Bound) | 79.57 | 52.25 | 42.90 | — |

### Ablation Study (Based on reported gain analysis)

| Configuration | Acc Gain | mIoU Gain | Description |
|------|---------|----------|------|
| Source-Only $\to$ EventFly | +25.48 | +17.63 | Full framework vs. no adaptation |
| Compared to best baseline PLSR | +4.56 | +2.98 | Additional gain from EventFly |
| Compared to MIC | +6.06 | +3.80 | Outperforming context-guided methods |

### Key Findings
- EventFly achieves extremely significant improvements compared to Source-Only (Acc +25.5%, mIoU +17.6%), proving that cross-platform domain differences severely affect event camera perception.
- Compared to PLSR, the best-performing traditional frame-based domain adaptation method, EventFly still yields gains of 4.56% in Acc and 2.98% in mIoU, demonstrating that adaptation strategies tailored to event data characteristics are necessary.
- Adaptation for small target categories such as *fence*, *person*, and *sign* remains the most difficult (mIoU remains very low), as the appearance of these categories under the downward-facing viewpoint of UAVs differs drastically from the forward-facing vehicle perspective.
- A significant gap (~10% mIoU) still exists between the Target Upper Bound (79.57% Acc, 42.90% mIoU) and EventFly, indicating that cross-platform event domain adaptation remains an open challenge.

## Highlights & Insights
- **EAP (Event Activation Prior)** is the most insightful design—it leverages the unique "sparse activation" characteristic of event camera data to apply regularization only to regions with event activity, which is more reasonable than full-image entropy minimization. This concept can be transferred to domain adaptation for other sparse data modalities, such as LiDAR and radar.
- **EventBlend's physically-driven data mixing** outperforms random mixing—the mixing mask is determined by the physical similarity of event activation patterns between the two domains rather than random cropping. This philosophy of "letting data characteristics guide data augmentation" can be applied to other domain adaptation tasks.
- **EXPo benchmark** is the first large-scale event perception dataset (~90K samples) covering three platforms: vehicles, UAVs, and quadruped robots, filling a major evaluation gap.
- For the first time, the problem of "cross-platform adaptation for event cameras" is systematically defined and studied, establishing a complete formulation and evaluation protocol.

## Limitations & Future Work
- The model of event cameras may differ across platforms (e.g., resolution, contrast threshold C), but the impact of sensor discrepancies is not explicitly discussed.
- The threshold $\tau$ in EventBlend requires tuning; the optimal values may differ for different platform pairs.
- Currently, only semantic segmentation has been addressed. Cross-platform adaptation for other dense prediction tasks, such as object detection and depth estimation, remains to be explored.
- Pairing the three platforms yields 6 adaptation directions, but their difficulty levels vary significantly (e.g., vehicle $\to$ UAV vs. UAV $\to$ quadruped), which might require asymmetric adaptation strategies.
- Future work could consider leveraging temporal dimension information of event data (such as differences in motion patterns) to further enhance cross-platform adaptation.

## Related Work & Insights
- **vs Ev-Transfer / ESS**: These methods perform cross-modal domain adaptation (RGB $\to$ Event), whereas EventFly focuses on cross-platform event-to-event adaptation. The underlying source of domain discrepancy is completely different (modality discrepancy vs. platform discrepancy).
- **vs DACS / MIC**: These general-purpose domain adaptation methods are designed for frames and do not account for the sparse activation characteristics of event data. EventFly's EAP and EventBlend specifically exploit event activation patterns, achieving superior performance.
- **vs HPL-ESS**: Also performs event domain adaptation but is confined to frame-to-event migration, whereas EventFly is the first to consider event-to-event adaptation across different robotic platforms.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to define the cross-platform event domain adaptation problem; the designs of EAP and EventBlend effectively exploit event data characteristics.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation on the large-scale EXPo benchmark, comparing multiple baselines across 6 adaptation directions.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, systematic and complete method description, though math symbols are dense.
- **Value**: ⭐⭐⭐⭐ Pioneers the new direction of cross-platform adaptation for event cameras; the benchmark is highly valuable for subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)
- [\[CVPR 2026\] Event Structural Valley: A Unified Theoretical and Practical Framework for Event Camera Autofocus](../../CVPR2026/3d_vision/event_structural_valley_a_unified_theoretical_and_practical_framework_for_event_.md)
- [\[CVPR 2025\] SphereUFormer: A U-Shaped Transformer for Spherical 360 Perception](sphereuformer_a_u-shaped_transformer_for_spherical_360_perception.md)
- [\[CVPR 2025\] AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis](aerialmegadepth_learning_aerial-ground_reconstruction_and_view_synthesis.md)
- [\[CVPR 2025\] PS-EIP: Robust Photometric Stereo Based on Event Interval Profile](ps-eip_robust_photometric_stereo_based_on_event_interval_profile.md)

</div>

<!-- RELATED:END -->
