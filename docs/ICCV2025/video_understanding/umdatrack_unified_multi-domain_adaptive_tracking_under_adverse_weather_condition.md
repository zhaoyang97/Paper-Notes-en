---
title: >-
  [Paper Note] UMDATrack: Unified Multi-Domain Adaptive Tracking Under Adverse Weather Conditions
description: >-
  [ICCV 2025][Video Understanding][visual object tracking] UMDATrack proposes the first unified multi-domain adaptive tracking framework. It leverages text-guided diffusion models to synthesize a small number (<2% of frames) of unlabeled multi-weather videos, employs Domain-Customized Adapters (DCA) to efficiently transfer object representations across weather domains, and introduces Target-aware Confidence Alignment (TCA) based on optimal transport to enhance cross-domain localization consistency. The framework substantially outperforms existing state-of-the-art trackers under nighttime, hazy, and rainy conditions.
tags:
  - ICCV 2025
  - Video Understanding
  - visual object tracking
  - multi-domain adaptation
  - adverse weather
  - domain-customized adapter
  - optimal transport
  - text-to-image diffusion
  - teacher-student
date: 2026-05-08
content_hash: d03af417d7e3f842
---

# UMDATrack: Unified Multi-Domain Adaptive Tracking Under Adverse Weather Conditions

**Conference**: ICCV 2025
**arXiv**: [2507.00648](https://arxiv.org/abs/2507.00648)
**Code**: [https://github.com/Z-Z188/UMDATrack](https://github.com/Z-Z188/UMDATrack)
**Area**: Visual Object Tracking / Domain Adaptation / Adverse Weather
**Keywords**: visual object tracking, multi-domain adaptation, adverse weather, domain-customized adapter, optimal transport, text-to-image diffusion, teacher-student

## TL;DR
UMDATrack proposes the first unified multi-domain adaptive tracking framework. It leverages text-guided diffusion models to synthesize a small number (<2% of frames) of unlabeled multi-weather videos, employs Domain-Customized Adapters (DCA) to efficiently transfer object representations across weather domains, and introduces Target-aware Confidence Alignment (TCA) based on optimal transport to enhance cross-domain localization consistency. The framework substantially outperforms existing state-of-the-art trackers under nighttime, hazy, and rainy conditions.

## Background & Motivation

Visual object tracking (VOT) has achieved remarkable progress under well-lit daytime conditions, yet suffers severe performance degradation under adverse weather (nighttime, haze, rain, etc.):

**Large domain shift**: Mainstream trackers (OSTrack, ARTrackV2, etc.) trained on daytime datasets such as LaSOT and TrackingNet exhibit significant performance drops when transferred to adverse weather due to substantial appearance distribution shifts.

**Limitations of single-domain methods**: Existing cross-domain trackers (e.g., UDAT) are designed for only a single weather condition. UDAT, for instance, is optimized for nighttime and suffers drastic performance degradation in hazy conditions, lacking multi-condition generalization.

**High cost of data synthesis**: Existing methods require generating large volumes of target-domain samples for knowledge transfer, which is time-consuming and processes each domain independently, neglecting the intrinsic associations of target objects across domains.

**Redundant parameters**: Introducing independent feature alignment parameters for each weather condition prohibits efficient cross-domain interaction.

**Core Motivation**: Can a unified framework be designed that covers multiple weather conditions with minimal synthetic data and achieves efficient multi-domain transfer via lightweight adapters?

## Method

### Overall Architecture
UMDATrack consists of three core components: a Controllable Scene Generator (CSG), an encoder network with Domain-Customized Adapters (DCA), and a localization head with Target-aware Confidence Alignment (TCA). The overall training follows a teacher-student paradigm.

### Component 1: Controllable Scene Generator (CSG)

- Employs Stable Diffusion-Turbo to translate daytime-domain video frames into various weather conditions.
- Input: source-domain video frame $x$ + text prompt $c_X$ (e.g., "Car in the night/haze/rain/snow").
- Output: target-domain frame $y = G_{SDT}(x, c_X, \varepsilon)$, with structural details preserved via skip connections and Zero-Convs.
- **Efficient synthesis**: Requires only 1–4 inference steps; switching weather conditions is as simple as changing the text prompt.
- **Minimal data**: Only GOT-10k is used for synthesis; target-domain frames account for less than 2% of source-domain frames.

### Component 2: Domain-Customized Adapter (DCA)

DCA transfers object representations from the source domain to multiple target domains without retraining the entire backbone:

1. **Frozen backbone**: The pretrained ViT-Base remains fixed; only the DCA module is trained.
2. **Architecture**:
    - A lightweight ResNet block transforms the target-domain search image $X^T$ into a query $Q \in \mathbb{R}^{K \times C}$.
    - A Gaussian random variable is initialized as a learnable token bank $B \in \mathbb{R}^{L' \times C}$.
    - The token bank is projected into Key and Value via two FC layers.
    - Structured tokens $S = \text{Softmax}(QK^T/\sqrt{d_k})V$ are computed, encoding the latent image content of the target domain.
3. **Injection**: Structured tokens $S$ are concatenated with source-domain template-search tokens and fed into the frozen ViT, enabling rapid convergence to the optimal representation for each weather condition.
4. **Training efficiency**: Only 50 additional epochs are required to train the DCA for each weather condition, with no repeated backbone training. Total training time across all weather conditions is approximately one and a half days.

### Component 3: Target-aware Confidence Alignment (TCA)

TCA aligns the localization confidence distributions of the source and target domains based on optimal transport (OT) theory:

1. **Pseudo-label propagation**: The teacher network generates pseudo-labels for the target domain; the student network is updated accordingly.
2. **Challenge**: Pseudo-labels may be noisy, and incorrect labels can mislead target state prediction.
3. **OT cost matrix design**: Both spatial and confidence discrepancies are considered:
    - Confidence cost $C^{Conf}$: measures the difference in response values at the highest-confidence positions between student and teacher.
    - Position cost $C^{Pos}$: measures the spatial displacement between the highest-confidence positions of student and teacher.
    - Total cost $C = C^{Conf} + C^{Pos}$.
4. **Position-Sensitive Optimal Transport loss (PSOT)**: The dual form of the OT problem is solved via the Sinkhorn distance algorithm to minimize the cost of transporting the source-domain confidence distribution to the target domain.
5. **Joint loss**: $\mathcal{L} = \mathcal{L}_t + \lambda \mathcal{L}_p$, where $\mathcal{L}_t = \mathcal{L}_{cls} + \beta \mathcal{L}_1 + \gamma \mathcal{L}_{GIoU}$ (classification + $L_1$ regression + GIoU).

### Loss & Training
- **Two-stage training**:
    - Stage 1 (Backbone training, 250 epochs): DCA is not introduced; domain adaptation is performed using target supervision loss + PSOT loss. Four source-domain datasets + three synthetic datasets (ratio 1:1:1:1:4:4:4).
    - Stage 2 (DCA training, 50 epochs): Backbone is frozen; only the DCA module is trained.
- At inference, template features are initialized and cached, then reused for subsequent frames.

## Key Experimental Results

### Synthetic Dataset Results

| Tracker | GOT-10k-Foggy AO | DTB70-Foggy AUC/P | GOT-10k-Dark AO | DTB70-Dark AUC/P | GOT-10k-Rainy AO | DTB70-Rainy AUC/P |
|---------|---:|---:|---:|---:|---:|---:|
| **UMDATrack** | **66.6** | **66.21/86.05** | **65.4** | **66.07/85.72** | **68.5** | **66.75/87.60** |
| DCPT | 61.6 | 58.31/75.33 | 62.4 | 61.87/80.11 | 62.3 | 61.68/82.56 |
| UDAT-CAR | 51.5 | 50.21/69.41 | 56.8 | 57.20/75.80 | 59.5 | 56.42/75.36 |
| ARTrackV2 | 64.8 | 62.25/80.15 | 63.1 | 62.87/80.56 | 66.2 | 63.84/83.32 |

UMDATrack achieves comprehensive superiority across all three weather conditions on synthetic datasets. On DTB70-Dark, it surpasses the second-best method by 3.06% AUC and 4.15% Precision.

### Real-World Dataset Results

| Tracker | NAT2021 AUC/P | UAVDark70 AUC/P | AVisT AUC/P |
|---------|---:|---:|---:|
| **UMDATrack** | **54.58/70.78** | **60.05/73.35** | **60.50/59.01** |
| DCPT | 52.55/69.01 | 56.86/70.16 | 55.66/52.41 |
| UDAT-CAR | 48.75/65.96 | 51.25/70.22 | 38.91/33.65 |

State-of-the-art performance is also achieved on real-world nighttime and mixed adverse weather datasets.

### Efficiency Comparison
- **Inference speed**: UMDATrack achieves the highest inference speed with relatively low parameter count and computational cost.
- **Training time**: Total training across all weather conditions requires only approximately one and a half days (single backbone training + 50 epochs of DCA per domain).

## Highlights & Insights

- **First unified multi-domain adaptive tracker**: A single framework covers multiple adverse weather conditions (nighttime, haze, rain, snow) without requiring condition-specific model designs.
- **Efficient transfer with minimal synthetic data**: Fewer than 2% of source-domain frames are synthesized, substantially reducing data collection and annotation costs.
- **Elegant DCA design**: Frozen backbone + learnable token bank + structured attention enables flexible multi-domain adaptation with negligible parameter overhead. Extending to new weather conditions requires only changing the text prompt.
- **Novel application of OT theory**: Optimal transport is applied to cross-domain localization alignment in tracking, jointly considering spatial and confidence dimensions—a more comprehensive formulation than simple KL divergence constraints.
- **Synergy of teacher-student paradigm and DCA**: Structured tokens generated by DCA are injected into the frozen backbone, while EMA updates the teacher model, forming a closed loop of progressive domain knowledge transfer.

## Limitations & Future Work

- CSG relies on the quality of Stable Diffusion-Turbo; synthesis fidelity may be insufficient for certain complex weather conditions (e.g., heavy rain combined with strong wind).
- Text prompt templates are relatively simple (e.g., "Car in the night") and may not capture complex real-world weather variations.
- DCA still requires 50 independent training epochs per weather condition, which introduces non-trivial overhead when the number of weather types is large.
- Experiments cover only four conditions (fog, night, rain, snow); extreme scenarios such as sandstorms and waterlogged surface reflections remain unvalidated.
- The choice of ViT-Base backbone may limit representational capacity; larger models are not explored.
- Sensitivity analysis of hyperparameters such as $\lambda$ in the PSOT loss is insufficient.

## Related Work & Insights

- **vs. UDAT**: UDAT targets nighttime only, employing a Transformer bridging layer for single-domain knowledge transfer. UMDATrack covers multiple domains via a unified DCA + CSG design with lower data requirements.
- **vs. DCPT**: DCPT also addresses cross-domain tracking but processes each domain independently. UMDATrack achieves parameter-efficient multi-domain adaptation through a shared backbone with domain-specific DCAs.
- **vs. SAM-DA**: SAM-DA relies on enhancement to unify representations, following a "Track-by-Enhancement" paradigm in which enhancement quality becomes the performance bottleneck. UMDATrack performs domain adaptation directly in feature space, enabling a more end-to-end approach.
- **Inspiration from ControlNet**: The use of Zero-Convs in CSG to preserve structural details is inspired by ControlNet, demonstrating the value of controllable generation for tracking data augmentation.
- Multi-Target Domain Adaptation (MTDA) has been explored in detection and classification; this work is the first to introduce it to tracking, inspiring future applications of MTDA in fine-grained visual tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ First unified multi-domain adaptive tracking framework; DCA is concise and effective; applying OT to tracking localization alignment is a notable highlight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 8 benchmarks across synthetic and real-world datasets; comprehensive evaluation under three weather conditions; comparison against 14+ baselines.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear; problem motivation is well articulated; the comparison figure of three tracking paradigms is intuitive.
- Value: ⭐⭐⭐⭐ Addresses practical pain points in adverse weather tracking; the unified framework design provides meaningful guidance for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions](../../CVPR2026/video_understanding/egoxtreme_a_dataset_for_robust_object_pose_estimation_in_egocentric_views_under_.md)
- [\[ICCV 2025\] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](aim_adaptive_inference_of_multi-modal_llms_via_token_merging_and_pruning.md)
- [\[AAAI 2026\] Lifelong Domain Adaptive 3D Human Pose Estimation](../../AAAI2026/video_understanding/lifelong_domain_adaptive_3d_human_pose_estimation.md)
- [\[ICCV 2025\] What You Have is What You Track: Adaptive and Robust Multimodal Tracking](what_you_have_is_what_you_track_adaptive_and_robust_multimodal_tracking.md)
- [\[CVPR 2026\] UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking](../../CVPR2026/video_understanding/utptrack_towards_simple_and_unified_token_pruning_for_visual_tracking.md)

</div>

<!-- RELATED:END -->
