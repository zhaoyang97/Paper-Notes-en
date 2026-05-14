---
title: >-
  [Paper Note] Bridging Day and Night: Target-Class Hallucination Suppression in Unpaired Image Translation
description: >-
  [AAAI 2026 (Oral)][Autonomous Driving][Day-to-night translation] This paper is the first to systematically address the "target-class hallucination" problem in unpaired day-to-night image translation. By combining a dual-…
tags:
  - "AAAI 2026 (Oral)"
  - "Autonomous Driving"
  - "Day-to-night translation"
  - "hallucination suppression"
  - "Schrödinger Bridge"
  - "SAM2 pseudo-labels"
  - "prototype contrastive learning"
date: 2026-05-08
content_hash: c50138884625cb73
---

# Bridging Day and Night: Target-Class Hallucination Suppression in Unpaired Image Translation

**Conference**: AAAI 2026 (Oral)
**arXiv**: [2602.15383](https://arxiv.org/abs/2602.15383)
**Code**: Unavailable
**Area**: Autonomous Driving
**Keywords**: Day-to-night translation, hallucination suppression, Schrödinger Bridge, SAM2 pseudo-labels, prototype contrastive learning

## TL;DR
This paper is the first to systematically address the "target-class hallucination" problem in unpaired day-to-night image translation. By combining a dual-head discriminator (style head + SAM2 pseudo-label segmentation head) for hallucination detection and class-prototype contrastive learning for suppression, the method improves mAP from 15.08 to 17.40 (+15.5%) on BDD100K day-to-night domain adaptation detection, with traffic light AP improving by 31.7%.

## Background & Motivation

**Background**: Day-to-night unpaired image translation is a key technique for domain adaptation in autonomous driving. Existing approaches include GAN-based methods such as CycleGAN and CUT, diffusion-based methods such as UNSB, and instance-aware methods that incorporate bounding-box annotations (INIT, DUNIT, MGUIT, InstaFormer).

**Limitations of Prior Work**: Existing translation methods suffer from severe "target-class hallucination"—when translating to nighttime scenes, the generator erroneously synthesizes light effects in background regions that resemble traffic lights, headlights, and taillights. Although instance-aware methods improve translation quality within bounding boxes, they entirely neglect semantic consistency in background regions outside the boxes.

**Key Challenge**: Traditional discriminators focus solely on whether the style resembles nighttime, yet the visual characteristics of nighttime lighting are highly similar to those of target-class objects (traffic lights, headlights). The discriminator effectively "rewards" hallucinations—generating more light-like artifacts makes the image appear more like a genuine nighttime scene.

**Goal**: (a) How to precisely localize hallucination pixels using only bounding-box annotations (without pixel-level labels)? (b) How to constrain the semantic boundary between background and foreground in feature space? (c) How to intervene during the intermediate translation steps rather than applying post-hoc correction?

**Key Insight**: SAM2 is leveraged to generate pseudo-segmentation labels from bounding boxes, enabling pixel-level hallucination detection; class prototypes serve as semantic anchors, and contrastive learning suppresses hallucinations in feature space.

**Core Idea**: A dual mechanism of "hallucination detection + hallucination suppression" is embedded within the multi-step Schrödinger Bridge translation framework—the dual-head discriminator identifies the locations of hallucinations, while prototype contrastive learning pushes hallucination features away from target-class representations.

## Method

### Overall Architecture
Built upon the UNSB Schrödinger Bridge multi-step translation framework. A daytime input image $x_0$ is progressively translated through a Markov chain (each step mixing the current state, the predicted target, and Gaussian noise). Intermediate predicted images are fed into the dual-head discriminator for hallucination detection; detected hallucination features are pushed away from target-class prototypes via prototype contrastive learning, ultimately yielding a semantically consistent nighttime image $x_{t_N}$.

### Key Designs

1. **Schrödinger Bridge Multi-Step Translation**:

    - Function: Models day-to-night translation as a Markov chain that progressively transitions from the source domain to the target domain.
    - Mechanism: Given a time partition $\{t_j\}_{j=0}^N$, at each step a neural network predicts the target-domain image $x_1(x_{t_j})$, and the next state is generated via the interpolation formula: $x_{t_{j+1}} = s_{j+1} x_1(x_{t_j}) + (1-s_{j+1}) x_{t_j} + \sigma_{j+1} \epsilon$, where $s_{j+1}$ controls the interpolation ratio.
    - Design Motivation: Multi-step translation reduces the per-step domain gap and allows hallucination detection and suppression to be applied at intermediate steps rather than only after one-shot translation.

2. **Dual-Head Discriminator (Hallucination-Aware Discriminator)**:

    - Function: Simultaneously evaluates global style authenticity and pixel-level semantic segmentation to detect which pixels constitute hallucinations.
    - Mechanism: A frozen Hiera-T (SAM2 visual backbone) encoder $D_{enc}$ is shared. The style head $D_{sty}$ assesses whether the global style resembles nighttime; the segmentation head $D_{seg}$ (UNet decoder) performs pixel-level semantic segmentation to identify pixels belonging to target classes. The segmentation head is trained with SAM2-generated pseudo-segmentation labels—instance masks are generated by prompting SAM2 with bounding boxes, and bounding boxes enlarged by 10% are used for a second-pass confirmation (retaining results with IoU > 0.9).
    - Design Motivation: A traditional discriminator that considers only style rewards hallucinations. The addition of a segmentation head enables the discriminator to distinguish between genuine nighttime objects and synthesized false objects, resolving the information asymmetry problem.

3. **Hallucination Loss $\mathcal{L}_{hl}$**:

    - Function: Penalizes any pixel predicted as a target class in background regions outside bounding boxes.
    - Mechanism: $\mathcal{L}_{hl} = \frac{1}{|S_{bg}|} \sum_{(w,h) \in S_{bg}} \sum_{c=1}^{C} (\text{softmax}(\hat{S})_{cwh})^2$, applying squared penalties to the predicted probabilities of all foreground classes in background regions.
    - Design Motivation: Directly localizes and penalizes target-class activations in the background from segmentation predictions, providing the most direct hallucination suppression signal.

4. **Class-Prototype Contrastive Suppression**:

    - Function: Pushes hallucination features away from target-class prototypes in feature space.
    - Mechanism: A prototype $p_c$ is constructed for each target class $c$ (a class-mean feature vector updated via EMA) from real annotated instances in the target domain. Hallucination pixel features serve as anchors, features at the corresponding locations in the source image serve as positives, and features at other locations plus class prototypes serve as negatives. InfoNCE loss: $\mathcal{L}_{supp} = -\log \frac{\exp(\hat{\mathbf{v}} \cdot \mathbf{v}^+ / \tau)}{\exp(\hat{\mathbf{v}} \cdot \mathbf{v}^+ / \tau) + \sum_n \exp(\hat{\mathbf{v}} \cdot \mathbf{v}_n^- / \tau) + \text{PDist}}$
    - Design Motivation: $\mathcal{L}_{hl}$ supervises only in output space, while prototype contrastive learning establishes semantic boundaries in feature space—the two are complementary. EMA prototypes address the issue of certain classes being absent within a single batch.

### Loss & Training
Total loss: $\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{adv} + \lambda_2 \mathcal{L}_{SB} + \lambda_3 \mathcal{L}_{seg} + \lambda_4 \mathcal{L}_{cont} + \lambda_5 \mathcal{L}_{supp} + \lambda_6 \mathcal{L}_{hl}$, where $\lambda_1$–$\lambda_5 = 1$ and $\lambda_6 = 0.2$. Training runs for 100 epochs on 8× RTX 3090 GPUs with the Adam optimizer, batch size 8, and learning rate 0.0001.

## Key Experimental Results

### Main Results

| Method | mAP | Person | Car | T. Light | T. Sign |
|--------|-----|--------|-----|----------|---------|
| Lower Bound (day training) | 13.75 | 12.99 | 25.21 | 8.28 | 18.55 |
| CUT | 14.10 | 14.13 | 28.31 | 5.36 | 19.19 |
| UNSB | 14.27 | 14.65 | 28.35 | 5.93 | 14.88 |
| MGUIT | 15.08 | 14.52 | 27.48 | 6.18 | 18.83 |
| InstaFormer | 14.93 | 14.04 | 27.25 | 6.33 | 18.19 |
| **Ours** | **17.40** | **15.35** | **30.01** | **8.55** | **22.01** |
| Upper Bound (night training) | 17.86 | 14.43 | 32.59 | 11.93 | 23.83 |

### Ablation Study

| Configuration | mAP | T. Light AP | Note |
|---------------|-----|-------------|------|
| w/o $\mathcal{L}_{hl}$ & $\mathcal{L}_{supp}$ | 14.11 | 5.48 | Baseline, no hallucination suppression |
| w/o $\mathcal{L}_{supp}$ | 15.55 | 7.01 | Segmentation detection only, no feature suppression |
| w/o $\mathcal{L}_{hl}$ | 16.43 | 7.45 | Feature suppression only, no pixel penalty |
| Full model | **17.40** | **8.55** | Both components combined, best performance |

### Key Findings
- The two hallucination suppression components are complementary: $\mathcal{L}_{hl}$ supervises in output space while $\mathcal{L}_{supp}$ constrains in feature space; their combination improves mAP by 1.85 and 0.97 respectively over each component used alone.
- The traffic light category benefits most from hallucination suppression (AP: 5.48 → 8.55), as it is the most susceptible to erroneous synthesis.
- The method nearly reaches the Upper Bound (17.40 vs. 17.86), demonstrating the effectiveness of hallucination suppression.
- State-of-the-art results are also achieved on cross-dataset (KITTI → Cityscapes) and cross-weather tasks.

## Highlights & Insights
- **Pioneering Problem Formulation**: The paper is the first to systematically define and quantify the "target-class hallucination" problem, which had been overlooked by all prior methods. The key insight is that the discriminator's style evaluation is itself an "accomplice" in generating hallucinations.
- **Effective Use of SAM2 Pseudo-Labels**: Prompting SAM2 with bounding boxes to generate pixel-level pseudo-labels elegantly resolves the bottleneck of having bounding-box annotations but no segmentation labels; the 10%-enlargement second-pass confirmation mechanism further improves label quality.
- **Intermediate-Step Intervention**: The multi-step nature of the Schrödinger Bridge is exploited to suppress hallucinations during translation rather than post-hoc, which is more efficient. This paradigm of "in-process supervision" is transferable to other multi-step generation tasks.

## Limitations & Future Work
- Hallucination suppression is limited to object-detection categories with bounding-box annotations; hallucinations in other semantic categories (e.g., road markings, building textures) remain unaddressed.
- The method depends on SAM2's segmentation quality, which may degrade under extreme lighting conditions or heavy occlusion.
- EMA prototypes may be insufficiently fine-grained for categories with high intra-class variation (e.g., vehicles of different models); multi-prototype or clustering approaches could be explored.
- The framework is extensible to other domain adaptation scenarios (e.g., clear → rain, synthetic → real).

## Related Work & Insights
- **vs. InstaFormer**: InstaFormer employs a Transformer encoder to improve in-box translation with contrastive learning, but entirely ignores background regions outside boxes. The proposed dual-head discriminator fills this blind spot.
- **vs. CUT**: CUT applies contrastive learning for style transfer but lacks object-level constraints. The proposed $\mathcal{L}_{supp}$ also uses InfoNCE but with a fundamentally different objective—suppressing hallucinations rather than maintaining style consistency.
- **vs. UNSB**: Hallucination detection and suppression modules are directly incorporated into the UNSB framework, demonstrating that multi-step translation frameworks are better suited than single-step methods for embedding intermediate supervision.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel problem formulation; technically sound combination, though not revolutionary.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset evaluation with detailed ablations and qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear and illustrations are intuitive.
- Value: ⭐⭐⭐⭐ Practically meaningful for the domain adaptation community; the hallucination suppression approach is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration](../../CVPR2026/autonomous_driving/lirec-net_a_target-free_and_learning-based_network_for_lidar_rgb_and_event_calib.md)
- [\[AAAI 2026\] Hierarchical Prompt Learning for Image- and Text-Based Person Re-Identification](hierarchical_prompt_learning_for_image-_and_text-based_person_re-identification.md)
- [\[AAAI 2026\] MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation](mambaseg_harnessing_mamba_for_accurate_and_efficient_image-e.md)
- [\[ICLR 2026\] Single Pixel Image Classification using an Ultrafast Digital Light Projector](../../ICLR2026/autonomous_driving/single_pixel_image_classification_using_an_ultrafast_digital_light_projector.md)
- [\[CVPR 2026\] SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](../../CVPR2026/autonomous_driving/searchad_large-scale_rare_image_retrieval_dataset_for_autonomous_driving.md)

</div>

<!-- RELATED:END -->
