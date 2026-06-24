---
title: >-
  [Paper Note] Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation
description: >-
  [CVPR 2025][Human Understanding][Synthetic-to-real domain transfer] This paper presents the first systematic study of the synthetic-to-real domain gap in 3D hand pose estimation. By designing a controllable data synthesis pipeline, the authors decompose and analyze the impacts of four key factors: forearms, spectral statistics, pose distribution, and object occlusion. The study demonstrates that with proper integration of these factors, purely synthetic data can achieve accur…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Synthetic-to-real domain transfer"
  - "3D hand pose estimation"
  - "domain gap analysis"
  - "data synthesis pipeline"
  - "occlusion analysis"
date: 2026-05-08
content_hash: 3077a5422d40cc8f
---

# Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation

**Conference**: CVPR 2025  
**arXiv**: [2503.19307](https://arxiv.org/abs/2503.19307)  
**Code**: [https://github.com/delaprada/HandSynthesis](https://github.com/delaprada/HandSynthesis)  
**Area**: Human Understanding / 3D Hand Pose Estimation  
**Keywords**: Synthetic-to-real domain transfer, 3D hand pose estimation, domain gap analysis, data synthesis pipeline, occlusion analysis

## TL;DR

This paper presents the first systematic study of the synthetic-to-real domain gap in 3D hand pose estimation. By designing a controllable data synthesis pipeline, the authors decompose and analyze the impacts of four key factors: forearms, spectral statistics, pose distribution, and object occlusion. The study demonstrates that with proper integration of these factors, purely synthetic data can achieve accuracy on par with real data.

## Background & Motivation

**Background**: 3D hand pose estimation heavily relies on large-scale annotated 3D data for training, but labeling real-world data is expensive and time-consuming. While synthetic data has reached state-of-the-art levels in face recognition and human body pose estimation, a significant synthetic-to-real domain gap still persists in hand pose estimation.

**Limitations of Prior Work**: Existing synthetic hand datasets (e.g., RHD, ObMan, DARTset, RenderIH) possess various limitations: simplistic backgrounds, limited textures, lack of forearms, or absent object interaction. Because these datasets differ from real data across multiple dimensions simultaneously, isolating the individual contribution of each factor has been impossible.

**Key Challenge**: Compared to faces and human bodies, hands present more severe self-occlusions and object occlusions. Coupled with the varying skeletal topologies used across different datasets (such as MANO with 21 joints vs. NIMBLE with 25 joints), the origins of the domain gap are complex and highly intertwined.

**Core Idea**: To design a controllable data synthesis pipeline capable of independently regulating various image components (hand texture, background, arm, object, and pose distribution), thereby decomposing and analyzing the contribution of each component to the domain gap.

## Method

### Overall Architecture

A high-quality synthesis pipeline based on the NIMBLE hand model and Blender rendering is designed to support independent control of hand textures (linear interpolation of 38 genuine textures), backgrounds (669 HDRI scenes), pose distributions, arms, and object occlusions. By segmenting arms and objects from real images using Grounding DINO and SAM, and combining them into synthetic images, controllable comparative experiments are enabled.

### Key Designs

1. **High-Quality Hand Model Rendering**:

    - Function: Renders 3D hands with realistic bones, muscles, skin, and textures based on the NIMBLE model.
    - Mechanism: NIMBLE provides a finer mesh than MANO (5,990 vertices vs. 778 vertices). The texture model $\mathcal{A}(\alpha) = \bar{A} + \Phi\alpha$ achieves diversity through linear interpolation across 38 real hand texture assets, including diffuse, specular, and normal maps.
    - Design Motivation: MANO textures are limited and unrealistic; thus, a higher-fidelity hand model is required to bridge the domain gap.

2. **Decomposition & Composition**:

    - Function: Segments forearms and objects from real images and blends them into synthetic images.
    - Mechanism: Bounding boxes are extracted using Grounding DINO, and segmentation masks are generated with SAM. The composition is computed as: $\tilde{I}_{syn}^j = (1 - M_{obj}^i - M_{arm}^i) \odot I_{syn}^j + M_{obj}^i \odot I_{real}^i + M_{arm}^i \odot I_{real}^i$.
    - Design Motivation: This approach is more practical than direct rendering of arm and object assets, and provides a superior controlled-variable experimental setup.

3. **Amplitude Spectrum Augmentation**:

    - Function: Enhances synthetic-to-real robustness by perturbing amplitudes in the frequency domain.
    - Key findings: Synthesized images exhibit smaller amplitude variance across the entire spectrum compared to real images (not limited to high frequencies, see Fig. 2b). Amplitude spectrum augmentation enhances model robustness by perturbing amplitude information while preserving phase spectrum information (representing hand structures).
    - Contribution Quantification: Removing Amplitude Spectrum Augmentation increases the PA-MPJPE on SynFrei from 1.02 to 1.11 (+0.09 cm).

4. **VAE Object Occlusion Prior**:

    - Function: Reconstructs occluded hand joints using a VAE.
    - Mechanism: A VAE prior is trained with the loss function $L_{VAE} = \lambda L_{KL} + \|\hat{x}_{3D} - x_{3D}\|_2^2$. Random masking of joints during training enhances reconstruction diversity. At inference, the pretrained prior refines the predicted joints.
    - Key findings: The occlusion prior significantly reduces the domain gap in object-interaction scenarios, allowing the model to associate specific hand poses with specific objects.

### Loss & Training

S2HAND, CMR, METRO, MeshGraphormer, and simpleHand are employed as baselines, trained respectively on FreiHAND and the synthetic SynFrei dataset. Pose distributions are kept consistent by fitting the NIMBLE mesh to the MANO mesh. Rendering takes approximately 1 second per image on an RTX A5000.

## Key Experimental Results

### Main Results: Competing Synthetic vs. Real Data Training (Tested on FreiHAND Evaluation Set)

| Method | Real PA-MPJPE/MPVPE | Synthetic PA-MPJPE/MPVPE | Syn→Real Ratio |
|------|:-:|:-:|:-:|
| S2HAND | 0.99/1.02 | 1.02/1.05 | **97%** |
| CMR | 0.77/0.78 | 0.85/0.88 | 91% |
| METRO | 0.69/0.71 | 0.78/0.79 | 88% |
| MeshGraphormer | 0.69/0.70 | 0.76/0.78 | 91% |
| simpleHand | 0.65/0.66 | 0.77/0.79 | 84% |

### Ablation Study: Component Contribution to the Domain Gap

| Component | Arm | Amplitude Aug. | Object | SynFrei PA-MPJPE↓ | SynDex PA-MPJPE↓ |
|------|:-:|:-:|:-:|:-:|:-:|
| (i) No Arm | ✗ | ✓ | ✓ | 1.07 | 0.90 |
| (ii) No Amplitude Aug. | ✓ | ✗ | ✓ | 1.11 | 0.89 |
| (iii) No Object | ✓ | ✓ | ✗ | 1.07 | 0.95 |
| (iv) Random Arm + Object | ~✓ | ✓ | ~✓ | 1.04 | 0.92 |
| (v) Full | ✓ | ✓ | ✓ | **1.02** | **0.87** |

### Key Findings

- **Purely synthetic data achieves 97% of real performance**: S2HAND trained on synthetic data exhibits a PA-MPJPE only 0.03 cm worse than when trained on real data, demonstrating for the first time that hand pose estimation can rely almost entirely on synthetic data.
- **The forearm is a critical cue**: Without forearms, models tend to misidentify wrist positions (confusing forearm regions as wrists). Incorporating the forearm reduces PA-MPJPE by 0.05 cm.
- **Amplitude Spectrum Augmentation is indispensable**: This is the single most contributing component (-0.09 cm), addressing the lack of frequency-domain diversity in synthetic images.
- **Saturation effect in pose distribution**: Utilizing only 20% of real poses achieves 90% of the performance, and 40% achieves 97%. This indicates that the primary value of synthetic data lies in learning visual representations; hence, an excessive number of core hand poses is not necessary.
- **Background/texture diversity also saturates**: 300 HDRI scenes (approx. 50% of the assets) are sufficient; further additions yield no significant improvement.
- **Mixed training is superior**: Training on a mix of real and synthetic data outperforms training on real data alone in both in-domain and cross-domain generalization.

## Highlights & Insights

- **Systematic Decomposition Analysis**: The hand pose domain gap is decomposed into four orthogonal dimensions (appearance, pose, occlusion, skeletal topology) for the first time, providing definitive conclusions after analyzing each dimension independently. This "decomposition-analysis-composition" paradigm serves as an exemplary methodology.
- **Surprising performance with random RGB values**: Even when replacing real arm/object mask regions with random RGB values, the model still achieves 95% of performance. This suggests that the model mainly learns the concept of "an object occluding the hand" rather than the precise appearance of the arm or object.
- **Practical value of saturation effects**: Finding that 20% of poses and 50% of background/texture can achieve near-optimal performance provides critical guidance for synthetic data generation; infinite expansion of data diversity is unnecessary.
- **Impact of skeletal topology differences**: Discrepancies in joint definitions between NIMBLE and MANO cause the PA-MPJPE to increase from 1.02 to 1.28 (+25%), an oversight that significantly affects results.

## Limitations & Future Work

- The study evaluates only single-hand scenarios (FreiHAND, Dex-YCB), leaving two-hand interactions and hand-hand occlusions unexplored.
- In the composition analysis, arms and objects are segmented from real images, making it impossible to fully control all variables.
- Stronger backbones (e.g., simpleHand) generalize worse on synthetic data (84% vs. 97% for S2HAND), implying that larger models are more prone to overfitting the synthetic data distribution.
- The object occlusion prior relies on existing hand-object interaction datasets (such as Dex-YCB), leaving a remaining gap for unseen objects.

## Related Work & Insights

- **vs. DARTset**: DARTset contains a more diverse pose distribution but lacks active backgrounds, object interactions, and amplitude augmentation. The dataset presented in this work is only half the size of DARTset yet achieves superior performance (CMR: 0.85 vs. 2.56 PA-MPJPE), proving that quality matters more than quantity.
- **vs. RenderIH**: RenderIH focuses on two-hand interactions but suffers from low-resolution backgrounds (1K), whereas this work utilizes 4K HDRI backgrounds to achieve higher realism.
- **General insights for synthetic-to-real transfer**: The decomposition analysis methodology proposed in this paper can be transferred to other synthetic data domains, such as human bodies and faces.

## Rating

- Novelty: ⭐⭐⭐⭐ The first systematic analysis of the hand pose domain gap with a delicately designed controllable synthesis pipeline, although the core technologies (NIMBLE + Blender rendering) rely on existing tools.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly exhaustive analyses spanning 5 methods and multi-dimensional ablations (forearm, texture, background, pose, object, and skeleton).
- Writing Quality: ⭐⭐⭐⭐ Clear analytical logic and rich charts/plots (spectral analyses, saturation curves, occlusion level analyses) make it an exemplary academic work.
- Value: ⭐⭐⭐⭐ Provides best practices for using synthetic data in the hand pose estimation community, lessening the dependency on expensive real-world annotations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation](unihope_a_unified_approach_for_hand-only_and_hand-object_pose_estimation.md)
- [\[CVPR 2025\] HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation](hipart_hierarchical_pose_autoregressive_transformer_for_occluded_3d_human_pose_e.md)
- [\[ECCV 2024\] 3D Hand Pose Estimation in Everyday Egocentric Images](../../ECCV2024/human_understanding/3d_hand_pose_estimation_in_everyday_egocentric_images.md)
- [\[CVPR 2025\] WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild](wilor_end-to-end_3d_hand_localization_and_reconstruction_in-the-wild.md)
- [\[CVPR 2025\] GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding](ga3ce_unconstrained_3d_gaze_estimation_with_gaze-aware_3d_context_encoding.md)

</div>

<!-- RELATED:END -->
