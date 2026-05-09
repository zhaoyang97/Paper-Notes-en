---
title: >-
  [Paper Note] FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation
description: >-
  [CVPR 2026][Human Understanding][cattle pose estimation] This paper proposes FSMC-Pose, a lightweight top-down framework that achieves cattle mounting pose estimation in dense and cluttered farm environments via the frequency-spatial fusion backbone CattleMountNet and the multiscale self-calibration head SC2Head, attaining 89% AP with only 2.698M parameters.
tags:
  - CVPR 2026
  - Human Understanding
  - cattle pose estimation
  - mounting detection
  - estrus recognition
  - frequency-spatial fusion
  - lightweight model
date: 2026-05-08
content_hash: a70f72c1e726ab2d
---

# FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation

**Conference**: CVPR 2026
**arXiv**: [2603.16596](https://arxiv.org/abs/2603.16596)
**Code**: [Github](https://github.com)
**Area**: Human/Animal Understanding
**Keywords**: cattle pose estimation, mounting detection, estrus recognition, frequency-spatial fusion, lightweight model

## TL;DR

This paper proposes FSMC-Pose, a lightweight top-down framework that achieves cattle mounting pose estimation in dense and cluttered farm environments via the frequency-spatial fusion backbone CattleMountNet and the multiscale self-calibration head SC2Head, attaining 89% AP with only 2.698M parameters.

## Background & Motivation

1. **Background**: Cattle mounting behavior is a key visual indicator of estrus, and its accurate recognition is critical for livestock production efficiency.
2. **Limitations of Prior Work**: Existing animal pose estimation methods are largely transferred from human pose estimation and perform poorly in complex agricultural scenes—cluttered backgrounds, frequent inter-animal occlusion, and entangled limbs and joints lead to identity confusion.
3. **Key Challenge**: High accuracy demands complex models, yet agricultural deployments require real-time inference and low-cost hardware. Furthermore, no publicly available mounting dataset exists.
4. **Goal**: Achieve lightweight and efficient mounting pose estimation in dense, cluttered environments.
5. **Key Insight**: Frequency-domain analysis to separate cattle from background, combined with multiscale self-calibration to correct structural misalignment under occlusion.
6. **Core Idea**: SFEBlock employs wavelet decomposition and Gaussian smoothing to separate foreground from background; RABlock aggregates multiscale context; SC2Head corrects occlusion-induced misalignment via spatial-channel self-calibration.

## Method

### Overall Architecture

A top-down paradigm: detect individual cattle → crop → backbone feature extraction → prediction head outputs keypoint heatmaps. The backbone is CattleMountNet and the prediction head is SC2Head.

### Key Designs

1. **SFEBlock (Spatial-Frequency Enhancement Block)**: Applies wavelet decomposition to separate high- and low-frequency components, and Gaussian smoothing to suppress background noise, enhancing the separability between cattle and cluttered backgrounds.
2. **RABlock (Receptive field Aggregation Block)**: Aggregates contextual information at different scales via multi-scale dilated convolutions, adapting to body parts of varying sizes.
3. **SC2Head (Spatial-Channel Self-Calibration Head)**: Attends to spatial and channel dependencies; a self-calibration branch corrects structural misalignment caused by occlusion.

### Loss & Training

Standard keypoint heatmap MSE loss.

## Key Experimental Results

### Main Results

| Method | AP | AP75 | AR | Params | GFLOPs |
|---|---|---|---|---|---|
| FSMC-Pose | **89.0%** | **92.5%** | **89.9%** | **2.698M** | 4.41 |
| RTMPose | 87.6% | 89.5% | 89.0% | 13.5M | - |

### Key Findings
- Parameter count reduced by 80% relative to RTMPose, with a 1.4% AP improvement.
- Frequency-spatial fusion yields the largest gain in cluttered-background scenarios (+2.8% AP).
- The MOUNT-Cattle dataset contains 1,176 mounting instances, filling a critical gap in the field.

### Ablation Study

| Configuration | AP | Params |
|---|---|---|
| Full FSMC-Pose | **89.0%** | **2.698M** |
| w/o SFEBlock | 86.2% | 2.1M |
| w/o RABlock | 87.1% | 2.3M |
| w/o SC2Head | 87.5% | 2.4M |
| Spatial only (w/o frequency) | 86.8% | 2.5M |

- Parameters reduced by 80% vs. RTMPose, with 1.4% AP gain.
- Frequency-spatial fusion produces the largest improvement in cluttered-background scenes.
- The constructed MOUNT-Cattle dataset, containing 1,176 mounting instances, fills an important gap.

## Highlights & Insights

- This is the first dataset and method dedicated to cattle mounting pose estimation, directly serving smart livestock farming.
- The strategy of using frequency-domain analysis to separate foreground from background is generalizable to other dense animal pose scenarios (e.g., poultry flocks, pig herds).

## Limitations & Future Work

- The dataset is relatively small (1,176 instances), posing a risk of overfitting.
- Validation is limited to mounting behavior; other behaviors (e.g., feeding, resting, rumination) are not addressed.
- Night/low-light conditions are insufficiently validated, despite significant illumination variation in outdoor farms.
- The computational overhead of wavelet decomposition may partially offset the lightweight advantage.
- The top-down paradigm depends on detector quality; pose estimation fails when detection fails.
- Video-level temporal modeling is not explored; the current approach operates on single frames only.
- Body shape variation across cattle breeds may affect generalization.
- No comparison is made against recent VLM-based animal pose estimation methods.

## Related Work & Insights

- **vs. DeepLabCut**: DeepLabCut performs poorly under occlusion; FSMC-Pose's self-calibration mechanism alleviates this issue.
- **vs. RTMPose**: RTMPose is general-purpose but parameter-heavy; FSMC-Pose is optimized for cattle scenarios and is substantially more lightweight.

### Supplementary Discussion
- The core innovation lies in transforming the problem from a single dimension to multiple dimensions, enabling a more comprehensive analytical perspective.
- The experimental design covers diverse scenarios and baseline comparisons, with statistically significant results.
- The modular design facilitates extension to related tasks and new datasets.
- Open-sourcing the code and data is of significant value for community reproduction and follow-up research.
- Compared to concurrent work, this paper demonstrates greater depth in problem formulation and comprehensiveness in experimental analysis.
- The paper's logical flow—from problem definition to method design to experimental validation—forms a complete and coherent narrative.
- The computational overhead is reasonable, rendering the method deployable in practical applications.
- Future work may consider fusion with additional modalities (e.g., audio, 3D point clouds).
- Validating scalability on larger data and models is an important subsequent direction.
- Combining the proposed method with reinforcement learning for end-to-end optimization is worth exploring.
- Cross-domain transfer is a direction worth investigating; the method's generalizability requires further validation.
- For edge computing and mobile deployment scenarios, a further-lightweighted variant of the method warrants investigation.

## Rating

- Novelty: ⭐⭐⭐ Component-level innovations are incremental, though the scenario and dataset are novel.
- Experimental Thoroughness: ⭐⭐⭐ Ablations are thorough, but scenarios are limited.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear and well-structured.
- Value: ⭐⭐⭐⭐ Demonstrates direct application value for smart livestock farming.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)
- [\[CVPR 2026\] Efficient Onboard Spacecraft Pose Estimation with Event Cameras and Neuromorphic Hardware](efficient_onboard_spacecraft_pose_estimation_with_event_cameras_and_neuromorphic_hardware.md)
- [\[CVPR 2026\] PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement](phase-net_physics-grounded_harmonic_attention_system_for_efficient_remote_photop.md)
- [\[CVPR 2026\] EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR](egoposeformer_v2_accurate_egocentric_human_motion_estimation_for_arvr.md)
- [\[CVPR 2026\] CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation](cigpose_causal_intervention_graph_neural_network_for_whole-body_pose_estimation.md)

</div>

<!-- RELATED:END -->
