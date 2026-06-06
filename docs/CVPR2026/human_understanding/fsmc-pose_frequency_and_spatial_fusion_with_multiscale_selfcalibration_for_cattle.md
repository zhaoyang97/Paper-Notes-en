---
title: >-
  [Paper Note] FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation
description: >-
  [CVPR 2026][Human Understanding][cattle pose estimation] FSMC-Pose proposes a lightweight top-down framework for cattle mounting pose estimation…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "cattle pose estimation"
  - "estrus detection"
  - "frequency-spatial fusion"
  - "multiscale self-calibration"
  - "lightweight backbone"
date: 2026-05-08
content_hash: 6df8d5bbdc649a70
---

# FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation

**Conference**: CVPR 2026
**arXiv**: [2603.16596](https://arxiv.org/abs/2603.16596)  
**Code**: [https://github.com/](https://github.com/)  
**Area**: Human Understanding
**Keywords**: cattle pose estimation, estrus detection, frequency-spatial fusion, multiscale self-calibration, lightweight backbone

## TL;DR

FSMC-Pose proposes a lightweight top-down framework for cattle mounting pose estimation, comprising the frequency-spatial fusion backbone CattleMountNet (which employs wavelet transform and Gaussian filtering in the SFEBlock for foreground-background separation, and multi-scale dilated convolutions in the RABlock for context aggregation) and the multiscale self-calibration head SC2Head (spatial-channel co-calibration with a self-calibration branch to correct structural displacement). The paper also introduces MOUNT-Cattle, the first dataset for cattle mounting behavior, achieving 89% AP in complex group-housing environments at extremely low computational cost (4.41 GFLOPs, 2.698M parameters).

## Background & Motivation

1. **Background**: Estrus detection in cattle is critical to the economic performance of the livestock industry. Mounting behavior is the most intuitive visual indicator of estrus. Existing animal pose estimation methods predominantly adapt human pose estimation approaches (e.g., DeepLabCut, HRNet) under two paradigms: bottom-up and top-down.
2. **Limitations of Prior Work**: (1) No publicly available dataset for cattle mounting behavior exists; (2) estrous cattle tend to cluster, making mounting scenes denser than typical farm environments; (3) cluttered backgrounds, severe inter-animal occlusion, and similar coat patterns cause keypoint confusion and identity ambiguity; (4) existing methods are computationally expensive and unsuitable for real-time production monitoring.
3. **Key Challenge**: Mounting pose estimation in dense group-housing environments requires simultaneous handling of background interference, occlusion, and multi-scale keypoints, yet no existing lightweight method can address all these challenges jointly.
4. **Goal**: Construct a dedicated dataset and design a lightweight, high-accuracy method for mounting pose estimation.
5. **Key Insight**: Enhance features from two complementary perspectives: the frequency domain (wavelet decomposition) and the spatial domain (multi-scale context).
6. **Core Idea**: Frequency-spatial fusion for foreground separation + multi-scale receptive fields for scale variation + self-calibration to correct occlusion-induced displacement.

## Method

### Overall Architecture

FSMC-Pose adopts a top-down design following the RTMPose framework, using MobileNet as the base. The CattleMountNet backbone extracts multi-level features, and the SC2Head prediction head performs keypoint regression. The input is a cropped cattle image, and the output consists of 16 keypoint coordinates.

### Key Designs

1. **Spatial-Frequency Enhancement Block (SFEBlock)**:

    - **Function**: Enhances foreground-background separation in cluttered farm environments.
    - **Mechanism**: Combines wavelet transform convolution (WTConv) and Gaussian filtering. WTConv decomposes the input into low- and high-frequency sub-bands, applies convolutions within each sub-band to capture multi-scale frequency features, and reconstructs the signal via inverse wavelet transform. A fixed 5×5 Gaussian kernel smooths noise. The two feature streams are summed and compressed via 1×1 convolution; element-wise multiplication refines the spatial response, and a residual connection preserves input information: $F_{\text{out}} = \text{Conv}^{3\times3}(F_{\text{WTconv}} \otimes F_{\text{temp}}) + F_{\text{in}}$
    - **Design Motivation**: Mud, shadows, and illumination variations in farm environments cause cattle texture to resemble the background, blurring keypoints under low contrast. Frequency-domain modeling enlarges the receptive field while preserving local structure.

2. **Receptive Field Aggregation Block (RABlock)**:

    - **Function**: Handles the large scale variation of cattle keypoints, from small hooves to large torso regions.
    - **Mechanism**: Augments an inverted residual unit with three parallel 3×3 depth-wise separable convolutions with dilation rates of 1, 3, and 5 to capture local, mid-range, and long-range context, respectively. The three outputs are summed and normalized with LayerNorm: $\mathbf{H}_{l-1} = \text{LN}(\mathbf{H}^1 + \mathbf{H}^2 + \mathbf{H}^3)$, combined with HardSwish activation and a residual connection.
    - **Design Motivation**: Single-scale features cannot simultaneously capture small joints and large body regions.

3. **Spatial-Channel Self-Calibration Head (SC2Head)**:

    - **Function**: Corrects structural displacement and keypoint misassociation caused by inter-animal occlusion.
    - **Mechanism**: Three-branch design — the Spatial Attention Branch (SAB) generates spatial weights via average and max pooling; the Channel Attention Branch (CAB) generates channel weights via channel-level pooling; the Self-Calibration Branch (SCB) provides structural correction. The three branches are fused via $C_o = f_{1\times1}([\text{SA}, \text{CA}]) \odot \text{SC} + X$.
    - **Design Motivation**: The SFEBlock and RABlock in the backbone primarily operate during early feature extraction; the prediction head must still resolve structural confusion.

### Loss & Training

Follows RTMPose's SimCC coordinate regression strategy with KL divergence loss supervision.

## Key Experimental Results

### Main Results

| Method | AP↑ | AP75↑ | AR↑ | GFLOPs | Params |
|--------|-----|-------|-----|--------|--------|
| RTMPose-s | 87.6 | 89.5 | 89.0 | 5.47 | 13.49M |
| HRNet-w32 | 86.8 | 88.1 | 88.3 | 9.83 | 28.54M |
| SimpleBaseline | 85.4 | 87.2 | 87.5 | 8.90 | 34.00M |
| **FSMC-Pose** | **89.0** | **92.5** | **89.9** | **4.41** | **2.698M** |

FSMC-Pose achieves the highest accuracy with the lowest computational cost and parameter count.

### Ablation Study

| Configuration | AP | AP75 | Note |
|---------------|----|------|------|
| MobileNet baseline | 86.2 | 87.8 | w/o SFE/RA |
| +SFEBlock | 87.5 | 89.2 | contribution of frequency enhancement |
| +RABlock | 88.1 | 90.8 | contribution of multi-scale aggregation |
| +SC2Head (full) | 89.0 | 92.5 | contribution of self-calibration |

### Key Findings

- SFEBlock yields the largest gain in high-occlusion scenarios, confirming the effectiveness of frequency-domain foreground-background separation.
- The improvement in AP75 (strict threshold) exceeds that in AP (+3.0% vs. +1.4%), indicating enhanced precise localization capability.
- With only 2.698M parameters (80% reduction relative to RTMPose) and 4.41 GFLOPs, the model supports real-time inference on commodity GPUs.
- The MOUNT-Cattle dataset contains 1,176 mounting instances and is the first dataset dedicated to mounting behavior.

## Highlights & Insights

- **First mounting pose dataset**: Fills the data gap in visual estrus detection for cattle; adopts COCO format for plug-and-play training.
- **Dual frequency-spatial modeling**: The application of wavelet transform to animal pose estimation is novel.
- **Extreme lightweight design**: 2.698M parameters and 4.41 GFLOPs achieving 89% AP demonstrate strong practical deployment value.

## Limitations & Future Work

- The dataset scale is limited (1,176 instances); generalization to different farms and breeds requires more data.
- Only 16 keypoints are considered; finer-grained behavioral analysis may require a denser keypoint definition.
- End-to-end estrus determination integrating behavior recognition is not addressed.
- Future work may extend to video-level temporal behavior recognition.

## Related Work & Insights

- **vs. DeepLabCut**: DeepLabCut suffers from severe identity confusion in crowded scenes; FSMC-Pose addresses this via self-calibration.
- **vs. RTMPose**: RTMPose offers strong generality but has a large parameter count; FSMC-Pose is customized for cattle scenes and is more efficient.
- **vs. CMBN**: CMBN compresses HRNet but remains bottom-up, leading to keypoint misassociation in dense scenes.

## Rating

- Novelty: ⭐⭐⭐ The method combines existing modules, though the application scenario is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dataset construction is solid; comparisons are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear.
- Value: ⭐⭐⭐⭐ Practical value for smart livestock farming.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)
- [\[CVPR 2026\] Efficient Onboard Spacecraft Pose Estimation with Event Cameras and Neuromorphic Hardware](efficient_onboard_spacecraft_pose_estimation_with_event_cameras_and_neuromorphic_hardware.md)
- [\[CVPR 2026\] CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation](cigpose_causal_intervention_graph_neural_network_for_whole-body_pose_estimation.md)
- [\[CVPR 2026\] GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion](gazeonce360_fisheye-based_360_multi-person_gaze_estimation_with_global-local_fea.md)
- [\[NeurIPS 2025\] RAPTR: Radar-Based 3D Pose Estimation Using Transformer](../../NeurIPS2025/human_understanding/raptr_radar-based_3d_pose_estimation_using_transformer.md)

</div>

<!-- RELATED:END -->
