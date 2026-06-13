---
title: >-
  [Paper Note] Multi-view Gaze Target Estimation
description: >-
  [ICCV 2025][Human Understanding][Gaze Target Estimation] This paper is the first to extend Gaze Target Estimation (GTE) from single-view to multi-view settings. By integrating three modules — Head Information Aggregation…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "Gaze Target Estimation"
  - "Multi-view"
  - "Cross-view"
  - "Epipolar Attention"
  - "Uncertainty"
date: 2026-05-08
content_hash: 3dd601c3330e594c
---

# Multi-view Gaze Target Estimation

**Conference**: ICCV 2025
**arXiv**: [2508.05857](https://arxiv.org/abs/2508.05857)  
**Code**: [https://www3.cs.stonybrook.edu/~cvl/multiview_gte.html](https://www3.cs.stonybrook.edu/~cvl/multiview_gte.html)  
**Area**: Human Understanding
**Keywords**: Gaze Target Estimation, Multi-view, Cross-view, Epipolar Attention, Uncertainty

## TL;DR
This paper is the first to extend Gaze Target Estimation (GTE) from single-view to multi-view settings. By integrating three modules — Head Information Aggregation (HIA), Uncertainty-based Gaze Selection (UGS), and Epipolar-based Scene Attention (ESA) — the method fuses information across multiple cameras. It significantly outperforms single-view state-of-the-art methods on the newly introduced MVGT dataset and enables cross-view estimation that single-view methods cannot handle.

## Background & Motivation

**Background**: Gaze Target Estimation (GTE) aims to determine where a person is looking using ordinary scene cameras. Deep learning-based approaches have advanced substantially in recent years, leveraging depth maps, 3D field-of-view (FoV) heatmaps, and Transformer architectures.

**Limitations of Prior Work**: All existing methods are confined to a single viewpoint. When the subject's face is occluded, multiple candidate targets exist within the field of view, or the gaze target lies outside the frame, single-view methods fail entirely.

**Key Challenge**: A single camera has a limited field of view and cannot obtain appearance information from occluded heads to accurately estimate gaze direction, nor can it predict out-of-frame gaze targets.

**Goal**: (a) How to leverage clearer facial observations from multiple camera viewpoints to improve gaze direction estimation accuracy; (b) how to propagate scene context information across different viewpoints; (c) how to handle cross-view scenarios where the subject and the gaze target appear in different cameras.

**Key Insight**: Multi-camera systems are widely deployed in environments such as supermarkets and classrooms, naturally providing broader coverage and multi-angle observations. Rather than performing full 3D reconstruction, the authors exploit calibrated multi-camera geometry to efficiently fuse multi-view information at the feature level.

**Core Idea**: Multi-view GTE is achieved by aggregating facial information via cross-view head attention, selecting reliable gaze directions through uncertainty estimation, and sharing scene context with epipolar-constrained attention.

## Method

### Overall Architecture
The inputs consist of a pair of calibrated camera images $\mathbf{I}_1, \mathbf{I}_2$, the corresponding head bounding boxes, and camera intrinsic/extrinsic parameters. The model first aggregates head features from both views via the HIA module, then the UGS module selects the more reliable gaze vector to generate an FoV heatmap, and finally a multi-view scene encoder incorporating the ESA module fuses cross-view scene features. A gaze decoder outputs the target heatmap and in/out probability.

### Key Designs

1. **Head Information Aggregation (HIA) Module**:

    - **Function**: Aggregates facial appearance information from the head image of another viewpoint to enhance the head embedding of the current view.
    - **Mechanism**: ResNet-18 extracts head features $\mathbf{F}^h$ from both views; these are flattened into tokens and aggregated via cross-attention. The key innovation lies in concatenating the relative camera rotation matrix $\mathbf{R}_{21} = \mathbf{R}_1 \mathbf{R}_2^{-1}$ into the Key and Value to inform the model of the geometric relationship between views: $\tilde{\mathbf{F}}^h_1 = \mathbf{F}^h_1 + \text{CrossAtt}(\mathbf{Q}^h_1, \mathbf{K}^h_1, \mathbf{V}^h_1)$
    - **Design Motivation**: When the subject faces away from one camera, the other may capture the frontal view; encoding geometric relationships enables the model to understand the correspondence between head poses across views.

2. **Uncertainty-based Gaze Selection (UGS) Module**:

    - **Function**: Selects the more reliable gaze vector from the two predicted by each viewpoint.
    - **Mechanism**: The gaze estimator is extended to jointly predict an uncertainty score $\sigma$, trained with an uncertainty-aware loss: $\mathcal{L}_{gaze} = \frac{1}{2\sigma^2}(1 - \cos(\mathbf{g}, \hat{\mathbf{g}})) + \frac{1}{2}\log(\sigma^2)$. The gaze vector from the view with smaller $\sigma$ is selected and transformed to the other view via $\mathbf{g}'_j = \mathbf{R}_j \mathbf{R}_i^{-1} \mathbf{g}_i$.
    - **Design Motivation**: Even with HIA-enhanced information propagation, prediction quality may still differ across views (e.g., occlusion in one view). Aleatoric uncertainty naturally reflects the difficulty of gaze estimation from a given input image.

3. **Epipolar-based Scene Attention (ESA) Module**:

    - **Function**: Propagates cross-view scene context information within the multi-view scene encoder.
    - **Mechanism**: For each token in the scene feature map, the corresponding epipolar line in the other view is computed via the fundamental matrix $\mathbb{F}$, and 48 feature vectors are uniformly sampled along this line for cross-attention. This is more efficient than dense cross-attention and exploits epipolar geometric constraints.
    - **Design Motivation**: Epipolar constraints ensure attention focuses only on features near corresponding 3D locations, which is particularly beneficial for handling occlusions — information about an occluded region can be retrieved from the complementary view.

### Loss & Training
The total loss is $\mathcal{L} = \alpha \mathcal{L}_{hm} + \beta \mathcal{L}_{io} + \lambda \mathcal{L}_{gaze}$, where $\mathcal{L}_{hm}$ is the heatmap MSE loss, $\mathcal{L}_{io}$ is the in/out binary classification BCE loss, and $\mathcal{L}_{gaze}$ is the uncertainty-aware gaze loss. The model is first pretrained as a single-view version on GazeFollow, then fine-tuned as the full multi-view model on MVGT.

Additionally, the paper proposes an extension for cross-view GTE: when the subject appears only in the reference view and the target only in the primary view, Dust3R is used to pre-reconstruct the scene and obtain absolute depth. The gaze vector from the reference view is then transformed into the primary view to generate the FoV heatmap, with an added feature transform module and a learnable "outside embedding."

## Key Experimental Results

### Main Results
Leave-one-scene-out evaluation on the MVGT dataset (results when the head in the reference view and the target are both visible):

| Method | Dist. ↓ | AP ↑ | Notes |
|--------|---------|------|-------|
| Chong [CVPR'20] | 0.159 | 0.855 | RGB input only |
| Miao [CVPR'22] | 0.141 | 0.886 | Uses depth |
| Tafasca [ECCV'24] | 0.149 | 0.893 | FoV heatmap |
| Ours-Single | 0.151 | 0.877 | Single-view baseline |
| **Ours** | **0.129** | **0.909** | Full multi-view model |

When the head in the reference view is visible, the proposed method reduces Dist. by 14.6% relative to the single-view baseline; error reductions are 23.7% and 23.2% when the frontal and side face are visible, respectively.

### Ablation Study

| Configuration | Dist. ↓ (Head+Target visible) | AP ↑ |
|--------------|-------------------------------|------|
| Single-view baseline | 0.151 | 0.877 |
| + σ uncertainty | 0.145 | 0.874 |
| + HIA | 0.135 | 0.896 |
| + HIA + UGS | 0.130 | 0.902 |
| + HIA + UGS + ESA (full) | 0.129 | 0.909 |

### Key Findings
- **HIA contributes most**: Adding HIA reduces Dist. from 0.145 to 0.135 (−6.9%), demonstrating that fusing head appearance from another viewpoint is the most effective improvement.
- **UGS is less effective when the head is invisible in the reference view**: Without a visible head, UGS cannot exploit its selection advantage.
- **ESA is more effective when the target is visible**: This confirms that ESA's role is to supplement scene information about the target region from the complementary view.
- **Cross-view estimation**: The proposed method achieves Dist.=0.188 and AP=0.820 on the cross-view task, substantially outperforming the adapted Recasens baseline (Dist.=0.271, AP=0.542).

## Highlights & Insights
- **Efficient cross-view attention via epipolar constraints**: Rather than applying dense full-image attention, features are sampled along epipolar lines, leveraging multi-view geometric priors while substantially reducing computation. This design is transferable to other multi-view understanding tasks.
- **Uncertainty-driven multi-view fusion**: Aleatoric uncertainty is used to automatically assess prediction quality from each view, elegantly addressing the question of which view is more reliable — a principle broadly applicable to multi-sensor fusion.
- **Non-intrusive data collection protocol**: Gaze targets are annotated using a laser pointer, which is then turned off before capturing the image, avoiding annotation artifacts in the scene. This yields more precise ground truth than conventional subjective annotation.

## Limitations & Future Work
- **Dependence on calibrated camera parameters**: In practice, camera calibration may be inaccurate or dynamically changing; geometry-aware feature learning to reduce reliance on precise calibration warrants future investigation.
- **Cross-view task requires pre-reconstruction**: The use of Dust3R for scene pre-reconstruction to obtain absolute depth increases deployment overhead.
- **Limited dataset scale**: Only 4 scenes and 28 subjects are included, leaving generalizability to be verified.
- **Restricted to pairwise views**: While more than two views can be handled by aggregating multiple pairwise results, an end-to-end fusion scheme for more than two views is absent.

## Related Work & Insights
- **vs. Tafasca [ECCV'24]**: Tafasca uses FoV heatmaps as gaze priors; the proposed method builds on this by applying UGS to select the more reliable gaze vector, producing higher-quality FoV heatmaps.
- **vs. Multi-view 3D pose estimation**: Multi-view methods for pose estimation typically require substantial viewpoint overlap, whereas GTE scenes may have minimal overlap between views.
- **vs. DVGaze [ICCV'23]**: DVGaze explores dual-view gaze direction estimation but requires visible faces and rectified images, making it inapplicable to GTE directly.

## Rating
- Novelty: ⭐⭐⭐⭐ First exploration of multi-view GTE; the three modules are well-motivated, though individually not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablations and convincing cross-view experiments, though the dataset is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is clear, method description is fluent, and figures are well-crafted.
- Value: ⭐⭐⭐⭐ Opens a new direction for multi-view GTE; dataset and code are publicly available.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion](../../CVPR2026/human_understanding/gazeonce360_fisheye-based_360_multi-person_gaze_estimation_with_global-local_fea.md)
- [\[NeurIPS 2025\] OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild](../../NeurIPS2025/human_understanding/omnigaze_reward-inspired_generalizable_gaze_estimation_in_the_wild.md)
- [\[NeurIPS 2025\] A Generalized Label Shift Perspective for Cross-Domain Gaze Estimation](../../NeurIPS2025/human_understanding/a_generalized_label_shift_perspective_for_crossdomain_gaze_e.md)
- [\[ICCV 2025\] GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation](genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)
- [\[ICCV 2025\] MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation](mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)

</div>

<!-- RELATED:END -->
