---
title: >-
  [Paper Note] Can Generative Video Models Help Pose Estimation?
description: >-
  [CVPR 2025][Image Generation][Pose Estimation] InterPose is proposed, which leverages pre-trained video generative models to "hallucinate" intermediate frames between two images with little or no overlap. Combined with a self-consistency score to select the best video, it consistently improves pose estimation accuracy across four datasets on top of DUSt3R.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Pose Estimation"
  - "Video Generative Models"
  - "Frame Interpolation"
  - "Self-Consistency Score"
  - "Low-Overlap Images"
date: 2026-05-08
content_hash: f53f78ca384d0130
---

# Can Generative Video Models Help Pose Estimation?

**Conference**: CVPR 2025  
**arXiv**: [2412.16155](https://arxiv.org/abs/2412.16155)  
**Code**: [https://Inter-Pose.github.io](https://Inter-Pose.github.io) (Project Page)  
**Area**: Image Generation  
**Keywords**: Pose Estimation, Video Generative Models, Frame Interpolation, Self-Consistency Score, Low-Overlap Images

## TL;DR
InterPose is proposed, which leverages pre-trained video generative models to "hallucinate" intermediate frames between two images with little or no overlap. Combined with a self-consistency score to select the best video, it consistently improves pose estimation accuracy across four datasets on top of DUSt3R.

## Background & Motivation

**Background**: Pairwise camera pose estimation is a fundamental task in 3D vision. Traditional methods rely on feature matching and correspondence computation, which require sufficient overlap between images. Deep learning methods like DUSt3R are trained on large-scale 3D data and generalize well, but still struggle under large viewpoint changes with almost no overlap.

**Limitations of Prior Work**: When the viewpoint difference between two images is extreme (e.g., two opposite walls of a classroom), reliable feature correspondences cannot be established, causing both traditional and learning-based methods to fail. The root cause is the lack of "bridging" information from intermediate viewpoints. Training stronger 3D models requires massive 3D annotated data, which is far less diverse and scalable than video data.

**Key Challenge**: Pose estimation requires geometrically consistent visual correspondences, which are lacking in low-overlap scenarios; video generative models possess rich scene priors but do not guarantee geometric consistency, often producing videos with artifacts and inconsistent geometry.

**Goal**: How to leverage the scene priors of video generative models to improve pose estimation for low-overlap image pairs.

**Key Insight**: Humans can infer spatial relationships from two almost non-overlapping images of a classroom because they possess prior knowledge of typical classroom layouts. Similarly, video models trained on massive video datasets learn powerful prior knowledge of scene motion. Utilizing video models to interpolate frames between two images creates a dense visual transition, transforming the low-overlap problem into a multi-frame dense-overlap problem.

**Core Idea**: Use a video frame interpolation model to hallucinate transition frames between two images, thereby "filling" the visual gap. A self-consistency score is then employed to select the most geometrically consistent video from multiple candidates, assisting DUSt3R in improving pose estimation.

## Method

### Overall Architecture
Given an image pair $(I_A, I_B)$, GPT-4o is used to generate two descriptive captions. Combined with forward and reversed sequences ($A \to B$ and $B \to A$), this generates $n=4$ interpolated videos. For each video, $m=11$ subsets of frames (each containing $k=5$ frames, including the original two frames) are randomly sampled. DUSt3R is applied to estimate the poses for each subset, and a self-consistency score is computed to select the best video. Its medoid pose is output as the final prediction.

### Key Designs

1. **Video Interpolation as a Scene Prior**

    - Function: Create dense visual transitions between low/non-overlapping image pairs, transforming a difficult wide-baseline problem into a simpler narrow-baseline multi-frame problem.
    - Mechanism: Utilize off-the-shelf video interpolation models $f_{vid}(I_A, I_B, p) = [I_1, ..., I_N]$, where $I_1=I_A, I_N=I_B$. Three models are validated: DynamiCrafter (open-source), Runway Gen-3 (commercial), and Luma Dream Machine (commercial). The generated intermediate frames, along with the original image pair, are fed into the multi-frame extension of DUSt3R for pose estimation. The video models do not require any modifications or fine-tuning.
    - Design Motivation: Video models are trained on video data that is orders of magnitude larger than 3D datasets, learning stronger priors about scene layouts and motion. Even if the generated frames are imperfect, they provide more geometric clues to DUSt3R than having only two frames.

2. **Self-Consistency Score**

    - Function: Select the geometrically most consistent video from multiple generated candidates.
    - Mechanism: For each video, $m$ frame subsets are randomly sampled, and the pose estimation $\hat{T}^{(i)}$ is computed for each subset. The medoid distance is used to measure consistency: $D_{med} = \min_i \frac{1}{m-1}\sum_{j \neq i} dist(\hat{T}^{(i)}, \hat{T}^{(j)})$. A low medoid distance implies different frame subsets yield nearly identical pose predictions, indicating high geometric consistency in the video. To prevent degenerate cases (e.g., low-quality videos consistently giving wrong predictions), an anchoring term is added: $D_{total} = D_{med} + dist(\hat{T}_{med}, f_{pose}(\{I_A, I_B\}))$. The video with the lowest $D_{total}$ is selected, and its medoid pose is output.
    - Design Motivation: Video models often generate artifacts (sudden appearance/disappearance of objects, morphing, geometric inconsistency). Simply averaging the predictions of all videos actually degrades performance (in experiments, averaging Dream Machine results degraded the MRE on Cambridge from 13.28° to 21.85°). The medoid score distinguishes good and bad videos without requiring ground-truth labels.

3. **Symmetric Generation Strategy**

    - Function: Mitigate the inherent motion bias of video generative models.
    - Mechanism: Generate both forward ($A \to B$) and reversed ($B \to A$) videos for each image pair simultaneously. Diversity is further enhanced using two different captions, resulting in a total of 4 videos per pair.
    - Design Motivation: Video models typically bias toward generating left-to-right camera motion. Reversing the input sequence compensates for this bias.

### Loss & Training
InterPose is a completely training-free, inference-time method—it does not modify any models and merely combines the video generative model and DUSt3R as black boxes. Pose distance is measured as the sum of rotation geodesic distance and translation angular error.

## Key Experimental Results

### Main Results

| Dataset | Method | MRE↓ | MTE↓ | AUC30↑ |
|--------|------|------|------|--------|
| Cambridge (Outdoor) | DUSt3R only | 13.28° | — | 77.23 |
| Cambridge | **InterPose+Runway Medoid** | **10.78°** | — | **80.59** |
| ScanNet (Indoor) | DUSt3R only | 21.31° | 24.72° | 60.34 |
| ScanNet | **InterPose+DreamMachine Medoid** | **17.65°** | **15.88°** | **63.06** |
| DL3DV-10K | DUSt3R only | — | 13.08° | 66.99 |
| DL3DV-10K | **InterPose+DreamMachine** | — | **8.72°** | **69.44** |
| NAVI (Object) | DUSt3R only | 8.65° | 7.88° | — |
| NAVI | **InterPose+DreamMachine** | **7.85°** | **6.51°** | — |

### Ablation Study

| Configuration | Cambridge MRE↓ | Description |
|------|---------------|------|
| DUSt3R only | 13.28° | Baseline |
| + DreamMachine Avg | 21.85° | Simple averaging degrades performance |
| + DreamMachine Medoid | **11.96°** | Self-consistency score restores improvement |
| Oracle (Select best video) | 3.65° | Upper bound, indicating massive potential for better video selection |

### Key Findings
- Self-consistency scoring is crucial: simply averaging predictions across all videos degrades performance for some models (Cambridge MRE worsened from 13.28° to 21.85°), whereas medoid selection yields consistent improvements.
- The Oracle upper bound (selecting the video closest to ground-truth pose) shows that MRE can be as low as 3.65°, indicating that video models possess correct geometric priors and the bottleneck lies in video selection.
- Runway and DreamMachine generally outperform the open-source DynamiCrafter, with commercial models showing more stable quality.
- Reversing the input sequence helps mitigate the rightward motion bias.
- The method is consistently effective across four different types of scenes (outdoor, indoor, center-focused, and object-centric).

## Highlights & Insights
- The cross-modal concept of **using video models as scene priors** is highly inspiring: video data is far more abundant than 3D data, and this approach opens up a new direction for "bootstrapping 3D understanding using generative models."
- The **self-consistency score** evaluates video quality without any labels, based on the simple principle that "a good reconstruction should be robust to input perturbations."
- **Completely plug-and-play**: It does not modify any models and merely combines them, making the method extremely simple yet consistently effective.

## Limitations & Future Work
- Video generative models are **expensive and slow**—the total cost for commercial models was around $5,500, limiting the scale of the experiments.
- Video models do not guarantee multi-view consistency; sometimes all generated videos are of low quality.
- Sensitive to prompts, camera intrinsics, and aspect ratios, requiring meticulous design.
- The gap between the Oracle and actual performance is large (3.65° vs 10.78°), making better video selection/scoring an important future direction.
- Currently, only DUSt3R has been tested as the pose estimator; combinations with traditional methods like COLMAP remain unexplored.

## Related Work & Insights
- **vs DUSt3R**: DUSt3R's performance degrades in low-overlap scenarios. InterPose improves on it without modifications by providing intermediate frames to expand the effective overlap. The two are complementary.
- **vs Flow Matching/Keypoint Methods**: SIFT+NN and LoFTR largely fail in low-overlap scenarios (MRE > 30°/64°) because they require correspondence points to function.
- **vs Probabilistic Pose Methods**: Diffusion-based pose models can handle ambiguity but do not leverage external scene priors. InterPose provides more information via generative priors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to leverage video generative models to improve pose estimation. The direction is novel and highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on four datasets, three video models, with complete ablation and oracle analyses, though the sample size is limited due to budget constraints.
- Writing Quality: ⭐⭐⭐⭐⭐ Intuitive description of motivation (the classroom example), excellent figures and tables, and the method is simple yet easy to understand.
- Value: ⭐⭐⭐⭐ Opens up a new paradigm of using video models for 3D understanding, though its practicality is currently constrained by video generation costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer](animer_animal_pose_and_shape_estimation_using_family_aware_transformer.md)
- [\[CVPR 2025\] PhysicsGen: Can Generative Models Learn from Images to Predict Complex Physical Relations?](physicsgen_can_generative_models_learn_from_images_to_predict_complex_physical_r.md)
- [\[CVPR 2025\] Goku: Flow Based Video Generative Foundation Models](goku_flow_based_video_generative_foundation_models.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)
- [\[CVPR 2025\] VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary](vlog_video-language_models_by_generative_retrieval_of_narration_vocabulary.md)

</div>

<!-- RELATED:END -->
