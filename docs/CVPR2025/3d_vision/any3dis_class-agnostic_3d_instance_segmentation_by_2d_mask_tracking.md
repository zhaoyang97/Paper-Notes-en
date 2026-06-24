---
title: >-
  [Paper Note] Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking
description: >-
  [CVPR 2025][3D Vision][3D Instance Segmentation] Any3DIS is proposed, which replaces traditional unsupervised merging strategies with 3D-aware 2D mask tracking (utilizing SAM-2 to track the 2D segmentations of each superpoint across multiple frames) and optimizes 3D proposals using dynamic programming. It achieves state-of-the-art (SOTA) results in class-agnostic, open-vocabulary, and open-ended 3D instance segmentation tasks on ScanNet200 and ScanNet++.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Instance Segmentation"
  - "2D Mask Tracking"
  - "SAM-2"
  - "Class-Agnostic"
  - "Dynamic Programming Optimization"
date: 2026-05-08
content_hash: ebf6c68c1c117e46
---

# Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking

**Conference**: CVPR 2025  
**arXiv**: [2411.16183](https://arxiv.org/abs/2411.16183)  
**Code**: [https://any3dis.github.io/](https://any3dis.github.io/)  
**Area**: 3D Computer Vision / 3D Instance Segmentation  
**Keywords**: 3D Instance Segmentation, 2D Mask Tracking, SAM-2, Class-Agnostic, Dynamic Programming Optimization

## TL;DR
Any3DIS is proposed, which replaces traditional unsupervised merging strategies with 3D-aware 2D mask tracking (utilizing SAM-2 to track the 2D segmentations of each superpoint across multiple frames) and optimizes 3D proposals using dynamic programming. It achieves state-of-the-art (SOTA) results in class-agnostic, open-vocabulary, and open-ended 3D instance segmentation tasks on ScanNet200 and ScanNet++.

## Background & Motivation
3D instance segmentation is a core task in computer vision with extensive applications in autonomous navigation, augmented reality, and scene understanding. Existing class-agnostic 3D instance segmentation methods (such as Open3DIS) adopt a "segment-then-associate" strategy: they generate dense 2D masks in each frame using models like SAM, and then lift them to 3D space for association based on heuristic rules. This leads to two core limitations of prior work: (1) over-segmentation, which generates a large number of redundant 3D proposals; (2) association inconsistency, where 2D segmentation results from different viewpoints lack coherence, resulting in poor association quality. The root cause is that the heuristic association process cannot exploit the temporal consistency of objects in video sequences. The core idea of this paper is "tracking instead of association," which leverages the video tracking capability of SAM-2 to directly obtain temporally consistent 2D mask trajectories across frames, where each trajectory naturally corresponds to a 3D object.

## Method
Any3DIS simplifies 3D instance segmentation into two main parts: first obtaining the cross-frame mask trajectory of each object using 2D video segmentation and tracking, and then lifting these trajectories into 3D proposals and refining them through optimization.

### Overall Architecture
The inputs are a 3D point cloud and the corresponding RGB-D frame sequence. First, superpoint partition and farthest point sampling (FPS) are applied to the point cloud to select initial superpoints. For each superpoint, the "pivot view" with the highest visibility across all frames is identified. In this pivot view, SAM-2 is used to segment the target object, which is then tracked forward and backward to get a complete 2D mask trajectory. This trajectory is then lifted to a 3D superpoint set to form a candidate proposal. Finally, the superpoint selection is optimized using a dynamic programming algorithm to output refined 3D masks. Uncovered superpoints are iteratively processed until all objects are segmented.

### Key Designs
1. **3D-Aware 2D Mask Tracking**:
    - **Function**: Generates a consistent 2D mask trajectory across all frames for each sampled superpoint.
    - **Mechanism**: For each superpoint $\mathbf{S}_l$, its visible points in each frame are calculated and weighted by the visibility of neighboring superpoints to construct a histogram $\psi_t^l = |\rho_t^l| \cdot s_t^l$. The frame corresponding to the maximum value of the histogram is selected as the pivot view. In the pivot view, 3 projected points are sampled using FPS to serve as point prompts for SAM-2, and the resulting 2D segmentation is used as a mask prompt for bidirectional (forward and backward) tracking.
    - **Design Motivation**: Traditional methods starting from the first frame tend to miss objects that appear in later frames, and the initial mask may not be the optimal visible area. Weighing by neighboring superpoints ensures that the selected view is not only optimal for the currently observed superpoint but also provides a good view of the overall object. For cases where objects disappear and reappear, superpoint projection points are utilized to provide query points again.

2. **3D Proposal Refinement Optimization (3D Mask Optimization)**:
    - **Function**: Selects the optimal subset from the candidate superpoint set to form the final 3D proposal.
    - **Mechanism**: A binary optimization problem is defined to maximize the number of projected points of selected superpoints inside the 2D masks across all views (positive term) and minimize those outside the masks (negative term). Since brute-force enumeration is NP-hard, dynamic programming is adopted: traversing all views sequentially, each step chooses the option with a higher objective value between "keep the current scheme" or "include all newly visible superpoints in this view."
    - **Design Motivation**: Directly selecting all superpoints with an IoU exceeding a threshold neglects multi-view consistency—a superpoint might have high overlap in one view but very low overlap in others. Although the DP algorithm is not globally optimal, it is highly efficient and its practical performance surpasses all previous methods.

3. **Iterative 3D Object Sampling**:
    - **Function**: Ensures all objects in the scene are covered.
    - **Mechanism**: After each round of processing, assigned superpoints are marked. The process of FPS sampling $\rightarrow$ tracking $\rightarrow$ optimization is repeated for the unassigned superpoints until no free superpoints remain.
    - **Design Motivation**: Single-round sampling might miss objects that are occluded by larger objects or are relatively small.

### Loss & Training
The method is training-free and uses SAM-2 (ViT-L) for zero-shot inference. Key hyperparameters: 2D-3D occlusion depth threshold of 0.1, 3D lifting IoU threshold $\tau=0.5$, and RGB-D frame sampling interval of 10 frames.

## Key Experimental Results

### Main Results

| Dataset | Task | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|------|----------|------|
| ScanNet++ | Class-Agnostic | AP | 22.2 | 20.7 (Open3DIS) | +1.5 |
| ScanNet200 | Class-Agnostic (2D only) | AP | 32.5 | 31.5 (Open3DIS) | +1.0 |
| ScanNet200 | Class-Agnostic (3D+2D) | AP | 42.5 | 41.5 (Open3DIS) | +1.0 |
| ScanNet200 | Open-Vocab (3D+2D) | AP | 25.8 | 23.7 (Open3DIS) | +2.1 |
| ScanNet++ | Open-Ended | AP | 20.1 | 18.4 (Point-Wise) | +1.7 |
| ScanNet200 | Open-Ended | AP | 19.1 | 16.0 (Point-Wise) | +3.1 |

### Ablation Study

| Configuration | AP | AP50 | AP25 | Description |
|------|-----|------|------|------|
| Baseline (w/o 3D-AMT, w/o MaskOpt) | 17.0 | 30.1 | 42.6 | First frame pivot + all superpoints |
| +3D-AMT | 17.9 | 31.2 | 45.1 | 3D-aware pivot selection |
| +MaskOpt | 20.3 | 33.7 | 44.5 | DP-optimized superpoint selection |
| +3D-AMT +MaskOpt (Full) | 22.2 | 35.8 | 47.0 | Complementary to each other |

### Key Findings
- MaskOpt makes the most significant contribution (+3.3 AP), indicating that refining superpoint selection is more critical than improving tracking starting points.
- In the comparison of mask optimization strategies, DP optimization (AP 22.2) is far superior to the exhaustive search of top-1 view (19.8) or top-10 views (20.8).
- Under the open-vocabulary setting, the improvement on tail classes ($AP_{\text{tail}}=26.4$) is particularly significant, demonstrating that the method is more friendly to rare classes.
- On ScanNet++, even without using 3D proposals, the 2D-only scheme outperforms Open3DIS which utilizes SAM-HQ.

## Highlights & Insights
- The concept of "tracking instead of association" is simple yet profound—leveraging the temporal consistency of SAM-2 naturally resolves the inconsistency of multi-view 2D masks.
- 3D-aware pivot view selection is highly practical, and neighboring visibility weighting is a clever trick.
- The design of using dynamic programming to solve the NP-hard problem is worth learning; although it does not guarantee global optimality, its practical performance is outstanding.
- The training-free paradigm allows the method to generalize directly to novel scenes and datasets.

## Limitations & Future Work
- It heavily relies on the segmentation and tracking quality of SAM-2; any 2D errors from SAM-2 will propagate to the 3D results.
- The DP algorithm can only decide "include all or none" at each step and cannot make fine-grained choices for individual superpoints, which may not yield the global optimum.
- For highly occluded or overlapping objects, tracking can be lost or confused.
- Incorporating quantum computing or more advanced optimization algorithms could be considered to improve the solving of the NP-hard problem.

## Related Work & Insights
- Relation to Open3DIS: Shares the basic paradigm of superpoints + lifting 2D masks to 3D, but replaces association with tracking, which fundamentally changes the pipeline.
- The integration of SAM-2 is key; its video segmentation capability extends the 2D-to-3D lifting from independent frames to temporal consistency.
- Insight: The application of foundation models in 3D tasks still possesses huge potential; the key lies in how to design pipelines that exploit their strengths.

## Supplementary Analysis

### Method Complexity
- For each superpoint, the projection histogram across $T$ frames needs to be calculated. The complexity scales linearly with the number of frames.
- The time complexity of DP optimization is $O(T \cdot L)$, where $L$ is the number of superpoints, which runs efficiently in practice.
- Iterative sampling typically requires only 2-3 rounds to cover the vast majority of objects in the scene.

### Generality with 2D Tracking Models
- Currently, SAM-2 ViT-L is used as the segmentation and tracking backbone.
- SAM-2 can be replaced with other video segmentation models (e.g., XMem), showcasing the framework's generality.
- The memory window limitation of SAM-2 (7 frames) is resolved via the projection point re-querying mechanism.
- Future performance improvements of the SAM family of models will directly benefit this method.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The core idea of "tracking instead of association" is novel and intuitive, though the overall framework is still based on the existing paradigm of superpoints + 2D masks.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three evaluation settings (class-agnostic, open-vocabulary, open-ended) across two datasets, with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ The description of the method is clear, and the algorithm pseudocode and mathematical derivations are formal.
- **Value**: ⭐⭐⭐⭐ Makes significant methodological contributions to standard training-free 3D instance segmentation, offering valuable guidance for future works.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ASSIST-3D: Adapted Scene Synthesis for Class-Agnostic 3D Instance Segmentation](../../AAAI2026/3d_vision/assist-3d_adapted_scene_synthesis_for_class-agnostic_3d_instance_segmentation.md)
- [\[ICCV 2025\] CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](../../ICCV2025/3d_vision/cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [\[CVPR 2025\] Sketchy Bounding-Box Supervision for 3D Instance Segmentation](sketchy_bounding-box_supervision_for_3d_instance_segmentation.md)
- [\[CVPR 2025\] Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation](relation3d_enhancing_relation_modeling_for_point_cloud_instance_segmentation.md)
- [\[CVPR 2025\] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting](rethinking_end-to-end_2d_to_3d_scene_segmentation_in_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
