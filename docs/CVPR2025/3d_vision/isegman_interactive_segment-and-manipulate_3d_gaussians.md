---
title: >-
  [Paper Note] iSegMan: Interactive Segment-and-Manipulate 3D Gaussians
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] iSegMan proposes an interactive 3DGS segmentation and manipulation framework that requires no scene-specific training. It achieves precise 3D region control via Epipolar-guided Interaction Propagation (EIP) and Visibility-based Gaussian Voting (VGV), paired with a comprehensive manipulation toolbox supporting various functions such as semantic editing, colormapping, scaling, copying-and-pasting, combining, and deleting.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Interactive Segmentation"
  - "Scene Manipulation"
  - "Epipolar Constraint"
  - "SAM"
date: 2026-05-08
content_hash: 0a34168c60fb35ec
---

# iSegMan: Interactive Segment-and-Manipulate 3D Gaussians

**Conference**: CVPR 2025  
**arXiv**: [2505.11934](https://arxiv.org/abs/2505.11934)  
**Code**: [https://zhao-yian.github.io/iSegMan](https://zhao-yian.github.io/iSegMan)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Interactive Segmentation, Scene Manipulation, Epipolar Constraint, SAM

## TL;DR
iSegMan proposes an interactive 3DGS segmentation and manipulation framework that requires no scene-specific training. It achieves precise 3D region control via Epipolar-guided Interaction Propagation (EIP) and Visibility-based Gaussian Voting (VGV), paired with a comprehensive manipulation toolbox supporting various functions such as semantic editing, colormapping, scaling, copying-and-pasting, combining, and deleting.

## Background & Motivation

1. **Background**: 3D Gaussian Splatting (3DGS) has driven rapid developments in 3D scene manipulation due to its highly efficient rendering and explicit representation. Existing methods like GaussianEditor perform text-driven editing, while Instruct-GS2GS executes global editing via instructions.

2. **Limitations of Prior Work**: (1) Existing 3DGS manipulation methods struggle with precise control over manipulation areas, easily affecting irrelevant regions. (2) Text-prompt-based region control (e.g., GaussianEditor) is limited by the coarse granularity of text descriptions, failing to segment fine-grained areas. (3) Existing 3D interactive segmentation methods (e.g., SA3D, SAGA) require scene-specific parameter training (feature distillation), which hinders efficiency and flexibility.

3. **Key Challenge**: Scene manipulation requires precise region control, yet existing approaches either offer coarse control (text-driven) or require time-consuming training (feature distillation). How to achieve "zero training, precise control, and interactive feedback" becomes the key challenge.

4. **Goal**: To build a training-free 3D region control module that supports user interaction via 2D clicks from any viewpoint, and on top of this, provide a comprehensive scene manipulation toolbox.

5. **Key Insight**: Leverage epipolar geometric constraints instead of brute-force feature matching to propagate user interactions across perspectives, and utilize the alpha-blended visibility information inherent in Gaussian Splatting instead of feature training to extract 3D regions.

6. **Core Idea**: Epipolar constraint to narrow the search space + visibility-weighted voting to replace feature distillation = training-free precise 3D segmentation.

## Method

### Overall Architecture
iSegMan takes user-provided 2D clicks (positive/negative) from an arbitrary perspective and outputs a segmented subset of 3D Gaussians. The overall pipeline is: EIP propagates user clicks to multiple views $\rightarrow$ SAM generates 2D segmentation masks for each view $\rightarrow$ VGV extracts the 3D region based on the masks and Gaussian visibility voting $\rightarrow$ the manipulation toolbox executes user-specified functions on the selected region. The entire process requires no scene-specific training.

### Key Designs

1. **Epipolar-guided Interaction Propagation (EIP)**:

    - **Function**: Efficiently and robustly propagate 2D user clicks from a single perspective to other perspectives.
    - **Mechanism**: Given a 2D click $p_v$ in view $v$, it is projected onto a 3D ray $r_{p_v}$ using camera intrinsics and extrinsics, and then the epipolar line $e_{p_v}^{\tilde{v}}$ of this ray on the new view $\tilde{v}$ is computed. The matching search is restricted to the epipolar line, utilizing DINO features to perform similarity matching on the line: $p_{\tilde{v}} = \text{Upsample}(I_{\tilde{v}}[\text{argmax}(\mathcal{A}_{p_v}^{\tilde{v}})])$, where the affinity $\mathcal{A}$ is calculated only on the epipolar sample points.
    - **Design Motivation**: Directly performing feature matching across the entire image results in a large search space, high noise, and low robustness. The epipolar constraint strictly reduces the 2D search space to a 1D line segment, significantly enhancing efficiency and accuracy. Bresenham's algorithm is adopted to efficiently sample discrete features along the epipolar line.

2. **Visibility-based Gaussian Voting (VGV)**:

    - **Function**: Extract target 3D Gaussians based on multi-view 2D segmentation masks without any feature training.
    - **Mechanism**: Treat 2D pixels as "voting participants" and 3D Gaussians as "candidates." The voting weight of each participant is determined by the visibility of the pixel to the Gaussian (alpha blending weight): $\Upsilon_{i,j} = \sigma_i \cdot \alpha_i \prod_{k=1}^{i-1}(1-\alpha_k)$. The votes from all pixels across all views are summed to obtain the total vote count $\Psi_j$ for each Gaussian, and Gaussians exceeding a threshold are selected. Additionally, an Iterative Inspection Mechanism (IIM) is introduced: it checks whether the intersection between the SAM-predicted mask and the rendered mask of the currently selected Gaussians is valid, filtering out erroneous segmentations caused by occlusion or out-of-view movements.
    - **Design Motivation**: Existing methods achieve 2D-to-3D mapping by training a 3D semantic feature field, which is time-consuming and inflexible. The alpha-blended rendering of Gaussian Splatting inherently provides the contribution weights from pixels to Gaussians. Leveraging this information directly for voting completely eliminates the training stage. The asymmetry of voting weights (higher visibility $\rightarrow$ larger voting weight) naturally handles occlusions.

3. **Manipulation Toolbox**:

    - **Function**: Provide a complete set of 3D scene manipulation functions.
    - **Mechanism**: (1) Semantic Editing: InstructPix2Pix is utilized to edit the rendered views, iteratively updating the Gaussian parameters of the selected region via L1 and perceptual losses, accompanied by an annealing strategy to improve stability; (2) Colormapping: Directly modify the color attributes of Gaussians, supporting both color replacement and balanced coloring modes; (3) Scaling: Scale orientation vectors and scale factors simultaneously to preserve rigid body transformation invariance; (4) Copying-and-Pasting/Combining/Deleting: Directly manipulate the collection of Gaussians.
    - **Design Motivation**: Precise region control builds the foundation for various manipulation functions, and the toolbox-based design makes the framework highly extensible.

### Loss & Training
Update loss for semantic editing: $\nabla_\theta \Theta_s = \mathbb{E}_v[(\frac{\partial \|I_v^e - I_v\|_1}{\partial I_v} + \frac{\partial \mathcal{D}(I_v, I_v^e)}{\partial I_v}) \cdot \frac{\partial I_v}{\partial \theta}]$, containing the L1 image loss and LPIPS perceptual distance. An annealing strategy is adopted to progressively decay the update step size. The VGV module itself requires no training.

## Key Experimental Results

### Main Results: Interactive 3D Segmentation (SPIn-NeRF Dataset)

| Method | Requires Training | mIoU(%) | mAcc(%) | Feature Time | Segmentation Time |
|------|--------|---------|---------|---------|---------|
| SA3D | ✓ | 91.9 | 98.8 | 5min | 30s |
| SAGA | ✓ | 88.0 | 98.5 | ~1.5h | 10ms |
| LangSplat | ✓ | 69.5 | 94.5 | ~2.5h | - |
| **iSegMan** | ✗ | **92.4** | **99.1** | **52s** | **6s** |

### Quantitative Comparison on Semantic Editing

| Metric | Instruct-GS2GS | GaussianEditor | iSegMan |
|------|----------------|----------------|---------|
| User study↑ | 2.10±0.20 | 3.32±0.40 | **4.52±0.20** |
| CLIP dir↑ | 0.1647 | 0.2071 | **0.2189** |

### Key Findings
- iSegMan achieves the best segmentation performance (mIoU 92.4%) while completely avoiding scene-specific training.
- The feature extraction time is only 52 seconds (vs 1.5 hours for SAGA and 5 minutes for SA3D) because it does not require training a feature field.
- In semantic editing, iSegMan avoids affecting irrelevant regions through precise region control.
- The text-based region control of GaussianEditor lacks sufficient precision, easily leading to editing artifacts.
- The Iterative Inspection Mechanism (IIM) effectively eliminates erroneous SAM masks caused by occlusion or entering/leaving the field of view.

## Highlights & Insights
- **Clever Application of Epipolar Constraints**: Incorporating classic geometric constraints from multi-view stereo into 2D interaction propagation reduces the 2D matching problem down to a 1D search along epipolar lines. This idea is transferable to any task requiring cross-perspective propagation of 2D annotations.
- **Voting Mechanism Replacing Feature Training**: Using the alpha-blended weights of Gaussian Splatting as voting weights completely bypasses the feature distillation phase, demonstrating that the rendering process of 3DGS natively contains rich 2D-3D correspondence information.
- **Interactive Editing Loop**: Supporting progressive editing for complex requirements (e.g., "turning a person into a bronze statue wearing a green shirt and yellow pants") while reusing intermediate results to enhance efficiency.

## Limitations & Future Work
- The semantic editing part relies on InstructPix2Pix, and is thus limited by the performance of the 2D editor.
- Epipolar matching utilizes DINO features, which may fail in areas with repetitive textures or weak textures.
- The voting threshold needs to be manually set, and different scenes may require different thresholds.
- Currently, it only supports 2D click interactions and does not support richer interactive forms (such as scribbles, or text + click hybrid).
- Future improvement: Combine VGV with SAM 2 to handle video or dynamic scenes.

## Related Work & Insights
- **vs SA3D**: SA3D alternately trains the 3D mask using mask inverse rendering and cross-view self-prompting, requiring a training phase. iSegMan replaces training with voting, achieving faster speed and better performance.
- **vs SAGA**: SAGA extracts masks for all views using SAM first and then distills 3D features, taking about 1.5 hours in pre-processing. iSegMan only processes views to which user interactions are propagated, computing on demand.
- **vs GaussianEditor**: GaussianEditor employs text to locate editing areas, which is coarse-grained. iSegMan utilizes click interactions, enabling precision down to arbitrary fine-grained regions.

## Rating
- Novelty: ⭐⭐⭐⭐ The designs of EIP and VGV are novel and practical, with training-free segmentation being a significant practical innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive quantitative and qualitative evaluations on both segmentation and editing tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; the formulation of the voting mechanism is formal yet intuitive.
- Value: ⭐⭐⭐⭐⭐ High practical value, offering a truly interactive tool for 3DGS scene manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Click-Gaussian: Interactive Segmentation to Any 3D Gaussians](../../ECCV2024/3d_vision/click-gaussian_interactive_segmentation_to_any_3d_gaussians.md)
- [\[CVPR 2025\] IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments](iaao_interactive_affordance_learning_for_articulated_objects_in_3d_environments.md)
- [\[CVPR 2025\] Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation](vid2sim_realistic_and_interactive_simulation_from_video_for_urban_navigation.md)
- [\[CVPR 2025\] Textured Gaussians for Enhanced 3D Scene Appearance Modeling](textured_gaussians_for_enhanced_3d_scene_appearance_modeling.md)
- [\[CVPR 2025\] RNG: Relightable Neural Gaussians](rng_relightable_neural_gaussians.md)

</div>

<!-- RELATED:END -->
