---
title: >-
  [Paper Note] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions
description: >-
  [CVPR 2025][3D Vision][Human-Object Interaction Reconstruction] The authors propose Open3DHOI, the first open-vocabulary in-the-wild 3D Human-Object Interaction (HOI) dataset (comprising over 2.5k images, 133 object categories, and 120 action categories). They also design Gaussian-HOI, a 3D Gaussian Splatting-based HOI optimizer that reconstructs spatial human-object interactions and learns contact regions via Gaussian rendering.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Human-Object Interaction Reconstruction"
  - "Open-Vocabulary"
  - "3D Gaussian Splatting"
  - "Contact Region Estimation"
  - "HOI Dataset"
date: 2026-05-08
content_hash: 9c4a9206b4b63c00
---

# Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions

**Conference**: CVPR 2025  
**arXiv**: [2503.15898](https://arxiv.org/abs/2503.15898)  
**Code**: [GitHub](https://wenboran2002.github.io/3dhoi/)  
**Area**: 3D Vision  
**Keywords**: Human-Object Interaction Reconstruction, Open-Vocabulary, 3D Gaussian Splatting, Contact Region Estimation, HOI Dataset

## TL;DR

The authors propose Open3DHOI, the first open-vocabulary in-the-wild 3D Human-Object Interaction (HOI) dataset (comprising over 2.5k images, 133 object categories, and 120 action categories). They also design Gaussian-HOI, a 3D Gaussian Splatting-based HOI optimizer that reconstructs spatial human-object interactions and learns contact regions via Gaussian rendering.

## Background & Motivation

- Reconstructing 3D human-object interactions (HOI) from a single image is a fundamental vision problem but is severely constrained by the scarcity of 3D data.
- Existing 3D HOI datasets (e.g., BEHAVE, InterCap) are primarily recorded in fixed indoor environments and contain very few object classes (8-40 classes), far lagging behind the diversity of 2D HOI datasets.
- Although WildHOI and 3DIR utilize in-the-wild images, they only include a small number of object categories and rely on unrealistic CAD models.
- 2D HOI datasets (e.g., HICO-DET, HAKE) provide rich 2D annotations and diverse object categories.
- The maturation of single-image 3D reconstruction technologies (e.g., InstantMesh) enables reconstructed 3D assets from 2D HOI images.
- Existing training-free 3D HOI reconstruction methods (e.g., PHOSA) only optimize object poses using silhouette losses, which limits their performance.
- While training-based methods perform well on predefined object categories, they struggle to generalize to open-world environments.
- There is currently a lack of unified metrics to evaluate 3D interaction quality.

## Method

### Overall Architecture

The proposed framework consists of two parts: a **data annotation pipeline** and a **3D HOI reconstruction optimizer**. The annotation pipeline selects over 15k images from 2D HOI datasets (HAKE + SWIG-HOI), reconstructs the objects using InstantMesh and the humans using OSX, and obtains over 2.5k high-quality 3D HOI annotations through coarse reconstruction (depth projection alignment) followed by fine annotation (using Blender and web-based tools). The reconstruction optimizer, Gaussian-HOI, is based on 3D Gaussian Splatting; it jointly optimizes human SMPL-X parameters and 6D object poses while learning human-object contact regions through Gaussian properties.

### Key Designs

**Design 1: Coarse-to-Fine 3D HOI Annotation Pipeline**
- **Function**: Efficiently obtains high-quality 3D HOI annotations from single-view images.
- **Mechanism**: First, occluded object areas are healed using occlusion-complemented Stable Diffusion. Next, monocular depth estimation and mask extraction generate depth point clouds for both the human and the object, and coarse reconstruction is achieved through point cloud alignment. Finally, the annotations are refined using a Filtering Tool (to evaluate SMPL-X and object reconstruction quality, and annotate contact regions) and a 3D Interaction Tool (for coarse adjustment in Blender and fine-tuning via web-based tools).
- **Design Motivation**: By starting directly from 2D annotations and existing reconstruction tools, this approach avoids the high cost of multi-view RGB-D capture. The coarse-to-fine strategy significantly reduces the human annotation workload.

**Design 2: HOI-Gaussian Optimizer**
- **Function**: Reconstructs 3D human-object interactions from a single image without training.
- **Mechanism**: The human Gaussians $g_h$ are initialized using the SMPL-X vertices, and the object Gaussians $g_o$ are initialized using the object mesh vertices. The 6D object pose is optimized via learnable parameters $W_{obj}$. The joint interaction Gaussians are defined as $g_{hoi} = g_h \oplus g_o$. 2D alignment is achieved through Gaussian rendering losses, and the 3D spatial relationship is optimized by combining collision, depth, and contact losses.
- **Design Motivation**: Compared to PHOSA, which only utilizes silhouette losses, 3D Gaussians exploit color matching and depth information, reducing scenarios where silhouettes look similar but actual 3D poses differ significantly.

**Design 3: Gaussian-based Contact Region Learning**
- **Function**: Automatically identifies potential contact regions between humans and objects.
- **Mechanism**: This leverages the physical characteristic that the opacity $\alpha$ of occluded areas naturally decreases during Gaussian rendering. First, a low opacity is assigned to back-facing points based on normal directions. During optimization, $\alpha$ decreases in occluded areas. Combined with a Chamfer distance constraint, the contact score is computed as $c = w_\alpha \cdot \text{Norm}(\alpha^h) + w_d \cdot d_C(p^h, p^o)^h$.
- **Design Motivation**: Direct determination of contact regions from monocular images is challenging. However, the occluded portions of the human body can be indirectly inferred as potential contact areas using the changes in opacity during Gaussian rendering.

### Loss & Training

The total loss is formulated as $\mathcal{L} = w_r \cdot \mathcal{L}_r + w_{hoi} \cdot \mathcal{L}_{hoi}$:
- Rendering loss $\mathcal{L}_r$: L1 + L2 mask + SSIM + LPIPS calculated separately for $g_{hoi}$, $g_h$, and $g_o$.
- HOI loss $\mathcal{L}_{hoi} = \mathcal{L}_{cont} + \mathcal{L}_{colli} + \mathcal{L}_{depth}$: Includes contact loss (Chamfer distance between the human contact region and the object), collision loss, and ordinal depth loss.

## Key Experimental Results

### Main Results: Object Pose Reconstruction Comparison

| Method | Scale↓ | Translation(cm)↓ | Rotation↓ | Chamfer Dist.(cm) |
|------|--------|-------------------|-----------|-------------------|
| PHOSA | 0.39 | 77.79 | 0.95 | 49.1 |
| Ours w/o HOI Loss | 0.25 | 38.66 | 0.45 | 16.9 |
| **Ours** | **0.16** | **38.44** | **0.41** | **19.3** |

### Ablation Study: Collision-Contact Evaluation ($Co^2$ Metric)

| Method | $Co^2$↓ | Collision↓ | Contact↓ |
|------|---------|------------|----------|
| PHOSA | 0.431 | 0.105 | 0.326 |
| Coarse Recon | 0.248 | 0.083 | 0.165 |
| Gs only | 0.287 | 0.136 | 0.151 |
| Gs & depth & colli | 0.188 | 0.045 | 0.143 |
| **Gs & depth & colli & cont** | **0.181** | **0.053** | **0.128** |

### Key Findings
- Gaussian-HOI outperforms PHOSA by a large margin across all object pose metrics (reducing translation error by over 50%).
- Solely relying on Gaussian optimization is insufficient to improve 3D interaction quality; it must be coupled with the HOI loss.
- While contact loss decreases the Contact score, it may slightly increase collision (as the object is pulled closer to the contact region). The $Co^2$ metric balances this trade-off.
- PointLLM exhibits limited ability to comprehend 3D HOIs. However, introducing the object name significantly enhances its action reasoning performance (Top-1 accuracy rises from 20% to 47%).

## Highlights & Insights

1. **First Open-Vocabulary In-the-Wild 3D HOI Dataset**: Includes 133 object categories and 120 action categories, vastly outperforming BEHAVE (10 object categories) and 3DIR (21 categories).
2. **Innovative Application of Gaussian Rendering**: Used not only for 2D alignment optimization but also cleverly leverages opacity attributes to infer contact regions.
3. **Proposed the $Co^2$ Evaluation Metric**: Unifies the evaluation of physical collision and contact quality, filling a gap in 3D HOI reconstruction assessment.
4. **Scalability of the Annotation Method**: The pipeline can leverage future, more powerful 3D-AIGC tools to further improve efficiency.

## Limitations & Future Work

- The dataset scale (over 2.5k) remains insufficient for training large-scale models.
- Object reconstruction quality heavily depends on InstantMesh; fine-grained interactions (such as finger grasping) are still difficult to reconstruct accurately.
- Only single-person scenes are annotated; multi-person interactions are not yet covered.
- General 3D understanding models like PointLLM perform poorly on HOI tasks, necessitating more fine-grained, data-driven improvements.
- Combining multi-view inputs or video sequences could be explored to further enhance reconstruction quality in the future.

## Related Work & Insights

- Compared to silhouette-based optimization methods like PHOSA, 3D Gaussians leverage richer color and depth information.
- GauHuman demonstrates that 3D Gaussians can be used to optimize human parameters; this paper extends that concept to human-object interaction scenarios.
- The concept of the $Co^2$ metric (balancing collision and contact) can be extended to other tasks that require evaluating physical plausibility.

## Rating

⭐⭐⭐⭐ — Building this dataset is highly valuable, and the design of the Gaussian-HOI optimizer is solid. However, the scale of the test set is still limited, and there remains room for improvement in reconstructing complex interactions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Reconstructing Animals and the Wild](reconstructing_animals_and_the_wild.md)
- [\[CVPR 2025\] Guiding Human-Object Interactions with Rich Geometry and Relations](guiding_human-object_interactions_with_rich_geometry_and_relations.md)
- [\[CVPR 2025\] Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning](reconstructing_close_human_interaction_with_appearance_and_proxemics_reasoning.md)
- [\[CVPR 2025\] HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)
- [\[CVPR 2025\] SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)

</div>

<!-- RELATED:END -->
