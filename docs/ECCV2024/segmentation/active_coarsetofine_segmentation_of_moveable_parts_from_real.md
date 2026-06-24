---
title: >-
  [Paper Note] Active Coarse-to-Fine Segmentation of Moveable Parts from Real Images
description: >-
  [ECCV 2024][Segmentation][Moveable Part Segmentation] Proposes the first active learning framework for instance segmentation of moveable parts in real-world indoor RGB images. Utilizing a pose-aware masked attention network, the framework achieves coarse-to-fine segmentation. It requires manual annotation of only 11.45% of the images to obtain fully verified high-quality segmentation results, saving 60% of manual effort compared to the best non-active learning methods.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Moveable Part Segmentation"
  - "Active Learning"
  - "Coarse-to-Fine"
  - "Pose-Aware Attention"
  - "Real-World Annotation"
date: 2026-05-08
content_hash: 28bf67c2091342cf
---

# Active Coarse-to-Fine Segmentation of Moveable Parts from Real Images

**Conference**: ECCV 2024  
**arXiv**: [2303.11530](https://arxiv.org/abs/2303.11530)  
**Code**: None  
**Area**: Semantic Segmentation / Active Learning / Articulated Object Understanding  
**Keywords**: Moveable Part Segmentation, Active Learning, Coarse-to-Fine, Pose-Aware Attention, Real-World Annotation  

## TL;DR
Proposes the first active learning framework for instance segmentation of moveable parts in real-world indoor RGB images. Utilizing a pose-aware masked attention network, the framework achieves coarse-to-fine segmentation. It requires manual annotation of only 11.45% of the images to obtain fully verified high-quality segmentation results, saving 60% of manual effort compared to the best non-active learning methods.

## Background & Motivation
Most daily objects (such as cabinets, refrigerators, and dishwashers) contain moveable parts (doors, drawers). Understanding these parts is crucial for robotic manipulation and embodied AI. Although existing methods like OPD and OPDMulti can detect openable parts from images, their training data sources are problematic—annotated on 3D reconstructed meshes and then projected onto 2D images, which introduces reconstruction and projection errors. Crucially, models trained on synthetic data suffer a drastic performance drop in real-world scenarios (segmentation accuracy drops from ~75% to ~30%). The core bottleneck is that high-quality part-level segmentation annotation on real images is extremely costly; direct manual annotation is impractical, while indirect 3D projection annotation yields poor quality.

## Core Problem
How can high-precision instance segmentation annotations of moveable parts on real-world RGB images be obtained with minimal human labor cost? The paper formulates this problem as: given a batch of unlabeled real-world images, the goal is to obtain manually verified and precise segmentation results for all images while keeping the number of images that require manual annotation as small as possible.

## Method

### Overall Architecture
The method consists of two major parts: a two-stage pose-aware masked attention network (for segmentation prediction) and a coarse-to-fine active learning strategy (for iterative annotation optimization). The input is a single RGB image, and the output is the instance segmentation masks and semantic labels (door/drawer) of all moveable parts of the articulated objects in the image.

### Key Designs
1. **Coarse Stage—Object-level Perception**: The input image is fed into a MaskRCNN backbone to extract multi-scale features and object bounding boxes. The normalized coordinates of the bounding boxes are encoded into object query embeddings, which are sent to a modified Deformable DETR encoder-decoder. The decoded queries predict object class, 6DoF pose, interaction direction (one of six discrete directions indicating the principal direction of part movement), and object mask via four MLP heads. Utilizing the estimated 3D pose and interaction direction, the 2D region corresponding to the object's interacting surface is calculated. This is combined with the object mask to crop a refined mask containing only the interacting surface, filtering out background and irrelevant object surface interference.

2. **Fine Stage—Part-level Segmentation**: This stage is based on a Mask2Former masked attention decoder composed of three cascaded layers. Taking the refined mask output from the coarse stage and upsampled features from the pixel decoder as input, it produces instance segmentation masks, bounding boxes, and semantic labels of the moveable parts through layer-by-layer refinement. The pose-aware refined mask forces the network to focus only on features within the object's interacting surface region, significantly lowering the segmentation difficulty.

3. **Coarse-to-Fine Active Learning Strategy**: It proceeds in two rounds: (a) Coarse AL—manual verification/correction of the interaction direction predictions to establish a reliable refined mask prior; (b) Fine AL—sorting part segmentation results into three categories: *perfect* (directly added to the training set), *missed* (manually annotated from scratch using Labelme and then added to the training set), and *fair* (retained in the evaluation set to be reassessed in the next round). After each iteration, the training set expands, the model becomes stronger, and the proportion requiring manual annotation continuously decreases until the evaluation set is cleared.

### Loss & Training
The total loss is $L = L_{class} + L_{dir} + L_{om} + L_{pos} + L_{fine}$, where $L_{pos} = \lambda_t L_t + \lambda_{rot} L_{rot}$ (L2 translation loss + geodesic rotation loss, $\lambda_t=2, \lambda_{rot}=1$). The model is first pre-trained on PartNet-Mobility synthetic data (~32K images) for 2K epochs, and then fine-tuned on real images for 4.5K epochs. During fine-tuning, the MLP weights are frozen (since real-world data lacks ground-truth pose annotations).

## Key Experimental Results

| Dataset | Metric | Ours (full) | OPDFormer-C | Ours w/o AL | Gain |
|--------|------|------|----------|----------|------|
| Self-built Dataset (500 images) | segm mAP@0.5 | **91.3** | 68.4 | 77.3 | +22.9 vs SOTA |
| OPDReal | segm mAP@0.5 | **51.6** | 46.3 | - | +5.3 |
| OPDMulti | segm mAP@0.5 | **31.5** | 27.6 | - | +3.9 |

Annotation efficiency comparison (2000 images):

| Method | No. of Images Requiring Manual Annotation | Total Time (h) |
|------|-----------------|----------|
| Grounded-SAM (Non-AL) | 1,888 | 35.5 |
| OPDFormer-C (Non-AL) | 792 | 16.3 |
| Ours (AL) | **229** | **6.5** |

### Ablation Study
- The complete system (mask+pose+interaction direction+AL) achieves 91.3 mAP. Removing AL $\rightarrow$ 77.3 (-14.0), removing pose and interaction direction $\rightarrow$ 89.1 (-2.2), removing all except AL $\rightarrow$ 87.3 (-4.0).
- Coarse-to-fine AL vs. Fine AL only: The performance gap is only 4.5% on 500 images, but widens to over 13% on 2000 images, demonstrating that the coarse-to-fine strategy is more effective for large-scale annotation.
- The introduction of pose and interaction direction estimation is key to increasing efficiency—filtering out irrelevant regions via refined masks reduces *fair* predictions and minimizes AL iteration cycles.

## Highlights & Insights
- **Clever utilization of problem hierarchy**: The mapping from object $\rightarrow$ interacting surface $\rightarrow$ part naturally aligns with the coarse-to-fine strategy. Instead of brute-forcing the application of a framework, this design genuinely leverages the structure of the task.
- **Interaction direction estimation is a key design aspect**: Utilizing a 6DoF pose combined with discrete interaction directions to locate the object's "interacting surface" is far simpler than segmenting parts directly. Furthermore, any errors can be quickly corrected manually.
- **Practical three-way sorting strategy in AL**: Categorizing samples into *perfect*, *missed*, and *fair* ensures that annotators only need to handle the most challenging cases, while retaining *fair* samples prevents contamination of the training set.

## Limitations & Future Work
- Only supports 6 classes of indoor articulated objects (e.g., Storage, Fridge, Dishwasher) with only two part labels (door/drawer). Generalization to more object types and motion types (e.g., knobs, sliding) remains unverified.
- Relies on 6DoF pose estimation, while pose ground truth is unavailable during fine-tuning, potentially leading to pose drift.
- The dataset contains only 2,550 images, which is relatively small; annotation quality depends heavily on the consistency of human annotators.
- Lacks deeper integration and comparative evaluation against the SAM series' few-shot or prompt-based segmentation.

## Related Work & Insights
- **vs. OPDFormer-C**: Though both are based on Mask2Former, OPDFormer-C does not exploit object pose and interaction direction. Consequently, its segmentation suffers from significant interference from backgrounds and irrelevant object surfaces. Ours squeezes the segmentation region to the vicinity of the interacting surface via pose-aware filtering in the coarse stage, significantly outperforming OPDFormer-C (91.3 vs 68.4).
- **vs. Grounded-SAM**: General-purpose vision-language foundation models exhibit weak zero-shot transfer on fine-grained tasks like moveable part segmentation (mAP is only 23.1). This demonstrates that general models cannot replace task-specific designs.
- **vs. Active Segmentation Methods (ECCV 2022 point/region supervision)**: Existing active annotation methods employ click or region inputs to correct masks. However, for multi-part articulated objects, point-click supervision introduces ambiguity. Ours opts for image-level classification (*perfect*/*missed*/*fair*) rather than pixel-level correction, making it more tailored for multi-instance scenarios.

## Related Work & Insights
The core insight of this paper is that when annotation cost is the bottleneck, the hierarchical structure of a task can be leveraged to design coarse-to-fine active learning strategies. Initially using low-cost, high-accuracy coarse judgments (such as interaction direction verification) constrains the search space for the more challenging, fine-grained tasks (part segmentation). This "easy-to-hard, step-by-step constraint" paradigm is transferable to other hierarchical annotation tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The first to introduce the active learning framework to moveable part segmentation. Combining coarse and fine details to leverage task hierarchy is a solid idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons across three datasets, comprehensive ablation studies, and efficiency analyses.
- Writing Quality: ⭐⭐⭐⭐ Clarified motivations and a complete pipeline description, although mathematical notations are somewhat redundant.
- Value: ⭐⭐⭐⭐ Highly practical for data preparation in robotic manipulation and embodied AI, though restricted to a relatively narrow set of scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Universal 3D Shape Matching via Coarse-to-Fine Language Guidance](../../CVPR2026/segmentation/universal_3d_shape_matching_via_coarse-to-fine_language_guidance.md)
- [\[ECCV 2024\] SPIN: Hierarchical Segmentation with Subpart Granularity in Natural Images](spin_hierarchical_segmentation_with_subpart_granularity_in_natural_images.md)
- [\[CVPR 2026\] SegGBC: Justifiable Coarse-to-Fine Granular-Ball Computing for Enhancing Clustering Image Segmentation](../../CVPR2026/segmentation/seggbc_justifiable_coarse-to-fine_granular-ball_computing_for_enhancing_clusteri.md)
- [\[ECCV 2024\] Efficient and Versatile Robust Fine-Tuning of Zero-shot Models](efficient_and_versatile_robust_fine-tuning_of_zero-shot_models.md)
- [\[NeurIPS 2025\] Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation](../../NeurIPS2025/segmentation/diffusion-driven_two-stage_active_learning_for_low-budget_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
