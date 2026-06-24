---
title: >-
  [Paper Note] 3×2: 3D Object Part Segmentation by 2D Semantic Correspondences
description: >-
  [ECCV 2024][3D Vision][3D part segmentation] Proposes a training-free 3D object part segmentation method, 3-By-2, which utilizes 2D semantic correspondences from diffusion models (DIFT) to transfer part labels from annotated 2D datasets or a small number of 3D annotated objects to 3D, achieving state-of-the-art (SOTA) performance under both zero-shot and few-shot settings.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D part segmentation"
  - "diffusion features"
  - "semantic correspondence"
  - "training-free"
  - "SAM"
date: 2026-05-08
content_hash: 15ca84c9cb7d49ed
---

# 3×2: 3D Object Part Segmentation by 2D Semantic Correspondences

**Conference**: ECCV 2024  
**arXiv**: [2407.09648](https://arxiv.org/abs/2407.09648)  
**Code**: [https://ngailapdi.github.io/projects/3by2/](https://ngailapdi.github.io/projects/3by2/) (Project Page)  
**Area**: 3D Vision / Zero-Shot & Few-Shot Learning / Part Segmentation  
**Keywords**: 3D part segmentation, diffusion features, semantic correspondence, training-free, SAM

## TL;DR
Proposes a training-free 3D object part segmentation method, 3-By-2, which utilizes 2D semantic correspondences from diffusion models (DIFT) to transfer part labels from annotated 2D datasets or a small number of 3D annotated objects to 3D, achieving state-of-the-art (SOTA) performance under both zero-shot and few-shot settings.

## Background & Motivation
3D object part segmentation is crucial in fields such as robotics and computer graphics, but 3D annotated data is scarce and expensive (existing large-scale 3D part datasets like PartNet are synthetic). Existing methods primarily rely on language input (e.g., PartSLIP uses GLIP for text-driven part detection). However, language descriptions of parts contain inherent ambiguities—the same part can be described by different phrases, and expressing fine-grained parts linguistically is particularly difficult (e.g., "back_frame_vertical_bar"). In contrast, images contain rich shape, texture, and spatial relationship information. The visual similarity of identical parts across different objects can be directly compared, independent of linguistic differences.

## Core Problem
**How to leverage existing 2D part segmentation datasets to achieve high-precision 3D object part segmentation without 3D annotations or language inputs?** The challenges include: (1) precisely determining the part boundaries of 3D point clouds is inherently difficult; (2) the method must flexibly adapt to diverse granularities of part definitions; (3) aligning and aggregating multi-view predictions consistently.

## Method

### Overall Architecture
Given a 3D query object, the method outputs part segmentation labels for its point cloud. The pipeline consists of four steps:
1. Render 2D RGB images of the query object from $K$ camera viewpoints.
2. Segment parts on each rendered image using 2D semantic correspondences (with labels sourced from a database).
3. Fuse multi-view 2D predictions using a Mask-Consistency aggregation module.
4. Back-project 2D predictions to the 3D point cloud using depth information.

The database $\mathcal{D}$ can be a 2D part dataset (e.g., PACO, used for zero-shot) or multi-view renderings of a few annotated 3D objects (few-shot).

### Key Designs

1. **DIFT-based Single-Pixel Label Transfer**: For each foreground pixel $p$ in the query image, diffusion features (DIFT) are utilized to calculate the cosine similarity with all pixels of all images in the database, finding the best-matching pixel $p'$ and assigning its label to $p$. This is the first work to utilize the semantic correspondence capabilities of diffusion models for label transfer in 3D part segmentation. To address the computational overhead, a **coarse-to-fine correspondence search** is proposed: it first localizes the best-matched region on the coarse DIFT feature maps, and then performs a fine search in a $3\times3$ local window, accelerating the query on $800\times800$ images by about 2000 times.

2. **Non-overlapping 2D Mask Proposal**: SAM can generate high-quality boundary masks, but its multi-granularity output is highly overlapping. This paper proposes a non-overlapping mask generation module: masks output by SAM are sorted in descending order of their areas, and smaller masks are overlaid on top of larger ones. The visible part of each mask is then taken as the final mutually exclusive mask. This preserves SAM's precise boundaries while eliminating granularity ambiguity caused by overlaps.

3. **Mask-level Label Transfer**: Combining the first two modules, pixels are sparsely sampled within each non-overlapping mask to perform label transfer, followed by a weighted majority vote (weighted by the cosine similarity of the match) within each mask to determine its overall label. This is more efficient and boundary-clear than pixel-wise transfer, and more flexible than using pure SAM masks.

4. **Mask-Consistency Aggregation Module**: Construct an undirected graph $G$, where each vertex corresponds to a 2D mask in a certain view, and edges are established between masks that project to the same 3D points across views. By detecting "under-segmented" masks (cases where masks from the same view in a correspondence set have different labels) and discarding their contributions, a final label is determined by a majority vote within the mask correspondence set. Intuition: if a part is incorrectly labeled in a few difficult views, multi-view majority voting can correct it; mask-level consistency ensures that points within the same mask eventually obtain the same label.

### Loss & Training
**Training-free method**—requires no training or fine-tuning. All majority voting operations use a truncation threshold of 0.6. SAM uses a $64\times64$ grid point sampling as prompts. DIFT features are extracted at the first upsampling block of the U-Net with timestep $t=261$ and an empty prompt input.

## Key Experimental Results

### Few-shot on PartNet-Ensembled (45 classes)

| Dataset | Metric | Ours (3-By-2) | PartSLIP++ | Gain |
|--------|------|------|----------|------|
| PartNetE (17 classes) | mIoU | 0.604 | 0.574 | +3.0% |
| PartNetE (28 classes) | mIoU | 0.665 | 0.642 | +2.3% |
| PartNetE (all 45 classes) | mIoU | 0.642 | 0.615 | +2.7% |

### Zero-shot on PartNetE (using PACO database, 18 classes)

| Dataset | Metric | Ours | PartSLIP | Gain |
|--------|------|------|----------|------|
| PartNetE-PACO overlap | mIoU | 0.430 | 0.341 | +8.9% |

### Few-shot on PartNet Level-3 (Fine-grained, 10 classes)

| Dataset | Metric | Ours | MvDeCor | PartSLIP |
|--------|------|------|---------|----------|
| PartNet L3 | mIoU | 0.281 | 0.277 | 0.125 |

### Ablation Study Key Points
- **Non-overlapping mask vs. Original SAM**: The non-overlapping module significantly improves performance on all classes (e.g., Bottle from 0.004 to 0.810, Suitcase from 0.285 to 0.813).
- **Mask-consistency vs. Point-consistency**: Mask-level consistency shows significant improvements on small parts (Printer 0.009 to 0.085, Clock 0.363 to 0.458).
- **Robustness to Multi-Class Database**: Expanding the database from 1 class to 8 classes only led to a slight drop in Kettle's average mIoU from 0.815 to 0.752, demonstrating the robustness of the method.
- **Cross-category Label Transfer**: Transferring the "wheel" label of Chair to Table improved the "wheel" mIoU from 0 to 0.600, while "leg" improved from 0.586 to 0.641.
- **Number of Database Instances**: Even with only 1 annotated instance, the mIoU reaches 0.809 (compared to 0.815 with 8 instances), far exceeding PartSLIP's 0.200.
- **Number of Views**: Performance stabilizes once the number of views exceeds 10.

## Highlights & Insights
- **Training-free & Language-free**: Relies solely on visual similarity for label transfer, avoiding ambiguities of linguistic part descriptions, which is particularly advantageous for fine-grained parts.
- **Coarse-to-Fine Correspondence Search**: Reduces the computational overhead of 2D label transfer by ~2000x, making high-resolution processing practically feasible.
- **Non-overlapping Mask Generation**: A simple and elegant design—overlaying masks in descending order of area to obtain visible parts, perfectly resolving the SAM multi-granularity overlap problem.
- **Cross-category Part Transfer Capability**: Demonstrates an interesting finding about the compositionality of object parts—the wheel label of a chair can be directly transferred to a table, suggesting that object parts form a finite set shareable across categories.
- **Works with a Single Instance**: The few-shot database requires only 1 annotated object to reach near-optimal performance.

## Limitations & Future Work
- **Dependency on SAM and DIFT Performance Bounds**: For parts without clear boundaries (e.g., animal/human parts), SAM's mask quality might degrade.
- **Requires Calibrated Cameras or Known Rendering Parameters**: Used for 3D back-projection. Applying this to real-world scans requires additional steps like Structure-from-Motion (SfM).
- **Noisy Database Annotations**: Even with accurate correspondence matching, incorrect labels in the database will still be propagated.
- **Potential Extensions**: Investigating applications in 3D scene semantic segmentation or object classification; exploring stronger correspondence models (e.g., SD3 features, combined DINOv2+SD features) to replace DIFT.

## Related Work & Insights
- **vs. PartSLIP/PartSLIP++**: PartSLIP relies on GLIP's text-driven part detection, which performs poorly on fine-grained parts (e.g., "back_frame_vertical_bar"). 3-By-2 is entirely vision-correspondence-based and outperforms PartSLIP by 15.6% on PartNet Level-3.
- **vs. MvDeCor**: MvDeCor requires pre-training on the target data distribution + category-level fine-tuning, as well as ground-truth depth and normal maps. 3-By-2 matches its performance (0.281 vs. 0.277) without any training.
- **vs. SAMPro3D+OpenMask3D**: Applying scene segmentation methods directly to part segmentation yielded poor results (mIoU 0.146 vs. 0.430), illustrating that 3D part segmentation requires specific designs.

### Inspirations & Connections
- **Semantic Correspondence using Diffusion Features**: This paradigm is highly valuable—not only for part segmentation but also extensible to other tasks requiring cross-instance semantic matching (e.g., human parsing, organ correspondence in medical images).
- **Cross-category Part Transfer**: The discovery of cross-category part transfer inspires a new direction: Is it possible to establish a unified "part vocabulary" shared across all object categories?

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to apply diffusion model semantic correspondence to 3D part segmentation, though the overall pipeline (multi-view 2D to 3D) is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on multiple datasets and settings (zero-shot/few-shot), with comprehensive ablation studies, cross-category experiments, and real scan demonstrations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich illustrations and clear methodological descriptions.
- **Value**: ⭐⭐⭐⭐ High practicality as a training-free method, and the finding on cross-category transfer is inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally](flashsplat_2d_to_3d_gaussian_splatting_segmentation_solved_optimally.md)
- [\[ECCV 2024\] Open-Vocabulary 3D Semantic Segmentation with Text-to-Image Diffusion Models](open-vocabulary_3d_semantic_segmentation_with_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Unlocking 3D Affordance Segmentation with 2D Semantic Knowledge](../../CVPR2026/3d_vision/unlocking_3d_affordance_segmentation_with_2d_semantic_knowledge.md)
- [\[ICLR 2026\] PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data](../../ICLR2026/3d_vision/partsam_a_scalable_promptable_part_segmentation_model_trained_on_native_3d_data.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)

</div>

<!-- RELATED:END -->
