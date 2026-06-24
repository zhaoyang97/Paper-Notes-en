---
title: >-
  [Paper Note] Dataset Enhancement with Instance-Level Augmentations
description: >-
  [ECCV 2024][Segmentation][Data Augmentation] An instance-level data augmentation method based on pretrained diffusion models is proposed. By repainting object instances individually in the image while maintaining the original labels, the method significantly enhances the performance of salient object detection, semantic segmentation, and object detection, while also supporting data anonymization.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Data Augmentation"
  - "Instance-Level Augmentation"
  - "Diffusion Model"
  - "Image Inpainting"
  - "Data Privacy"
date: 2026-05-08
content_hash: 9dafd820df983b4d
---

# Dataset Enhancement with Instance-Level Augmentations

**Conference**: ECCV 2024  
**arXiv**: [2406.08249](https://arxiv.org/abs/2406.08249)  
**Code**: [https://github.com/kupynorest/instance-augmentation](https://github.com/kupynorest/instance-augmentation)  
**Area**: Image Segmentation  
**Keywords**: Data Augmentation, Instance-Level Augmentation, Diffusion Model, Image Inpainting, Data Privacy

## TL;DR
An instance-level data augmentation method based on pretrained diffusion models is proposed. By repainting object instances individually in the image while maintaining the original labels, the method significantly enhances the performance of salient object detection, semantic segmentation, and object detection, while also supporting data anonymization.

## Background & Motivation

**Background**: Deep learning increasingly relies on large-scale annotated datasets. However, collecting and annotating large-scale, high-quality datasets is computationally expensive and raises privacy and ethical concerns. Many small, classic datasets (such as DUTS, Pascal VOC) have been gradually marginalized due to scale limitations.

**Limitations of Prior Work**:
1. Traditional data augmentations (flips, crops, color jitters) provide limited visual diversity, fundamentally remaining simple pixel-level transformations.
2. Image mixing techniques (CutMix, MixUp, etc.) often introduce visual artifacts, leading models to learn shortcut solutions.
3. Generating samples by training generative models on target datasets fails to introduce out-of-distribution (OOD) novel information.
4. Full-image generation methods require generating corresponding annotations simultaneously, making the verification of labeling quality difficult.

**Key Challenge**: How to introduce out-of-distribution visual diversity using generative models while preserving exact original annotations?

**Key Insight**: Instead of repainting the entire image, the repainting is performed at the instance level. By leveraging large-scale pretrained diffusion models to inpaint object instances one by one, the original segmentation masks, bounding boxes, and category labels remain unchanged.

**Core Idea**: Utilize pretrained latent diffusion models (LDM) for instance-level image inpainting, combined with depth and edge maps as ControlNet conditioning. This ensures that the shape and location of the repainted objects are consistent with the original annotations, achieving label-preserving, appearance-diverse data augmentation.

## Method

### Overall Architecture
Given an image $I$ and an annotation set $\mathcal{Y} = \{(M_i, c_i)\}_{1 \le i \le N}$ (binary mask and class for each instance), the method repaints each instance sequentially via a conditional LDM to generate a new image $I^*$, where the mapping $F(I, \mathcal{Y}) = I^*$ preserves the annotations. The pipeline consists of: (1) estimating the depth map and edge map of the entire image; (2) sorting instances by depth (from far to near); (3) inpainting each instance individually with a conditional inpainting model; and (4) combining the final image through Alpha blending.

### Key Designs

1. **Independent Repainting and Alpha Blending**: To avoid quality degradation (such as progressive drops in PSNR and SSIM) caused by repeated encoding/decoding through the LDM autoencoder, the inpainting for each instance is performed independently starting from the original image:
    $I_i^* = I_{i-1}^* \odot (\mathbb{1} - M_i) + G(I, M_i, T_i) \odot M_i$
   Thus, only $N$ inpainting operations are needed to generate $2^N$ combinations of original/repainted variants. Sorting by depth map (drawing distant objects first) ensures correct occlusion relationships.

2. **ControlNet Condition Control (Better Inpainting)**: Vanilla inpainting models cannot guarantee that (a) the generated object categories are correct, (b) the shapes conform to the original masks, and (c) small objects are not erased. Hence, ControlNet is introduced to inject depth maps (estimated by DepthAnything) and edge maps (extracted by HED) as auxiliary conditions. This significantly reduces instances where the generated objects deviate from original boundaries. The depth map preserves overall scene structure, while the edge map retains detailed boundaries.

3. **Prompt Engineering for Enhanced Diversity**:

    - **Object Description Extension**: Use synset lemmas from WordNet to expand category names, stabilizing CLIP text encoding.
    - **Color and Illumination Randomization**: Randomly sample color and lighting conditions and add them to the prompts for color-sensitive classes (e.g., cars, backpacks) to increase appearance diversity.
    - **Person Action Description**: For the "person" class, use BLIP-VQA to predict actions and incorporate them into the prompt to handle person repainting in complex scenarios.

4. **Mask Refinement**: For salient object detection datasets like DUTS, TRACER-7 is used to re-estimate tight saliency masks on the cropped region of each repainted object, eliminating potential mismatches between the generated objects and the original annotations.

### Loss & Training
This method is a data augmentation pipeline rather than end-to-end training. During training, each instance has a 30% probability of being replaced with its repainted version at each iteration to ensure training diversity. Up to 3 synthetic replacement variants are generated for each object. Strict filtering of NSFW content is also applied.

## Key Experimental Results

### Main Results
**Object Detection (COCO)**:

| Model | Data | AP | AP50 | AP75 |
|------|------|-----|------|------|
| Deformable-DETR | Original | 39.3 | 60.0 | 42.0 |
| Deformable-DETR | + Ours | **40.5** | 60.2 | **43.4** |
| RT-DETR | Original | 51.4 | 69.6 | 55.4 |
| RT-DETR | + Ours | **52.4** | 69.7 | **56.5** |
| YOLOv5m | Original | 44.1 | 63.4 | 47.8 |
| YOLOv5m | + Ours | **45.7** | 64.0 | **49.7** |

**Salient Object Detection (DUTS training)**:

| Model | Data | ECSSD F_max | DUTS-TE F_max | HKU-IS F_max |
|------|------|-------------|---------------|--------------|
| U2Net | Original | 0.944 | 0.863 | 0.930 |
| U2Net | + Ours | **0.948** | **0.874** | **0.935** |
| TRACER-4 | Original | 0.956 | 0.911 | 0.944 |
| TRACER-4 | + Ours | **0.960** | **0.918** | **0.948** |

Among 36 evaluations, 34 show improvements.

### Ablation Study

| Configuration | F_max↑ | MAE↓ | S_m↑ | F_avg↑ |
|------|--------|------|------|--------|
| Full Method | 0.892 | 0.033 | 0.885 | 0.853 |
| w/o Instance-level (Full-image Inpainting) | 0.889 | 0.035 | 0.881 | 0.848 |
| w/o Edge + Depth Control | 0.886 | 0.036 | 0.881 | 0.846 |
| w/o Edge Control | 0.890 | 0.034 | 0.880 | 0.850 |
| w/o Prompt Engineering | 0.889 | 0.034 | 0.880 | 0.849 |
| w/o Mask Refinement | 0.888 | 0.035 | 0.879 | 0.844 |

### Key Findings
- **Data Scarcity Scenarios**: When training on only 10% to 100% of the COCO training set, the augmentation consistently yields a +2-3 AP improvement.
- **Anonymization without Performance Loss**: Replacing all real people with synthetic counterparts leads to only a 1-point drop in overall AP (with 30.5% of instances repainted), while the face re-identification rate drops to 0.14%.
- **Complementary to Traditional Augmentation**: This method is complementary to traditional augmentations (flip, rotate, blur, etc.), resulting in even better performance when used jointly.
- **Semantic Segmentation**: On Pascal VOC, Mask2Former under synthesis + fine-tuning achieves 78.2 mIoU, significantly outperforming Diffumask's 77.6.

## Highlights & Insights
- **Conceptual Elegance**: The idea of instance-level repainting to preserve annotations while introducing out-of-distribution knowledge is simple yet effective, and generalized to multiple tasks like detection, segmentation, and saliency.
- **Exponential Variants**: An image with $N$ objects can yield $2^N$ original/synthetic combination variants, making the data scaling highly efficient.
- **Added Value of Privacy Protection**: The same pipeline naturally supports data anonymization without hurting the performance of downstream tasks.
- **Independent Repainting Strategy** cleverly avoids noise accumulation issues caused by repeated encoding/decoding in latent space.

## Limitations & Future Work
- The performance upper bound is constrained by the capabilities of the inpainting model and ControlNet; small or occluded objects are sometimes directly erased.
- It is dependent on the domain of the text-to-image diffusion model (typically real-world scenes), making it difficult to directly generalize to specialized domains like satellite or medical imaging.
- The depth and edge controls of ControlNet are occasionally inaccurate in complex scenes, which can lead to misalignments between annotations and the image.
- Future work can replace the generative model with stronger ones (e.g., SDXL) to further improve quality.

## Related Work & Insights
- ControlNet and T2I-Adapter introduce spatial control into diffusion models $\rightarrow$ This work utilizes this to maintain label consistency.
- Diffumask and DatasetDM generate both images and annotations simultaneously $\rightarrow$ This work highlights that preserving real annotations is more reliable.
- Copy-Paste augmentation only performs simple pasting $\rightarrow$ This work achieves semantically consistent instance replacement through repainting.
- Insight: This approach can be integrated with active learning to selectively enhance training samples of hard or rare categories.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of instance-level repainting for augmentation is novel, and the approach of preserving labels while introducing out-of-distribution knowledge is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across three tasks (detection, semantic segmentation, saliency), six datasets, and multiple models. Anonymization and generalization experiments are also included.
- Writing Quality: ⭐⭐⭐⭐ Clear logic; natural transitions from motivation to methodology and experiments.
- Value: ⭐⭐⭐⭐ Highly practical as a general-purpose data augmentation tool; release of datasets and code further adds to its real-world utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement](visage_video_instance_segmentation_with_appearance-guided_enhancement.md)
- [\[CVPR 2026\] DIMOS: Disentangling Instance-level Moving Object Segmentation](../../CVPR2026/segmentation/dimos_disentangling_instance-level_moving_object_segmentation.md)
- [\[ECCV 2024\] Part2Object: Hierarchical Unsupervised 3D Instance Segmentation](part2object_hierarchical_unsupervised_3d_instance_segmentation.md)
- [\[ECCV 2024\] ProMerge: Prompt and Merge for Unsupervised Instance Segmentation](promerge_prompt_and_merge_for_unsupervised_instance_segmentation.md)
- [\[ECCV 2024\] UniFS: Universal Few-Shot Instance Perception with Point Representations](unifs_universal_few-shot_instance_perception_with_point_representations.md)

</div>

<!-- RELATED:END -->
