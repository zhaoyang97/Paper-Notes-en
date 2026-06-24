---
title: >-
  [Paper Note] Robust 3D Shape Reconstruction in Zero-Shot from a Single Image in the Wild
description: >-
  [CVPR 2025][Segmentation][Single-view 3D reconstruction] ZeroShape-W proposes an occlusion-aware single-view 3D shape reconstruction model. It estimates the complete 3D shape (including the occluded parts) by jointly regressing the visible mask, occlusion mask, depth map, and camera intrinsics. Simultaneously, it designs a scalable synthetic data pipeline to simulate diverse foregrounds, occluders, and backgrounds. With only 194M parameters, it significantly outperforms state…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Single-view 3D reconstruction"
  - "zero-shot generalization"
  - "occlusion-aware"
  - "synthetic data"
  - "domain randomization"
date: 2026-05-08
content_hash: 1d7816aa4d5cb7f8
---

# Robust 3D Shape Reconstruction in Zero-Shot from a Single Image in the Wild

**Conference**: CVPR 2025  
**arXiv**: [2403.14539](https://arxiv.org/abs/2403.14539)  
**Code**: [https://ZeroShape-W.github.io](https://ZeroShape-W.github.io)  
**Area**: Image Segmentation  
**Keywords**: Single-view 3D reconstruction, zero-shot generalization, occlusion-aware, synthetic data, domain randomization

## TL;DR

ZeroShape-W proposes an occlusion-aware single-view 3D shape reconstruction model. It estimates the complete 3D shape (including the occluded parts) by jointly regressing the visible mask, occlusion mask, depth map, and camera intrinsics. Simultaneously, it designs a scalable synthetic data pipeline to simulate diverse foregrounds, occluders, and backgrounds. With only 194M parameters, it significantly outperforms state-of-the-art (SOTA) methods utilizing >1100M parameters on the Pix3D benchmark.

## Background & Motivation

1. **Background**: Single-view 3D shape reconstruction has developed rapidly in recent years. Large-scale 3D datasets (ShapeNet, Objaverse) have given rise to zero-shot reconstruction models with strong generalization capabilities (LRM, ZeroShape, etc.). These methods are categorized into two types: generation-based methods (such as Zero-1-to-3 diffusion multi-view synthesis) and regression-based methods (such as ZeroShape single forward pass inference).
2. **Limitations of Prior Work**: Existing SOTA methods assume that the inputs are clean, segmented object images (without background or occlusion), thus requiring pre-processing with segmentation models like SAM in real-world scenes. This introduces two issues: (a) errors from the segmentation model accumulate in the reconstruction; (b) segmentation removes the occluded parts of the object, leading to incomplete reconstructions.
3. **Key Challenge**: Object occlusion is common in real-world images (57.4% of images in Pix3D contain occlusions), yet existing methods assume fully visible objects. Addressing occlusions requires additional amodal completion models (such as pix2gestalt), which further increases error accumulation and parameter overhead.
4. **Goal**: To construct an end-to-end regression model that directly processes real-world images (containing backgrounds and occlusions) without requiring any external segmentation or completion models.
5. **Key Insight**: Unifying segmentation and reconstruction into a single regression framework, while explicitly modeling the contours of occluding objects to assist in reasoning about the occluded parts of the target object.
6. **Core Idea**: Jointly regressing the visible mask + occlusion mask + depth map + camera intrinsics to construct the visible 3D shape, and then using the visible shape and occlusion information via cross-attention to infer the complete 3D occupancy field (including occluded parts), alongside designing a synthetic data pipeline for domain-randomized training.

## Method

### Overall Architecture

ZeroShape-W takes an RGB image (224×224) containing the target object as input and outputs the complete 3D shape of the object (implicit occupancy field $\rightarrow$ Marching Cubes to extract the mesh). The pipeline is divided into two stages: **Pixel-Level Regression**—using a DPT-Hybrid backbone to extract global and fine-grained features from the image, separately regressing the camera intrinsics $K$, depth map $M_D$, visible mask $M_V$ (visible region of the object), and occlusion mask $M_O$ (location of the occluding object); then, the visible 3D shape $S_V$ is obtained through back-projection using $K$, $M_D$, and $M_V$. **3D Point-Level Regression**—concatenating and encoding $S_V$ and $M_O$ as key/value pairs, then predicting the occupancy value for any 3D query point through cross-attention layers to recover the complete 3D shape. A VLM can optionally be used to estimate the object category as a prior.

### Key Designs

1. **Occlusion-Aware Pixel-Level Regression**:
    - **Function**: Simultaneously regressing the visible region of the object, occluding object location, depth, and camera parameters from a single image.
    - **Mechanism**: Utilizing a DPT-Hybrid backbone to extract global features $X_G$ (used for camera intrinsics) and fine-grained features $X_F$ (used for depth/masks). Feature modulation is applied to the visible mask regression using the category prior provided by a CLIP text encoder: $\bar{X}_{F_{ij}} = (1+\gamma)X_{F_{ij}} + \beta$, where $\gamma, \beta$ are estimated from the CLIP text embedding via an FFN. The occlusion mask regression additionally utilizes the visible mask as input (concatenated along the channel dimension of $X_F$). The visible 3D shape is obtained through back-projection: $S_{V_{ij}} = \mathbb{1}_{\{M_{V_{ij}} \geq \eta\}} \cdot (M_{D_{ij}} K^{-1}[i,j,1]^T)$.
    - **Design Motivation**: Jointly regressing on the shared feature map $X_F$ allows the model to have extremely few parameters (194M vs. >1100M of competitors); CLIP text modulation assists in identifying the target object more accurately under complex backgrounds or occlusions.

2. **Occlusion-Aware 3D Point-Level Regression**:
    - **Function**: Inferring the complete 3D shape of the object (including occluded parts) from the visible 3D shape and occlusion information.
    - **Mechanism**: Concatenating and encoding $S_V$ (visible 3D shape) and $M_O$ (occlusion mask) into a sequence of $Z$-dimensional vectors to act as the key/value for the cross-attention. For each 3D query point, a query vector is constructed (by concatenating the point coordinate embedding and the CLIP text embedding), and is processed through $L=2$ layers of cross-attention followed by an FFN to predict the occupancy value. Cross-attention allows each query point to independently attend to its relevant spatial features.
    - **Design Motivation**: The occlusion mask informs the model about which regions are occluded. Complemented by the learned 3D shape priors, cross-attention enables the model to hallucinate the geometry of the occluded regions—a capability that has not been explored in previous regression-based methods.

3. **Scalable Synthetic Data Pipeline**:
    - **Function**: Generating large-scale, diverse training data without requiring real-world 3D annotations.
    - **Mechanism**: A multi-step pipeline—(a) **Object Rendering**: Rendering from 94K 3D objects in ShapeNet/Objaverse with various camera parameters yields over 1 million object images alongside precise 3D, depth, and mask annotations; (b) **Appearance Diversification**: Using ControlNet (conditioned spatially on the depth map, with "a [color][material][object]" as the text condition) to synthesize diverse object appearances. To prevent contour deformation, the rendered image is added with noise to serve as the initial guidance (inspired by SDEdit); (c) **Background Diversification**: Generating various scene backgrounds using an object-aware background outpainting model, ensuring that the object contour remains unaltered; (d) **Occlusion Augmentations**: Inserting synthetic foreground objects online as occluders via Copy-Paste during training.
    - **Design Motivation**: Real-world 3D annotations are extremely scarce; traditional rendering methods are limited by high-quality texture and environment resources; utilizing generative models can generate near-infinite appearance and background variations to achieve domain randomization.

### Loss & Training

- **Visible Mask and Occlusion Mask**: Binary cross-entropy loss
- **Occupancy Value**: Binary cross-entropy loss
- **Depth Map**: Scale-and-shift-invariant MAE loss, divided into two parts—GT depth for the visible region, and pseudo-depth from Depth Anything V2 for the global area (auxiliary depth loss)
- **Camera Intrinsics**: MSE loss (comparing the visible 3D shape with the GT)
- The auxiliary depth loss is a key design: while the synthetic pipeline only provides depth for the object region, the auxiliary loss fills in the depth supervision for the non-object regions, preventing overfitting.

## Key Experimental Results

### Main Results

**Pix3D Benchmark (Zero-Shot 3D Reconstruction):**

| Model | External Model | Params | FS@$\tau$↑ | FS@$2\tau$↑ | CD↓ |
|------|---------|--------|-------|--------|-----|
| LRM (w/ SAM) | SAM | >1100M | 31.0 | 54.5 | 0.121 |
| LRM (w/ SAM+pix2gestalt) | SAM+pix2gestalt | >2400M | 31.1 | 54.9 | - |
| ZeroShape (w/ SAM) | SAM | >800M | 32.1 | 56.8 | 0.116 |
| ZeroShape (w/ SAM+pix2gestalt) | SAM+pix2gestalt | >2100M | 33.6 | 59.0 | 0.110 |
| **ZeroShape-W (Ours)** | **None** | **194M** | **38.2** | **65.3** | **0.097** |

**Occluded / Non-Occluded Evaluation:**

| Model | Non-Occluded FS@$\tau$↑ | Non-Occluded CD↓ | Occluded FS@$\tau$↑ | Occluded CD↓ |
|------|-------------|-----------|-----------|---------|
| LRM (w/ SAM) | 33.5 | 0.111 | 29.1 | 0.128 |
| ZeroShape (w/ SAM) | 34.6 | 0.106 | 30.2 | 0.123 |
| **ZeroShape-W** | **43.6** | **0.082** | **34.2** | **0.107** |

### Ablation Study

| Configuration | FS@$\tau$↑ | FS@$5\tau$↑ | CD↓ | Description |
|------|-------|--------|-----|------|
| w/o Auxiliary Depth Loss | 35.9 | 90.7 | 0.105 | Inaccurate depth in non-object regions |
| w/o Occlusion Simulation | 37.7 | 91.2 | 0.102 | Inability to reconstruct occluded parts |
| w/o Text Prompts | 38.0 | 91.5 | 0.101 | Object recognition affected |
| Full (category-specific) | 39.6 | 92.8 | 0.094 | Optimal |
| Full (category-agnostic) | 38.2 | 92.5 | 0.097 | VLM Not Required |

### Key Findings

- **End-to-End Methods Far Outperform Pipeline-Based Methods**: ZeroShape-W outperforms the strongest baseline (ZeroShape+SAM+pix2gestalt) by 13.7% in FS@$\tau$ while requiring only 1/12th of its parameter count.
- **Auxiliary Depth Loss Contributes the Largest Single Improvement**: Removing it drops FS@$\tau$ from 39.6 to 35.9 (-3.7), indicating that global depth supervision is critical for generalization.
- **Occlusion Augmentations Exhibit Significant Effects in Occluded Scenes**: FS@$\tau$ increases by 4.0 (34.2 vs. ~30) on occluded images, though it also yields benefits in non-occluded scenarios.
- **Category Priors Help but Are Not Mandatory**: The category-specific configuration outperforms the category-agnostic one by 0.9 FS@$\tau$, demonstrating that the model performs well even without knowing the object category.

## Highlights & Insights

- **Unifying segmentation and reconstruction into end-to-end regression** is the most significant innovation—eliminating the error accumulation of external segmentation models while concurrently and dramatically reducing parameter counts. Utilizing the shared feature map $X_F$ for the joint estimation of depth, masks, and occlusions is a highly elegant design.
- **Using the occlusion mask as an additional information source for cross-attention** is extremely clever: it informs the model that "an object is blocking the target here," allowing the network to leverage learned shape priors to reason about the occluded parts. This concept is transferable to any 3D task requiring occlusion handling.
- **The initial guidance strategy of the synthetic data pipeline** (using the rendered image with added noise as the starting point for ControlNet) effectively resolves the issue of generative models distorting object contours, presenting a highly practical trick.

## Limitations & Future Work

- **Only Reconstructs Geometry, Lack of Textures**: Inability to output textured 3D models (such as LRM which can output NeRF).
- **Limited Input Resolution (224×224)**: May lose fine-grained details for small objects.
- **Domain Gap Between Synthetic and Real Data**: Although mitigated by domain randomization, it might still remain insufficient for certain specialized domains.
- **Single-Object Assumption**: Only handles a single salient object in the image, failing to perform scene-level reconstruction.
- **Future Directions**: Adding texture prediction branches; increasing input resolution; extending to multi-object scene-level reconstruction.

## Related Work & Insights

- **vs. LRM**: LRM is a Transformer-based large-model solution (>1100M parameters) that outputs textured NeRF, but requires SAM pre-processing and carries a huge parameter count; ZeroShape-W has only 1/6th of its parameter size yet accomplishes superior reconstruction accuracy.
- **vs. ZeroShape**: ZeroShape is the predecessor of ZeroShape-W, which is also a regression-based method but ignores occlusions and realistic backgrounds; ZeroShape-W addresses the in-the-wild deployment obstacles by introducing an occlusion mask branch and a synthetic data pipeline.
- **vs. Generative Methods (e.g., Zero-1-to-3 / Wonder3D)**: Generative methods leverage diffusion to synthesize multi-views, yielding high-quality results but incurring massive inference costs; ZeroShape-W operates via a single forward pass, which is higher in efficiency by an order of magnitude.
- **vs. Domain Randomization Methods**: Traditional domain randomization synthesizes random scenes, whereas this work synthesizes appearances and backgrounds meaningfully via generative models, aligning closer with real-world distributions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First occlusion-aware zero-shot regression-based 3D reconstruction method. The end-to-end design eliminates dependencies on external models.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Quantitative results on Pix3D + qualitative evaluations on three additional datasets + comprehensive ablations. Evaluating occluded vs. non-occluded instances separately is highly convincing.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly described methodology, high-quality illustrations, and an intuitive presentation of the synthetic data pipeline.
- **Value**: ⭐⭐⭐⭐⭐ Solves a key obstacle (occlusion and segmentation dependency) for deploying single-view 3D reconstruction in real-world environments. It is highly practical due to its compact parameter size and high performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ZIM: Zero-Shot Image Matting for Anything](../../ICCV2025/segmentation/zim_zero-shot_image_matting_for_anything.md)
- [\[ECCV 2024\] Efficient and Versatile Robust Fine-Tuning of Zero-shot Models](../../ECCV2024/segmentation/efficient_and_versatile_robust_fine-tuning_of_zero-shot_models.md)
- [\[NeurIPS 2025\] HumanCrafter: Synergizing Generalizable Human Reconstruction and Semantic 3D Segmentation](../../NeurIPS2025/segmentation/humancrafter_synergizing_generalizable_human_reconstruction_and_semantic_3d_segm.md)
- [\[CVPR 2025\] Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video](uni4d_unifying_visual_foundation_models_for_4d_modeling_from_a_single_video.md)
- [\[CVPR 2025\] MammAlps: A Multi-view Video Behavior Monitoring Dataset of Wild Mammals in the Swiss Alps](mammalps_a_multi-view_video_behavior_monitoring_dataset_of_wild_mammals_in_the_s.md)

</div>

<!-- RELATED:END -->
