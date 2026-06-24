---
title: >-
  [Paper Note] ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping
description: >-
  [CVPR 2025][Robotics][hand-object interaction] This paper proposes a Multi-Layer Occlusion (MLO) representation to learn 3D hand-object occlusion relationships and integrates the large-scale Objaverse 3D object dataset into training, achieving the first hand-object manipulation video generation framework that supports both dexterous bimanual manipulation and generalizable object appearances.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "hand-object interaction"
  - "video generation"
  - "multi-layer occlusion"
  - "diffusion model"
  - "Objaverse"
date: 2026-05-08
content_hash: 7ffdffee1783a798
---

# ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping

**Conference**: CVPR 2025  
**arXiv**: [2412.16212](https://arxiv.org/abs/2412.16212)  
**Code**: To be confirmed  
**Area**: 3D Vision  
**Keywords**: hand-object interaction, video generation, multi-layer occlusion, diffusion model, Objaverse

## TL;DR

This paper proposes a Multi-Layer Occlusion (MLO) representation to learn 3D hand-object occlusion relationships and integrates the large-scale Objaverse 3D object dataset into training, achieving the first hand-object manipulation video generation framework that supports both dexterous bimanual manipulation and generalizable object appearances.

## Background & Motivation

**Background**: Hand-object interaction (HOI) image generation based on diffusion models has achieved preliminary success, where methods typically utilize 2D conditional signals such as depth maps, normal maps, and hand skeletons to guide generation.

**Limitations of Prior Work**:
1. **Insufficient occlusion modeling**: Existing methods rely solely on 2D conditional signals, failing to handle self-occlusion among fingers and mutual occlusion between the hand and object, which leads to penetration artifacts and structural errors in generation results.
2. **Poor object generalization**: HOI video datasets usually contain only a dozen object categories, making it difficult for trained models to generalize to diverse unseen objects.
3. **Video temporal consistency**: Extending from image to video requires additional temporal consistency mechanisms.

**Key Challenge**: The hand has extremely high degrees of freedom and fine joint structures. Resolving 3D occlusion relationships from 2D conditional signals is an ill-posed problem; at the same time, HOI video data is extremely scarce.

**Key Insight**: Designing a 3D-aware MLO representation to replace 2D conditional signals, while introducing Objaverse to address data scarcity, and unifying the learning through a multi-dataset joint training strategy.

## Method

### Overall Architecture

ManiVideo is based on a conditional diffusion model, where inputs are bimanual MANO model parameters $(θ,β)$ and the motion sequence of the 3D object, and the output is a temporally consistent hand-object manipulation video. The pipeline includes:
1. Converting the raw hand-object signals into MLO representations (occlusion-free normal maps and occlusion confidence maps) and object representations (appearance and geometry).
2. Embedding the MLO representations into the UNet in two ways (initial noise and Transformer block).
3. Injecting the object representations into the UNet via AppearanceNet and geometric embeddings.
4. Two-stage training: image stage $\rightarrow$ temporal stage.

### Key Designs

**1. Multi-Layer Occlusion (MLO) Representation**
- **Function**: Decomposing the 3D hand-object model into multiple independent layers (object, palm, thumb, index, middle, ring, pinky), and rendering occlusion-free normal maps $H$ for each layer independently to compensate for occluded hidden areas.
- **Mechanism**: Inspired by Multi-Plane Images (MPI), a multi-layer 3D structure is constructed from back to front. Concurrently, an occlusion confidence map $D$ (based on depth maps) is introduced, where darker regions represent more severe occlusions. The model utilizes $D$ to distinguish visible/hidden areas and uses $H$ to complete the occluded parts.
- **Design Motivation**: 2D conditional signals (depth maps, masks, etc.) are ill-posed inputs that fail to represent occlusion relationships when fingers are densely arranged. MLO provides a complete 3D perspective, enabling the model to perceive the geometry of occluded fingers.

**2. Dual Embedding of MLO**
- **Function**: Injecting the MLO structure into the UNet in two ways: (a) extracting features of $H$ via a Pose Guider (4-layer convolution) and adding them to the initial noise $z_t' = z_t + G([H])$; (b) concatenating $H$ and $D$, extracting an embedding $E_F$ via convolution and an MLP, and injecting it into the Transformer blocks through cross-attention.
- **Mechanism**: The initial layer learns coarse-grained spatial correspondence, while the deep Transformer blocks perceive complex occlusion relations.
- **Design Motivation**: Ablation studies show that using only a single embedding method (w/o MLO*) leads to penetration artifacts, whereas dual embedding utilizes MLO information in a complementary manner.

**3. Object Representation and Objaverse Integration**
- **Function**: Rendering 6-view appearance images $O_I$ for each Objaverse object, which are combined with a background reference image $O_B$ and injected into the UNet via AppearanceNet $R$. Concurrently, rendering the object normal map $H_o$ and sampling a point cloud $P \in \mathbb{R}^{2048 \times 3}$ to extract a geometric embedding $E_N$, which is injected via cross-attention.
- **Mechanism**: Leveraging the scale advantage of Objaverse's 800K+ 3D models, rotation $Q$ and translation $L$ motion trajectories are randomly generated to simulate object movement, compensating for the scarcity of HOI video data.
- **Design Motivation**: HOI datasets contain extremely few object categories (~15 classes). Training solely on HOI data would cause the model to overfit to the texture dynamics of specific objects, whereas Objaverse provides rich object appearance and geometric diversity.

### Loss & Training

- **Two-stage training**: The image stage freezes temporal layers and trains for approximately 20K iterations, while the temporal stage freezes image layers, adds temporal layers, and trains for approximately 30K iterations.
- **Multi-dataset mixture**: In each iteration, data is sampled in equal proportions from Objaverse data, HOI video data, and human body data. Since Objaverse data lacks hands, the hand-related layers in the MLO are zeroed out; since human body data lacks objects, all object conditions are zeroed out.
- **Learning rate**: $1 \times 10^{-5}$ for the image stage, $8 \times 10^{-6}$ for the temporal stage, using the Adam optimizer.
- **Human extension**: Optionally extracting human skeletons $S$ and injecting them into the UNet via an additional Pose Guider $G_1$ to support human-centric HOI video generation.

## Key Experimental Results

### Main Results

| Method | DexYCB FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ | MPJPE↓ |
|---|---|---|---|---|---|
| HOGAN | 64.74 | 0.102 | 29.50 | 0.896 | 60.95 |
| ADiff | 53.95 | 0.093 | 29.96 | 0.903 | 59.12 |
| CDiff | 84.74 | 0.127 | 28.27 | 0.835 | 68.01 |
| **ManiVideo** | **49.96** | **0.079** | **30.10** | **0.913** | **57.30** |

Self-collected dataset:

| Method | FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ | MPJPE↓ |
|---|---|---|---|---|---|
| ADiff | 39.91 | 0.127 | 29.17 | 0.898 | 37.45 |
| CDiff | 45.50 | 0.133 | 28.33 | 0.883 | 42.89 |
| **ManiVideo** | **37.70** | **0.113** | **29.59** | **0.905** | **32.89** |

### Ablation Study

| Configuration | FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ | MPJPE↓ |
|---|---|---|---|---|---|
| w/o Objaverse | 61.60 | 0.121 | 27.99 | 0.895 | 37.33 |
| w/o MLO | 46.67 | 0.115 | 28.26 | 0.869 | 39.41 |
| w/o MLO* (Initial noise only) | 40.60 | 0.117 | 28.30 | 0.881 | 34.02 |
| **Full Model** | **37.70** | **0.113** | **29.59** | **0.905** | **32.89** |

### Key Findings

1. **MLO is crucial for occlusion modeling**: Removing MLO degrades the FID from 37.70 to 46.67 and the SSIM from 0.905 to 0.869, with particularly significant performance gaps observed in scenarios featuring densely arranged fingers and invisible finger bending.
2. **Complementary dual embedding**: Although using only the initial noise embedding (w/o MLO*) outperforms completely removing MLO, penetration issues still occur, indicating that deep occlusion relationship perception from Transformer blocks is indispensable.
3. **Objaverse enhances object generalization**: Without Objaverse, FID exhibits the most severe degradation (61.60), as the model overfits to the object textures in the training set.

## Highlights & Insights

- **First framework**: Achieves the first HOI video generation framework supporting both dexterous bimanual manipulation and generalizable objects simultaneously.
- **3D-aware conditional design**: The MLO representation reformulates the ill-posed 2D conditional signal problem into complete 3D modeling, which is novel and effective.
- **Unified multi-dataset training**: Ingeniously unifies Objaverse (objects), HOI videos (interactions), and human body data within the same framework via conditional zero-out.
- **Value**: Supports human-centric HOI video generation, which is directly applicable to digital humans and VR scenarios.

## Limitations & Future Work

- Performance is constrained by the accuracy of the driving signals (dependent on MANO fitting quality).
- Generalization to complex object textures is still affected by the synthetic-to-real domain gap.
- 4D representations (spatiotemporally consistent object appearance modeling) could be explored to further improve texture consistency.
- The scale of current HOI training data remains limited (722 videos, 15 objects); scaling up the dataset is expected to yield further improvements.

## Related Work & Insights

- **HOGAN**: A GAN-based method that achieves HOI image editing using flow warping, but optical flow fails to capture 3D occlusions.
- **Affordance Diffusion**: Generates HOI images given an object reference image, but only utilizes 2D mask conditions.
- **Animate Anyone**: ManiVideo's architectural design references its AppearanceNet + UNet framework.
- **Objaverse**: A large-scale dataset of 800K+ 3D models; this work is the first to systematically integrate it into HOI video generation training.

## Rating ⭐⭐⭐⭐

**Novelty**: ⭐⭐⭐⭐ MLO representation and unified multi-dataset training are both novel designs.  
**Experimental Thoroughness**: ⭐⭐⭐⭐ Comparison across two datasets and comprehensive ablation studies.  
**Writing Quality**: ⭐⭐⭐⭐ Clear structure and intuitive illustrations.  
**Value**: ⭐⭐⭐⭐ Direct application prospects in digital humans and VR fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AffordGen: Generating Diverse Demonstrations for Generalizable Object Manipulation with Affordance Correspondence](../../CVPR2026/robotics/affordgen_generating_diverse_demonstrations_for_generalizable_object_manipulatio.md)
- [\[CVPR 2025\] DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness](dexgrasp_anything_towards_universal_robotic_dexterous_grasping_with_physics_awar.md)
- [\[CVPR 2025\] ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning](maniptrans_efficient_dexterous_bimanual_manipulation_transfer_via_residual_learn.md)
- [\[CVPR 2025\] GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities](gigahands_a_massive_annotated_dataset_of_bimanual_hand_activities.md)
- [\[CVPR 2025\] 3D-MVP: 3D Multiview Pretraining for Robotic Manipulation](3d-mvp_3d_multiview_pretraining_for_manipulation.md)

</div>

<!-- RELATED:END -->
