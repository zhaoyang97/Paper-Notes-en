---
title: >-
  [Paper Note] Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes
description: >-
  [CVPR 2025][3D Vision][Mesh Saliency] This paper proposes Mesh Mamba, the first unified mesh saliency prediction model based on State Space Models (SSMs). By incorporating texture alignment, subgraph embedding, and bidirectional SSMs, it achieves high-quality visual attention prediction for both textured and non-textured 3D meshes. Additionally, it constructs the first dataset that systematically compares saliency differences under textured and non-textured conditions.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Mesh Saliency"
  - "State Space Model"
  - "Mamba"
  - "Texture Alignment"
  - "Subgraph Embedding"
date: 2026-05-08
content_hash: 48c45d9900d25df5
---

# Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes

**Conference**: CVPR 2025  
**arXiv**: [2504.01466](https://arxiv.org/abs/2504.01466)  
**Code**: [https://github.com/kaviezhang/MeshMamba](https://github.com/kaviezhang/MeshMamba)  
**Area**: 3D Vision  
**Keywords**: Mesh Saliency, State Space Model, Mamba, Texture Alignment, Subgraph Embedding

## TL;DR
This paper proposes Mesh Mamba, the first unified mesh saliency prediction model based on State Space Models (SSMs). By incorporating texture alignment, subgraph embedding, and bidirectional SSMs, it achieves high-quality visual attention prediction for both textured and non-textured 3D meshes. Additionally, it constructs the first dataset that systematically compares saliency differences under textured and non-textured conditions.

## Background & Motivation

1. **Background**: 3D mesh saliency prediction aims to identify regions on the mesh surface that attract visual attention. Existing methods mainly target non-textured meshes, predicting saliency based on geometric features (curvature, normals, etc.). Regarding datasets, although saliency data collected from VR eye-tracking experiments exist, systematic research under textured conditions is lacking.

2. **Limitations of Prior Work**: (1) Existing datasets either focus solely on non-textured meshes or only consider simple vertex colors, lacking a systematic study of the saliency differences of the same model under textured and non-textured conditions; (2) Existing methods cannot simultaneously process textured and non-textured meshes; (3) Point cloud and mesh methods struggle to model the global context.

3. **Key Challenge**: Both geometric structures and texture information jointly affect the visual attention distribution, but their interaction remains unclear, and there is a lack of a unified framework to simultaneously utilize both types of features.

4. **Goal** (1) Construct a comparative dataset; (2) Design a unified model to handle both types of meshes; (3) Effectively model local and global features.

5. **Key Insight**: Leveraging the linear complexity and global modeling capability of Mamba/SSM, combined with graph convolutions to process local topological relations, and preserving mesh topology through subgraph embedding.

6. **Core Idea**: Using subgraph embedding to preserve topology and bidirectional SSMs to realize global context modeling, thereby uniformly addressing saliency prediction for both textured and non-textured meshes.

## Method

### Overall Architecture
The input is a 3D triangular mesh (with an optional texture image), and the output is the saliency value of each triangular face. The model consists of three components: a graph convolutional encoder (texture alignment + geometric feature extraction + graph convolution) → subgraph embedding → Mamba Block (bidirectional SSM + feature diffusion & aggregation) → feature propagation (voting interpolation upsampling).

### Key Designs

1. **Texture Alignment Module**:

    - **Function**: Accurately maps 2D texture image features onto the 3D mesh surface.
    - **Mechanism**: Utilizing UV mapping to correspond triangular faces to pixel positions in the texture image, alignment is then achieved in the high-dimensional feature domain through an implicit representation (latent code map). For each triangular face, texture features within its UV range are uniformly sampled. Bilinear implicit interpolation is adopted to process sub-pixel locations to prevent feature discontinuities. Each triangular face adjusts its sampling range based on its aspect ratio to ensure an undistorted receptive field.
    - **Design Motivation**: Directly using vertex colors to represent texture information is too coarse and fails to capture the visual details of complex textures. Accurate texture-geometry alignment is achieved through continuous implicit representation.

2. **Subgraph Embedding**:

    - **Function**: Splits the mesh into topology-preserving local patches to serve as the input token sequence for the SSM.
    - **Mechanism**: First, Farthest Point Sampling (FPS) is used to select $L$ center faces. Then, Random Walk Sampling (RWS) is applied to each center face to gather a subgraph of length $M$, generating connectivity-preserving local patches. Unlike KNN clustering, RWS traverses along mesh edges, naturally preserving adjacency, and its randomness also enhances the model's robustness against noise.
    - **Design Motivation**: Traditional patch embedding (such as KNN clustering) destroys the topological connectivity of meshes. Unlike the regular grid structure of images, meshes cannot be simply cut into rectangular patches. Subgraph embedding preserves the inherent positional relationships and geometric integrity.

3. **Bidirectional SSM Mamba Block**:

    - **Function**: Performs global context modeling on the subgraph patch token sequence.
    - **Mechanism**: A learnable $[cls]$ token and positional encodings are introduced to extend the sequence, followed by expanding the receptive fields of local features via feature diffusion and aggregation operations—generating $l$ pseudo-adjacent faces for each token for feature propagation. Crucially, a bidirectional SSM is utilized: $z_t = SSM_+(f(z_{t-1})) + SSM_-(f(z_{t-1})) + f(z_{t-1})$, simultaneously considering both forward and backward sequence contexts to overcome the directional bias issues of unidirectional SSMs.
    - **Design Motivation**: Unidirectional SSMs can only process sequences in one direction, whereas patch tokens on meshes have no natural linear order. Bidirectional processing ensures that each token can fuse global context information, enhancing the understanding of the overall structure and content.

### Loss & Training
Trained using the L1 loss function with the AdamW optimizer. The initial learning rate is 1e-3, decaying by a factor of 0.1 every 50 epochs, for a total of 150 training epochs. The dataset is split into 80% for training and 20% for testing.

## Key Experimental Results

### Main Results

Non-textured mesh saliency prediction (geometry features only):

| Method | CC↑ | SIM↑ | KLD↓ | SE↓ |
|------|-----|------|------|-----|
| PointTrans | 0.5114 | 0.6861 | 0.3475 | 0.0314 |
| MeshNet | 0.5423 | 0.7000 | 0.3390 | 0.0309 |
| Mamba3D | 0.5993 | 0.7088 | 0.3345 | 0.0285 |
| **Ours** | **0.6140** | **0.7171** | **0.3067** | **0.0284** |

Textured mesh saliency prediction (geometry + texture):

| Method | CC↑ | SIM↑ | KLD↓ | SE↓ |
|------|-----|------|------|-----|
| PointTrans | 0.5201 | 0.6817 | 0.3578 | 0.0297 |
| MeshNet | 0.5605 | 0.7002 | 0.3371 | 0.0286 |
| Mamba3D | 0.5013 | 0.6769 | 0.3655 | 0.0372 |
| **Ours** | **0.6305** | **0.7232** | **0.2888** | **0.0265** |

The improvement is particularly significant under textured conditions: compared with Mamba3D, CC improved by 25.8% and KLD decreased by 21.0%.

### Ablation Study

| Configuration | CC↑ | SIM↑ | KLD↓ | SE↓ |
|------|-----|------|------|-----|
| Full model | 0.6305 | 0.7232 | 0.2888 | 0.0265 |
| w/o Texture | 0.6066 | 0.7113 | 0.3134 | 0.0267 |
| w/o Shape | 0.5403 | 0.6903 | 0.3634 | 0.0324 |
| w/o Subgraph (with KNN) | 0.6237 | 0.7208 | 0.3048 | 0.0298 |
| w/o Feature D&A | 0.5889 | 0.7106 | 0.3123 | 0.0294 |
| w/o SSM- | 0.6203 | 0.7186 | 0.2935 | 0.0272 |
| w/o SSM+ | 0.6199 | 0.7164 | 0.2947 | 0.0275 |
| w/ Backbone-T (Transformer) | 0.6113 | 0.7204 | 0.2975 | 0.0270 |

### Key Findings
- **Shape features contribute the most**: Removing the shape features of triangular faces (w/o Shape) causes a 14.3% drop in CC (0.6305→0.5403), representing the most significant impact among all ablations.
- **Texture is crucial for textured meshes**: Adding texture improves CC from 0.6066 to 0.6305; however, it is detrimental to non-textured meshes.
- **Subgraph embedding outperforms KNN**: Subgraph embedding performs 5.2% better in KLD than KNN clustering, validating the importance of preserving topology.
- **Bidirectional SSM components are both indispensable**: Removing either direction degrades performance, with the forward and backward SSMs contributing roughly equally.
- **Linear growth of computational complexity**: FLOPs grow linearly with the number of subgraphs and subgraph length, demonstrating good scalability.

## Highlights & Insights
- **Dataset Contribution**: The first dataset to systematically compare the saliency differences of the same 3D model under textured/non-textured conditions, utilizing VR eye-tracking experiments with 60 participants, presenting a solid methodology. This provides a valuable resource for studying the effects of texture on visual attention.
- **Subgraph Embedding Design**: Replacing KNN with Random Walks to segment mesh patches is a highly clever approach that naturally preserves topological information. This concept can be generalized to any task requiring patchification on non-Euclidean geometric data.
- **Texture-Geometry Interaction Finding**: Experiments reveal that texture information is unhelpful or even harmful for non-textured mesh saliency prediction, but significantly improves performance for textured meshes. This indicates that the model indeed learns conditionally dependent feature representations.

## Limitations & Future Work
- The dataset scale is limited, and the diversity of the Free3D asset library may not be sufficient to represent all scenarios.
- Only static mesh saliency is considered, without involving visual attention online in dynamic/interactive scenes.
- Subgraphs sampled by random walks might struggle to cover all critical regions of the mesh, and FPS might miss small areas with high detail.
- The paper does not discuss the performance and generalization capability of the model on meshes of different resolutions.
- The saliency-driven mesh simplification application (Sec 5.4) is only a preliminary demonstration and can be further explored.

## Related Work & Insights
- **vs Mamba3D**: Mamba3D directly applies SSMs to point clouds, whereas this work applies them to mesh subgraphs, preserving topological information and showing a distinct advantage on textured meshes (CC 0.6305 vs 0.5013).
- **vs SAL3D**: SAL3D is based on PointNet2 and only processes non-textured meshes. This work offers a unified framework handling both mesh types, with all-around superior performance.
- **vs DiffusionNet**: DiffusionNet processes meshes using Heat Kernel Signatures and diffusion operators but performs poorly on the saliency prediction task, likely due to a lack of global context modeling.

## Rating
- Novelty: ⭐⭐⭐⭐ First to apply Mamba/SSM to mesh saliency prediction; the subgraph embedding design is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual validation with a self-built dataset and public datasets, compared against 16 baselines, detailed ablation, and cross-validation design.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured, but some mathematical notation is quite dense.
- Value: ⭐⭐⭐⭐ Both the dataset and methodological contributions are substantial, directly driving research in 3D visual perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MVSMamba: Multi-View Stereo with State Space Model](../../NeurIPS2025/3d_vision/mvsmamba_multi-view_stereo_with_state_space_model.md)
- [\[ICCV 2025\] MeshMamba: State Space Models for Articulated 3D Mesh Generation and Reconstruction](../../ICCV2025/3d_vision/meshmamba_state_space_models_for_articulated_3d_mesh_generation_and_reconstructi.md)
- [\[CVPR 2025\] Textured Gaussians for Enhanced 3D Scene Appearance Modeling](textured_gaussians_for_enhanced_3d_scene_appearance_modeling.md)
- [\[ECCV 2024\] CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model](../../ECCV2024/3d_vision/crm_single_image_to_3d_textured_mesh_with_convolutional_reconstruction_model.md)
- [\[CVPR 2025\] Continuous 3D Perception Model with Persistent State](continuous_3d_perception_model_with_persistent_state.md)

</div>

<!-- RELATED:END -->
