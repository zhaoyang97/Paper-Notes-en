---
title: >-
  [Paper Note] "CADCrafter: Generating Computer-Aided Design Models from Unconstrained Images"
description: >-
  [CVPR2025][3D Vision][Paper Note] Academic paper note for "CADCrafter: Generating Computer-Aided Design Models from Unconstrained Images".
tags:
  - CVPR2025
  - 3D Vision
date: 2025-04-07
content_hash: ade2db5d9eb5b683
---
# CADCrafter: Generating Computer-Aided Design Models from Unconstrained Images

**Authors**: Cheng Chen, Xiaohui Zeng, Yuwei Li, Hippolyte Music, Tobias Ritschel, Sanja Fidler, Florian Shkurti  
**Affiliations**: NTU / A*STAR / UT Austin  
**Conference**: CVPR 2025  

## Background & Motivation

Generating editable CAD models from a single image is a long-standing goal in computer vision and computer graphics. Existing methods typically rely on parametric surface fitting or template-based retrieval, but these approaches face severe limitations:

**Discreetness of CAD programs**: CAD models consist of a sequence of discrete modeling operations (e.g., extrusion, chamfering, Boolean operations), which traditional continuous optimization methods struggle to handle directly.

**Huge semantic gap from image to CAD**: RGB images contain non-geometric information such as textures and illumination, whereas CAD programs focus purely on geometric operations, making the mapping between them extremely challenging.

**Data scarcity**: High-quality paired image-CAD data is extremely scarce; existing datasets (such as DeepCAD) contain only CAD programs without corresponding images.

**Hard-to-guarantee generation quality**: Generated CAD programs may contain syntax errors or geometric inconsistencies, preventing successful compilation by CAD compilers.

These issues motivate the authors to propose CADCrafter, an Image-to-CAD system based on a latent diffusion model. It bridges the semantic gap between images and CAD programs via a geometry encoder and ensures generation quality using Direct Preference Optimization (DPO) fine-tuning.

## Method

### Overall Architecture

CADCrafter adopts a two-stage training strategy:

1. **Stage 1**: Train the VAE and latent diffusion model for CAD sequences.
2. **Stage 2**: Introduce geometric conditions and DPO fine-tuning.

### Geometry Encoder

To bridge the semantic gap between RGB images and CAD programs, a multimodal geometry encoder is designed:

| Feature Type | Extraction Method | Function |
|----------|----------|------|
| Depth Map (Depth) | Pre-trained monocular depth estimation | Provides global 3D shape information |
| Normal Map (Normal) | Pre-trained normal estimation | Captures local surface orientation |
| DINO-v2 Semantic Features | Pre-trained DINO-v2 | Provides high-level semantic understanding |

The three types of features are integrated through a Feature Fusion Module and then injected into the cross-attention layers of the diffusion model as conditions.

### Latent Diffusion Model

A CAD program is represented as a sequence of commands $S = \{c_1, c_2, ..., c_N\}$, where each command $c_i$ contains an operation type and parameters. The detailed pipeline is as follows:

1. **VQ-VAE Encoding**: Encodes the CAD command sequence into a latent vector $z = E(S)$.
2. **Forward Diffusion**: $q(z_t | z_{t-1}) = \mathcal{N}(z_t; \sqrt{1-\beta_t} z_{t-1}, \beta_t I)$
3. **Reverse Denoising**: Conditional diffusion model $p_\theta(z_{t-1} | z_t, c_{geo})$, where $c_{geo}$ represents the geometric condition.
4. **Decoding**: $\hat{S} = D(\hat{z}_0)$

### Direct Preference Optimization (DPO)

One of the key innovations is using a CAD compiler as an automatic quality evaluator:

- **Positive samples**: Generated programs that can be successfully compiled by the CAD compiler.
- **Negative samples**: Generated programs that fail to compile or exhibit large geometric errors.
- **DPO Loss**: $\mathcal{L}_{DPO} = -\log \sigma(\beta (\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}))$
- Hyperparameter $\beta = 20$, controlling the strength of preference learning.

DPO fine-tuning significantly improves the compilation success rate and geometric accuracy of the generated programs.

### Multi-view to Single-view Distillation

Multi-view renderings are used as conditions during training (providing richer geometric information), while knowledge distillation is leveraged during inference so that the model can operate using only single-view inputs:

- Teacher model: Conditioned on multi-view geometric features.
- Student model: Conditioned on single-view geometric features.
- Distillation loss: Aligns the denoising predictions of both models in the latent space.

## Key Experimental Results

### DeepCAD Dataset

| Method | Acc_cmd ↑ | Acc_param ↑ | Med CD ↓ | Invalid ↓ |
|------|-----------|-------------|----------|-----------|
| DeepCAD | 78.41% | 72.15% | 0.082 | 12.3% |
| SkexGen | 80.56% | 74.33% | 0.067 | 9.8% |
| CADCrafter (Ours) | **83.23%** | **77.89%** | **0.049** | **4.2%** |

### Ablation Study

| Configuration | Acc_cmd | Med CD |
|------|---------|--------|
| w/o Geometry Encoder | 76.8% | 0.091 |
| Depth Only | 79.4% | 0.072 |
| Depth + Normal | 81.5% | 0.058 |
| All Features (D+N+DINO) | 82.1% | 0.053 |
| + DPO Fine-tuning | **83.23%** | **0.049** |

### RealCAD Dataset

The authors construct the RealCAD dataset to evaluate performance in real-world scenarios:
- It contains multi-angle captured images of 3D-printed physical models.
- CADCrafter demonstrates robust CAD reconstruction capabilities on real-world images as well.
- Qualitative results show that the generated CAD models maintain sound topological structures and editability.

## Highlights & Insights

1. **Geometric conditioning bridge**: Effectively bridges the semantic gap between images and CAD programs using a multimodal geometry encoder composed of Depth + Normal + DINO-v2.
2. **DPO fine-tuning strategy**: Innovatively utilizes a CAD compiler as an automatic preference labeling tool, improving generation quality through DPO.
3. **Multi-view distillation**: Leverages multi-view information during training while requiring only a single-view input during inference.
4. **RealCAD dataset**: The first Image-to-CAD evaluation dataset featuring 3D-printed physical objects.

## Limitations & Future Work

- Currently, only CAD models formed by extrusion operations are supported, while complex operations such as revolution and sweeping are not yet supported.
- For images with heavy occlusions or complex textures, depth and normal estimations may be inaccurate.
- DPO fine-tuning relies on binary feedback (success/failure) from the CAD compiler, lacking fine-grained quality evaluation.

## Related Work & Insights

- DeepCAD: Pioneering work in Transformer-based CAD sequence generation.
- SkexGen: CAD generation based on sketch-extrusion separation.
- Point2CAD: CAD reconstruction from point clouds.
- Alignment applications of DPO in LLMs/diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] 3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting](3dgut_enabling_distorted_cameras_and_secondary_rays_in_gaussian_splatting.md)
- [\[ICCV 2025\] NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement](../../ICCV2025/3d_vision/neuraleaf_neural_parametric_leaf_models_with_shape_and_deformation_disentangleme.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](../../ICCV2025/3d_vision/repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images](../../ICCV2025/3d_vision/auto-regressively_generating_multi-view_consistent_images.md)
- [\[ICCV 2025\] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](../../ICCV2025/3d_vision/one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)

</div>

<!-- RELATED:END -->
