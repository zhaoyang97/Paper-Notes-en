---
title: >-
  [Paper Note] Matrix3D: Large Photogrammetry Model All-in-One
description: >-
  [CVPR 2025][3D Vision][Photogrammetry] Matrix3D proposes a unified photogrammetry model based on a multi-modal diffusion Transformer. Through a masked learning strategy, it simultaneously performs pose estimation, depth prediction, and novel view synthesis within a single model. It achieves a pose estimation rotation accuracy of 96.5% on CO3D, significantly outperforming all specialized methods.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Photogrammetry"
  - "Multi-modal Diffusion"
  - "Pose Estimation"
  - "Depth Prediction"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: 72febf4fe1987a93
---

# Matrix3D: Large Photogrammetry Model All-in-One

**Conference**: CVPR 2025  
**arXiv**: [2502.07685](https://arxiv.org/abs/2502.07685)  
**Code**: [https://nju-3dv.github.io/projects/matrix3d](https://nju-3dv.github.io/projects/matrix3d) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Photogrammetry, Multi-modal Diffusion, Pose Estimation, Depth Prediction, Novel View Synthesis

## TL;DR
Matrix3D proposes a unified photogrammetry model based on a multi-modal diffusion Transformer. Through a masked learning strategy, it simultaneously performs pose estimation, depth prediction, and novel view synthesis within a single model. It achieves a pose estimation rotation accuracy of 96.5% on CO3D, significantly outperforming all specialized methods.

## Background & Motivation
**Background**: The traditional photogrammetry pipeline consists of multiple independent stages—feature detection, SfM, MVS, etc.—each employing different algorithms that cannot be jointly optimized. Additionally, reconstruction quality degrades severely under sparse-view inputs, typically requiring hundreds of images to achieve reliable results.

**Limitations of Prior Work**: First, traditional workflows require dense image collection, which is difficult to satisfy in practical applications. Second, different steps in multi-stage pipelines operate independently, leading to accumulated errors and suboptimal final results. Existing methods such as PF-LRM and DUSt3R attempt to perform pose estimation and reconstruction with a single feedforward model, but they still cannot simultaneously support pose estimation, depth, and novel view synthesis.

**Key Challenge**: The key challenge of unified multi-task modeling is incomplete training data—existing datasets often provide only partial modal annotations (such as only image-pose pairs or image-depth pairs) rather than a complete combination of all three modalities.

**Goal**: (1) Build a unified model capable of simultaneously supporting pose estimation, depth prediction, and novel view synthesis; (2) Solve the problem of multi-modal training under partially annotated data; (3) Achieve high-quality reconstruction under sparse-view inputs.

**Key Insight**: The authors observe that all modalities can be unified into 2D representations—Plücker ray maps for cameras, disparity maps for depth, and VAE latent codes for images—thereby leveraging the capabilities of image generation models. Borrowing the masked learning strategy from MAE, the model is trained by randomly masking inputs of different views/modalities, allowing the complete tri-modal model to be trained even when only bi-modal data is available.

**Core Idea**: Restructure several sub-tasks of photogrammetry into a "modality translation" problem using a multi-modal diffusion Transformer and masked learning, enabling full-modality training from incomplete data.

## Method

### Overall Architecture
The input to Matrix3D is a set of sparse-view images (up to 8 views), and the outputs can be any combination of target-view camera poses (ray maps), depth maps, or novel-view RGB images. The overall architecture adopts an encoder-decoder diffusion Transformer based on pretrained Hunyuan-DiT weights: the encoder processes known conditional information (known data of each view and modality), and the decoder denoises the noisy target modality. Finally, 3DGS optimization is integrated to generate complete 3D reconstructions.

### Key Designs

1. **Unified Multi-modal Representation and Encoding**:

    - **Function**: Unify images, camera poses, and depth into 2D representations processable by diffusion models.
    - **Mechanism**: RGB images are encoded into a low-dimensional latent space via SDXL's VAE; camera poses are represented by Plücker ray maps, where each pixel encodes a ray originating from the camera, naturally forming image-like 2D data; depth is converted into disparity maps (inverse depth) to bound the data range. All modalities undergo fixed shifting and scaling to align their distributions with standard Gaussian.
    - **Design Motivation**: The unified 2D representation allows three highly distinct modalities to share the same generative framework, eliminating the need to design specialized networks for each modality.

2. **View-Level Masked Learning Strategy**:

    - **Function**: Enable flexible input/output configurations in multi-view, multi-modal scenarios and allow training on partially annotated data.
    - **Mechanism**: Unlike MAE which masks patches within a single image, Matrix3D performs masking at the view and modality levels. During training, tasks are allocated in a ratio of NVS : pose estimation : depth prediction : fully random = 3:3:3:1, where masked views/modalities are filled with noise as denoising targets. During inference, different tasks are executed by selecting different "condition-target" combinations. In addition, conditioning is dropped with a 10% probability to support classifier-free guidance.
    - **Design Motivation**: Existing datasets often only contain bi-modal data (e.g., image-pose or image-depth). View-level masking enables the model to learn tri-modal generation capabilities from these incomplete source data, significantly increasing the volume of usable training data.

3. **Multi-View Encoder-Decoder DiT Architecture**:

    - **Function**: Achieve cross-view, cross-modal feature fusion and generation.
    - **Mechanism**: The encoder consists of multiple self-attention blocks that process the latent codes of all conditional data and project them to a shared latent space. The decoder also consists of self-attention blocks but incorporates additional cross-attention layers to integrate conditional features from the encoder. The latent codes of all views and modalities are concatenated in sequence and passed through the Transformer layers to capture cross-view correspondences. Three types of positional encodings are used to encode patch token coordinates (RoPE), view IDs, and modality IDs (sinusoidal positional encodings with different base frequencies).
    - **Design Motivation**: The pure self-attention mechanism enables each token to attend to information across all views and modalities, which is inherently suited for photogrammetry tasks that require multi-view consistency.

### Loss & Training
Training utilizes the v-prediction loss: $\mathcal{L} = \|D(E(\mathbf{x}_c), \mathbf{x}_{g,t}, t) - \mathbf{v}\|^2$, where $\mathbf{v} = \alpha_t \epsilon - \sigma_t \mathbf{x}_0$. Training is conducted in three stages: first training on 4 views at 256 resolution for 180K steps, then expanding to 8 views for 20K steps, and finally climbing to 512 resolution for another 20K steps. A mixture of six datasets—Objaverse, MVImgNet, CO3D-v2, RealEstate10k, Hypersim, and ARKitScenes—is used, covering both object-level and scene-level.

## Key Experimental Results

### Main Results

| Dataset / Task | Metric | Ours | Prev. SOTA | Gain |
|------------|------|----------|----------|------|
| CO3D Pose (8-view) | RRA@15° | 96.1% | 92.4% (RayDiffusion) | +3.7% |
| CO3D Pose (8-view) | CCA@0.1 | 87.8% | 81.9% (RayDiffusion) | +5.9% |
| GSO NVS | PSNR | 20.45 | 19.22 (SyncDreamer) | +1.23 |
| DTU Monocular Depth | AbsRel | 0.036 | 0.064 (Metric3D v2) | -43.8% |
| DTU Monocular Depth | δ₁ | 0.985 | 0.969 (Metric3D v2) | +1.6% |

### Ablation Study

| Configuration | RRA@15° (2-view) | CCA@0.1 (2-view) | Description |
|------|---------------|----------------|------|
| RGB Only | 95.6% | 100% | Only RGB conditioning |
| RGB + Depth | 95.8% | 100% | Added depth conditioning |

### Key Findings
- Pose estimation significantly outperforms RayDiffusion and DUSt3R across all viewpoints (2-8 views), with a particularly pronounced advantage in camera center accuracy.
- Monocular depth prediction performance is significantly superior to Metric3D v2 and Depth Anything v2, even though the model was fundamentally designed as a multi-view method.
- When extra depth conditions are provided, both pose estimation and NVS quality improve further, validating the efficacy of joint multi-modal inference.
- NVS supports novel view synthesis of arbitrary viewpoints without constraints from fixed view configurations, demonstrating far greater flexibility than methods like SyncDreamer and Wonder3D.

## Highlights & Insights
- **Elegant Unified Multi-Task Modeling**: Pose, depth, and NVS are all formulated as a "modality translation" problem. Task switching is realized entirely by changing the masking patterns, avoiding the tedious process of training separate models for each task. The brilliance lies in the mutual benefits tasks provide to each other.
- **Masking Strategy Tackling Data Incompleteness**: Drawing inspiration from MAE but expanding it to the view-by-modality level, this strategy permits bi-modal annotated data to participate in tri-modal training. This concept is transferable to any scenario requiring multi-modal data but suffering from incomplete annotations.
- **Multi-Turn Interactive 3D Creation**: Users can achieve finely controlled 3D content creation by sequentially providing conditions (e.g., first estimating poses, then generating novel views, and finally adding depth), which holds high value in real-world applications.

## Limitations & Future Work
- The model supports a maximum of 8 views, which is still insufficient for large-scale scenes (e.g., city-scale reconstruction), limited by the quadratic complexity of Transformers.
- It relies on a pretrained VAE for image encoding, allowing the reconstruction errors of the VAE to propagate to the final NVS quality.
- The paper does not report execution time and GPU memory consumption, implying inference efficiency could be a bottleneck in practical deployment.
- The 3DGS optimization phase requires additional handling of multi-view inconsistency, indicating that the multi-view images generated by diffusion still retain some degree of inconsistency.

## Related Work & Insights
- **vs DUSt3R**: DUSt3R is a discriminative method that recovers poses via predicting 3D point maps followed by PnP. Matrix3D is a generative method that directly generates ray maps via diffusion. Matrix3D offers higher accuracy but potentially demands greater computational resources.
- **vs RayDiffusion**: Both methods utilize ray representations and diffusion models for pose estimation. However, RayDiffusion supports only pose estimation, whereas Matrix3D unifiedly supports three tasks and acquires superior features through multi-modal training.
- **vs InstantMesh/Wonder3D**: These methods are confined to NVS and are typically constrained to fixed-view configurations. Matrix3D supports arbitrary viewpoints and can simultaneously output depth and poses.

## Rating
- Novelty: ⭐⭐⭐⭐ Extending masked learning to a multi-modal multi-view level to unify three tasks is a highly novel concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compares against multiple SOTAs across all three tasks, though efficiency analysis is missing.
- Writing Quality: ⭐⭐⭐⭐ The paper is logically structured, progressing steadily from problem definition to methodology design.
- Value: ⭐⭐⭐⭐⭐ The practical value of an all-in-one photogrammetry model is exceptionally high, representing the development trend of 3D vision towards unified models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] One Diffusion to Generate Them All](one_diffusion_to_generate_them_all.md)
- [\[ICCV 2025\] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation](../../ICCV2025/3d_vision/robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)
- [\[CVPR 2025\] LIM: Large Interpolator Model for Dynamic Reconstruction](lim_large_interpolator_model_for_dynamic_reconstruction.md)
- [\[CVPR 2025\] Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)](gaussian_splatting_for_efficient_satellite_image_photogrammetry.md)
- [\[CVPR 2025\] PartRM: Modeling Part-Level Dynamics with Large Cross-State Reconstruction Model](partrm_modeling_part-level_dynamics_with_large_cross-state_reconstruction_model.md)

</div>

<!-- RELATED:END -->
