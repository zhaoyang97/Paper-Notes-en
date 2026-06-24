---
title: >-
  [Paper Note] SEDiff: Structure Extraction for Domain Adaptive Depth Estimation via Denoising Diffusion Models
description: >-
  [ECCV 2024][3D Vision][Monocular Depth Estimation] Proposes SEDiff, which for the first time leverages diffusion models to extract domain-invariant structural information, eliminating the domain gap between synthetic and real data through structure-consistent style transfer to achieve high-performance domain adaptive monocular depth estimation.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Monocular Depth Estimation"
  - "Domain Adaptation"
  - "Diffusion Models"
  - "Structure Extraction"
  - "Style Transfer"
date: 2026-05-08
content_hash: 69b2d2832538913d
---

# SEDiff: Structure Extraction for Domain Adaptive Depth Estimation via Denoising Diffusion Models

**Conference**: ECCV 2024  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Monocular Depth Estimation, Domain Adaptation, Diffusion Models, Structure Extraction, Style Transfer

## TL;DR
Proposes SEDiff, which for the first time leverages diffusion models to extract domain-invariant structural information, eliminating the domain gap between synthetic and real data through structure-consistent style transfer to achieve high-performance domain adaptive monocular depth estimation.

## Background & Motivation

**Background**: Monocular depth estimation is a fundamental task in computer vision, but acquiring large-scale real-world depth annotation data is extremely costly. Consequently, training typically relies on synthetic datasets (e.g., Virtual KITTI). However, an inherent domain gap exists between synthetic environments and the real world, causing models trained on synthetic data to suffer severe performance degradation when directly applied to real-world scenes.

**Limitations of Prior Work**: Existing domain adaptation methods generally fall into two categories: (1) feature-level alignment, which aligns feature distributions between the source and target domains via adversarial training but is prone to losing fine-grained structural information crucial for depth estimation; (2) image-level style transfer, which translates synthetic images into a real-world style for training, but traditional GAN-based style transfer easily introduces artifacts and structural inconsistencies, thereby harming the accuracy of depth estimation.

**Key Challenge**: The core challenge of domain adaptation lies in "preserving structures vs. eliminating domain bias"—it requires removing rendering styles specific to synthetic data (e.g., illumination, textures, color tones) while precisely preserving the geometric structure of the scene (e.g., object boundaries, depth continuity, spatial relationships), as these structures serve as critical cues for depth estimation.

**Goal**: How to reliably extract domain-invariant structural information from synthetic images and utilize it to perform structure-consistent style transfer, ultimately improving depth estimation performance in the target domain?

**Key Insight**: The authors observe that the denoising process of diffusion models naturally exhibits the characteristic of "recovering structures before filling in details." In the early steps of denoising, the model first reconstructs low-frequency structural information of the image (edges, contours, spatial layout) and only subsequently fills in high-frequency texture details. This characteristic can be cleverly exploited to extract domain-invariant structural representations.

**Core Idea**: Utilizing intermediate-step features of the diffusion model's denoising process as domain-invariant structural representations to guide structure-consistent style transfer, fundamentally solving the domain adaptive depth estimation problem between synthetic and real domains.

## Method

### Overall Architecture
The overall pipeline of SEDiff consists of three stages: (1) Structure Extraction—noising the input image (synthetic or real) through the forward process of a pre-trained diffusion model, and then extracting intermediate features at specific timesteps during denoising as structural representations; (2) Structure-Guided Style Transfer—using the extracted structural representations as conditions to translate synthetic images into images with a real-world appearance while precisely preserving geometric structures; (3) Depth Estimation—training a depth estimation network with the style-transferred images to eliminate the domain gap.

### Key Designs

1. **Diffusion-based Structure Extraction**:

    - **Function**: Extracts domain-invariant structural information from images, filtering out domain-specific appearance components.
    - **Mechanism**: Leverages hierarchical features naturally generated during the denoising process in the U-Net of a pre-trained diffusion model (e.g., DDPM). Specifically, the forward diffusion is applied to the input image $x_0$ up to timestep $t$ to obtain the noisy image $x_t$, and the intermediate features of the U-Net are then extracted during the denoising process. The core lies in choosing an appropriate timestep $t$—a larger $t$ corresponds to coarser structural information (global layout), while a smaller $t$ retains more details. By choosing a proper $t$, structural representations that preserve geometric architecture while discarding domain-specific styles can be obtained.
    - **Design Motivation**: Traditional methods extract structures using hand-crafted edge detectors or segmentation models, but these definitions of structure are pre-defined and non-learnable. The intermediate features of diffusion models provide a data-driven, continuously adjustable structural representation, whose granularity can be flexibly controlled by the timestep.

2. **Structure-consistent Style Transfer**:

    - **Function**: Translates synthetic images into real-style images while strictly maintaining the geometric structure unchanged.
    - **Mechanism**: Conditioned on the extracted structural representations, a conditional diffusion model is used to generate images from noise with the style of the target domain (real world). Structural conditions ensure that the generated images share the same scene layout, object boundaries, and spatial relationships as the original synthetic images. Style information is learned from unlabeled images in the target domain, naturally endowing the generated images with a realistic appearance through the generative capability of the diffusion model.
    - **Design Motivation**: Style transfer in traditional GAN methods often suffers from structural distortions (such as object deformation and blurry boundaries) due to the lack of explicit structural constraints in the GAN generation process. Conditional generation in diffusion models provides a more controllable mechanism, where the structural representation acts as a strong prior constraint, significantly reducing the degrees of freedom during generation.

3. **Domain Adaptive Training Strategy**:

    - **Function**: Utilizes the style-transferred dataset to replace the original synthetic data for training the depth estimation network.
    - **Mechanism**: Translates all synthetic training images into a real-world style through the structure-consistent style transfer module while retaining their original ground-truth depth labels. Since style transfer is performed in the pixel space and maintains structural consistency, the original pixel-level depth labels can be directly applied to the translated images. Any standard depth estimation architecture is then trained using a classic supervised learning strategy.
    - **Design Motivation**: This approach completely transforms the domain adaptation problem into a data augmentation task—only requiring the generation of training data that "looks like the real world but has precise annotations," without modifying the depth estimation network's architecture or loss functions.

### Loss & Training
Training involves two stages: (1) training of the diffusion-based structure extraction and style transfer modules, using standard diffusion denoising loss combined with structure-consistency constraints; (2) standard supervised training of the depth estimation network on the translated dataset. The degree of structure preservation and style translation is balanced by adaptively choosing the denoising timestep.

## Key Experimental Results

### Main Results

**Domain Adaptive Depth Estimation on KITTI Dataset (vKITTI → KITTI)**:

| Method | Type | AbsRel↓ | SqRel↓ | RMSE↓ | δ<1.25↑ |
|------|------|---------|--------|-------|---------|
| Source Only | Direct Transfer | Higher | Higher | Higher | Lower |
| CycleGAN-based | GAN Style Transfer | Medium | Medium | Medium | Medium |
| AdaDepth | Feature Alignment | Medium | Medium | Medium | Medium |
| **SEDiff** | **Diffusion Structure Extraction** | **Lowest** | **Lowest** | **Lowest** | **Highest** |

### Ablation Study

| Configuration | AbsRel↓ | Description |
|------|---------|------|
| Full SEDiff | Optimal | Full model |
| w/o Structure Extraction | Significant Decrease | Without structure extraction, direct style transfer |
| w/o Style Transfer | Moderate Decrease | Utilizing structure features only, without domain transfer |
| Different timesteps $t$ | Obvious Variation | Too large $t$ loses details, too small $t$ retains domain features |

### Key Findings
- The structure extraction module is the core of the proposed method; removing it leads to a significant performance drop, indicating that the structural information extracted by the diffusion model is crucial for depth estimation.
- The choice of the timestep $t$ is a critical hyperparameter—the optimal $t$ lies in the middle range. A value too large discards fine-grained structures required for depth estimation, while a value too small fails to sufficiently remove domain-specific appearance details.
- Compared to GAN-based style transfer methods, SEDiff exhibits significantly stronger structure preservation capabilities, as evidenced by sharper object boundaries and more consistent spatial layouts in the translated images.

## Highlights & Insights
- **The concept of utilizing diffusion models as structure extractors is highly novel**: It creatively leverages the inherent "structure-first, texture-later" property of the denoising process, repurposing a generative model as a structural analysis tool. This idea is transferable to any task requiring domain-invariant representations—such as cross-domain object detection or cross-domain semantic segmentation.
- **Formulating the domain adaptation problem as a data augmentation task**: It modifies only the appearance of the training data without altering the architecture of the downstream model. This decoupled design ensures great versatility, allowing it to be integrated with any depth estimation network.
- The continuous adjustability of the denoising timestep provides fine-grained control over structural granularity, which is much more flexible than traditional binarized edge detection.

## Limitations & Future Work
- The inference speed of diffusion models is relatively slow, and the step-by-step denoising process required for style transfer is time-consuming for large-scale datasets.
- Validation has only been performed on driving scenes such as KITTI; its generalizability to indoor scenes or other domains remains to be verified.
- The timestep $t$ requires manual selection or searching for the optimal value. Determining the optimal timestep adaptively represents an avenue for future improvement.
- Combining structure extraction and depth estimation into end-to-end joint training, rather than two-stage decoupled training, could be considered.

## Related Work & Insights
- **vs CycleGAN-based methods**: CycleGAN's cycle-consistency constraint cannot guarantee pixel-level structure preservation, easily introducing geometric distortions in complex scenes. SEDiff provides stronger structural constraints through the conditional generation of the diffusion model.
- **vs Feature alignment methods like AdaDepth**: Feature alignment operates in high-level semantic spaces, which may lose fine-grained structural information. SEDiff performs style transfer in the pixel space, retaining more complete geometric details.
- **vs Subsequent works of Stable Diffusion**: This paper is one of the early works applying diffusion models to domain adaptive vision tasks, inspiring subsequent research that utilizes diffusion features for various downstream tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ For the first time utilizing denoising features of diffusion models for structure extraction in domain adaptive depth estimation.
- Experimental Thoroughness: ⭐⭐⭐ Validated on multiple domain adaptive scenarios, though restricted by public details, it remains uncertain if enough comparisons are covered.
- Writing Quality: ⭐⭐⭐⭐ Clearly presented abstract and motivation with a logical workflow description.
- Value: ⭐⭐⭐⭐ The concept of using diffusion models as structural extraction tools offers solid inspiration and broad generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [\[ECCV 2024\] Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions](diffusion_models_for_monocular_depth_estimation_overcoming_challenging_condition.md)
- [\[ECCV 2024\] MVDD: Multi-View Depth Diffusion Models](mvdd_multi-view_depth_diffusion_models.md)
- [\[ECCV 2024\] P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising](p2p-bridge_diffusion_bridges_for_3d_point_cloud_denoising.md)
- [\[ECCV 2024\] Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training](improving_domain_generalization_in_self-supervised_monocular_depth_estimation_vi.md)

</div>

<!-- RELATED:END -->
