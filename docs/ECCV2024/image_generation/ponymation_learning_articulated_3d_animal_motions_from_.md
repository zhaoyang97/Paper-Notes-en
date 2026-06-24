---
title: >-
  [Paper Note] Ponymation: Learning Articulated 3D Animal Motions from Unlabeled Online Videos
description: >-
  [ECCV 2024][Image Generation][3D animal motion] This paper proposes a method for learning articulated 3D animal motion generative models from unlabeled internet videos. By decomposing videos into static shape, appearance, and motion latent codes via a video photo-geometric autoencoding framework, the method enables the generation of diverse 4D animations from a single image during inference without requiring any pose annotations or parametric shape models.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "3D animal motion"
  - "unlabeled learning"
  - "video autoencoding"
  - "motion VAE"
  - "differentiable rendering"
date: 2026-05-08
content_hash: ebf6d9dff461a2d3
---

# Ponymation: Learning Articulated 3D Animal Motions from Unlabeled Online Videos

**Conference**: ECCV 2024  
**arXiv**: [2312.13604](https://arxiv.org/abs/2312.13604)  
**Code**: [https://keqiangsun.github.io/projects/ponymation](https://keqiangsun.github.io/projects/ponymation)  
**Area**: Image Generation  
**Keywords**: 3D animal motion, unlabeled learning, video autoencoding, motion VAE, differentiable rendering

## TL;DR
This paper proposes a method for learning articulated 3D animal motion generative models from unlabeled internet videos. By decomposing videos into static shape, appearance, and motion latent codes via a video photo-geometric autoencoding framework, the method enables the generation of diverse 4D animations from a single image during inference without requiring any pose annotations or parametric shape models.

## Background & Motivation

**Background**: 3D human motion synthesis has made significant progress with parametric models like SMPL and large-scale MoCap data. However, 3D modeling of animal motion severely suffers from a lack of data—lacking large-scale 3D scans, parametric shape models, or motion capture data. Existing methods either require multi-view videos or 2D keypoint annotations.

**Limitations of Prior Work**: Methods like MagicPony can learn 3D animal models from single-view image collections, but they only handle static images and ignore dynamic motion information. Methods like LASR require tedious optimization pipelines and do not model motion distributions. BKinD performs self-supervised keypoint discovery but is limited to 2D representations. General 4D generative models (such as diffusion-based methods) remain limited in motion quality and diversity.

**Key Challenge**: Learning the distribution of 3D animal motion requires registering unstructured internet videos (where each video features a different individual, different motion, and different perspective) to a unified 3D canonical model, while disentangling shape, appearance, and motion—all without any external annotations.

**Goal**: To learn 3D articulated animal motion generative models solely from raw internet videos, supporting the sampling of new motions from a latent space and the automatic generation of 4D animations from a single image.

**Key Insight**: Leveraging self-supervised features from DINO-ViT to provide cross-instance semantic correspondences, combined with a known coarse skeleton (e.g., "quadruped"), to train a motion VAE end-to-end under a video re-rendering objective.

**Core Idea**: A video photo-geometric autoencoding framework is designed to decompose videos into a rest-pose 3D mesh, texture, and joint pose sequence. The system is trained by reconstructing original video frames via differentiable rendering. The core component is a motion VAE, where a spatial-temporal Transformer encoder encodes the video into a motion latent code, and a decoder decodes it into a sequence of skeletal rotations.

## Method

### Overall Architecture
Input: Internet video collection of an animal category. Training pipeline: (1) DINO-ViT extracts features for each frame; (2) A spatial Transformer aggregates the skeletal features of each frame into a pose representation, and a temporal Transformer encodes the sequence of frames into motion VAE parameters $(\hat{\mu}, \hat{\Sigma})$; (3) $z$ is sampled and decoded by a spatio-temporal decoder into joint pose sequences; (4) Linear Blend Skinning (LBS) animates the mesh, and differentiable rendering reconstructs the RGB frames and masks. During inference, new motions can be generated simply by sampling $z$.

### Key Designs

1. **Spatial-Temporal Transformer Motion Encoder**:

    - **Function**: Extracts joint motion information from videos to encode into the VAE latent space.
    - **Mechanism**: For each frame, a skeletal feature descriptor $\nu_{t,b}$ is constructed (containing global DINO features, local key token features, 3D bone positions, and image projection positions). A spatial Transformer $E_s$ aggregates all skeletal features of the same frame into a single pose feature $\nu_{t,*}$, and a temporal Transformer $E_t$ maps the frame sequence to VAE distribution parameters. The decoder is symmetrically designed: the temporal decoder decodes $z$ into a sequence of frame features, and the spatial decoder decodes each frame feature into individual joint rotations.
    - **Design Motivation**: The spatial dimension captures the pose relationships across bones, while the temporal dimension captures the sequential patterns of motion. Decoupling these two dimensions makes the encoding more efficient.

2. **Video Photo-Geometric Autoencoding**:

    - **Function**: Trains the entire system using only video reconstruction objectives without pose annotations.
    - **Mechanism**: Decompose the video into a shared base mesh (SDF MLP + DMTet), instance deformation (conditional MLP), texture (MLP), and motion sequence (VAE). After animating the mesh via LBS, differentiable rendering is used to reconstruct RGB frames and masks. Losses include mask reconstruction ($L_2$ + distance transform), RGB reconstruction ($L_1$), DINO feature reconstruction, temporal smoothness regularization, and shape regularization.
    - **Design Motivation**: This is an "analysis-by-synthesis" strategy. By requiring the system to re-synthesize the input video from learned representations, it forces the system to learn meaningful 3D decomposition, fully bypassing the need for annotations.

3. **DINO Semantic Correspondence-Driven Cross-Instance Learning**:

    - **Function**: Establishes correspondences across different animal individuals without annotations.
    - **Mechanism**: The PCA features of DINO-ViT naturally contain cross-instance, part-level semantic correspondences. A feature field MLP $\psi(\mathbf{x})$ is learned in the canonical space, requiring rendered feature maps to match DINO feature maps, thereby establishing cross-instance correspondences at the 3D level.
    - **Design Motivation**: This is key to registering videos of different individual animals to a unified 3D model; otherwise, each video would only learn its own geometry, failing to model the shared motion distribution.

### Loss & Training
Total loss = mask loss ($L_2$ + distance transform) + RGB $L_1$ loss + DINO feature matching loss + VAE KL divergence + temporal smoothness regularization + shape regularization (Eikonal + joint rotation magnitude + deformation magnitude). A multi-hypothesis viewpoint prediction mechanism is used to handle viewpoint ambiguity.

## Key Experimental Results

### Main Results

| Method | 3D Reconstruction Quality | Motion Diversity | Motion Quality (FID) | Speed |
|---|---|---|---|---|
| MagicPony | Static only | No motion modeling | - | Fast |
| LASR | Video-level optimization | No generative ability | - | Extremely slow |
| 4D Diffusion Models | Fair | Limited | Poor | Slow |
| **Ponymation** | **Good** | **Diverse** | **Best** | **Inference in seconds** |

### Ablation Study

| Configuration | Effect | Explanation |
|---|---|---|
| W/o DINO features | Degraded shape and motion quality | Loss of cross-instance correspondence |
| W/o temporal Transformer | Incoherent motion | Inability to exchange information across frames |
| W/o VAE (direct regression) | Reduced motion diversity | Inability to sample new motions |
| Full model | Best | All components are complementary |

### Key Findings
- Plausible 3D animal motion distributions (such as horses running, walking, and standing) can be learned solely from internet videos.
- DINO features are critical for cross-instance pose registration—without them, the model completely fails to learn a shared motion space.
- Inference from a single image to a 4D animation takes only a few seconds, which is significantly faster than optimization-based methods (e.g., LASR takes hours).
- Generated motions outperform contemporary 4D generation methods in both visual quality and diversity.

## Highlights & Insights
- **Fully Unlabeled 3D Motion Learning**: No MoCap data, parametric models, or keypoint annotations are required; learning is achieved purely from YouTube videos. This dramatically lowers the barrier to 3D animal motion modeling.
- **Spatio-Temporal Decomposed Transformer VAE**: Resolves complex 4D problems by elegantly and efficiently processing spatial (inter-bone) and temporal (inter-frame) dimensions separately.
- **Clever Application of DINO Features**: Leverages the semantic correspondence capability of pre-trained vision features for 3D registration, completely avoiding the need for manual annotations.

## Limitations & Future Work
- Assumes a known coarse skeleton structure (e.g., "quadruped"), making it unable to handle completely unknown animal categories.
- Shape modeling is relatively coarse, making it difficult to recover high-frequency geometric details (such as fur).
- Limited by the quality of DINO features—correspondence might be inaccurate between individuals with significant appearance gaps.
- Can be extended to more animal categories or combined with diffusion models for higher-quality motion generation.

## Related Work & Insights
- **vs MagicPony**: Both leverage DINO features and SDF meshes, but while MagicPony only handles static images, Ponymation extends this to video motion modeling.
- **vs Human Motion Methods (e.g., MotionDiffuse)**: Human-centric methods rely on SMPL models and large-scale MoCap data, whereas Ponymation demonstrates these requirements can be bypassed using raw videos.
- The concept of video photo-geometric autoencoding can be generalized to motion learning for other non-rigid objects.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to learn 3D animal motion generative models from unlabeled videos.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple ablations and comparisons are provided, but quantitative evaluation metrics are limited by the lack of ground truth (GT).
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed descriptions of the methodology.
- Value: ⭐⭐⭐⭐⭐ Pioneering work that opens new avenues for non-human 3D motion modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VideoWorld: Exploring Knowledge Learning from Unlabeled Videos](../../CVPR2025/image_generation/videoworld_exploring_knowledge_learning_from_unlabeled_videos.md)
- [\[ECCV 2024\] Generating Human Interaction Motions in Scenes with Text Control](generating_human_interaction_motions_in_scenes_with_text_control.md)
- [\[ECCV 2024\] NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation](neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)
- [\[ECCV 2024\] NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)

</div>

<!-- RELATED:END -->
