---
title: >-
  [Paper Note] IDOL: Instant Photorealistic 3D Human Creation from a Single Image
description: >-
  [CVPR 2025][Video Generation][Single-image human reconstruction] IDOL achieves instant (<1s) high-fidelity animatable 3D human reconstruction from a single image input by constructing HuGe100K, a large-scale multi-view dataset of 100k human subjects, and training a Transformer-based feed-forward model, significantly outperforming existing methods in quality and generalization.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Single-image human reconstruction"
  - "3D Gaussian Splatting"
  - "SMPL-X"
  - "Large-scale dataset"
  - "Feed-forward reconstruction"
date: 2026-05-08
content_hash: 4314ae733130b6d5
---

# IDOL: Instant Photorealistic 3D Human Creation from a Single Image

**Conference**: CVPR 2025  
**arXiv**: [2412.14963](https://arxiv.org/abs/2412.14963)  
**Code**: [https://yiyuzhuang.github.io/IDOL/](https://yiyuzhuang.github.io/IDOL/)  
**Area**: Video Generation  
**Keywords**: Single-image human reconstruction, 3D Gaussian Splatting, SMPL-X, Large-scale dataset, Feed-forward reconstruction

## TL;DR

IDOL achieves instant (<1s) high-fidelity animatable 3D human reconstruction from a single image input by constructing HuGe100K, a large-scale multi-view dataset of 100k human subjects, and training a Transformer-based feed-forward model, significantly outperforming existing methods in quality and generalization.

## Background & Motivation

**Background**: Single-image 3D human reconstruction is a fundamental task in virtual reality, gaming, and 3D content creation. Existing methods are mainly divided into three categories: (1) implicit representation-based methods (e.g., PIFu, SIFU) rely on pixel alignment and have limited receptive fields; (2) iterative-optimization methods (e.g., GTA, SIFU) require minutes of inference and suffer from dependency on SMPL estimation accuracy; (3) general large-scale reconstruction models (e.g., LGM) lack human-specific prior knowledge.

**Limitations of Prior Work**: The core bottlenecks lie across three aspects. In terms of data, the largest public human dataset, MVHumanNet, contains only 4,500 subjects, which is far from sufficient for training domain-generalized large models; in terms of models, existing methods either suffer from error accumulation in SMPL parameter estimation or require time-consuming optimization; in terms of representation, the reconstructed results are usually not directly animatable and require extra post-processing.

**Key Challenge**: High-quality animatable human reconstruction demands large-scale and diverse 3D human training data. However, real-world 3D human data acquisition is extremely expensive and limited in scale, leaving this data-quality conflict unresolved for a long time.

**Goal**: (1) How to synthesize high-quality multi-view human data at scale? (2) How to design an efficient feed-forward model to achieve instant reconstruction? (3) How to make the reconstructed results directly animatable and editable?

**Key Insight**: This work returns to first principles, rethinking the challenge through three dimensions: data, model, and representation. Generative models are leveraged to synthesize large-scale training data, a unified representation based on UV space is designed, and scalable feed-forward reconstruction is achieved.

**Core Idea**: Synthesizing a multi-view human dataset of 100k subjects using generative models, and training a Transformer-based feed-forward model to predict Gaussian attribute maps in UV space, achieving instant, high-fidelity, and animatable 3D human reconstruction.

## Method

### Overall Architecture

The pipeline of IDOL consists of two phases. Data phase: Utilize the FLUX text-to-image model and an improved Champ video model (MVChamp) to generate the HuGe100K dataset (100k subjects × 24 views = 2.4 million images). Model phase: Given a single 1024×1024 human image, extract high-resolution features using the Sapiens high-resolution encoder, align and fuse the image tokens with learnable UV tokens via the UV-Alignment Transformer, and decode the features through the UV Decoder to predict Gaussian attribute maps in the SMPL-X UV space. With predicted SMPL-X parameters, animatable 3D humans are generated.

### Key Designs

1. **HuGe100K Dataset Construction Pipeline**:

    - **Function**: Provides large-scale, high-quality, and diverse multi-view human training data.
    - **Mechanism**: Generated in a two-stage pipeline: (a) Use GPT-4 templates to uniformly sample human attributes (age, body shape, clothing, ethnicity, gender) to generate text prompts, followed by FLUX synthesizing 90k synthetic images + 10k DeepFashion real images; (b) Train MVChamp (an improved version of Champ) to convert the single image + SMPL-X pose sequence into 24-view consistent multi-view images. MVChamp is fine-tuned on THuman 2.1 scan data for temporal layers to enhance 3D consistency, and introduces a Temporal Shift Denoising strategy to improve continuity between start and end frames.
    - **Design Motivation**: Real 3D human data acquisition is expensive and small in scale (MVHumanNet is only 4,500 subjects), while model generalization requires diverse training data at the 100k scale. Generative dataset construction is key to breaking through this data bottleneck.

2. **Gaussian Human Representation based on UV Space**:

    - **Function**: Transforms 3D human reconstruction into attribute map prediction in 2D UV space, reducing computational complexity and enabling natural animation.
    - **Mechanism**: Utilizing the pre-defined UV parameterization of SMPL-X, the attributes of each Gaussian primitive (position offset $\delta\mu$, rotation offset $\delta r$, scale offset $\delta s$, color $c$, opacity $\alpha$) are modeled as offsets relative to the SMPL-X vertices, encoded in 2D UV maps. Animation is directly driven via LBS (Linear Blend Skinning), where skinning weights are obtained through voxel field and barycentric coordinate interpolation.
    - **Design Motivation**: Directly predicting all Gaussian primitives in 3D space is computationally expensive. UV space leverages the geometric/semantic prior of SMPL-X to ensure semantic consistency of corresponding body parts across different humans, and naturally supports animation and editing.

3. **High-Resolution Encoder + UV-Alignment Transformer**:

    - **Function**: Extracts rich features from high-resolution input images and maps irregular image features onto the regular UV space.
    - **Mechanism**: Sapiens-1B (a ViT pre-trained with MAE on 300 million human images) is adopted as the frozen encoder to encode the 1024×1024 image into patch tokens. These tokens are then concatenated with learnable spatial UV tokens and undergo self-attention fusion through $D$-layer Transformer blocks to output enhanced UV tokens, which are decoded into Gaussian attribute maps via CNN upsampling.
    - **Design Motivation**: Prior methods were limited by low-resolution encoders (DINOv2 supports up to 448) and could not utilize high-resolution information. Sapiens is specifically pre-trained on human data, maintaining fine-grained texture and diverse pose information better.

### Loss & Training

End-to-end differentiable training: For each sample, the frontal view is selected as the reference input, and several multi-view images are randomly sampled for supervision. Multi-view predictions are generated via differentiable rendering, and the loss function is MSE + perceptual loss: $\mathcal{L} = \sum_{i=1}^{N}(\|I_{gt,i} - I_{pred,i}\|^2 + \lambda L_{vgg})$. The encoder parameters are frozen, and only the 0.5B-parameter Transformer and UV Decoder are trained.

## Key Experimental Results

### Main Results

| Method | MSE ↓ | PSNR ↑ | LPIPS ↓ | Inference Speed |
|------|-------|--------|---------|---------|
| SIFU | 0.042 | 14.204 | 1.612 | ~Minutes |
| GTA | 0.041 | 14.282 | 1.629 | ~Minutes |
| DreamGaussian | - | - | - | ~2min |
| **IDOL (full)** | **0.008** | **21.673** | **1.138** | **<1s** |

IDOL significantly outperforms all baselines across all metrics, with PSNR over 7.4dB higher than the best baseline, while being hundreds of times faster in inference speed.

### Ablation Study

| Configuration | MSE ↓ | PSNR ↑ | LPIPS ↓ |
|------|-------|--------|---------|
| Full model | 0.008 | 21.673 | 1.138 |
| w/o HuGe100K | 0.017 | 19.225 | 1.326 |
| w/o Sapiens (using DINOv2) | - | Quality degradation | Blurry textures |

### Key Findings

- HuGe100K dataset is the largest contributor: without it, MSE doubles, PSNR drops by 2.4dB, and severe color bleeding and detail blurring occur.
- Sapiens encoder is significantly better than DINOv2 in texture details and clothing wrinkles; high-resolution input is crucial.
- Pixel-aligned methods like SIFU/GTA perform drastically worse under non-orthogonal projections (focal length 35-80mm), exposing the vulnerability of their orthogonal projection assumption.
- MVChamp's 3D consistency fine-tuning and Temporal Shift Denoising strategies make significant contributions to data quality.

## Highlights & Insights

- **Paradigm innovation in data engineering**: The concept of synthesizing training data at scale (100k subjects) using generative models is highly inspiring. This paradigm can be directly transferred to domains where real 3D data acquisition is limited (e.g., medical, industrial).
- **UV space representation unifies reconstruction and animation**: Drives animation without post-processing, greatly simplifying the pipeline from reconstruction to application. Compared with implicit representations like NeRF, the combination of Gaussian Splatting + UV map holds obvious advantages in real-time rendering and editability.
- **Insightful choice of Sapiens encoder**: Foundation models specifically pre-trained on human data far outperform general-purpose models in downstream tasks, underlining the value of domain-specific foundation models.

## Limitations & Future Work

- Insufficient fine modeling of facial expressions and identity; the current architecture lacks a dedicated facial design.
- Only supports single-frame fixed view synthesis; future work can extend to generating longer motion sequences.
- Poor performance in handling half-body inputs; the data generation strategy needs improvement to cover such scenarios.
- The dataset depends on the generation quality ceiling of MVChamp; hand and shoe details under some views still exhibit artifacts.
- Domain bias of the generated dataset may impact performance generalization in real-world scenes.

## Related Work & Insights

- **vs SIFU/GTA**: These two methods are based on pixel alignment + SMPL optimization, suffering from slow inference and relying heavily on the orthogonal projection assumption. IDOL trains a feed-forward model through large-scale data, achieving instant reconstruction and higher robustness to projection models.
- **vs LGM**: LGM is a general 3D reconstruction model that lacks human-specific priors. IDOL introduces strong human priors through SMPL-X UV space and a human-specific encoder.
- **vs E3Gen**: E3Gen also predicts Gaussian attributes in UV space but does not support arbitrary image inputs. IDOL achieves open-domain generalization through training on the large-scale dataset.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined scheme of large-scale generative datasets + UV space feed-forward reconstruction is systematic, though individual components are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Both data generation ablation and reconstruction ablation are thorough, and qualitative comparisons are highly convincing.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, organized from the three dimensions of data, model, and representation.
- Value: ⭐⭐⭐⭐⭐ The HuGe100K dataset with 100k subjects and the instant reconstruction capability significantly advance the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis](levitor_3d_trajectory_oriented_image-to-video_synthesis.md)
- [\[ECCV 2024\] SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion](../../ECCV2024/video_generation/sv3d_novel_multi-view_synthesis_and_3d_generation_from_a_single_image_using_late.md)
- [\[CVPR 2025\] HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation](hoigen-1m_a_large-scale_dataset_for_human-object_interaction_video_generation.md)
- [\[CVPR 2025\] Pathways on the Image Manifold: Image Editing via Video Generation](pathways_on_the_image_manifold_image_editing_via_video_generation.md)
- [\[CVPR 2025\] Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion](zero-1-to-a_zero-shot_one_image_to_animatable_head_avatars_using_video_diffusion.md)

</div>

<!-- RELATED:END -->
