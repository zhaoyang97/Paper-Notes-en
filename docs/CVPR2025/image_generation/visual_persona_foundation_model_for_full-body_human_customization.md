---
title: >-
  [Paper Note] Visual Persona: Foundation Model for Full-Body Human Customization
description: >-
  [CVPR 2025][Image Generation][Full-Body Human Customization] Visual Persona is proposed as the first foundation model for full-body human customization. Through large-scale paired dataset curation (580K images / 100K identities) and a body-part partitioned Transformer decoder architecture, it achieves high-fidelity full-body appearance preservation and diverse text-guided generation.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Full-Body Human Customization"
  - "Text-to-Image Generation"
  - "Diffusion Models"
  - "Identity Preservation"
  - "Data Curation"
date: 2026-05-08
content_hash: 6eed54197a853d28
---

# Visual Persona: Foundation Model for Full-Body Human Customization

**Conference**: CVPR 2025  
**arXiv**: [2503.15406](https://arxiv.org/abs/2503.15406)  
**Code**: [Project Page](https://cvlab-kaist.github.io/Visual-Persona)  
**Area**: Image Generation  
**Keywords**: Full-Body Human Customization, Text-to-Image Generation, Diffusion Models, Identity Preservation, Data Curation

## TL;DR

Visual Persona is proposed as the first foundation model for full-body human customization. Through large-scale paired dataset curation (580K images / 100K identities) and a body-part partitioned Transformer decoder architecture, it achieves high-fidelity full-body appearance preservation and diverse text-guided generation.

## Background & Motivation

- Text-to-image customized generation has made significant progress in recent years, but most approaches focus solely on preserving facial identity.
- Full-body human customization (including the complete preservation of the face, clothing, and accessories) is a severely overlooked area.
- Achieving full-body customization requires large-scale paired human datasets (multiple images of the same person with consistent full-body identities), which are extremely difficult to obtain.
- Existing methods such as StoryMaker can only be trained with a single image (without paired data), leading to a trade-off between identity preservation and text alignment.
- Global encoding approaches using CLIP or face recognition models fail to capture fine-grained local features of the entire body.
- Compressing input features into a small number of token embeddings (usually $l_H=16$) loses crucial body part details.
- A new data curation method and model architecture are required to simultaneously achieve identity preservation and text alignment.

## Method

### Overall Architecture

Visual Persona incorporates two major innovations: (1) A data curation pipeline that leverages vision-language models (VLMs) to filter paired data with consistent full-body identities from large amounts of unpaired human images, constructing the Visual Persona-500K dataset; (2) A model architecture based on a body-part decomposed Transformer encoder-decoder architecture, adapted to a pre-trained T2I diffusion model (with frozen parameters), which conditions the diffusion process via dense identity embeddings. Only the body-partitioned Transformer decoder and identity cross-attention modules are trained.

### Key Designs

**1. Visual Persona-500K Dataset Curation**
- **Function**: To construct a large-scale paired human dataset with consistent full-body identities.
- **Mechanism**: A three-stage curation pipeline is employed: first, a face recognition model is used with cosine similarity to filter image subsets with consistent facial identities; second, LLaVA is used to assess full-body clothing consistency (with the prompt: "Are they wearing exactly the same clothes?"); finally, Phi-3 is used to generate descriptive texts free of identity information (focusing on expressions, poses, actions, and environments).
- **Design Motivation**: Paired data is key to achieving both identity preservation and text alignment simultaneously (single-image training easily overfits to identity-irrelevant attributes like positions and poses). VLMs are simple and effective tools for evaluating complex full-body appearance consistency. Ultimately, 580K images of 100K identities were collected.

**2. Body Part Decomposition + DINOv2 Encoder**
- **Function**: Decomposes the input human image into independent body regions (totaling $N=5$: full-body, face, upper body, legs, and shoes) to extract fine-grained local appearance features.
- **Mechanism**: Utilizes an off-the-shelf body parsing method to segment body regions, which are then cropped, zero-padded, and resized into independent images. A pre-trained DINOv2 encoder is used to extract local token features $F \in \mathbb{R}^{N \times h \times w \times d_F}$ (utilizing all local tokens rather than just the CLS token).
- **Design Motivation**: The self-supervised training of DINOv2 makes it superior to CLIP at capturing fine-grained structural and textural information. Body decomposition allows the diffusion model to focus independently on each part, preventing the blending of local details caused by global encoding.

**3. Body-Partitioned Transformer Decoder**
- **Function**: Projects the encoded features of each body part into corresponding dense identity embeddings to condition the diffusion model.
- **Mechanism**: Each layer consists of cross-attention (associating learnable latent embeddings $H^{i,j}$ with the corresponding body part features $F^i$) + self-attention (learning internal relationships among embeddings) + MLP. After $M$ layers of iteration, all body part embeddings are concatenated along the token length: $C_H^* = \text{Concat}([C_H^1, ..., C_H^N]) \in \mathbb{R}^{(N \times l_H) \times d_H}$. They are then injected into the diffusion model via decoupled cross-attention.
- **Design Motivation**: Using dense embeddings ($l_H = 16 \times 16$) retains significantly more details than traditional methods ($l_H=16$). The partitioned design prevents the mixing of features from different body parts; ablation studies demonstrate that body decomposition improves D-T from 6.13 to 6.67.

### Loss & Training

Standard diffusion model noise prediction loss:

$$L = \mathbb{E}_{z_{Y,t}, \epsilon, t, C_T, C_H^*}\left[\|\epsilon - \epsilon_\theta(z_{Y,t}, t, C_T, C_H^*)\|_2^2\right]$$

Only the parameters of the body-partitioned Transformer decoder and the identity cross-attention modules are updated during backpropagation.

## Key Experimental Results

### Main Results: Quantitative Comparison (GPT-based DreamBench++ Evaluation)

| Method | D-I↑ (SSHQ) | D-T↑ | D-H↑ | D-I↑ (PPR10K) | D-T↑ | D-H↑ |
|------|------------|------|------|--------------|------|------|
| IP-Adapter-FaceID| 1.78 | 7.50 | 2.76 | 1.86 | 7.49 | 2.81 |
| InstantID | 1.52 | 6.94 | 2.37 | 1.70 | 7.12 | 2.63 |
| PhotoMaker | 1.70 | 7.72 | 2.64 | 2.03 | 7.64 | 3.03 |
| StoryMaker | 6.74 | 7.08 | 6.71 | 6.80 | 6.77 | 6.63 |
| **Visual Persona** | **7.10** | **7.15** | **6.99**| **7.30** | 6.67 | **6.85** |

*Visual Persona significantly leads in identity preservation (D-I) and achieves the highest harmonic mean (D-H).*

### Ablation Study

| Config | D-I↑ | D-T↑ | D-H↑ |
|------|------|------|------|
| (I) MLP only | 6.66 | 7.11 | 6.74 |
| (II) + Self-Attention | 6.54 | 7.01 | 6.63 |
| (III) + Cross-Attention | 7.47 | 6.13 | 6.40 |
| **(IV) + Body Part Decomp.** | **7.30** | **6.67** | **6.85** |

| Token Length $l_H$ | D-I↑ | D-T↑ | D-H↑ |
|-------------------|------|------|------|
| $4 \times 4$ | 5.51 | 6.90 | 5.81 |
| $8 \times 8$ | 6.56 | 6.50 | 6.52 |
| $16 \times 16$ | **7.30** | **6.67** | **6.85** |

### Key Findings

- Body part decomposition is crucial for improving text alignment while maintaining identity preservation (D-T: 6.13 to 6.67).
- Dense identity embeddings ($16 \times 16$) improve D-I by up to 32.5% compared to sparse embeddings ($4 \times 4$).
- Cross-image training (paired data) supports large geometric deformations better than reconstruction training (single-image).

## Highlights & Insights

1. **Data-Driven Breakthrough**: Evaluates full-body consistency automatically via VLMs, overcoming the core bottleneck of paired data acquisition.
2. **Dense Embedding Paradigm**: Breaks the convention of compressing identities into a small number of tokens, proving that more tokens better preserve full-body details.
3. **Zero-Shot Multi-Scenario Applications**: Supports multi-person customization, virtual try-on, character stylization, and consistent story generation without additional training.

## Limitations & Future Work

- Depends on the accuracy of the VLM's assessment of clothing consistency, where subtle differences may be ignored.
- The quality of the body parsing model directly affects the ultimate performance.
- Currently fixed at $N=5$ body regions; more flexible dynamic partitioning could further improve performance.
- There may be biases in the ethnicity and age distribution of the dataset.

## Related Work & Insights

- Compared with facial customization methods like IP-Adapter and InstantID, Visual Persona extends the customization scope from the face to the entire body.
- The choice of DINOv2 as the encoder highlights the advantages of self-supervised features in fine-grained preservation tasks.
- The paradigm of paired data + cross-image training can be extended to other identity-consistent generation tasks.

## Rating

⭐⭐⭐⭐ — Systematically addresses the two major bottlenecks of full-body human customization: data and model. The methodology design is sound, and the ablation analysis is thorough. The data curation pipeline has high practical value, and multiple downstream applications demonstrate the versatility of the approach. The evaluation metrics rely on GPT-based scoring, which may present certain limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](intermimic_towards_universal_whole-body_control_for_physics-based_human-object_i.md)
- [\[CVPR 2025\] DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image](decloth_decomposable_3d_cloth_and_human_body_reconstruction_from_a_single_image.md)
- [\[ECCV 2024\] HIMO: A New Benchmark for Full-Body Human Interacting with Multiple Objects](../../ECCV2024/image_generation/himo_a_new_benchmark_for_full-body_human_interacting_with_multiple_objects.md)
- [\[CVPR 2025\] Goku: Flow Based Video Generative Foundation Models](goku_flow_based_video_generative_foundation_models.md)
- [\[CVPR 2025\] MCA-Ctrl: Multi-party Collaborative Attention Control for Image Customization](mca_ctrl_attention_control_customization.md)

</div>

<!-- RELATED:END -->
