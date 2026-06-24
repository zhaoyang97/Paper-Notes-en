---
title: >-
  [Paper Note] SAR3D: Autoregressive 3D Object Generation and Understanding via Multi-scale 3D VQVAE
description: >-
  [CVPR 2025][3D Vision][3D Generation] SAR3D proposes an autoregressive framework based on multi-scale 3D VQVAE, achieving high-quality 3D object generation in 0.82 seconds via "next-scale prediction" (instead of next-token prediction). Furthermore, the same set of VQVAE tokens can fine-tune LLMs to enable detailed 3D object understanding and description.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Generation"
  - "Autoregressive Model"
  - "VQVAE"
  - "3D Understanding"
  - "Multi-scale Representation"
date: 2026-05-08
content_hash: 8fa42f9c66a21e17
---

# SAR3D: Autoregressive 3D Object Generation and Understanding via Multi-scale 3D VQVAE

**Conference**: CVPR 2025  
**arXiv**: [2411.16856](https://arxiv.org/abs/2411.16856)  
**Code**: [https://cyw-3d.github.io/projects/SAR3D/](https://cyw-3d.github.io/projects/SAR3D/)  
**Area**: 3D Vision  
**Keywords**: 3D Generation, Autoregressive Model, VQVAE, 3D Understanding, Multi-scale Representation

## TL;DR

SAR3D proposes an autoregressive framework based on multi-scale 3D VQVAE, achieving high-quality 3D object generation in 0.82 seconds via "next-scale prediction" (instead of next-token prediction). Furthermore, the same set of VQVAE tokens can fine-tune LLMs to enable detailed 3D object understanding and description.

## Background & Motivation

**Background**: 3D object generation methods are mainly divided into three categories—SDS distillation-based methods (optimizing 3D representations using 2D diffusion priors, e.g., DreamFusion), feed-forward reconstruction methods (e.g., LGM, OpenLRM, achieving fast 3D generation via multi-view reconstruction), and native 3D diffusion models (e.g., LN3Diff, training diffusion models on a 3D latent space).

**Limitations of Prior Work**: (1) SDS distillation methods suffer from slow optimization, mode collapse, and Janus issues; (2) Feed-forward reconstruction methods are constrained by the quality of multi-view generation, resulting in view inconsistencies and difficulty in scaling to high resolutions; (3) Native 3D diffusion models exhibit slow inference speeds (requiring multi-step denoising), and their latent spaces are not easily leveraged for 3D understanding. Mesh generation methods (e.g., MeshAnything) are slow for face-by-face prediction and lack fine details.

**Key Challenge**: How to achieve sub-second 3D generation while ensuring generation quality? How to enable a single 3D representation to support both efficient generation and detailed understanding?

**Goal**: To extend the autoregressive next-scale prediction paradigm to the 3D domain, constructing a unified framework that simultaneously supports rapid 3D generation (conditioned on text/images) and 3D object understanding (captioning).

**Key Insight**: Next-scale prediction proposed by VAR in 2D image generation has proven more efficient than next-token prediction—by predicting the entire token map of the next scale instead of a single token, it drastically reduces the number of generation steps. 3D objects can be represented as triplanes, which possess a natural spatial structure well-suited for multi-scale quantization.

**Core Idea**: To encode 3D objects (multi-view RGB-D + camera parameters) into a hierarchical triplane token sequence using a multi-scale 3D VQVAE, then employ a GPT-style transformer for next-scale prediction to achieve high-speed generation; truncated low-scale tokens are used to fine-tune LLMs for 3D understanding.

## Method

### Overall Architecture

SAR3D consists of three parts: (1) **Multi-scale 3D VQVAE**—takes 6-view RGB-D renderings and Plücker camera embeddings as inputs, encodes them into a latent triplane, obtains token maps $R = (r_1, \ldots, r_K)$ across $K=10$ scales through multi-scale quantization, and decodes them back to the triplane for rendering supervision; (2) **Autoregressive 3D Generation**—a GPT-style transformer is conditioned on images/text to predict triplane tokens scale-by-scale; (3) **SAR3D-LLM**—uses truncated tokens from the first $K-2$ scales (incorporating only 37.5% of the total token count) to fine-tune LLaMA for 3D captioning.

### Key Designs

1. **Multi-scale 3D VQVAE**:

    - **Function**: Encodes 3D objects into hierarchical discrete triplane token sequences.
    - **Mechanism**: The input consists of 6-view $\tilde{M} = [I \oplus \text{Depth} \oplus \mathbf{P}] \in \mathbb{R}^{H \times W \times 10}$ (concatenation of RGB, depth, and Plücker coordinates). A multi-view convolutional encoder extracts the latent triplane $f \in \mathbb{R}^{3 \times h \times w \times C}$. Multi-scale quantization is then performed on $f$: $f$ is interpolated into 10 scales totaling $3 \times (1^2, 2^2, \ldots, 16^2)$, and the three representation planes of each scale independently search for nearest vectors in a shared codebook $Z \in \mathbb{R}^{V \times C}$ ($V=16384, C=8$). The decoder reconstructs the quantized triplane, which is supervised via volume rendering/Flexicubes across multiple views.
    - **Design Motivation**: Triplanes possess spatial inductive bias, making them inherently suitable for the concept of scales (where low-resolution scales capture global structures and high-resolution scales capture fine details), which is compatible with the multi-scale quantization design of VAR. Utilizing $\ell_2$-normalized codebook vectors and low-dimensional codewords ($C=8$) ensures high codebook utilization.

2. **Conditional Autoregressive 3D Generation**:

    - **Function**: Generates 3D objects at high speed conditioned on text or a single image.
    - **Mechanism**: A GPT-style decoder-only transformer with AdaLN layers is employed to autoregressively predict $p(r_k | r_1, \ldots, r_{k-1})$. Different triplane planes $r_k^i$ are distinguished by learnable positional embeddings. Text conditions are injected via cross-attention utilizing a CLIP text encoder. Image conditions are injected via a pre-cross-attention block using patch features extracted by DINOv2, with CLIP/DINOv2 pooled features serving as the start-of-sequence token. Classifier-free guidance is achieved by randomly dropping 10% of the conditions during training, mapped to $r_g = r_u + s(r_c - r_u)$ during inference.
    - **Design Motivation**: Next-scale prediction predicts an entire scale's token map per step instead of a single token, completing token generation in only $K=10$ steps. It is trained using a standard cross-entropy loss, bypassing the complicated sampling processes of diffusion models. DINOv2 is utilized for image conditioning since its patch features are better suited for spatial alignment.

3. **SAR3D-LLM (3D Understanding Module)**:

    - **Function**: Enables LLMs to understand and describe 3D objects.
    - **Mechanism**: Truncated scale tokens $\tilde{R} = (r_1, \ldots, r_{K-2})$ (only the first 8 scales, comprising 37.5% of the total token volume) are projected into LLaMA's text embedding space using an MLP projector and concatenated with static instruction text tokens for input into the LLM. A two-stage fine-tuning process is used: (1) training the projector to align 3D-text features while freezing the LLM; (2) jointly fine-tuning the LLM and the projector. This supports understanding both VQVAE-encoded and autoregressively generated 3D tokens.
    - **Design Motivation**: Low-scale tokens already encompass sufficient global semantic information for understanding tasks (with high-resolution scales primarily containing texture details), and truncation drastically reduces the input sequence length for the LLM. Unlike PointLLM which uses point clouds, VQVAE tokens encode much richer RGB and geometric information.

### Loss & Training

VQVAE training loss: $\mathcal{L} = \lambda_{\text{render}}\mathcal{L}_{\text{render}} + \lambda_{\text{VQ}}\mathcal{L}_{\text{VQ}} + \lambda_{\text{GAN}}\mathcal{L}_{\text{GAN}}$, where the rendering loss corresponds to MAE + perceptual loss, the VQ loss is the embedding error + commitment loss, and the GAN loss encourages a perceptually rich latent space. Normal loss and regularization are added in the subsequent Flexicubes fine-tuning phase. The autoregressive model is trained utilizing standard cross-entropy loss.

## Key Experimental Results

### Main Results

| Method | FID↓ | KID(%)↓ | MUSIQ↑ | COV(%)↑ | MMD(‰)↓ | Latency (s)↓ |
|------|------|---------|--------|---------|---------|---------|
| Splatter-Image | 48.80 | 3.65 | 30.33 | 37.66 | 30.69 | 0.83 |
| OpenLRM | 38.41 | 1.87 | 45.46 | 39.33 | 29.08 | 7.21 |
| LGM (V=4) | 19.93 | 0.55 | 54.78 | 50.83 | 22.06 | 3.87 |
| LN3Diff | 29.08 | 0.89 | 50.39 | 55.17 | 19.94 | 7.51 |
| **SAR3D-NeRF** | **22.55** | **0.42** | **65.77** | **74.17** | **13.63** | **1.64** |
| **SAR3D-Flex** | 27.30 | 0.63 | **67.24** | 71.50 | 15.25 | 2.92 |

### Ablation Study

| Configuration | Description |
|------|------|
| Full-scale tokens (K=10) | Used for 3D generation, completed in 0.82s |
| Truncated tokens (K-2=8) | Only 37.5% of the token volume, sufficient for high-quality 3D captioning |
| Separate generation & understanding vs. Unified | SAR3D unifies the two tasks to share the VQVAE, avoiding training different encoders |

### Key Findings

- SAR3D requires only **0.82 seconds** to complete 3D generation (NeRF version) on an A6000 GPU, which is significantly faster than LN3Diff's 7.51 seconds and CRM's 22.10 seconds, while heavily leading in MUSIQ and 3D quality metrics (COV/MMD).
- 3D captioning results show that SAR3D-LLM can describe fine-grained details (color, shape, spatial relationships) that PointLLM fails to capture, as VQVAE tokens embed complete appearance and geometric information.
- The discovery regarding truncated scales is intriguing: understanding tasks do not require full high-resolution inputs; low-scale tokens are sufficient to capture crucial semantic information.
- The same tokens generated by VQVAE can be used both for autoregressive generation and immediate understanding by LLMs, realizing a pipeline of "generation + instant understanding."

## Highlights & Insights

- **Successful Validation of Next-scale Prediction in 3D**: Promoting the VAR paradigm from 2D images to 3D triplanes demonstrates that multi-scale autoregression remains highly efficient in 3D generation. This points out a novel direction for other 3D tasks such as scene generation and 4D generation.
- **Ingenious Finding of Using Truncated Tokens for Understanding**: High-resolution scale tokens mainly encode high-frequency texture details, holding limited value for semantic understanding. Truncating 62.5% of the tokens does not degrade understanding quality but significantly mitigates the computational burden of LLMs.
- **Unified VQVAE Design**: Enables a single encoder to serve two vastly different tasks—generation and understanding—avoiding the overhead of training independent models for each task.

## Limitations & Future Work

- Generation and understanding currently still utilize two independent autoregressive models (one for 3D token generation, and an LLM for understanding); future work should fuse them into a genuinely multimodal unified model.
- Geometric and texture quality are constrained by volume rendering; deploying more efficient 3D representations or cascaded generation could yield further enhancements.
- Scaling laws remain unverified—due to constraints in computational resources, the performance of larger models was not tested.
- Exclusively trained on the Objaverse dataset, so the generalization capability to real-world 3D objects remains to be validated.

## Related Work & Insights

- **vs. LN3Diff**: LN3Diff similarly employs triplane representations but generates them via a diffusion model, requiring multi-step denoising during inference (7.51s). SAR3D completes the generation in 10 autoregressive steps (0.82s) and achieves superior quality.
- **vs. LGM**: LGM is a multi-view-to-3D method that reports the lowest FID (19.93). However, its 3D quality indicators (COV/MMD) are far inferior to SAR3D, indicating that superior 2D rendering quality does not guarantee accurate 3D geometry.
- **vs. PointLLM**: PointLLM feeds point clouds into LLMs as 3D representations, where point clouds neglect substantial appearance and detail information. SAR3D utilizes VQVAE tokens (containing RGB, depth, and normal information) to generate significantly richer descriptions.

## Rating

- Novelty: ⭐⭐⭐⭐ Extending VAR's next-scale prediction to 3D successfully is a valuable and meaningful development.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on both generation and understanding tasks, though ablation studies are somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with detailed descriptions of methodologies.
- Value: ⭐⭐⭐⭐ The 0.82-second 3D generation latency and the unified generation-understanding framework carry high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing](treemeshgpt_artistic_mesh_generation_with_autoregressive_tree_sequencing.md)
- [\[CVPR 2025\] Structured 3D Latents for Scalable and Versatile 3D Generation](structured_3d_latents_for_scalable_and_versatile_3d_generation.md)
- [\[CVPR 2025\] Digital Twin Catalog: A Large-Scale Photorealistic 3D Object Digital Twin Dataset](digital_twin_catalog_a_large-scale_photorealistic_3d_object_digital_twin_dataset.md)
- [\[CVPR 2025\] FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting](fruitninja_3d_object_interior_texture_generation_with_gaussian_splatting.md)
- [\[CVPR 2025\] HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos](hot3d_hand_and_object_tracking_in_3d_from_egocentric_multi-view_videos.md)

</div>

<!-- RELATED:END -->
