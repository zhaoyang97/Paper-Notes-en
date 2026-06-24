---
title: >-
  [Paper Note] Editable Image Elements for Controllable Synthesis
description: >-
  [ECCV 2024][Image Generation][Image Editing] This work proposes an "Editable Image Elements" representation that decomposes an input image into a set of semantically aligned patch embeddings (similar to superpixels). Each patch is associated with spatial position and size attributes. Users can directly edit these attributes (moving, scaling, deleting), and a Stable Diffusion-based decoder then synthesizes realistic images from them.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Image Editing"
  - "Diffusion Models"
  - "Editable Representation"
  - "Superpixel Decomposition"
  - "Spatial Control"
date: 2026-05-08
content_hash: 020b67811f88318a
---

# Editable Image Elements for Controllable Synthesis

**Conference**: ECCV 2024  
**arXiv**: [2404.16029](https://arxiv.org/abs/2404.16029)  
**Code**: [Yes](https://jitengmu.github.io/Editable_Image_Elements/)  
**Area**: Image Generation / Image Editing  
**Keywords**: Image Editing, Diffusion Models, Editable Representation, Superpixel Decomposition, Spatial Control

## TL;DR

This work proposes an "Editable Image Elements" representation that decomposes an input image into a set of semantically aligned patch embeddings (similar to superpixels). Each patch is associated with spatial position and size attributes. Users can directly edit these attributes (moving, scaling, deleting), and a Stable Diffusion-based decoder then synthesizes realistic images from them.

## Background & Motivation

### Problem Introduction

Diffusion models have made significant progress in text-guided image synthesis, but **editing user-provided images** remains challenging. The high-dimensional noise input space of diffusion models is inherently unsuitable for image inversion or spatial editing. Existing methods face a fundamental trade-off: **faithful reconstruction of the original image** vs. **editable representation**.

### Limitations of Prior Work

**Noise inversion methods** (e.g., Null-text Inversion): Invert images into noise space, but there is an inherent conflict between faithful reconstruction and editability.

**ControlNet-like methods**: Guide diffusion models with conditions such as edge maps, but fail to retain image attributes missing from the conditioning signal.

**Concept tuning methods** (e.g., DreamBooth): Generate new instances of a concept rather than modifying the original image.

**Self-Guidance**: Perform spatial editing by caching attention maps, but editing results tend to lack realism.

The core problem is that the representations used in these methods (noise maps or edge maps) **are not suitable for precise spatial control**.

### Key Insight

This paper proposes a new image representation: decomposing an image into a set of "image elements," where each element is a semantically meaningful patch associated with an appearance embedding, position, and size. This representation can faithfully reconstruct the original image while naturally supporting spatial editing operations—moving, scaling, or deleting a patch directly corresponds to the movement, scaling, and removal of an object.

## Method

### Overall Architecture

The entire pipeline consists of three stages:
1. **Image Element Extraction**: Performs SLIC clustering based on Segment Anything Model (SAM) features to decompose the image into $N=256$ semantically aligned superpixel patches.
2. **Content Encoder Training**: Trains a patch encoder using an autoencoder framework to encode the appearance of each patch into an embedding decoupled from its spatial position.
3. **Diffusion Decoder Training**: Adds image element cross-attention layers on top of Stable Diffusion, training the model to decode edited image elements back into realistic images.

### Key Designs

#### 1. **Image Element Extraction (Image Elements)**

**Function**: Decomposes an image $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$ into a set of $N=256$ disjoint continuous patches $\mathbf{A} = \{\mathbf{a}_1, ..., \mathbf{a}_N\}$.

**Mechanism**: Modifies the SLIC algorithm to operate in the SAM feature space. Starting with $16 \times 16$ uniformly spaced query points, each pixel $m$ is assigned to a query element $n$ by combining SAM's semantic affinity map $\mathbf{s}(m,n)$ and Euclidean spatial distance $\mathbf{d}(m,n)$:

$$g(m) = \arg\max_{n \in \{1,...,N\}} [\mathbf{s}(m,n) - \beta \cdot \mathbf{d}(m,n)]$$

Here, $\beta=64$ balances feature similarity and spatial distance. This formula is equivalent to running one iteration of SLIC in the high-quality semantic affinity space of SAM.

**Design Motivation**: Conventional grid-based latent codes are unsuitable for spatial editing (deleting a code at a certain position cannot leave a blank), whereas semantically aligned superpixels naturally exhibit editability. Unlike direct SAM segmentation, the SLIC constraint guarantees uniform patch sizes (averaging 1024 pixels), preventing encoding/decoding difficulties caused by extreme size variations in segmentation results.

#### 2. **Content Encoder**

**Function**: Encodes the appearance of each patch into an embedding decoupled from its spatial position.

**Mechanism**: Employs a convolutional encoder with the same architecture as the Stable Diffusion KL-autoencoder (4 downsampling layers). All patches are resized to the same dimensions before being fed into the encoder, ensuring size information is decoupled. The encoder is optimized by jointly training a lightweight Transformer decoder $\mathcal{D}_{\text{light}}$ (8 self-attention layers + 4 cross-attention layers):

$$\mathcal{E}^* = \arg\min_{\mathcal{E}} \min_{\mathcal{D}_{\text{light}}} \ell_2(\mathbf{x}, \mathcal{D}_{\text{light}}(\mathbf{S}))$$

where $\mathbf{S} = \{(\mathcal{E}(\mathbf{a}_n), \mathbf{p}_n)\}$ is the set of encoded image elements, and $\mathbf{p}_n = (x_n, y_n, w_n, h_n)$ represents the spatial position and size attributes of the patch.

**Design Motivation**: Stage-wise training is proven to be necessary—jointly training the content encoder and diffusion decoder yields worse reconstruction and editing quality compared to stage-wise training (MSE 0.0138 vs 0.0069, with a user preference of only 34.2%). Freezing the encoder after training ensures the stability of decoder training.

#### 3. **Diffusion Decoder**

**Function**: Generates realistic images based on edited image elements.

**Mechanism**: In the UNet of Stable Diffusion v1.5, a new cross-attention layer $\theta_{\mathcal{S}}$ is inserted after each existing text cross-attention layer, using image elements as key/value. The outputs of both text and image element cross-attentions are added to the self-attention features with equal weights. The training objective is:

$$\mathcal{L}_{SD}^{\text{new}} = \mathbb{E}_{\mathbf{z}, \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), t} ||\epsilon - \mathcal{U}(\mathbf{z}_t, t, \mathbf{C}, \mathbf{S}; \theta_{\mathcal{U}}, \theta_{\mathcal{S}})||_2^2$$

During inference, classifier-free guidance is applied, utilizing the same guidance weight $w=3.0$ for both text and image elements, with 50 DDIM sampling steps.

**Design Motivation**: A lightweight decoder trained solely with MSE produces blurry reconstructions, and edited elements introduce out-of-distribution biases not seen during training. Stable Diffusion's strong image prior helps fill in the unspecified information of edited elements.

### Loss & Training

**Image Element Dropout Training Strategy**:

A directly trained decoder performs well in reconstruction, but the visual realism of edited images drops sharply (as cases with missing elements, overlaps, and gaps are unseen during training). The solution is to randomly drop out image elements during training:

- Semantic SAM is used to obtain an object mask database, and image elements in areas covered by randomly selected masks are dropped out.
- **Random Partition Trick**: Random segmentation results from another image are used as dropout masks to prevent correlation between object boundaries and image element boundaries. Without random partition, user preference drops to 27.3%.

Training is split into two stages:
1. Content Encoder + Lightweight Decoder: MSE loss, 30 epochs
2. Diffusion Decoder: Freeze the content encoder, standard diffusion loss, ~180k iterations

## Key Experimental Results

### Main Results

**2AFC User Study (900+ judgments, evaluating editing quality and image realism)**:

| Method Comparison | Ours Preference | Opponent Preference | Notes |
|---------|-----------|-----------|------|
| Ours vs Self-Guidance | ~75% | ~25% | Self-guidance exhibits unstable editing on SDXL |
| Ours vs Paint-by-Example | ~80% | ~20% | Two-stage operations lead to quality degradation |
| Ours vs InstructPix2Pix | ~70% | ~30% | InstructPix2Pix struggles with precise spatial control |

Supported editing operations: Object scaling, rearrangement, dragging, de-occlusion, removal, variation, and image composition.

### Ablation Study

**Design Choices Ablation (Reconstruction Quality + User Editing Preference)**:

| Configuration | MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | Editing Preference |
|------|------|-------|-------|--------|------|---------|
| **Default (Stage-wise + Frozen + Random Partition)** | **0.0069** | **22.98** | **0.6354** | 0.3376 | 10.82 | Baseline |
| Joint training (non-stage-wise) | 0.0138 | 19.74 | 0.5208 | 0.3697 | 11.91 | 34.2% |
| Stage-wise but without freezing encoder | 0.0097 | 21.35 | 0.5962 | 0.3238 | 10.48 | 37.1% |
| Stage-wise + Frozen, without random partition | 0.0066 | 23.15 | 0.6389 | 0.3262 | 9.75 | 27.3% |

### Key Findings

1. **Stage-wise training is key**: The PSNR difference between joint training and stage-wise training is 3.24 dB, and the editing preference is only 34.2%.
2. **Freezing the content encoder is important**: Without freezing, the encoder co-degenerates with the diffusion decoder, with an editing preference of only 37.1%.
3. **Random Partition is crucial for editing quality**: Without it, the model tends to inpaint along element boundaries, resulting in unnatural image boundaries, with a preference rate of only 27.3%.
4. The method is particularly prominent in object scaling scenarios—it can automatically handle de-occlusion (e.g., automatically completing missing corners when a car is scaled down).

## Highlights & Insights

- **Highly innovative representation design**: Representing images as a set of semantically aligned, editable elements naturally unifies the usually conflicting objectives of "faithful reconstruction" and "editability".
- **Elegant position-appearance decoupling**: Patch encoding is independent of location, and location is represented by editable continuous coordinates, making movement, scaling, and deletion as simple as modifying attribute values.
- **Random Partition training strategy**: A clever insight—the conditioning inpainting probability should be independent of the image element segmentation pattern, thus utilizing random segmentations instead of the actual ones for dropout during training.
- Bridges the fields of superpixel segmentation and diffusion-model-based editing.

## Limitations & Future Work

- **Imperfect reconstruction quality** (PSNR 22.98): Editing high-resolution user-provided images remains a challenge.
- **Uneditable appearance embeddings**: The current method supports spatial editing but does not support style editing.
- The number of image elements is fixed at 256, which might lack flexibility for very simple or complex scenes.
- Based on Stable Diffusion v1.5; upgrading to newer base models (like SDXL) could yield significant improvements.
- Future directions include exploring image elements as a more compact and controllable latent space for image generation (e.g., training a prior model to directly generate image elements).

## Related Work & Insights

- **BlobGAN**: Similarly exposes manipulable object anchors, but cannot be used to edit input images.
- **Swapping Autoencoder / Diffusion Autoencoder**: Edits in the latent space, but inherits a trade-off between reconstruction accuracy and editability.
- **SAM / SLIC**: This work elegantly combines the two, performing SLIC clustering in the SAM feature space to obtain uniform semantic superpixels.
- **ControlNet**: Image-level conditional guidance, which is unsuitable for precise spatial manipulation.
- Can be extended to scenarios such as video editing and 3D scene editing.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The concept of editable image elements is highly novel, elegantly unifying reconstruction and editing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The user study is well-designed and the ablation is comprehensive, but lacks large-scale quantitative evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem definitions and explanations of methodology, with intuitive diagrams.
- **Value**: ⭐⭐⭐⭐ — Provides a new paradigm for image editing, with massive potential, particularly in spatial editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ∞-Brush: Controllable Large Image Synthesis with Diffusion Models in Infinite Dimensions](inftybrush_controllable_large_image_synthesis_with_diffusion.md)
- [\[ECCV 2024\] FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis](fouriscale_a_frequency_perspective_on_training-free_high-resolution_image_synthe.md)
- [\[ECCV 2024\] Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)
- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)

</div>

<!-- RELATED:END -->
