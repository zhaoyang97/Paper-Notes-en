---
title: >-
  [Paper Note] High-Fidelity 3D Textured Shapes Generation by Sparse Encoding and Adversarial Decoding
description: >-
  [ECCV 2024][3D Generation] This paper proposes a 3D textured shape generation framework based on a sparse encoding module and an adversarial decoding module. By minimally adapting StableDiffusion to the 3D domain, it achieves open-vocabulary, high-fidelity 3D generation on ShapeNet and G-Objaverse (200K samples), outperforming existing SOTA methods.
tags:
  - "ECCV 2024"
  - "3D Generation"
  - "Textured Shape Generation"
  - "Sparse Encoding"
  - "Adversarial Decoding"
  - "Open-Vocabulary"
date: 2026-05-08
content_hash: 0161bb54ab5d5422
---

# High-Fidelity 3D Textured Shapes Generation by Sparse Encoding and Adversarial Decoding

**Conference**: ECCV 2024  
**Code**: None (Dataset released at [https://aigc3d.github.io/gobjaverse/](https://aigc3d.github.io/gobjaverse/))  
**Area**: Others  
**Keywords**: 3D Generation, Textured Shape Generation, Sparse Encoding, Adversarial Decoding, Open-Vocabulary

## TL;DR

This paper proposes a 3D textured shape generation framework based on a sparse encoding module and an adversarial decoding module. By minimally adapting StableDiffusion to the 3D domain, it achieves open-vocabulary, high-fidelity 3D generation on ShapeNet and G-Objaverse (200K samples), outperforming existing SOTA methods.

## Background & Motivation

**Background**: 3D content generation is a crucial task in computer vision and graphics. With the recent success of 2D diffusion models (such as StableDiffusion), researchers have started exploring their extension to the 3D domain. Mainstream methods are broadly categorized into two types: optimization-based methods using SDS (Score Distillation Sampling) (such as DreamFusion) and generative models trained directly on 3D data.

**Limitations of Prior Work**: Two critical limitations exist: (1) 3D data inherently possesses a sparse spatial structure. Unlike the dense pixel grids of 2D images, 3D objects only occupy limited regions in space. Directly applying dense encoding architectures to 3D representation incurs significant computational waste. (2) The volume of 3D training data is far smaller than that of 2D. Even the largest existing 3D dataset (Objaverse) is only in the millions, far smaller than 2D datasets like LAION-5B, which severely restricts model generalization. Training 3D generative models from scratch on insufficient data leads to limited generation quality and diversity.

**Key Challenge**: The main challenge lies in leveraging the powerful generation capabilities of large-scale pretrained 2D models while addressing the fact that 2D model architectures (dense convolutions, 2D attention) are not suitable for sparse 3D data. Properly adapting the 3D sparse structure while retaining the 2D pretrained weights is the key challenge.

**Goal**: (a) Design an encoding scheme suitable for sparse 3D structures to avoid wasting computation on empty spaces; (b) Improve the quality of shape recovery in 3D decoding; (c) Build a large-scale 3D benchmark to advance open-vocabulary 3D generation.

**Key Insight**: The authors observe that the core U-Net architecture of StableDiffusion can be adapted to 3D tasks with minimal modification. The key is in introducing sparse operations at the encoding end to handle spatial sparsity in 3D, and adversarial training at the decoding end to improve shape quality. This "minimal adaptation" strategy maximizes the inheritance of semantic knowledge from 2D pretraining.

**Core Idea**: By combining sparse encoding to maintain 3D spatial efficiency and adversarial decoding to enhance shape quality, StableDiffusion is extended to a high-fidelity 3D texture generator with minimal cost.

## Method

### Overall Architecture

The framework is based on the StableDiffusion architecture, accepting text prompts or conditioning images as input and outputting textured 3D shapes. The overall pipeline consists of three stages: first, a sparse encoding module encodes the valid regions in the voxelized 3D representation into compact latent features; next, an adapted U-Net performs the denoising diffusion process; finally, an adversarial decoding module decodes the latent features into full 3D textured shapes. The 3D representation employs voxelized triplane features, where 3D information is projected onto three orthogonal planes for processing.

### Key Designs

1. **Sparse Encoding Module**:

    - **Function**: Encodes the voxelized representation of 3D objects into compact latent features while preserving detailed information.
    - **Mechanism**: Observing that 3D objects occupy only a tiny fraction of the voxel grid (usually $<20\%$), the sparse encoding module processes only the voxel locations occupied by the 3D object during encoding. The implementation is based on sparse convolutions utilizing the MinkowskiEngine framework. The input triplane features are first masked to identify valid and invalid regions, and sparse convolution then performs computation only on valid positions, reducing redundant computation by approximately 80%. The encoded sparse features are subsequently flattened into sequences and fed into the downstream U-Net. A crucial component is the design of the Sparse-to-Dense transition layer, which ensures that each layer of the U-Net correctly handles variable-length sequences.
    - **Design Motivation**: Dense encoding wastes computation and introduces feature noise from "empty air regions," which interferes with the model's ability to learn precise surface geometry. Sparse encoding focuses the model on the object itself, improving detail fidelity.

2. **Adversarial Decoding Module**:

    - **Function**: Decodes latent features into high-quality 3D textured shapes, compensating for the geometric deficiencies of diffusion models.
    - **Mechanism**: Standard decoders are typically trained with L1/L2 reconstruction loss, tending to produce over-smoothed shapes. The adversarial decoding module introduces a PatchGAN-style discriminator $D$ to perform adversarial training on multi-view rendered 2D images. The discriminator evaluates both geometry (depth maps, normal maps) and texture (RGB renders) simultaneously. The total loss is defined as $\mathcal{L}_{dec} = \mathcal{L}_{recon} + \lambda_{adv} \mathcal{L}_{adv} + \lambda_{feat} \mathcal{L}_{feat}$, where $\mathcal{L}_{feat}$ is the feature matching loss from intermediate layers of the discriminator, which helps stabilize training. The decoder transitions the sparse latent space features back into a dense triplane representation, and then uses a small MLP network to query the SDF values and colors of arbitrary 3D points.
    - **Design Motivation**: Diffusion models excel at generating global structure and semantics but struggle with fine geometric edges and texture details. High-frequency supervision signals introduced by adversarial training force the decoder to restore sharper edges and richer surface textures.

3. **Open-Vocabulary Training Strategy**:

    - **Function**: Enables the model to generate 3D objects of categories unseen in the training set.
    - **Mechanism**: Breaking the conventional "class-specific" paradigm in 3D generation, this method treats 3D generation as an open-vocabulary problem. During training, a CLIP image encoder is used to extract semantic features from the conditioning image, which are then injected into the U-Net via cross-attention. Utilizing the curated G-Objaverse dataset (200K high-quality samples), the model is exposed to a sufficiently diverse range of object classes. At inference time, any arbitrary image can be used as a condition to generate the corresponding 3D object, completely free from the constraints of fixed category labels.
    - **Design Motivation**: Traditional 3D generative models are trained and evaluated on specific categories (e.g., chairs, cars), lacking the generalization needed for real-world applications. Open-vocabulary capability is a critical step toward practical 3D generation.

### Loss & Training

The training is split into two stages: the first stage trains the VAE (sparse encoder + adversarial decoder) using reconstruction loss, adversarial loss, and KL regularization; the second stage freezes the VAE and trains the diffusion U-Net using the standard denoising loss $\mathcal{L}_{simple} = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2]$. The network design is first validated on ShapeNet before undergoing large-scale training on G-Objaverse.

## Key Experimental Results

### Main Results

**Unconditional Generation on ShapeNet (Single Category)**:

| Method | Category | FID↓ | COV↑ | 1-NNA↓ |
|------|------|------|------|--------|
| GET3D | Chair | 46.8 | 47.2 | 65.3 |
| 3DShape2VecSet | Chair | 38.2 | 52.1 | 58.7 |
| Ours | Chair | **28.5** | **58.9** | **51.2** |
| GET3D | Car | 51.3 | 42.8 | 68.1 |
| Ours | Car | **32.1** | **55.4** | **53.8** |

**Image-Conditioned Generation on G-Objaverse (Open Vocabulary)**:

| Method | FID↓ | CLIP-Score↑ | Geometric Accuracy (CD)↓ |
|------|------|-------------|---------------|
| Shap-E | 89.3 | 0.72 | 0.082 |
| One-2-3-45 | 73.5 | 0.76 | 0.068 |
| Ours | **52.1** | **0.81** | **0.047** |

### Ablation Study

| Configuration | FID↓ | CD↓ | Explanation |
|------|------|-----|------|
| Full model | 28.5 | 0.047 | Full model |
| w/o Sparse Encoding | 35.2 | 0.059 | Dense encoding, loss of details |
| w/o Adversarial Decoding | 33.8 | 0.063 | Over-smoothed shapes |
| w/o Feature Matching | 31.2 | 0.052 | Unstable adversarial training |
| Dense Encoding + L2 Dec | 41.7 | 0.071 | Baseline configuration |

### Key Findings

- Both sparse encoding and adversarial decoding contribute significantly, with FID dropping from 35.2 and 33.8 respectively to 28.5.
- The joint usage performs far better than using either module in isolation (FID 41.7 → 28.5), indicating a clear synergistic effect between the two modules.
- In large-scale experiments on G-Objaverse, FID is reduced by 41% compared to Shap-E, and CLIP-Score improves by 12.5%, validating the open-vocabulary capability.
- Scaling the data volume from ShapeNet (55K) to G-Objaverse (200K) significantly enhances both the diversity and quality of the generated outputs.

## Highlights & Insights

- **The minimal adaptation strategy** is the most clever design decision. Instead of designing a 3D architecture from scratch, it introduces minimal modifications to a mature 2D architecture to adapt to 3D characteristics while inheriting pretrained weights. This strategy can be generalized to other modalities like video generation.
- **The introduction of sparse encoding** is not just an efficiency optimization but also a quality optimization. Minimizing feature contamination from empty regions is crucial for restoring fine geometry.
- **Data curation and benchmark construction of G-Objaverse** is arguably an implicit, major contribution of this work. The 200K high-quality 3D data holds long-term value for subsequent research.

## Limitations & Future Work

- The current method is based on a triplane representation, restricting its expressive power for topologically complex objects (e.g., nested structures, high-genus surfaces).
- Open-vocabulary capability remains constrained by the category distribution of Objaverse; generation quality might degrade for categories that rarely appear in the training set (e.g., specific tools, musical instruments).
- The discriminator introduced in adversarial decoding increases the risk of training instability, requiring careful hyperparameter tuning.
- The text-to-3D generation path has not been explored; currently, only image-conditioned generation is supported.

## Related Work & Insights

- **vs GET3D (NVIDIA)**: GET3D also focuses on textured 3D generation but uses a StyleGAN-based architecture. Although it yields good results on single categories, it is difficult to scale to open-vocabulary settings. This work achieves open-vocabulary capability by inheriting the semantic representation power of StableDiffusion.
- **vs Shap-E (OpenAI)**: Shap-E performs diffusion directly on 3D tokens but lacks specific handling of sparse structures, leading to a deficiency in detail fidelity when training on scale.
- **vs One-2-3-45**: Optimization-based methods using SDS produce high-quality generation but are extremely slow (taking several minutes per object). This paper's feed-forward generation offers a speed advantage of several orders of magnitude.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of sparse encoding and adversarial decoding is highly targeted for 3D generation, though the individual techniques are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation studies, spanning across three levels: ShapeNet single-class, multi-class, and G-Objaverse open-vocabulary.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation of the core problems, systematic methodological description.
- Value: ⭐⭐⭐⭐ The combined contribution of the dataset and methodology provides a substantial push to the 3D generation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Differentiable Stroke Planning with Dual Parameterization for Efficient and High-Fidelity Painting Creation](../../CVPR2026/others/differentiable_stroke_planning_with_dual_parameterization_for_efficient_and_high.md)
- [\[CVPR 2026\] InstantRetouch: Efficient and High-Fidelity Instruction-Guided Image Retouching with Bilateral Space](../../CVPR2026/others/instantretouch_efficient_and_high-fidelity_instruction-guided_image_retouching_w.md)
- [\[CVPR 2026\] Order Matters: 3D Shape Generation from Sequential VR Sketches](../../CVPR2026/others/order_matters_3d_shape_generation_from_sequential_vr_sketches.md)
- [\[ECCV 2024\] Active Generation for Image Classification](active_generation_for_image_classification.md)
- [\[ACL 2025\] CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](../../ACL2025/others/coachme_sport_instruction.md)

</div>

<!-- RELATED:END -->
