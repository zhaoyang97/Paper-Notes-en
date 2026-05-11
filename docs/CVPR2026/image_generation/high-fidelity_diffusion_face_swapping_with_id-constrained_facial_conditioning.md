---
title: >-
  [Paper Note] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning
description: >-
  [CVPR 2026][Image Generation][Face Swapping] This paper proposes an identity-constrained attribute tuning framework for diffusion-based face swapping: the method first constrains the identity solution space…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Face Swapping"
  - "Diffusion Models"
  - "Identity Constraint"
  - "Condition Decoupling"
  - "Multi-Stage Training"
date: 2026-05-08
content_hash: 276af68495117090
---

# High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning

**Conference**: CVPR 2026
**arXiv**: [2503.22179](https://arxiv.org/abs/2503.22179)
**Code**: N/A
**Area**: Diffusion Models / Image Generation
**Keywords**: Face Swapping, Diffusion Models, Identity Constraint, Condition Decoupling, Multi-Stage Training

## TL;DR

This paper proposes an identity-constrained attribute tuning framework for diffusion-based face swapping: the method first constrains the identity solution space, then injects attribute conditions, and finally performs end-to-end refinement with identity and adversarial losses. Combined with a decoupled condition injection design, it achieves state-of-the-art FID (3.61) and identity retrieval accuracy (97.9% Top-1) on FFHQ.

## Background & Motivation

Face swapping transfers the identity of a source face onto a target face while preserving the target's expression, pose, and other attributes. It has important applications in film production, gaming, and digital twins.

Traditional GAN-based methods (SimSwap, E4S, InfoSwap) are constrained by inherent GAN image quality limitations and mode collapse issues. Diffusion models have emerged as a promising alternative due to their superior generative capability, but existing diffusion-based methods (DiffFace, DiffSwap, REFace) face two core challenges:

**Identity-Attribute Priority Conflict**: In face swapping, identity preservation should take priority over attribute consistency — the result must first resemble the source face, and only secondarily align with the target's expression and pose. Existing methods typically inject all conditions jointly without priority control.

**Identity-Attribute Condition Conflict**: The identity condition drives the output toward the source face, while the attribute condition drives it toward the target face — these two objectives are directionally opposed during training (see Fig. 3 of the paper), and joint training tends to fall into suboptimal solutions.

## Method

### Overall Architecture

The framework is built upon a Stable Diffusion 1.5 conditional inpainting pipeline. Given a source face image and a target face image, the model outputs a target image with the source identity transferred. The core mechanism follows a **constrain identity → tune attributes → end-to-end refinement** paradigm, implemented via three sequential training stages that progressively narrow the solution space.

### Key Designs

1. **Decoupled Facial Condition Injection**:

    - **Data-level decoupling**: Unlike prior methods that apply augmentations to a single image to generate condition pairs (which risks leaking both identity and attribute information), this work uses paired images of **the same person with different attributes**, fundamentally decoupling identity and attribute features.
    - **Dual-path extraction**: The identity path uses ArcFace to extract a $d$-dimensional feature, which is expanded into an $n \times d$ token sequence $c_{\text{face}}$ via an MLP. DINOv2 then extracts spatial detail features $c_{\text{dino}}$, fused via cross-attention:
    $c_{\text{id}} = c_{\text{face}} + \lambda_{\text{id}} \cdot \text{Attention}(c_{\text{face}}, c_{\text{dino}}, c_{\text{dino}})$
    The attribute path uses SimSwap's 3-layer downsampling network to extract expression features $c_{\text{attr}}$ from the target face.
    - **Attention-based fusion**: Identity features serve as queries and attribute features as key-values, with a fusion factor $\lambda_{\text{fuse}}$ controlling attribute injection intensity:
    $c_{\text{fuse}} = c_{\text{id}} + \lambda_{\text{fuse}} \cdot \text{Attention}(c_{\text{id}}, c_{\text{attr}}, c_{\text{attr}})$
    Setting $\lambda_{\text{fuse}} = 0$ reduces the model to pure identity conditioning. The fused features are injected into UNet cross-attention layers via a GLIGEN adapter.

2. **Identity-Constrained Multi-Stage Training**:

    - **Stage 1 — Identity-Guided Tuning**: The UNet input layer is extended to accept the noisy latent $x_t$, an inpainting mask $m$, and background context $(1-m) \odot x_t$. Only identity conditions are used ($\lambda_{\text{fuse}} = 0$), with no attribute constraints, contracting the solution space to the identity-consistent output region.
    - **Stage 2 — Attribute Tuning**: Attribute conditions are enabled ($\lambda_{\text{fuse}} = 1$), guiding the model to align with the target expression and pose while respecting identity constraints. Two key implementation details: (a) the fusion module output layer is zero-initialized to prevent attribute injection from disrupting the learned identity features; (b) the identity spatial augmentation factor $\lambda_{\text{id}}$ is reduced to 0.2 to prevent overly strong identity conditioning from neglecting attributes.
    - **Stage 3 — End-to-End Refinement**: The 50-step DDIM sampling process is treated as a cascaded end-to-end generative model, with identity and adversarial losses applied to the sampled outputs:
    $\mathcal{L} = \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{\text{id}} \mathcal{L}_{\text{id}}$
    To address memory overhead from backpropagation, only $k$ steps randomly sampled from the 50 DDIM steps are used for gradient computation per mini-batch.

3. **Key Distinctions from GAN-Based Methods**:

    - DiffSwap/DiffFace directly add an ID loss to the noise prediction loss, which relaxes the ELBO theoretical bound and degrades generation quality.
    - REFace applies ID loss to multi-step DDIM outputs, but incurs high computational cost.
    - This work isolates ID supervision to the dedicated Stage 3, avoiding interference with the diffusion training ELBO, while an SNGAN discriminator further enhances realism.

### Loss & Training

- **Stages 1–2**: Standard diffusion noise prediction MSE loss $\mathcal{L}_{\text{diff}} = \sum_t \lambda_t \|\epsilon_\theta(x_t; t) - \epsilon\|_2^2$
- **Stage 3**: SNGAN hinge loss $\mathcal{L}_{\text{adv}}$ + ArcFace identity loss $\mathcal{L}_{\text{id}}$, computed over randomly sampled $k$ steps
- Training data: 4.5 million internet-sourced paired face images (same identity, different attributes), with text descriptions annotated via BLIP-2
- Stage 3 uses randomly paired face data filtered from LAION-5B
- Output resolution: $512 \times 512$

## Key Experimental Results

### Main Results

Evaluated on 1,000 pairs from the FFHQ validation set:

| Method | FID↓ | Pose↓ | Expr.↓ | ID Top-1↑(%) | ID Top-5↑(%) |
|--------|------|-------|--------|--------------|--------------|
| SimSwap (GAN) | 13.74 | 2.62 | **0.95** | 93.37 | 97.29 |
| E4S (GAN) | 12.22 | 4.55 | 1.32 | 77.80 | 87.40 |
| InfoSwap (GAN) | 4.26 | 3.26 | 1.00 | 92.82 | 97.69 |
| DiffFace | 8.82 | 3.76 | 1.31 | 91.50 | 97.50 |
| DiffSwap | 5.80 | **2.43** | 1.01 | 67.00 | 81.90 |
| REFace | 5.62 | 3.75 | 1.04 | 89.10 | 96.10 |
| **Ours** | **3.61** | 3.69 | 0.97 | **97.90** | **99.70** |

The proposed method achieves a substantially lower FID (3.61 vs. the second-best 4.26) and significantly outperforms all baselines on identity retrieval accuracy (97.9% vs. 93.4%), while matching the best result on expression distance.

### Ablation Study

| Configuration | Key Observation | Explanation |
|---------------|----------------|-------------|
| Non-decoupled condition injection | Weak identity preservation; overfitting to attribute following | Same-image augmentation causes ID/attribute leakage |
| Stage 1 only | Good identity; poor expression/pose alignment | No attribute condition guidance |
| Stage 1+2 | Good identity; improved expression, pose, and lighting | Attribute tuning is effective; SimSwap encoder also captures lighting |
| Stage 1+2+3 | Significant gains in identity similarity and realism | End-to-end refinement with GAN+ID loss is necessary |

### Key Findings

- **Decoupled injection is central**: Without decoupling, the model overfits to attribute following, leading to a substantial drop in identity preservation.
- **Multi-stage training is effective**: Each stage yields clear and measurable improvements; Stage 2 additionally produces unexpected benefits in lighting alignment and face shape adjustment.
- **End-to-end refinement improves realism**: Stage 3's ID loss + GAN loss significantly enhances identity similarity and generation fidelity.
- **Unique advantages of diffusion models**: The pretrained backbone provides out-of-the-box generalization to stylized images (e.g., oil paintings, cartoons, comprising <1% of training data).
- **User study** (39 participants): The proposed method received 57.1% of votes on fidelity (far ahead of the second-place at 15.3%), and ranked first on attribute consistency with 33.2%.

## Highlights & Insights

- **The identity-first intuition is elegant and principled**: The notion of "identity first, then alignment" is formalized as constrained optimization — first contracting the solution space to the identity-consistent region, then optimizing attribute alignment within that subspace, avoiding mutual interference between the two conditions in the full space.
- **Staged loss design avoids theoretical contradictions**: Isolating ID supervision to the dedicated Stage 3 avoids polluting the ELBO in Stages 1–2, representing a more principled approach than DiffSwap/DiffFace.
- **Stage 2's zero-initialization + weakened identity factor** is a subtle yet effective design choice that prevents catastrophic forgetting and condition imbalance.
- **Unexpected benefit of the SimSwap encoder**: Beyond encoding expression and pose, the encoder implicitly captures lighting conditions — a noteworthy byproduct.

## Limitations & Future Work

- Built on SD 1.5 ($512 \times 512$), the method is constrained by the base model's resolution and does not adopt more modern architectures such as DiT or FLUX.
- The pose distance (3.69) is inferior to DiffSwap (2.43), indicating that identity constraints partially sacrifice pose alignment.
- Stage 3 requires 50-step DDIM sampling with backpropagation, leading to high training costs (partially mitigated by random $k$-step gradient computation).
- The 4.5 million training images are sourced from the internet; data quality and privacy concerns are not discussed.
- Systematic evaluation on challenging scenarios such as cross-race and cross-age face swapping is absent.

## Related Work & Insights

- The concept of **condition priority** can generalize to other multi-condition generation tasks (e.g., determining priority when simultaneously controlling identity, style, and layout).
- The **staged condition injection** paradigm is applicable to any diffusion model fine-tuning scenario with condition conflicts.
- End-to-end DDIM refinement with random-step gradient computation represents a general post-training enhancement strategy applicable to other generative quality optimization tasks.
- The reuse of the SimSwap attribute encoder within a diffusion framework demonstrates that modules from the GAN era retain transferable value.

## Rating

- Novelty: ⭐⭐⭐⭐ The identity-constrained multi-stage training paradigm and decoupled condition injection design are novel with clear intuition.
- Experimental Thoroughness: ⭐⭐⭐⭐ Six comparison methods, quantitative evaluation, user study, comprehensive ablation, and visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; Fig. 2/3 provides highly intuitive visualization of the condition conflict.
- Value: ⭐⭐⭐⭐ The ideas of multi-condition decoupling and priority-based training are broadly applicable beyond the specific task of face swapping.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)
- [\[CVPR 2026\] APPLE: Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping](apple_attribute-preserving_pseudo-labeling_for_diffusion-based_face_swapping.md)
- [\[AAAI 2026\] Realistic Face Reconstruction from Facial Embeddings via Diffusion Models](../../AAAI2026/image_generation/realistic_face_reconstruction_from_facial_embeddings_via_diffusion_models.md)
- [\[CVPR 2026\] CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment](cognitioncapturerpro_towards_highfidelity_visual_d.md)
- [\[CVPR 2026\] ExpressEdit: Fast Editing of Stylized Facial Expressions with Diffusion Models in Photoshop](expressedit_fast_editing_of_stylized_facial_expressions_with_diffusion_models_in.md)

</div>

<!-- RELATED:END -->
