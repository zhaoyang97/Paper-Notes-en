---
title: >-
  [Paper Note] MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion
description: >-
  [ECCV 2024][Image Generation] The diffusion model is applied to skeleton representation learning for the first time, proposing the Masked Conditional Diffusion (MacDiff) framework. It extracts representations of the masked skeleton via a semantic encoder to guide a conditional diffusion decoder for denoising, thereby unifying the discriminative and generative modeling of skeletons.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: c190f8a173fc4dfc
---

# MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion

**Conference**: ECCV 2024  
**arXiv**: [2409.10473](https://arxiv.org/abs/2409.10473)  
**Area**: Image Generation

## TL;DR

The diffusion model is applied to skeleton representation learning for the first time, proposing the Masked Conditional Diffusion (MacDiff) framework. It extracts representations of the masked skeleton via a semantic encoder to guide a conditional diffusion decoder for denoising, thereby unifying the discriminative and generative modeling of skeletons.

## Background & Motivation

Self-supervised learning of skeleton data mainly falls into two paradigms: contrastive learning and reconstruction. Contrastive learning methods rely on the construction of positive and negative sample pairs, suffering from false negative problems and only learning discriminative information, which limits their generalization ability. Reconstruction methods (such as MAE) focus excessively on low-level signal reconstruction, and the learned representations contain excessive amounts of information irrelevant to high-level semantics.

Although diffusion models have achieved great success in the field of image generation, their representation learning capability has not been fully explored, especially in the face of the spatial sparsity and temporal redundancy of skeleton data. Directly using diffusion models to predict noise does not explicitly learn meaningful discriminative representations. Therefore, how to unleash the potential of diffusion models in skeleton representation learning while maintaining generative capabilities has become a direction worth exploring.

## Method

### Overall Architecture

MacDiff consists of two core components:
1. **Semantic Encoder**: Receives the masked skeleton sequence and extracts high-level compact representations.
2. **Denoising Decoder**: Performs conditional diffusion denoising conditioned on the representations output by the encoder.

The input skeleton sequence is first split into patches along the temporal dimension and embedded as tokens. A high mask ratio of 90% random masking is applied to the encoder input, which both introduces an information bottleneck to remove redundancy and accelerates training. After the encoder outputs local representations, the global representation is obtained through pooling.

### Key Designs

1. **Patchify + Random Masking**: Splits the T₀×V×3 skeleton data into patches along the temporal dimension, and a high mask ratio of 90% is used to construct a compact information bottleneck.
2. **AdaLN Conditional Injection**: Replaces standard LN with Adaptive Layer Norm in the decoder, injecting encoder representations into the denoising process via scale and shift operations.
3. **Local-Global Representation Fusion**: Fills unmasked positions with local representations and masked positions with global representations, addressing the over-smoothing issue and preserving token diversity.
4. **Inverse-cosine Noise Schedule**: Pulls the noise levels of all timesteps toward a moderate level, which outperforms the commonly used cosine or linear schedules.
5. **Diffusion Data Augmentation**: Utilizes the pre-trained diffusion decoder to generate label-preserving training data, which significantly improves fine-tuning performance when labeled data is scarce.

### Loss & Training

Standard conditional diffusion loss using $\epsilon$-prediction:

$$\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon}\left[\|\epsilon - \mathcal{D}(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t, \mathcal{E}(\mathcal{M}(x_0)))\|^2\right]$$

Theoretical analysis shows that the generative objective of MacDiff is equivalent to a contrastive learning objective (aligning the mutual information $I(Z; X_t)$ of the masked view and the noisy view) plus a complementary reconstruction objective ($I(X; Z|X_t)$), where the latter requires the representation to contain more task-relevant information missed by contrastive learning.

## Key Experimental Results

### Main Results

| Method | Type | NTU60 xsub | NTU60 xview | NTU120 xsub | NTU120 xset | PKU I |
|------|------|------------|-------------|-------------|-------------|-------|
| 3s-CrosSCLR | Contrastive Learning | 77.8 | 83.4 | 67.9 | 66.7 | 84.9 |
| 3s-AimCLR | Contrastive Learning | 78.9 | 83.8 | 68.2 | 68.8 | 87.4 |
| 3s-ActCLR | Contrastive Learning | 84.3 | 88.8 | 74.3 | 75.7 | - |
| MAMP | Reconstruction (J) | 84.9 | 89.1 | 78.6 | 79.1 | 92.2 |
| PCM3 | Multi-task (J) | 83.9 | 90.4 | 76.5 | 77.5 | - |
| **MacDiff** | **Generative (J)** | **86.4** | **91.0** | **79.4** | **80.2** | **92.8** |

In linear evaluation, MacDiff surpasses all three-stream ensemble methods and previous MAE methods using only a single stream (Joint).

### Ablation Study

| Configuration | NTU60 xsub | NTU60 xview |
|------|------------|-------------|
| SkeletonMAE (Transformer) | 88.5 | 94.7 |
| MAMP (Transformer) | 93.1 | 97.5 |
| MotionBERT (DSTformer) | 93.0 | 97.2 |
| UPS (Transformer, Supervised) | 92.6 | 97.0 |
| **MacDiff (Transformer)** | **92.7** | **97.3** |

Under the supervised fine-tuning protocol, MacDiff achieves performance comparable to the fully supervised unified models UPS and MotionBERT, validating that the representations learned by the diffusion model are highly versatile.

### Semi-supervised Fine-tuning and Data Augmentation

| Method | NTU60 xsub 1% | NTU60 xsub 10% | NTU60 xview 1% | NTU60 xview 10% |
|------|---------------|----------------|----------------|-----------------|
| SkeletonMAE | 54.4 | 80.6 | 54.6 | 83.5 |
| MAMP | 66.0 | 88.0 | 68.7 | 91.5 |
| MacDiff (w/o Augmentation) | 65.6 | 88.2 | 77.3 | 92.5 |
| **MacDiff (w/ Augmentation)** | **72.0** | **89.2** | **79.2** | **93.1** |

Diffusion data augmentation yields a significant improvement of 6.4% (65.6 → 72.0) under 1% labeled data, outperforming MAMP by 6.0%.

### Key Findings

- A 90% mask ratio is the optimal choice; 50%/80% ratios yield 82.7/83.8, respectively, whereas 0% (no masking) yields only 79.3.
- The inverse-cosine noise schedule significantly outperforms cosine and linear schedules.
- The optimal ratio of augmented data to real data decreases as the amount of labeled data increases: 2.0, 0.5, and 0.25 for 1%, 2%, and 10% labeled data, respectively.
- In motion reconstruction, the MPJPE of MacDiff (0.033) is only 1/6 of that of SkeletonMAE (0.191).
- The local-global representation fusion effectively mitigates the over-smoothing issue of Transformers.

## Highlights & Insights

1. **First to demonstrate that diffusion models can serve as effective skeleton representation learners**, breaking the stereotype that generative models lack sufficient representation capability.
2. **Solid theoretical analysis**: Proves from the perspective of mutual information that the MacDiff objective encapsulates contrastive learning and additionally preserves more task-relevant semantics, offering stronger theoretical guarantees for downstream performance.
3. **Unified framework**: The same model simultaneously supports both action recognition (discriminative) and skeleton generation/data augmentation (generative), avoiding the waste of pre-trained components.
4. **Clever choice of mask ratio**: The high mask ratio of 90% constrains the representation dimensionality while significantly reducing computational overhead.

## Limitations & Future Work

- Currently, evaluation is limited to skeleton data; whether it can generalize to more complex sequential data (e.g., video, point clouds) remains to be explored.
- Diffusion sampling still requires multi-step iterations of DDIM, which makes generation slower compared to MAE-based methods.
- The fusion of multi-modal inputs (e.g., RGB + skeleton) has not been discussed.
- The decoupled analysis of the contributions of contrastive learning and generative learning is not sufficiently detailed.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to apply diffusion models to self-supervised skeleton representation learning with a novel theoretical analysis perspective.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Rigorous and complete theoretical analysis of mutual information, with a finely designed information bottleneck.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated across three major datasets with multiple evaluation protocols and extensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, closely integrating theory and experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [\[ECCV 2024\] Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [\[ECCV 2024\] Memory-Efficient Fine-Tuning for Quantized Diffusion Model](memory-efficient_fine-tuning_for_quantized_diffusion_model.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)

</div>

<!-- RELATED:END -->
