---
title: >-
  [Paper Note] Infinite-ID: Identity-preserved Personalization via ID-semantics Decoupling Paradigm
description: >-
  [ECCV 2024][Image Generation][Identity-preserved Generation] Infinite-ID is proposed to separate identity information and text semantic information via an ID-semantic decoupling paradigm. In the training phase, text cross-attention is disabled to focus on learning identity embeddings. In the inference phase, the two streams of information are merged via a mixed attention mechanism and an AdaIN-mean operation, achieving both high-fidelity identity preservation and semantic con…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Identity-preserved Generation"
  - "Personalized Text-to-Image"
  - "Style Control"
  - "Attention Mechanism"
  - "SDXL"
date: 2026-05-08
content_hash: a933a369092a4bfc
---

# Infinite-ID: Identity-preserved Personalization via ID-semantics Decoupling Paradigm

**Conference**: ECCV 2024  
**arXiv**: [2403.11781](https://arxiv.org/abs/2403.11781)  
**Code**: [Project Page](https://infinite-id.github.io/)  
**Area**: Image Generation  
**Keywords**: Identity-preserved Generation, Personalized Text-to-Image, Style Control, Attention Mechanism, SDXL

## TL;DR

Infinite-ID is proposed to separate identity information and text semantic information via an ID-semantic decoupling paradigm. In the training phase, text cross-attention is disabled to focus on learning identity embeddings. In the inference phase, the two streams of information are merged via a mixed attention mechanism and an AdaIN-mean operation, achieving both high-fidelity identity preservation and semantic consistency with a single reference image.

## Background & Motivation

Identity-preserved personalization in text-to-image synthesis aims to generate new images (with new scenes, actions, and styles) while maintaining a specific facial identity based on one or more reference photos. This has enormous potential in commercial applications such as AI portraits and virtual try-ons.

**Key Challenge**: There is a severe trade-off between ID fidelity and semantic consistency, which stems from the entanglement of image and text information.

Two typical perspectives of existing tuning-free methods:

| Method Type | Representative Methods | Fusion Mode | Advantages | Limitations |
|----------|----------|----------|------|------|
| Text-space Fusion | PhotoMaker | Merging ID information in the text encoder space | Good semantic consistency | Image features are compressed, resulting in low identity fidelity |
| U-Net Space Fusion | IP-Adapter | Injecting ID information via extra cross-attention | Stronger ID information | Bias towards the image branch during training, leading to poor semantic consistency |

Both approaches entangle image and text information. The core idea of Infinite-ID is: **completely decouple ID and semantics during training**, and then re-fuse them through carefully designed mechanisms during inference.

## Method

### Overall Architecture

Infinite-ID is built upon SDXL and consists of three key components:

1. **Identity-enhanced Training**: Completely decouples ID and text during the training phase.
2. **Mixed Attention**: Fuses ID and semantic information during the inference phase.
3. **AdaIN-mean Operation**: Controls the style of generated images during the inference phase.

### Key Designs

#### 1. Identity-enhanced Training (Core of Training Phase)

Unlike conventional methods that train with text-image pairs, Infinite-ID adopts a brand-new strategy:

- **Exclude text prompt inputs** and disable the original text cross-attention module in the U-Net.
- Training pairs consist of **different photos of the same person** (different angles, expressions) to promote more comprehensive identity learning.
- Only target training on the **Face Mapper, CLIP Mapper, and image cross-attention modules**, while freezing the parameters of the diffusion model.

The core benefit of this approach is that the image branch is not disturbed by text signals during training, allowing it to **completely and faithfully** learn how to represent the identity information of the reference image.

The training loss is simplified to a pure identity-conditioned diffusion loss:

$$L_{\text{diffusion}} = E_{z_t, t, c_{id}, \epsilon}\left[\|\epsilon - \epsilon_\theta(z_t, t, c_{id})\|_2^2\right]$$

#### 2. Face Embeddings Extractor

A dual-feature extraction strategy is adopted to complementarily capture identity information:

**(a) CLIP Image Encoder**:
- Use OpenCLIP ViT-H/14.
- Extract the last hidden states (sequence embeddings of $N=257$ tokens).
- Project them to the feature dimension of the diffusion model via CLIP Mapper.
- Mainly capture the **structural information** of the face.

**(b) Face Recognition Backbone**:
- Use the ArcFace backbone.
- Extract global image embeddings (512-dimensional).
- Align the dimensions via Face Mapper.
- Mainly capture **facial feature details**.

The final identity embedding:
$$c_{id} = \text{Concat}(M_{\text{clip}}(E_{\text{clip}}(FA(x))), M_{\text{face}}(E_{\text{face}}(FA(x))))$$

where $FA(\cdot)$ represents the face alignment module.

#### 3. Mixed Attention Mechanism (Core of Inference Phase)

During inference, both identity information and text semantics need to be utilized simultaneously. The mixed attention achieves fusion in the self-attention layer:

$$\text{Attn}_{\text{mix}}(Q, K, V) = \text{Attn}(Q, \hat{K}, \hat{V})$$

where:
$$\hat{K} = \text{Concat}(K_{id}, K_t), \quad \hat{V} = \text{Concat}(V_{id}, V_t)$$

- $K_{id}, V_{id}$: From the identity stream (spatial feature projections generated by the image cross-attention module).
- $K_t, V_t$: From the text stream (self-attention features obtained by original SDXL denoising with only the text prompt).

This design allows identity features to naturally compete and fuse with semantic features in the attention computation, with different resolution layers automatically balancing the two streams of information.

#### 4. Cross-Attention Merging

Semantic control is further reinforced in the cross-attention layer:

$$\text{Attn}_{\text{cross}} = \text{Attn}(Q, K'_{id}, V'_{id}) + \text{Attn}(Q, K'_t, V'_t)$$

The outputs of the image cross-attention and text cross-attention are directly added, preserving information from both paths.

#### 5. AdaIN-mean Style Fusion

To achieve style control (e.g., anime style, sketch style), an Adaptive Instance Normalization Mean (AdaIN-mean) operation is introduced:

$$\text{AdaIN-m}(x, y) = x - \mu(x) + \mu(y)$$

where $\mu(x) \in \mathbb{R}^{d_k}$ is the mean of features across pixels.

It is applied to the ID features in the mixed attention and cross-attention:

$$K_{id} = \text{AdaIN-m}(K_{id}, K_t), \quad V_{id} = \text{AdaIN-m}(V_{id}, V_t)$$

**Why use AdaIN-mean instead of standard AdaIN?** Standard AdaIN aligns both the mean and variance, which alters the distribution range of ID features and consequently reduces identity fidelity. Aligning only the mean preserves the intrinsic distribution structure of the ID features.

### Loss & Training

**Training Details:**
- Base Model: SDXL
- Image Encoders: OpenCLIP ViT-H/14 + ArcFace
- Attach an additional image cross-attention module to each of the 70 cross-attention layers in SDXL.
- 16 x A100 GPUs, batch size = 4 per GPU, trained for 1 million steps in total.
- AdamW optimizer, lr=1e-4, weight decay=0.01.
- Inference: DDIM Sampler, 30 steps, guidance scale = 5.0.
- Training Data: LAION-2B + LAION-Face + web images.

## Key Experimental Results

### Main Results

Quantitative comparison with tuning-free methods:

| Method | CLIP-T ↑ | CLIP-I ↑ | $M_{\text{FaceNet}}$ ↑ |
|------|:---:|:---:|:---:|
| FastComposer | 0.292 | 0.887 | 0.556 |
| IP-Adapter | 0.274 | 0.905 | 0.474 |
| IP-Adapter-Face | 0.313 | 0.919 | 0.513 |
| PhotoMaker | **0.343** | 0.814 | 0.502 |
| **Infinite-ID** | 0.340 | 0.913 | **0.689** |

Key observations:
- Infinite-ID **substantially leads** in FaceNet identity similarity (0.689 vs. the second-best 0.556), representing a 23.9% gain.
- CLIP-T is on par with PhotoMaker (0.340 vs. 0.343), demonstrating that semantic consistency is unaffected.
- CLIP-I is 0.913, which is close to IP-Adapter-Face's 0.919, while the FaceNet score is substantially higher.

### Ablation Study

Contribution of each component:

| Variant | CLIP-T ↑ | CLIP-I ↑ | $M_{\text{FaceNet}}$ ↑ |
|------|:---:|:---:|:---:|
| **Infinite-ID (Full)** | **0.340** | 0.913 | **0.689** |
| w/o Identity-enhanced Training | 0.329 | 0.891 | 0.593 |
| w/o Mixed Attention | 0.331 | 0.905 | **0.700** |
| Mixed Attn $\rightarrow$ Mutual Attn | 0.316 | 0.808 | 0.398 |

AdaIN-mean ablation:

| Variant | Identity Fidelity | Style Consistency |
|------|:---:|:---:|
| w/o AdaIN-mean | ★★★★★ | ✗ Cannot achieve stylization |
| AdaIN-mean $\rightarrow$ AdaIN | ★★★☆☆ | ✓ Achievable but ID fidelity decreases |
| **AdaIN-mean** | **★★★★☆** | **✓ Stylization + Identity preservation** |

### Key Findings

1. **Identity-enhanced training improves FaceNet by 16.2%** (0.593 $\rightarrow$ 0.689): Validates the efficacy of the decoupling strategy to exclude text interference during training.
2. **Mixed Attention outperforms Mutual Attention**: Mutual Attention causes the FaceNet score to plunge to 0.398, indicating that simply replacing key/value pairs destroys identity information.
3. **AdaIN-mean is a superior alternative to standard AdaIN**: Aligning only the mean preserves the intrinsic structure of the ID embeddings, whereas standard AdaIN's variance alignment distorts identity features.
4. **Removing Mixed Attention actually yields a higher FaceNet score (0.700)**: This suggests that Mixed Attention incurs a slight cost in ID fidelity while improving semantic consistency, making it a better overall trade-off.
5. **Stylization capability**: Under stylized prompts, IP-Adapter still generates images with the style of the reference image, indicating that its text-image space is distorted. Infinite-ID avoids this issue via decoupled training.

## Highlights & Insights

1. The paradigm design of **decoupling during training and fusing during inference** is clear and effective: Disabling text cross-attention to train the image branch not only purifies ID learning but also leaves flexible room for subsequent fusion designs.
2. **Clever design of AdaIN-mean**: By aligning only the mean rather than the variance, it maximizes the preservation of the distribution of identity features while securing style control capability.
3. **Complementarity of dual features**: CLIP captures structure while ArcFace captures facial features; concatenating the two provides a comprehensive identity representation.
4. **Sufficient training scale**: 1 million training steps on LAION-scale datasets ensure the model's generalization capabilities.

## Limitations & Future Work

1. **No support for multi-subject personalization**: The current framework handles only a single identity and cannot preserve multiple distinct identities simultaneously.
2. **Small-face artifacts**: When the face occupies only a small area of the image, artifacts may occur (a limitation inherited from the underlying diffusion model).
3. **High training cost**: Training 1 million steps on 16 x A100 GPUs makes replication challenging for researchers with limited resources.
4. **Dependence on face alignment**: Features can only be extracted after face detection and alignment, which is less robust under extreme poses or occlusions.
5. Extrapolating the ID-semantic decoupling paradigm to other personalization scenarios (such as objects, pets, etc.) is worth exploring.

## Related Work & Insights

- **PhotoMaker**: Fuses ID embeddings in the text space, achieving good semantic consistency but insufficient ID fidelity.
- **IP-Adapter / IP-Adapter-Face**: Inject ID information into U-Net; while the ID is strong, the semantics are disrupted.
- **MasaCtrl**: Proposes Mutual Self-Attention for consistent editing; Infinite-ID demonstrates that Mixed Attention is superior in this scenario.
- **StyleAligned**: Uses AdaIN + attention sharing to realize style alignment, inspiring the design of AdaIN-mean.
- **Insights**: The decoupling strategy during training may also serve as a reference for other multi-conditioned generation tasks.

## Rating

- **Novelty**: ★★★★☆ — The paradigm design of decoupled training + mixed attention fusion is novel and effective.
- **Experimental Thoroughness**: ★★★★★ — Quantitative comparisons, ablations, and reference/stylized photo generation are comprehensively analyzed.
- **Writing Quality**: ★★★★☆ — The methodology description is clear with plenty of illustrative content.
- **Value**: ★★★★☆ — Commercial-grade identity-preserved generation, though training cost poses a bottleneck.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](../../ICCV2025/image_generation/dynamicid_zero-shot_multi-id_image_personalization_with_flexible_facial_editabil.md)
- [\[ECCV 2024\] MagicEraser: Erasing Any Objects via Semantics-Aware Control](magiceraser_erasing_any_objects_via_semantics-aware_control.md)
- [\[ECCV 2024\] Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement](unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)
- [\[ICLR 2026\] WithAnyone: Toward Controllable and ID Consistent Image Generation](../../ICLR2026/image_generation/withanyone_toward_controllable_and_id_consistent_image_generation.md)
- [\[ECCV 2024\] Generating 3D House Wireframes with Semantics](generating_3d_house_wireframes_with_semantics.md)

</div>

<!-- RELATED:END -->
