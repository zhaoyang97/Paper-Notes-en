---
title: >-
  [Paper Note] LCA: Large-scale Codec Avatars - The Unreasonable Effectiveness of Large-scale Avatar Pretraining
description: >-
  [CVPR 2026][Human Understanding][3D avatars] LCA is the first work to apply the large-scale pretraining/post-training paradigm to 3D avatar modeling: it pretrains on 1 million in-the-wild videos to acquire broad appearance and geometry priors, then post-trains on high-quality multi-view studio data to enhance fine-grained expression fidelity, effectively breaking the inherent trade-off between generalizability and fidelity.
tags:
  - CVPR 2026
  - Human Understanding
  - 3D avatars
  - large-scale pretraining
  - feed-forward generation
  - Gaussian splatting
  - expression control
date: 2026-05-08
content_hash: fb4339d9decb4e49
---

# LCA: Large-scale Codec Avatars - The Unreasonable Effectiveness of Large-scale Avatar Pretraining

**Conference**: CVPR 2026
**arXiv**: [2604.02320](https://arxiv.org/abs/2604.02320)
**Code**: [https://junxuan-li.github.io/lca](https://junxuan-li.github.io/lca)
**Area**: Human Understanding / 3D Vision
**Keywords**: 3D avatars, large-scale pretraining, feed-forward generation, Gaussian splatting, expression control

## TL;DR

LCA is the first work to apply the large-scale pretraining/post-training paradigm to 3D avatar modeling: it pretrains on 1 million in-the-wild videos to acquire broad appearance and geometry priors, then post-trains on high-quality multi-view studio data to enhance fine-grained expression fidelity, effectively breaking the inherent trade-off between generalizability and fidelity.

## Background & Motivation

High-quality 3D avatar modeling faces a fundamental trade-off: studio data enables high-fidelity avatars but generalizes poorly (only to captured subjects); in-the-wild data generalizes to more identities but yields lower quality due to 3D ambiguity and distortion.

**Core Insight**: Inspired by LLMs and visual foundation models—large-scale pretraining learns general priors, while a small amount of high-quality post-training data aligns the model to the target task. This work is the first to demonstrate that this paradigm is equally effective in the 3D avatar domain.

## Method

### Overall Architecture

A two-branch architecture: reference image tokens + template body mesh tokens → large Transformer fusion → canonical MLP outputting Gaussian attributes + correction MLP outputting offsets under driving signals → LBS transformation to target pose → 3DGS rendering. Pretraining is conducted on 1M in-the-wild videos; post-training is conducted on multi-view studio data covering thousands of identities.

### Key Designs

1. **Scalable Dual-Branch Architecture**:

    - Function: Jointly supports training on both studio and in-the-wild data.
    - Mechanism: Image tokens are derived from a general-purpose visual encoder; geometry tokens are derived from a canonicalized body template mesh. The Transformer backbone employs a hybrid attention scheme (alternating global attention and per-image self-attention) to handle a variable number of input images. The canonical branch outputs canonical Gaussian attributes; the correction branch outputs attribute offsets conditioned on driving signals.
    - Design Motivation: Eliminates the need for high-quality conditioning data (e.g., geometry and texture maps), enabling seamless switching between different data sources.

2. **Pretraining → Post-training Paradigm**:

    - Function: Achieves the optimal balance between generalizability and fidelity.
    - Mechanism: The pretraining stage learns broad priors over human appearance and geometry from 1M in-the-wild videos. The post-training stage specializes on multi-view studio data to enhance the granularity and 3D consistency of facial expressions. Post-training augments rather than overwrites the generalization capability acquired during pretraining.
    - Design Motivation: Analogous to LLM pretraining + RLHF: pretraining provides capability, post-training provides quality.

3. **Self-supervised Expression Encoding**:

    - Function: Learns fine-grained facial expression control signals.
    - Mechanism: A FACS-inspired self-supervised approach is used to learn latent expression codes, which serve as driving signals for the correction branch. Combined with SMPL-X body and hand poses, this enables fine-grained full-body control.
    - Design Motivation: Expression is the most critical control dimension for avatars, requiring accuracy beyond that of parametric face models.

### Loss & Training

3DGS rendering loss (L1 + D-SSIM) + perceptual loss + identity preservation loss. Pretraining is performed on 1M in-the-wild videos; post-training is performed on multi-view studio data.

## Key Experimental Results

### Main Results

| Capability | LCA | Prev. SOTA | Notes |
|------|-----|---------|------|
| Identity generalization | World-scale population coverage | Thousands of identities | Hair / clothing / skin tone / accessories |
| Expression control | Fine-grained facial + finger-level | Coarse-grained | Significantly enhanced by post-training |
| 3D consistency | Strong | Weak in in-the-wild methods | Synergy of pretraining + post-training |
| Feed-forward inference | Efficient | Requires optimization | Generated from a few images |

### Ablation Study

| Configuration | Generalization | Expression Accuracy | 3D Consistency | Notes |
|------|------|---------|---------|------|
| Pretrain only | Strong | Weak (blurry expressions) | Moderate (3D distortion) | Broad priors but insufficient precision |
| Post-train only | Weak | Strong | Strong | High quality but poor generalization |
| Pretrain + post-train | Strong | Strong | Strong | Best balance |

### Key Findings

- **Emergent capabilities** are observed: without direct supervision, the model spontaneously generalizes to relighting, loose clothing support, and zero-shot robustness to stylized images.
- Priors acquired during pretraining are not overwritten during post-training—analogous to capability retention in LLMs.
- The scale of 1 million videos is critical for generalization ability.

## Highlights & Insights

- **Transfer of the LLM paradigm to 3D**: This is the first work to demonstrate that the pre/post-training paradigm breaks the generalization–fidelity trade-off in the 3D avatar domain.
- **Emergent capabilities**: The emergence of relighting support and stylization robustness suggests that large-scale data enables the model to acquire deep physical and semantic understanding.
- **Efficient feed-forward inference**: High-quality avatars can be generated from only a few images, making the approach suitable for practical deployment.

## Limitations & Future Work

- Pretraining on 1 million videos incurs extremely high computational costs (Meta-scale resources required).
- The fidelity of the body region is lower than that of the face.
- The degree of open-sourcing remains uncertain, and reproducibility has yet to be verified.

## Related Work & Insights

- **vs. Codec Avatars series**: Traditional Codec Avatars require per-identity optimization; LCA is feed-forward.
- **vs. TRELLIS/Rodin**: These methods operate at a smaller scale; LCA is the first to achieve million-scale pretraining.
- **vs. Real3D-Portrait**: Single-image methods have limited fidelity; LCA benefits from multi-image input and large-scale pretraining.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First successful application of the pretraining/post-training paradigm to 3D avatars.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive demonstrations, though quantitative evaluation could be more standardized.
- Writing Quality: ⭐⭐⭐⭐⭐ Fluent narrative with deep insights.
- Value: ⭐⭐⭐⭐⭐ Transformative implications for the 3D digital human industry.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling](../../ICCV2025/human_understanding/imhead_a_large-scale_implicit_morphable_model_for_localized_head_modeling.md)
- [\[CVPR 2026\] LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction](laser_layer-wise_scale_alignment_for_training-free_streaming_4d_reconstruction.md)
- [\[CVPR 2026\] AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video](avatar_reinforcement_learning_to_see_hear_and_reason_over_video.md)
- [\[CVPR 2026\] FlexAvatar: Learning Complete 3D Head Avatars with Partial Supervision](flexavatar_learning_complete_3d_head_avatars_with_partial_supervision.md)
- [\[CVPR 2026\] UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking](unils_end-to-end_audio-driven_avatars_for_unified_listening_and_speaking.md)

<!-- RELATED:END -->
