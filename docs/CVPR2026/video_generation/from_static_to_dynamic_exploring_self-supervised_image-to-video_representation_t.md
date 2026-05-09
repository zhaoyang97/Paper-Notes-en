---
title: >-
  [Paper Note] From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning
description: >-
  [CVPR 2026][Video Generation][Image-to-video transfer] This paper proposes Co-Settle, a framework that trains a lightweight linear projection layer on top of a frozen image-pretrained encoder. Using temporal cycle consistency loss and semantic separability constraints, the method achieves consistent improvements across multi-granularity video downstream tasks on 8 image foundation models with only 5 epochs of self-supervised training.
tags:
  - CVPR 2026
  - Video Generation
  - Image-to-video transfer
  - self-supervised learning
  - temporal consistency
  - semantic separability
  - lightweight projection
date: 2026-05-08
content_hash: 61ef6710d3896abe
---

# From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning

**Conference**: CVPR 2026
**arXiv**: [2603.26597](https://arxiv.org/abs/2603.26597)
**Code**: [https://github.com/yafeng19/Co-Settle](https://github.com/yafeng19/Co-Settle)
**Area**: Video Generation
**Keywords**: Image-to-video transfer, self-supervised learning, temporal consistency, semantic separability, lightweight projection

## TL;DR

This paper proposes Co-Settle, a framework that trains a lightweight linear projection layer on top of a frozen image-pretrained encoder. Using temporal cycle consistency loss and semantic separability constraints, the method achieves consistent improvements across multi-granularity video downstream tasks on 8 image foundation models with only 5 epochs of self-supervised training.

## Background & Motivation

**Background**: Transferring image-pretrained models to video tasks has become the dominant paradigm in video representation learning. Existing methods typically attach temporal modeling modules (e.g., temporal attention, 3D convolutions, adapters) to pretrained image encoders and fine-tune on video data.

**Limitations of Prior Work**: Fine-tuning heavy temporal modules degrades **inter-video semantic separability**—the ability to distinguish different objects across different videos—because video datasets have limited category diversity, which easily induces catastrophic forgetting. Conversely, restricting trainable parameters to preserve separability leads to insufficient **intra-video temporal consistency**—the stability of representations for the same object within a video.

**Key Challenge**: Image-to-video transfer involves an inherent **trade-off between temporal consistency and semantic separability**. Existing methods either over-fine-tune and lose semantic discriminability, or operate under parameter constraints that prevent learning adequate temporal correspondences.

**Goal**: To identify an efficient method that enhances temporal consistency while preserving or even improving semantic separability.

**Key Insight**: The authors observe that image-pretrained models already possess approximate temporal consistency (due to geometric augmentations during pretraining) but lack modeling of real-world temporal dynamics. A single extremely lightweight projection layer suffices to adjust the representation space to balance both properties.

**Core Idea**: Freeze the image encoder and train only a linear projection layer; use a cycle consistency objective to enhance temporal correspondences and a KL-divergence constraint to preserve semantic separability, achieving efficient image-to-video representation transfer.

## Method

### Overall Architecture

Two frames $\mathbf{v}_{t_1}$ and $\mathbf{v}_{t_2}$ are sampled from each video. Patch-level features are extracted via the frozen image encoder and then mapped through a learnable lightweight projection layer $g$ (linear layer + LayerNorm, only 0.59M parameters) into an adjusted representation space. Temporal correspondences are learned via a cycle consistency loss in this space, while a KL-divergence constraint preserves semantic structure. The total loss is $\mathcal{L}_{total} = \mathcal{L}_{cyc} + \lambda \mathcal{L}_{reg}$.

### Key Designs

1. **Positional Encoding Augmentation (PEA)**:

    - **Function**: Prevents positional shortcuts in cycle consistency learning.
    - **Mechanism**: The explicit positional encodings in ViT cause patches to be matched by absolute position rather than semantic content, trivially minimizing the cycle consistency loss. PEA applies interpolation-based scaling and random cropping to the positional encodings of the backward frame, producing an augmented variant $\tilde{\mathbf{E}}_{pos}$ that disrupts exact positional matching while preserving local positional relationships. This asymmetric design is inspired by information bottleneck theory.
    - **Design Motivation**: Experiments reveal that even with unrelated videos or shuffled patches, the CRW loss converges rapidly to zero—demonstrating that the model exploits positional shortcuts rather than learning genuine visual correspondences.

2. **Temporal Cycle Consistency Learning**:

    - **Function**: Establishes precise temporal correspondences between patches across frames.
    - **Mechanism**: A forward-backward cycle $\mathbf{v}_{t_1} \to \mathbf{v}_{t_2} \to \mathbf{v}_{t_1}$ is constructed. The forward affinity matrix $\mathbf{A}_{t_1 t_2}$ and the asymmetric backward affinity matrix $\tilde{\mathbf{A}}_{t_2 t_1}$ (computed using PEA-processed backward frames) are computed. The objective encourages $\mathbf{A}_{t_1 t_2} \tilde{\mathbf{A}}_{t_2 t_1}$ to approximate the identity matrix, i.e., each patch should return to its original position after one full cycle: $\mathcal{L}_{cyc} = -\sum_{i=1}^{N} \log P(X_d = \tilde{\mathbf{q}}_{t_1}^b(i) | X_s = \mathbf{q}_{t_1}^f(i))$
    - **Design Motivation**: Cycle consistency provides a direct and explicit learning signal for temporal correspondences, which is more efficient than indirect auxiliary tasks.

3. **Semantic Separability Constraint**:

    - **Function**: Prevents the projection layer from forgetting semantic discriminability during temporal consistency optimization.
    - **Mechanism**: A KL-divergence constraint enforces consistency between feature distributions before and after projection: $\mathcal{L}_{reg} = \frac{1}{|\mathcal{S}|} \sum_{(\mathbf{p},\mathbf{z}) \in \mathcal{S}} \sum_{i=1}^{d} P(i) \log \frac{P(i)}{Z(i)}$, where $P = \text{softmax}(\mathbf{p})$ and $Z = \text{softmax}(\mathbf{z})$. This forces the projection layer to maintain a near-isometric mapping and prevents dimensional collapse.
    - **Design Motivation**: Optimizing consistency solely on video data with limited category diversity leads to catastrophic forgetting. The KL constraint protects semantic structure without impeding consistency learning.

### Loss & Training

- Total loss: $\mathcal{L}_{total} = \mathcal{L}_{cyc} + \lambda \mathcal{L}_{reg}$
- Only the projection layer $g$ (0.59M parameters) is updated; the encoder is fully frozen.
- Trained on Kinetics-400 for 5 epochs, batch size 512, on 4 RTX 4090 GPUs, using AdamW optimizer with learning rate $1 \times 10^{-4}$ and cosine decay.
- Theoretical analysis (Theorem 1–2) proves that the optimal linear projection exhibits soft-thresholding behavior—dimensions with high temporal variance are suppressed while those with low variance are amplified—ultimately increasing the margin between inter-video and intra-video distances.

## Key Experimental Results

### Main Results (Dense-level Tasks, ViT-B/16)

| Method | VIP mIoU | DAVIS J&F | JHMDB PCK@0.1 |
|--------|----------|-----------|---------------|
| MAE (original) | 29.3 | 52.4 | 41.6 |
| MAE + Co-Settle | **33.8** | **59.6** | **48.4** |
| CLIP (original) | 38.1 | 54.9 | 36.9 |
| CLIP + Co-Settle | **39.2** | **58.3** | **40.6** |
| DINOv2 (original) | 38.4 | 63.1 | 46.6 |
| DINOv2 + Co-Settle | **39.9** | **63.7** | **47.3** |

### Ablation Study

| Configuration | VIP mIoU | DAVIS J&F | JHMDB PCK | Notes |
|---------------|----------|-----------|-----------|-------|
| $\mathcal{L}_{cyc}$ only (no PEA) | 16.2 | 26.2 | 38.5 | Positional shortcuts cause severe degradation |
| $\mathcal{L}_{cyc}$ + PEA | 33.3 | 59.3 | 48.1 | PEA effectively suppresses shortcuts |
| $\mathcal{L}_{cyc}$ + $\mathcal{L}_{reg}$ + PEA | **33.8** | **59.6** | **48.4** | Full model |
| Linear layer (default) | 33.8 | 59.6 | 47.9 | Simple linear layer is sufficient |
| 2-layer MLP | 33.2 | 59.2 | 47.5 | Greater complexity not necessarily better |
| 3-layer MLP | 32.6 | 58.4 | 47.4 | Excessive depth hurts semantics |

### Key Findings

- **All 8 image-pretrained models show consistent improvements**, covering masked modeling, contrastive learning, and self-distillation paradigms.
- Training requires only 5 epochs, 0.59M parameters, and 1.2 RTX 4090 GPU-days—approximately 13× faster than conventional methods.
- A linear layer performs on par with or better than MLPs, consistent with the theoretical analysis showing similar soft-thresholding behavior between linear and MLP projections.
- Replacing DINOv2 features with Co-Settle features in the DINO-Tracker pipeline improves BADJA tracking accuracy from 62.73 to 70.52.

## Highlights & Insights

- **Extreme simplicity**: Frozen encoder + one linear layer + 5 epochs = consistent gains across all models and all task granularities. This "less is more" design philosophy is remarkable.
- **The discovery process behind PEA is highly instructive**: By designing controlled experiments with unrelated videos and shuffled patches, the existence of positional shortcuts is rigorously verified, demonstrating exemplary experimental design thinking.
- **Well-grounded theoretical support**: Beyond being an empirical method, the paper provides spectral analysis proofs that reveal the soft-thresholding mechanism of the projection layer, making the approach interpretable.

## Limitations & Future Work

- Training is conducted only on Kinetics-400; the impact of video dataset content diversity on transfer performance remains insufficiently explored.
- The current work is limited to ViT architectures; applicability to CNN or hybrid architectures is not validated.
- Whether the projected features conflict across different downstream tasks warrants further investigation.
- Extension to multi-frame settings (currently only 2 frames are used) could be considered to capture more complex long-range temporal correspondences.

## Related Work & Insights

- **vs. AIM/ST-Adapter/ZeroI2V**: These CLIP-based methods require 11–14M parameters and substantial GPU time for supervised adaptation, whereas Co-Settle achieves superior performance on dense tasks with only 0.59M parameters and unsupervised training.
- **vs. SiamMAE/CropMAE**: These video pretraining methods require 400–1600 epochs of large-scale training; Co-Settle achieves comparable performance at a fraction of the cost.
- The core insight of **explicitly modeling the trade-off** is transferable to other representation transfer scenarios, such as cross-modal or cross-resolution transfer.

## Rating

- Novelty: ⭐⭐⭐⭐ — Explicitly modeling and theoretically proving the trade-off is a highlight, though the lightweight projection layer concept is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 8 models, multi-granularity tasks, efficiency comparisons, and theoretical validation; exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — The motivation–method–theory–experiment logical chain is very clear; the theoretical section is self-contained.
- Value: ⭐⭐⭐⭐ — Provides an efficient baseline and theoretical framework for image-to-video transfer, though the applicable scenarios are relatively specialized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Let Your Image Move with Your Motion! – Implicit Multi-Object Multi-Motion Transfer](let_your_image_move_with_your_motion_--_implicit_multi-object_multi-motion_trans.md)
- [\[ICCV 2025\] Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning](../../ICCV2025/video_generation/disentangled_world_models_learning_to_transfer_semantic_knowledge_from_distracti.md)
- [\[CVPR 2026\] Diff4Splat: Repurposing Video Diffusion Models for Dynamic Scene Generation](diff4splat_controllable_4d_scene_generation_with_latent_dynamic_reconstruction_m.md)
- [\[AAAI 2026\] SphereDiff: Tuning-free Omnidirectional Panoramic Image and Video Generation via Spherical Latent Representation](../../AAAI2026/video_generation/spherediff_tuning-free_360_static_and_dynamic_panorama_generation_via_spherical_.md)
- [\[CVPR 2026\] Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning](goal-driven_reward_by_video_diffusion_models_for_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
