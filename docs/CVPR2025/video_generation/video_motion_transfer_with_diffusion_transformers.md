---
title: >-
  [Paper Note] DiTFlow: Video Motion Transfer with Diffusion Transformers
description: >-
  [CVPR 2025][Video Generation][Motion Transfer] DiTFlow proposes the first motion transfer method designed specifically for Diffusion Transformers (DiTs). By analyzing cross-frame attention maps, it extracts Attention Motion Flow (AMF) as patch-wise motion signals and guides the generation of new videos to replicate the motion patterns of reference videos in a training-free optimization manner.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Motion Transfer"
  - "Diffusion Transformer"
  - "Attention Motion Flow"
  - "Position Embedding Optimization"
  - "Zero-shot"
date: 2026-05-08
content_hash: 9d98afebe5694a9f
---

# DiTFlow: Video Motion Transfer with Diffusion Transformers

**Conference**: CVPR 2025  
**arXiv**: [2412.07776](https://arxiv.org/abs/2412.07776)  
**Code**: [ditflow.github.io](https://ditflow.github.io)  
**Area**: Image Restoration / Video Motion Transfer  
**Keywords**: Motion Transfer, Diffusion Transformer, Attention Motion Flow, Position Embedding Optimization, Zero-shot

## TL;DR

DiTFlow proposes the first motion transfer method designed specifically for Diffusion Transformers (DiTs). By analyzing cross-frame attention maps, it extracts Attention Motion Flow (AMF) as patch-wise motion signals and guides the generation of new videos to replicate the motion patterns of reference videos in a training-free optimization manner.

## Background & Motivation

Although video diffusion models can generate realistic video content, precise control over motion remains challenging using only text prompts due to inherent ambiguities in describing fine-grained temporal variations. Motion transfer addresses this issue by using reference videos as motion guidance. However, most existing methods are based on the UNet architecture with decoupled temporal and spatial attention, failing to fully leverage the benefits of DiTs.

DiTs process spatiotemporal information jointly using a full spatiotemporal attention mechanism, which allows them to extract higher-quality motion information compared to UNet-based methods. However, disentangling motion patterns from content within the full spatiotemporal attention of DiTs is more difficult. Existing UNet-based methods (such as spatial averaging in SMM and motion channel discovery in MOFT) assume decoupled temporal attention, making them inapplicable to DiTs.

Core insight of DiTFlow: tokens representing similar semantic content in a DiT actively attend to each other across frames. These cross-frame attention relations can be leveraged to build patch-level displacement maps that capture motion patterns.

## Method

### Overall Architecture

DiTFlow consists of two steps: (1) Extracting AMF motion signals from the reference video by passing the reference video into a specific transformer block of a pre-trained DiT and analyzing cross-frame attention to construct displacement matrices; (2) During the denoising process of generating a new video, optimizing latent variables or position embeddings to align the AMF of the generated video with the reference AMF.

### Key Design 1: Attention Motion Flow (AMF) Extraction

- **Function**: Extract patch-wise motion displacement signals from the cross-frame attention of DiTs
- **Mechanism**: Encode the reference video into a latent representation $z_{\text{ref}}$. At step $t=0$, extract the averaged $Q$ and $K$ through the $n$-th block of the DiT. Compute the cross-frame attention $A_{i,j}^{\otimes} = \sigma(\tau \frac{Q_i K_j^T}{\sqrt{d_k}})$, and use argmax to locate the corresponding patch coordinates in frame $j$ that frame $i$'s patches attend to most. This is used to construct the displacement matrix $\Delta_{i,j}[(u,v)] = (u'-u, v'-v)$. Aggregate all frame pairs to constitute the AMF
- **Design Motivation**: Feature analysis at $t=0$ yields cleaner motion signals compared to high-noise steps, bypassing expensive DDIM inversion. Employing argmax instead of softmax produces cleaner displacement maps

### Key Design 2: AMF-Guided Latent Optimization

- **Function**: Replicate reference motion in the generated video by optimizing latent variables during the denoising process
- **Mechanism**: During the denoising process, extract $\tilde{Q}$ and $\tilde{K}$ through the DiT for the current step's latent variable $z_t$. Compute a differentiable displacement matrix $\tilde{\Delta}_{i,j}$ using soft argmax (weighted sum) to maintain gradients. Minimize the AMF loss $\mathcal{L}_{\text{AMF}} = \|\text{AMF}(z_{\text{ref}}) - \text{AMF}(z_t)\|_2^2$, optimizing for $K_{\text{opt}}=5$ steps in each of the first 20% of denoising steps
- **Design Motivation**: Optimizing latent variables directly modifies the generated content to align with the reference motion, achieving peak performance across all metrics

### Key Design 3: Position Embedding Optimization for Zero-Shot Transfer

- **Function**: Achieve generalizable motion transfer by optimizing the position embeddings $\rho$ of the DiT, eliminating the need for repeated optimization for each new video
- **Mechanism**: Backpropagate the gradient of the AMF loss to the position embeddings $\rho_t$ instead of the latent variables $z_t$. Position embeddings encode the spatiotemporal positions of patches; manipulating position information guides the reorganization of patches to achieve motion transfer. Once optimized, the learned embeddings can be directly applied to the generation of new prompts
- **Design Motivation**: Position embeddings are decoupled from content. Manipulating spatial position information does not affect content encoding, leading to superior generalizability. This is a property that is impossible to achieve with UNet-based methods

### Loss & Training

AMF loss: $\mathcal{L}_{\text{AMF}}(z_{\text{ref}}, z_t) = \|\text{AMF}(z_{\text{ref}}) - \text{AMF}(z_t)\|_2^2$, computed as the element-wise Euclidean distance across the displacement matrices of all frame pairs.

## Key Experimental Results

### Main Results: Motion Transfer Evaluation on CogVideoX-5B

| Method | MF(Caption)↑ | MF(Subject)↑ | MF(Scene)↑ | MF(All)↑ | IQ(All)↑ |
|------|-------------|-------------|------------|---------|---------|
| Backbone | 0.524 | 0.502 | 0.544 | 0.523 | 0.315 |
| MotionClone | 0.635 | 0.640 | 0.628 | 0.634 | 0.318 |
| SMM | 0.782 | 0.741 | 0.776 | 0.766 | 0.315 |
| MOFT | 0.728 | 0.728 | 0.722 | 0.726 | 0.318 |
| **DiTFlow** | **0.790** | **0.775** | **0.789** | **0.785** | **0.319** |

### Comparison on CogVideoX-2B

| Method | MF(All)↑ | IQ(All)↑ |
|------|---------|---------|
| SMM | 0.688 | 0.312 |
| MOFT | 0.504 | 0.312 |
| **DiTFlow** | **0.726** | **0.317** |

### Key Findings

- DiTFlow consistently achieves top performance in Motion Fidelity (MF), yielding 0.785 on the 5B model compared to 0.766 for SMM.
- SMM exhibits a significant drop in MF under Subject prompts (0.741 vs. 0.782 for Caption), indicating entanglement between spatially averaged features and reference content.
- DiTFlow delivers consistent performance across all prompt types, demonstrating superior motion-content disentanglement.
- In human evaluation, DiTFlow significantly outperforms baselines in both motion fidelity and prompt consistency.
- Although position embedding optimization achieves slightly lower performance than latent optimization, it facilitates optimization-free zero-shot transfer.

## Highlights & Insights

1. **First DiT-Specific Motion Transfer Method**: Fully exploits the joint spatiotemporal attention of DiTs to extract patch-wise motion signals.
2. **Innovative Insight on Position Embedding Optimization**: Leverages the unique position encoding mechanism of DiTs to achieve zero-shot motion transfer—a capability unavailable in UNet architectures.
3. **Patch-wise Motion-Content Disentanglement**: Unlike the spatial averaging of SMM and position-independent biases of MOFT, AMF captures explicit spatiotemporal relationships between patches.

## Limitations & Future Work

- Latent optimization requires additional optimization time per video (8 minutes vs. 5 minutes for baselines).
- A performance gap still exists for the zero-shot mode enabled by position embedding optimization.
- Fine-grained control over motion transfer (e.g., transferring only partial motion) remains unexplored.
- Future work could integrate AMF guidance with other conditional controls.

## Related Work & Insights

- **SMM**: Utilizes spatially-averaged motion features; its global nature presents difficulties when modifying local semantics.
- **MOFT**: Highlights motion channels in diffusion features, but independent processing of locations is prone to misguiding incorrect elements.
- **MotionClone**: A UNet-based attention motion transfer method. However, its assumption of decoupled temporal attention makes it inapplicable to DiTs.
- Insight: While the joint spatiotemporal attention of DiTs makes motion disentanglement more challenging, it inherently contains richer spatiotemporal association information.

## Rating

⭐⭐⭐⭐ — The first motion transfer method designed specifically for DiT architectures. The concept of AMF is elegant and effective. The zero-shot capability enabled by position embedding optimization is a unique contribution. Robustness is well demonstrated by consistent improvements across scaling models and prompt types.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](../../ICCV2025/video_generation/decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)
- [\[CVPR 2025\] ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer](conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)
- [\[CVPR 2025\] Towards Precise Scaling Laws for Video Diffusion Transformers](towards_precise_scaling_laws_for_video_diffusion_transformers.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](../../ICCV2025/video_generation/efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [\[CVPR 2025\] Motion Prompting: Controlling Video Generation with Motion Trajectories](motion_prompting_controlling_video_generation_with_motion_trajectories.md)

</div>

<!-- RELATED:END -->
