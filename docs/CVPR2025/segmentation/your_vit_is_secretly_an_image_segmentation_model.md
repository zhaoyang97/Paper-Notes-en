---
title: >-
  [Paper Note] Your ViT is Secretly an Image Segmentation Model
description: >-
  [CVPR 2025][Segmentation][Vision Transformer] This paper proposes the Encoder-only Mask Transformer (EoMT), demonstrating that with large-scale pre-training and sufficiently large models, plain ViT can achieve high-quality image segmentation without task-specific components such as CNN adapters, pixel decoders, and Transformer decoders, while being up to 4x faster.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Vision Transformer"
  - "Image Segmentation"
  - "Encoder-only Architecture"
  - "Mask Annealing"
  - "DINOv2"
date: 2026-05-08
content_hash: b1214f639cd203b0
---

# Your ViT is Secretly an Image Segmentation Model

**Conference**: CVPR 2025  
**arXiv**: [2503.19108](https://arxiv.org/abs/2503.19108)  
**Code**: [https://www.tue-mps.org/eomt/](https://www.tue-mps.org/eomt/)  
**Area**: Segmentation  
**Keywords**: Vision Transformer, Image Segmentation, Encoder-only Architecture, Mask Annealing, DINOv2

## TL;DR
This paper proposes the Encoder-only Mask Transformer (EoMT), demonstrating that with large-scale pre-training and sufficiently large models, plain ViT can achieve high-quality image segmentation without task-specific components such as CNN adapters, pixel decoders, and Transformer decoders, while being up to 4x faster.

## Background & Motivation

**Background**: State-of-the-art image segmentation methods (semantic/instance/panoptic segmentation) usually stack multiple task-specific components on top of ViT: ViT-Adapter (CNN adapter for multi-scale feature extraction) $\rightarrow$ pixel decoder (multi-scale feature fusion) $\rightarrow$ Transformer decoder (object queries + cross-attention for prediction generation). A typical representative is ViT-Adapter + Mask2Former.

**Limitations of Prior Work**: These additional components introduce significant computational overhead and implementation complexity. The convolutional modules introduced by ViT-Adapter require multiple interactions with ViT, the pixel decoder uses multi-scale deformable attention, and the masked cross-attention in the Transformer decoder requires predicting intermediate masks at every layer—all representing performance bottlenecks. ViT-Adapter + M2F only achieves 32 FPS on ViT-B.

**Key Challenge**: These task-specific components essentially compensate for two "drawbacks" of ViT—the lack of multi-scale features and the lack of local processing capability. However, are these so-called drawbacks due to the inherent insufficiency of ViT, or are they due to insufficient pre-training?

**Goal**: To validate the hypothesis that as model scale and pre-training quality increase, ViT can implicitly learn these inductive biases, making task-specific components redundant.

**Key Insight**: Gradually remove each component and observe the changes in performance. If the performance degradation under a large model and strong pre-training is negligible, it proves these components are unnecessary.

**Core Idea**: Treat ViT as both the encoder and decoder—concatenating learnable queries directly into the patch tokens, replacing cross-attention with ViT's own self-attention, and utilizing a mask annealing strategy to enable highly efficient segmentation without mask attention at inference time.

## Method

### Overall Architecture
The input to EoMT is the token sequence after image patch embedding. The first $L_1$ ViT blocks process patch tokens normally. Then, $K$ learnable queries are concatenated into the token sequence. The subsequent $L_2$ blocks jointly process both patch tokens and query tokens. Finally, the query tokens predict classes and segmentation masks through a lightweight masking module.

### Key Designs

1. **逐步移除任务特定组件**:

    - Function: Verify the necessity of each component
    - Mechanism: Start from ViT-Adapter + M2F and gradually remove components: (1) Remove ViT-Adapter—use simple transposed/standard convolutions to construct a feature pyramid from single-scale ViT outputs (ViTDet-style), which only drops PQ by 0.4; (2) Remove the pixel decoder—directly feed the feature pyramid into the TF decoder, keeping PQ basically unchanged; (3) Remove multi-scale processing—queries only attend to single-scale ViT outputs, dropping PQ by another 0.2; (4) Remove the independent TF decoder—directly inject queries into the last few layers of ViT, which drops PQ by 0.5 but boosts FPS to 61
    - Design Motivation: Each ablation step quantifies the "actual value" of the corresponding component. It is discovered that on ViT-L pre-trained with DINOv2, all components combined only contribute 1.1 PQ, while introducing a 4.4x speed penalty

2. **查询注入机制**:

    - Function: Allow ViT's self-attention to simultaneously handle query-to-query interactions and query-to-image feature transfer
    - Mechanism: Concatenate $K$ learnable queries to the end of the patch tokens after the $L_1$-th layer. In the subsequent $L_2$ self-attention layers, a single MHSA operation simultaneously computes: self-attention between queries (coordinating queries), cross-attention from queries to patches (acquiring visual info), reverse attention from patches to queries (feeding query info back to visual features), and the original self-attention among patches. For ViT-L ($L=24$), setting $L_1=20$ and $L_2=4$ only increases the token processing workload by 2.1%
    - Design Motivation: Standard segmentation methods perform self-attention and cross-attention sequentially in two distinct steps. EoMT exploits the natural "all-to-all" property of ViT's self-attention mechanism to complete all interactions within a single step, which is more efficient and simpler

3. **掩码退火（Mask Annealing）策略**:

    - Function: Use masked attention during training (for performance) and no masked attention during inference (for speed)
    - Mechanism: Masked attention in M2F constrains each query to focus only on the image region corresponding to its predicted mask, which aids learning but degrades efficiency. EoMT fully enables masked attention in the early training phases ($P_{mask}=1.0$), and then block-by-block and step-by-step anneals the masking probability to 0, ensuring the model adapts to a mask-free attention scheme by the end of training. The annealing schedule uses polynomial decay, lifting masks from early layers first, until the mask probability reaches 0 for all layers at the end of training
    - Design Motivation: Training without masked attention directly leads to a 3.0 PQ drop, while simply removing the mask during inference causes a catastrophic drop of 28.8 PQ. Mask annealing acts as a progressive transition strategy, leveraging masks to assist learning during training while requiring absolutely none at inference, offering the best of both worlds

### Loss & Training
The loss function follows M2F: $\mathcal{L}_{tot} = \lambda_{bce}\mathcal{L}_{bce} + \lambda_{dice}\mathcal{L}_{dice} + \lambda_{ce}\mathcal{L}_{ce}$, with weights of 5.0, 5.0, and 2.0, respectively. Training employs the AdamW optimizer with layer-wise learning rate decay (factor 0.8) and a polynomial learning rate schedule. The patch embedding and position encodings of DINOv2 pre-trained weights are adjusted to the target resolution using the FlexiViT method.

## Key Experimental Results

### Main Results

| Method | Backbone | PQ (COCO) | FPS | Params |
|------|----------|-----------|-----|--------|
| ViT-Adapter + M2F | ViT-L | 57.1 | 29 | 349M |
| EoMT w/ Masking | ViT-L | 56.2 | 61 | 316M |
| **EoMT** | **ViT-L** | **56.0** | **128** | **316M** |
| ViT-Adapter + M2F | ViT-g | 57.7 | 20 | 1209M |
| EoMT | ViT-g | 57.0 | 55 | 1164M |
| EoMT (high resolution) | ViT-L | 58.3 | 30 | 322M |

### Ablation Study

| Training Strategy | Inference Strategy | FPS | PQ | Description |
|---------|---------|-----|------|------|
| Full masked attention | Masked attention | 61 | 56.2 | Complete but slow |
| Full masked attention | No mask | 128 | 27.4 | Catastrophic failure -28.8 |
| Full mask-free | No mask | 128 | 53.2 | Acceptable but worse -3.0 |
| **Mask annealing** | **No mask** | **128** | **56.0** | Drops only 0.2, doubles speed |

### Key Findings
- **Pre-training quality is decisive**: Under DINOv2/EVA-02 pre-training, the gap between EoMT and complex models is only 1.1–1.2 PQ, but this gap widens to 6.1 under ImageNet-1K pre-training, confirming the hypothesis that "large-scale pre-training makes task-specific components redundant".
- **Larger models make components more redundant**: The gap between EoMT and M2F is 5.8 PQ on ViT-S but only 0.7 PQ on ViT-g, showing a clear shrinking trend.
- **Mask annealing is the key innovation**: It yields a 2.1x speedup (61 $\rightarrow$ 128 FPS) at the cost of only 0.2 PQ, and is also effective on ViT-Adapter + M2F (extremely versatile).
- EoMT also performs excellently on out-of-distribution generalization: posting an OOD confidence score of 89.7 on the BRAVO benchmark vs 68.7 for ViT-Adapter+M2F.
- Compatible with token merging (ALGM), achieving an additional 31% throughput improvement without sacrificing mIoU.

## Highlights & Insights
- **A paradigm of "subtractive" thinking**: Instead of adding new modules, it proves that existing modules can be removed. This simplification philosophy is especially valuable in the era of large models—allocating computing resources to scaling up the ViT itself rather than adding complex peripherals.
- **Ingenious design of mask annealing**: Utilizing masks to assist learning during training while requiring absolutely none at inference elegantly resolves the training-inference inconsistency problem. This strategy can be generalized to other training frameworks that use intermediate constraints.
- **Ecosystem compatibility**: EoMT is entirely based on plain ViT, enabling it to directly benefit from all Transformer ecosystem optimizations such as FlashAttention, token merging, and ViT-specific hardware acceleration, while complex architectures remain bottlenecked by custom components.

## Limitations & Future Work
- The performance decline in instance segmentation (-2.4 AP) is larger than in panoptic segmentation (-1.1 PQ), indicating room for improvement in tasks that require fine-grained instance discrimination.
- Validation is currently limited to DINOv2 and EVA-02 pre-training; future, stronger vision foundation models might narrow the gap further.
- The performance gap on small models (ViT-S) remains significant (-5.8 PQ), making EoMT more suitable for medium-to-large deployment scenarios.
- The scheduling of mask annealing requires pre-set hyperparameters; adaptive scheduling strategies are worth exploring.

## Related Work & Insights
- **vs ViT-Adapter + Mask2Former**: EoMT challenges this "standard setup" by simplifying it, proving that with DINOv2, the marginal utility of complex components is small, while the speed penalty is huge.
- **vs YOLOS**: Both use an encoder for detection, but YOLOS does not perform segmentation and has not verified the benefits of larger models.
- **vs UViT**: Uses single-scale ViT features for instance recognition, but still relies on complex task decoders.
- **vs SegFormer/OneFormer**: These methods rely on non-plain-ViT architectures like Swin/DiNAT, preventing them from leveraging pre-training from vision foundation models like DINOv2.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative mask annealing strategy; "subtractive design" offers unique insights under current trends.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across three major segmentation tasks, multiple datasets, multiple model sizes, and various pre-training methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely elegant hypothesis-driven, progressive ablation narrative structure.
- Value: ⭐⭐⭐⭐⭐ Provides a crucial direction for simplifying the design paradigm of ViT segmentation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](../../CVPR2026/segmentation/videomt_your_vit_is_secretly_also_a_video_segmentation_model.md)
- [\[ICML 2025\] FeatSharp: Your Vision Model Features, Sharper](../../ICML2025/segmentation/featsharp_your_vision_model_features_sharper.md)
- [\[CVPR 2025\] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)
- [\[ICLR 2026\] TRACE: Your Diffusion Model is Secretly an Instance Edge Detector](../../ICLR2026/segmentation/trace_your_diffusion_model_is_secretly_an_instance_edge_detector.md)
- [\[CVPR 2025\] 2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification](2dmamba_efficient_state_space_model_for_image_representation_with_applications_o.md)

</div>

<!-- RELATED:END -->
