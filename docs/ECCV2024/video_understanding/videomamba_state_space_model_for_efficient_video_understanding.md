---
title: >-
  [Paper Note] VideoMamba: State Space Model for Efficient Video Understanding
description: >-
  [ECCV 2024][Video Understanding][State Space Model] This work innovatively adapts Mamba's selective state space model to the video domain, proposing VideoMamba, a pure SSM architecture. It achieves efficient spatiotemporal context modeling with linear complexity, demonstrating superior performance on both short and long video understanding tasks.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "State Space Model"
  - "Mamba"
  - "Linear Complexity"
  - "Long Video Modeling"
date: 2026-05-08
content_hash: 13b15ea82f37fde2
---

# VideoMamba: State Space Model for Efficient Video Understanding

**Conference**: ECCV 2024  
**arXiv**: [2403.06977](https://arxiv.org/abs/2403.06977)  
**Code**: [https://github.com/OpenGVLab/VideoMamba](https://github.com/OpenGVLab/VideoMamba)  
**Area**: Video Understanding  
**Keywords**: State Space Model, Mamba, Video Understanding, Linear Complexity, Long Video Modeling

## TL;DR

This work innovatively adapts Mamba's selective state space model to the video domain, proposing VideoMamba, a pure SSM architecture. It achieves efficient spatiotemporal context modeling with linear complexity, demonstrating superior performance on both short and long video understanding tasks.

## Background & Motivation

**Background**: The core of video understanding lies in mastering spatiotemporal representations, which faces two major challenges: substantial spatiotemporal redundancy in short video clips and complex spatiotemporal dependencies in long contexts.

**Limitations of Prior Work**: 3D CNNs (e.g., SlowFast) excel at local modeling but fail to capture long-range dependencies; Video Transformers (e.g., TimeSformer, ViViT) can model long-range dependencies, but the quadratic complexity of self-attention leads to extremely high computational costs; UniFormer attempts to combine the advantages of both but still struggles with long videos.

**Key Challenge**: It is difficult to achieve both high efficiency and strong modeling capabilities simultaneously—when processing 64-frame videos, the throughput and GPU memory consumption of TimeSformer are highly unacceptable.

**Goal**: Design a video understanding architecture that possesses both linear complexity and strong spatiotemporal dynamic modeling capabilities.

**Key Insight**: Leverage the emerging Mamba (selective SSM) from the NLP field and extend it bidirectionally to 3D video sequences.

**Core Idea**: Build a pure SSM video model based on the simple architecture of vanilla ViT, replacing the self-attention layers with bidirectional Mamba blocks.

## Method

### Overall Architecture

VideoMamba strictly follows the architecture design of vanilla ViT. An input video $\mathbf{X}^v \in \mathbb{R}^{3 \times T \times H \times W}$ is first projected into $L = t \times h \times w$ non-overlapping spatiotemporal patch tokens $\mathbf{X}^p \in \mathbb{R}^{L \times C}$ via a 3D convolution (kernel $1 \times 16 \times 16$). Subsequently, learnable spatial position embeddings $\mathbf{p}_s$ and temporal position embeddings $\mathbf{p}_t$ are added, and a [CLS] token is appended to the front of the sequence. The token sequence sequentially passes through $L$ stacked B-Mamba (bidirectional Mamba) blocks, and the final [CLS] token is normalized and classified via a linear layer.

### Key Designs

1. **Selective State Space Model (S6)**: The core operator is based on Mamba's selective scan mechanism. Unlike traditional linear time-invariant SSMs, the parameters $\mathbf{B}$, $\mathbf{C}$, and $\boldsymbol{\Delta}$ in S6 are dynamically generated from the input data, providing context awareness. The continuous system is discretized via Zero-Order Hold (ZOH):

    $$\bar{\mathbf{A}} = \exp(\boldsymbol{\Delta} \mathbf{A}), \quad \bar{\mathbf{B}} = (\boldsymbol{\Delta} \mathbf{A})^{-1}(\exp(\boldsymbol{\Delta} \mathbf{A}) - \mathbf{I}) \cdot \boldsymbol{\Delta} \mathbf{B}$$

    $$h_t = \bar{\mathbf{A}} h_{t-1} + \bar{\mathbf{B}} x_t, \quad y_t = \mathbf{C} h_t$$

   This data-dependent parameterization allows the model to adaptively adjust weights, achieving content-aware context modeling while maintaining a linear complexity of $\mathcal{O}(n_h \cdot n_w \cdot n_t)$.

2. **Bidirectional Mamba Block (B-Mamba)**: The original Mamba is designed for unidirectional 1D sequences, lacking spatial awareness. Drawing inspiration from Vision Mamba, VideoMamba adopts a bidirectional SSM to process both forward and backward sequences simultaneously, enhancing spatial perception. Each B-Mamba block contains: linear projection ($384 \to 768$) → 1D convolution → bidirectional ST-SSM → linear projection ($768 \to 384$).

3. **Spatiotemporal Scanning Strategy**: Extending 2D bidirectional scanning to 3D video, four strategies were explored:

    - **Spatial-First**: Organizing tokens by spatial positions and stacking them frame by frame—the simplest and most effective strategy.
    - **Temporal-First**: Arranging temporal tokens by frame and stacking them along the spatial dimension.
    - **Spatiotemporal**: A hybrid of both, where v1 executes half of each, and v2 executes all (2× computation).
   
   Ablation studies show that **Spatial-First bidirectional scanning** yields the best performance, as it seamlessly leverages 2D pre-training knowledge.

4. **Self-Distillation**: Large-scale Mamba models (such as VideoMamba-B) are prone to overfitting during training. The solution is to use a pre-trained small model (such as VideoMamba-S) as a teacher to guide the training of the large model (student) by aligning their final feature maps via L2 loss. This strategy achieves better convergence and scalability with minimal extra computational overhead.

5. **Masked Modeling**: To enhance temporal sensitivity, mask alignment methods from UMT were adapted. Addressing the characteristic that the 1D convolution in B-Mamba prefers continuous tokens, a **Row Masking** strategy (clip-row and frame-row) is designed, and **Attention Masking** is introduced to retain meaningful adjacent relationships among neighboring tokens. Due to the architectural differences between SSM and Transformer, aligning only the final output layer yields the best results.

### Loss & Training

- **Supervised Training**: Initialized with ImageNet-1K pre-trained weights, utilizing the AdamW optimizer + cosine learning rate scheduler; trained for 50 epochs on K400, with the learning rate linearly scaled as $2e^{-4} \cdot \frac{batchsize}{256}$.
- **Self-Distillation**: Employs L2 loss to align the final feature maps of the teacher and the student.
- **Self-Supervised (UMT-style)**: Distills VideoMamba-M using CLIP-ViT-B for 800 epochs; optimal mask ratio is 80%, paired with stronger regularization (droppath=0.4).
- **Multimodal Pre-training**: Conducted on WebVid-2M + CC3M, featuring four objectives: vision-text contrastive learning, matching, MLM, and unmasked token alignment.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (VideoMamba-M) | SOTA | Gain/Comparison |
|--------|------|------|------|------|
| ImageNet-1K | Top-1 Acc | 84.0% (576²) | DeiT-B 83.1% | +0.9% (Fewer params) |
| Kinetics-400 (Supervised) | Top-1 Acc | 83.3% (64f, 384²) | ViViT-L 81.3% | +2.0% |
| SthSthV2 (Supervised) | Top-1 Acc | 68.4% (16f, 288²) | ViViT-L 65.4% | +3.0% |
| K400 (Self-Supervised) | Top-1 Acc | 85.0% (64f) | UMT-B 85.7% | Close (Fewer params) |
| Breakfast (Long Video) | Top-1 Acc | 97.9% | ViS4mer 88.2% | +9.7% |
| COIN (Long Video) | Top-1 Acc | 90.4% | Distant Sup. 90.0% | +0.4% |

### Ablation Study

| Configuration | SSV2 Top-1 | Description |
|------|------|------|
| Spatial-First scan | 65.1% | Optimal scanning strategy |
| Temporal-First scan | 62.4% | Worst, loses spatial info |
| ST-Bidirectional v2 | 64.2% | Spatiotemporal hybrid, 2× computation |
| Attention Masking | 68.5% | Optimal masking type |
| Random Masking | 67.4% | Baseline masking |
| Mask ratio 80% | 68.5% | Optimal ratio |
| Droppath 0.4 | 68.5% | Optimal regularization strength for self-supervised |

### Efficiency Comparison

| Model | Frames | Speed (Relative) | GPU Memory (Relative) |
|------|------|------|------|
| VideoMamba | 64f | 6× faster than TimeSformer | 40× less than TimeSformer |
| VideoMamba-Ti | 16f, 224² | 17 GFLOPs | 7M Params |
| TimeSformer-L | 96f, 224² | 2380 GFLOPs | 121M Params |

### Key Findings

- Spatial-First scanning is the most effective because it seamlessly leverages 2D pre-training knowledge.
- Self-distillation effectively solves the overfitting problem of large-scale Mamba, with negligible extra computational overhead.
- Distillation aligning only the final output layer performs best (due to the architectural differences between SSM and Transformer).
- Masked modeling is also applicable to Mamba, and Row Masking paired with 1D convolution yields the best results.
- End-to-end training for long videos performs significantly better than methods based on pre-extracted features.

## Highlights & Insights

1. **Minimalist Architecture**: Strictly maintains the isotropic design of ViT, without downsampling layers or extra depthwise convolutions, proving the feasibility of pure SSM architectures in the video domain.
2. **Comprehensive Four-dimensional Validation**: Covers scalability, short-video sensitivity, long-video superiority, and multimodal compatibility—providing an extremely thorough evaluation perspective.
3. **Stunning Efficiency Advantage**: 6× faster and consumes 40× less GPU memory than TimeSformer on 64-frame videos, making end-to-end training on long videos feasible.
4. **Self-Distillation Solves Overfitting**: Elegantly addresses the overfitting issue of SSM models during scaling up, eliminating the need for pre-training on large-scale datasets.

## Limitations & Future Work

1. Under the self-supervised paradigm, a performance gap remains between VideoMamba and UMT due to architectural inconsistencies (82.0% vs 85.7%); cross-architecture distillation warrants further in-depth study.
2. Highly scaled models (VideoMamba-B) are still constrained/excluded even with self-distillation, leaving the upper limit of scalability unclear.
3. The scale of multimodal pre-training experiments is relatively small (WebVid-2M + CC3M), and its performance on larger-scale data remains unknown.
4. The pure SSM architecture might underperform hybrid architectures like UniFormer in tasks requiring explicit local interactions.

## Related Work & Insights

- **Vision Mamba / VMamba**: VideoMamba is built upon Vim but removes the middle CLS token and RoPE, yielding a +0.8% improvement on ImageNet.
- **UniFormer**: A representative of hybrid CNN+Attention architectures, whereas VideoMamba demonstrates that a pure SSM can achieve comparable performance.
- **ViS4mer**: An early attempt to apply S4 to long videos (feature-based), which VideoMamba substantially outperforms via end-to-end training.

## Rating

- Novelty: ⭐⭐⭐⭐ First work to comprehensively adapt Mamba to video understanding, with an ingenious self-distillation strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers ImageNet, K400, SSV2, Breakfast, COIN, LVU, and multimodal retrieval with extremely detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The organization of the four major capacities is clear, though table densities are somewhat high.
- Value: ⭐⭐⭐⭐⭐ Pioneering work that lays the foundation for video SSMs, with fully open-sourced code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](videomamba_spatio-temporal_selective_state_space_model.md)
- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](../../NeurIPS2025/video_understanding/pass_path-selective_state_space_model_for_event-based_recognition.md)
- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](../../CVPR2025/video_understanding/mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [\[ECCV 2024\] PiTe: Pixel-Temporal Alignment for Large Video-Language Model](pite_pixel-temporal_alignment_for_large_video-language_model.md)
- [\[ECCV 2024\] Towards Model-Agnostic Dataset Condensation by Heterogeneous Models](towards_model-agnostic_dataset_condensation_by_heterogeneous_models.md)

</div>

<!-- RELATED:END -->
