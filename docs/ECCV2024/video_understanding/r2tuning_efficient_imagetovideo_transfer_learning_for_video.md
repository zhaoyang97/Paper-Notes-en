---
title: >-
  [Paper Note] R²-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding
description: >-
  [ECCV 2024][Video Understanding][Video Temporal Grounding] This paper proposes R²-Tuning, which appends a lightweight R² Block (only 1.5% of total parameters) recursively in a backward manner onto the last several layers of a frozen CLIP model. It enables query-modulated spatial pooling and coarse-to-fine temporal refinement, outperforming state-of-the-art (SOTA) methods that require additional temporal backbones on 6 VTG benchmarks across 3 tasks with only 2.7M parameters.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Video Temporal Grounding"
  - "CLIP"
  - "Transfer Learning"
  - "Parameter-Efficient"
  - "Temporal Modeling"
date: 2026-05-08
content_hash: e2b5f931d12438ed
---

# R²-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding

**Conference**: ECCV 2024  
**arXiv**: [2404.00801](https://arxiv.org/abs/2404.00801)  
**Code**: [https://github.com/yeliudev/R2-Tuning](https://github.com/yeliudev/R2-Tuning)  
**Area**: Video Understanding  
**Keywords**: Video Temporal Grounding, CLIP, Transfer Learning, Parameter-Efficient, Temporal Modeling

## TL;DR

This paper proposes R²-Tuning, which appends a lightweight R² Block (only 1.5% of total parameters) recursively in a backward manner onto the last several layers of a frozen CLIP model. It enables query-modulated spatial pooling and coarse-to-fine temporal refinement, outperforming state-of-the-art (SOTA) methods that require additional temporal backbones on 6 VTG benchmarks across 3 tasks with only 2.7M parameters.

## Background & Motivation

**Background**: Video Temporal Grounding (VTG) is a fine-grained video-language understanding problem, containing three sub-tasks: Moment Retrieval (MR), Highlight Detection (HD), and Video Summarization (VS). Current mainstream methods rely on the frame-level final-layer features extracted by CLIP, supplemented by additional temporal backbones (e.g., SlowFast) and meticulously designed temporal reasoning modules.

**Limitations of Prior Work**: This "post-processing" paradigm suffers from two fundamental limitations. First, simultaneously utilizing two backbones with similar capabilities is counter-intuitive and computationally inefficient; a single model with both visual-text alignment and spatio-temporal modeling capabilities is more desirable. Second, the granularity of VTG queries varies widely from coarse (e.g., "a family traveling") to fine (e.g., "the moment when the white-haired man hands me the golf club"). Relying solely on final-layer frame-level features fails to flexibly adapt to different granularities.

**Key Challenge**: CLIP itself possesses strong potential for spatio-temporal modeling—where each of its layers encodes useful information at different granularities—but this potential is severely underutilized by existing "last-layer-only" approaches. Preliminary experiments also confirm that methods utilizing only the final-layer CLIP features fall far short of unleashing the full temporal modeling capabilities of CLIP.

**Goal**: How to efficiently transfer vision-language foundation models to video temporal grounding? Specifically, the goals are: (1) parameter- and memory-efficient; (2) flexible in granularity to adapt to queries of varying complexities.

**Key Insight**: The authors observe that multi-layer CLIP features provide progressive information from low-level details to high-level semantics, which can be fused recursively in a backward (last-to-first) manner to achieve coarse-to-fine spatio-temporal modeling. The key is to design a lightweight side network so that gradients do not need to backpropagate through the frozen CLIP encoders.

**Core Idea**: A weight-shared, lightweight R² Block is recursively attached in a backward manner to the last $K$ layers of the frozen CLIP model. It sequentially performs query-modulated spatial pooling and temporal refinement, achieving complete spatio-temporal modeling with only 1.5% additional parameters.

## Method

### Overall Architecture

The input video $V$ and text query $Q$ are passed through the frozen CLIP vision and text encoders, respectively, to obtain multi-layer features $e_v \in \mathbb{R}^{B \times N \times T \times (P+1) \times D_v}$ and $e_q \in \mathbb{R}^{B \times N \times L \times D_q}$. A learnable R² Block processes the features of the last $K$ layers recursively in a backward direction (from the final layer to the front), maintaining and progressively refining a hidden state $h \in \mathbb{R}^{B \times T \times C}$ as the frame-level spatio-temporal feature. Once the refinement is complete, $h$ is used to build a temporal feature pyramid via 1D convolutions, which are finally passed to three task heads to predict boundary regression for MR, saliency scores for HD, and foreground/background classification for VS.

### Key Designs

1. **Query-Modulated Spatial Pooling**:

    - **Function**: Adapts the pooling of patch-level features in each frame into a single token based on the text query.
    - **Mechanism**: First, two MLPs are used to project the visual features $\hat{e}_v^n$ and query features $\hat{e}_q^n$ into the same space, and then the similarity between each token-patch pair is computed as $a = \text{softmax}(\frac{(w_q \hat{e}_q^n)^\top w_v \hat{e}_v^n}{\sqrt{C}})$. This attention weight is used to pool the visual features into individual tokens, which are then max-pooled along the token dimension to obtain $e_{token}^n$. Finally, $e_{token}^n$ is combined with the [CLS] token via a zero-initialized learnable gating factor $g^k \in (-1, 1)$: $e_{pool}^n = e_v^{n,0} + g^k \cdot e_{token}^n$.
    - **Design Motivation**: Different queries attend to different regions of video frames. By guiding the spatial pooling with the query via a cross-attention mechanism, the model focuses on query-relevant spatial regions. The gating factor allows negative values to remove irrelevant information from the [CLS] token.

2. **Recurrent Temporal Refinement**:

    - **Function**: Merges and refines temporal features layer-by-layer starting from the final layer of CLIP and moving backward.
    - **Mechanism**: In each step $k$, a learnable gating factor $\varphi^k \in (0,1)$ is first used to fuse the pooled features of the current layer $e_{pool}^n$ with the previous hidden state $h^{k-1}$: $\hat{h}^{k-1} = \varphi^k \cdot e_{pool}^n + (1-\varphi^k) \cdot h^{k-1}$. The hidden state is then updated sequentially through multi-head cross-attention (with the query as key/value), multi-head self-attention, and an FFN: $h^k = \text{FFN}(\text{MHSA}(\text{MHCA}(\hat{h}^{k-1}, \hat{e}_q^n)))$.
    - **Design Motivation**: The backward fusion order (from last to first) achieves a "coarse-to-fine" spatio-temporal modeling—capturing high-level semantic outlines first, then progressively incorporating low-level details. Recursively sharing weights significantly reduces the parameter count.

3. **Granularity Calibration**:

    - **Function**: Aligns the feature granularity between CLIP visual and text encoders across different layers.
    - **Mechanism**: Two contrastive losses are designed for calibration. The video-level contrastive loss $\mathcal{L}_{video}$ performs contrastive learning across samples in the same batch and averages over $K$ layers, ensuring the diversity of features for different video-query pairs. The layer-level contrastive loss $\mathcal{L}_{layer}$ performs contrastive learning across different layers, encouraging different layers to distill distinctive information.
    - **Design Motivation**: The visual and textual encoders of CLIP are trained independently during pre-training, which does not guarantee that the visual and textual features at the same layer correspond to the same level of granularity. Thus, additional alignment constraints are required.

### Loss & Training

The total loss is the sum of five components: $\mathcal{L} = \mathcal{L}_{video} + \mathcal{L}_{layer} + \mathcal{L}_{cls} + \mathcal{L}_{reg} + \mathcal{L}_{sal}$

| Loss | Weight $\lambda$ | Description |
|------|--------------|------|
| $\mathcal{L}_{video}$ | 0.1 | Video-level contrastive loss |
| $\mathcal{L}_{layer}$ | 0.1 | Layer-level contrastive loss |
| $\mathcal{L}_{cls}$ | 1.0 | Foreground/background classification Focal Loss ($\alpha=0.9, \gamma=2.0$) |
| $\mathcal{L}_{reg}$ | 0.1 | Boundary regression L1 Loss |
| $\mathcal{L}_{sal}$ | 0.1 | Saliency prediction contrastive loss (temperature $\tau=0.07$) |

CLIP is completely frozen, and only the R² Block, the pyramid, and the task heads are trainable, with a total of 2.7M parameters. DropPath ($p=0.1$) is used to prevent over-fitting, and the NMS IoU threshold is set to 0.7.

## Key Experimental Results

### Main Results: Joint Evaluation of MR + HD on QVHighlights Test Set

| Method | Backbone | Extra Pre-training | R1@0.5 | R1@0.7 | mAP Avg. | HD mAP | Params |
|------|------|----------|--------|--------|----------|--------|--------|
| Moment-DETR | CLIP+SlowFast | None | 52.89 | 33.02 | 30.73 | 35.69 | 4.8M |
| UMT | CLIP+SlowFast | None | 56.23 | 41.18 | 36.12 | 38.18 | 14.9M |
| QD-DETR | CLIP+SlowFast | None | 62.40 | 44.98 | 39.86 | 38.94 | 7.6M |
| CG-DETR | CLIP+SlowFast | None | 65.43 | 48.38 | 42.86 | 40.33 | 12.0M |
| TR-DETR | CLIP+SlowFast | None | 64.66 | 48.96 | 42.62 | 39.91 | 7.9M |
| UniVTG | CLIP+SlowFast | 4.2M corpus | 65.43 | 50.06 | 43.63 | 40.54 | 41.3M |
| **R²-Tuning** | **CLIP only** | **None** | **68.03** | **49.35** | **46.17** | **40.75** | **2.7M** |

With only the CLIP backbone (no extra backbone) and 2.7M parameters, R²-Tuning outperforms UniVTG (41.3M parameters), which requires the dual CLIP+SlowFast backbone and 4.2M pre-training corpus. R²-Tuning achieves a 2.54% mAP gain on MR, using only 1/15 of the parameters.

### Ablation Study: Effectiveness of Granularity Calibration (QVHighlights Val Set)

| $\mathcal{L}_{video}$ | $\mathcal{L}_{layer}$ | MR R1@0.5 | MR mAP | HD mAP | HD HIT@1 |
|----------------------|----------------------|-----------|--------|--------|----------|
| ✗ | ✗ | 64.48 | 44.01 | 37.94 | 62.67 |
| ✓ | ✗ | 67.68 | 46.74 | 39.81 | 65.16 |
| ✗ | ✓ | 64.71 | 44.60 | 38.91 | 63.35 |
| ✓ | ✓ | **68.71** | **47.59** | **40.59** | **64.32** |

The video-level contrastive loss contributes the most (MR mAP +2.73), and the combination of both yields the optimal result (+3.58).

### Key Findings

- **Outperforming SOTA Without Extra Backbones**: R²-Tuning proves that CLIP itself has sufficient spatio-temporal modeling potential, rendering additional backbones like SlowFast unnecessary.
- **Significant Advantage in High-Precision Retrieval (R@0.7)**: On Charades-STA, R@0.7 reaches 37.02 (vs UniVTG 35.65); on TACoS, R@0.7 reaches 25.12 (vs UniVTG 17.35, +44.8%), demonstrating strong fine-grained temporal modeling capabilities.
- **Outstanding Generalization on Long Queries**: While training queries are mostly coarse-grained with $\le 30$ words, R²-Tuning achieves an MR mAP of 72.38 on queries with $\ge 41$ words, significantly outperforming QD-DETR (26.67) and UniVTG (31.11).
- **Backward Fusion Surpasses Forward**: "Backward" recursive fusion from high-to-low layers consistently outperforms "forward" fusion from low-to-high layers, validating the superiority of coarse-to-fine modeling.
- **State-of-the-Art on Video Summarization and Highlight Detection**: On YouTube Highlights, the average mAP reaches 76.1 (vs UniVTG 75.2); on TVSum, the average Top-5 mAP is 85.2 (vs UMT 83.1).

## Highlights & Insights

- **Elegant Intuition of "Backward Recursion"**: Merging lower-layer features (detailed textures) progressively by starting from the last layer of CLIP (the most abstract semantics) perfectly matches the cognitive pattern of "looking at the big picture first, then details." This design is simple yet highly effective.
- **Extreme Parameter Efficiency**: 2.7M trainable parameters vs 41.3M (UniVTG) and 87.9M (UnLoc), demonstrating that proper architectural design is more critical than stacking parameters.
- **Transferable Query-Modulated Spatial Pooling**: This approach of guiding visual feature spatial aggregation using textual queries can be generalized to other cross-modal grounding tasks, such as Referring Video Object Segmentation.
- **Contrastive Learning Idea for Granularity Calibration**: Utilizing contrastive losses across both video and layer dimensions to align the granularity of multi-layer features provides a paradigm of orthogonal constraints that can be applied to other multi-layer feature fusion scenarios.

## Limitations & Future Work

- **Evaluated Only on CLIP ViT-B/32**: The method has not been validated on larger-scale CLIP models (e.g., ViT-L/14). Whether its advantages scale to larger models remains to be verified.
- **Fixed Choice of $K=4$**: Using only the last 4 layers is a relatively static choice; more flexible, adaptive layer selection might yield better performance.
- **Limitations of Recursive Weight Sharing**: Using identical parameters for all $K$ steps limits representation capability as different layers might warrant distinct processing strategies.
- **Lack of Comparison with Recent Methods (e.g., GroundingDINO)**: The VTG field is evolving rapidly, and comparisons with more recent baselines are needed.

## Related Work & Insights

- **vs UniVTG (Lin et al., 2023)**: UniVTG unifies the three tasks of MR/HD/VS but relies on a dual CLIP+SlowFast backbone and a 4.2M pre-training corpus. R²-Tuning outperforms it using fewer parameters and requiring no extra backbone or pre-training, proving the value of fully exploiting multi-layer CLIP information.
- **vs EVL (Lin et al., 2022)**: EVL is also a side-tuning method that learns a temporal decoder in parallel alongside CLIP. However, EVL employs forward fusion and lacks query modulation. R²-Tuning's backward recursion + query-guided mechanism proves more effective for VTG.
- **vs QD-DETR (Moon et al., 2023)**: QD-DETR uses query-dependent video representations and dynamic anchors but still relies on pre-extracted offline features. R²-Tuning introduces query guidance right from the feature extraction stage, making it more end-to-end.

## Rating

- Novelty: ⭐⭐⭐⭐ The designs of backward recursive multi-layer feature fusion and query-modulated spatial pooling are novel, though the overall framework still falls into the broader category of adapter/side-tuning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 6 datasets and 3 tasks, with detailed ablation studies and thorough visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, the method description progressive, and illustrations intuitive.
- Value: ⭐⭐⭐⭐ Proves the significant potential of multi-layer CLIP features for VTG, providing a strong baseline for parameter-efficient video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Efficient Transfer Learning for Video-language Foundation Models](../../CVPR2025/video_understanding/efficient_transfer_learning_for_video-language_foundation_models.md)
- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)
- [\[CVPR 2026\] VidPrism: Heterogeneous Mixture of Experts for Image-to-Video Transfer](../../CVPR2026/video_understanding/vidprism_heterogeneous_mixture_of_experts_for_image-to-video_transfer.md)
- [\[CVPR 2026\] Learning to Refuse: Refusal-Aware Reinforcement Fine-Tuning for Hard-Irrelevant Queries in Video Temporal Grounding](../../CVPR2026/video_understanding/learning_to_refuse_refusal-aware_reinforcement_fine-tuning_for_hard-irrelevant_q.md)
- [\[CVPR 2025\] Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding](../../CVPR2025/video_understanding/seq2time_sequential_knowledge_transfer_for_video_llm_temporal_grounding.md)

</div>

<!-- RELATED:END -->
