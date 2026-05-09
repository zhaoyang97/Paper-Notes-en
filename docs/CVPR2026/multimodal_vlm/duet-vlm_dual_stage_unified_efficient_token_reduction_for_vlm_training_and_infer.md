---
title: >-
  [Paper Note] DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference
description: >-
  [CVPR2026][Multimodal VLM][VLM token compression] This paper proposes DUET-VLM, a dual-stage visual token compression framework. Stage 1 operates within the visual encoder: dominant tokens are selected via V2V self-attention, and remaining tokens are merged into contextual tokens through attention-guided local cluster aggregation. Stage 2 operates within the LLM, progressively pruning visual tokens via T2V cross-attention across multiple layers. On LLaVA-1.5-7B, DUET-VLM achieves 67% token compression while retaining 99%+ accuracy, and 89% compression while retaining 97%+ accuracy, with a 31% reduction in training time.
tags:
  - CVPR2026
  - Multimodal VLM
  - VLM token compression
  - visual token redundancy
  - dual-stage token pruning
  - attention-guided aggregation
  - hierarchical pruning
date: 2026-05-08
content_hash: e3b3fdd697fba2e5
---

# DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference

**Conference**: CVPR2026  
**arXiv**: [2602.18846](https://arxiv.org/abs/2602.18846)  
**Code**: [https://github.com/AMD-AGI/DUET-VLM](https://github.com/AMD-AGI/DUET-VLM)  
**Area**: Multimodal VLM  
**Keywords**: VLM token compression, visual token redundancy, dual-stage token pruning, attention-guided aggregation, hierarchical pruning

## TL;DR
This paper proposes DUET-VLM, a dual-stage visual token compression framework. Stage 1 operates within the visual encoder: dominant tokens are selected via V2V self-attention, and remaining tokens are merged into contextual tokens through attention-guided local cluster aggregation. Stage 2 operates within the LLM, progressively pruning visual tokens via T2V cross-attention across multiple layers. On LLaVA-1.5-7B, DUET-VLM achieves 67% token compression while retaining 99%+ accuracy, and 89% compression while retaining 97%+ accuracy, with a 31% reduction in training time.

## Background & Motivation

1. **State of the Field**: VLMs (e.g., LLaVA, InternVL) rely on large numbers of visual tokens to convey image information to the LLM, yet visual tokens exhibit severe redundancy—many tokens correspond to background or repetitive texture regions rather than semantically critical content.
2. **Limitations of Prior Work**: Existing token compression methods are **unilateral**—they either compress only on the visual encoder side (VisionZip, HiRED) or only on the LLM side (FastV, PyramidDrop), and cannot leverage information from both sides for optimal compression.
3. **Root Cause**: Vision-only methods lack text-guided signals and cannot determine which visual tokens are truly relevant to the current query; language-only methods can only perform post-hoc processing within the LLM, having already wasted computation in earlier layers.
4. **Paper Goals**: To design a unified dual-stage framework that performs complementary token compression within both the visual encoder and the LLM, applicable to both training and inference.
5. **Starting Point**: Stage 1 uses V2V self-attention among visual tokens for coarse-grained compression; Stage 2 uses T2V cross-attention from text to visual tokens for fine-grained pruning.
6. **Core Idea**: The V2V stage preserves spatial context via attention-guided local cluster aggregation (local clustering with fixed window width $w$ rather than global averaging); the T2V stage progressively drops low-relevance visual tokens through hierarchical pruning.

## Method

### Overall Architecture
DUET-VLM consists of two stages: (1) the V2V (Vision-to-Vision) stage, executed within the final layer of the visual encoder (e.g., CLIP ViT); and (2) the T2V (Text-to-Vision) stage, executed across multiple intermediate layers of the LLM decoder. The two stages are applied sequentially and are compatible with both training and inference.

### Key Designs

1. **V2V Stage — Token Compression within the Visual Encoder**:

    - **Function**: Within the final layer of the visual encoder, V2V self-attention is used to compress $N$ visual tokens into $k_1 + k_2$ tokens.
    - **Mechanism**: (a) Self-attention scores are computed for all visual tokens (column-wise summation yields each token's "attended-to" score), and the top-$k_1$ tokens are selected as **dominant tokens**—those most globally important for semantics. (b) The remaining $N - k_1$ tokens are merged into $k_2$ **contextual tokens** via **attention-guided local cluster aggregation**—for each contextual token center, neighbors within a fixed window of width $w$ are selected and aggregated through a weighted average based on attention scores.
    - **Design Motivation**: Global average pooling (as in VisionZip's contextual tokens) dilutes information by mixing semantically dissimilar tokens. Local clustering with fixed width $w$ ensures that merged tokens are spatially proximate and semantically similar, thereby avoiding information loss.

2. **T2V Stage — Hierarchical Visual Token Pruning within the LLM**:

    - **Function**: Visual tokens are progressively pruned across multiple intermediate layers of the LLM.
    - **Mechanism**: (a) A set $S$ of **salient text tokens** is first selected—comprising the last token (serving as an attention sink) and the text tokens with the highest attention scores. (b) At each pruning stage, T2V cross-attention scores from text tokens in $S$ to visual tokens are computed, and the bottom $\lambda$ fraction of visual tokens by score are discarded. (c) Pruning is applied at multiple stages (e.g., at LLM layers $l_1, l_2, \ldots$), enabling progressive compression.
    - **Design Motivation**: Text tokens encode "what information is needed to answer the query," making T2V attention a natural indicator of visual token relevance. Hierarchical pruning is safer than one-shot pruning, as shallow-layer attention is less mature and progressive decisions are more reliable.

3. **Dual-Stage Compression during Training**:

    - **Function**: Both compression stages are applied during training to reduce resource consumption.
    - **Mechanism**: The same compression strategy used at inference is applied during training, reducing FLOPs and memory by decreasing the number of tokens fed to the LLM. Dominant/contextual token selection and T2V pruning both employ a straight-through estimator to preserve gradient flow.
    - **Design Motivation**: Most existing methods (FastV, PyramidDrop) compress only at inference while leaving training costs unchanged. DUET-VLM unifies the compression strategy across training and inference, enabling training-time acceleration.

### Loss & Training
- Standard autoregressive language modeling loss, consistent with LLaVA.
- Dual-stage token compression is applied directly during training without additional distillation or auxiliary losses.
- V2V stage hyperparameters $k_1, k_2, w$ and T2V stage hyperparameters $\lambda$ and pruning layers are treated as fixed hyperparameters.

## Key Experimental Results

### Main Results — LLaVA-1.5-7B Inference

| Method | Token Compression Rate | Retained Accuracy | Notes |
|--------|----------------------|-------------------|-------|
| FastV | 50%↓ | ~98% | LLM-only pruning |
| PyramidDrop | 50%↓ | ~98% | LLM-only hierarchical |
| VisionZip | 67%↓ | ~97% | Vision-only |
| HiRED | 67%↓ | ~96% | Vision-only hierarchical |
| FitPrune | 67%↓ | ~98% | Training-aware pruning |
| **DUET-VLM** | **67%↓** | **99%+** | Dual-stage |
| **DUET-VLM** | **89%↓** | **97%+** | Dual-stage extreme compression |

### Dual-Stage Compression during Training

| Compression Rate | Retained Accuracy | Training Time Saved |
|-----------------|-------------------|---------------------|
| 67%↓ | 99.7% | ~31% |
| 89%↓ | 97.6% | ~31% |

### Video-LLaVA-7B

| Compression Rate | Retained Accuracy | Notes |
|-----------------|-------------------|-------|
| 53.1%↓ | 100%+ (exceeds baseline) | Compression improves over baseline |
| 93.4%↓ | 97.6% | Extreme compression |

### Key Findings
- **Dual-stage > Unilateral**: V2V-only or T2V-only compression is inferior to their combination, confirming that the two stages provide complementary information.
- **Local clustering > Global averaging**: Local cluster aggregation with fixed width $w$ substantially outperforms VisionZip's global contextual token strategy.
- **Video benefits more**: Video-LLaVA surpasses the baseline at 53.1% compression, indicating that token redundancy is more severe in video and that moderate compression acts as denoising.
- **Training compression is viable**: Applying 67% compression during training incurs only 0.3% accuracy loss while reducing training time by 31%.
- **Outperforms all prior methods**: At equivalent compression rates, DUET-VLM surpasses VisionZip, FastV, PyramidDrop, HiRED, and FitPrune across all benchmarks.

## Highlights & Insights
- **Philosophy of complementary dual-stage design**: The V2V stage performs coarse filtering using intra-visual information (independent of text); the T2V stage performs fine-grained filtering guided by text. The two stages address redundancy at different levels, avoiding the information blind spots of unilateral methods.
- **Simplicity and effectiveness of local clustering**: Using a fixed window width $w$ for local clustering avoids complex global clustering algorithms (e.g., k-means), incurring minimal computational overhead while significantly outperforming global averaging.
- **Unified training and inference**: Most token compression methods target inference efficiency alone; DUET-VLM is effective at training time as well, offering practical value in large-scale VLM training.
- **Video compression surpasses baseline**: Accuracy improves at 53.1% compression, indicating that redundant tokens not only waste resources but also introduce noise.

## Limitations & Future Work
- Validation is limited to LLaVA-1.5-7B and Video-LLaVA-7B; experiments on larger-scale models (e.g., 13B/34B) are absent.
- The V2V stage hyperparameters $k_1, k_2, w$ are fixed; adaptive adjustment for different images or tasks may be necessary.
- Compatibility with more recent architectures such as InternVL2 and Qwen-VL has not been verified.
- The salient text token selection in the T2V stage relies on the attention sink assumption; robustness to non-standard prompt formats remains unknown.
- In-depth analysis of how attention patterns change after compression is lacking.

## Related Work & Insights
- **vs. VisionZip**: VisionZip selects dominant and contextual tokens on the visual encoder side, but contextual tokens are computed via global averaging, leading to information dilution. DUET-VLM replaces this with local cluster aggregation and adds the T2V stage.
- **vs. FastV/PyramidDrop**: Both perform attention-based pruning on the LLM side but lack preliminary filtering in the visual encoder. The V2V stage in DUET-VLM provides coarse pre-filtering, reducing the burden on the T2V stage.
- **vs. FitPrune**: FitPrune optimizes pruning strategies in a training-aware manner but remains a unilateral method. DUET-VLM applies dual-stage compression in both training and inference.
- **vs. HiRED**: HiRED performs hierarchical attention-based compression within the visual encoder but does not involve the LLM side. DUET-VLM applies hierarchical compression on both sides.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual-stage V2V+T2V framework is proposed for the first time; the local cluster aggregation design is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both image and video settings, validates both training and inference, and includes complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, method descriptions are detailed, and figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ A practical solution for VLM token compression; the 31% training acceleration carries significant engineering value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [\[CVPR 2026\] ApET: Approximation-Error Guided Token Compression for Efficient VLMs](apet_approximation-error_guided_token_compression_for_efficient_vlms.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] Scene-VLM: Multimodal Video Scene Segmentation via Vision-Language Models](scene-vlm_multimodal_video_scene_segmentation_via_vision-language_models.md)
- [\[CVPR 2026\] SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning](spatialstack_layered_geometry-language_fusion_for_3d_vlm_spatial_reasoning.md)

<!-- RELATED:END -->
