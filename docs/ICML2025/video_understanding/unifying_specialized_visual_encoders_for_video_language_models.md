---
title: >-
  [Paper Note] Unifying Specialized Visual Encoders for Video Language Models
description: >-
  [ICML 2025][Video Understanding][Multi-encoder fusion] MERV proposes a multi-encoder video representation method that integrates four visual encoders with different areas of expertise (DINOv2, ViViT, SigLIP, LanguageBind) into a single VideoLLM through spatio-temporal alignment and cross-attention fusion. It improves performance on video reasoning benchmarks by up to 4.62% compared to the baseline Video-LLaVA, validating the complementary strengths of different encoders.
tags:
  - "ICML 2025"
  - "Video Understanding"
  - "Multi-encoder fusion"
  - "VideoLLM"
  - "Visual representation"
  - "Cross-attention"
date: 2026-05-08
content_hash: 4facd3de058a5766
---

# Unifying Specialized Visual Encoders for Video Language Models

**Conference**: ICML 2025  
**arXiv**: [2501.01426](https://arxiv.org/abs/2501.01426)  
**Code**: [Available](https://github.com/princetonvisualai/merv)  
**Area**: Video Understanding / Video Language Models  
**Keywords**: Multi-encoder fusion, VideoLLM, Visual representation, Cross-attention, Video understanding

## TL;DR

MERV proposes a multi-encoder video representation method that integrates four visual encoders with different areas of expertise (DINOv2, ViViT, SigLIP, LanguageBind) into a single VideoLLM through spatio-temporal alignment and cross-attention fusion. It improves performance on video reasoning benchmarks by up to 4.62% compared to the baseline Video-LLaVA, validating the complementary strengths of different encoders.

## Background & Motivation

### Background
Current VideoLLMs (such as Video-LLaVA) rely on a single visual encoder (typically contrastive learning models like CLIP or LanguageBind), which limits the upper bound of the model's capabilities. Different encoders possess distinct strengths across various tasks—CLIP excels at vision-language alignment but lacks fine-grained object understanding, DINOv2 is strong at object-level comprehension but weak in language grounding, and ViViT is proficient at temporal modeling but deficient in language understanding.

### Limitations of Prior Work
The inherent weaknesses of a single encoder directly limit the reasoning capacity of VideoLLMs. For instance, certain questions can only be answered correctly by models using ViViT (which requires temporal reasoning), while others can only be solved by CLIP-based models (which requires semantic understanding). Traditional views assume that employing multiple encoders introduces redundant computational overhead, but this assumption overlooks the complementary value of different encoders.

### Ours
This paper proposes MERV, which integrates the features of four encoders into a unified representation via (1) spatio-temporal alignment of each encoder's output; (2) a lightweight pre-fusion projection; and (3) a cross-attention blending strategy. It utilizes parallel visual processing to minimize computational overhead.

## Method

### Overall Architecture

MERV follows the LLaVA/PrefixLM paradigm. The video input is fed into four visual encoders to extract features, which are then spatio-temporally aligned and fused via cross-attention. Finally, they are concatenated with text tokens and input to LLaMA-2 7B. The four encoders process inputs in parallel, allowing training to complete within 24 hours on 8 L40 GPUs.

### Key Designs

1. **Multi-Encoder Feature Extraction**: Selects four complementary encoders:

    - **Spatial Expert DINOv2**: Unsupervised learning, with strong understanding of object parts and semantics.
    - **Temporal Expert ViViT**: Video-supervised learning, modeling long-term dependencies with space-time attention.
    - **Image-Language Contrastive Expert SigLIP**: Sigmoid contrastive learning, understanding vision-language associations via joint embedding space.
    - **Video-Language Contrastive Expert LanguageBind**: Multimodal joint learning, understanding high-level semantics of video and text.

2. **Spatio-Temporally Aligned Representations**: Different encoders yield different output shapes (e.g., ViViT outputs $8 \times 14 \times 14$, LanguageBind outputs $16 \times 16 \times 16$). Alignment is achieved through:

    - Temporal alignment: Adjust the number of input frames so that each encoder outputs the same temporal dimension $t$.
    - Spatial alignment: Unified spatial dimension $h \times w$ using adaptive 2D average pooling.
    - Dimension projection: A linear layer maps the dimension of different encoders $d_e$ to the LLM dimension $d$.
    $\mathbf{x}_e := \mathcal{P}(\mathbf{v}_e) W_e \in \mathbb{R}^{\ell \times d}, \quad \ell = t \times h \times w$
   The projector contains only $d \times \sum_e d_e$ trainable parameters, making it highly lightweight.

3. **Cross-Attention Feature Fusion**: Uses a single learnable query $\mathbf{Q} \in \mathbb{R}^{1 \times d}$. The key is the mean of the feature sequence of each encoder $\overline{\mathbf{X}} \in \mathbb{R}^{N \times d}$, and the value is the original feature $\mathbf{X} \in \mathbb{R}^{N \times \ell \times d}$:
    $$ \mathbf{O} = \text{Softmax}\left(\frac{\mathbf{Q}\overline{\mathbf{X}}^\top}{\sqrt{d}}\right) \mathbf{X} \in \mathbb{R}^{\ell \times d} $$
   This produces a weighted linear mixture representation, fusing information from all encoders. The dynamic weights are determined by visual features.

### Loss & Training

Two training strategies:
- **MERV (frozen)**: Only Stage 2 instruction tuning, learning rate $2 \times 10^{-5}$, batch size 128, only training the projector and fusion modules.
- **MERV (full)**: Stage 1 pre-training (unfrozen LLM) + Stage 2 fine-tuning, Stage 1 learning rate $1 \times 10^{-4}$.

MERV (frozen) requires only 43% of the training time of Video-LLaVA while achieving comparable or superior performance.

## Key Experimental Results

### Main Results

| Dataset | Metric | MERV (frozen) | Video-LLaVA | Gain |
|--------|------|---------------|-------------|------|
| MSVD-QA | Acc | 70.97 | 67.74 | +3.23 |
| MSRVTT-QA | Acc | 59.03 | 56.90 | +2.13 |
| TGIF-QA | Acc | 51.10 | 47.99 | +3.11 |
| Perception Test | Acc | 46.21 | 44.22 | +1.99 |
| ActivityNet-QA | Acc | 50.87 | 47.08 | +3.79 |
| NExT-QA | Acc | 63.09 | 59.61 | +3.48 |
| TVQA | Acc | 42.28 | 37.66 | **+4.62** |

MERV (full) achieves 48.41% on the Perception Test, outperforming SeViLA's 46.2% (+2.2%).

### Ablation Study

| Configuration | Average Accuracy | FLOPs | Description |
|------|----------|-------|------|
| Cross-Attention (Default) | 56.83 | 17.19T | Optimal fusion strategy |
| Concat (Seq.) | 54.45 | 43.09T | Sequence concatenation is computationally expensive |
| Concat (Ch.) | 56.64 | 16.29T | Channel concatenation yields similar results |
| Learnable W | 55.01 | 16.24T | Static weights perform poorly |
| 64 tokens/frame | 69.08 (MSVD) | - | Optimal number of projection tokens |
| 2D Avg pooling | 55.86 | 2.1M FLOPs | Optimal projector (zero parameters) |

### Key Findings

1. **Validation of Encoder Complementarity**: Removing any single encoder decreases performance, and the magnitude of the drop is proportional to the strength of that encoder's specialty.
2. **Temporal Expertise of ViViT**: On the temporally sensitive subset of SSv2-MCQ, ViViT achieves 39.77%, outperforming the runner-up by 9.19%, though it lags on the full dataset.
3. **Interpretable Cross-Attention Weights**: High-motion videos activate ViViT, videos containing text activate SigLIP, and static scenes activate DINOv2/LanguageBind.
4. **High Efficiency of Parallel Encoding**: The step-time overhead introduced by extra encoders is minimal, being dominated by the slowest single encoder.

## Highlights & Insights

- **Breaking the Single-Encoder Paradigm**: The first to systematically validate the value of utilizing multiple encoders in VideoLLMs.
- **Elegant Spatio-Temporal Alignment**: Achieves optimal feature projection using only 2D average pooling (zero parameters), which is simple and highly effective.
- **Compelling SSv2-MCQ Analysis**: Quantitatively demonstrates ViViT's temporal understanding advantages (e.g., pushing vs. pulling, left vs. right) through a temporally sensitive subset.
- **Excellent Scalability**: The architecture can easily incorporate more encoders, with the computational overhead absorbed by parallel processing.

## Limitations & Future Work

- The selection of the 4 encoders is heuristic, lacking a systematic encoder search or auto-selection mechanism.
- The dataset is restricted to Video-LLaVA data; higher-quality training data could yield larger performance gains.
- The fusion strategy is input-independent (attention based on sequence means); a more input-adaptive fusion approach could be beneficial.
- Decoupled from other modalities like audio, which might omit valuable information.

## Related Work & Insights

- Related to multi-encoder image LLMs like SPHINX and Cambrian-1, but focuses specifically on the spatio-temporal alignment challenges in the video domain.
- The concept of complementary encoder expertise can inspire other multimodal areas (such as joint audio-visual-language encoding).
- The discovery that 2D average pooling outperforms complex projectors suggests that feature selection might be more critical than feature transformation.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-encoder fusion idea is novel, though the fusion method itself is relatively simple.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 benchmarks, comprehensive ablations, qualitative analysis, and in-depth SSv2 analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, in-depth analysis, and rich visualizations.
- Value: ⭐⭐⭐⭐ Provides new scaling directions and practical methodologies for VideoLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Scaling Video-Language Models to 10K Frames via Hierarchical Differential Distillation](scaling_video-language_models_to_10k_frames_via_hierarchical_differential_distil.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](../../CVPR2025/video_understanding/video_summarization_with_large_language_models.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](../../NeurIPS2025/video_understanding/enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[CVPR 2025\] Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks](../../CVPR2025/video_understanding/omni-rgpt_unifying_image_and_video_region-level_understanding_via_token_marks.md)

</div>

<!-- RELATED:END -->
