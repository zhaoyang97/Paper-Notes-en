---
title: >-
  [Paper Note] Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection
description: >-
  [CVPR 2025][LLM Pretraining][CLIP training efficiency] This paper proposes CLIP-PGS (Patch Generation-to-Selection), a simple yet effective masking strategy. Through a progressive "generation-to-selection" process—pre-selecting candidate masked patches, preserving critical semantic regions with Sobel edge detection, and then refining the selection using optimal transport normalization—it improves CLIP training efficiency (reducing training time to $0.5\text{--}0.6\times$) whi…
tags:
  - "CVPR 2025"
  - "LLM Pretraining"
  - "CLIP training efficiency"
  - "patch masking strategy"
  - "edge detection"
  - "optimal transport regularization"
  - "semantic preservation"
date: 2026-05-08
content_hash: d4d4076275006dc7
---

# Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection

**Conference**: CVPR 2025  
**arXiv**: [2503.17080](https://arxiv.org/abs/2503.17080)  
**Code**: [GitHub](https://github.com/NUST-Machine-Intelligence-Laboratory/CLIP-PGS)  
**Area**: LLM Evaluation  
**Keywords**: CLIP training efficiency, patch masking strategy, edge detection, optimal transport regularization, semantic preservation

## TL;DR

This paper proposes CLIP-PGS (Patch Generation-to-Selection), a simple yet effective masking strategy. Through a progressive "generation-to-selection" process—pre-selecting candidate masked patches, preserving critical semantic regions with Sobel edge detection, and then refining the selection using optimal transport normalization—it improves CLIP training efficiency (reducing training time to $0.5\text{--}0.6\times$) while achieving state-of-the-art (SOTA) performance on zero-shot classification, retrieval, and other tasks.

## Background & Motivation

**Background**: Vision-language pre-training models like CLIP have demonstrated powerful zero-shot capabilities by learning from large-scale image-text pairs. However, training is extremely computationally expensive. Recently, masking strategies (such as FLIP, MaskCLIP, A-CLIP, and E-CLIP) have improved training efficiency by selectively removing image patches.

**Limitations of Prior Work**:
- **Random Masking** (FLIP): May accidentally remove critical semantic regions, disrupting visual-text alignment.
- **Attention-based Masking** (A-CLIP): Requires extra attention modules, increasing computational complexity.
- **Clustering-based Masking** (E-CLIP): Maintains coherent visual structures but lacks fine-grained semantic preservation, potentially masking regions that correspond to text descriptions.

**Key Challenge**: A higher masking ratio accelerates training, but more masking increases the likelihood of losing critical semantic information, which harms alignment quality. How can semantic integrity be maximized under high masking ratios?

**Goal**: To design a masking strategy that efficiently reduces the number of input patches while preserving key semantic content.

**Key Insight**: A progressive generation-to-selection process—first coarsely filtering candidate patches, then utilizing edge detection to protect main object regions, and finally employing inter-patch similarity paired with optimal transport for refined selection.

**Core Idea**: Protect object boundaries through edge detection and balance patch similarity distribution via optimal transport normalization, achieving precise masking that "masks heavily without covering critical regions."

## Method

### Overall Architecture

CLIP-PGS incorporates a preprocessing step into the standard CLIP training workflow: before the image enters the ViT image encoder, the PGS strategy is applied to select which patches to retain. The text side remains unchanged. Finally, the standard InfoNCE contrastive loss is used for alignment.

### Key Designs

1. **Progressive Dynamic Masking Ratio**:
    - **Function**: Starts with a lower initial masking ratio and gradually increases it to the target masking ratio.
    - **Mechanism**: Initially applies a random masking ratio of only $5\%$ (compared to FLIP's $50\%$) to pre-select a small number of candidate patches as potential mask regions. Two variants are proposed: CLIP-PGS_0.5 (fixed $0.5$ masking ratio) and CLIP-PGS_0.3 (dynamically adjusted between $[0.3, 0.5]$).
    - **Design Motivation**: Progressively expanding the masked area starting from a small ratio preserves key semantics far better than a one-step random masking strategy.

2. **Sobel Edge Detection (ED)**:
    - **Function**: Generates edge maps to protect object boundaries and high-contrast regions.
    - **Mechanism**: Applies the Sobel operator to the entire image to generate an edge map. If a patch initially marked for masking exhibits a high edge score, it is retained; candidate patches with low edge scores are more likely to be masked. The additional computational overhead is only around $1\%$.
    - **Design Motivation**: Object boundaries are the most semantically dense regions; thus, protecting edge areas is equivalent to protecting the most critical semantic info.

3. **Optimal Transport Normalization (OTN)**:
    - **Function**: Optimizes mask selection by balancing the similarity distribution between patches.
    - **Mechanism**: Computes the cosine similarity matrix $S$ between patches, fusing feature similarity with image similarity (where the weight $\alpha$ is adjusted dynamically across training epochs). The Sinkhorn algorithm is then used to iteratively normalize $S$ into a doubly stochastic matrix, yielding balanced similarity scores. It retains patches with high similarity to neighboring patches (acting as representatives of redundant regions that can be safely masked) and masks patches with low similarity (which contain unique information).
    - **Design Motivation**: Edge detection alone cannot capture the semantic redundancy relationships among patches. OTN supplements this at the feature level. The additional overhead is around $1\%$.

### Loss & Training

- **Loss Function**: Standard InfoNCE contrastive loss (same as CLIP).
- **Training Setup**:
    - Dataset: CC12M (approx. 12 million image-text pairs)
    - Architecture: ViT-B/16 image encoder + 12-layer text encoder (512-dim, 8 heads)
    - Optimizer: AdamW, $\text{lr}=1\times 10^{-3}$, $\beta_1=0.9$, $\beta_2=0.98$, weight decay $0.2$
    - Training: 32 epochs, batch size 4096, 8 × V100 GPUs
    - Computational overhead: ED + OTN total $< 3\%$

## Key Experimental Results

### Main Results

**Zero-Shot Classification (Average Top-1 Accuracy across 17 datasets)**:

| Method | Training Time | Average Accuracy |
|------|---------|-----------|
| CLIP | 1.0× | 35.1% |
| FLIP | 0.5× | 33.0% |
| A-CLIP | 1.1× | 35.9% |
| E-CLIP | 0.6× | 36.9% |
| CLIP-PGS_0.5 | **0.5×** | 37.6% |
| CLIP-PGS_0.3 | **0.6×** | **39.5%** |

**Zero-Shot Retrieval (MS-COCO Text R@1 / Image R@1)**:
- CLIP-PGS_0.3: $36.0\% / 25.1\%$ (both are the best)

**Linear Probing (ImageNet-1K)**:
- CLIP-PGS_0.3: $64.4\%$ (vs. E-CLIP $62.7\%$, CLIP $62.3\%$)

**Robustness Evaluation (Average over ImageNet variants)**:
- CLIP-PGS_0.3: $32.9\%$ overall average, $31.8\%$ OOD average (both are the best)

### Ablation Study

| Configuration | ZS (IN-1K) | LP (IN-1K) | TR (COCO) | IR (COCO) |
|------|-----------|-----------|-----------|-----------|
| CLIP Baseline | 36.1 | 62.3 | 34.6 | 23.5 |
| FLIP Random Masking | 34.4 | 61.3 | 32.6 | 22.6 |
| PGS_0.3 (w/o ED, w/o OTN) | 35.9 | 61.7 | 33.5 | 23.0 |
| PGS_0.3 + ED | 36.8 | 63.2 | 34.3 | 24.0 |
| PGS_0.3 + OTN | 36.7 | 63.0 | 34.5 | 23.8 |
| PGS_0.3 + ED + OTN | **38.6** | **64.4** | **36.0** | **25.1** |

### Key Findings

- **ED and OTN are complementary**: Each is effective individually, and their combination yields the best results ($38.6\%$ vs $36.8\%/36.7\%$).
- **Sobel outperforms Canny**: Sobel edge detection slightly outperforms Canny ($38.6\%$ vs $38.5\%$).
- **Initial masking ratio of $5\%$ is optimal**: An excessively high initial masking ratio degrades performance.
- **Dynamic masking ratio is superior**: PGS_0.3 (dynamic $0.3\text{--}0.5$) outperforms PGS_0.5 (fixed $0.5$).
- **ViT-B/16 is the best**: ViT-B/16 > ViT-S/16 > ViT-B/32.
- **Minimal computational overhead**: The total extra overhead of ED + OTN is $< 3\%$.

## Highlights & Insights

1. **Simple yet effective methodology**: A three-stage progressive strategy (pre-selection $\rightarrow$ edge preservation $\rightarrow$ OTN refinement) that requires no additional trainable modules.
2. **Comprehensive outperformance over prior work**: Outperforms CLIP, FLIP, A-CLIP, and E-CLIP across the board under equivalent training times ($0.5\text{--}0.6\times$).
3. **Semantic preservation strategy**: The combination of edge detection and optimal transport elegantly preserves semantic information from two distinct perspectives.
4. **Minimal overhead**: Achieves superior results within the same training timeframe as FLIP, with ED + OTN introducing $< 3\%$ overall overhead.
5. **Improved linguistic compositionality**: Improvements on the SugarCrepe dataset show that a better masking strategy indeed enhances the quality of vision-language alignment.

## Limitations & Future Work

1. **Limited dataset scale**: Only trained on CC12M; performance on larger datasets (e.g., LAION) remains unverified.
2. **Restricted to ViT architectures**: Future work could extend the method to CNN-based architectures (e.g., ConvNeXt).
3. **Limited to dual-encoder models**: Currently designed specifically for CLIP's dual-encoder setup; future work could explore adaptation to self-supervised methods like MAE.
4. **Heuristic masking strategy**: The current strategy is heuristic; future work could investigate end-to-end learnable mask selection.

## Related Work & Insights

- **CLIP**: The foundational contrastive learning framework.
- **FLIP**: Pioneering work using random masking to improve CLIP training efficiency.
- **A-CLIP**: Attention-based adaptive masking.
- **E-CLIP**: Clustering-based masking strategy.
- **MaskCLIP**: Self-distillation combining masked image modeling with contrastive learning.
- **Insights for Future Research**: Traditional CV tools like edge detection and optimal transport remain highly useful in deep learning preprocessing.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] The Double-Ellipsoid Geometry of CLIP](../../ICML2025/llm_pretraining/the_double-ellipsoid_geometry_of_clip.md)
- [\[CVPR 2025\] Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction](improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio.md)
- [\[CVPR 2025\] ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)
- [\[ICML 2025\] Whitened CLIP as a Likelihood Surrogate of Images and Captions](../../ICML2025/llm_pretraining/whitened_clip_as_a_likelihood_surrogate_of_images_and_captions.md)
- [\[ICCV 2025\] SynCity: Training-Free Generation of 3D Worlds](../../ICCV2025/llm_pretraining/syncity_training-free_generation_of_3d_worlds.md)

</div>

<!-- RELATED:END -->
