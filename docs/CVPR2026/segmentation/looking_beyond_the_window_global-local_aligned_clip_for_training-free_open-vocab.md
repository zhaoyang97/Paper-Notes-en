---
title: >-
  [Paper Note] Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation
description: >-
  [CVPR 2026][Segmentation][Open-vocabulary semantic segmentation] This paper proposes GLA-CLIP to address cross-window semantic inconsistency introduced by sliding-window inference in training-free open-vocabulary semantic segmentation. Three mechanisms—global key-value extension, proxy anchor attention, and dynamic normalization—are introduced to integrate global context across windows, achieving state-of-the-art average mIoU of 44.0% across 8 benchmarks.
tags:
  - CVPR 2026
  - Segmentation
  - Open-vocabulary semantic segmentation
  - CLIP
  - sliding window
  - training-free
  - global-local alignment
date: 2026-05-08
content_hash: db22aaa9e8c2c17a
---

# Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.23030](https://arxiv.org/abs/2603.23030)
**Code**: [https://github.com/2btlFe/GLA-CLIP](https://github.com/2btlFe/GLA-CLIP)
**Area**: Segmentation / Multimodal VLM
**Keywords**: Open-vocabulary semantic segmentation, CLIP, sliding window, training-free, global-local alignment

## TL;DR

This paper proposes GLA-CLIP to address cross-window semantic inconsistency introduced by sliding-window inference in training-free open-vocabulary semantic segmentation. Three mechanisms—global key-value extension, proxy anchor attention, and dynamic normalization—are introduced to integrate global context across windows, achieving state-of-the-art average mIoU of 44.0% across 8 benchmarks.

## Background & Motivation

**Background**: Open-vocabulary semantic segmentation (OVSS) leverages CLIP's vision-language alignment space to enable pixel-level annotation beyond fixed category sets. Training-free approaches have attracted attention for requiring no additional training, with common strategies including modifying CLIP attention mechanisms (e.g., MaskCLIP, SCLIP, ClearCLIP) or incorporating visual foundation model features (e.g., ProxyCLIP with DINO).

**Limitations of Prior Work**: CLIP's pretraining resolution is only 224×224, necessitating a sliding-window strategy for high-resolution images. Since each window is processed independently, semantic inconsistencies arise across windows—pixels near adjacent window boundaries often receive different predictions, producing visible grid-like artifacts. The authors introduce the Boundary Error Rate (BER) to quantify this issue, finding that ProxyCLIP exhibits high prediction inconsistency near window boundaries.

**Key Challenge**: Sliding windows preserve CLIP's pretraining resolution advantage but sacrifice global context—especially when large or continuous semantic regions span multiple windows, preventing each window from accessing complete scene information for consistent inference.

**Goal**: How can each window obtain global contextual information while maintaining local spatial precision, without any training, so as to eliminate cross-window semantic inconsistency?

**Key Insight**: Extend each window's attention range from local to global—allowing query tokens to attend to key-value tokens from all windows, while using proxy anchors to remove local bias and dynamic normalization to adapt to objects of varying scales.

**Core Idea**: Achieve global-local semantic alignment in training-free CLIP via cross-window key-value extension, proxy anchors, and dynamic normalization.

## Method

### Overall Architecture

A high-resolution input image is partitioned into $L$ overlapping windows, each fed into a frozen VFM (e.g., DINO) and CLIP for feature extraction. For each window, VFM features serve as queries, while the concatenation of all windows' VFM features forms the global key, and the concatenated values from the final transformer layer of CLIP across all windows form the global value. Cross-attention aggregation followed by a projection layer yields the final visual feature $\mathbf{F}_{visual}$, which is compared with text features via cosine similarity for classification.

### Key Designs

1. **Key-Value Extension (KVE)**:

    - **Function**: Expands each window's attention range from local to global.
    - **Mechanism**: VFM features from all $L$ windows are concatenated to form the global key $\mathbf{K}_{global} \in \mathbb{R}^{(LN)\times D}$, and CLIP's final transformer layer values are similarly concatenated to form $\mathbf{V}_{global}$. Each window's query $\mathbf{Q}$ attends to the global key via $\mathbf{A}_{ext} = \mathbf{Q}\cdot\mathbf{K}_{global}^\top \in \mathbb{R}^{N\times(LN)}$, and the resulting attention is used to aggregate global values into the final feature.
    - **Design Motivation**: Breaking window boundaries allows local queries to leverage semantically relevant tokens from distant regions, maintaining consistency particularly when large objects span multiple windows.

2. **Proxy Anchor Attention**:

    - **Function**: Eliminates local bias in attention, enabling fair attention allocation between tokens inside and outside the current window.
    - **Mechanism**: Even with global key-value extension, query tokens tend to over-attend to tokens within their own window (since queries are generated from local window features). To address this, for each query token, the set of high-confidence tokens $\mathcal{P}_i^{(0)}$ in the global key with cosine similarity exceeding threshold $\rho$ is identified. Iterative aggregation (akin to mean-shift) yields the proxy $\mathbf{Q}_i^{(T)}$, which resides at the centroid of high-similarity embeddings and naturally balances contributions from inside and outside the window.
    - **Design Motivation**: The original query is constrained by local window context and lacks global awareness. The proxy serves as a stable semantic anchor, distributing attention based on semantic consistency rather than window membership.

3. **Dynamic Normalization**:

    - **Function**: Adaptively adjusts attention intensity based on target scale, preventing small objects from being overwhelmed by irrelevant global tokens.
    - **Mechanism**: Two adaptive variables replace fixed hyperparameters. The offset variable $\mathbf{u} = 1 + \lambda_1\log(1+L)$ grows with the number of windows, suppressing noise from extended tokens. The scaling variable $\mathbf{w}_i = 1 + \lambda_2 / |\mathcal{P}_i|$ is inversely proportional to the number of high-confidence tokens—small objects correspond to few positive tokens, yielding a larger $\mathbf{w}_i$ that amplifies the weight of relevant tokens.
    - **Design Motivation**: Global attention extension increases the risk of attending to irrelevant tokens, with small objects being especially vulnerable. Dynamic normalization achieves per-query, scale-aware attention modulation, eliminating the need for dataset-specific hyperparameters required by conventional methods.

### Loss & Training

This method is entirely training-free and involves no loss functions or training procedures. All CLIP and DINO parameters are fully frozen; improvements are achieved solely by modifying the attention computation at inference time. Hyperparameters $\rho=0.6$, proxy iteration steps $T=2$, $\lambda_1=0.3$, and $\lambda_2=30$ are shared across datasets.

## Key Experimental Results

### Main Results

| Dataset | Metric | ProxyCLIP | GLA-CLIP (Ours) | Gain |
|--------|--------|-----------|----------------|------|
| Pascal VOC21 | mIoU | 61.3 | 66.3 | +5.0 |
| Pascal Context60 | mIoU | 35.3 | 36.1 | +0.8 |
| COCO-Object | mIoU | 37.5 | 37.7 | +0.2 |
| ADE20K | mIoU | 20.2 | 20.0 | -0.2 |
| Cityscapes | mIoU | 38.1 | 40.8 | +2.7 |
| **Average (8 datasets)** | mIoU | 42.3 | 44.0 | +1.7 |

Note: GLA-CLIP uses no dataset-specific hyperparameters yet still surpasses CASS (44.4 with dataset-specific hyperparameters).

### Ablation Study

| Configuration | Avg. mIoU | Notes |
|------|----------|------|
| Baseline (intra-window DINO attention only) | 30.8 | No normalization, no extension |
| + KVE + Dynamic Normalization | 43.1 | Global extension yields substantial gains |
| + Proxy + Dynamic Normalization (no KVE) | 43.0 | Proxy alone is also effective |
| + KVE + Proxy + Dynamic Normalization | 44.0 | Full method achieves best performance |
| + KVE + Proxy + Dataset-specific hyperparams | 44.3 | Manual tuning yields only marginal improvement |

### Key Findings

- Dynamic normalization adapts across datasets, eliminating the need for per-dataset hyperparameter tuning.
- The number of high-confidence tokens is highly correlated with object scale: in Cityscapes, the Road class yields approximately 135 positive tokens while the Person class yields only approximately 5.
- GLA-CLIP functions as a plug-in module, improving multiple baselines: ClearCLIP by +1.2%, SCLIP by +1.6%, and ProxyCLIP by +0.6%.

## Highlights & Insights

- This work is the first to systematically identify and address the problem of cross-window semantic inconsistency in sliding-window inference.
- The three components (KVE, Proxy Anchor, Dynamic Normalization) form a progressive and mutually reinforcing pipeline: KVE provides global information → Proxy eliminates local bias → Dynamic Normalization handles scale variation.
- The method is entirely training-free with no additional parameters, and can be applied as a plug-and-play extension to existing methods.
- The number of high-confidence tokens serves as a free proxy signal for object scale, elegantly avoiding explicit scale estimation.

## Limitations & Future Work

- The computational complexity of global key-value extension is $O(N \cdot LN)$, and attention overhead grows with the number of windows.
- Iterative construction of proxy anchors introduces additional inference latency, albeit requiring only 2 steps.
- $\lambda_1$ and $\lambda_2$ in dynamic normalization remain globally fixed values; more refined adaptive strategies may yield further improvements.
- Gains on datasets with a large number of categories, such as ADE20K, remain limited, potentially requiring more sophisticated category-aware mechanisms.

## Related Work & Insights

- ProxyCLIP's proxy attention mechanism serves as the direct foundation of this work; GLA-CLIP extends it from single-window to global scope.
- The mean-shift clustering concept is adopted for constructing proxy anchors and is applicable to other settings requiring cross-region semantic consistency.
- Training-free OVSS is emerging as an active research direction, circumventing the need for training data and mitigating overfitting risks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Systematically addresses the previously overlooked problem of cross-window semantic inconsistency.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluations on 8 datasets, integration with multiple baselines, and detailed ablation and visualization analyses.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation, precise problem definition (BER metric), and rigorous methodological derivation.
- **Value**: ⭐⭐⭐⭐ A general plug-and-play solution with practical impact on the training-free OVSS field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [\[CVPR 2026\] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](../../ICCV2025/segmentation/training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2026\] DeBias-CLIP: CLIP Is Shortsighted — Paying Attention Beyond the First Sentence](clip_is_shortsighted_paying_attention_beyond_the_first_sentence.md)
- [\[CVPR 2026\] INSID3: Training-Free In-Context Segmentation with DINOv3](insid3_training-free_in-context_segmentation_with_dinov3.md)

</div>

<!-- RELATED:END -->
