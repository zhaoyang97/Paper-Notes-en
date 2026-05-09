---
title: >-
  [Paper Note] EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation
description: >-
  [CVPR 2026][Image Generation][video tokenizer] This paper proposes EVATok, a four-stage framework that first uses a proxy tokenizer to estimate the optimal token allocation for each video, then trains a lightweight router to predict these allocations in a single forward pass, and finally trains an adaptive tokenizer that flexibly assigns token counts according to content complexity. On UCF-101, EVATok achieves state-of-the-art generation quality with a 24.4% reduction in token count.
tags:
  - CVPR 2026
  - Image Generation
  - video tokenizer
  - adaptive tokenization
  - autoregressive generation
  - proxy reward
  - Q-Former
date: 2026-05-08
content_hash: 137560709bbdaa79
---

# EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation

**Conference**: CVPR 2026
**arXiv**: [2603.12267](https://arxiv.org/abs/2603.12267)
**Code**: [Project Page](https://silentview.github.io/EVATok/)
**Area**: Image Generation
**Keywords**: video tokenizer, adaptive tokenization, autoregressive generation, proxy reward, Q-Former

## TL;DR

This paper proposes EVATok, a four-stage framework that first uses a proxy tokenizer to estimate the optimal token allocation for each video, then trains a lightweight router to predict these allocations in a single forward pass, and finally trains an adaptive tokenizer that flexibly assigns token counts according to content complexity. On UCF-101, EVATok achieves state-of-the-art generation quality with a 24.4% reduction in token count.

## Background & Motivation

**Background**: The core pipeline of autoregressive video generation first compresses pixels into a discrete token sequence via a video tokenizer, then models the token sequence with an AR model. The length of the token sequence directly determines the computational cost of downstream generation—longer sequences lead to quadratically increasing attention complexity.

**Limitations of Prior Work**: Nearly all video tokenizers allocate the same number of tokens to different videos and different temporal segments. However, information density in video is highly non-uniform—segments with static backgrounds or repetitive textures carry little information, while segments with rapid motion or scene transitions are information-dense. This one-size-fits-all fixed allocation wastes tokens on simple segments (where reconstruction quality has already saturated) and under-allocates tokens to complex segments (leading to quality degradation from under-representation).

**Key Challenge**: Adaptive allocation requires knowing "what the optimal allocation is," but (1) how should "optimal" be defined? A quantifiable quality-efficiency trade-off metric is needed; (2) searching for the optimal allocation per video is computationally prohibitive; (3) the tokenizer architecture must support variable-length input. Prior methods such as ElasticTok rely on threshold-based heuristic search and AdapTok uses mini-batch ILP—both yielding locally sub-optimal solutions.

**Key Insight**: EVATok defines a proxy reward metric to quantify the quality-cost trade-off of a given allocation, uses brute-force search to find the optimal allocation per video as supervision, and trains a lightweight router to predict the optimal allocation in a single forward pass, thereby bypassing the search stage. **Core Idea**: The problem of "finding the optimal allocation" is reformulated as a classification task, replacing expensive per-sample search with a single forward pass of a small model.

## Method

### Overall Architecture

EVATok proceeds through four sequential stages: Stage 1 trains a proxy tokenizer capable of reconstructing video under any token allocation → Stage 2 uses the proxy tokenizer to brute-force search optimal allocations over 100k videos, constructing a (video, optimal allocation) training set → Stage 3 trains a lightweight ViT-S router, modeling optimal allocation prediction as a classification task → Stage 4 uses the router's guidance to train the final adaptive tokenizer from scratch.

### Key Designs

1. **Proxy Reward and Optimal Allocation Definition**:

    - **Function**: Quantify the quality-cost trade-off of each token allocation scheme for each video.
    - **Mechanism**: Define $R_{\text{proxy}} = w_q Q(\mathcal{E},x,a) - w_l L(a)$, where $Q$ is reconstruction quality (normalized LPIPS), $L(a)$ is normalized token length, and $w_q, w_l$ are preference weights. All $5^4=625$ candidate allocations are enumerated per video, and the one maximizing the proxy reward is designated as the optimal allocation $a^*$.
    - **Design Motivation**: Prior methods lack an explicit definition of "optimality" and rely on heuristic search prone to local optima. The proxy reward unifies quality and cost into a single scalar, making the optimal allocation computable and comparable.

2. **Lightweight Router**:

    - **Function**: Predict the optimal token allocation for an input video in a single forward pass, replacing brute-force search.
    - **Mechanism**: A ViT-S architecture (19.9M parameters) patchifies the video and appends a [CLS] token, outputting probabilities over $m^T$ allocation categories. Trained as a classification task with cross-entropy loss on the 100k-sample dataset constructed in Stage 2.
    - **Design Motivation**: Brute-force search requires 625 forward passes per video; the router compresses this to one. Experiments show that the router's predictions approach the Pareto frontier of brute-force search and generalize to datasets unseen during training.

3. **Q-Former-Style 1D Variable-Length Tokenizer**:

    - **Function**: An encode-decode architecture supporting different numbers of tokens for different temporal blocks.
    - **Mechanism**: After spatio-temporal patchification of the input video, 1D queries of varying counts are initialized according to the allocation scheme $a=(k_1,...,k_T)$. These interact with 3D embeddings via Q-Former encoder layers, then undergo VQ quantization to produce discrete tokens. The decoder initializes 3D queries from the first 1D token of each block for reconstruction.
    - **Design Motivation**: This avoids two problems of tail-token-dropping: (1) dropped tail tokens still consume computation during encoding; (2) tail queries have an ambiguous role during encoding (unaware of whether they will be dropped). Fixing the length at the query initialization stage is more efficient.

### Loss & Training

The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{vqgan}} + \lambda \mathcal{L}_{\text{align}} + \gamma \mathcal{L}_{\text{entropy}}$, where:

- $\mathcal{L}_{\text{vqgan}}$: L1 reconstruction + perceptual loss + GAN adversarial loss + VQ codebook loss
- $\mathcal{L}_{\text{align}}$: cosine similarity alignment between intermediate 3D decoder features and pre-trained V-JEPA2-L features, $\lambda=0.7$
- $\mathcal{L}_{\text{entropy}}$: LFQ entropy loss promoting codebook utilization, $\gamma=0.02$

**Advanced Design**: The final tokenizer training (Stage 4) additionally employs VideoMAE-B as a semantic discriminator, feeding its multi-layer features into a trainable 1D CNN head for real/fake discrimination, significantly improving reconstruction and downstream generation quality.

## Key Experimental Results

### Main Results

| Method | Params (Tok+Gen) | Recon. rFVD↓ | Gen. gFVD↓ | Recon. Token Count | Gen. Token Count |
|--------|-----------------|-------------|-----------|-------------------|-----------------|
| LARP-L-Long | 173M+632M | 20 | 57 | 1024 | 1024 |
| AdapTok | 195M+633M | 36 | 67 | 1024 | 1024 |
| OmniTokenizer | 82M+650M | 42 | 191 | 1280 | 1280 |
| **EVATok** | **145M+633M** | **9.7** | **48** | **774 (−24.4%)** | **756 (−26.2%)** |

### Ablation Study

| Configuration | rFVD↓ | Token Count | Notes |
|---------------|-------|------------|-------|
| Uniform allocation (Proxy Tok.) | 73 | 1024 | Fixed allocation baseline |
| Uniform allocation (Final Tok.) | 63 | 1024 | Final tokenizer outperforms |
| Router (Proxy Tok.) | 50 | 721 (−29.6%) | Router allocation yields significant gain |
| Router (Final Tok.) | 33 | 721 (−29.6%) | Both improvements combined are best |
| +VideoMAE discriminator | 9.2 | 721 (−29.6%) | Semantic discriminator brings large gain |

### Key Findings

- Adaptive allocation consistently dominates fixed allocation on the quality-cost curve at the same average token count: 56% token savings on WebVid and 42% on UCF at equivalent rFVD.
- The final tokenizer significantly outperforms the proxy tokenizer (under equivalent training), demonstrating the importance of eliminating the training-inference gap in variable-length tokenizers.
- The router generalizes to the UCF dataset unseen during training, approaching the optimal Pareto frontier of brute-force search.
- Introducing the VideoMAE semantic discriminator reduces rFVD from 33 to 9.2, making it the single largest quality improvement factor.

## Highlights & Insights

- The paradigm of "define optimality → brute-force label → train a classifier to imitate" is elegant: it converts a seemingly continuous optimization problem into a discrete classification task, offering both theoretical optimality guarantees and practical efficiency. This design pattern—using a small model to predict the optimal configuration of a large model—has strong reuse potential in other settings.
- The design choice to avoid tail-token-dropping is insightful: fixing the length at query initialization eliminates both the wasted computation of encoding tokens that will be discarded and the role ambiguity of tail queries during encoding.

## Limitations & Future Work

- The candidate allocation space grows exponentially as $m^T$ (625 in this work); for longer videos or finer granularity, the search space explodes and more efficient allocation space design is needed.
- The router uses a global ViT-S and produces a single prediction per video, which may lack flexibility for long videos with abrupt local complexity changes.
- The interaction between the fixed codebook size (8192/16384) and adaptive token length has not been explored for optimality.

## Related Work & Insights

- **vs. LARP/AdapTok**: Both methods also perform adaptive video tokenization but rely on heuristic allocation strategies (threshold search / mini-batch ILP). EVATok provides an explicit definition of "optimal allocation" via proxy reward and achieves better router generalization.
- **vs. ElasticTok**: ElasticTok achieves variable length through tail-token-dropping. EVATok demonstrates the efficiency and performance drawbacks of this approach and instead directly determines the token count at the query initialization stage.

## Rating

- Novelty: ⭐⭐⭐⭐ — Complete four-stage framework, elegant proxy reward definition, and the idea of replacing search with a router are all well-conceived.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Quality-cost curve analysis, ablation studies, and system-level comparisons are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — The four-stage narrative is logically clear and the problem formulation is rigorous.
- Value: ⭐⭐⭐⭐ — Token savings of 24.4%–29.6% have direct deployment value for video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Depth Adaptive Efficient Visual Autoregressive Modeling](depthvar_depth_adaptive_var.md)
- [\[ICCV 2025\] Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization](../../ICCV2025/image_generation/efficient_autoregressive_shape_generation_via_octree-based_adaptive_tokenization.md)
- [\[CVPR 2026\] ViHOI: Human-Object Interaction Synthesis with Visual Priors](vihoi_human-object_interaction_synthesis_with_visual_priors.md)
- [\[CVPR 2026\] Causal Motion Diffusion Models for Autoregressive Motion Generation](causal_motion_diffusion_models_for_autoregressive_motion_generation.md)
- [\[CVPR 2026\] Image Generation as a Visual Planner for Robotic Manipulation](image_generation_as_a_visual_planner_for_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
