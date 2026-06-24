---
title: >-
  [Paper Note] GOAL: Global-Local Object Alignment Learning
description: >-
  [CVPR 2025][Information Retrieval & RAG][Global-Local Alignment] Proposes the GOAL method, which enhances CLIP's understanding of long text descriptions through two modules: Local Image-Sentence Matching (LISM) and Token Similarity-based Learning (TSL). By introducing local semantic alignment on top of global alignment, it significantly improves image-text retrieval performance.
tags:
  - "CVPR 2025"
  - "Information Retrieval & RAG"
  - "Global-Local Alignment"
  - "CLIP Fine-tuning"
  - "Long-Text Image-Text Retrieval"
  - "SAM Segmentation"
  - "Token Similarity Learning"
date: 2026-05-08
content_hash: 49345e0cca120605
---

# GOAL: Global-Local Object Alignment Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.17782](https://arxiv.org/abs/2503.17782)  
**Code**: [GitHub](https://perceptualai-lab.github.io/GOAL)  
**Area**: Information Retrieval  
**Keywords**: Global-Local Alignment, CLIP Fine-tuning, Long-Text Image-Text Retrieval, SAM Segmentation, Token Similarity Learning

## TL;DR

Proposes the GOAL method, which enhances CLIP's understanding of long text descriptions through two modules: Local Image-Sentence Matching (LISM) and Token Similarity-based Learning (TSL). By introducing local semantic alignment on top of global alignment, it significantly improves image-text retrieval performance.

## Background & Motivation

**Background**:
CLIP has demonstrated powerful zero-shot transfer capabilities through contrastive learning on large-scale image-text pairs. However, CLIP's training data mainly consists of short image descriptions (up to 77 tokens), focusing on high-level concepts of images.

**Limitations of Prior Work**:
1. CLIP performs poorly when processing longer, detailed text descriptions because its unified embedding space is optimized for concise descriptions.
2. Existing models only perform global image-text matching, aligning the entire image and the full caption as a single unit, which lacks fine-grained local correspondence.
3. Although Long-CLIP extends the text length, it requires multimodal LLMs to generate synthetic long descriptions, resulting in high data preparation costs.

**Key Challenge**:
CLIP's global contrastive learning paradigm cannot capture the fine-grained correspondence between local image regions and specific description sentences in the text, leading to a substantial loss of detailed information when processing long texts.

**Goal**:
How to efficiently fine-tune CLIP to enable it to understand local semantic details in long texts while maintaining global alignment capabilities.

**Key Insight**:
"Global" is defined as the entire image or full text, while "local" is defined as image fragments or individual sentences in the text. Global representations are enhanced by constructing local pseudo-pairs and propagating local attention.

**Core Idea**:
Segment images using SAM and split sentences to construct local image-sentence pseudo-pairs, then propagate local information to global representations through Token-level Similarity-based Learning.

## Method

### Overall Architecture

GOAL consists of two core components: (1) LISM (Local Image-Sentence Matching) for extracting local pseudo-pairs from global image-long text pairs; (2) TSL (Token Similarity-based Learning) for efficiently propagating local attention to global Token representations through matched local pairs. The overall objective function combines global contrastive, local contrastive, and TSL losses.

### Key Designs

1. **LISM (Local Image-Sentence Matching Pipeline)**:
    - **Function**: Automatically extract the best local image-sentence matching pairs from a pair of image-text data.
    - **Mechanism**: Segment the image into semantic regions using SAM (filtering small areas <1%), and split the caption by sentences; extract CLS embeddings of each region and sentence using the CLIP encoder, calculate cosine similarity to perform maximum similarity matching, and select the local pair $(I_l, T_l)$ with the highest similarity.
    - **Design Motivation**: Leverage SAM's strong segmentation capabilities and CLIP's own matching ability to obtain high-quality local correspondences without additional annotations.

2. **TSL (Token Similarity-based Learning)**:
    - **Function**: Propagate local semantic information to global Token representations.
    - **Mechanism**: For the text side, locate the tokens in the global text corresponding to the local sentence, map them through a projection layer after average pooling, and maximize the similarity with the local CLS embedding; for the visual side, locate the patch tokens corresponding to the region in the global image based on the bounding box of the local image, map them through a projection layer after average pooling, and maximize the similarity with the local CLS embedding.
    - **Design Motivation**: By making a subset of global tokens approximate the CLS representation of the corresponding local part, the encoder is encouraged to focus on key local elements in the image/text.

3. **Positional Encoding Interpolation**:
    - **Function**: Support processing of long texts exceeding 77 tokens.
    - **Mechanism**: Adopt the positional encoding interpolation technique from Long-CLIP to extend the maximum sequence length of the text encoder.
    - **Design Motivation**: The 77-token limit of the original CLIP cannot handle multi-sentence long descriptions.

### Loss & Training

- Total loss: $\mathcal{L}_{total} = \lambda_{global}\mathcal{L}_{global} + \lambda_{local}\mathcal{L}_{local} + \lambda_{TSL}\mathcal{L}_{TSL}$
- $\lambda_{global}=1$, $\lambda_{local}=0.5$, $\lambda_{TSL}=1$
- $\mathcal{L}_{global}$ and $\mathcal{L}_{local}$ are standard CLIP contrastive losses
- $\mathcal{L}_{TSL}$ uses MSE loss to maximize the similarity between the projected Token and the corresponding local CLS embedding
- Trained for 10 epochs with a batch size of 16, taking about 1 hour on a single RTX 4090.

## Key Experimental Results

### Main Results

**DOCCI Dataset (Original Test Set, ViT-L/14)**

| Method | T2I R@1 | T2I R@5 | I2T R@1 | I2T R@5 |
|------|---------|---------|---------|---------|
| Global fine-tuning | 74.00 | 93.84 | 73.55 | 93.94 |
| Local fine-tuning | 67.39 | 90.67 | 66.33 | 90.41 |
| w/o TSL | 74.75 | 94.31 | 74.55 | 94.37 |
| **GOAL** | **84.37** | **97.55** | **82.57** | **97.37** |

**DCI Dataset (Original Test Set, ViT-L/14)**

| Method | T2I R@1 | T2I R@5 | I2T R@1 | I2T R@5 |
|------|---------|---------|---------|---------|
| Global fine-tuning | 65.73 | 84.24 | 65.73 | 86.04 |
| GOAL | **76.89** | **91.05** | **76.59** | **91.20** |

### Ablation Study

**DOCCI Dataset (ViT-B/16)**

| Setting | Global | Local | TSL | T2I R@1 | I2T R@1 |
|------|--------|-------|-----|---------|---------|
| Global Only | ✓ | | | 72.41 | 72.04 |
| Local Only | | ✓ | | 65.82 | 65.73 |
| Global + Local | ✓ | ✓ | | 72.08 | 71.80 |
| GOAL (Full) | ✓ | ✓ | ✓ | **79.47** | **79.43** |

### Key Findings

1. **TSL is the core contribution**: Simply adding local contrastive loss (w/o TSL) yields almost no improvement or even a slight decline, but adding TSL leads to a significant performance leap (ViT-L/14 DOCCI: +12.87% R@1).
2. **Local alignment cannot be used in isolation**: Using only local contrastive loss results in performance significantly lower than the global baseline due to the loss of global context.
3. **Equally effective on joint Global-Local test sets**: GOAL also significantly outperforms baselines on test sets containing both global and local queries (mAP@10 increases by 3-6 percentage points).
4. **Resource-friendly**: Training ViT-B/16 can be completed on a single RTX 4090 GPU in approximately 1 hour.

## Highlights & Insights

1. **Simple yet effective**: The method design is remarkably simple, substantially boosting CLIP's ability to handle long texts using only two modules: the LISM pipeline and the TSL loss.
2. **No extra annotations**: Automatically constructs local pseudo-pairs utilizing SAM and CLIP's inherent capabilities without requiring manual annotations.
3. **Key insight of TSL**: Instead of merely adding contrastive learning between global and local views, it forces a subset of global Tokens to "approximate" the corresponding local representation. This Token-level alignment strategy is significantly more effective than CLS-level local contrastive learning.
4. **New Benchmarks**: Proposes three new evaluation benchmarks targeting long-text image-text retrieval (DOCCI, DCI, Urban1k).

## Limitations & Future Work

1. The LISM pipeline only selects a single best local matched pair, potentially losing other valuable local correspondences.
2. It relies heavily on SAM for image segmentation, making the quality of local pairs directly dependent on the segmentation quality.
3. Validation is limited to only three datasets, with relatively small data scales (e.g., DOCCI has only about 10k samples).
4. Lacks a systematic comparison with large-scale pre-training methods (e.g., CLOC).
5. Positional encoding interpolation may have limited efficacy on extremely long texts.

## Related Work & Insights

- **CLIP/Long-CLIP**: The base models and direct competitors of this work.
- **CLOC**: A large-scale pre-training method that establishes local correspondences via OWL-v2 detectors but requires 2 billion image-text pairs.
- **SAM**: Provides strong zero-shot image segmentation capabilities, serving as a critical dependency for the LISM pipeline.
- **Insight**: The key to local semantic alignment lies not in learning local features independently, but in how to effectively propagate local information into global representations.

## Rating

- Novelty: ⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation](preserving_clusters_in_prompt_learning_for_unsupervised_domain_adaptation.md)
- [\[ICML 2026\] Vector Linking based on Cross-Model Local Isometric Consistency](../../ICML2026/information_retrieval/vector_linking_via_cross-model_local_isometric_consistency.md)
- [\[ACL 2025\] Divide-Then-Align: Honest Alignment based on the Knowledge Boundary of RAG](../../ACL2025/information_retrieval/divide_then_align_rag_knowledge_boundary.md)
- [\[ICLR 2026\] ELViS: Efficient Visual Similarity from Local Descriptors that Generalizes Across Domains](../../ICLR2026/information_retrieval/elvis_efficient_visual_similarity_from_local_descriptors_that_generalizes_across.md)
- [\[ICLR 2026\] Deep Global-sense Hard-negative Discriminative Generation Hashing for Cross-modal Retrieval](../../ICLR2026/information_retrieval/deep_global-sense_hard-negative_discriminative_generation_hashing_for_cross-moda.md)

</div>

<!-- RELATED:END -->
