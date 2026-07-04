---
title: >-
  [Paper Note] Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation
description: >-
  [CVPR 2025][Segmentation][Open-vocabulary semantic segmentation] SemLA proposes a training-free test-time domain adaptation framework. By building a LoRA adapter library indexed by CLIP, it dynamically retrieves and fuses the most relevant adapters at inference based on the embedding distance between the input image and domain centroids. This achieves on-the-fly and highly efficient domain adaptation for open-vocabulary semantic segmentation models.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Open-vocabulary semantic segmentation"
  - "LoRA adapters"
  - "domain adaptation"
  - "test-time adaptation"
  - "CLIP"
date: 2026-05-08
content_hash: 4a380e2c08e8c3a1
---

# Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2503.21780](https://arxiv.org/abs/2503.21780)  
**Code**: [https://thegoodailab.org/semla](https://thegoodailab.org/semla)  
**Area**: Segmentation / Domain Adaptation  
**Keywords**: Open-vocabulary semantic segmentation, LoRA adapters, domain adaptation, test-time adaptation, CLIP

## TL;DR

SemLA proposes a training-free test-time domain adaptation framework. By building a LoRA adapter library indexed by CLIP, it dynamically retrieves and fuses the most relevant adapters at inference based on the embedding distance between the input image and domain centroids. This achieves on-the-fly and highly efficient domain adaptation for open-vocabulary semantic segmentation models.

## Background & Motivation

**Background**: Open-vocabulary (OV) semantic segmentation maps visual layouts to pixel-level classifications using vision-language alignment, defining classes via arbitrary textual queries. Models like CAT-Seg show impressive flexibility in zero-shot scenarios. However, distribution shifts between the training domain and the test domain (domain drift) significantly degrade performance—including shifts in visual appearance and lexical misalignment.

**Limitations of Prior Work**: (1) Traditional unsupervised domain adaptation (UDA) methods typically target a static single target domain and require access to source domain data, leading to a slow process that can degrade source domain performance. (2) Test-time adaptation (TTA) methods generally incur high computational overheads, making them unsuitable for real-time applications. (3) Data is abundant but highly heterogeneous—with varying label spaces, different annotation styles, and potential privacy restrictions—leading to a paradox where "data is rich but models are fragile."

**Key Challenge**: There is a fundamental tension between the flexibility of open-vocabulary segmentation and the robustness of domain adaptation. Models are expected to respond to arbitrary textual queries while adapting to diverse, varying visual domains. Traditional adaptation methods fail to adapt to new domains efficiently while preserving open-vocabulary capabilities.

**Goal**: Achieve training-free test-time domain adaptation for OV semantic segmentation—enabling on-the-fly adaptation to the domain of any arbitrary input image at inference without further training.

**Key Insight**: Inspired by the combination of LoRA adapters and retrieval in the LLM community (e.g., Hugging Face's adapter ecosystem), the authors observe that lightweight LoRA adapters can be trained for different domains and indexed using CLIP embeddings. At test time, the most relevant adapters can be retrieved and fused based on the CLIP embedding of the input image. This mimics a library—even if the exact book wanted is unavailable, one can still acquire the necessary knowledge by synthesizing several related books.

**Core Idea**: Build a library of LoRA adapters indexed by CLIP centroids. At test time, compute the CLIP embedding of the input image, determine its distance to each domain's centroid, select the Top-K nearest adapters, and merge them with distance-weighted concat-based fusion to generate a customized model for each input image.

## Method

### Overall Architecture

SemLA consists of two phases:
1. **Offline: Build the LoRA Adapter Library**—Train a LoRA adapter for each training domain and compute the centroid of CLIP embeddings for images in that domain as the index, yielding the library $\mathcal{L} = \{(\mathbf{c}_i, \Delta\mathcal{W}_i)\}$.
2. **Online: Dynamic Test-Time Adaptation**—For each testing image, compute its CLIP embedding $\rightarrow$ retrieve the Top-K nearest adapters $\rightarrow$ compute distance-based weights $\rightarrow$ fuse the LoRA weights via concatenation $\rightarrow$ make predictions.

### Key Designs

1. **CLIP Embedding Centroid Index (Domain Embeddings from CLIP)**:

    - **Function**: Creates a compact semantic index for each domain's LoRA adapter.
    - **Mechanism**: For each image in training domain $\mathcal{D}_i$, compute its CLIP image embedding $\mathbf{e}_j = \text{CLIP}_\text{image}(\mathbf{x}_j)$. The centroid is the average embedding $\mathbf{c}_i = \frac{1}{N_i}\sum_{j=1}^{N_i} \mathbf{e}_j$. At test time, the CLIP embedding $\mathbf{e}_t$ of the input image is computed, and similarity is measured via Euclidean distance $d_i = \|\mathbf{e}_t - \mathbf{c}_i\|_2$.
    - **Design Motivation**: The embedding space of CLIP naturally captures the semantic characteristics of domains. Computing centroids is simple and efficient, requiring no training or fine-tuning of CLIP.

2. **Distance-Based Adapter Weighting and Concat Merging**:

    - **Function**: Merges knowledge from multiple domains based on relevance to construct a customized model.
    - **Mechanism**: Select the Top-K nearest adapters (index set $\mathcal{K}$), and compute weights using softmax with a temperature parameter $\tau$: $w_i = \frac{\exp(1/(d_i \cdot \tau))}{\sum_{k \in \mathcal{K}} \exp(1/(d_k \cdot \tau))}$. The fusion adopts a concatenation mechanism: the $\mathbf{A}$ matrices of the selected adapters are scaled by their weights and vertically concatenated, while the $\mathbf{B}$ matrices are horizontally concatenated, obtaining $\Delta\mathbf{W}_\text{fused} = \mathbf{B}_\text{fused} \mathbf{A}_\text{fused}$.
    - **Design Motivation**: Compared to uniform merging, distance-based weighting allows more relevant domains to contribute higher weights. Concat-based merging preserves the low-rank structures of individual adapters, retaining unique domain knowledge better than direct weighted averaging of $\Delta\mathbf{W}$.

3. **Library Scalability Design**:

    - **Function**: Adds new domain adapters dynamically at any time without retraining or affecting existing adapters.
    - **Mechanism**: For a new domain, compute its centroid $\mathbf{c}_*$, train the LoRA adapter $\Delta\mathcal{W}_*$, and append them to the library: $\mathcal{L} = \mathcal{L} + (\mathbf{c}_*, \Delta\mathcal{W}_*)$.
    - **Design Motivation**: Real-world domains emerge continuously. This incremental expansion of the library avoids global retraining or re-optimization.

### Loss & Training

Each domain's LoRA adapter is trained independently on its respective domain data using standard semantic segmentation loss (cross-entropy). During training, the original weights of CAT-Seg are frozen, and only the LoRA parameters are optimized. All adapters utilize the same rank $r$ to ensure dimensional alignment during fusion.

Evaluation adopts a leave-one-out strategy: when evaluating on a target domain, its corresponding adapter is removed from the library, ensuring the model has never direct access to target-specific knowledge.

## Key Experimental Results

### Main Results

**20-Domain Benchmark (CAT-Seg backbone, leave-one-out, mIoU):**

| Method | ACDC rain | ACDC fog | ACDC night | CS | BDD | ADE150 | IDD | h-mean |
|------|-----------|----------|------------|------|------|--------|------|--------|
| Zero-shot | 46.5 | 47.1 | 37.9 | 47.1 | 47.9 | 37.8 | 35.4 | 39.4 |
| Uniform merging | 67.4 | 69.7 | 50.0 | 62.2 | 58.2 | 37.3 | 38.8 | 51.9 |
| **SemLA** | **67.7** | **71.9** | **51.7** | **63.9** | **57.3** | **38.2** | **40.2** | **54.2** |
| Oracle (Upper Bound) | 70.9 | 70.0 | 51.6 | 67.5 | 60.1 | 54.0 | 64.3 | 61.1 |

### Ablation Study

| Fusion Strategy | h-mean mIoU |
|---------|-------------|
| Zero-shot (No adaptation) | 39.4 |
| Uniform merging | 51.9 |
| SemLA Late Fusion (Output-level fusion) | 52.1 |
| Uniform Late Fusion | 49.3 |
| **SemLA (Weight-level fusion)** | **54.2** |

### Key Findings

- **SemLA improves h-mean by 2.27 points over Uniform merging** and yields improvements across most domains, proving that selective fusion outperforms global uniform merging.
- **Improvements are particularly pronounced under adverse weather domains**: ACDC fog (+2.21), ACDC night (+1.75), and typically +2~7 points across the MUSES suite. This indicates that as domain specificity increases, the selective advantage of SemLA becomes more pronounced.
- **Occasionally outperforms the Oracle**: On ACDC fog and ACDC night, SemLA surpasses the single adapter trained directly on target domain data, suggesting that fusing multi-domain knowledge can provide complementary gains.
- **Weight-level fusion outperforms output-level fusion**: SemLA (54.2) > SemLA Late Fusion (52.1), indicating that fusing in the parameter space is more effective than fusing in the prediction space.
- **CLIP embeddings serve as effective domain navigators**: Without training an auxiliary retriever, CLIP's zero-shot embeddings can accurately identify which domains input images belong to.

## Highlights & Insights

- The **library analogy** is elegant and intuitive: adapters are "books," the CLIP centroid is the "index system," and fusion is "synthesizing knowledge from many related books." This framework transforms domain adaptation from a "training problem" to a "retrieval and fusion problem."
- **Interpretability is a natural byproduct**: By observing which adapters are chosen and what their weights are, one can understand why the model made a specific prediction. This is highly valuable in sensitive domains such as healthcare.
- **Data privacy-friendly**: There is no need to access any training data at test time; data of each domain remains local, with only LoRA parameters and CLIP centroids shared. This facilitates federated-style collaboration.
- **Plug-and-play**: SemLA does not depend on a specific backbone and can be applied to any CLIP-based OV segmentation model.

## Limitations & Future Work

- **Bounded domain coverage of the adapter library**: If the domain of an input image is far from all domains in the library, fusion performance may degrade. A larger and more diverse library is needed.
- **Upper bound on the discriminative capability of CLIP embeddings**: Some domains with significant visual differences but similar semantics (e.g., the same scene under day vs. night) might end up close in CLIP space, leading to suboptimal selection.
- **Evaluation is limited to CAT-Seg**: Although claimed to be backbone-agnostic, the main experiments only validate performance on CAT-Seg.
- **Centroid representation is somewhat coarse**: A single average vector struggles to capture intra-domain diversity (e.g., when a domain contains multiple sub-scenes).
- Future directions include: exploring fine-grained domain representations (such as GMM), utilizing active learning to select the most valuable new domains for training, and incorporating text embeddings for multi-modal retrieval.

## Related Work & Insights

- **vs. Uniform LoRA Merging (Model Soups series)**: Model Soups / AdapterSoup uniformly merge all adapters, effectively giving all "books" equal weights. SemLA selectively fuses them based on relevance, achieving better performance.
- **vs. LoraRetriever**: LoraRetriever trains a retriever via instruction fine-tuning, whereas SemLA directly utilizes the zero-shot capabilities of CLIP, offering a simpler and more elegant approach.
- **vs. Test-Time Adaptation (TTA)**: TTA methods (such as entropy minimization) require gradient updates at test time, while SemLA is entirely training-free, involving only a single CLIP forward pass and matrix operations.
- **vs. UDA**: UDA requires both source and target domain data to coexist during optimization and is limited to adapting to a static target domain. SemLA requires no source data and adapts to arbitrary domains concurrently.

## Rating

- Novelty: ⭐⭐⭐⭐ Fusing and retrieving LoRAs for domain adaptation in OV segmentation presents a novel combination, though the individual technical components (LoRA, CLIP retrieval) are existing concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The 20-domain benchmark provides extensive coverage, the leave-one-out evaluation is rigorous, and the comparisons with Oracle and ablations are comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ The library analogy is consistent throughout, the framework diagram is clear, and the method description is mathematically rigorous.
- Value: ⭐⭐⭐⭐ The training-free and privacy-preserving characteristics give the method strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Effective SAM Combination for Open-Vocabulary Semantic Segmentation](effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Exploring Simple Open-Vocabulary Semantic Segmentation](exploring_simple_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation](dpseg_dual-prompt_cost_volume_learning_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Universal Domain Adaptation for Semantic Segmentation](universal_domain_adaptation_for_semantic_segmentation.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](../../ICCV2025/segmentation/training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
