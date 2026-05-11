---
title: >-
  [Paper Note] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning
description: >-
  [ICCV 2025][Multimodal VLM][Test-Time Adaptation] This paper proposes Latte, a framework that enables collaborative test-time adaptation of vision-language models (e.g.…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Test-Time Adaptation"
  - "Federated Learning"
  - "Vision-Language Models"
  - "Memory Cache"
  - "CLIP"
date: 2026-05-08
content_hash: 6a283be124e940b4
---

# LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning

**Conference**: ICCV 2025
**arXiv**: [2507.21494](https://arxiv.org/abs/2507.21494)
**Code**: [GitHub](https://github.com/baowenxuan/Latte)
**Area**: Multimodal VLM
**Keywords**: Test-Time Adaptation, Federated Learning, Vision-Language Models, Memory Cache, CLIP

## TL;DR

This paper proposes Latte, a framework that enables collaborative test-time adaptation of vision-language models (e.g., CLIP) in decentralized federated learning settings. Through a dual-memory mechanism combining local and external memory, Latte achieves cross-client knowledge sharing while preserving client-level personalization.

## Background & Motivation

Pre-trained vision-language models (VLMs) such as CLIP demonstrate strong zero-shot image classification performance, yet suffer from **domain shift** when deployed to specific downstream domains—the alignment between visual and textual embeddings may no longer hold. Test-time adaptation (TTA) is an effective remedy, and **memory-based methods** are particularly attractive due to their training-free, backpropagation-free nature.

However, existing memory-based TTA methods rely on a critical assumption: **a single domain with sufficient data**. In federated learning (FL) settings, multiple clients perform the same task but with heterogeneous data distributions and limited local data, giving rise to two conflicting strategies:

**Independent adaptation**: Each client runs TTA independently, leading to poor memory quality due to data scarcity and degraded performance.

**Global sharing**: All clients share a single global memory, which fails to accommodate the unique distribution of each client.

The core problem addressed in this paper is: **How can memory information be securely and efficiently shared across decentralized heterogeneous clients—leveraging the data advantage of in-distribution clients while remaining robust to out-of-distribution ones?** Latte addresses this via a dual-memory design combining local and external memory.

## Method

### Overall Architecture

The Latte pipeline consists of four steps:
1. Encode the input image to obtain embedding $\boldsymbol{f}$ and an initial prediction.
2. Update the local memory $\boldsymbol{L}^i$ using $\boldsymbol{f}$.
3. Obtain an adapted prediction using local memory $\boldsymbol{L}^i$ and external memory $\boldsymbol{E}^i$.
4. (Optionally) communicate with the server to update external memory $\boldsymbol{E}^i$.

### Key Designs

1. **Local Memory Construction (Priority Queue)**: Each client maintains a class-partitioned memory $\boldsymbol{L}^i \in \mathbb{R}^{c \times k_l \times d}$, where each class corresponds to an entropy-sorted priority queue. When a new test sample arrives, it is inserted directly if the queue is not full; if the queue is full and the new sample has lower entropy (higher confidence), it replaces the entry with the highest entropy. This ensures that only the most reliable sample embeddings are retained in memory.

2. **Global Memory and External Memory (Server-Coordinated Selective Sharing)**: The server maintains a global memory $\boldsymbol{G} \in \mathbb{R}^{c \times n \times d}$. Each client uploads **entropy-weighted prototypes** (entropy-weighted averages followed by normalization) from its local memory. Crucially, rather than receiving the entire global memory, each client uses its own prototype as a query vector to retrieve the top-$k_e$ most similar prototypes from other clients as its external memory. This achieves **coarse-grained filtering**, reducing the transmission of irrelevant prototypes.

3. **Adaptive Prediction with Fused Memory**: After merging local and external memory, aggregation weights are computed by jointly considering **embedding similarity** and **uncertainty (entropy)**:

$$w_{y,\kappa}^i = \exp(\beta \cdot \boldsymbol{f}^\top \boldsymbol{m}_{y,\kappa}^i) \cdot \exp(-\gamma \cdot H(\boldsymbol{m}_{y,\kappa}^i))$$

This design assigns higher weights to samples with high similarity and low uncertainty, thereby conferring robustness to OOD prototypes and misclassified samples. The final prediction is a weighted sum of the original CLIP logits and the memory-based logits.

4. **Decoupling Communication from Inference**: The communication process depends only on local memory and not on current test samples, allowing clients to perform offline inference between communication rounds and substantially reducing the number of communication rounds required.

### Loss & Training

Latte is a **training-free** framework—no backpropagation or gradient computation is required. Adaptation is achieved entirely through memory construction, sharing, and weighted retrieval. Hyperparameters include local memory size $k_l$, external memory size $k_e$, similarity sharpness $\beta$, uncertainty sharpness $\gamma$, and fusion coefficient $\alpha$.

## Key Experimental Results

### Main Results

Evaluation is conducted on domain adaptation benchmarks (VLCS, TerraIncognita) and corruption benchmarks (CIFAR-10-C, CIFAR-100-C).

| Method | VLCS (ViT-B/16) | TerraIncognita (ViT-B/16) | CIFAR-10-C (ViT-B/16) |
|------|:---:|:---:|:---:|
| CLIP | 80.83 | 31.84 | 65.58 |
| VTE | 81.75 | 38.56 | 67.64 |
| TDA (local) | 81.44 | 34.24 | 66.58 |
| TDA (global) | 80.29 | 36.19 | 65.58 |
| DMN-ZS (local) | 81.12 | 33.65 | 67.42 |
| DMN-ZS (global) | 80.55 | 37.64 | 63.90 |
| **Latte** | **82.57 (+1.74)** | **40.95 (+9.11)** | **68.27 (+2.69)** |

Latte achieves the best performance on all benchmarks. Notably, on TerraIncognita, Latte outperforms CLIP zero-shot by 9.11%, far exceeding all other methods. Global sharing strategies (TDA global, DMN-ZS global) even exhibit negative transfer in certain settings.

### Ablation Study

| Ablation | VLCS Accuracy | Note |
|--------|:---:|------|
| Local memory only | ~81.5 | Lacks cross-client information |
| External memory only | ~81.0 | Lacks local personalization |
| Latte (full) | **82.57** | Complementary combination |
| w/o similarity weight | ~81.0 | Entropy weighting alone is insufficient |
| w/o uncertainty weight | ~81.5 | Similarity weighting alone is insufficient |
| Latte (full) | **82.57** | Both components are necessary |

### Key Findings

- As the degree of data decentralization increases (number of clients per domain from 1 to 50), DMN-ZS and TDA performance degrades significantly, whereas Latte remains stable.
- Computational overhead is minimal: Latte adds only 871K MACs compared to 17.6G MACs for CLIP inference.
- Communication is efficient: each round transmits less than 0.4% of the CLIP visual encoder size; accuracy remains nearly unchanged for communication intervals $T \leq 50$.
- In-distribution clients predominantly retrieve prototypes from clients with similar distributions; merging memory significantly reduces entropy and yields tighter intra-class clustering.

## Highlights & Insights

- **Elegant design**: The dual-memory and prototype retrieval mechanism effectively balances sharing and personalization while remaining conceptually simple.
- **Theoretical guarantees**: It is proven that Latte's error monotonically decreases as the number of in-distribution clients grows, and the error bound with respect to OOD clients is unaffected.
- **Practical applicability**: Decoupling communication from inference makes Latte genuinely suitable for real-world FL systems rather than idealized settings.
- The training-free nature makes it well-suited for resource-constrained edge devices.

## Limitations & Future Work

- Validation is limited to image classification; extension to detection, segmentation, and other vision tasks remains unexplored.
- Extreme heterogeneity scenarios (e.g., completely non-overlapping label spaces across clients) are not thoroughly discussed.
- The top-$k$ prototype retrieval strategy is relatively simple; finer-grained image-level memory sharing could be explored.
- Theoretical analysis relies on a simplified binary classification assumption, which may not fully reflect practical multi-class settings.

## Related Work & Insights

- Building upon memory-based TTA methods such as TDA and DMN-ZS, the core innovation lies in introducing a collaborative mechanism for the federated setting.
- Compared to federated TTA approaches (e.g., FedTHE, ATP), Latte is distinguished by its focus on pre-trained VLMs without any fine-tuning.
- The top-$k$ retrieval strategy for external memory shares conceptual parallels with retrieval-augmented generation (RAG).
- Future work could explore extending this paradigm to distributed inference scenarios involving multimodal large language models.

## Rating

- **Novelty**: 7/10 — The dual-memory and collaborative retrieval combination is novel in the VLM+FL setting.
- **Technical Quality**: 8/10 — The method is complete, with solid theoretical analysis and comprehensive experiments.
- **Practicality**: 8/10 — Training-free, low communication overhead, and decoupled communication and inference.
- **Writing Quality**: 8/10 — Well-structured with thorough notation definitions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models](multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)
- [\[ICCV 2025\] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[NeurIPS 2025\] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](../../NeurIPS2025/multimodal_vlm/mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)

</div>

<!-- RELATED:END -->
