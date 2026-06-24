---
title: >-
  [Paper Note] Untitled
description: >-
  [ACL 2025][Multimodal VLM][K-Means] Academic paper note for Untitled.
tags:
  - ACL 2025
  - Multimodal VLM
  - K-Means
  - RQ-VAE
date: 2026-05-08
content_hash: 3da5480c3c04234b
---
# CART: A Generative Cross-Modal Retrieval Framework with Coarse-To-Fine Semantic Modeling

## Basic Information

- **Conference**: ACL 2025
- **arXiv**: [2406.17507](https://arxiv.org/abs/2406.17507)
- **Code**: None
- **Area**: Information Retrieval
- **Keywords**: Generative Retrieval, Cross-Modal Retrieval, Semantic Identifier, K-Means, RQ-VAE, Coarse-To-Fine
- **TL;DR**: Proposes CART, the first generative cross-modal retrieval framework supporting text-to-image/audio/video, which constructs coarse-to-fine semantic identifiers via K-Means + RQ-VAE and combines feature fusion strategies to achieve an outstanding balance between retrieval performance and efficiency.

## Background & Motivation

Cross-modal retrieval aims to search for semantically related instances through the interaction of multi-modal data. Existing methods primarily fall into two categories:

**Single-tower models** (e.g., BLIP-2, InternVL-G): Perform fine-grained interaction (e.g., cross-attention) between the query and candidates, offering high retrieval accuracy but massive latency, making them unsuitable for large-scale retrieval.

**Dual-tower models** (e.g., CLIP, CLAP): Map different modalities into a joint embedding space to compute similarity, which is more efficient but suffers from accuracy limitations due to the modality gap.

**Generative retrieval** is an emerging paradigm that assigns identifiers to each candidate, transforming the retrieval problem into a sequence-to-sequence generation problem. Its advantages include:
- Retrieval speed is independent of the dataset size
- No need to maintain a unified embedding space
- Leverages the powerful capabilities of generative models to improve performance

**Key Challenge**: Extending generative retrieval from document retrieval to cross-modal retrieval faces three issues:
1. Multi-modal data lacks text that can be directly used as identifiers (unlike documents which have titles/keywords).
2. Identifiers constructed from low-level visual/auditory information have a semantic gap with natural language queries.
3. Generative retrieval lacks an explicit interaction process between the query and candidates.

## Method

### Overall Architecture

CART (Cross-modal Autoregressive Retrieval Transformer) contains three modules:
1. **Semantic Identifier Generation**: Constructs hierarchical semantic identifiers for each candidate.
2. **Caption Enhancement**: Generates text descriptions for multi-modal data to act as queries.
3. **Feature Fusion**: Fuses multi-layer features within an encoder-decoder architecture.

### Key Designs

#### Key Design 1: Coarse-to-Fine Semantic Identifier Generation

The identifier consists of three parts: **coarse-grained token + fine-grained token + unique token**

**Coarse-grained Token (Coarse Token)**:
- Uses ImageBind to encode the embeddings of all candidates.
- Performs K-Means clustering on the embeddings.
- The cluster ID serves as the **first token** of the identifier.
- Intuition: The first token is crucial; if predicted incorrectly, the subsequent generation is meaningless. K-Means captures global semantic classification.

**Fine-grained Token (Fine Token)**:
- Computes the residual between the original embedding and the K-Means cluster center (highlighting subtle differences).
- Uses RQ-VAE (Residual Quantized Variational Autoencoder) to perform multi-layer quantization on the residual.
- RQ-VAE contains $M$ independent codebooks, recursively quantizing the residual: $v_m = \arg\min_k \|r_{m-1} - e_m^k\|$
- Each layer of quantization captures feature differences at different levels of granularity.
- The training loss consists of reconstruction loss and commitment loss.

**Unique Token**:
- Maintains a prefix database to detect identifier conflicts.
- Appends a counter value to conflicting identifiers, ensuring each candidate has a unique identifier.

Final identifier format: $(k, v_1, v_2, \cdots, v_M, u)$

#### Key Design 2: Caption Enhancement

- Uses a pre-trained multi-modal model to generate text descriptions for each candidate.
- Uses the descriptions as additional queries, paired with identifiers for training.
- Effectively bridges the semantic gap between multi-modal identifiers and natural language queries.

#### Key Design 3: Coarse-to-Fine Feature Fusion

Uses a standard encoder-decoder architecture. Since each encoder layer captures hierarchical semantic representations, a two-branch fusion strategy is designed:

**Coarse Fusion**:
$$Z = W[E_1, E_2, \ldots, E_S] + b$$
Concatenates the outputs of all encoder layers, passes them through a fusion layer, performs cross-attention with decoder inputs, and applies self-gated sigmoid post-processing.

**Fine Fusion**:
$$L(Y, E(q)) = \sum_{i=1}^{S} \alpha_i \odot \mathcal{C}(Y, E_i)$$
Adopting a MoE-like approach, each encoder layer is treated as an "expert." The outputs of each layer interact independently with the decoder, and their contributions are adjusted by learnable weights.

Finally, the outputs of the coarse and fine fusions are added to serve as the input for the next decoder layer.

### Loss & Training

- Standard cross-entropy loss: Maximizes the probability of generating the correct identifier.
- **Bidirectional KL divergence loss** (R-Drop): Keeps the output distributions of two forward passes (with different dropouts) consistent to prevent overfitting.
$$\mathcal{L}(\theta) = \sum_{(q,d)} (\log p(d|E(q), \theta) + \omega \mathcal{L}_{KL})$$

### Inference
- Uses **constrained beam search**: Builds a prefix tree using the prefix database to restrict the model to only generate valid identifiers.

## Experiments

### Datasets
- Text-Image: Flickr30K, MS-COCO
- Text-Audio: Clotho, AudioCaps
- Text-Video: MSR-VTT, MSVD

### Main Results

**vs. Single-Tower Models (Table 1)**:

| Method | Flickr30K R@1 | Flickr30K R@10 | Throughput |
|------|-------------|---------------|--------|
| BLIP-2 | 89.7 | 98.9 | 1.68/s |
| InternVL-G | 85.0 | 98.6 | 2.03/s |
| **CART** | **81.8** | **98.4** | **105.8/s** |

CART performs comparably to single-tower models in Recall (R@10 is only 0.5 lower), while achieving a **63x improvement in throughput**.

**vs. Dual-Tower Models (Table 2 Selected)**:

| Task | Method | R@1 | R@5 | R@10 |
|------|------|-----|-----|------|
| Text-Image (Flickr) | ImageBind | 74.9 | 93.0 | 96.1 |
| | **CART** | **81.8** | **96.1** | **98.4** |
| Text-Audio (Clotho) | ONE-PEACE | 22.4 | 49.0 | 62.7 |
| | **CART** | **46.4** | **70.6** | **76.0** |
| Text-Video (MSR-VTT) | Cap4Video | 49.3 | 74.3 | 83.8 |
| | **CART** | **52.6** | **75.4** | **84.2** |

**The advantage is most significant in audio retrieval**, with an R@1 gain of over 100%.

**vs. Generative Retrieval Models (Table 3)**:
CART significantly outperforms GRACE (which uses predefined identifiers), achieving 81.78 vs. 68.4 (Atomic ID) on Flickr30K R@1.

### Ablation Study (Table 4, Flickr30K)

| Setting | R@1 | R@10 | MRR@10 |
|------|-----|------|--------|
| w/o consistency loss | 81.64 | 98.04 | 87.85 |
| w/o fusion strategy | 75.54 | 96.72 | 83.11 |
| w/o K-Means | 79.50 | 97.52 | 86.12 |
| w/o RQ-VAE | 76.22 | 96.16 | 83.31 |
| **CART (Full)** | **81.78** | **98.38** | **88.04** |

- **Fusion strategy** has the greatest impact (removing it drops R@1 by 6.2%), indicating that multi-layer feature interaction is crucial.
- Removing RQ-VAE (using only hierarchical K-Means) also causes a significant performance drop, as hierarchical K-Means loses semantic information between clusters.
- The prior knowledge provided by K-Means contributes significantly to the accurate prediction of the first token.

### Efficiency Analysis
- As the number of candidates increases, the throughput of CLIP/CLAP continues to drop (since similarity must be computed individually).
- The throughput of CART remains stable on both CPU and GPU, independent of the number of candidates (as the identifiers are already encoded in the model parameters).
- The advantage is extremely significant in scenarios with 1M candidates and 100 concurrent queries.

## Highlights & Insights

1. **The first generative cross-modal retrieval framework that fully supports text-to-image/audio/video**
2. **Complementary design of K-Means + RQ-VAE**: K-Means provides global semantic classification, while RQ-VAE captures subtle differences.
3. **Excellent balance of efficiency and performance**: Performance is close to or even exceeds dual-tower models, while efficiency far outperforms single-tower models.
4. **Breakthrough in audio retrieval**: Drastically outperforms all baselines on Clotho and AudioCaps.
5. **Engineering wisdom of unique tokens**: A simple prefix-database scheme elegantly resolves the problem of identifier conflicts.

## Limitations & Future Work

1. **Not validated on ultra-large datasets**: Experimental dataset scales are limited (Flickr30K has only 31K images).
2. **Model update cost**: Adding new candidates requires regenerating identifiers and fine-tuning the model.
3. **Identifier quality depends on ImageBind**: Embedding quality directly affects the semantic quality of identifiers.
4. **Only text queries supported**: Other cross-modal directions, such as image-to-image or audio-to-text, have not been explored.
5. **Training requires 4 V100 GPUs**: Compared to contrastive learning in dual-tower models, the training paradigm is more complex.

## Related Work & Insights

- **Cross-Modal Retrieval**: CLIP, BLIP-2, InternVL, ImageBind, LanguageBind
- **Generative Retrieval**: DSI, NCI, GENRE, SEAL, GRACE
- **Vector Quantization**: RQ-VAE, VQ-VAE, SoundStream
- **Information Retrieval**: BM25, Dense Retrieval, Meshed-Memory Transformer

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ — Extends generative retrieval comprehensively to cross-modal scenarios for the first time, with a novel identifier design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — A comprehensive comparison across three modalities, six datasets, and three retrieval paradigms.
- **Practicality**: ⭐⭐⭐⭐ — Significant efficiency advantages in large-scale retrieval scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and rich diagrams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Maximal Matching Matters: Preventing Representation Collapse for Robust Cross-Modal Retrieval](maximal_matching_matters_preventing_representation_collapse_for_robust_cross-mod.md)
- [\[CVPR 2026\] Intra-class Distribution-guided Generative Hashing with Neighbor Refinement for Cross-modal Retrieval](../../CVPR2026/multimodal_vlm/intra-class_distribution-guided_generative_hashing_with_neighbor_refinement_for_.md)
- [\[CVPR 2025\] NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval](../../CVPR2025/multimodal_vlm/neighborretr_balancing_hub_centrality_in_cross-modal_retrieval.md)
- [\[ACL 2025\] REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)
- [\[CVPR 2026\] POGA: Paraphrased and Oppositional Graph Alignment for Fine-Grained Cross-Modal Retrieval](../../CVPR2026/multimodal_vlm/poga_paraphrased_and_oppositional_graph_alignment_for_fine-grained_cross-modal_r.md)

</div>

<!-- RELATED:END -->
