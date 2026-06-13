---
title: >-
  [Paper Note] Inductive Transfer Learning for Graph-Based Recommenders
description: >-
  [NeurIPS 2025][Audio & Speech][Graph Neural Networks] This paper proposes NBF-Rec, a graph-based recommendation model built upon the Neural Bellman-Ford Network…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "Graph Neural Networks"
  - "Transfer Learning"
  - "Recommender Systems"
  - "Inductive Inference"
  - "Zero-Shot Recommendation"
date: 2026-05-08
content_hash: 3b03e4df67812aea
---

# Inductive Transfer Learning for Graph-Based Recommenders

**Conference**: NeurIPS 2025
**arXiv**: [2510.22799](https://arxiv.org/abs/2510.22799)  
**Code**: None  
**Area**: Audio & Speech
**Keywords**: Graph Neural Networks, Transfer Learning, Recommender Systems, Inductive Inference, Zero-Shot Recommendation

## TL;DR

This paper proposes NBF-Rec, a graph-based recommendation model built upon the Neural Bellman-Ford Network, which supports inductive transfer learning across datasets with completely disjoint users and items, enabling zero-shot cross-domain recommendation and lightweight fine-tuning adaptation.

## Background & Motivation

**Background**: Graph neural network-based recommender systems (e.g., LightGCN) perform well within a single domain, but are primarily trained in a transductive manner and cannot generalize to new users, new items, or new datasets.

**Limitations of Prior Work**:
   - Existing cross-domain recommendation methods assume overlapping users or items between the source and target domains, limiting their applicability.
   - Methods based on adversarial training, contrastive disentanglement, and meta-learning still rely on aligned entity spaces or domain-specific supervision.
   - Large-scale pre-trained models (P5, GPTRec) require substantial pre-training and inference resources, and depend on textual or visual side information.

**Key Challenge**: Transfer learning has become standard practice in NLP and CV, yet it remains largely unexplored in graph-based recommendation—particularly in scenarios with completely disjoint users and items.

**Goal**: Enable inductive transfer learning across user-item graphs with fully disjoint entities, supporting both zero-shot recommendation and fine-tuning adaptation.

**Key Insight**: The path-aggregation message-passing mechanism of NBFNet is leveraged to dynamically compute node representations (rather than pre-learned embeddings), thereby achieving inductive generalization; edge feature encoding is integrated to enhance the capture of interaction-level information.

**Core Idea**: Rather than learning node-specific parameters, the model learns the message-passing process itself, enabling generalization to entirely unseen user-item graphs.

## Method

### Overall Architecture

NBF-Rec builds upon NBFNet (Neural Bellman-Ford Network) and frames recommendation as a link prediction task on a bipartite graph. Given a query user $u$, the model dynamically computes representations for all nodes via multi-layer message passing and produces a score for each candidate item.

### Key Designs

1. **Query-Conditioned Initialization**

    - **Function**: Initialize representations of all nodes conditioned on the query user.
    - **Mechanism**: $h_v^{(0)} = \mathbf{1}(u = v)$, i.e., only the node corresponding to the query user is initialized to 1, while all others are set to 0. All information propagates outward from the query user.
    - **Design Motivation**: Ensures that representations are dynamically computed for each query without relying on pre-computed node embeddings.

2. **Edge Feature Embedding**

    - **Function**: Encode raw edge features (ratings, timestamps, categories, play counts, etc.) into embeddings suitable for message passing.
    - **Mechanism**: A two-level MLP structure:
    $g(r) = \text{MLP}_{\text{emb}}(\text{MLP}_{\text{proj}}(r))$
        - $\text{MLP}_{\text{proj}}$: a dataset-specific projection MLP that handles heterogeneous edge features across different datasets
        - $\text{MLP}_{\text{emb}}$: a shared backbone embedding MLP
    - **Novelty**: While the original NBFNet relies solely on graph structural information, NBF-Rec incorporates edge features to enable learning of richer interaction patterns.

3. **Message Passing Mechanism**

    - **Function**: At each layer $t$, aggregate neighbor messages to update node representations.
    - **Mechanism**:
    $M_v^{(t)} = \{\text{MESSAGE}(h_x^{(t-1)}, \mathbf{w}_q(x,r,v)) \mid (x,r,v) \in \mathcal{E}(v)\}$
        - Edge weights: $\mathbf{w}_q(x,r,v) = \text{MLP}_t(g(r))$, with a separate MLP per layer
        - Message function: non-parametric DistMult operation
        - Node update: aggregation (summation) + linear transformation + layer normalization + activation
        - Residual connection incorporating initial embeddings: $\text{AGGREGATE}(M_v^{(t)} \cup \{h_v^{(0)}\})$

4. **Score Generation**

    - **Function**: After $T$ layers of message passing, compute a recommendation score for each node.
    - **Mechanism**:
    $\text{score}(u,q,v) = \text{MLP}_{\text{score}}(\text{concat}(h_v^{(T)}, h_v^{(0)}))$
    - The final-layer embedding and the initial embedding are concatenated and passed through an MLP to produce a scalar score.

5. **Key to Inductive Generalization**

    - The model learns no node-specific parameters; all parameters reside in the message-passing MLPs and aggregation operations.
    - Heterogeneous edge feature formats across datasets are handled via dataset-specific projection MLPs.
    - Representations are computed dynamically at inference time, requiring no pre-computed embeddings.

### Loss & Training

- **Cross-Entropy Loss**:
  $$\mathcal{L} = -\log p(u,q,v) - \sum_{i=1}^{n} \frac{1}{n} \log(1-p(u'_i, q, v'_i))$$
  where $(u,q,v)$ is a positive sample and $\{(u'_i, q, v'_i)\}$ are strictly negative samples drawn via uniform random sampling (i.e., not present in the training set).
- **Batch-Edge Removal During Training**: Edges in the current batch are removed from the message-passing graph, forcing the model to rely on non-trivial paths rather than direct connections when learning relational patterns.
- **Three Settings**: end-to-end training / zero-shot transfer / fine-tuning.

### Computational Complexity

The total forward-pass complexity is $\mathcal{O}(T|E| + |V|)$, linear in the number of nodes and edges. Although inference overhead is higher than that of LightGCN (which pre-computes embeddings), the model supports inductive generalization.

## Key Experimental Results

### Datasets

Seven real-world recommendation datasets:

| Dataset | #Users | #Items | #Interactions | Domain |
|--------|--------|--------|---------------|--------|
| ML-1M | 5,950 | 2,811 | 364,654 | Movies |
| LastFM | 1,867 | 1,867 | 39,717 | Music |
| Amazon B. | 52,204 | 57,289 | 293,912 | E-commerce |
| Gowalla | 29,858 | 70,839 | 712,504 | Location check-in |
| Epinions | 21,008 | 13,887 | 266,791 | Product reviews |
| BookX | 12,720 | 18,318 | 276,334 | Books |
| Yelp18 | 31,668 | 38,048 | 1,097,007 | Local businesses |

### Main Results

#### Zero-Shot / Fine-Tuning / End-to-End Comparison

Pre-training sources: Amazon Beauty + Epinions

| Setting | ML-1M | LastFM | Amazon B. | Gowalla | Epinions | BookX | Yelp18 |
|---------|-------|--------|-----------|---------|----------|-------|--------|
| Zero-shot | Competitive | Below par | — | Below par | — | Near baseline | Near baseline |
| Fine-tuning | Improved | Substantially improved | Improved | Substantially improved | Improved | Improved | Improved |
| End-to-end | Baseline | Baseline | Baseline | Baseline | Baseline | Baseline | Baseline |

Key observations:
- On ML-1M, BookX, and Yelp18, zero-shot performance falls **within 5%** of the end-to-end baseline.
- Fine-tuning consistently improves performance across all datasets, with the most notable gains on LastFM and Gowalla.

### Ablation Study

#### Cross-Dataset Transfer Heatmap

| Key Finding | Description |
|-------------|-------------|
| Asymmetric transfer | Transfer from LastFM→ML-1M is effective, but not vice versa |
| Self-transfer not optimal | BookX and Amazon Fashion benefit more from transfer from other datasets than from themselves |
| Edge feature impact | Low-information edge features (BookX, LastFM) generally reduce transferability |
| Graph scale vs. features | Gowalla (large graph, sparse features) transfers better than Yelp (rich features but weak transferability) |

#### NBF-Rec vs. NBFNet Comparison

- NBF-Rec consistently outperforms NBFNet (graph structure only) in zero-shot and fine-tuning settings.
- In end-to-end training, NBF-Rec performs on par with or slightly better than NBFNet.
- This confirms the contribution of edge feature embeddings to transfer learning.

### Key Findings

- **Inductive transfer learning is feasible for graph-based recommendation**: Even when users and items are completely disjoint, NBF-Rec achieves meaningful zero-shot recommendation through the learned message-passing process.
- **Pre-training on a different domain can outperform pre-training on the target domain**: Cross-domain inductive biases are sometimes more effective than in-domain signals.
- **Edge features are a double-edged sword**: Rich edge features improve in-domain performance but do not necessarily enhance cross-domain transferability.
- **Lightweight fine-tuning bridges the gap**: A small amount of in-domain supervision is sufficient to bring the zero-shot model close to fully supervised performance.

## Highlights & Insights

- **This is the first demonstration of scalable inductive transfer across fully disjoint user-item graphs**, representing an important milestone in graph-based recommendation research.
- **Elegant design**: Built as an extension of NBFNet, the core modification is the introduction of edge feature embeddings and a dataset-specific projection MLP, making the engineering implementation relatively lightweight.
- **The discovery of asymmetric transferability** is particularly intriguing and suggests that source domain selection is a research question worthy of deeper investigation.
- **No reliance on textual or visual side information**: The model operates purely on interaction graph structure and edge features, granting it broad applicability.

## Limitations & Future Work

- **Inference cost**: Each query requires a full message-passing forward pass, resulting in higher inference cost than methods that pre-compute embeddings.
- **Medium-scale datasets**: The largest dataset, Yelp18, contains only approximately one million interactions; scalability to industrial-scale data remains unverified.
- **Edge feature engineering**: Different datasets require different feature preprocessing pipelines, and the dataset-specific projection MLPs add implementation complexity.
- **No direct comparison with large-scale pre-trained recommendation models (P5, GPTRec) under identical conditions.**
- **The negative sampling strategy is relatively simple** (uniform random); more advanced strategies could potentially yield further gains.
- Future directions include multi-graph joint pre-training, evaluation on larger-scale datasets, and more efficient inference strategies.

## Related Work & Insights

- **NBFNet**: The Neural Bellman-Ford Network was originally developed for knowledge graph link prediction; this paper successfully transfers the framework to recommender systems.
- **ULTRA**: Zero-shot transfer for knowledge graph completion, representing the closest methodological relative.
- **LightGCN**: A representative transductive method, serving as the primary comparison baseline.
- **Insight**: The message-passing mechanism itself can serve as transferable "knowledge" across domains, as opposed to node embeddings. This observation may generalize to other graph learning tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ First demonstration of scalable inductive recommendation transfer across purely interaction-based graphs with disjoint entities; the problem formulation is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 7 datasets, three settings, and cross-domain transfer heatmap analysis
- Writing Quality: ⭐⭐⭐⭐ Clear methodology, well-specified equations, and well-designed experiments
- Value: ⭐⭐⭐⭐ Opens a new direction for transfer learning in graph-based recommendation; the lightweight design has practical potential

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation](can_llms_outshine_conventional_recommenders_a_comparative_evaluation.md)
- [\[NeurIPS 2025\] Enabling Differentially Private Federated Learning for Speech Recognition: Benchmarks, Adaptive Optimizers and Gradient Clipping](enabling_differentially_private_federated_learning_for_speech_recognition_benchm.md)
- [\[ICLR 2026\] AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](../../ICLR2026/audio_speech/ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)
- [\[ICML 2026\] Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration](../../ICML2026/audio_speech/polyphonia_zero-shot_timbre_transfer_in_polyphonic_music_with_acoustic-informed_.md)
- [\[ICCV 2025\] Learning to See Inside Opaque Liquid Containers using Speckle Vibrometry](../../ICCV2025/audio_speech/learning_to_see_inside_opaque_liquid_containers_using_speckle_vibrometry.md)

</div>

<!-- RELATED:END -->
