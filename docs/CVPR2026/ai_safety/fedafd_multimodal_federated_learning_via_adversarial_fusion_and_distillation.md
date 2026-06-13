---
title: >-
  [Paper Note] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation
description: >-
  [CVPR 2026][AI Safety][Multimodal Federated Learning] This paper proposes FedAFD, a framework that simultaneously improves model performance for both heterogeneous clients and the server in multimodal federated learning…
tags:
  - "CVPR 2026"
  - "AI Safety"
  - "Multimodal Federated Learning"
  - "Adversarial Alignment"
  - "Feature Fusion"
  - "knowledge distillation"
  - "Model Heterogeneity"
date: 2026-05-08
content_hash: b84008a1e0a41248
---

# FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation

**Conference**: CVPR 2026
**arXiv**: [2603.04890](https://arxiv.org/abs/2603.04890)  
**Code**: [Chao2433/FedAFD](https://github.com/Chao2433/FedAFD)  
**Area**: AI Safety / Federated Learning
**Keywords**: Multimodal Federated Learning, Adversarial Alignment, Feature Fusion, knowledge distillation, Model Heterogeneity

## TL;DR

This paper proposes FedAFD, a framework that simultaneously improves model performance for both heterogeneous clients and the server in multimodal federated learning through a three-stage design comprising bi-level adversarial alignment, granularity-aware feature fusion, and similarity-guided ensemble distillation.

## Background & Motivation

Multimodal Federated Learning (MFL) enables clients with different modalities to collaboratively train models without sharing raw data, yet faces three major challenges:

**Modality/Task Heterogeneity**: Different clients may process different modalities (image, text) and handle different tasks (classification, retrieval), leading to inconsistent feature spaces and model drift.

**Insufficient Personalization**: Existing methods often sacrifice local model performance in pursuit of improved global model performance.

**Model Heterogeneity**: Clients employ encoders with different architectures, precluding direct parameter-level aggregation.

Existing methods such as CreamFL focus solely on global model performance, neglecting local personalization, and lack effective mechanisms for handling modality/task discrepancies. The core mechanism of FedAFD is an edge–cloud collaborative framework designed to simultaneously enhance both global and local model performance.

## Method

### Overall Architecture

FedAFD consists of three iteratively executed training stages:
- **Stage ①**: The server trains on a public dataset and extracts global public features.
- **Stage ②**: Clients receive global representations and encoders, then train local models on private data via bi-level adversarial alignment and granularity-aware fusion.
- **Stage ③**: Clients extract local features from the public dataset, upload them to the server, and the server performs similarity-guided ensemble distillation to update the global model.

The system comprises three types of clients: $N_I$ unimodal image clients (image classification), $N_T$ unimodal text clients (text classification), $N_M$ multimodal clients (image–text retrieval), and a public multimodal dataset $\mathcal{P}$.

### Key Designs

1. **Bi-level Adversarial Alignment (BAA)**: The representation inconsistency between clients and the server is formulated as a federated domain adaptation problem. Each client is equipped with two adversarial discriminators:

    - **Intra-modal discriminator** $\mathcal{D}_c^{in}$: distinguishes between local and global representations within the same modality (e.g., $i_p^{c,k}$ vs. $i_p^{g,k}$).
    - **Cross-modal discriminator** $\mathcal{D}_c^{cr}$: distinguishes between local and global representations across modalities (e.g., $i_p^{c,k}$ vs. $t_p^{g,k}$).

   The adversarial loss is:
    $\mathcal{L}_{adv} = \frac{1}{|\mathcal{P}|}\sum_{k=1}^{|\mathcal{P}|}(\mathcal{L}_{in}^k + \mathcal{L}_{cr}^k)$
   where $\mathcal{L}_{in}^k = \log \mathcal{D}_c^{in}(i_p^{g,k}) + \log(1-\mathcal{D}_c^{in}(i_p^{c,k}))$, with an analogous formulation for the cross-modal term. The discriminators maximize this loss while the encoders minimize it, thereby reducing the distributional discrepancy between client and server representations.

2. **Granularity-aware Feature Fusion (GFF)**: After BAA aligns the feature distributions, excessive injection of global knowledge may degrade local performance. GFF adaptively fuses local and global features at the sample level via an attention mechanism:

   First-level fusion:
    $h_c^k = M(i_c^k + i_g^k) \otimes i_c^k + (1-M(i_c^k + i_g^k)) \otimes i_g^k$
   Second-level fusion (refinement):
    $\widetilde{i}_c^k = M(h_c^k) \otimes i_c^k + (1-M(h_c^k)) \otimes i_g^k$

   The attention weights are $M(x) = \sigma(T_1(x) + T_2(x))$, where $T_1$ and $T_2$ are parallel nonlinear transformations that capture multi-scale contextual information. The fused features are used to compute the task loss $\mathcal{L}_{task}$.

3. **Similarity-guided Ensemble Distillation (SED)**: This component addresses model heterogeneity on the server side by dynamically assigning aggregation weights based on feature similarity:

   Similarity score:
    $s^{c,k} = \log \frac{\exp(sim(i_p^{c,k}, i_p^{g,k}))}{\sum_{j=1}^{|\mathcal{P}|}\exp(sim(i_p^{c,k}, i_p^{g,j}))}$

   Normalized aggregation weight: $w^{c,k} = \frac{\exp(s^{c,k})}{\sum_{c'\in\pi_{img}}\exp(s^{c',k})}$

   Aggregated teacher representation: $i_{agg}^k = \sum_{c\in\pi_{img}} w^{c,k} \cdot i_p^{c,k}$

### Loss & Training

- **Client loss**: $\mathcal{L}_{task} + \beta \cdot \mathcal{L}_{adv}$, with $\beta=0.5$.
- **Server distillation loss**: $\mathcal{L}_{kd} = \frac{1}{|\mathcal{P}|}\sum_{k}(\|i_{agg}^k - i_p^{g,k}\|_2 + \|t_{agg}^k - t_p^{g,k}\|_2)$, with $\gamma=0.4$.
- Training protocol: 40 communication rounds, 5 local epochs per round, totaling 200 local updates.
- Client discriminators and encoders are trained alternately in an adversarial fashion.

## Key Experimental Results

### Main Results

Setting: 3 image clients (CIFAR-100), 3 text clients (AGNEWS), 4 multimodal clients (Flickr30k); server task: MS-COCO retrieval.

| Method | CIFAR-100 acc@1 | AGNEWS acc@1 | Flickr30k i2t R@1 | MS-COCO rsum R@1 | Convergence Rounds |
|------|----------------|-------------|-------------------|-----------------|---------|
| LOCAL | 28.07 | 48.35 | 22.33 | 57.54 | 29 |
| FedMD | 22.54 | 48.18 | 19.13 | 58.47 | 25 |
| FedGEMS | 22.84 | 48.30 | 18.93 | 58.62 | 27 |
| CreamFL | 22.14 | 42.16 | 18.38 | 59.61 | 21 |
| FedET | 31.86 | 49.38 | 22.63 | 58.92 | 27 |
| FedMKD | 24.99 | 47.99 | 22.33 | 59.18 | 21 |
| FedDFA | 23.09 | 43.79 | 19.68 | 59.10 | 26 |
| **FedAFD** | **33.18** | **51.98** | **32.48** | **60.16** | **20** |

Results are under a Non-IID setting. The advantage is more pronounced under IID settings: on CIFAR-100, FedAFD achieves 61.04% vs. FedET's 46.44%; on AGNEWS, 89.34% vs. 86.07%. FedAFD significantly outperforms all baselines on both client and server sides, with a particularly notable improvement of **+10 points** on Flickr30k i2t retrieval. It is worth noting that many baselines yield client performance even below LOCAL, indicating that global optimization harms personalization — FedAFD is the only method capable of simultaneously improving performance on both ends.

### Ablation Study

| Configuration | CIFAR-100 | AGNEWS | MS-COCO rsum | Note |
|------|----------|--------|-------------|------|
| FedAFD (Full) | 33.18 | 51.98 | 60.16 | Complete framework |
| w/o BAA | 33.56 | 49.03 | 59.29 | Removing adversarial alignment degrades server performance |
| w/o GFF | 24.94 | 44.46 | 59.72 | Removing feature fusion causes sharp drop in client performance |
| w/o SED | 32.21 | 50.20 | 59.56 | Removing ensemble distillation degrades global performance |

### Key Findings

- **GFF is critical for client performance**: Removing GFF causes CIFAR-100 accuracy to drop from 33.18% to 24.94% (−8.24%).
- **BAA primarily benefits server performance**: Its removal reduces rsum from 60.16 to 59.29, while text client performance also suffers.
- **Effect of public data size**: Increasing public data from 10k to 30k improves server rsum from 60.16 to 78.09, though client performance slightly decreases.
- **Convergence efficiency**: FedAFD reaches the baseline target (57.50) in only 20 rounds, whereas other methods require 21–29 rounds.

## Highlights & Insights

1. **Unified three-stage design**: FedAFD is the first framework to jointly address cross-modal/task alignment, task-aware personalization, and architecture-agnostic aggregation within a single system.
2. **Formulating MFL as domain adaptation**: Adversarial learning is employed to minimize the distributional discrepancy between client and server representations, offering a theoretically grounded perspective.
3. **Bidirectional optimization**: Unlike methods that focus exclusively on either global or local objectives, FedAFD simultaneously improves performance on both ends.
4. **Representation-level distillation**: Knowledge is transferred across heterogeneous models without requiring parameter-level compatibility.

## Limitations & Future Work

1. **Reliance on public data**: The framework depends on a public multimodal dataset $\mathcal{P}$, whose acquisition may be restricted in privacy-sensitive scenarios.
2. **Discriminator overhead**: Each client must maintain two additional discriminators, increasing computational and communication costs.
3. **Limited scalability validation**: Experiments involve only 10 clients; performance at large scales (100+ clients) remains unexplored.
4. **Limited modality coverage**: Validation is restricted to image and text modalities; extension to audio, video, and other modalities remains to be explored.

## Related Work & Insights

- **CreamFL**: Employs intra-modal and cross-modal contrastive regularization but neglects local performance; FedAFD's GFF module addresses this limitation.
- **FedDFA**: Uses boundary-aware distillation weights; FedAFD's SED further introduces sample-level dynamic weighting.
- **Domain adaptation theory**: Modeling modality/task discrepancies in federated learning as a domain adaptation problem offers a novel perspective for MFL research.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The three-module co-design is systematic, and the domain adaptation perspective is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive ablation study, IID/Non-IID dual settings, T-SNE visualization, and public data size analysis; the appendix includes hyperparameter and communication overhead analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with detailed mathematical derivations.
- **Value**: ⭐⭐⭐⭐ Represents a relatively complete solution for MFL and serves as a meaningful reference for heterogeneous federated learning research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning](../../AAAI2026/ai_safety/healsplit_towards_self-healing_through_adversarial_distillation_in_split_federat.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)
- [\[CVPR 2026\] UniGame: Turning a Unified Multimodal Model Into Its Own Adversary](unigame_turning_a_unified_multimodal_model_into_its_own_adversary.md)
- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)

</div>

<!-- RELATED:END -->
