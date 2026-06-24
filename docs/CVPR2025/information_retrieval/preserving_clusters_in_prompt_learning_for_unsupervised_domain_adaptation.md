---
title: >-
  [Paper Note] Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation
description: >-
  [CVPR 2025][Information Retrieval & RAG][Unsupervised Domain Adaptation] The CRPL framework is proposed to improve prompt learning of CLIP in unsupervised domain adaptation (UDA) through source-augmented pseudo-labeling and an optimal transport-based cluster preservation strategy, ensuring that the text embeddings of target prompts better cover the cluster structures of visual embeddings.
tags:
  - "CVPR 2025"
  - "Information Retrieval & RAG"
  - "Unsupervised Domain Adaptation"
  - "Prompt Learning"
  - "Optimal Transport"
  - "Cluster Preserving"
  - "CLIP"
date: 2026-05-08
content_hash: be251adaff999ecf
---

# Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation

**Conference**: CVPR 2025  
**arXiv**: [2506.11493](https://arxiv.org/abs/2506.11493)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: Unsupervised Domain Adaptation, Prompt Learning, Optimal Transport, Cluster Preserving, CLIP

## TL;DR

The CRPL framework is proposed to improve prompt learning of CLIP in unsupervised domain adaptation (UDA) through source-augmented pseudo-labeling and an optimal transport-based cluster preservation strategy, ensuring that the text embeddings of target prompts better cover the cluster structures of visual embeddings.

## Background & Motivation

CLIP-based prompt learning has become a mainstream method for unsupervised domain adaptation (DAPL, MPA, PGA), but it suffers from three key issues:

1. **Poor pseudo-label quality**: Existing methods rely on CLIP zero-shot predictions as target domain pseudo-labels. However, when there is a significant distribution shift between the target domain and CLIP pre-training data, the accuracy of pseudo-labels drops sharply (e.g., only 50.4% on the Clipart domain of OfficeHome).

2. **Low-quality target prompts**: The authors surprisingly discover through experiments that the learned target domain prompts in existing methods perform very poorly in practice (Table 1: accuracy is only 47.6% when inferring solely with target prompts). Their seemingly decent overall performance completely relies on the contribution of source domain prompts. The source-target prompt alignment strategy has limited effectiveness.

3. **Unexploited cluster structure**: Visual embeddings of CLIP naturally exhibit strong clustering properties (Table 1: supervised training of target prompts can reach 99.6%), but existing methods ignore this geometric structure.

## Method

### Overall Architecture

CRPL (Clustering Reinforcement Prompt Learning) consists of two innovative components: (1) Source-augmented Pseudo-Labeling (SPL), which utilizes predictions from source prompts to improve the quality of target domain pseudo-labels; and (2) Cluster Preserving Regularization ($\mathcal{L}_\mathcal{W}$), which forces text embeddings to become the centroids of visual clusters by minimizing the Wasserstein distance between the target domain visual embedding distribution and the text embedding distribution.

### Key Designs

1. **Source-augmented Pseudo-Labeling (SPL)**:
    - Function: Provides more accurate target domain pseudo-labels than CLIP zero-shot predictions.
    - Mechanism: For each target domain sample, calculate the distance between its visual embedding and each source domain category centroid. Multiple source prompt predictions are aggregated and weighted using distance-aware weights $w_{i,k}(x) = \frac{\exp(\|z_{pre} - c_k^i\|_2)}{\sum_{i'}\exp(\|z - c_k^{i'}\|_2)}$, and then averaged with the base prompt prediction: $\boldsymbol{\tau}_{ave}^k(x) = \frac{1}{2}\boldsymbol{\tau}_{base}^k + \frac{w_{k,i}(x)}{2}\sum_i \boldsymbol{\tau}_{S_i}^k$.
    - Design Motivation: Different source domains possess distinct transfer capabilities to the target domain (Table 1: contributions of source domains to different target domains vary significantly in multi-source settings), requiring adaptive weighting based on sample-source domain distances.

2. **Optimal Transport-based Cluster Preserving ($\mathcal{L}_\mathcal{W}$)**:
    - Function: Forces target domain text embeddings to act as the centroids of visual embedding clusters.
    - Mechanism: Treat text embeddings $\{\boldsymbol{\tau}_T^k\}_{k=1}^K$ as a discrete distribution $\mathbb{P}_{\tau,\pi}$ over $K$ centroids, and treat target domain visual embeddings as an empirical distribution $\mathbb{P}^T$. Then, minimize the Wasserstein distance between them: $\mathcal{L}_\mathcal{W} = \mathcal{W}_{d_z}(\mathbb{P}_{\tau,\pi}, \mathbb{P}^T)$, where $d_z$ is the cosine distance.
    - Design Motivation: Even after enhancement, pseudo-labels still contain errors, causing text embeddings to bias towards the centroid of correctly labeled samples rather than the centroid of the entire cluster (Figure 1b). Minimizing the Wasserstein distance is equivalent to optimal cluster assignment (Lemma 1), which can automatically correct the position of text embeddings.

3. **Joint Training Strategy**:
    - Function: Coordinates three objectives: source domain supervision, target domain pseudo-labeling, and cluster preservation.
    - Mechanism: The total loss is $\mathcal{L}_{total} = \mathcal{L}_S + \lambda_T \mathcal{L}_T + \lambda_\mathcal{W} \mathcal{L}_\mathcal{W}$, where the source domain uses cross-entropy loss, and the target domain uses soft cross-entropy loss based on augmented pseudo-labels.
    - Design Motivation: SPL and $\mathcal{L}_\mathcal{W}$ mutually reinforce each other—SPL provides directional guidance (preventing text embeddings from drifting to incorrect clusters), while $\mathcal{L}_\mathcal{W}$ ensures that text embeddings cover the overall cluster instead of being biased.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_S + \lambda_T \mathcal{L}_T + \lambda_\mathcal{W} \mathcal{L}_\mathcal{W}$$

- $\mathcal{L}_S$: Source domain cross-entropy loss (labeled)
- $\mathcal{L}_T$: Target domain soft cross-entropy loss (SPL pseudo-labels)
- $\mathcal{L}_\mathcal{W}$: Wasserstein clustering regularization
- $\lambda_T = \lambda_\mathcal{W} = 0.5$, $\pi_k = 1/K$ (uniform prior)

## Key Experimental Results

### Main Results

| Dataset | Setting | CRPL | PGA (SOTA) | DAPL | Gain |
|--------|------|------|----------|------|------|
| ImageCLEF | Source-combined Avg | **90.3%** | 88.2% | 87.1% | +2.1% |
| Office-Home | Source-combined Avg | **77.6%** | 74.5% | 72.8% | +3.1% |
| DomainNet | Source-combined Avg | **55.8%** | 55.4% | 52.0% | +0.4% |
| Office-Home | Multi-source Avg | **78.9%** | 75.5% | - | +3.4% |
| DomainNet | Multi-source Avg | **56.2%** | 55.4% | - | +0.8% |
| Office-Home | 12 Single-Source Pairs Avg | **74.4%** | 73.9% | 73.3% | +0.5% |

### Ablation Study (Office-Home)

| Configuration | Prompt for Inference | Ar | Cl | Pr | Rw | Description |
|------|---------|------|------|------|------|------|
| CPL (CLIP pseudo-labels) | $\tau_T$ | 47.6 | 29.0 | 53.5 | 63.5 | Baseline target prompt is extremely poor |
| SPL (source-augmented pseudo-labels) | $\tau_T$ | **75.8** | **62.9** | **86.6** | **87.2** | SPL significantly boosts target prompt |
| CPL + $\mathcal{L}_W$ | $\tau_T$ | 7.9 | 4.7 | 80.4 | 82.3 | Weak pseudo-labels + OT collapses |
| SPL + $\mathcal{L}_W$ | $\tau_T$ | **76.8** | **63.5** | **87.5** | **87.6** | Complementary, optimal combination |

### Key Findings

- **Target prompts in existing methods are "false" champions**: PGA's target prompt alone achieves only 72.3% (Ar) accuracy, and its reported good performance relies on the average inference of source prompt + target prompt. In contrast, CRPL's target prompt alone reaches 76.8%.
- **Strong complementarity between SPL and $\mathcal{L}_\mathcal{W}$**: Applying OT clustering alone (CPL + $\mathcal{L}_\mathcal{W}$) collapses in domains with weak pseudo-labels (Ar drops from 47.6% to 7.9%), but yields further improvements when paired with SPL.
- **Limited effectiveness of source-target prompt alignment**: This strongly challenges the existing paradigm and offers a new direction for the field.

## Highlights & Insights

- **Deep questioning of existing methods**: Dissecting inference experiments (target prompt only vs. averaged prompt) exposes the fact that target prompts in methods like DAPL/PGA are essentially inactive, providing a highly insightful analysis.
- **Elegant combination of clustering assumption and optimal transport**: Utilizing the intrinsic clustering behavior of CLIP visual embeddings, it pulls text embeddings towards cluster centroids via Wasserstein distance, which enjoys stronger theoretical guarantees (Lemma 1) than simple alignment losses.
- **Distance-aware cross-domain transfer weights**: Assigning different weights to different source domains is far more reasonable than simple averaging.

## Limitations & Future Work

- When the target domain significantly differs from the source domain (e.g., Quickdraw in DomainNet), the improvement from SPL is limited (only 10.6%), because the source domain prompt itself lacks transferability to this domain.
- The simple setting of $\lambda_T = \lambda_\mathcal{W} = 0.5$ may not be optimal for all scenarios.
- Computing the Wasserstein distance is computationally expensive when the number of classes $K$ is very large.
- The vision encoder remains completely frozen; future work could explore the effects of parameter-efficient fine-tuning.

## Related Work & Insights

- Relationship with DAPL, MPA, and PGA: All of them perform UDA using CLIP-based prompt learning, but this work exposes their implicit reliance on source prompts.
- Application of optimal transport in domain adaptation (e.g., DeepJDOT, OTDA): This work innovatively applies it to the optimization of prompt text embeddings rather than feature alignment.
- Inspiration: In other prompt learning scenarios, one could also inspect whether the learned prompts possess true independent representation capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing optimal transport into prompt learning for UDA is novel, and the critique of existing methods is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive ablation studies across three benchmarks: ImageCLEF, OfficeHome, and DomainNet.
- Writing Quality: ⭐⭐⭐⭐ Deep motivational analysis (the design of experiments in Table 1) and clear mathematical derivations.
- Value: ⭐⭐⭐⭐ Provides a fresh perspective and practical improvements for prompt-based UDA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval](../../ACL2026/information_retrieval/unite_uncertainty-based_iterative_document_sampling_for_domain_adaptation_in_inf.md)
- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](../../ACL2026/information_retrieval/domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[CVPR 2025\] COBRA: COmBinatorial Retrieval Augmentation for Few-Shot Adaptation](cobra_combinatorial_retrieval_augmentation_for_few-shot_adaptation.md)
- [\[ACL 2026\] More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval](../../ACL2026/information_retrieval/more_than_efficiency_embedding_compression_improves_domain_adaptation_in_dense_r.md)
- [\[CVPR 2025\] GOAL: Global-Local Object Alignment Learning](goal_global-local_object_alignment_learning.md)

</div>

<!-- RELATED:END -->
