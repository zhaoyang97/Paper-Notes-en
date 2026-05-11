---
title: >-
  [Paper Note] Synergy over Discrepancy: A Partition-Based Approach to Multi-Domain LLM Fine-Tuning
description: >-
  [NeurIPS 2025][LLM/NLP][multi-domain fine-tuning] This paper proposes a partition-based multi-stage fine-tuning framework that strategically partitions multiple domains into subsets (stages) to maximize inter-domain syne…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "multi-domain fine-tuning"
  - "inter-domain synergy"
  - "partition strategy"
  - "generalization bound"
  - "Adapter"
date: 2026-05-08
content_hash: 53ef8b56469838f7
---

# Synergy over Discrepancy: A Partition-Based Approach to Multi-Domain LLM Fine-Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2511.07198](https://arxiv.org/abs/2511.07198)
**Code**: Not released
**Area**: LLM/NLP
**Keywords**: multi-domain fine-tuning, inter-domain synergy, partition strategy, generalization bound, Adapter

## TL;DR

This paper proposes a partition-based multi-stage fine-tuning framework that strategically partitions multiple domains into subsets (stages) to maximize inter-domain synergy while minimizing negative transfer, and derives a novel generalization bound to theoretically support the partitioning strategy.

## Background & Motivation

Single-domain fine-tuning of LLMs is well-studied (LoRA, Adapter, etc.), but practical scenarios frequently require **simultaneous adaptation to multiple heterogeneous domains** (e.g., clinical text, social media, legal documents), a problem that remains largely underexplored.

Limitations of existing multi-domain methods:
- **Joint fine-tuning**: Training all domains together leads to mutual interference among domain features (negative transfer).
- **Independent fine-tuning**: Training each domain separately fails to exploit inter-domain synergy.
- **Adversarial training / distribution alignment methods**: Approaches such as MDAN and M3SDA neglect inter-domain synergistic relationships.
- **Adapter methods**: Reduce parameter counts but do not leverage domain partitioning to maximize overall performance.

Core Problem: **How to effectively and efficiently fine-tune a single LLM across multiple heterogeneous domains, exploiting inter-domain synergy while mitigating negative transfer?**

## Method

### Overall Architecture

The $k$ source domains are partitioned into $M$ disjoint stages $S_1, \ldots, S_M$, within each of which the assigned domains are fine-tuned jointly. The framework consists of two components:
1. **Partition optimization**: Maximize inter-domain synergy while minimizing intra-stage discrepancy and capacity overhead.
2. **Multi-stage Adapter fine-tuning**: Sequentially train the LLM backbone and domain-specific Adapters stage by stage.

### Key Design 1: Partition Objective

The partition $(S_1, \ldots, S_M)$ is optimized to maximize:

$$\mathcal{G}(S_1,\ldots,S_M) = -\sum_{t=1}^{M}\left[\underbrace{\sum_{i<j \in S_t} d(\mathcal{D}_i, \mathcal{D}_j)}_{\text{discrepancy}} - \lambda \underbrace{\sum_{i<j \in S_t} s(\mathcal{D}_i, \mathcal{D}_j)}_{\text{synergy}} + \underbrace{\mu_\theta \|\Delta\theta^t\|^2 + \mu_\phi \sum_{j \in S_t} \|\phi_j^t\|^2}_{\text{capacity overhead}}\right]$$

where:
- Discrepancy measure: $d(\mathcal{D}_i, \mathcal{D}_j) = \text{JS}(P_i, P_j)$ (Jensen–Shannon divergence)
- Synergy measure: $s(\mathcal{D}_i, \mathcal{D}_j) = \frac{1}{2}(\text{Jacc}(V_i, V_j) + \cos(\mu_i, \mu_j))$ (vocabulary overlap + embedding cosine similarity)
- $\lambda$ balances the relative weight of synergy versus discrepancy

### Key Design 2: Generalization Bound

**Theorem 3.1 (Multi-source concurrent generalization bound)**:

$$\sum_j \alpha_j \mathcal{L}(\theta, \{\phi_i\}; \mathcal{D}_j) \leq \sum_j \alpha_j \hat{\mathcal{L}} + \underbrace{2LB(\rho_\theta + \sum_j \alpha_j \rho_\phi)}_{\Gamma: \text{capacity term}} + \underbrace{\frac{\beta}{k}\sum_{i,j} d(\mathcal{D}_i, \mathcal{D}_j)}_{\text{discrepancy term}} + O\left(\sqrt{\frac{\ln(1/\delta)}{n}}\right)$$

**Theorem 3.2 (Optimality of multi-stage partition)**: The optimal partition $(S_1^*, \ldots, S_M^*)$ minimizes the right-hand side of the above bound.

**Corollary 3.1**: High-synergy subsets tend to be assigned to the same stage — when $\Lambda > \lambda^{-1}(\gamma + \text{Cap}(U))$, a high-synergy domain set $U$ is necessarily grouped together in the optimal partition.

### Key Design 3: Algorithm

Procedure:
1. Compute pairwise discrepancy and synergy matrices for all domain pairs ($O(k^2)$).
2. Solve for the optimal partition via single-linkage hierarchical clustering ($O(k^2 \log k)$) or ILP.
3. Initialize $\theta^0 = \theta^*$ (pretrained parameters) and $\phi_j^0 = 0$.
4. Stage-wise optimization: for stage $t$, jointly fine-tune $\theta$ and $\{\phi_j\}_{j \in S_t}$ on the data of domains in $S_t$.

Constraints: $\|\theta^t - \theta^{t-1}\|_2 \leq \rho_\theta$, $\|\phi_j^t\|_2 \leq \rho_\phi$.

### Loss & Training

Each stage employs a weighted multi-domain loss: $\sum_{j \in S_t} \alpha_j^t \mathcal{L}(\theta, \{\phi_i\}; \mathcal{D}_j)$, with parameter norm constraints to preserve the implicit regularization of pretraining. Adapters for out-of-stage domains remain frozen: $\phi_j^t = \phi_j^{t-1}$ for $j \notin S_t$.

## Key Experimental Results

### Main Results

Performance across three LLM backbones on four tasks (LLaMA2-7B / LLaMA2-13B / Falcon-40B):

| Method | NSum | Q&A | Sent | Topic | Type |
|--------|------|-----|------|-------|------|
| FULL | 41.2/42.1/43.2 | 64.7/66.3/68.2 | 89.0/89.8/90.4 | 86.5/87.1/88.3 | Baseline |
| LoRA | 41.0/42.0/42.5 | 63.9/65.1/66.5 | 88.4/89.1/— | 86.2/86.9/— | Single-domain |
| MDAN | 39.7/40.5/41.7 | 62.8/64.0/66.1 | 88.1/88.9/89.3 | 85.9/86.3/87.0 | Domain adaptation |
| M3SDA | 40.5/41.7/42.3 | 63.1/64.9/66.6 | 88.6/89.4/89.9 | 86.1/86.7/87.4 | Domain adaptation |
| **PMS-FT (Ours)** | **42.1/43.1/44.3** | **66.1/67.8/69.5** | **89.9/90.5/91.1** | **87.4/88.0/89.0** | Partition multi-stage |

PMS-FT outperforms all baselines across all models and tasks.

### Ablation Study

- **Partition vs. no partition**: The partitioning strategy outperforms full-domain joint fine-tuning across all configurations, validating the value of reducing inter-domain interference.
- **Contribution of synergy measure**: Removing the synergy term degrades performance, demonstrating that reducing discrepancy alone is insufficient and inter-domain complementarity must be exploited.
- **Effect of stage count $M$**: The optimal $M$ depends on the number of domains and their inter-domain relationships; too few stages fail to isolate conflicting domains, while too many prevent synergy exploitation.
- **Effect of capacity constraints**: Appropriate $\rho_\theta, \rho_\phi$ constraints prevent catastrophic forgetting while retaining sufficient adaptation capacity.

### Key Findings

1. Inter-domain synergy and discrepancy are equally important dimensions in multi-domain fine-tuning — attending to only one is insufficient.
2. The additional computational overhead of partitioning is negligible ($O(k^2 \log k)$), yet the performance gains are substantial.
3. The theoretical generalization bound is consistent with experimental trends: domain groupings with low discrepancy and high synergy consistently yield better performance.
4. The method scales to LLMs of varying sizes (validated from 7B to 40B parameters).

## Highlights & Insights

1. **Formal modeling of inter-domain synergy**: This work is the first to incorporate "synergy" as an explicit optimization objective in multi-domain fine-tuning.
2. **Unification of theory and practice**: The generalization bound not only explains why partitioning is effective but also directly guides the design of the partitioning algorithm.
3. **Generality of the framework**: The approach is compatible with any parameter-efficient fine-tuning method (LoRA, Adapter, etc.).
4. **Practical usability**: The partitioning step incurs negligible computation and does not increase GPU memory during fine-tuning.

## Limitations & Future Work

1. **Limited domain scale**: Experiments cover at most $k \leq 10$ domains; performance under large-scale domain settings (e.g., hundreds of domains) remains unknown.
2. **Heuristic synergy measure**: Vocabulary Jaccard similarity combined with embedding cosine similarity may lack precision; more sophisticated domain relationship modeling offers room for improvement.
3. **Stage ordering not optimized**: The current approach assumes stage order is irrelevant, yet sequential effects across stages may exist in practice.
4. **Static partitioning**: Partitions are determined prior to training and are not adjusted during training — dynamic partitioning may yield further gains.
5. **Focus on NLP tasks**: The method has not been validated on multimodal or non-textual domains.
6. **Loose capacity bounds**: The Rademacher complexity bound for Adapter capacity may not be sufficiently tight.

## Related Work & Insights

- **Distinction from Ganin & Lempitsky (2015) adversarial domain training**: No adversarial objective is used; domain relationships are instead managed directly through partitioning.
- **Complementarity with LoRA (Hu et al., 2021)**: LoRA is a parameter-efficient single-domain method; the proposed partitioning strategy can be applied on top of it.
- **Distinction from continual learning (Xu et al., 2025)**: Continual learning assumes tasks arrive sequentially, whereas this work assumes all domains are simultaneously available.
- **Broader implications**: The synergy–discrepancy partitioning paradigm generalizes to multi-task learning, federated learning, and other multi-source settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The partitioning strategy and synergy modeling constitute novel framework-level contributions.
- **Theoretical Depth**: ⭐⭐⭐⭐ — The generalization bound derivation is complete, though some assumptions (Lipschitz continuity, bounded norms) are standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validation across multiple models and tasks is thorough, though the number of domains tested is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — The structure is clear, with tight integration between theory and experiments.
- **Value**: ⭐⭐⭐⭐ — The method is simple, effective, and readily integrable into existing fine-tuning pipelines.
- **Overall**: ⭐⭐⭐⭐ (8/10) — Addresses a practical pain point in multi-domain fine-tuning with a well-grounded combination of theory and practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning](sparse_mezo_less_parameters_for_better_performance_in_zeroth-order_llm_fine-tuni.md)
- [\[NeurIPS 2025\] Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs](triplets_better_than_pairs_towards_stable_and_effective_self-play_fine-tuning_fo.md)
- [\[NeurIPS 2025\] SPACE: Noise Contrastive Estimation Stabilizes Self-Play Fine-Tuning for Large Language Models](space_noise_contrastive_estimation_stabilizes_self-play_fine-tuning_for_large_la.md)
- [\[NeurIPS 2025\] MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery](moose-chem2_exploring_llm_limits_in_fine-grained_scientific_hypothesis_discovery.md)
- [\[ACL 2026\] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](../../ACL2026/llm_nlp/grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)

</div>

<!-- RELATED:END -->
