---
title: >-
  [Paper Note] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning
description: >-
  [NeurIPS 2025][LLM Safety][Federated Learning] This paper proposes FedLEASE, which addresses two critical challenges in federated LoRA fine-tuning: (1) automatically determining the optimal number of experts and their as…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Federated Learning"
  - "LoRA"
  - "Mixture of Experts"
  - "Adaptive Clustering"
  - "Parameter-Efficient Fine-Tuning"
date: 2026-05-08
content_hash: 0c6043926bd702ed
---

# Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2509.15087](https://arxiv.org/abs/2509.15087)  
**Code**: N/A  
**Area**: AI Safety
**Keywords**: Federated Learning, LoRA, Mixture of Experts, Adaptive Clustering, Parameter-Efficient Fine-Tuning

## TL;DR
This paper proposes FedLEASE, which addresses two critical challenges in federated LoRA fine-tuning: (1) automatically determining the optimal number of experts and their assignment via LoRA B-matrix similarity clustering, and (2) enabling adaptive top-M expert selection through an expanded routing space of $2M-1$ dimensions, allowing each client to determine how many experts to use. FedLEASE achieves an average improvement of 5.53% over the strongest baseline on GLUE.

## Background & Motivation

**Background**: Federated Learning (FL) enables privacy-preserving distributed LLM fine-tuning, while LoRA provides a parameter-efficient fine-tuning paradigm. However, a single shared LoRA module struggles to handle heterogeneous data across clients spanning different tasks and domains.

**Limitations of Prior Work**: (a) Existing methods such as FedIT and FedSA have all clients share a single LoRA module, which performs poorly under task heterogeneity; (b) assigning a dedicated LoRA to each client leads to redundancy and eliminates cross-client knowledge sharing; (c) LoRA-MoE approaches require manually specifying a fixed top-$k$ expert count, whereas the optimal $k$ differs across clients.

**Key Challenge**: Too few experts (one shared) fail to capture domain diversity, while too many experts (one per client) introduce redundancy and performance degradation.

**Key Insight**: Two key observations motivate this work — (a) cosine similarity of B matrices reflects task similarity (A matrices do not); (b) different clients require different numbers of experts.

**Core Idea**: Use B-matrix clustering to determine the number of experts and their assignment, and use an expanded routing space to allow each client to adaptively decide how many experts to employ.

## Method

### Overall Architecture
The framework consists of two phases: (1) **Initialization** — each client trains for a few local rounds to obtain LoRA parameters, uploads B matrices to the server, which then applies silhouette-coefficient-based clustering to determine $M$ experts and initialize them; (2) **Iterative Training** — clients update only their assigned expert and router, the server aggregates within clusters, and this process repeats.

### Key Designs

1. **Adaptive LoRA Expert Allocation**:

    - **Function**: Automatically determine the number of experts $M$ and the client-to-expert mapping.
    - **Mechanism**:
        - Each client trains locally for $E$ epochs to obtain $(A_i, B_i)$.
        - Cosine distance between B matrices is computed as: $d(i,j) = \frac{1}{|L|}\sum_l (1 - \frac{\mathbf{B}_i^l \cdot \mathbf{B}_j^l}{\|\mathbf{B}_i^l\|\|\mathbf{B}_j^l\|})$
        - Hierarchical clustering is performed for $k \in \{2,...,M_{max}\}$, and the optimal $M = \arg\max S(k)$ is selected via the silhouette coefficient $S(k)$.
        - The LoRA parameters of clients within each cluster are averaged to initialize the corresponding expert.
    - **Design Motivation**: LoRA B matrices encode task-specific information (empirically verified), while A matrices encode general linguistic features.

2. **Adaptive Top-M Expert Selection**:

    - **Function**: Allow each client to automatically determine how many experts (1 to $M$) to use for each input.
    - **Mechanism**: The router output is expanded from $\mathbb{R}^{M \times d}$ to $\mathbb{R}^{(2M-1) \times d}$.
        - The first $M$ outputs are all connected to the client's own expert $E_j$.
        - The remaining $M-1$ outputs are each connected to one of the other experts.
        - Standard top-$M$ selection: if all $M$ selected slots are occupied by the first $M$ outputs (i.e., the client's own expert), only one expert is used; if some slots are assigned to other experts, multiple experts are engaged.
    - **Novelty**: This design guarantees that the client's own expert always participates (all first $M$ positions point to it), while allowing flexible incorporation of other experts without manually tuning $k$.

3. **Federated Aggregation**:

    - Each round, clients upload and download only their own expert and router, ensuring communication efficiency comparable to baselines.
    - Intra-cluster aggregation: expert parameters within the same cluster are averaged.
    - Cross-cluster knowledge sharing: realized through forward passes over routers and other experts.

### Loss & Training
- NLU: RoBERTa-Large (355M), GLUE benchmark, 16 clients, 25 rounds.
- NLG: LLaMA-2-7B (8-bit quantization), FLAN dataset, 8 clients, 10 rounds.
- LoRA applied to Q and V projection matrices.

## Key Experimental Results

### Main Results (GLUE, RoBERTa-Large)

| Method | SST-2 | QNLI | MRPC | QQP | Avg. | Δ |
|--------|-------|------|------|-----|------|---|
| FedIT | 93.33 | 85.43 | 76.35 | 73.82 | 82.23 | - |
| FedDPA | 91.90 | 83.13 | 81.60 | 81.35 | 84.49 | +2.26 |
| FedSA | 91.97 | 82.70 | 82.08 | 81.65 | 84.60 | +2.37 |
| **FedLEASE** | **93.33** | **87.22** | **86.93** | **83.57** | **87.76** | **+5.53** |

### Ablation Study

| Configuration | Performance | Remarks |
|---------------|-------------|---------|
| Fixed $M$ (manually specified) | Inferior to adaptive | Silhouette-based $M$ selection is more effective |
| Uniform assignment (no clustering) | Significant drop | Clustering-based allocation is critical |
| Fixed top-$k$ (non-adaptive) | Inferior to top-$M$ | Optimal $k$ varies across clients |
| Without guaranteed own-expert participation | Performance drop | Own expert must always be engaged |

### Key Findings
- **B Matrix as Task Fingerprint**: After only a few training rounds, cosine similarity of B matrices accurately distinguishes clients with different tasks, outperforming A matrices and BA products while being more computationally economical.
- **Substantial Variance in Adaptive $k$**: Experiments show that the optimal $k$ per client ranges from 2 to 4 (when $M=4$), confirming that a fixed $k$ is inevitably suboptimal for a subset of clients.
- **Cross-Cluster Knowledge Sharing Is Beneficial**: Compared to IFCA+LoRA (clustering with inter-cluster isolation), FedLEASE's routing mechanism allows knowledge to flow across clusters.
- **No Communication Overhead**: Clients upload only their own expert and router, keeping communication costs on par with baselines.

## Highlights & Insights
- **B Matrix as a Task Similarity Proxy**: The observation that LoRA B encodes task-specific features while A encodes general features is transferable to other settings requiring task similarity estimation, such as curriculum learning and transfer learning.
- **Elegant Routing Space Expansion**: Expanding from $M$ to $2M-1$ dimensions, with the first $M$ outputs all pointing to the client's own expert, guarantees own-expert participation more elegantly than constrained optimization.
- **Data-Driven Expert Count Selection**: Using the silhouette coefficient to select the optimal $M$ eliminates the need for manual hyperparameter tuning, making the method fully data-driven.

## Limitations & Future Work
- **One-Time Clustering**: Clustering is performed only during initialization; inter-client task relationships may evolve throughout training.
- **Limitations of Silhouette Coefficient**: May yield suboptimal $M$ for non-convex cluster structures.
- **Scalability**: How to set $M_{max}$ when the number of clients is very large (e.g., hundreds) warrants further investigation.
- **Future Directions**: (1) Dynamic re-clustering (updating groupings every few rounds); (2) layer-wise adaptive allocation, as different layers may benefit from different numbers of experts.

## Related Work & Insights
- **vs. FedIT**: FedIT relies on a single shared LoRA, which is insufficient for heterogeneous tasks; FedLEASE employs multiple experts with clustering-based assignment.
- **vs. FedDPA**: FedDPA uses a binary global-plus-local structure with limited granularity; FedLEASE determines the number of experts in a data-driven manner.
- **vs. MoLoRA/LoRAMoE (centralized)**: Centralized LoRA-MoE does not face federated heterogeneity or communication constraints; FedLEASE addresses expert allocation and adaptive selection specifically within the federated setting.

## Rating
- Novelty: ⭐⭐⭐⭐ — B-matrix clustering combined with expanded routing space is creative and supported by complete theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both NLU and NLG settings with multiple baselines, thorough ablations, and three empirically motivated key observations.
- Writing Quality: ⭐⭐⭐⭐ — Clear logical flow from observations to method to experiments, with intuitive figures.
- Value: ⭐⭐⭐⭐ — Addresses practical pain points in federated LoRA fine-tuning with a plug-and-play method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[AAAI 2026\] FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA](../../AAAI2026/llm_safety/fedalt_federated_fine-tuning_through_adaptive_local_training_with_rest-of-world_.md)
- [\[ICLR 2026\] SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](../../ICLR2026/llm_safety/she-lora_selective_homomorphic_encryption_for_federated_tuning_with_heterogeneou.md)
- [\[NeurIPS 2025\] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](../../ICLR2026/llm_safety/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)

</div>

<!-- RELATED:END -->
