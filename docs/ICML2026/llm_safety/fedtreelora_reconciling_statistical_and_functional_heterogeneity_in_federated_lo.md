---
title: >-
  [Paper Note] FedTreeLoRA: Reconciling Statistical and Functional Heterogeneity in Federated LoRA Fine-Tuning
description: >-
  [ICML2026][LLM Safety][Federated Learning] Addressing the issue where existing methods treat "client statistical heterogeneity" and "LLM layer-wise functional heterogeneity" as isolated dimensions in Federated LoRA…
tags:
  - "ICML2026"
  - "LLM Safety"
  - "Federated Learning"
  - "LoRA"
  - "Personalized Fine-Tuning"
  - "Hierarchical Clustering"
  - "Heterogeneity"
date: 2026-05-08
content_hash: 1f3380f58a65d03a
---

# FedTreeLoRA: Reconciling Statistical and Functional Heterogeneity in Federated LoRA Fine-Tuning

**Conference**: ICML2026  
**arXiv**: [2603.13282](https://arxiv.org/abs/2603.13282)  
**Code**: To be confirmed  
**Area**: llm_safety (Federated Learning / Privacy-Preserving Fine-Tuning)  
**Keywords**: Federated Learning, LoRA, Personalized Fine-Tuning, Hierarchical Clustering, Heterogeneity  

## TL;DR
Addressing the issue where existing methods treat "client statistical heterogeneity" and "LLM layer-wise functional heterogeneity" as isolated dimensions in Federated LoRA, FedTreeLoRA utilizes a global hierarchical clustering tree with layer-wise adaptive depth search. This allows shallow layers to be shared while deep layers specialize, improving average metrics on GLUE and FLAN from 91.19 / 61.77 to 92.36 / 63.19 with minimal parameter overhead.

## Background & Motivation

**Background**: LoRA combined with Federated Learning (FL) has become the standard for privacy-preserving LLM fine-tuning. Research typically falls into two camps: training a single global LoRA (FedIT, SLoRA) or implementing personalization via dual-modules (FedDPA, FedALT) or client clustering (FedLEASE).

**Limitations of Prior Work**: Existing methods implicitly rely on a **Flat-Model Assumption**, treating LoRA as a monolithic block and assuming that the decision to "share or not" must be uniform across all layers.

**Key Challenge**: The authors identify two critical facts through pilot experiments: (1) **Vertical Heterogeneity**—aggregating only shallow layers is significantly better than aggregating deep layers, as deep layers handle semantic/task specialization and are highly sensitive to client data distribution shifts; (2) **Coupled Heterogeneity**—the "safe sharing depth" depends on client similarity. The more heterogeneous the data, the further the optimal sharing boundary shifts toward shallow layers. Consequently, the flat assumption is inherently suboptimal.

**Goal**: To design a mechanism that provides **layer-specific** decisions on sharing granularity while maintaining topological consistency across layers to avoid semantic discontinuity caused by repeated regrouping.

**Key Insight**: Model client relationships as a **global hierarchical tree**. The root represents full sharing, leaves represent full personalization, and each intermediate cut represents a grouping scheme. Each Transformer layer selects a cut on this tree (constrained to be monotonically deeper), ensuring consistency while allowing layer-wise adaptation.

**Core Idea**: Construct a global tree using **agglomerative hierarchical clustering (AHC) on client LoRA $B$ matrices**. For each Transformer layer, search for the optimal cluster count $c_l^*$ using the Silhouette coefficient within a search window that starts from the previous layer's granularity and expands by at most $K$ clusters. This couples horizontal and vertical heterogeneity into a unified framework.

## Method

### Overall Architecture
The federated system consists of $N$ clients, each with private data $\mathcal{D}_k$ and a shared frozen backbone $W_0$. Each client learns a set of personalized LoRA parameters $\boldsymbol{\Theta}_k$. The FedTreeLoRA pipeline consists of three steps: (1) **Warmup** via local fine-tuning for $E_{warm}$ rounds to obtain initial $B$ matrices; (2) **Global Topology Modeling** using the Frobenius distance of $B$ matrices to construct an $N\times N$ global distance matrix $D^{global}$, followed by AHC to generate a binary merge tree $\mathcal{T}$; (3) **Layer-wise Cut Selection on $\mathcal{T}$** from shallow to deep layers, using the Silhouette coefficient to select $c_l^*$ while enforcing $c_l \geq c_{l-1}$ for monotonic specialization; (4) Local training where clients construct **Cluster Expert** and **External Expert** modules, mixed via a learnable scalar $\lambda_{l,k}$. Local updates only affect the Cluster Expert and $\lambda$, while External Experts are frozen.

### Key Designs

1.  **Global Topology Tree + Distance Metric**:
    - **Function**: Utilizes a binary merge tree $\mathcal{T}$ to encompass all candidate client grouping schemes as a "backbone" for subsequent layer-wise cuts.
    - **Mechanism**: Warmup rounds generate layer-wise LoRA matrices $\{A_{l,k}, B_{l,k}\}$. The global distance between clients $i$ and $j$ is calculated as $D^{global}_{i,j} = \frac{1}{L}\sum_l \text{dist}(B_{l,i}, B_{l,j})$ (using the $B$ matrix as it encodes task-specific semantics). AHC then condenses $D^{global}$ into $\mathcal{T}$.
    - **Design Motivation**: Independent clustering per layer would lead to "topological drift" (e.g., groups $\{1,2\},\{3,4\} \to \{1,3\},\{2,4\}$), disrupting semantic continuity in the forward pass. The global tree ensures that clients separated in shallow layers remain specialized in deeper layers, making the specialized paths monotonic and interpretable.

2.  **Adaptive Layer-wise Depth Alignment**:
    - **Function**: Selects the optimal cluster count $c_l^*$ for each Transformer layer $l$, enabling coarse sharing in shallow layers and fine specialization in deep layers.
    - **Mechanism**: A layer-specific distance matrix $D^{(l)}_{i,j}$ is calculated. The search space is restricted to $\Omega_l = \{c \in \mathbb{Z} \mid c_{l-1}^* \leq c < \min(N, c_{l-1}^* + K)\}$, where $K$ controls the granularity increments. The scoring function $\phi(c; D^{(l)})$ uses a threshold $\tau$ for $c=1$ and the Silhouette coefficient for $c \geq 2$.
    - **Design Motivation**: Sharing depth should vary with data heterogeneity. The restricted search window ensures alignment with $\mathcal{T}$, while $\tau$ provides a prior bias toward global sharing when heterogeneity is low.

3.  **Cluster-External Expert Mixing**:
    - **Function**: Translates the grouping $P_{c_l^*}$ into actual forward-pass LoRA parameters, allowing clients to absorb peer-group consensus while retaining global knowledge.
    - **Mechanism**: For client $k$ at layer $l$, the Cluster Expert $\bar{\Phi}_{l,k}^{\text{clus}}$ is aggregated from its cluster $\mathcal{S}_k^{(l)}$, and the External Expert $\bar{\Phi}_{l,k}^{\text{ext}}$ from the remaining clients $\mathcal{R}_k^{(l)}$. The forward pass is defined as $h_l(x) = W_{0,l}x + \lambda_{l,k}(\bar{B}^{\text{clus}}\bar{A}^{\text{clus}}x) + (1-\lambda_{l,k})(\bar{B}^{\text{ext}}\bar{A}^{\text{ext}}x)$, where $\lambda_{l,k} \in [0,1]$ is a **learnable scalar**.
    - **Design Motivation**: Ablations show that topological alignment itself is the primary performance driver. Using scalars instead of complex MoE routers keeps additional trainable parameters at $\approx 0.020\%$ and minimizes communication costs while preventing information isolation.

### Loss & Training
Each client performs $E$ local SGD steps on the Cluster Expert $(\bar{A}^{\text{clus}}_{l,k}, \bar{B}^{\text{clus}}_{l,k})$ and $\lambda_{l,k}$, with the External Expert frozen. Under standard federated assumptions ($\sigma$-smoothness, bounded gradients, etc.), the authors prove a convergence rate of $\mathcal{O}(1/\sqrt{T})$, consistent with FedAvg, indicating that the tree structure does not impede training stability.

## Key Experimental Results

### Main Results

**NLU (RoBERTa-Large, 20 clients, Dirichlet $\alpha=0.5$, GLUE Average Accuracy, rank=4)**

| Method | % Param | MNLI | QNLI | SST2 | QQP | Average | $\Delta$ |
|--------|---------|------|------|------|-----|---------|----------|
| FedIT | 0.1107% | 83.18 | 87.03 | 93.65 | 84.93 | 87.20 | - |
| FedLEASE | 0.1521% | 86.21 | 92.56 | 95.63 | 90.36 | 91.19 | +3.99 |
| **Ours** | **0.1107%** | **88.15** | **93.37** | **96.56** | **91.35** | **92.36** | **+5.16** |

**NLG (LLaMA-2-7B 8-bit, 8 clients, FLAN Average ROUGE-1, rank=8)**

| Method | % Param | Text Edit | Struct2Text | Sentiment | Reasoning | Average | $\Delta$ |
|--------|---------|-----------|-------------|-----------|-----------|---------|----------|
| FedIT | 0.0622% | 59.84 | 51.71 | 44.53 | 74.42 | 57.62 | - |
| FedALT | 0.0699% | 67.61 | 54.06 | 48.57 | 76.84 | 61.77 | +4.15 |
| **Ours** | **0.0622%** | **68.63** | **55.59** | **51.27** | **77.27** | **63.19** | **+5.57** |

The results show that FedTreeLoRA achieves SOTA performance using the **minimum parameter budget**, even outperforming FedLEASE despite lower costs.

### Ablation Study

| Configuration | Avg. Acc | Description |
|---------------|----------|-------------|
| Fixed $k=1$ | 87.20 | Global sharing (equivalent to FedIT); underfits due to late heterogeneity. |
| Fixed $k=4$ | 91.45 | Fixed clusters; better but remains flat. |
| Layer-wise Adaptive $c_l^*$ | **92.36** | Full FedTreeLoRA. |
| Independent Clustering | 89.47 | Performance drops by 3% due to topological drift across layers. |
| Scalar-Mixed (Ours) | **92.36** | Parameter increase only +0.020%; highest cost-efficiency. |
| MoE Router | 92.02 | 25% more parameters with slightly lower performance. |

### Key Findings
- **Topological Alignment is Crucial**: Even a "Cluster-Only" variant without External Experts outperforms FedLEASE, suggesting that proper layer-wise grouping is the primary factor in resolving heterogeneity.
- **Global Tree Ensures Stability**: Removing the global tree leads to independent clustering drift, dropping accuracy to 89.47, which confirms that cross-layer topological consistency is essential.
- **Fixed Depth is Suboptimal**: Adaptive strategies outperform all fixed-cluster counts ($k=1, 4, 8$), validating that "one-size-fits-all" granularity harms either shallow sharing or deep specialization.

## Highlights & Insights
- **Deconstructing Heterogeneity**: The distinction between horizontal (data) and vertical (layer) heterogeneity, and the demonstration that "safe sharing depth" is a function of data similarity, provides a clear and novel motivation for the tree structure.
- **Elegant AHC + Monotonic Cut**: Using a global candidate tree with constrained cuts allows for layer-wise flexibility while preventing the "random jumping" of clusters between layers, ensuring semantic continuity.
- **Topology vs. Capacity**: The finding that expert routing/capacity is less important than correct topological grouping is a significant insight that could guide future federated fine-tuning designs.

## Limitations & Future Work
- The convergence rate is a standard $\mathcal{O}(1/\sqrt{T})$; the specific theoretical benefits provided by the tree structure specifically are not fully quantified.
- The warmup phase requires $E_{warm}$ local rounds, which may not be ideal for dynamic client participation; online tree updates are not discussed.
- The use of $B$ matrices for distance is based on task-specific priors; comparisons with other metrics like $A$ or $BA$ are limited.
- The choice of threshold $\tau$ and window $K$ are critical priors; though sensitivity analysis is mentioned, a universal guide for setting these remains for future work.

## Related Work & Insights
- **vs. FedLEASE**: While FedLEASE uses clustering, it is "flat" (same clusters for all layers). FedTreeLoRA introduces nested hierarchical groupings and layer-specific cuts.
- **vs. FedDPA/FedALT**: These dual-branch methods assume sharing decisions are uniform across the model. FedTreeLoRA treats "sharing depth" as an adaptive variable.
- **vs. FedPer**: FedTreeLoRA generalizes the "shallow-shared, deep-private" concept from CNNs to Transformer LoRA, specifically solving the problem of *where* and *how much* to split.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [\[ICLR 2026\] SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](../../ICLR2026/llm_safety/she-lora_selective_homomorphic_encryption_for_federated_tuning_with_heterogeneou.md)
- [\[ICML 2026\] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning](decoupled_training_with_local_reinforcement_fine-tuning_in_federated_learning.md)
- [\[AAAI 2026\] FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA](../../AAAI2026/llm_safety/fedalt_federated_fine-tuning_through_adaptive_local_training_with_rest-of-world_.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](../../ICLR2026/llm_safety/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)

</div>

<!-- RELATED:END -->
