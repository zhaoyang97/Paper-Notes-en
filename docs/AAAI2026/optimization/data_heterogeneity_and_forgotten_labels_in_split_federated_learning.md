---
title: >-
  [Paper Note] Data Heterogeneity and Forgotten Labels in Split Federated Learning
description: >-
  [AAAI 2026][Optimization][Split Federated Learning] This paper systematically investigates catastrophic forgetting (CF) caused by data heterogeneity in Split Federated Learning — with particular focus on intra-round forg…
tags:
  - "AAAI 2026"
  - "Optimization"
  - "Split Federated Learning"
  - "Catastrophic Forgetting"
  - "Data Heterogeneity"
  - "Multi-head"
  - "Processing Order"
date: 2026-05-08
content_hash: 948d2bd5c7ca3a05
---

# Data Heterogeneity and Forgotten Labels in Split Federated Learning

**Conference**: AAAI 2026
**arXiv**: [2511.09736](https://arxiv.org/abs/2511.09736)  
**Code**: [GitHub](https://github.com/jtirana98/Hydra-CF-in-SFL)  
**Area**: Optimization
**Keywords**: Split Federated Learning, Catastrophic Forgetting, Data Heterogeneity, Multi-head, Processing Order

## TL;DR
This paper systematically investigates catastrophic forgetting (CF) caused by data heterogeneity in Split Federated Learning — with particular focus on intra-round forgetting induced by the server-side processing order — and proposes Hydra, a multi-head method that partitions and trains the final layers of part-2 in groups before aggregation, significantly reducing the performance gap (PG) across labels by up to 75.4%.

## Background & Motivation

### State of the Field

**Background**: Split Federated Learning (SFL/SplitFedv2) divides the model into part-1 (client-side) and part-2 (server-side). Part-1 is trained in parallel across clients and aggregated as in standard FL; part-2 is trained on the server by sequentially processing each client's activations.

The authors identify two forms of CF in SFL: (1) **inter-round forgetting** in part-1, which mirrors standard FL model drift caused by aggregation; and (2) **intra-round forgetting (intraCF)** in part-2, caused by sequential server-side processing — analogous to continual learning — where the model tends to retain knowledge of labels associated with the last-processed client while forgetting earlier ones. Experiments on CIFAR-10 under non-IID settings show that the accuracy of the last-processed labels far exceeds that of the first-processed ones, with overall accuracy dropping to only 43% (vs. 80% under IID).

Existing CF mitigation methods from CL and FL (e.g., EWC, Scaffold) cannot be directly applied to SFL due to incompatible assumptions: EWC assumes sequential data streams, while Scaffold requires access to the full model.

### Starting Point

**Goal**: How can intra-round catastrophic forgetting caused by sequential server-side processing in SFL be characterized and mitigated? How do the cut layer choice and processing order affect the degree of forgetting?

## Method

### Overall Architecture
Hydra introduces a multi-head architecture on the server side of SFL: part-2 is further divided into a shared part-2a and multiple heads (part-2b). Part-2a is updated sequentially (preserving the original SFL behavior), while each head is assigned to a group of clients with similar data distributions and updated in parallel, with aggregation at the end of each round.

### Key Designs
**Client Grouping**: Prior to training, each client transmits its label distribution vector of length $L$. A greedy algorithm assigns clients with similar label distributions to the same group. With $G$ groups (heads) in total, each group exhibits a more uniform data distribution, reducing gradient conflicts.

**Head Design**: Each head corresponds to the final few layers of part-2. During the forward pass, the server routes the output of part-2a to the corresponding head according to a client-to-group mapping, and reverses this routing during backpropagation. At the end of each round, all heads are aggregated into a single model via FedAvg — unlike continual learning approaches that retain separate heads at inference time, Hydra ultimately produces a single-head model.

**Processing Order Analysis**: The authors define two ordering strategies — cyclic order (processing clients by dominant label in a cyclic fashion) and random order — and find that:
- Cyclic order consistently outperforms random order, as structured processing yields greater stability.
- Smaller values of $\phi$ (i.e., fewer consecutive clients from the same class) lead to better performance, confirming the importance of diversity in the training sequence.

**Cut Layer Impact**: A shallow cut layer results in a larger part-2, amplifying intraCF; a deeper cut layer yields a larger part-1, leading to inter-round forgetting analogous to standard FL. Experiments confirm that intraCF is more harmful than the model drift incurred in part-1.

### Forgetting Metrics
- **Performance Gap (PG)**: $\mathcal{PG}(r) = \frac{1}{L}\sum_{l=1}^{L}\max_{k}|min(0, A_l^r - A_k^r)|$, measuring the accuracy disparity across labels.
- **Backward Transfer (BW)**: measuring the difference between peak accuracy and final accuracy.

## Key Experimental Results

### Main Results

| Method | Global Acc.↑ | PG↓ | BW↓ |
|--------|:---:|:---:|:---:|
| SFL (baseline) | 43 | 43.4 | 28 |
| SplitFedV1/FL | 55 | 15.7 | 14 |
| MergeSFL | 65 | 21 | 13 |
| MultiHead | 47 | 36.6 | 25 |
| **SFL+Hydra** | **66.5** | **13.4** | **9** |

(MobileNet, CIFAR-10, 80%-DL)

Hydra outcomes:
- PG reduced by up to **75.4%** (ResNet101); accuracy improved by up to **112.2%**.
- Effective across diverse data partitioning strategies including Sharding and Dirichlet ($\alpha$=0.1/0.3).
- Small heads (last 2 layers only) outperform larger heads, with memory overhead of only $G \times 6$MB.

## Highlights & Insights
- First systematic characterization of intra-round catastrophic forgetting in SFL, clearly distinguished from inter-round forgetting in FL.
- Exceptionally thorough experiments: 2 models × 4 datasets × multiple data partitioning schemes × multiple processing orders, with ≥10 repeated runs per configuration and median reporting.
- Hydra is elegantly designed, leveraging the natural split structure of SFL with zero additional overhead on the client side.
- Provides detailed empirical validation and comparison of CL theory applicability within the SFL setting.

## Limitations & Future Work
- Client grouping relies on sharing label distributions prior to training, which may be restricted in privacy-sensitive scenarios.
- The number of heads $G$ must be predefined; adaptive grouping strategies remain unexplored.
- Only classification tasks are considered; forgetting patterns in generative or regression settings may differ.
- No integration with momentum-based methods such as SMoFi is explored, though the two approaches may be complementary.

## Related Work & Insights
- **FL forgetting methods (EWC/Scaffold)**: Require a complete model or sequential data assumptions, rendering them incompatible with SFL.
- **SplitFedV1**: Avoids sequential processing by maintaining a separate model copy per client on the server, but is effectively equivalent to FL and does not exploit the split structure.
- **MergeSFL**: Merges features and adjusts batch size, achieving 65% accuracy but with PG remaining at 21; Hydra reduces PG to 13.4.
- **Traditional multi-head (CL)**: Retains multiple heads for separate inference; Hydra instead aggregates all heads into a single one, better suited for the SFL setting.

## Related Work & Insights
- The discovery of intraCF provides important guidance for SFL research: server-side processing strategy must not be overlooked, and random ordering may introduce instability.
- Hydra's multi-head aggregation paradigm can be extended to personalization in heterogeneous federated learning.
- Strong complementarity with SMoFi: SMoFi addresses momentum divergence in SFLv1, while Hydra addresses sequential forgetting in SFLv2; combining the two is a promising direction.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [\[AAAI 2026\] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)
- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](../../NeurIPS2025/optimization/efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)
- [\[ICML 2026\] Learning Locally, Revising Globally: Global Reviser for Federated Learning with Noisy Labels](../../ICML2026/optimization/learning_locally_revising_globally_global_reviser_for_federated_learning_with_no.md)
- [\[ICML 2026\] HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation](../../ICML2026/optimization/ho-sfl_hybrid-order_split_federated_learning_with_backprop-free_clients_and_dime.md)

</div>

<!-- RELATED:END -->
