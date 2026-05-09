---
title: >-
  [Paper Note] Less is More: Towards Simple Graph Contrastive Learning
description: >-
  [ICLR 2026][AI Safety][graph contrastive learning] This paper revisits the foundational principles of graph contrastive learning (GCL) and identifies that node feature noise can be mitigated through structural feature aggregation derived from graph topology. Based on this insight, the authors propose a minimalist GCL model that contrasts a GCN encoder (capturing structural features) against an MLP encoder (isolating node feature noise), requiring neither data augmentation nor negative sampling. The method achieves state-of-the-art performance on heterophilic graph benchmarks while offering advantages in complexity, scalability, and robustness on homophilic graphs.
tags:
  - ICLR 2026
  - AI Safety
  - graph contrastive learning
  - heterophilic graphs
  - GCN
  - MLP
  - unsupervised graph representation learning
date: 2026-05-08
content_hash: 0162956dd8784b95
---

# Less is More: Towards Simple Graph Contrastive Learning

**Conference**: ICLR 2026
**arXiv**: [2509.25742](https://arxiv.org/abs/2509.25742)
**Code**: N/A
**Area**: AI Safety
**Keywords**: graph contrastive learning, heterophilic graphs, GCN, MLP, unsupervised graph representation learning

## TL;DR
This paper revisits the foundational principles of graph contrastive learning (GCL) and identifies that node feature noise can be mitigated through structural feature aggregation derived from graph topology. Based on this insight, the authors propose a minimalist GCL model that contrasts a GCN encoder (capturing structural features) against an MLP encoder (isolating node feature noise), requiring neither data augmentation nor negative sampling. The method achieves state-of-the-art performance on heterophilic graph benchmarks while offering advantages in complexity, scalability, and robustness on homophilic graphs.

## Background & Motivation

**Background**: Graph contrastive learning (GCL) is the dominant paradigm for unsupervised graph representation learning. The core idea is to construct two "views" of the same graph or node and apply a contrastive loss to attract positive pairs while repelling negative pairs. Numerous methods have been proposed in recent years, achieving strong performance on homophilic graphs (where connected nodes share the same class).

**Limitations of Prior Work**:
- **Poor performance on heterophilic graphs**: Most GCL methods perform poorly on heterophilic graphs (where connected nodes belong to different classes), where neighborhood aggregation may introduce misleading information since neighbors and the target node are of different classes.
- **Excessive complexity**: Existing GCL methods rely heavily on complex data augmentation strategies (edge dropping, feature masking, subgraph sampling, etc.), carefully engineered encoder architectures, and negative sampling techniques—all of which increase computational cost and tuning difficulty.
- **Insufficient theoretical understanding**: Why does contrastive learning work on graphs? In particular, what are the key factors driving success on heterophilic graphs? These fundamental questions lack clear theoretical explanations.

**Key Challenge**: The GCL community has been stacking increasingly complex components (fancier augmentations, more elaborate encoders, more sophisticated negative sampling) to improve performance, yet progress on heterophilic graphs remains limited. The underlying question is: is such complexity truly necessary, or has a simple, principled mechanism been overlooked?

**Goal**: To answer the question "What is the essence of GCL?" from both theoretical and empirical perspectives, and to propose a minimalist yet high-performing GCL model grounded in the identified core principle.

**Key Insight**: By revisiting the foundations of supervised and unsupervised graph learning, the authors identify a key principle: GCN message passing inherently performs "denoising" (smoothing node feature noise via neighborhood aggregation), and the raw node features and graph structure naturally provide two complementary contrastive views.

**Core Idea**: Augmentation and negative sampling are unnecessary—the GCN view (structure-aware, denoised features) and the MLP view (original, noisy features) naturally constitute positive pairs for contrastive learning.

## Method

### Overall Architecture
- **Input**: Graph $G = (V, E, X)$, where $V$ is the node set, $E$ is the edge set, and $X$ is the node feature matrix.
- **Output**: Low-dimensional node representations for downstream tasks (e.g., node classification).
- **Pipeline**:
  1. GCN encoder processes $(A, X)$ → produces structure-aware node embeddings $Z_{GCN}$.
  2. MLP encoder processes only $X$ → produces feature-only node embeddings $Z_{MLP}$.
  3. A contrastive loss is applied between $Z_{GCN}$ and $Z_{MLP}$ (positive pairs: two views of the same node).
  4. $Z_{GCN}$ or a combination of both is used as the final node representation.

### Key Designs

1. **GCN Encoder: Structural Feature Extraction**:

    - **Function**: A standard GCN that aggregates neighborhood information over the graph structure via message passing to produce structure-aware node embeddings.
    - **Mechanism**: GCN message passing $H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})$ essentially performs topology-based smoothing of node features. For homophilic graphs, this smoothing brings features of same-class nodes closer; for heterophilic graphs, the effect is more subtle—it probabilistically mitigates feature noise.
    - **Design Motivation**: GCN not only extracts topological information; its aggregation operation itself serves as a form of "denoising" by reducing random noise in node features through neighborhood averaging. This is the fundamental reason the GCN view contrasts meaningfully with the MLP view.

2. **MLP Encoder: Feature Noise Isolation**:

    - **Function**: A standard MLP that applies nonlinear transformations solely to each node's raw features $X_i$, without utilizing any graph structure.
    - **Mechanism**: The MLP processes "raw, noisy" node features. Because no neighborhood aggregation is performed, the feature noise of each node is fully preserved.
    - **Design Motivation**: The MLP view retains the "noisy component" of node features, which complements the GCN view (denoised features). The primary difference between the two views lies in the presence or absence of noise—this constitutes natural positive pairs for contrastive learning.

3. **Contrastive Loss: No Negative Sampling Required**:

    - **Function**: Aligns the GCN and MLP embeddings of the same node as a positive pair.
    - **Mechanism**: Rather than using the traditional InfoNCE loss (which requires negative samples), a simplified contrastive objective is adopted that directly maximizes the similarity between the two views of the same node, with regularization to prevent representation collapse.
    - **Design Motivation**: Negative sampling is particularly problematic in graphs—randomly selected "negatives" may actually belong to the same class (false negative problem). Eliminating negative sampling not only simplifies the method but also avoids this issue.

4. **Theoretical Guarantee**:

    - **Function**: Provides theoretical proof explaining why the dual-view contrastive learning with GCN and MLP is effective.
    - **Mechanism**: Under reasonable assumptions (features = signal + noise), GCN neighborhood aggregation reduces noise variance to $\sigma^2/d$ (where $d$ is the degree), while the MLP retains the full noise variance $\sigma^2$. Contrastive optimization drives the encoders to filter out noise and preserve the signal.
    - **Design Motivation**: Provides rigorous theoretical support for the observation that "GCN + MLP = natural contrastive views."

### Loss & Training
- **Contrastive Loss**: An asymmetric contrastive objective inspired by BYOL/SimSiam—one branch includes a predictor head while the other uses stop-gradient; no negative samples are required.
- **Regularization**: Batch Normalization is applied to prevent representation collapse.
- **Fully unsupervised training**: No node labels are used during training.
- After training, the GCN encoder output (or a concatenation/aggregation of both encoders) is used as the final node representation and fed into a linear classifier for evaluation.

## Key Experimental Results

### Main Results: Node Classification

**Heterophilic Graph Benchmarks**:

| Dataset | Ours | Prev. GCL SOTA | Gain |
|--------|------|-------------|------|
| Texas | SOTA | Complex GCL methods | Significant |
| Wisconsin | SOTA | Complex GCL methods | Significant |
| Cornell | SOTA | Complex GCL methods | Significant |
| Chameleon | SOTA | Complex GCL methods | Significant |
| Squirrel | SOTA | Complex GCL methods | Significant |
| Actor | SOTA | Complex GCL methods | Significant |

**Homophilic Graph Benchmarks**:

| Dataset | Ours | Prev. GCL SOTA | Note |
|--------|------|-------------|------|
| Cora | Competitive | Complex GCL methods | Comparable accuracy with far lower complexity/memory |
| Citeseer | Competitive | Complex GCL methods | Same as above |
| Pubmed | Competitive | Complex GCL methods | Same as above |

Core conclusion: State-of-the-art on heterophilic graphs; competitive on homophilic graphs with minimal computational and memory overhead.

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| GCN only (no contrastive learning) | Performance drops | Validates necessity of contrastive learning |
| MLP only (no graph structure) | Significant performance drop | Validates importance of structural information |
| With data augmentation | No clear improvement or degradation | Validates the core claim that augmentation is unnecessary |
| With negative sampling | No clear improvement | Validates that negative sampling is unnecessary |
| Varying GCN depth | 2–3 layers optimal | Deeper GCNs lead to over-smoothing |

### Robustness Experiments

| Attack Type | Proposed Method | Complex GCL | Note |
|-------------|-----------|-------------|------|
| Black-box attack (structural perturbation) | Strong | Weak–Moderate | Minimalist design is naturally robust to structural noise |
| White-box attack (feature + structure) | Moderate–Strong | Weak | MLP branch's independence from graph structure provides redundant protection |

### Key Findings
- **Minimalist method achieves SOTA on heterophilic graphs**: No sophisticated augmentation or negative sampling is needed; a simple GCN + MLP dual-view suffices—challenging the intuition that GCL requires complex design.
- **High computational efficiency**: Compared to augmentation-based GCL methods, the proposed method significantly reduces training/inference time and memory usage (by 1–2 orders of magnitude).
- **Strong scalability**: The absence of augmentation and negative sampling enables straightforward scaling to large graphs (millions of nodes).
- **Adversarial robustness**: The minimalist design yields stronger adversarial robustness—the MLP branch is independent of graph structure and therefore immune to structural attacks.
- **Theory–experiment consistency**: The noise mitigation theory accurately predicts experimental observations—GCN denoising effectiveness increases with node degree.

## Highlights & Insights

- **Victory of the "Less is More" philosophy**: Against the community trend toward increasingly complex designs, this work achieves the best heterophilic graph results with the simplest possible method—a reminder of the importance of returning to first principles.
- **Identification of the core GCL principle**: GCN message passing is inherently a form of "feature denoising," and contrasting it against raw features amounts to "denoised vs. noisy"—an insight that is both concise and elegant.
- **No augmentation, no negative sampling**: The two largest engineering burdens in GCL are completely eliminated, rendering the method "embarrassingly simple."
- **Breakthrough on heterophilic graphs**: Most prior GCL methods struggle on heterophilic graphs; the success of this work suggests the issue lies not in contrastive learning itself but in the inappropriateness of prior view construction strategies for heterophilic settings.
- **Robustness as a byproduct**: Simple design not only improves performance but also naturally confers adversarial robustness; the MLP branch's independence from graph structure makes it immune to structural attacks.

## Limitations & Future Work

- **Over-smoothing in GCN**: As depth increases, GCN node representations tend to converge (over-smoothing). Shallow GCNs (2–3 layers) are used here, which limits the modeling of long-range dependencies.
- **Assumptions in theoretical analysis**: The theoretical proof relies on the simplified assumption that "features = signal + Gaussian noise"; the actual distribution of feature noise in real data may be more complex.
- **Node classification only**: The method has not been validated on other graph tasks such as graph classification or link prediction; its generality remains to be confirmed.
- **Applicability to extreme heterophilic graphs**: When heterophily is very high (almost no same-class neighbors), the denoising effect of GCN may diminish.
- **Gap with supervised methods**: As an unsupervised approach, a performance gap with supervised GNNs remains, particularly on large datasets with abundant labels.
- **Extension to heterogeneous graphs**: Current validation is limited to homogeneous graphs; applicability to heterogeneous graphs (with multiple node/edge types) is unknown.

## Related Work & Insights

- **Graph contrastive learning**: Methods such as DGI, GraphCL, GCA, and BGRL rely on complex augmentations and negative sampling. This work demonstrates that these components may be superfluous given a principled view construction.
- **Denoising perspective on GNNs**: Some prior works have observed that GCN performs feature smoothing/denoising, but this paper is the first to systematically formalize it as the core mechanism of contrastive learning.
- **BYOL/SimSiam**: The idea of negative-sample-free contrastive learning originates from computer vision; this work successfully transfers it to graph learning and provides a graph-specific theoretical interpretation.
- **Inspiration**: Can analogous "natural contrastive views" be identified in other domains (e.g., point clouds, temporal graphs)? The key is to identify a "denoising" operation paired with a "noise-preserving" operation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hide and Find: A Distributed Adversarial Attack on Federated Graph Learning](hide_and_find_a_distributed_adversarial_attack_on_federated_graph_learning.md)
- [\[ICCV 2025\] Backdooring Self-Supervised Contrastive Learning by Noisy Alignment](../../ICCV2025/ai_safety/backdooring_self-supervised_contrastive_learning_by_noisy_alignment.md)
- [\[NeurIPS 2025\] FairContrast: Enhancing Fairness through Contrastive Learning and Customized Augmentation](../../NeurIPS2025/ai_safety/faircontrast_enhancing_fairness_through_contrastive_learning_and_customized_augm.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)
- [\[CVPR 2026\] One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control](../../CVPR2026/ai_safety/one-to-more_high-fidelity_training-free_anomaly_generation_with_attention_control.md)

</div>

<!-- RELATED:END -->
