---
title: >-
  [Paper Note] HarmonyGNNs: Harmonizing Heterophily and Homophily in GNNs via Self-Supervised Node Encoding
description: >-
  [ICLR 2026][Graph Learning][Graph Neural Networks] HarmonyGNNs achieves goal harmony via "Teacher-Student Prediction SSL (JEPA-style) + Node Difficulty-Driven Dynamic Masking" and representation harmony via "Linear/MLP Projection + Weighted GCN + Feature-level Self-Attention + Hierarchical Fusion," allowing a single unlabeled framework to achieve SOTA on both homophilic and heterophilic graphs simultaneously.
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "Graph Neural Networks"
  - "Self-Supervised Learning"
  - "Heterophilic Graphs"
  - "Homophilic Graphs"
  - "Teacher-Student Prediction Architecture"
  - "JEPA"
  - "Dynamic Masking"
date: 2026-05-08
content_hash: b52fa2f7dc3407fe
---

# HarmonyGNNs: Harmonizing Heterophily and Homophily in GNNs via Self-Supervised Node Encoding

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=mGxtoQY3GA](https://openreview.net/forum?id=mGxtoQY3GA)  
**Code**: Open-sourced (GitHub link provided in the paper)  
**Area**: Graph Self-Supervised Learning / Graph Representation Learning  
**Keywords**: Graph Neural Networks, Self-Supervised Learning, Heterophilic Graphs, Homophilic Graphs, Teacher-Student Prediction Architecture, JEPA, Dynamic Masking  

## TL;DR
HarmonyGNNs achieves goal harmony via "Teacher-Student Prediction SSL (JEPA-style) + Node Difficulty-Driven Dynamic Masking" and representation harmony via "Linear/MLP Projection + Weighted GCN + Feature-level Self-Attention + Hierarchical Fusion," allowing a single unlabeled framework to achieve SOTA on both homophilic and heterophilic graphs simultaneously.

## Background & Motivation
- **Background**: Graph Self-Supervised Learning (Graph SSL) has become mainstream in low-label scenarios by eliminating dependency on labels. Existing methods are divided into two categories: contrastive (DGI/GRACE/BGRL), which rely on complex negative sampling and data augmentation, and generative (GraphMAE), which directly reconstruct original features.
- **Limitations of Prior Work**: Real-world graphs often mix **homophily** (connected nodes share similar labels, e.g., Cora homophily ratio 0.81) and **heterophily** (connected nodes have different labels, e.g., Roman-Empire ratio 0.05), with varying intensities across datasets. Most Graph SSL methods perform poorly on heterophilic graphs, limiting their generalization—especially in SSL settings where the model must rely solely on structure and features without label guidance.
- **Key Challenge**: (1) **Goal level**: Contrastive methods use relative targets like InfoNCE, lacking a stable global reference, making it unclear which pattern should dominate in mixed graphs and preventing convergence to a unified latent space. Generative methods force reconstruction of original features, which provides contradictory supervision signals when heterophilic neighbors have dissimilar attributes. (2) **Representation level**: Homophilic regions require "smoothing" to capture similarity, while heterophilic regions require "differentiation" to preserve variance. Existing methods fail to adaptively balance these, often biasing toward a single pattern.
- **Goal**: To develop a single unified framework that automatically spans the full spectrum of homophily and heterophily without any prior knowledge of graph properties.
- **Core Idea**: **[Goal Harmony]** Use a teacher-student prediction architecture to provide stable global supervision anchors + node difficulty-driven dynamic masking to generate challenging and informative proxy tasks; **[Representation Harmony]** Encode nodes into a joint latent space that preserves both node specificity and structural awareness, using feature-level self-attention to adaptively blend the two patterns.

## Method

### Overall Architecture
HarmonyGNNs is an end-to-end JEPA-style graph SSL framework. The teacher network processes the complete graph, while the student network processes a partially masked graph. The student is trained to predict the teacher's embeddings for **all nodes** (both masked and unmasked). Teacher parameters are an Exponential Moving Average (EMA) of the student's parameters. Each node first undergoes "Joint Structural Node Encoding"—treating linear projection, MLP projection, and two-layer weighted GCN outputs as four tokens, processed through a feature-level Transformer block and then hierarchically fused into the final representation. The masking strategy switches from random masking during warm-up to an "Exploitation-Exploration" strategy that prioritizes masking the most difficult nodes based on difficulty scores.

```mermaid
flowchart TD
    A[Input Graph G] --> B[Teacher Network T<br/>EMA Update · Full Graph View]
    A --> C[Random Mask M] --> D[Masked Graph Ḡ]
    D --> E[Student Network S<br/>End-to-End Training]
    subgraph Joint Structural Node Encoding
        F[Linear Projection] & G2[MLP Projection] & H[1-hop WGCN] & I[K-hop WGCN] --> J[4-token Feature-level Transformer]
        J --> K[Hierarchical Fusion → S·v·]
    end
    E --> Joint Structural Node Encoding
    B --> L[Full Node Prediction Loss<br/>StopGrad]
    K --> L
    L --> M[Difficulty Score Diffi·v·] --> N[Dynamic Masking → Next Round M]
    N -.-> C
```

### Key Designs

**1. Teacher-Student Prediction for All Nodes: Using EMA anchors instead of relative targets to avoid collapse and negative sampling.** Inspired by the success of JEPA in computer vision, the authors use a teacher-student rather than an encoder-decoder setup. For masked nodes $v\in V_m$, original features are replaced by learnable parameters initialized from white noise $f(v)\sim\mathcal{N}(0,1)$. The student $S(\cdot;\Phi)$ sees the masked graph, while the teacher $T(\cdot;\Psi)$ sees the full graph. The teacher is not trained but is an EMA of the student: $\Psi_i=\alpha\Psi_{i-1}+(1-\alpha)\Phi_i$. Crucially, the loss is calculated over the **entire graph**, not just masked nodes—since graph nodes are highly interconnected, masking affects unmasked nodes, and predicting only masked nodes is insufficient to learn a meaningful latent space: $L(\Phi)=\frac{1}{N}\sum_{v\in V}\lVert S(v;\Phi)-T(v;\Psi)\rVert_2^2$. Predicting features in the latent space with stable EMA targets avoids negative sampling and is less likely to produce contradictory signals in heterophilic regions compared to direct feature reconstruction. Theoretically (Thm.1), the contraction factor $1-\mu_E^2/\beta_E^2$ of the teacher-student model is smaller than the $1-\min(\mu_E^2,\mu_D^2)/\max(\beta_E^2,\beta_D^2)$ of encoder-decoders, leading to faster and more stable convergence.

**2. Node Difficulty-Driven Dynamic Masking: Focusing the proxy task on "nodes the student finds hardest to learn."** Pure random masking is insufficient for graphs with unknown topology. After a random masking warm-up, nodes are selected using an exploitation rate $r$ to pick $m=\lfloor M\times r\rfloor$ difficult nodes, with the rest randomly sampled to maintain exploration. Difficulty scores are derived directly from the prediction loss: $\text{Diffi}(v)=\lVert S(v)-T(v)\rVert_2^2$. Two variants: **+Diffi** ranks nodes by difficulty and masks the top $m$, but risks over-focusing on a few hard nodes; **+Prob** uses Bernoulli sampling $M(v)\sim\text{Bernoulli}(p_v)$, where $p_v=p_0+\delta_v$. The base probability $p_0=(1-r)\times R$ is constant for all nodes, while the difficulty increment $\delta_v=\frac{\text{Diffi}(v)}{\text{Diffi}_{\max}}\times r\times R$ increases the masking probability for hard nodes, balancing exploration and exploitation. This objective harmonizes easy/hard samples with homophilic/heterophilic signals.

**3. Weighted GCN (WGCN): Adaptive switching between "smoothing" and "sharpening" via learnable edge weights.** Traditional GCN is a smoothing operator, benefiting homophily but harming heterophily. WGCN makes each edge weight learnable: $H^{(l+1)}=\sigma(A\cdot H^{(l)}\cdot W^{(l)})$, where $A_{ij}$ are learnable parameters initialized as normalized adjacency with self-loops $\tilde{A}=\tilde{D}^{-1/2}(A+I)\tilde{D}^{-1/2}$. A Sigmoid function $a_e=\sigma(w_e)$ constrains weights to $(0,1)$ to resolve global rescaling ambiguity. This allows high weights for similar neighbors in homophilic regions to preserve smoothing, and reduced weights in heterophilic regions to prevent over-smoothing—accommodating both structures within a single efficient operator.

**4. Feature-level Self-Attention + Hierarchical Fusion: Adaptive mixing of "four perspectives" as tokens to avoid $O(N^2)$ attention.** For each node, linear projection $f^{(\text{Linear})}$, MLP projection $f^{(\text{Mlp})}$ (preserving node specificity for heterophily), and two-layer WGCN outputs $H^{(\ell)},H^{(\ell')}$ (preserving structural awareness) are concatenated into $f(v)\in\mathbb{R}^{4\times C}$. Treating each projection as a token, a pre-norm vanilla Transformer performs **feature-level** (not node-level) attention. Complexity is only $O(Ns^2h+Nsh^2)$ (where $s=4$) rather than the $O(N^2h)$ of Graph Transformers, ensuring scalability. Fusion is hierarchical, merging from near to far: first merging the two WGCN tokens, then gradually incorporating the two projection tokens: $S(v)=\sigma(\text{Linear}(X_{0,C}\Vert\sigma(\text{Linear}(X_{1,C}\Vert\sigma(\text{Linear}(X_{2,C}\Vert X_{3,C}))))))$. This allows the model to learn "coarse-to-fine" weighting of homophilic/heterophilic patterns with fewer parameters and better gradient flow. The overall encoder complexity $O(Ndh+|E|h+Nh^2)$ is comparable to standard GNNs and strictly superior to node-level Graph Transformers.

## Key Experimental Results

### Main Results Table (Linear probing for node classification, Accuracy %)
Across 8 datasets (4 heterophilic + 4 homophilic), HarmonyGNNs leads significantly on heterophilic graphs and remains competitive with SOTA on homophilic graphs:

| Method | Cornell | Texas | Wisconsin | Actor | Cora | CiteSeer | PubMed | Arxiv |
|------|---------|-------|-----------|-------|------|----------|--------|-------|
| Hom. Ratio | 0.30 | 0.11 | 0.21 | 0.22 | 0.81 | 0.74 | 0.80 | 0.66 |
| GraphMAE | 61.93 | 67.80 | 58.25 | 31.48 | 84.20 | 73.20 | 81.10 | 71.75 |
| GraphACL | 59.33 | 71.08 | 69.22 | 30.03 | 84.20 | 73.63 | 82.02 | 71.72 |
| MUSE | 82.00 | 83.98 | 88.24 | 36.15 | 82.22 | 71.14 | 82.90 | 70.98 |
| GREET | 73.51 | 83.80 | 82.94 | 35.79 | 83.84 | 73.25 | 80.29 | 71.09 |
| **HarmonyGNNs+Diffi** | 85.41 | **93.24** | 92.74 | 37.93 | 84.70 | 73.36 | 83.42 | 71.56 |
| **HarmonyGNNs+Prob** | **85.68** | 92.45 | **93.13** | **38.15** | **84.82** | 73.12 | 83.25 | **71.97** |

The gains are even more pronounced on three difficult heterophilic graphs (e.g., Roman-Empire with a homophily ratio of only 0.05):

| Method | Chameleon(filtered) | Squirrel(filtered) | Roman-Empire |
|------|--------|--------|--------|
| MUSE | 46.48 | 41.57 | 66.26 |
| GREET | 44.67 | 39.69 | 63.37 |
| **HarmonyGNNs+Prob** | **48.91** | **45.49** | **75.86** |

Compared to the previous SOTA, improvements include +7.1% on Texas, +9.6% on Roman-Empire, and +1.27% on Actor.

### Ablation Study (Stepwise removal of three components)

| Configuration | Cornell | Texas | Wisconsin | Actor | Roman | Cora | Arxiv |
|------|---------|-------|-----------|-------|-------|------|-------|
| HarmonyGNNs (Full) | 85.68 | 92.45 | 93.13 | 38.15 | 75.86 | 84.70 | 71.56 |
| w/o DynMsk | 84.26 | 90.16 | 90.08 | 36.98 | 74.01 | 84.10 | 71.00 |
| w/o T-S & DynMsk | 81.78 | 85.59 | 88.56 | 35.86 | 72.87 | 83.10 | 70.02 |
| w/o T-S & DynMsk & Attn | 79.86 | 82.46 | 86.98 | 34.11 | 70.12 | 78.36 | 68.65 |

### Key Findings
- **All components are effective**: Removing dynamic masking (DynMsk) causes a drop of up to 3.05%. Further removing the teacher-student architecture (T-S) leads to additional declines. Removing self-attention (Attn) causes a sharp drop on the homophilic Cora dataset (84.70→78.36), indicating its criticality for homophilic patterns.
- **+Prob evaluates better than +Diffi**: The exploration-exploitation balance of Bernoulli sampling is more stable than purely selecting the hardest nodes.
- **Efficiency advantages**: On Actor/Roman/Arxiv, total training time is significantly shorter than MUSE and GREET (e.g., 476s vs 627s/739s on Arxiv). Memory usage is on par with GREET and lower than MUSE, as the latter requires alternating training for contrastive learning, whereas HarmonyGNN's unified latent space allows faster end-to-end convergence.
- **Consistency in clustering**: K-means clustering performance on Texas/Cornell is 11.26%/12.51% higher than MUSE, demonstrating that embedding quality is robust across downstream tasks.

## Highlights & Insights
- **Reconceptualizing the "homophily vs. heterophily" debate into two orthogonal dimensions: Goal Harmony and Representation Harmony**. This is the paper's clearest conceptual contribution: the former addresses "whom to trust" (stable EMA anchors), while the latter addresses "how to represent" (adaptive blending of node specificity and structural awareness).
- **Migrating JEPA (Vision SSL) to graphs** and modifying the loss from "predicting only masked nodes" to "predicting all nodes" is a precise adaptation to the strong interconnectivity of graphs.
- **Feature-level token attention** instead of node-level attention is a clever way to bypass the quadratic complexity of Graph Transformers. Treating "four encoding perspectives" as a sequence provides adaptive capabilities while maintaining linear complexity.
- **Reusing the prediction loss as a difficulty score** enables "hard node mining" with zero additional computational overhead, creating a natural closed loop.
- The intuitive argument that "teacher-student models converge faster and more stably than encoder-decoders" is supported by comprehensive convergence theory (Thm.1).

## Limitations & Future Work
- **High component complexity**: The framework involves teacher-student EMA, dual dynamic masking, WGCN, feature-level Transformers, and hierarchical fusion. This leads to many hyperparameters ($R, r$, warm-up epochs, $\alpha$, WGCN layers), making tuning non-trivial. The paper acknowledges that some baselines (like MUSE) are difficult to replicate.
- **Strong theoretical assumptions**: Thm.1 relies on assumptions like strong convexity, smoothness, and Lipschitz continuity, which may differ from real deep networks, making the analysis more "heuristic" than a strict guarantee.
- **Limited dataset scale**: Testing stopped at Ogbn-Arxiv, without validating scalability and stability on massive graphs with millions or tens of millions of nodes.
- **Marginal gains on homophilic graphs**: Performance on CiteSeer/PubMed is merely on par or slightly lower, suggesting "harmony" primarily benefits the heterophily side, as homophily performance may have already plateaued with existing methods.
- Future work could link dynamic masking with graph structure learning, introduce multi-scale structural encoding into tokens, and extend the framework to more downstream tasks like link prediction and graph classification.

## Related Work & Insights
- **Two major schools of Graph SSL**: Contrastive (DGI, GRACE, BGRL, MVGRL) and Generative (GraphMAE, NWR-GAE). HarmonyGNNs points out their fundamental flaws in mixed graphs: lack of stable global references and feature space mismatch, respectively.
- **Heterophilic Graph SSL**: Methods like GREET, MUSE, GraphACL, and DSSL have improved heterophily performance but struggle to maintain homophily performance simultaneously. HarmonyGNNs emphasizes a "unified framework across the spectrum."
- **JEPA / Predictive SSL**: I-JEPA (Assran et al., 2023) and BYOL-style EMA teacher-student setups are the direct inspirations for Goal Harmony. This paper marks their successful adaptation to the graph domain.
- **Insight**: (1) When facing "opposing patterns," rather than designing inductive biases for one side, it is better to create an adaptive joint latent space with stable supervision anchors; (2) Treating multiple encodings as tokens for feature-level attention is a general technique to add "adaptive fusion" to any GNN without exploding complexity; (3) Using training loss itself as a difficulty signal to drive curriculum masking is nearly cost-free.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of JEPA-style prediction for graphs, full-node prediction, feature-level token attention, and difficulty-driven masking is novel and well-targeted. While individual components have precedents, the integrated "Goal + Representation" perspective is clear.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covering 11 datasets across the homophily/heterophily spectrum, dual downstream tasks (probing + clustering), three-component ablation, and efficiency/memory comparisons. Theoretical analysis is included, though validation on ultra-large graphs is missing.
- **Writing Quality**: ⭐⭐⭐⭐ The logic from motivation to challenges, methods, theory, and experiments is smooth. Diagrams are clear, though the numerous components lead to high information density.
- **Value**: ⭐⭐⭐⭐ Provides a unified, efficient, and reproducible strong baseline for handling homophily and heterophily without labels, offering practical reference value to the graph representation learning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Self-Consistency Improves the Trustworthiness of Self-Interpretable GNNs](self-consistency_improves_the_trustworthiness_of_self-interpretable_gnns.md)
- [\[ICLR 2026\] Forest-Based Graph Learning for Semi-Supervised Node Classification](forest-based_graph_learning_for_semi-supervised_node_classification.md)
- [\[AAAI 2026\] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption](../../AAAI2026/graph_learning/feature-centric_unsupervised_node_representation_learning_without_homophily_assu.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](../../NeurIPS2025/graph_learning/self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[ICML 2025\] HGOT: Self-supervised Heterogeneous Graph Neural Network with Optimal Transport](../../ICML2025/graph_learning/hgot_self-supervised_heterogeneous_graph_neural_network_with_optimal_transport.md)

</div>

<!-- RELATED:END -->
