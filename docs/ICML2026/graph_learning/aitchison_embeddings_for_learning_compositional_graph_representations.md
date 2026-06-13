---
title: >-
  [Paper Note] Aitchison Embeddings for Learning Compositional Graph Representations
description: >-
  [ICML2026][Graph Learning][Aitchison Geometry] This paper proposes AICoG, which represents nodes as mixtures of latent archetypes on a simplex and learns graph embeddings using Aitchison geometry and ILR isometric coordi…
tags:
  - "ICML2026"
  - "Graph Learning"
  - "Aitchison Geometry"
  - "Graph Representation Learning"
  - "Compositional Data"
  - "ILR Coordinates"
  - "Interpretable Embeddings"
date: 2026-05-08
content_hash: d36b5576aa7a5a24
---

# Aitchison Embeddings for Learning Compositional Graph Representations

**Conference**: ICML2026  
**arXiv**: [2605.00716](https://arxiv.org/abs/2605.00716)  
**Code**: https://github.com/Nicknakis/AICoG  
**Area**: Graph Learning / Interpretable Representation Learning  
**Keywords**: Aitchison Geometry, Graph Representation Learning, Compositional Data, ILR Coordinates, Interpretable Embeddings  

## TL;DR
This paper proposes AICoG, which represents nodes as mixtures of latent archetypes on a simplex and learns graph embeddings using Aitchison geometry and ILR isometric coordinates. While maintaining the same expressive power as Euclidean latent distance models, it provides an endogenous interpretation of node role similarity based on relative proportion trade-offs.

## Background & Motivation

**Background**: Graph representation learning typically maps nodes into Euclidean vector spaces, using random walks, matrix factorization, GNNs, or latent distance models to preserve structural proximity. While effective for link prediction and node classification, the embedding dimensions often lack semantics, making distances and directions difficult to interpret directly.

**Limitations of Prior Work**: Many networks do not merely exhibit "similarity between adjacent nodes" but possess continuous, overlapping structural roles. Nodes may simultaneously hold varying proportions of multiple latent archetypes, such as bridges, content producers, or community cores in social networks. Traditional mixed-membership models can express role mixtures but usually assume roles are discrete, identifiable, and axis-aligned. Standard Euclidean embeddings are flexible but fail to explain what relative role changes a specific direction corresponds to.

**Key Challenge**: Graph embeddings need both predictive performance and the ability to explain why nodes are similar. Euclidean spaces are expressive but semantically weak; discrete role models are interpretable but overly rigid. Continuous overlapping roles are better described as "relative trade-offs between multiple archetype proportions" rather than single coordinate values.

**Goal**: The authors aim to construct a graph embedding framework that explicitly models node roles as compositions on a simplex, using Aitchison geometry—which is suited for compositional data—to define distances. This ensures similarity naturally corresponds to log-ratio trade-offs between archetype proportions.

**Key Insight**: The essence of compositional data is that "proportions are meaningful, while absolute scales are not." Aitchison geometry is the standard tool for handling such relative information. Furthermore, the ILR transformation can isometrically map the simplex to an unconstrained Euclidean space, balancing geometric semantics with optimization convenience.

**Core Idea**: Each node is represented as a compositional proportion $\mathbf{z}_i$ of latent archetypes. ILR coordinates $\mathbf{x}_i=\operatorname{ILR}(\mathbf{z}_i)$ are used to preserve Aitchison distances, and the graph structure is learned via a latent distance likelihood.

## Method

### Overall Architecture

AICoG starts with an undirected simple graph $\mathcal{G}=(V,E)$ and learns a $K$-dimensional composition $\mathbf{z}_i\in\Delta^{K-1}$ for each node. Here, each dimension is not a typical Euclidean coordinate but the relative contribution of a latent archetype factor; all components are positive and sum to 1. A node role is not a single archetype but a continuous mixture within the simplex.

To avoid optimization directly on the constrained simplex, the method employs the isometric log-ratio (ILR) transformation. Given an orthonormal basis $\mathbf{V}$ of the contrast space, the ILR coordinates are $\mathbf{x}_i=\log(\mathbf{z}_i)^\top\mathbf{V}\in\mathbb{R}^{K-1}$. Since the Aitchison distance $d_A(\mathbf{z}_i,\mathbf{z}_j)$ equals the Euclidean distance $\|\mathbf{x}_i-\mathbf{x}_j\|_2$ in the ILR space, standard optimization can be performed in Euclidean coordinates while retaining the log-ratio semantics of compositional proportions for interpretation.

The graph structure is learned through a Bernoulli latent distance model. For a node pair $(i,j)$, the model defines the log-odds $\eta_{ij}=-\|\mathbf{x}_i-\mathbf{x}_j\|_2+\gamma_i+\gamma_j$, where $\gamma_i$ captures node degree heterogeneity. The training objective is to maximize the Bernoulli log-likelihood of all edges and non-edges. To avoid $O(N^2)$ all-pairs computation, the non-edge terms are approximated via uniform subsampling, reducing the per-iteration complexity to $O(|E|)$.

### Key Designs

1. **Simplex Node Roles and Aitchison Distance**:
	- **Function**: Allows node embeddings to naturally express "mixtures of archetype proportions."
	- **Mechanism**: The role of node $i$ is $\mathbf{z}_i=(z_{i1},...,z_{iK})$, focusing only on the relative proportions of archetypes. Two nodes are similar not because of absolute coordinate values, but because their log-ratio trade-offs are similar. Aitchison geometry ensures the distance depends only on proportional relationships and is scale-invariant.
	- **Design Motivation**: Many graph roles are continuous and overlapping. Forcing them into discrete roles or Euclidean axes loses semantics. Compositional representations align better with the intuition that "nodes mix multiple roles simultaneously."

2. **ILR Isometric Coordinates and Learnable Basis**:
	- **Function**: Utilizes unconstrained Euclidean optimization while maintaining Aitchison geometry.
	- **Mechanism**: ILR projects $\log(\mathbf{z}_i)$ onto $K-1$ dimensions using a contrast basis $\mathbf{V}$. Different valid ILR bases differ only by an orthogonal transformation, leaving distances and likelihoods unchanged. The paper uses both a fixed Helmert basis and a learned basis, which can be further interpreted via varimax rotation for sparser balances.
	- **Design Motivation**: Direct interpretation of simplex components falls back into the "axis-aligned role" limitation. ILR balances explain log-ratio contrasts between groups of archetypes, which are better suited for continuous role spaces.

3. **Subcompositional Coherence and Component Constraint Analysis**:
	- **Function**: Supports meaningful dimension/component removal rather than arbitrary dropping of Euclidean coordinates.
	- **Mechanism**: After selecting a subset of archetypes $S$, the subcomposition $\mathbf{z}_i^{(S)}$ is obtained via re-closure. Its ILR distance is equivalent to an orthogonal projection of the original ILR difference vector. This allows removing components and evaluating the retention of node classification performance without retraining.
	- **Design Motivation**: Euclidean dimensions lack inherent semantics, making dropout difficult to explain. Compositional components are archetype proportions; removing and renormalizing them has clear geometric meaning, useful for analyzing which archetype groups drive predictions.

### Loss & Training

Node compositions are parameterized via unconstrained logits $\tilde{\mathbf{z}}_i$ and converted to $\mathbf{z}_i$ using row-wise softmax. Edge probabilities are derived from $\eta_{ij}=-\|\mathbf{x}_i-\mathbf{x}_j\|_2+\gamma_i+\gamma_j$ piped into a logistic Bernoulli likelihood. The full log-likelihood is $\sum_{i<j}[Y_{ij}\eta_{ij}-\log(1+\exp(\eta_{ij}))]$. The authors prove that the ILR-compositional latent distance model is capable of representing the same set of edge probability matrices as a standard Euclidean latent distance model in $\mathbb{R}^{K-1}$, meaning compositional constraints do not sacrifice expressivity.

In experiments, AICoG uses the Adam optimizer to minimize the Bernoulli negative log-likelihood with a learning rate of $10^{-2}$ over 5000 iterations. Dimensions are set to $D=K-1$, evaluated at $D\in\{8,16,32,64\}$. Datasets include Cora, Citeseer, LastFM, DBLP, AstroPh, GrQc, and HepTh. Baselines include Node2Vec, Role2Vec, NetMF, MMSBM, MNMF, SLIM-Raa, HM-LDM, and Simplex-Euclidean.

## Key Experimental Results

### Main Results

| Task / Dataset | Dim | Strong Baseline | AICoG (HB) | AICoG (LB) | Main Conclusion |
|--------------|------|-------------|------------|------------|----------|
| Link prediction AstroPh AUC-ROC | 64 | SLIM-Raa 0.969 | 0.976 | 0.976 | AICoG achieves SOTA |
| Link prediction GrQc AUC-ROC | 64 | SLIM-Raa 0.949 | 0.961 | 0.961 | Significantly outperforms traditional mixed-membership |
| Link prediction HepTh AUC-ROC | 64 | SLIM-Raa 0.920 | 0.929 | 0.928 | Aitchison geometry shows steady lead |
| Link prediction Cora AUC-ROC | 64 | HM-LDM 0.806 | 0.851 | 0.852 | Compositional geometry benefits citation graphs |
| Node classification Cora Micro-F1 | 64 | Node2Vec 0.814 / HM-LDM 0.814 | 0.831 | 0.833 | Interpretable model does not sacrifice classification |
| Node classification LastFM Micro-F1 | 64 | Node2Vec 0.865 | 0.870 | 0.870 | Comparable to or slightly better than the strongest Euclidean baseline |

### Ablation Study

| Analysis | Setting | Key Metric | Description |
|------|------|---------|------|
| Aitchison vs. Simplex Euclidean | Simplex-Euclidean | Cora AUC-ROC (64d) is only 0.709 vs AICoG's 0.851 | The key is Aitchison geometry, not just the simplex constraint |
| Synthetic Membership Recovery | AICoG vs MMSBM | ILR-continuous: $\ell_1$ 0.900 vs 1.452; JS 0.154 vs 0.356 | AICoG is better at recovering continuous/interior membership |
| Membership Interiority | Cora | AICoG entropy 1.064, near-corner 5.55%; MMSBM entropy 0.191, near-corner 78.95% | AICoG learns more overlapping and interior roles that are label-informative |
| Single Balance Interpretation | Cora learned ILR basis | Single balance ~0.40 probe accuracy, ANOVA $F\approx319$ | A single log-ratio contrast can capture significant label structure |
| Subcomposition Evaluation | Cora 64d, random removal | AICoG shows strongest retention under aggressive compression | Re-closure of components preserves semantically meaningful structures |

### Key Findings
- AICoG is very strong in link prediction, particularly on Cora, GrQc, and HepTh. Results for fixed Helmert basis and learned basis are nearly identical, supporting the argument of ILR basis orthogonal invariance.
- Simplex-Euclidean shows a significant performance drop, indicating that simply placing nodes on a simplex is insufficient. Standard Euclidean distances fail to capture the relative semantics of compositional data.
- In node classification, pure Euclidean methods remain strong, but AICoG meets or exceeds Node2Vec/Role2Vec on Cora and LastFM, proving that interpretable geometry does not require a predictive performance trade-off.
- Synthetic experiments show that MMSBM tends to push memberships toward near-discrete corners, whereas AICoG is better suited for continuous, overlapping role structures.

## Highlights & Insights
- The biggest highlight is shifting graph role interpretation from "what a coordinate axis represents" to "what the relative trade-off between archetypes is." This avoids the rigid requirement of mixed-membership models to identify discrete corners.
- The use of ILR is clever: it allows the optimization to look like a standard Euclidean latent distance model, while all distances can be translated back into log-ratio differences in Aitchison space.
- Theoretical guarantees of expressive equivalence reduce concerns about adopting compositional geometry. The method does not trade expressivity for interpretability; it changes the geometric semantics under the same latent distance capacity.
- Subcompositional analysis provides a more natural way of explanation than post-hoc attribution. Removing a group of archetypes, re-closing, and observing prediction retention is a theoretically sound operation in compositional data theory.

## Limitations & Future Work
- AICoG is best suited for graphs where node roles possess inherently compositional semantics; it may not outperform Euclidean embeddings if the graph structure is driven by local homophily or non-proportional factors.
- The paper primarily evaluates featureless graphs and unsupervised representation learning, without direct competition against modern attributed GNNs or end-to-end supervised models.
- The training protocol assumes the graph is connected or dominated by a large giant component. Extensions for many small components or disconnected graphs remain for future work.
- While the ILR basis does not affect distance, it affects how humans read balances. Learned bases and varimax help, but automated alignment of archetypes with domain knowledge is still needed.
- The likelihood model is still a pairwise distance-based Bernoulli graph model. Extensions to complex relationships like directed edges, heterogeneous edges, and dynamic graphs are required.

## Related Work & Insights
- **vs. Node2Vec / DeepWalk**: These learn Euclidean embeddings via random walks, which are predictive but semantically weak; AICoG's distances map directly to archetype proportion trade-offs.
- **vs. Role2Vec / GraphWave**: These focus on structural roles but output standard vectors; AICoG models roles as continuous compositions and defines similarity via Aitchison geometry.
- **vs. MMSBM / Mixed-membership SBM**: MMSBM provides membership but leans toward discrete and axis-based interpretations; AICoG allows continuous interior compositions and explains roles via geometry rather than unique coordinates.
- **vs. SLIM-Raa / HM-LDM**: These simplex latent distance baselines are expressive and close to AICoG in some cases; AICoG's advantage lies in adding principled compositional semantics to simplex representations via Aitchison/ILR.
- **vs. Post-hoc Graph Explainability**: Common GNN explainers explain specific predictions or subgraphs; AICoG embeds interpretability into the representation space itself, making distances, balances, and subcompositions inherently interpretable.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematically introducing Aitchison geometry to graph role embeddings is highly distinctive, especially with the proof of non-diminished expressivity.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Link prediction, classification, synthetic recovery, interiority, basis visualization, and subcomposition analysis are well-covered; lacks direct comparison with attributed GNNs.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation and solid geometric explanation; some mathematical details might have a high entry barrier for readers unfamiliar with compositional data.
- Value: ⭐⭐⭐⭐☆ Highly insightful for interpretable graph representation learning, particularly for network analysis where roles are naturally continuous and overlapping.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](../../ICLR2026/graph_learning/towards_improved_sentence_representations_using_token_graphs.md)
- [\[ICLR 2026\] Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](../../ICLR2026/graph_learning/structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)
- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](../../ACL2026/graph_learning/agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ICML 2026\] View Space: Representation Learning Across Arbitrary Graphs](view_space_learning_representation_across_arbitrary_graphs.md)

</div>

<!-- RELATED:END -->
