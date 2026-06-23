---
title: >-
  [Paper Note] Aitchison Embeddings for Learning Compositional Graph Representations
description: >-
  [ICML 2026][Graph Learning][Paper Note] This paper proposes AICoG, which represents nodes as mixtures of latent archetypes on a simplex and learns graph embeddings using Aitchison geometry and Isometric Log-Ratio (ILR) coordinates. While maintaining the same expressiveness as Euclidean latent distance models, it ensures that node role similarity has an endog
tags:
  - ICML 2026
  - Graph Learning
date: 2026-05-08
content_hash: e8dfde8aeccdfdb1
---
# Aitchison Embeddings for Learning Compositional Graph Representations

**Conference**: ICML2026  
**arXiv**: [2605.00716](https://arxiv.org/abs/2605.00716)  
**Code**: https://github.com/Nicknakis/AICoG  
**Area**: Graph Learning / Interpretable Representation Learning  
**Keywords**: Aitchison geometry, Graph representation learning, Compositional data, ILR coordinates, Interpretable embeddings  

## TL;DR
This paper proposes AICoG, which represents nodes as mixtures of latent archetypes on a simplex and learns graph embeddings using Aitchison geometry and Isometric Log-Ratio (ILR) coordinates. While maintaining the same expressiveness as Euclidean latent distance models, it ensures that node role similarity has an endogenous interpretation based on relative trade-offs of proportions.

## Background & Motivation

**Background**: Graph representation learning typically maps nodes into Euclidean vector spaces, using random walks, matrix factorization, GNNs, or latent distance models to maintain structural proximity. While effective for link prediction and node classification, the embedding dimensions often lack semantic meaning, making distances and directions difficult to interpret directly.

**Limitations of Prior Work**: Many networks do not just exhibit "adjacent node similarity" but involve continuous, overlapping structural roles. Nodes may possess varying proportions of multiple latent archetypes, such as bridges, content producers, or community cores in social networks. Traditional mixed-membership models can express role mixtures but usually assume roles are discrete, identifiable, and axis-aligned. Ordinary Euclidean embeddings, though flexible, cannot explain what relative role changes a particular direction corresponds to.

**Key Challenge**: Graph embeddings need to be both predictive and explain why nodes are similar. Euclidean space is expressive but semantically weak; discrete role models are interpretable but too rigid. Continuous overlapping roles behave more like "relative trade-offs between multiple archetype proportions" rather than individual coordinate values.

**Goal**: The authors aim to construct a graph embedding framework that explicitly models node roles as compositions on a simplex, using Aitchison geometry—which is suited for compositional data—to define distances, making similarity naturally correspond to log-ratio trade-offs between archetype proportions.

**Key Insight**: The core of compositional data is that "proportions are meaningful, while absolute scale is not." Aitchison geometry is the standard tool for handling such relative information. The ILR transformation can isometrically map a simplex to an unconstrained Euclidean space, thereby balancing geometric semantics with optimization convenience.

**Core Idea**: Each node is represented as a composition proportion $\mathbf{z}_i$ of latent archetypes. ILR coordinates $\mathbf{x}_i=\operatorname{ILR}(\mathbf{z}_i)$ are used to preserve Aitchison distances, and graph structures are learned using a latent distance likelihood.

## Method

### Overall Architecture

AICoG starts from an undirected simple graph $\mathcal{G}=(V,E)$ and learns a $K$-dimensional composition $\mathbf{z}_i\in\Delta^{K-1}$ for each node. Here, each dimension is not a standard Euclidean coordinate but the relative contribution of a latent archetype factor; all components are positive and sum to 1. Node roles are not a single archetype but a continuous mixture within the simplex.

To avoid direct optimization on the constrained simplex, the method employs the isometric log-ratio (ILR) transformation. Given an orthonormal basis $\mathbf{V}$ of the contrast space, the ILR coordinates are $\mathbf{x}_i=\log(\mathbf{z}_i)^\top\mathbf{V}\in\mathbb{R}^{K-1}$. The Aitchison distance $d_A(\mathbf{z}_i,\mathbf{z}_j)$ is equal to the Euclidean distance in the ILR space $\|\mathbf{x}_i-\mathbf{x}_j\|_2$, allowing for standard optimization in Euclidean coordinates while retaining interpretation within the log-ratio semantics of compositional proportions.

The graph structure is learned via a Bernoulli latent distance model. For a pair of nodes $(i,j)$, the model defines the log-odds as $\eta_{ij}=-\|\mathbf{x}_i-\mathbf{x}_j\|_2+\gamma_i+\gamma_j$, where $\gamma_i$ captures node degree heterogeneity. The training objective is to maximize the Bernoulli log-likelihood of all edges and non-edges. To avoid $O(N^2)$ all-pairs computation, non-edge terms are approximated via uniform subsampling, reducing the complexity per iteration to $O(|E|)$.

### Key Designs

**1. Simplex Node Roles and Aitchison Geometry: Defining similarity via relative proportions rather than absolute coordinates**

Ordinary Euclidean embeddings and discrete role models both have shortcomings: the former's axes lack inherent semantics, and the latter (e.g., MMSBM) assumes roles are discrete, identifiable, and axis-aligned, struggling with "nodes mixing multiple continuous overlapping roles." AICoG represents node $i$'s role directly as a composition $\mathbf{z}_i=(z_{i1},\dots,z_{iK})\in\Delta^{K-1}$ on a simplex, where each dimension is the relative contribution of a latent archetype—simplex vertices correspond to pure roles dominated by a single archetype, while interior points represent mixed roles. Critically, Aitchison geometry rather than Euclidean distance is used to compare these compositions. Absolute scale (like degree or activity) is often a distractor; two nodes may have vastly different interaction volumes but identical relative interaction patterns. Aitchison geometry looks only at proportions (log-ratios) and is invariant to total scaling. Two nodes are similar if and only if their log-ratio trade-offs between archetypes are similar. Thus, "similarity" naturally resides in the "relative role distribution" rather than "absolute interaction volume," matching the semantics of continuous overlapping roles.

**2. ILR Isometric Coordinates and Learnable Basis: Transforming simplex optimization into unconstrained Euclidean optimization**

Direct gradient optimization on a constrained simplex is awkward, and directly interpreting each simplex component reverts to the "axis = role" problem. The isometric log-ratio (ILR) transformation resolves this dilemma: given an orthonormal basis $\mathbf{V}$ of the contrast space, it maps compositions to coordinates $\mathbf{x}_i=\log(\mathbf{z}_i)^\top\mathbf{V}\in\mathbb{R}^{K-1}$ and ensures that the Aitchison distance exactly equals the Euclidean distance of ILR coordinates $d_A(\mathbf{z}_i,\mathbf{z}_j)=\|\mathbf{x}_i-\mathbf{x}_j\|_2$. Consequently, the model can be optimized using standard gradient methods in unconstrained $\mathbb{R}^{K-1}$, while geometric semantics remain in the compositional proportions. Since any two valid ILR bases differ only by an orthogonal transformation—leaving distances and likelihoods unchanged—interpretability is a property of the representation space itself rather than a specific coordinate set. The paper uses both domain-independent fixed Helmert bases and learned bases trained jointly with the embeddings (which can be varimax-rotated for sparser balances). Each balance is a log-ratio contrast of one group of archetypes against another, which fits the continuous role space better than looking at single dimensions.

**3. Subcompositional Coherence: Semantically removing archetype components**

Dimensions in Euclidean embeddings have no inherent semantics, so discarding them lacks a clear interpretation. In contrast, components in compositional data represent archetype proportions, and "removing a subset of archetypes" is a legitimate operation with clear geometric meaning. AICoG leverages the subcompositional coherence of Aitchison geometry: after selecting a subset of archetypes $S$, the corresponding components are re-normalized (re-closure) to obtain a subcomposition $\mathbf{z}_i^{(S)}$. The paper proves (Lemma 3.1) that the ILR distance of the subcomposition is exactly equal to the norm of the orthogonal projection of the original ILR difference vector onto the corresponding subspace. This means one can remove archetypes and evaluate node classification retention without retraining, allowing researchers to probe "which archetype groups truly influence predictions." Compared to post-hoc attribution for black-box models, this interpretation is endogenous to the representation space during modeling.

### Loss & Training

Node compositions are parameterized via unconstrained logits $\tilde{\mathbf{z}}_i$ and passed through a row-wise softmax to obtain $\mathbf{z}_i$. Edge probabilities are derived from $\eta_{ij}=-\|\mathbf{x}_i-\mathbf{x}_j\|_2+\gamma_i+\gamma_j$ via a logistic Bernoulli likelihood. The full log-likelihood is $\sum_{i<j}[Y_{ij}\eta_{ij}-\log(1+\exp(\eta_{ij}))]$. The authors prove that the ILR-compositional latent distance model is representatively equivalent to the standard Euclidean latent distance model in $\mathbb{R}^{K-1}$ regarding the set of expressible edge probability matrices; thus, compositional constraints do not sacrifice expressiveness.

In experiments, AICoG uses the Adam optimizer to minimize the Bernoulli negative log-likelihood with a learning rate of $10^{-2}$ for 5000 iterations. Dimensions $D=K-1$ are evaluated at $D\in\{8,16,32,64\}$. Datasets include Cora, Citeseer, LastFM, DBLP, AstroPh, GrQc, HepTh. Baselines include Node2Vec, Role2Vec, NetMF, MMSBM, MNMF, SLIM-Raa, HM-LDM, and Simplex-Euclidean.

## Key Experimental Results

### Main Results

| Task / Dataset | Dim | Strong Baseline | AICoG (HB) | AICoG (LB) | Main Conclusion |
|--------------|------|-------------|------------|------------|----------|
| Link prediction AstroPh AUC-ROC | 64 | SLIM-Raa 0.969 | 0.976 | 0.976 | AICoG achieves SOTA |
| Link prediction GrQc AUC-ROC | 64 | SLIM-Raa 0.949 | 0.961 | 0.961 | Significantly outperforms traditional mixed-membership |
| Link prediction HepTh AUC-ROC | 64 | SLIM-Raa 0.920 | 0.929 | 0.928 | Aitchison geometry shows stable lead |
| Link prediction Cora AUC-ROC | 64 | HM-LDM 0.806 | 0.851 | 0.852 | Compositional geometry yields clear gains for citation graphs |
| Node classification Cora Micro-F1 | 64 | Node2Vec 0.814 / HM-LDM 0.814 | 0.831 | 0.833 | Interpretable model does not sacrifice classification performance |
| Node classification LastFM Micro-F1 | 64 | Node2Vec 0.865 | 0.870 | 0.870 | Comparable to or slightly better than strongest Euclidean baseline |

### Ablation Study

| Analysis Item | Setting | Key Metric | Description |
|------|------|---------|------|
| Aitchison vs Simplex-Euclidean | Simplex-Euclidean | Cora AUC-ROC at 64D is only 0.709, while AICoG is ~0.851 | The key is not the simplex constraint itself, but Aitchison geometry for compositional data |
| Synthetic membership recovery | AICoG vs MMSBM | ILR-continuous: $\ell_1$ 0.900 vs 1.452, JS 0.154 vs 0.356 | AICoG is better at recovering continuous/interior membership |
| Membership interiority | Cora | AICoG entropy 1.064, near-corner 5.55%; MMSBM entropy 0.191, near-corner 78.95% | AICoG learns more overlapping and interior roles that are label-informative |
| Single balance interpretation | Cora learned ILR basis | Single balance ~0.40 probe accuracy, ANOVA $F\approx319$, MI $\approx0.44$ | A single log-ratio contrast can capture meaningful label structure |
| Subcomposition evaluation | Cora 64D, random component removal | AICoG shows strongest retention under aggressive compression | Re-closure preserves semantically meaningful geometric structure |

### Key Findings
- AICoG is very strong in link prediction; on datasets like Cora, GrQc, and HepTh, fixed Helmert basis and learned basis yields nearly identical results, supporting the argument for ILR basis orthogonal invariance.
- Simplex-Euclidean shows a significant performance drop, indicating that simply placing nodes on a simplex is insufficient. Comparing proportions with standard Euclidean distance fails to capture the relative semantics of compositional data.
- In node classification, pure Euclidean methods remain strong, but AICoG reaches or exceeds Node2Vec/Role2Vec on Cora and LastFM, proving that interpretable geometry does not necessarily come at the cost of predictive performance.
- Synthetic experiments show MMSBM tends to push membership toward near-discrete corner points, while AICoG is better suited for continuous, overlapping role structures.

## Highlights & Insights
- The biggest highlight is shifting the interpretation of graph roles from "what an axis identifies" to "what the relative trade-offs between multiple archetypes are." This avoids the limitation of mixed-membership models that must identify discrete roles.
- The use of ILR is clever: it makes model optimization look like a standard Euclidean latent distance model while allowing all distances to be translated back into log-ratio differences in Aitchison space.
- The theoretical guarantee of expressive equivalence reduces concerns about adopting compositional geometry. The method does not trade expressiveness for interpretability; it changes geometric semantics under the same latent distance expressiveness.
- Subcompositional analysis provides a more natural way of explanation than post-hoc attribution. Removing archetypes and re-closing to see performance retention is a valid operation within compositional data theory.

## Limitations & Future Work
- AICoG is best suited for graphs where node roles naturally possess compositional semantics; if the graph structure is driven primarily by local homophily or non-proportional factors, it may not outperform Euclidean embeddings.
- Evaluations are primarily conducted on featureless graphs and in unsupervised settings, without direct competition against modern attributed GNNs or end-to-end supervised models.
- The training protocol assumes the graph is connected or dominated by a large component. Extensions to graphs with many small components or disconnected graphs remain a future direction.
- While the ILR basis does not affect distance, it affects how humans read balances; learned bases and varimax rotation help, but the automatic alignment of archetypes with domain knowledge is still unresolved.
- The likelihood model is still a pairwise distance-based Bernoulli graph model; extensions for complex relations like directed edges, heterogeneous edges, and dynamic graphs are needed.

## Related Work & Insights
- **vs Node2Vec / DeepWalk**: These learn Euclidean embeddings via random walks, which are predictive but lack semantic dimensions; AICoG's distances map to archetype proportions.
- **vs Role2Vec / GraphWave**: These focus on structural roles but output standard vectors; AICoG models roles as continuous compositions and defines similarity via Aitchison geometry.
- **vs MMSBM / Mixed-Membership SBM**: MMSBM provides membership but leans toward discrete and axis-based interpretation; AICoG allows continuous interior compositions and explains roles through geometry rather than unique coordinates.
- **vs SLIM-Raa / HM-LDM**: These simplex latent distance baselines are expressive and close to AICoG on some datasets; Ours' advantage lies in adding principled compositional semantics to simplex representations via Aitchison/ILR.
- **vs Post-hoc graph explainability**: Common GNN explainers explain specific predictions or subgraphs; AICoG embeds interpretability into the representation space itself (distances, balances, and subcompositions).

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematic introduction of Aitchison geometry into graph role embeddings is distinctive, with theoretical proof of no loss in expressiveness.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Link prediction, node classification, synthetic recovery, interiority, basis visualization, and subcomposition analysis are comprehensive; lacks direct comparison with attributed GNNs.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation is clear with solid geometric explanations; some mathematical details might have a steep learning curve for readers unfamiliar with compositional data.
- Value: ⭐⭐⭐⭐☆ Highly insightful for interpretable graph representation learning, especially for network analysis scenarios where roles are continuous, overlapping, and naturally proportional.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](../../ICLR2026/graph_learning/towards_improved_sentence_representations_using_token_graphs.md)
- [\[ICLR 2026\] CORDS - Continuous Representations of Discrete Structures](../../ICLR2026/graph_learning/cords_-_continuous_representations_of_discrete_structures.md)
- [\[ACL 2025\] Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings](../../ACL2025/graph_learning/predicate-conditional_conformalized_answer_sets_for_knowledge_graph_embeddings.md)
- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](../../ACL2026/graph_learning/agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
