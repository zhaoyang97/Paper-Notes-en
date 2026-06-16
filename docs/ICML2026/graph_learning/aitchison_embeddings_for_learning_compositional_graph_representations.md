---
title: >-
  [Paper Note] Aitchison Embeddings for Learning Compositional Graph Representations
description: >-
  [ICML 2026][Graph Learning][Paper Note] This paper proposes AICoG, which represents nodes as latent mixture archetypes on a simplex. It learns graph embeddings using Aitchison geometry and ILR isometric coordinates, maintaining expressive power equivalent to Euclidean latent distance models while providing endogenous interpretability of node role similarity
tags:
  - ICML 2026
  - Graph Learning
date: 2026-05-08
content_hash: 7464b7237e217a77
---
# Aitchison Embeddings for Learning Compositional Graph Representations

**Conference**: ICML2026  
**arXiv**: [2605.00716](https://arxiv.org/abs/2605.00716)  
**Code**: https://github.com/Nicknakis/AICoG  
**Area**: Graph Learning / Interpretable Representation Learning  
**Keywords**: Aitchison Geometry, Graph Representation Learning, Compositional Data, ILR Coordinates, Interpretable Embeddings  

## TL;DR
This paper proposes AICoG, which represents nodes as latent mixture archetypes on a simplex. It learns graph embeddings using Aitchison geometry and ILR isometric coordinates, maintaining expressive power equivalent to Euclidean latent distance models while providing endogenous interpretability of node role similarity based on relative proportion trade-offs.

## Background & Motivation

**Background**: Graph representation learning typically maps nodes to Euclidean vector spaces, maintaining structural proximity via random walks, matrix factorization, GNNs, or latent distance models. While effective for link prediction and node classification, these embedding dimensions often lack semantics, making distance and orientation difficult to interpret directly.

**Limitations of Prior Work**: Many networks do not just exhibit "similarity among neighbors" but possess continuous, overlapping structural roles. Nodes may hold proportions of multiple latent archetypes simultaneously, such as mixed roles like bridges, content producers, or community cores in social networks. Traditional mixed-membership models express role mixtures but often assume roles are discrete, identifiable, and axis-aligned. Standard Euclidean embeddings are flexible but cannot explain which relative role changes correspond to a specific direction.

**Key Challenge**: Graph embeddings need both predictive performance and the ability to explain node similarity. Euclidean spaces are expressive but semantically weak; discrete role models are interpretable but too rigid. Continuous overlapping roles resemble "relative trade-offs of multiple prototype proportions" rather than single coordinate values.

**Goal**: The authors aim to construct a graph embedding framework that explicitly models node roles as compositions on a simplex, defining distances through Aitchison geometry suitable for compositional data. This ensures similarity naturally corresponds to log-ratio trade-offs between prototype proportions.

**Key Insight**: The core of compositional data is that "proportions are meaningful, absolute scales are not." Aitchison geometry is the standard tool for processing such relative information. The ILR transformation isometrically maps the simplex to an unconstrained Euclidean space, balancing geometric semantics with optimization convenience.

**Core Idea**: Each node is represented as a composition proportion $\mathbf{z}_i$ of latent archetypes. The model uses ILR coordinates $\mathbf{x}_i=\operatorname{ILR}(\mathbf{z}_i)$ to maintain Aitchison distances and learns the graph structure using a latent distance likelihood.

## Method

### Overall Architecture

AICoG starts from an undirected simple graph $\mathcal{G}=(V,E)$ and learns a $K$-dimensional composition $\mathbf{z}_i\in\Delta^{K-1}$ for each node. Each dimension represents the relative contribution of a latent archetype factor; all components are positive and sum to 1. Node roles are defined as continuous mixtures within the simplex rather than single archetypes.

To avoid optimization directly on the constrained simplex, the method utilizes the isometric log-ratio (ILR) transformation. Given an orthogonal basis $\mathbf{V}$ of the contrast space, the ILR coordinates are $\mathbf{x}_i=\log(\mathbf{z}_i)^\top\mathbf{V}\in\mathbb{R}^{K-1}$. Since the Aitchison distance $d_A(\mathbf{z}_i,\mathbf{z}_j)$ equals the Euclidean distance $\|\mathbf{x}_i-\mathbf{x}_j\|_2$ in the ILR space, standard optimization can be performed in Euclidean coordinates while retaining the log-ratio semantics of compositional proportions for interpretation.

Graph structure is learned via a Bernoulli latent distance model. For a node pair $(i,j)$, the model defines the log-odds $\eta_{ij}=-\|\mathbf{x}_i-\mathbf{x}_j\|_2+\gamma_i+\gamma_j$, where $\gamma_i$ captures node degree heterogeneity. The training objective is to maximize the Bernoulli log-likelihood of all edges and non-edges. Non-edge terms are approximated via uniform subsampling to reduce complexity per iteration to $O(|E|)$.

### Key Designs

**1. Simplex Node Roles and Aitchison Geometry: Defining Similarity via Relative Proportions Instead of Absolute Coordinates**

Standard Euclidean embeddings and discrete role models have respective shortcomings: the former's axes lack inherent semantics, while the latter (e.g., MMSBM) assumes roles are discrete and axis-aligned, struggling with nodes mixing multiple continuous overlapping roles. AICoG represents node $i$'s role directly as a composition $\mathbf{z}_i=(z_{i1},\dots,z_{iK})\in\Delta^{K-1}$ on a simplex, where each dimension is the relative contribution of a latent archetype. Using Aitchison geometry—rather than Euclidean distance—to compare these compositions is critical. Since absolute scales (like degree or activity) are often distractors and two nodes might share identical relative interaction patterns despite different volumes, Aitchison geometry focuses only on proportions (log-ratios). Two nodes are similar if and only if their log-ratio trade-offs between archetypes are similar, aligning similarity with "relative role distribution."

**2. ILR Isometric Coordinates and Learnable Basis: Reforming Simplex Optimization as Unconstrained Euclidean Optimization**

Direct gradient optimization on a constrained simplex is efficient, and interpreting simplex components directly risks reviving the "coordinate=role" issue. The ILR transformation resolves this: given an orthogonal basis $\mathbf{V}$, compositions are mapped to coordinates $\mathbf{x}_i=\log(\mathbf{z}_i)^\top\mathbf{V}\in\mathbb{R}^{K-1}$, ensuring $d_A(\mathbf{z}_i,\mathbf{z}_j)=\|\mathbf{x}_i-\mathbf{x}_j\|_2$. Consequently, models can be optimized in unconstrained $\mathbb{R}^{K-1}$ while keeping geometric semantics within the compositional proportions. Since any two valid ILR bases differ only by an orthogonal transformation, interpretability is a property of the representation space itself. The paper utilizes both domain-independent fixed Helmert bases and learned bases trained jointly with the embeddings, which can be further refined with Varimax rotation for sparser balances.

**3. Subcompositional Coherence: Semantically Removing Archetype Components**

While Euclidean dimensions lack inherent semantics, components of compositional data represent archetype proportions, allowing the removal of a subset of prototypes as a geometrically valid operation. AICoG leverages the subcompositional coherence of Aitchison geometry. After selecting a subset of archetypes $S$, the corresponding components are re-normalized (re-closure) to obtain a subcomposition $\mathbf{z}_i^{(S)}$. The paper proves (Lemma 3.1) that their ILR distance equals the norm of the original ILR difference vector's orthogonal projection onto the corresponding subspace. This allows evaluating performance retention after removing archetypes without retraining, serving as an endogenous explanation within the representation space.

### Loss & Training

Node compositions are parameterized via unconstrained logits $\tilde{\mathbf{z}}_i$ and mapped to $\mathbf{z}_i$ using a row-wise softmax. Edge probabilities are derived from $\eta_{ij}=-\|\mathbf{x}_i-\mathbf{x}_j\|_2+\gamma_i+\gamma_j$ via a logistic Bernoulli likelihood. The full log-likelihood is $\sum_{i<j}[Y_{ij}\eta_{ij}-\log(1+\exp(\eta_{ij}))]$. The authors prove that the space of representable edge probability matrices for the ILR-compositional latent distance model is identical to that of an unconstrained Euclidean latent distance model in $\mathbb{R}^{K-1}$.

Training employs Adam to minimize the Bernoulli negative log-likelihood with a learning rate of $10^{-2}$ for 5000 iterations. Dimensions $D=K-1$ are evaluated for $D\in\{8,16,32,64\}$. Datasets include Cora, Citeseer, LastFM, DBLP, AstroPh, GrQc, and HepTh.

## Key Experimental Results

### Main Results

| Task / Dataset | Dimension | Strong baseline | AICoG (HB) | AICoG (LB) | Main Conclusion |
|--------------|------|-------------|------------|------------|----------|
| Link prediction AstroPh AUC-ROC | 64 | SLIM-Raa 0.969 | 0.976 | 0.976 | AICoG achieves optimal performance |
| Link prediction GrQc AUC-ROC | 64 | SLIM-Raa 0.949 | 0.961 | 0.961 | Significantly exceeds traditional mixed-membership |
| Link prediction HepTh AUC-ROC | 64 | SLIM-Raa 0.920 | 0.929 | 0.928 | Aitchison geometry maintains a steady lead |
| Link prediction Cora AUC-ROC | 64 | HM-LDM 0.806 | 0.851 | 0.852 | Compositional geometry benefits citation graphs |
| Node classification Cora Micro-F1 | 64 | Node2Vec 0.814 | 0.831 | 0.833 | Interpretable models preserve classification performance |
| Node classification LastFM Micro-F1 | 64 | Node2Vec 0.865 | 0.870 | 0.870 | Slightly superior to strong Euclidean baselines |

### Ablation Study

| Analysis Item | Setting | Key metric | Description |
|------|------|---------|------|
| Aitchison vs Simplex-Euclidean | Simplex-Euclidean | Cora AUC-ROC (64D) is only 0.709 vs AICoG ~0.851 | The benefit comes from Aitchison geometry, not just the simplex constraint |
| Synthetic membership recovery | AICoG vs MMSBM | ILR-continuous: $\ell_1$ 0.900 vs 1.452 | AICoG better recovers continuous/interior memberships |
| Membership interiority | Cora | AICoG entropy 1.064; MMSBM 0.191 | AICoG learns more overlapping, interior, and label-informative roles |
| Single balance interpretation | Cora learned ILR basis | Single balance probe accuracy ~0.40 | A single log-ratio contrast can capture partial label structure |
| Subcomposition evaluation | Cora (64D) | Aggressive component removal | AICoG shows superior performance retention under re-closure |

### Key Findings
- AICoG is highly effective for link prediction; the alignment between Helmert and learned bases suggests orthogonal invariance of the ILR basis.
- Simplex-Euclidean performance drops significantly, proving that a simplex representation without Aitchison geometry fails to capture compositional semantics.
- In node classification, AICoG meets or exceeds pure Euclidean methods (Node2Vec/Role2Vec) on several datasets.
- Synthetic experiments demonstrate that AICoG is better suited for continuous, overlapping role structures compared to MMSBM, which favors near-discrete corners.

## Highlights & Insights
- The shift from axis-based interpretation to trade-off-based interpretation between prototypes avoids the rigid identify-only-discrete-roles constraint of mixed-membership models.
- The use of ILR allows optimization to mirror standard Euclidean latent distance models while maintaining theoretical grounding in Aitchison space.
- Expressive equivalence ensures that interpretability is gained without compromising the model's ability to represent edge probability matrices.
- Subcompositional analysis provides a natural form of explainability rooted in the representation geometry rather than post-hoc attribution.

## Limitations & Future Work
- Highest utility is found in graphs with inherent compositional role semantics; performance may vary if local homophily dominates.
- Current evaluations focus on featureless graphs and unsupervised learning without direct competition against attributed GNNs.
- Training protocols assume strong connectivity; extensions for graphs with many small disjoint components are a future direction.
- While balances improve interpretability, automatic alignment between archetypes and specific domain knowledge remains an open problem.
- Scaling the likelihood model to directed, heterogeneous, or dynamic graphs requires further research.

## Related Work & Insights
- **vs Node2Vec / DeepWalk**: While Euclidean embeddings excel at prediction, AICoG provides distances directly interpretable as prototype log-ratio trade-offs.
- **vs Role2Vec / GraphWave**: AICoG models structural roles as continuous compositions within the simplex rather than generic vectors.
- **vs MMSBM / Mixed-membership SBM**: AICoG favors continuous interior compositions and geometric interpretations over discrete, identifiable coordinate axes.
- **vs SLIM-Raa / HM-LDM**: AICoG introduces a principled compositional semantic to simplex representations using Aitchison/ILR frameworks.
- **vs Post-hoc graph explainability**: Instead of explaining specific predictions, AICoG embeds interpretability into the representation space itself.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematic introduction of Aitchison geometry to graph role embeddings is highly distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Comprehensive across link prediction, classification, and synthetic recovery tasks.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation and robust geometric justification.
- Value: ⭐⭐⭐⭐☆ Significant for interpretable graph representation learning in overlapping role scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](../../ICLR2026/graph_learning/towards_improved_sentence_representations_using_token_graphs.md)
- [\[ACL 2025\] Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings](../../ACL2025/graph_learning/predicate-conditional_conformalized_answer_sets_for_knowledge_graph_embeddings.md)
- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](../../ACL2026/graph_learning/agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ACL 2026\] What Makes AI Research Replicable? Executable Knowledge Graphs as Scientific Knowledge Representations](../../ACL2026/graph_learning/what_makes_ai_research_replicable_executable_knowledge_graphs_as_scientific_know.md)

</div>

<!-- RELATED:END -->
