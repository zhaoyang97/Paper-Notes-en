---
title: >-
  [Paper Note] Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs
description: >-
  [ICML 2026][Graph Learning][SAT] This paper models CNF formulas as "clause–literal hypergraphs + clause interaction graphs" and decomposes variable-level representations into polarity-invariant and polarity-equivariant c…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "SAT"
  - "unsat core"
  - "hypergraph neural networks"
  - "polarity invariant–equivariant decomposition"
  - "consistency regularization"
date: 2026-05-08
content_hash: c074191007dfd81a
---

# Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs

**Conference**: ICML 2026  
**arXiv**: [2605.04819](https://arxiv.org/abs/2605.04819)  
**Code**: None  
**Area**: Graph Learning / Neuro-Symbolic Reasoning / SAT Solving  
**Keywords**: SAT, unsat core, hypergraph neural networks, polarity invariant–equivariant decomposition, consistency regularization

## TL;DR
This paper models CNF formulas as "clause–literal hypergraphs + clause interaction graphs" and decomposes variable-level representations into polarity-invariant and polarity-equivariant components. By training with polarity-reversal consistency regularization, it significantly improves the accuracy of unsat-core variable prediction.

## Background & Motivation
**Background**: GNN-based SAT learners (such as NeuroCore, NeuroSAT, and SATformer) typically encode CNF formulas as bipartite graphs or directed acyclic graphs. Nodes represent literals/variables and clauses, while edges encode the binary "literal appears in clause" relationship. These models learn embeddings to predict whether variables belong to the unsat core or backbone, or to provide variable activity priors for CDCL solvers.

**Limitations of Prior Work**: The authors identify two systemic limitations. First is "insufficient structural expressiveness": bipartite graphs only encode pairwise relations, whereas real clauses often contain multiple literals and exhibit high-order coupling through shared variables. Bipartite graphs can only capture these indirectly through deep GNN stacks, which leads to over-smoothing. Second is the "lack of polarity modeling": each variable $v_i$ possesses a pair of complementary literals $l_i, \neg l_i$. Existing methods either treat them as independent nodes or add an edge between them, but none explicitly enforce the intrinsic SAT properties of "shared source information" and "sign-flipping representations under polarity reversal."

**Key Challenge**: The contradiction between the need for high-order, polarity-aware expressiveness and the inductive bias of current graph structures, which only express binary relations and ignore the algebraic symmetry of variables. Circumnavigating this requires either deeper networks (causing over-smoothing) or manual heuristics, making systematization difficult.

**Goal**: (i) To find a graph representation that natively carries multi-literal clauses and multi-clause interactions; (ii) To explicitly model "polarity-invariant" and "polarity-equivariant" properties at the variable level; (iii) To inject polarity constraints into training via a "label-free" dual-view regularization.

**Key Insight**: The authors start from two mathematical observations: for any CNF formula, after flipping the polarity of all literals, its satisfiability and the set of unsat-core variables remain unchanged (Property 1), while the variable assignments are fully flipped (Property 2). This implies that the unsat-core prediction task is "invariant" to polarity reversal, which serves as a self-supervised training signal.

**Core Idea**: Construct the CNF as a hypergraph where "literals are nodes and clauses are hyperedges," combined with a clause–clause interaction graph for high-order propagation. Variable representations are decomposed into "invariant + equivariant" parts before being combined into positive/negative literal embeddings. By sharing parameters across the original and polarity-flipped formulas with a consistency loss, the model is forced to learn polarity symmetry.

## Method
The overall logic of paSAT is summarized as: "Capture high-order structure via hypergraphs, capture polarity symmetry via algebraic decomposition, and enforce constraints via dual-view consistency."

### Overall Architecture
Given a CNF formula $\phi$, the model first transforms it into a hypergraph $\mathcal{H}=(\mathcal{V}_H, \mathcal{E}_H)$: node $u_i$ corresponds to literal $l_i$, hyperedge $e_j$ corresponds to clause $c_j$, and the incidence matrix $\mathbf{H} \in \mathbb{R}^{2N \times M}$ has $\mathbf{H}_{ij}=1$ if and only if $l_i \in c_j$. Additionally, a "Clause Interaction Graph" $\mathcal{G}_C$ is constructed where nodes are clauses and edge weights $w^C_{ij}=|\mathcal{L}(c_i) \cap \mathcal{L}(c_j)| / |\mathcal{L}(c_i) \cup \mathcal{L}(c_j)|$ use the Jaccard similarity to measure clause overlap.

During training, two pipelines run simultaneously: the original formula $\phi$ and the polarity-flipped formula $\phi^{(flip)}$. Sharing all parameters, the model generates variable prediction distributions $\mathbf{s}$ and $\mathbf{s}^{(flip)}$. Optimization is performed using three losses: task loss, output consistency, and decomposition consistency. During inference, only the original formula is processed, and scores are fed to the CDCL solver as variable activity initializations (NeuroCore style, but executed once rather than periodically).

### Key Designs

1.  **High-Order Propagation via Hypergraph + Clause Interaction Graph**:
    *   **Function**: Simultaneously aggregates multi-way relations at both literal and clause levels, explicitly injecting high-order clause–clause dependencies into literal embeddings.
    *   **Mechanism**: In round $t$, hypergraph convolution is performed first as $\mathbf{M}^{(t)}_H = \mathbf{D}^{-1}\mathbf{H}\mathbf{B}^{-1}\mathbf{H}^\top \mathbf{L}^{(t)}\mathbf{W}^{(t)}$, aggregating literals to clauses and back. On the clause side, $\mathbf{C}^{(t)} = \mathbf{B}^{-1}\mathbf{H}^\top \mathbf{L}^{(t)}\mathbf{W}^{(t)}$ is fed into a GCN $\Delta\mathbf{C}^{(t)} = \mathbf{D}_C^{-1/2}\mathbf{A}_C\mathbf{D}_C^{-1/2}\mathbf{C}^{(t)}\mathbf{U}$. After a residual update $\mathbf{C}'^{(t)} = \mathbf{C}^{(t)} + \alpha\sigma(\Delta\mathbf{C}^{(t)})$, the refined clause information is back-propagated to literals $\mathbf{M}^{(t)} = \mathbf{D}^{-1}\mathbf{H}\mathbf{C}'^{(t)}$. Finally, literal representations are updated via $\mathbf{L}^{(t+1)} = f_{\text{update}}(\mathbf{L}^{(t)}, \mathbf{M}^{(t)}, \bar{\mathbf{L}}^{(t)})$ where $\bar{\mathbf{L}}^{(t)}$ represents complementary literals.
    *   **Design Motivation**: Pure hypergraphs only allow clauses to interact indirectly via shared literals, which lacks expressiveness for tasks like unsat-core detection that rely on multi-clause conflicts. Explicitly building a clause graph based on shared literals provides a shortcut for the model to learn clause-level constraints while naturally avoiding over-smoothing from excessive depth.

2.  **Polarity Invariant–Equivariant Variable Decomposition**:
    *   **Function**: Forces variable-level representations to reflect the SAT properties of "common source + opposite sign," preventing traditional concatenation-based aggregation from blurring polarity information.
    *   **Mechanism**: Each variable representation $\mathbf{v}_i^{(t)} \in \mathbb{R}^{2d}$ is split by two learnable mappings into an invariant component $\mathbf{v}_{i,\text{inv}}^{(t)}$ and an equivariant component $\mathbf{v}_{i,\text{eq}}^{(t)}$. Positive/negative literal embeddings are obtained via direct addition/subtraction: $\mathbf{l}_{x_i}^{(t)} = \mathbf{v}_{i,\text{inv}}^{(t)} + \mathbf{v}_{i,\text{eq}}^{(t)}$ and $\mathbf{l}_{\neg x_i}^{(t)} = \mathbf{v}_{i,\text{inv}}^{(t)} - \mathbf{v}_{i,\text{eq}}^{(t)}$. After hypergraph propagation, the components are recalculated as $\mathbf{v}_{\text{inv},i}^{(t+1)} = \frac{1}{2}(\mathbf{L}_{2i} + \mathbf{L}_{2i+1})$ and $\mathbf{v}_{\text{eq},i}^{(t+1)} = \frac{1}{2}(\mathbf{L}_{2i} - \mathbf{L}_{2i+1})$, and then recombined via MLPs. Finally, a linear head predicts the unsat-core probability $\mathbf{s} = g(f'_{\text{inv}}(\mathbf{V}_{\text{inv}}^{(T)}))$ using only the invariant component.
    *   **Design Motivation**: Since the unsat-core label is a "structural property" invariant to polarity flips, the prediction head should only focus on the invariant component. Using linear "+/-" combinations ensures that the negation operation acts as a group action on literal embeddings, embedding the symmetry inductive bias into the architecture itself.

3.  **Polarity-Reversal Consistency Regularization**:
    *   **Function**: Uses the polarity-flipped formula of the same CNF as a "dual view" to inject symmetry into training without requiring extra labels.
    *   **Mechanism**: For each $\phi$, $\phi^{(flip)}$ is constructed (flipping all literal signs but preserving structure). Sharing the network yields $\mathbf{s}$ and $\mathbf{s}^{(flip)}$. Output-level consistency $\mathcal{L}_{\text{cons}} = \frac{1}{|\mathcal{V}|}\|\mathbf{s} - \mathbf{s}^{(flip)}\|_2^2$ forces both to predict the same results. Decomposition-level consistency $\mathcal{L}_{\text{decomp}} = \frac{1}{|\mathcal{V}|}\sum_i \bigl[\|\mathbf{V}_{i,\text{inv}}^{(T)} - \mathbf{V}_{i,\text{inv}}^{(T)(flip)}\|_2^2 + \|\mathbf{V}_{i,\text{eq}}^{(T)} + \mathbf{V}_{i,\text{eq}}^{(T)(flip)}\|_2^2\bigr]$ explicitly constrains invariant components to be equal and equivariant components to have opposite signs.
    *   **Design Motivation**: Architecting "+/-" combinations does not guarantee that the learned components truly correspond to "invariant/equivariant" semantics. The decomposition-level consistency serves as a direct mathematical constraint in the loss function to force information separation.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{\text{core}} + \lambda_{\text{cons}}\mathcal{L}_{\text{cons}} + \lambda_{\text{decomp}}\mathcal{L}_{\text{decomp}}$. $\mathcal{L}_{\text{core}}$ follows the KL divergence format from NeuroCore $D_{\text{KL}}(\mathbf{p}^* \| \mathbf{p})$, where the target distribution uniformly assigns probability to unsat-core variables and zero to others. The entire network is trained end-to-end. When integrated with CDCL, the network runs once before the solver starts to provide variable activity scores; these are reset periodically during solving without re-inference.

## Key Experimental Results

### Main Results
Evaluation was conducted on SR (Random 3-SAT), CA (Community Structure), and PS (Pigeonhole) datasets across Easy/Medium/Hard difficulties using Precision, PR-AUC, and ROC-AUC.

| Dataset | Metric | GCN | NeuroCore | SATFormer | paSAT |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SR (Avg) | PR-AUC | 0.915 | 0.938 | 0.920 | **0.963** |
| SR (Avg) | ROC-AUC | 0.732 | 0.805 | 0.741 | **0.872** |
| CA (Avg) | PR-AUC | 0.215 | 0.333 | 0.246 | **Significantly Higher** |
| CA (Avg) | ROC-AUC | 0.518 | 0.605 | 0.527 | **Significantly Higher** |

On the most challenging CA dataset (sparse community structure, small unsat-core ratio, extreme class imbalance), paSAT shows significant improvements over NeuroCore in both PR-AUC and ROC-AUC. This indicates that high-order + polarity modeling provides greater gains in structurally complex scenarios. An 6–9 point increase in ROC-AUC on SR further demonstrates superior classification ranking stability.

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full paSAT | Optimal | Hypergraph + CIG + Polarity Decomposition + Consistency |
| w/o CIG | Performance Drop | High-order clause dependencies can only be learned indirectly |
| w/o Polarity Decomp | Performance Drop | Degenerates to "concat + MLP" aggregation, losing symmetry |
| w/o $\mathcal{L}_{\text{decomp}}$ | Performance Drop | Architecture exists, but lacks explicit supervision for invariant/equivariant semantics |
| w/o $\mathcal{L}_{\text{cons}}$ | Performance Drop | Lacks output-level dual-view supervision |

### Key Findings
*   The Clause Interaction Graph makes the largest contribution on the complex CA dataset, indicating that explicit modeling of clause–clause dependencies is the key bottleneck for high-order problems.
*   Architectural decomposition alone is insufficient; it must be paired with decomposition-level consistency loss to truly learn symmetric semantics, otherwise, the network may merge the two components.
*   Switching from "periodic calls" to "one-time initialization" when integrating with CDCL causes almost no performance drop while significantly saving GPU time, suggesting that paSAT’s prediction signals are strong enough to act as a powerful prior.

## Highlights & Insights
*   Translating the algebraic symmetry of SAT formulas into dual constraints of "architecture + loss" is an elegant way to inject inductive bias: the architecture uses $+/-$ to make flipping a group action, while the loss uses MSE to prevent information collapse.
*   Constructing a polarity-flipped dual view as "label-free self-supervision" is very similar to "equivariant augmentation" in contrastive learning. This idea can be transferred to any graph task with explicit group symmetry (e.g., mirror flips in molecular graphs, voltage reversals in circuit graphs).
*   The "hypergraph + interaction graph" structure serves as a reminder: when an object participates in high-order relations (clauses containing literals) and has its own binary relations (clauses sharing variables), forcing them into a single graph may confuse message passing. Splitting them into two task-specific graphs is cleaner.

## Limitations & Future Work
*   Experiments were only verified on the unsat-core task; it remains to be proven if polarity-reversal consistency is equally effective for tasks like backbone prediction or guidance for branching heuristics.
*   The number of edges in the clause interaction graph is $O(M^2)$, which may require further optimization (sparsity, sampling) for industrial-scale instances with millions of clauses.
*   The integration with CDCL only explores "initialization of activity," which is a weak coupling. The potential for periodic re-calls or dynamic use within the search tree was not explored.
*   While Property 1 (satisfiability invariance) is used, Property 2 (assignment reversal) is mentioned but not fully utilized for "assignment-dependent" tasks like backbone prediction. Future work could adapt the loss (using sign-reversal constraints).

## Related Work & Insights
*   **vs NeuroCore (Selsam & Bjørner 2019)**: This work adopts NeuroCore’s KL training objective and CDCL integration but replaces the bipartite graph with a hypergraph + CIG and explicitly models polarity. While NeuroCore learns polarity implicitly via LSTM-style complementary messages, paSAT hard-codes the constraint through architecture and loss, likely improving sample efficiency.
*   **vs SATFormer**: SATFormer uses Transformers for global attention between literals and clauses, theoretically capturing arbitrary-order interactions. However, paSAT significantly outperforms SATFormer on SR and CA, suggesting that inductive bias (especially polarity symmetry) is more important than pure architectural capacity.
*   **vs NeuroBack (Wang et al. 2024)**: NeuroBack also uses polarity-flipped formulas as dual views for backbone prediction but only supervises at the output level. paSAT extends this to "decomposition-level consistency" and introduces hypergraph structures, offering a more systematic approach to symmetry modeling.
*   **vs General HGNN Work**: The hypergraph convolution $\mathbf{D}^{-1}\mathbf{H}\mathbf{B}^{-1}\mathbf{H}^\top$ follows standard forms (Bai et al. 2021). The innovation lies in the combination of domain knowledge and task symmetry, suggesting that breakthroughs in GNNs for structured reasoning require deep dives into the algebraic structure of the task.

## Rating
*   Novelty: ⭐⭐⭐⭐ Polarity invariant–equivariant decomposition + dual-view consistency is a clear demonstration of "translating" SAT algebraic symmetry into a neural network.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Systematic comparison on SR/CA/PS datasets across multiple difficulties and metrics, with a full ablation study. Lacks end-to-end solving time comparisons after CDCL integration.
*   Writing Quality: ⭐⭐⭐⭐ Rigorous derivations and a core motivation based on mathematical properties. Figure 1 is intuitive, though some sections are heavily mathematical.
*   Value: ⭐⭐⭐⭐ Advances the SAT $\times$ GNN research line. The method is transferable to other structured prediction tasks with group symmetries, though industrial adoption barriers remain high.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)
- [\[ICML 2026\] View Space: Representation Learning Across Arbitrary Graphs](view_space_learning_representation_across_arbitrary_graphs.md)
- [\[AAAI 2026\] UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction](../../AAAI2026/graph_learning/unihr_hierarchical_representation_learning_for_unified_knowledge_graph_link_pred.md)
- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)
- [\[AAAI 2026\] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption](../../AAAI2026/graph_learning/feature-centric_unsupervised_node_representation_learning_without_homophily_assu.md)

</div>

<!-- RELATED:END -->
