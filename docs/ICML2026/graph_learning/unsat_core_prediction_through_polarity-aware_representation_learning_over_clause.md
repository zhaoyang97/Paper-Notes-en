---
title: >-
  [Paper Note] Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs
description: >-
  [ICML 2026][Graph Learning][SAT] This work models CNF formulas as a "clause–literal hypergraph + clause association graph," decomposes variable representations into polarity-invariant and polarity-equivariant components…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "SAT"
  - "unsat core"
  - "hypergraph neural network"
  - "polarity-invariant–equivariant decomposition"
  - "consistency regularization"
date: 2026-05-08
content_hash: fb6973bd43cb9a1b
---

# Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs

**Conference**: ICML 2026  
**arXiv**: [2605.04819](https://arxiv.org/abs/2605.04819)  
**Code**: None  
**Area**: Graph Learning / Neuro-symbolic Reasoning / SAT Solving  
**Keywords**: SAT, unsat core, hypergraph neural network, polarity-invariant–equivariant decomposition, consistency regularization

## TL;DR
This work models CNF formulas as a "clause–literal hypergraph + clause association graph," decomposes variable representations into polarity-invariant and polarity-equivariant components at the variable level, and trains with polarity-flip consistency regularization, significantly boosting unsat-core variable prediction accuracy.

## Background & Motivation
**Background**: GNN-based SAT learners (NeuroCore, NeuroSAT, SATformer, etc.) typically encode CNF formulas as bipartite or DAGs, where nodes are literals/variables and clauses, and edges represent the binary relation "literal appears in clause." An embedding is learned to predict whether a variable belongs to the unsat core, is a backbone, or to inject variable activity priors into CDCL solvers.

**Limitations of Prior Work**: The authors identify two systematic limitations. First, "insufficient structural expressiveness": bipartite graphs only encode pairwise relations, while real clauses often contain multiple literals and clauses are coupled via shared variables, which bipartite graphs can only capture indirectly with deep GNNs, leading to oversmoothing. Second, "lack of polarity modeling": each variable $v_i$ has a pair of complementary literals $l_i, \neg l_i$; existing methods either treat them as independent nodes with MLP aggregation or add an edge between complementary literals, but do not explicitly enforce "shared information from the same source" and "representation sign flip under polarity reversal," both intrinsic SAT properties.

**Key Challenge**: The contradiction between expressiveness (requiring higher-order and polarity-aware modeling) and inductive bias (existing graph structures only encode binary relations and ignore algebraic symmetry of variables). Overcoming this either requires deeper networks (worsening oversmoothing) or manual heuristics (edges/labels), which are hard to systematize.

**Goal**: (i) Find a graph representation that natively supports multi-literal clauses and multi-clause interactions; (ii) Explicitly model "polarity-invariant" and "polarity-flip" properties at the variable level; (iii) Inject polarity constraints into training via a "label-free" dual-view regularization.

**Key Insight**: The authors start from two mathematical observations—flipping all literal polarities in a CNF formula leaves its satisfiability and unsat-core variable set unchanged (Property 1); variable assignments are globally flipped (Property 2). This means the unsat-core prediction task is "invariant" to polarity flips, which can serve as a self-supervised training signal.

**Core Idea**: Construct the CNF as a hypergraph with literals as nodes and clauses as hyperedges, plus a clause–clause association graph for higher-order propagation; decompose variable representations into "invariant + equivariant" parts, then assemble positive/negative literal embeddings, and enforce polarity symmetry by sharing parameters and consistency loss between the original and polarity-flipped formulas.

## Method

The overall logic of paSAT can be summarized as: "first use hypergraphs to capture higher-order structure, then algebraic decomposition for polarity symmetry, and finally strong dual-view consistency constraints."

### Overall Architecture

Given a CNF formula $\phi$, the model first converts it into a hypergraph $\mathcal{H}=(\mathcal{V}_H,\mathcal{E}_H)$: node $u_i$ corresponds to literal $l_i$, hyperedge $e_j$ to clause $c_j$, with incidence matrix $\mathbf{H}\in\mathbb{R}^{2N\times M}$, where $\mathbf{H}_{ij}=1$ iff $l_i\in c_j$. Additionally, a "clause association graph" $\mathcal{G}_C$ is constructed, where nodes are clauses and edge weights $w^C_{ij}=|\mathcal{L}(c_i)\cap \mathcal{L}(c_j)|/|\mathcal{L}(c_i)\cup \mathcal{L}(c_j)|$ measure clause overlap via Jaccard similarity.

During training, two pipelines are run in parallel: the original formula $\phi$ and its polarity-flipped version $\phi^{(flip)}$, sharing all parameters and producing variable prediction distributions $\mathbf{s},\mathbf{s}^{(flip)}$. Three losses (task loss + output consistency + decomposition consistency) are jointly optimized. At inference, only the original formula is used, and the scores are fed to the CDCL solver as variable activity initialization (NeuroCore style, but run only once, saving GPU time).

### Key Designs

1. **Higher-order Propagation via Hypergraph + Clause Association Graph**:

    - **Function**: Aggregates multi-way relations on both literal and clause sides, explicitly injecting higher-order clause–clause dependencies into literal embeddings.
    - **Mechanism**: At iteration $t$, first perform hypergraph convolution $\mathbf{M}^{(t)}_H=\mathbf{D}^{-1}\mathbf{H}\mathbf{B}^{-1}\mathbf{H}^\top \mathbf{L}^{(t)}\mathbf{W}^{(t)}$, aggregating literals to clauses and back; then, on the clause side, feed $\mathbf{C}^{(t)}=\mathbf{B}^{-1}\mathbf{H}^\top \mathbf{L}^{(t)}\mathbf{W}^{(t)}$ into a GCN $\Delta\mathbf{C}^{(t)}=\mathbf{D}_C^{-1/2}\mathbf{A}_C\mathbf{D}_C^{-1/2}\mathbf{C}^{(t)}\mathbf{U}$, update with residual $\mathbf{C}'^{(t)}=\mathbf{C}^{(t)}+\alpha\sigma(\Delta\mathbf{C}^{(t)})$, then propagate refined clause information back to literals $\mathbf{M}^{(t)}=\mathbf{D}^{-1}\mathbf{H}\mathbf{C}'^{(t)}$, and finally aggregate complementary literal representations via $\mathbf{L}^{(t+1)}=f_{\mathrm{update}}(\mathbf{L}^{(t)},\mathbf{M}^{(t)},\bar{\mathbf{L}}^{(t)})$.
    - **Design Motivation**: Pure hypergraphs only allow clauses to interact indirectly via shared literals, which is insufficient for tasks like unsat-core prediction that depend on multi-clause joint conflicts. Explicitly building a clause graph provides a shortcut for learning clause-level mutual constraints and naturally avoids oversmoothing from deep stacking.

2. **Polarity-Invariant–Equivariant Variable Decomposition**:

    - **Function**: Enforces representations at the variable level to reflect "shared source + sign flip," avoiding the loss of polarity information in traditional concatenation-based aggregation.
    - **Mechanism**: Each variable representation $\mathbf{v}_i^{(t)}\in\mathbb{R}^{2d}$ is split via two learnable mappings into invariant $\mathbf{v}_{i,\mathrm{inv}}^{(t)}$ and equivariant $\mathbf{v}_{i,\mathrm{eq}}^{(t)}$ components; positive/negative literal embeddings are formed as $\mathbf{l}_{x_i}^{(t)}=\mathbf{v}_{i,\mathrm{inv}}^{(t)}+\mathbf{v}_{i,\mathrm{eq}}^{(t)}$, $\mathbf{l}_{\neg x_i}^{(t)}=\mathbf{v}_{i,\mathrm{inv}}^{(t)}-\mathbf{v}_{i,\mathrm{eq}}^{(t)}$. After hypergraph propagation, recover invariant/equivariant parts via $\mathbf{v}_{\mathrm{inv},i}^{(t+1)}=\tfrac{1}{2}(\mathbf{L}_{2i}+\mathbf{L}_{2i+1})$, $\mathbf{v}_{\mathrm{eq},i}^{(t+1)}=\tfrac{1}{2}(\mathbf{L}_{2i}-\mathbf{L}_{2i+1})$, then recompose new variable representations via two MLPs. The invariant component is used with a linear head to predict unsat-core probability $\mathbf{s}=g(f'_{\mathrm{inv}}(\mathbf{V}_{\mathrm{inv}}^{(T)}))$.
    - **Design Motivation**: The unsat-core label is a "structural property" and invariant to polarity flip, so the prediction head should only use the invariant component. The linear "+/-" combination ensures that the sign flip operation is naturally a group action on literal embeddings, embedding the symmetry inductive bias in the architecture rather than the loss.

3. **Polarity-Flip Consistency Regularization**:

    - **Function**: Uses the polarity-flipped CNF as a "dual view" to inject symmetry into training without extra labels.
    - **Mechanism**: For each $\phi$, construct $\phi^{(flip)}$ (flip all literal signs but keep structure), share the network to obtain $\mathbf{s}$ and $\mathbf{s}^{(flip)}$; output-level consistency $\mathcal{L}_{\mathrm{cons}}=\tfrac{1}{|\mathcal{V}|}\|\mathbf{s}-\mathbf{s}^{(flip)}\|_2^2$ enforces identical predictions; decomposition-level consistency $\mathcal{L}_{\mathrm{decomp}}=\tfrac{1}{|\mathcal{V}|}\sum_i\bigl[\|\mathbf{V}_{i,\mathrm{inv}}^{(T)}-\mathbf{V}_{i,\mathrm{inv}}^{(T)(flip)}\|_2^2 + \|\mathbf{V}_{i,\mathrm{eq}}^{(T)}+\mathbf{V}_{i,\mathrm{eq}}^{(T)(flip)}\|_2^2\bigr]$ explicitly constrains invariants to be equal and equivariants to be sign-flipped.
    - **Design Motivation**: Architectural "+/-" combinations alone cannot guarantee that the two components truly correspond to "invariant/equivariant" semantics; the model could put all information into the equivariant part and have it canceled out. Decomposition-level consistency regularization directly encodes the mathematical definition into the loss, forcing information separation.

### Loss & Training

The total loss is $\mathcal{L}=\mathcal{L}_{\mathrm{core}}+\lambda_{\mathrm{cons}}\mathcal{L}_{\mathrm{cons}}+\lambda_{\mathrm{decomp}}\mathcal{L}_{\mathrm{decomp}}$. $\mathcal{L}_{\mathrm{core}}$ follows NeuroCore's KL divergence form $D_{\mathrm{KL}}(\mathbf{p}^*\|\mathbf{p})$, where the target distribution assigns uniform probability to unsat-core variables and zero elsewhere. The network is trained end-to-end; when integrated with CDCL, the neural network is run only once before solving to initialize variable activities, which are periodically reset to fixed scores during solving, but no repeated inference is performed.

## Key Experimental Results

### Main Results

On three datasets—SR (random 3-SAT), CA (community structure), PS (pigeonhole style)—precision, PR-AUC, and ROC-AUC are evaluated at Easy/Medium/Hard difficulty.

| Dataset | Metric | GCN | NeuroCore | SATFormer | paSAT |
|---------|--------|------|-----------|-----------|-------|
| SR (Avg) | PR-AUC | 0.915 | 0.938 | 0.920 | **0.963** |
| SR (Avg) | ROC-AUC | 0.732 | 0.805 | 0.741 | **0.872** |
| CA (Avg) | PR-AUC | 0.215 | 0.333 | 0.246 | **Significantly higher** |
| CA (Avg) | ROC-AUC | 0.518 | 0.605 | 0.527 | **Significantly higher** |

On the most challenging CA dataset (sparse community structure, small unsat-core proportion, highly imbalanced classes), paSAT achieves significant improvements over NeuroCore in both PR-AUC and ROC-AUC, indicating that higher-order and polarity modeling provide greater gains in structurally complex scenarios. The 6–9 point ROC-AUC improvement on SR also demonstrates more stable ranking quality.

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| Full paSAT | Best | Hypergraph + CIG + polarity decomposition + polarity-flip consistency |
| w/o CIG | Drops | No clause–clause association graph; higher-order clause dependencies only learned indirectly |
| w/o polarity decomposition | Drops | Degrades to "positive/negative literal concatenation + MLP" aggregation, loses algebraic symmetry |
| w/o $\mathcal{L}_{\mathrm{decomp}}$ | Drops | Decomposition architecture present, but without explicit supervision for invariant/equivariant semantics |
| w/o $\mathcal{L}_{\mathrm{cons}}$ | Drops | Lacks output-level dual-view supervision |

### Key Findings
- The clause association graph contributes most on the structurally complex CA dataset, showing that explicit modeling of clause–clause dependencies is the key bottleneck for higher-order problems.
- Architectural invariant/equivariant decomposition alone is insufficient; decomposition-level consistency loss is necessary to truly learn symmetry semantics, otherwise the two components may be "conspired" by the network to blend together.
- When integrated with CDCL, switching from "periodic invocation" to "one-time initialization" of the neural network results in almost no performance drop but greatly saves GPU time, indicating that paSAT's prediction signal alone provides a strong prior.

## Highlights & Insights
- Translating the algebraic symmetry of SAT formulas into "architecture + loss" dual constraints is an elegant way to inject inductive bias: architecturally, "+/-" makes the flip operation a group action; in the loss, decomposition-level MSE prevents information collapse—the two are complementary.
- Using polarity-flip to construct a dual view as "label-free self-supervision" is similar to "equivariant augmentation" in contrastive learning and can be transferred to any graph task with explicit group symmetry (e.g., molecular graph mirror flips, circuit voltage reversals).
- The dual-graph structure of "hypergraph + association graph" reminds us: when an object participates in both higher-order (clauses with multiple literals) and binary relations (clauses sharing variables), forcing everything into a single graph may confuse message passing objectives; splitting into two graphs with distinct roles is cleaner.

## Limitations & Future Work
- Experiments are only validated on the unsat-core task; whether polarity-flip consistency is equally effective for "backbone prediction, model counting, guidance for branching heuristics," etc., remains to be shown.
- The clause association graph has $O(M^2)$ edges; scalability to industrial-scale instances with millions of clauses requires further engineering (sparsification, sampling).
- Integration with CDCL is limited to "activity initialization," the weakest coupling; periodic neural network recall or dynamic use of paSAT in the search tree is unexplored, limiting solving time improvements.
- While polarity-flip invariance of $\phi^{(flip)}$'s satisfiability is used as Property 1, "assignment flip" (Property 2) is mentioned as used by NeuroBack, but paSAT itself only applies it to "assignment-independent" tasks like unsat-core; how to adapt the loss for "assignment-dependent" tasks like backbone prediction (possibly using sign constraints) is a direction for future work.

## Related Work & Insights
- **vs NeuroCore (Selsam & Bjørner 2019)**: This work adopts NeuroCore's KL training objective and CDCL integration paradigm, but replaces the bipartite graph with a hypergraph + CIG and explicitly models polarity. NeuroCore learns polarity implicitly via LSTM-style complementary literal message passing, while paSAT hard-codes constraints via architectural symmetry + loss, potentially improving sample efficiency.
- **vs SATFormer**: SATFormer uses a Transformer for global attention between literals and clauses, theoretically capturing arbitrary-order interactions, but this work outperforms SATFormer on SR and CA, indicating that inductive bias (especially polarity symmetry) is more important than pure architectural capacity.
- **vs NeuroBack (Wang et al. 2024)**: NeuroBack also uses polarity-flip to construct dual formulas for backbone prediction training, but only supervises at the output level; paSAT extends dual-view to "decomposition-level consistency" and introduces hypergraph structure, providing a more systematic symmetry modeling.
- **vs HGNN General Hypergraph Work**: The hypergraph convolution form $\mathbf{D}^{-1}\mathbf{H}\mathbf{B}^{-1}\mathbf{H}^\top$ follows the standard in (Bai et al. 2021); the main innovation lies in combining "domain knowledge + task symmetry," suggesting that for structured reasoning tasks, general GNNs must deeply exploit the algebraic structure of the task itself.

## Rating
- Novelty: ⭐⭐⭐⭐ The polarity-invariant–equivariant decomposition + polarity-flip dual view is a clear demonstration of "translating" SAT algebraic symmetry into neural networks; the dual-graph structure for higher-order clause relations is also original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic comparison on three datasets (SR/CA/PS) × three difficulty levels × three metrics, with complete module/loss ablation, but lacking end-to-end solving time comparison with CDCL integration.
- Writing Quality: ⭐⭐⭐⭐ Rigorous formula derivations, with Properties 1/2 providing a solid motivation; Figure 1's pipeline view is intuitive, though some sections are somewhat mathematical and lack solver-side intuitive examples.
- Value: ⭐⭐⭐⭐ Advances the SAT × GNN research line; the method is transferable to other structured prediction tasks with group symmetry, but short-term adoption by industrial SAT solvers faces a high threshold.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction](../../AAAI2026/graph_learning/unihr_hierarchical_representation_learning_for_unified_knowledge_graph_link_pred.md)
- [\[AAAI 2026\] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption](../../AAAI2026/graph_learning/feature-centric_unsupervised_node_representation_learning_without_homophily_assu.md)
- [\[ICML 2026\] Learning Graph Foundation Models on Riemannian Graph-of-Graphs](learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)
- [\[NeurIPS 2025\] Solar-GECO: Perovskite Solar Cell Property Prediction with Geometric-Aware Co-Attention](../../NeurIPS2025/graph_learning/solar-geco_perovskite_solar_cell_property_prediction_with_geometric-aware_co-att.md)
- [\[ICLR 2026\] Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization](../../ICLR2026/graph_learning/embodied_agents_meet_personalization_investigating_challenges_and_solutions_thro.md)

</div>

<!-- RELATED:END -->
