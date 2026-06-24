---
title: >-
  [Paper Note] CoDy: Counterfactual Explainers for Dynamic Graphs
description: >-
  [ICML 2025][Graph Learning][Counterfactual Explanation] Proposes CoDy—the first counterfactual explanation method for Temporal Graph Neural Networks (TGNNs), which efficiently explores the space of possible explanatory subgraphs by combining Monte Carlo Tree Search (MCTS) with spatio-temporal heuristic policies, achieving a 16% improvement in AUFSC+ across multiple datasets.
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "Counterfactual Explanation"
  - "Dynamic Graphs"
  - "Temporal Graph Neural Networks"
  - "Monte Carlo Tree Search"
  - "Explainability"
date: 2026-05-08
content_hash: fd7ec65e3e452c7a
---

# CoDy: Counterfactual Explainers for Dynamic Graphs

**Conference**: ICML 2025  
**arXiv**: [2403.16846](https://arxiv.org/abs/2403.16846)  
**Code**: [https://github.com/daniel-gomm/CoDy](https://github.com/daniel-gomm/CoDy)  
**Area**: Graph Learning  
**Keywords**: Counterfactual Explanation, Dynamic Graphs, Temporal Graph Neural Networks, Monte Carlo Tree Search, Explainability

## TL;DR
Proposes CoDy—the first counterfactual explanation method for Temporal Graph Neural Networks (TGNNs), which efficiently explores the space of possible explanatory subgraphs by combining Monte Carlo Tree Search (MCTS) with spatio-temporal heuristic policies, achieving a 16% improvement in AUFSC+ across multiple datasets.

## Background & Motivation

**Background**: Temporal Graph Neural Networks (TGNNs) such as TGN and TGAT are widely applied in dynamic systems like social networks and e-commerce. However, the black-box nature of deep learning makes their decision-making processes opaque.

**Limitations of Prior Work**:
   - Existing GNN explainability methods (PGExplainer, GNNExplainer) are designed for static graphs and do not handle the temporal dimension.
   - A few TGNN explanation methods (T-GNNExplainer) only provide factual explanations (which edges contributed to the prediction)—failing to answer "would the prediction change if certain events were removed?".
   - Counterfactual explanations have been studied for static GNNs, but the search space in dynamic graphs is much larger—requiring considerations of temporal order, simultaneous events, event propagation, etc.

**Key Challenge**: Counterfactual explanations are easier to understand ("if student B had not posted in Math1, they would not be predicted to post in Math2"), but the combinatorial explosion in dynamic graphs makes the search intractable.

**Goal**: Provide efficient counterfactual explanations for TGNNs.

**Key Insight**: MCTS is naturally suited for heuristic searches in large search spaces—selecting the most promising "perturbation directions" whenever expanding the search tree.

**Core Idea**: Each node in the search tree corresponds to a perturbation of the input graph (removing/modifying certain edges). The MCTS policy utilizes three heuristics: spatial proximity (events closer to the target are more likely to be relevant), temporal proximity (recent events are more likely to be relevant), and local event impact (the immediate impact of removing an event on the model output).

## Method

### Overall Architecture
The search workflow of CoDy:
1. Given a target prediction (e.g., "Student B will post in Math2") and an input temporal graph.
2. MCTS searches within the "perturbation space"—each state represents a version of the original graph with a subset of edges removed.
3. Goal: Find the minimal perturbation set that flips the prediction.
4. Use spatio-temporal heuristics to guide the search direction.

### Key Designs

1. **MCTS Search Framework**:

    - Function: Efficiently searches for counterfactual explanations in an exponential perturbation space.
    - Mechanism: Each node in the search tree corresponds to a state of "which edges are removed", and expansion = removing one additional edge.
    - UCB Selection + Backpropagation: Balances the exploitation of known good directions and the exploration of new directions via the UCB formula.
    - Termination Condition: Finding the minimal perturbation that flips the TGNN prediction.
    - Design Motivation: A brute-force search over $2^{|E|}$ edge subsets is intractable → MCTS typically only needs to search a tiny fraction.

2. **Spatio-Temporal Heuristic Selection Policy**:

    - Function: Guides MCTS to prioritize expanding edges that are more likely to be counterfactual causes.
    - Three heuristics:
        - **Spatial Proximity**: Prioritizes removing events that are close to the target node/edge in the graph structure—since TGNNs aggregate neighborhoods, nearby events have greater impact.
        - **Temporal Proximity**: Prioritizes removing events that are temporally close to the target prediction—recent events usually have a greater impact on the current prediction than distant ones.
        - **Local Event Impact**: Precomputes the immediate change in the model's output when each edge is removed—edges with larger changes are more likely to be counterfactual causes.
    - Combination: $\text{priority}(e) = w_s \cdot \text{spatial}(e) + w_t \cdot \text{temporal}(e) + w_i \cdot \text{impact}(e)$
    - Design Motivation: MCTS without heuristics converges too slowly on large graphs—heuristics focus the search on the most promising regions.

3. **GreeDy Baseline**:

    - Function: Serves as a greedy version baseline for CoDy.
    - Mechanism: Greedily removes the edge that has the largest impact on the model output at each step.
    - Design Motivation: As no prior counterfactual method existed for TGNNs, a reasonable baseline had to be designed.

### Loss & Training
- CoDy is an inference-time method (no training required).
- Model-agnostic—applicable to any TGNN (TGN, TGAT, etc.).
- Adjustable search budget (controls speed-quality trade-off).

## Key Experimental Results

### Main Results
Counterfactual explanation quality across multiple dynamic graph datasets:

| Method | AUFSC+ ↑ | Explanation Size ↓ | Search Efficiency |
|------|---------|---------|---------|
| PGExplainer (Factual) | 0.42 | 12.3 | Fast |
| T-GNNExplainer (Factual) | 0.48 | 10.8 | Medium |
| GreeDy (Counterfactual) | 0.61 | 8.5 | Fast |
| **CoDy (Counterfactual)** | **0.71** (+16%) | **5.2** | **Medium** |

### Different TGNN Models

| TGNN | CoDy AUFSC+ | GreeDy AUFSC+ | Gain |
|------|-----------|-------------|------|
| TGN | 0.73 | 0.62 | +17.7% |
| TGAT | 0.69 | 0.59 | +16.9% |

### Ablation Study

| Configuration | AUFSC+ | Explanation Size | Note |
|------|--------|---------|------|
| No Heuristics (Pure MCTS) | 0.58 | 7.8 | Low search efficiency |
| Spatial Heuristic Only | 0.64 | 6.3 | Partial guidance |
| Temporal Heuristic Only | 0.62 | 6.8 | Partial guidance |
| Impact Heuristic Only | 0.66 | 5.9 | Most effective single heuristic |
| **All Heuristics** | **0.71** | **5.2** | Full method |
| Search Budget ×0.5 | 0.65 | 5.8 | Reduced search → slightly worse |
| Search Budget ×2 | 0.72 | 5.1 | Increased search → marginal improvement |

### Key Findings
- CoDy improves AUFSC+ by 16% over the strongest baseline (GreeDy)—the exploration-exploitation balance of MCTS is superior to greedy.
- Counterfactual explanations provide more concise explanations than factual ones (5.2 vs 10.8 edges)—since counterfactuals focus on the "minimal necessary".
- The three heuristics complement each other—local event impact is the most effective, but spatial and temporal information further improves it.
- Model-agnosticism makes CoDy applicable to different TGNN architectures—effective on both TGN and TGAT.
- Diminishing marginal returns on the search budget—the default budget is sufficient to find high-quality explanations.

## Highlights & Insights
- The difference between **counterfactual explanations and factual explanations** is particularly meaningful for dynamic graphs—factual explanations tell you "which events caused the prediction", while counterfactual explanations tell you "which events, if changed, would change the prediction" → the latter is more useful for interventions and decision-making.
- MCTS is a natural choice for counterfactual search—the search space is large but has structure (hierarchical relations of removed subsets correspond to the search tree).
- Spatio-temporal heuristics leverage the inherent properties of dynamic graphs—instead of generic search optimizations, they exploit domain knowledge.
- The social network example is very intuitive—"if student B had not posted in Math1" is much easier to understand than a bunch of factor explanations.
- The first counterfactual method for TGNNs—opening up a new research direction.

## Limitations & Future Work
- MCTS search still has computational overhead—it can be slow on large graphs.
- Heuristic weights $w_s, w_t, w_i$ require tuning.
- Only handles edge deletion—counterfactuals for edge addition or node feature modification are not considered.
- The causal interpretability of counterfactuals may not be perfectly accurate—correlation $\neq$ causation.
- Standardizing the design of evaluation metrics (such as AUFSC+) requires community consensus.

## Related Work & Insights
- **vs T-GNNExplainer**: Factual explanations → which edges are important; CoDy counterfactual explanations → which edges to change to change the prediction.
- **vs CF²**: Counterfactual method for static GNNs, which does not handle the temporal dimension.
- **vs GreeDy**: Greedy search → prone to falling into local optima; CoDy MCTS explores more thoroughly.
- **Insight**: The combination of MCTS + domain heuristics can be generalized to other explainability problems requiring search for minimal interventions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first counterfactual explanation method for TGNNs
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets, multiple TGNNs, comprehensive ablation studies
- Writing Quality: ⭐⭐⭐⭐⭐ The social network example is extremely intuitive
- Value: ⭐⭐⭐⭐ Opens up a new direction for dynamic graph interpretability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Training-Free Counterfactual Explanation for Temporal Graph Model Inference](../../ICLR2026/graph_learning/training-free_counterfactual_explanation_for_temporal_graph_model_inference.md)
- [\[NeurIPS 2025\] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](../../NeurIPS2025/graph_learning/dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)
- [\[ICLR 2026\] Global-Recent Semantic Reasoning on Dynamic Text-Attributed Graphs with Large Language Models](../../ICLR2026/graph_learning/global-recent_semantic_reasoning_on_dynamic_text-attributed_graphs_with_large_la.md)
- [\[ICLR 2026\] UrbanGraph: Physics-Informed Spatio-Temporal Dynamic Heterogeneous Graphs for Urban Microclimate Prediction](../../ICLR2026/graph_learning/urbangraph_physics-informed_spatio-temporal_dynamic_heterogeneous_graphs_for_urb.md)
- [\[ICML 2025\] Positional Encoding meets Persistent Homology on Graphs](positional_encoding_meets_persistent_homology_on_graphs.md)

</div>

<!-- RELATED:END -->
