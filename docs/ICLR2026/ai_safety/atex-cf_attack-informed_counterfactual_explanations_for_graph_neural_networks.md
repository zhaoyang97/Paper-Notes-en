---
title: >-
  [Paper Note] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks
description: >-
  [ICLR 2026][AI Safety][Graph Neural Networks] This paper proposes ATEX-CF, a framework that, for the first time, unifies the edge-addition strategy from adversarial attacks with the edge-deletion strategy from counterfac…
tags:
  - "ICLR 2026"
  - "AI Safety"
  - "Graph Neural Networks"
  - "Counterfactual Explanations"
  - "Adversarial Attacks"
  - "Explainability"
  - "Graph Structure Perturbation"
date: 2026-05-08
content_hash: c1aa08db8e86a9d4
---

# ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks

**Conference**: ICLR 2026
**arXiv**: [2602.06240](https://arxiv.org/abs/2602.06240)
**Code**: [https://github.com/zhangyuo/ATEX_CF](https://github.com/zhangyuo/ATEX_CF)
**Area**: AI Safety / GNN Explainability
**Keywords**: Graph Neural Networks, Counterfactual Explanations, Adversarial Attacks, Explainability, Graph Structure Perturbation

## TL;DR

This paper proposes ATEX-CF, a framework that, for the first time, unifies the edge-addition strategy from adversarial attacks with the edge-deletion strategy from counterfactual explanations. Through joint optimization of prediction flipping, sparsity, and plausibility, ATEX-CF generates more faithful, concise, and plausible instance-level counterfactual explanations for GNNs.

## Background & Motivation

**Pressing demand for GNN explainability**: GNNs are widely deployed in high-stakes domains such as healthcare and finance, yet their black-box reasoning undermines trust, motivating the development of counterfactual explanation methods.

**Limitations of Prior Work**: Classical methods such as CF2 and GCFExplainer primarily rely on edge deletion to identify critical substructures supporting predictions, overlooking the possibility that adding missing relations can also substantially alter predictions.

**Adversarial attacks reveal the power of edge addition**: Graph adversarial learning has demonstrated that adding a small number (e.g., 2) of carefully selected edges can effectively flip the prediction of a target node, and these edges often correspond to semantically meaningful structural relationships.

**Long-standing disconnect between two directions**: Although counterfactual explanation and adversarial attack share the objective of flipping predictions, their perturbation strategies differ fundamentally—the former favors deletion while the latter favors addition—and no existing method unifies them.

**Complementary explanatory coverage via edge addition**: Edge-deletion explanations reveal which existing relations are critical, while edge-addition explanations reveal which missing relations could change the outcome; only together do they provide a complete understanding of model decisions.

**Search space explosion**: Incorporating edge addition into counterfactual search leads to combinatorial explosion of the candidate space, necessitating efficient strategies from adversarial attack research to constrain the search.

## Method

### Overall Architecture

ATEX-CF formulates counterfactual generation as a constrained optimization problem. The core pipeline is as follows:

1. **Candidate edge selection**: Two streams are maintained—edge-deletion candidates $\mathcal{S}^-$ are drawn from existing edges within the $(l+1)$-hop neighborhood of the target node; edge-addition candidates $\mathcal{S}^+$ are provided by the GOttack adversarial method as high-influence edges.
2. **Signed mask optimization**: A continuous parameter $M_e \in [-1,1]$ is introduced for each candidate edge, where $M_e > 0$ denotes addition and $M_e < 0$ denotes deletion, optimized via gradient descent.
3. **Forward discretization**: Thresholding combined with Top-$\kappa$ sparsification retains at most $\kappa$ perturbed edges.
4. **Joint loss optimization**: $\mathcal{L} = \lambda_1 \mathcal{L}_{pred} + \lambda_2 \mathcal{L}_{dist} + \lambda_3 \mathcal{L}_{plau}$.
5. **Post-minimization pruning**: Redundant edges are greedily removed to ensure minimality of the explanation.

### Key Designs

- **Theoretical bridge (Hypothesis 1)**: Edges added during successful evasion attacks exhibit high graph similarity to the counterfactual explanation subgraph of the target node, providing a principled basis for unifying the two directions.
- **Prediction loss $\mathcal{L}_{pred}$**: An indicator function combined with negative log-likelihood is activated only when the prediction has not yet been flipped, driving the perturbed graph away from the original class.
- **Sparsity loss $\mathcal{L}_{dist}$**: An $\ell_0$-norm constraint on the number of perturbed edges ensures conciseness and interpretability.
- **Plausibility loss $\mathcal{L}_{plau}$**: Comprises a degree anomaly penalty DegAnom (suppressing abrupt degree changes) and a motif violation penalty MotifViol (preserving local clustering coefficient stability).
- **Straight-Through Estimator (STE)**: Approximates the discretization operation as an identity function during backpropagation, allowing gradients to flow through non-differentiable operations.
- **Asymmetric cost extension**: A scalar weight $C$ controls the cost asymmetry between edge addition and deletion, accommodating domain-specific constraints.

## Key Experimental Results

### Table 1: Meta Results — Average Rank across Six Datasets (lower is better)

| Method | Misclass. | Fidelity | ΔE | Plausibility | Time | Overall Avg. | Wins |
|--------|-----------|----------|-----|-------------|------|-------------|------|
| CF-GNNExplainer | 4.7 | 4.8 | 2.0 | 2.3 | 9.5 | 4.67 | 1 |
| PGExplainer | 8.2 | 7.7 | 4.2 | 5.8 | 1.0 | 5.37 | 6 |
| Nettack | 3.3 | 2.5 | 8.8 | 8.0 | 4.5 | 5.43 | 2 |
| GOttack | 4.8 | 4.3 | 8.8 | 8.0 | 3.0 | 5.80 | 0 |
| **ATEX-CF (ours)** | **1.2** | **1.3** | **1.0** | **1.2** | 7.3 | **2.40** | **20** |

> ATEX-CF ranks first in 20 out of 30 metric–dataset combinations, far exceeding the second-best method PGExplainer with 6 wins.

### Table 2: Asymmetric Addition Cost Experiment (Cora, GCN, κ=20)

| Addition Cost $C$ | Misclass. | Fidelity | ΔE (E+, E-) | Plausibility | Time |
|-------------------|-----------|----------|-------------|-------------|------|
| 0.5 | 0.70 | 0.23 | 1.78 (0.78, 1.00) | 0.72 | 6.1s |
| 1.0 (symmetric) | 0.70 | 0.23 | 1.78 (0.77, 1.01) | 0.71 | 10.3s |
| 5.0 | 0.70 | 0.23 | 1.82 (0.65, 1.17) | 0.69 | 10.9s |
| 20.0 | 0.54 | 0.15 | 1.42 (0.49, 0.93) | 0.68 | 11.5s |
| 21.0 (deletion-only) | 0.42 | 0.10 | 1.78 (0.00, 1.78) | 0.62 | 11.7s |

> When the addition cost $C$ is excessively high (≥20), the method degrades to pure deletion with significant performance drop, validating the importance of the edge-addition strategy.

## Highlights & Insights

- **First theoretical bridge**: Hypothesis 1 formalizes the high similarity between adversarial attack subgraphs and counterfactual explanation subgraphs, providing a principled foundation for unifying the two directions.
- **Complementary explanation paradigm**: Edge deletion answers "which existing relations are critical," while edge addition answers "which missing relations could change the outcome"; their combination yields a more complete understanding of model behavior.
- **Motivating practical case**: In a loan approval scenario, edge deletion fails to flip the prediction, whereas plausible edge addition succeeds, intuitively demonstrating the practical value of the approach.
- **Efficient search space constraint**: Graph orbit theory from GOttack is leveraged to narrow the edge-addition candidate set, resolving the combinatorial explosion problem.
- **Effective post-pruning**: Runtime is reduced from 6.12s to 3.00s and edge count from 1.71 to 1.62 without sacrificing prediction accuracy.

## Limitations & Future Work

1. **Structural perturbations only**: The current framework handles only edge addition and deletion, without considering node feature perturbations, and thus cannot capture counterfactuals at the feature level.
2. **High computational overhead**: Although overall ranking is best, the Time metric ranks 7.3 (near last), indicating that efficiency on large-scale graphs remains to be improved.
3. **Dependence on GOttack as the attack source**: Candidate edge additions rely entirely on the quality of GOttack, with no exploration or comparison of alternative attack methods.
4. **Static graphs only**: The framework has not been extended to dynamic graphs, temporal graphs, or other more complex graph structures.
5. **Limited plausibility metrics**: Only degree anomaly and clustering coefficient are used as plausibility indicators, which may be insufficient to capture domain-specific semantic constraints.

## Related Work & Insights

- **CF-GNNExplainer (Lucic et al., 2022)**: A classical counterfactual GNN explainer relying solely on edge deletion; the direct predecessor that this work improves upon.
- **GOttack (Alom et al., 2025, ICLR)**: A universal adversarial attack based on graph orbit learning, used in ATEX-CF as the generator of edge-addition candidates.
- **C2Explainer (Ma et al., 2025)**: A customizable masked counterfactual explainer supporting both edge and feature perturbations, but with inferior joint optimization performance compared to ATEX-CF.
- **Insights**: Adversarial attack and model explanation, two seemingly opposing directions, can mutually reinforce each other—attack strategies provide efficient search mechanisms, while explanation requirements impose plausibility constraints. This unified perspective may generalize to other domains (e.g., adversarial example explanation in NLP).

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | 4 | First to unify adversarial attacks and counterfactual explanations; theoretical contribution is clear |
| Technical Depth | 4 | Joint optimization framework is well-designed; theoretical hypothesis is experimentally supported |
| Experimental Thoroughness | 4 | 6 datasets × 3 GNN architectures × 10 baselines; comprehensive ablation and sensitivity analyses |
| Writing Quality | 4 | Motivation is clear, illustrative cases are vivid, framework diagrams are intuitive |
| Value | 3 | Code is open-sourced, but high time overhead and domain applicability require further validation |
| **Overall** | **3.8** | Solid ICLR contribution; the unified perspective constitutes the core contribution |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[ICLR 2026\] Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks](time_is_all_it_takes_spike-retiming_attacks_on_event-driven_spiking_neural_netwo.md)
- [\[NeurIPS 2025\] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](../../NeurIPS2025/ai_safety/influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)
- [\[ICLR 2026\] Hide and Find: A Distributed Adversarial Attack on Federated Graph Learning](hide_and_find_a_distributed_adversarial_attack_on_federated_graph_learning.md)
- [\[ICLR 2026\] Bridging Fairness and Explainability: Can Input-Based Explanations Promote Fairness in Hate Speech Detection?](bridging_fairness_and_explainability_can_input-based_explanations_promote_fairne.md)

</div>

<!-- RELATED:END -->
