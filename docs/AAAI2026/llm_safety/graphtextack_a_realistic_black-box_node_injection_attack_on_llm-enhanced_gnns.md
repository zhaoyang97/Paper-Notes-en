---
title: >-
  [Paper Note] GraphTextack: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs
description: >-
  [AAAI 2026][LLM Safety][Text-attributed graphs] This paper proposes GraphTextack — the first black-box multimodal node injection poisoning attack targeting LLM-enhanced GNNs. It jointly optimizes the graph structural con…
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "Text-attributed graphs"
  - "adversarial attacks"
  - "node injection"
  - "LLM-enhanced GNN"
  - "evolutionary optimization"
  - "black-box attack"
  - "multimodal attack"
date: 2026-05-08
content_hash: be971426f2007ae0
---

# GraphTextack: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs

**Conference**: AAAI 2026
**arXiv**: [2511.12423](https://arxiv.org/abs/2511.12423)
**Code**: To be confirmed
**Area**: AI Security / Adversarial Attacks
**Keywords**: Text-attributed graphs, adversarial attacks, node injection, LLM-enhanced GNN, evolutionary optimization, black-box attack, multimodal attack

## TL;DR

This paper proposes GraphTextack — the first black-box multimodal node injection poisoning attack targeting LLM-enhanced GNNs. It jointly optimizes the graph structural connections and semantic features of injected nodes via an evolutionary optimization framework, requiring neither internal model information nor surrogate models. GraphTextack significantly outperforms 12 baseline methods across 5 datasets and 2 types of LLM-GNN models.

## Background & Motivation

- **Background**: In text-attributed graphs (TAGs), nodes possess both structural relationships and textual descriptions. Recent LLM-enhanced GNN methods (e.g., One-for-all, E5+GCN) extract semantic embeddings via LLMs and then perform structural aggregation with GNNs, achieving state-of-the-art performance on tasks such as node classification. However, such integration inherits the dual vulnerabilities of GNNs to structural perturbations and LLMs to textual adversarial inputs.
- **Limitations of Prior Work**: (1) Most existing attacks are **unimodal** — perturbing either structure or text alone — and are thus limited against LLM-enhanced GNNs: text attacks typically reduce accuracy by less than 5%, while structural attacks require modifying more than 10% of edges or assume white-box access. (2) Most attacks rely on unrealistic assumptions — white-box access, direct modification of existing nodes/edges, or dependence on surrogate models. (3) In realistic scenarios, attackers can only create new entities (e.g., fake users or products) rather than modify existing data.
- **Key Challenge**: The vulnerability of LLM-enhanced GNNs is distributed across two mutually dependent modalities — structure and semantics — making unimodal attacks insufficient. Joint optimization, however, faces a discrete, high-dimensional, non-differentiable combinatorial search space.
- **Key Insight**: Node injection attacks — injecting new nodes rather than modifying existing data — simulate real-world scenarios such as creating fake accounts or products. A gradient-free evolutionary algorithm is adopted to navigate the joint search space.

## Method

### Overall Architecture

GraphTextack employs an iterative evolutionary optimization framework. For each node to be injected, a population of candidate injection strategies is maintained and iteratively optimized through selection, crossover, mutation, and fitness evaluation. The core innovations are: (1) a joint candidate representation encoding structural connections and semantic features; (2) multimodal crossover and mutation operators; and (3) a multi-objective fitness function that integrates local prediction perturbation and global graph influence.

### Key Designs

1. **Candidate Representation and Class-Conditional Semantic Feature Generation (C1+C2)**

    - Each candidate $s_i$ encodes: edge connections $E_i' \subseteq V' \times V$ for the injected node, plus a feature generation strategy (class label $c \in \mathcal{Y}$).
    - The number of edges is sampled from the degree distribution of the original graph, ensuring that the injected node's connectivity pattern resembles that of legitimate nodes (stealthiness).
    - Feature generation: given a specified class label $c$, node features are sampled from the empirical distribution of existing nodes belonging to that class, i.e., $X'(v') \sim p(X(v) | Y(v) = c)$ — pseudo-labels are obtained by querying the target model to construct this distribution.
    - **Design Motivation**: This avoids direct optimization over continuous embeddings (high-dimensional and unrealistic); class-conditional sampling enables injected nodes to blend naturally into the existing node distribution in the semantic space.

2. **Multimodal Multi-Objective Fitness Function (C3)**

    - $\text{Fitness}(s_i) = \alpha \cdot \Delta_{\text{conf}}(s_i) + \beta \cdot \text{PR}(s_i)$
    - **Local prediction shift** $\Delta_{\text{conf}}$: the average change in maximum confidence among nodes within the two-hop neighborhood before and after injection, $\frac{1}{|\mathcal{N}_2(v')|}\sum_{v \in \mathcal{N}_2(v')}|C_v - C_v'|$ — measures the semantic-level attack effect.
    - **Global PageRank influence** $\text{PR}$: the PageRank score of the injected node — higher centrality implies a greater reach through message passing.
    - **Design Motivation**: A single objective (prediction shift or centrality alone) is insufficient to capture the effect of poisoning attacks, since poisoning impact manifests only after model retraining.

3. **Multimodal Evolutionary Operators (C4)**

    - **Selection**: Candidates are ranked by fitness; the top-$N_e$ elite individuals are retained.
    - **Crossover**: A crossover point $j$ is randomly selected; the new candidate inherits the first $j$ edges from $s_1$ and the remaining edges from $s_2$, with the class label randomly inherited from one parent — enabling joint exploration of structural and semantic dimensions.
    - **Mutation**: Edge connections or feature assignments are randomly modified with probability $p_{\text{mut}}$ — maintaining population diversity.
    - Evolutionary optimization is run independently for each injection step, incrementally constructing the poisoned graph.

### Complexity Analysis

The search space is $O(|V|^{r \cdot d_{\max}} \times |\mathcal{F}|^r)$, which is exponentially intractable by exhaustive enumeration. With a fixed population size $N_p$ and number of generations $T_{\text{gen}}$, GraphTextack achieves a per-step complexity of $O(N_p \cdot T_{\text{gen}} \cdot (r \cdot d_{\max}^2 + |E|))$, scaling effectively linearly with graph size.

## Experiments

### Main Results (Representation-level Enhancer, One-for-all Model)

| Dataset (Clean Acc) | Method | r=0.01 | r=0.03 | r=0.05 |
|:--|:--|:--|:--|:--|
| Cora (80.95) | Best text* | — | 74.36 | — |
| | GANI | 77.18 | 70.98 | 66.06 |
| | WTGIA | 76.35 | 69.63 | 65.80 |
| | **GraphTextack** | **73.99** | **65.75** | **62.02** |
| PubMed (71.65) | WTGIA | 65.89 | 48.96 | 43.10 |
| | G²A²C | 62.10 | 54.96 | 43.78 |
| | **GraphTextack** | **60.78** | **48.43** | **42.05** |
| WikiCS (76.31) | WTGIA | 71.87 | **64.17** | **60.95** |
| | **GraphTextack** | **71.69** | 64.58 | 61.35 |
| ogbn-arxiv (75.44) | G²A²C | 72.63 | 69.50 | 66.67 |
| | **GraphTextack** | **71.95** | **68.23** | **66.61** |
| ogbn-products (83.51) | GANI | 78.93 | 74.83 | 69.51 |
| | **GraphTextack** | 78.99 | **74.26** | **69.05** |

### Ablation Study (WikiCS, Representation-level Enhancer)

| Variant | r=0.01 | r=0.05 |
|:--|:--|:--|
| GraphTextack (full) | **71.69** | **61.35** |
| w/o crossover | 73.50 | 64.42 |
| w/o mutation | 73.39 | 65.19 |
| w/o prediction shift | 73.77 | 66.83 |
| w/o PageRank | 71.92 | 61.97 |

### Key Findings

- **Advantage of multimodal attacks**: Even the strongest text attack — which modifies all node features — is less effective than GraphTextack injecting only 1–5% of nodes, confirming the necessity of joint structural and semantic attacks.
- **Advantage of eliminating surrogate models**: By querying the target model directly, GraphTextack avoids approximation errors introduced by architectural discrepancies between surrogate and target models, with particularly notable gains on representation-level enhancer models.
- **Best runtime efficiency**: By avoiding gradient computation and surrogate model training, GraphTextack achieves the lowest average injection runtime across all datasets.
- **Ablation validation**: All four components — crossover, mutation, prediction shift, and PageRank — are indispensable, with prediction shift having the largest individual impact on attack effectiveness.

## Highlights & Insights

- ⭐ **First black-box multimodal node injection attack**: Requires no model gradients, parameters, or surrogate models, while jointly optimizing both structural and semantic modalities.
- ⭐ **Highly realistic threat model**: Node injection (creating new entities) better reflects real-world attack scenarios than modifying existing data — e.g., injecting fake products into e-commerce networks or low-quality papers into academic citation graphs.
- ⭐ **Theoretical analysis provided**: Includes search space complexity, multimodal adversarial synergy (the cross term $\gamma \cdot d_E d_X$ in Lemma B.1), and a bound on local prediction shift (Lemma B.3).
- ⭐ Achieves the best or near-best overall attack performance among 12 baselines while maintaining the highest runtime efficiency.

## Limitations & Future Work

- **Simplistic class-conditional feature sampling**: Features are sampled directly from the empirical distribution of same-class nodes, without learning a more adaptive feature generation strategy.
- **Assumes unrestricted connection creation**: In practice, edges from injected nodes to existing nodes may be subject to constraints (e.g., mutual confirmation in social networks).
- **Sensitivity to evolutionary hyperparameters**: Population size, crossover/mutation probabilities, and the $\alpha$/$\beta$ balance coefficient require careful tuning.
- **Defense not evaluated**: The paper only mentions "laying the groundwork for future defense work" without assessing attack robustness against any defensive mechanism.

## Related Work & Insights

- **Graph adversarial attacks (structural)**: Nettack (Zügner et al. 2018) — greedy modification; PRBCD (Geisler et al. 2021) — projected gradient; Meta-attack (Zügner & Günnemann 2020).
- **Node injection attacks**: AFGSM (Wang et al. 2020) — approximate fast gradient sign; TDGIA (Zou et al. 2021) — topological defect injection; GANI (Fang et al. 2024) — genetic algorithm with surrogate model; G²A²C (Ju et al. 2023) — reinforcement learning.
- **Textual adversarial attacks**: BertAttack (Li et al. 2020), HotFlip (Ebrahimi et al. 2018), VIPER (Eger et al. 2019).
- **LLM-GNN attacks**: WTGIA (Lei et al. 2024) — text-level injection attack (unimodal); Guo et al. (2024) — white-box modification attack.

## Rating

⭐⭐⭐⭐ — Novel problem formulation (first black-box multimodal injection), comprehensive experiments (5 datasets × 2 models × 12 baselines), with both theoretical and empirical contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization](psm_prompt_sensitivity_minimization_via_llm-guided_black-box_optimization.md)
- [\[ACL 2026\] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework](../../ACL2026/llm_safety/rethinking_llm_watermark_detection_in_black-box_settings_a_non-intrusive_third-p.md)
- [\[ICLR 2026\] Enhancing Hallucination Detection through Noise Injection](../../ICLR2026/llm_safety/enhancing_hallucination_detection_through_noise_injection.md)
- [\[AAAI 2026\] Democratizing LLM Efficiency: From Hyperscale Optimizations to Universal Deployability](democratizing_llm_efficiency_from_hyperscale_optimizations_to_universal_deployab.md)
- [\[AAAI 2026\] LLM Targeted Underperformance Disproportionately Impacts Vulnerable Users](llm_targeted_underperformance_disproportionately_impacts_vulnerable_users.md)

</div>

<!-- RELATED:END -->
