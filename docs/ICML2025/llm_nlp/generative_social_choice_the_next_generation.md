---
title: >-
  [Paper Note] Generative Social Choice: The Next Generation
description: >-
  [ICML2025][LLM (Other)][Generative Social Choice] Extends the generative social choice framework to scenarios with cost/budget constraints and approximate queries, proposes the DemocraticProcess algorithm with near-optimal approximate proportional representation guarantees, and implements a practical system PROSE (based on GPT-4o) validated on drug review and urban governance datasets.
tags:
  - "ICML2025"
  - "LLM (Other)"
  - "Generative Social Choice"
  - "Proportional Representation"
  - "Participatory Budgeting"
  - "LLM Queries"
  - "Approximation Guarantees"
date: 2026-05-08
content_hash: 82497ac21e594b7d
---

# Generative Social Choice: The Next Generation

**Conference**: ICML2025  
**arXiv**: [2505.22939](https://arxiv.org/abs/2505.22939)  
**Code**: [github.com/sara-fish/gen-soc-choice-next-gen](https://github.com/sara-fish/gen-soc-choice-next-gen)  
**Area**: Social Choice / AI & Democracy  
**Keywords**: Generative Social Choice, Proportional Representation, Participatory Budgeting, LLM Queries, Approximation Guarantees

## TL;DR
Extends the generative social choice framework to scenarios with cost/budget constraints and approximate queries, proposes the DemocraticProcess algorithm with near-optimal approximate proportional representation guarantees, and implements a practical system PROSE (based on GPT-4o) validated on drug review and urban governance datasets.

## Background & Motivation
- **Collective response systems like Polis**: Online participants submit opinions and vote on each other's, and the system selects a subset of statements that represent diverse viewpoints. Such systems have been used for national policy-making in Taiwan, Australia, etc.
- **Limitations of Prior Work**: The generative social choice framework proposed by Fish et al. [2024] is a milestone, but relies on two overly strong assumptions:
  1. **Fixed k statements of equal length**: Unable to control the length of statements, where a small k might lead to excessively long summaries.
  2. **Exact queries**: Theoretical guarantees rely on exact responses to discriminative and generative queries, which LLMs cannot answer perfectly.
- **Core Motivation**: In practical deployments, LLM queries inevitably introduce errors, and statements exhibit varying lengths/costs, necessitating proportional allocation under a total budget constraint. This naturally generalizes committee elections to participatory budgeting.

## Method

### Problem Modeling
- **Statement universe** $\mathcal{U}$ (can be infinite), cost function $c: \mathcal{U} \to \mathbb{N}_0$ (e.g., word count)
- **$n$ agents**, where each agent $i$ has a utility function $u_i: \mathcal{U} \to [r]$ ($r$ utility levels, e.g., "strongly agree" $\to$ "strongly disagree")
- **Goal**: Under a total budget $B$ constraint, select a slate $W \subseteq \mathcal{U}$ to proportionally represent all agents.

### Query Models
Accessing the unknown $\mathcal{U}$ and utility functions through two types of queries:

| Query Type | Exact Version | Approximation Parameter |
|---------|---------|---------|
| **Discriminative Query** Disc$(i, \alpha)$ | Returns $u_i(\alpha)$ | $\beta$-accurate: error $\leq \beta$ |
| **Generative Query** Gen$(S, \ell, x)$ | Returns a statement with cost $\leq x$ that is supported at level $\ell$ by the most agents in $S$ | $(\gamma, \delta, \mu)$-accurate |

Three types of errors in generative queries:
- $\gamma$: multiplicative error in supporter count
- $\delta$: additive error in utility assessment
- $\mu$: multiplicative error in cost assessment (GPT-4o often underestimates target length)

Formalization of approximate generative queries: return $\alpha^*$ satisfying $c(\alpha^*) \leq x$ and

$$\frac{\mathrm{sup}(\alpha^*, S, \ell - \delta)}{\max_{\alpha \in \mathcal{U}: c(\alpha) \leq \lceil \mu x \rceil} \mathrm{sup}(\alpha, S, \ell)} \geq \gamma$$

### Proportional Representation Axiom: $(b, d)$-costBJR
Defines the cost-aware Balanced Justified Representation (cBJR):
- There exists a balanced mapping $\omega: N \to W$, and no coalition $S$, statement $\alpha$, and threshold $\theta$ simultaneously satisfy:
  1. $|S| \geq d \cdot \lceil c(\alpha) \cdot n / B \rceil$ (coalition is large enough)
  2. $u_i(\alpha) \geq \theta, \forall i \in S$ (coalition cohesively prefers $\alpha$)
  3. $u_i(\omega(i)) < \theta - b, \forall i \in S$ (current assignment is significantly worse than $\alpha$)
- The exact version of cBJR corresponds to $b=0, d=1$.

### DemocraticProcess Algorithm
Core Idea: Iterative greedy search—scanning from high utility levels to low levels, and attempting to generate and add statements with sufficient support in each round:

1. Outer loop: utility level $\ell$ decreases from $r$ to $1$.
2. Inner loop: iterate through the cost list $C$, calling the generative query for each cost $C[j]$.
3. Use discriminative queries to filter the set of agents $S_\alpha$ who support the returned statement at level $\ell$.
4. If the support of the best statement $\alpha^*$ is $\geq \lceil c(\alpha^*) \cdot n / B \rceil$, add it to the slate and remove the corresponding agents.
5. Otherwise, increase the cost and continue the search.

Two key variants:
- **Fast-DemocraticProcess**: $C = \{\lfloor j \cdot B/n \rfloor \mid j \in [n]\}$, $f(\ell) = \{\ell\}$ $\to$ guarantees cBJR under exact queries.
- **Complex-DemocraticProcess**: $C = [B]$, $f(\ell) = [\ell, r]$ $\to$ optimal guarantees under approximate queries.

### Theoretical Guarantees

**Theorem 3.1** (Exact Queries): Fast-DemocraticProcess satisfies $(0, 1/\gamma)$-cBJR under exact discriminative queries and $(\gamma, 0, 1)$-accurate generative queries.

**Theorem 3.2** (Approximate Queries, Core Result): Complex-DemocraticProcess satisfies $(2\beta + \delta, \frac{1}{\gamma\mu})$-cBJR under $\beta$-accurate discriminative queries and $(\gamma, \delta, \mu)$-accurate generative queries.

**Theorems 3.3–3.4** (Lower Bounds): Near-matching impossibility results, proving that the dependence of approximation guarantees on error parameters is close to optimal.

### PROSE: Proportional Slate Engine
Practical system implementation:
- Discriminative queries: Use GPT-4o as a human preference model to predict user utilities for statements.
- Generative queries: A two-step strategy—① Use text-embedding-3-large embeddings + clustering/nearest neighbors to identify cohesive subgroups, and ② Use GPT-4o to generate a consensus statement for each subgroup.
- Key Advantage: Requires only unstructured text data + target slate length as input, without needing dataset-specific hyperparameter tuning.

## Key Experimental Results

### Datasets

| Dataset | Source | Number of Agents | Budget $B$ |
|-------|------|---------|---------|
| Birth Control (Balanced) | UCI Drug Review | 80 | 160 words |
| Birth Control (Imbalanced) | UCI Drug Review | 80 | 160 words |
| Obesity | UCI Drug Review | 80 | 160 words |
| Bowling Green | Polis City Governance Discussion | 41 | 164 words |

### Baselines

| Method | Description |
|------|------|
| Contextless Zero-Shot | Provided only with the topic and length constraints, without user data |
| Zero-Shot | Provided with all user descriptions, generating the slate in a single step |
| Clustering | Embedding + affinity propagation clustering, generating one statement per cluster |
| PROSE-UnitCost | Equal-cost version, corresponding to the original framework of Fish et al. |

### Core Results
- PROSE outperforms all four baselines on both **user satisfaction** and **proportional representation** dimensions.
- Synthetic environment verification: The empirical performance of all algorithm variants is far superior to the worst-case theoretical guarantees (grey region in Fig 1).
- The Fast and Complex variants perform closely and both significantly outperform the Uniform variant.
- As error increases, BJR violations for Fast and Complex gradually grow, but the increase remains controlled.

## Highlights & Insights
1. **Natural generalization from committee elections to participatory budgeting**: Introducing cost/budget constraints renders the framework more realistic—controlling total slate length is more practical than fixing the number of statements.
2. **Fault-tolerant design**: The approximate query model elegantly quantifies the imperfection of LLMs, with theoretical guarantees degrading smoothly with error parameters instead of breaking down entirely.
3. **Near-matching upper and lower bounds**: The combination of Theorem 3.2 and Theorems 3.3–3.4 proves that the approximation ratio of the algorithm is close to the information-theoretic limit.
4. **Strong practicality**: PROSE requires only unstructured text input without dataset-specific parameter tuning, making it widely applicable.
5. **Error-agnostic execution**: DemocraticProcess does not need to know the exact values of $\beta, \gamma, \delta, \mu$ during execution.

## Limitations & Future Work
1. **Opacity of GPT-4o**: Query implementation relies on a black-box LLM, which cannot guarantee the quality of individual responses and might exhibit bias or hallucinations.
2. **High computational cost of Complex-DemocraticProcess**: $C=[B]$ causes the inner loop to iterate through all possible cost values, requiring trade-offs in practical deployment.
3. **Difficulty in implementing generative queries**: The authors observed that GPT-4o struggles to identify cohesive participant subgroups, shifting the reliance onto an external embedding-based clustering step.
4. **Limited dataset scale**: Experiments only involve 41–80 agents, leaving large-scale (thousands of users) scenarios unverified.
5. **Gap in the upper and lower bounds of $\gamma$ error**: Theorem 3.2 yields $1/\gamma$ while the lower bound is $|W|/(|W|\gamma+1)$. The gap remains non-negligible when $|W|$ is small.

## Related Work & Insights
- Fish et al. [2024]: Original framework of generative social choice, which this work directly extends.
- Polis [Small et al., 2021]: The most widely used collective response system.
- Tessler et al. [2024], Bakker et al. [2022]: Generating a single consensus statement using LLMs (as opposed to multi-statement proportional representation in this work).
- Peters et al. [2021]: Proportional representation theory in participatory budgeting.

## Rating
- Novelty: ⭐⭐⭐⭐ — The dual extension of cost/budget and approximate queries yields substantial theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ — Verified on both synthetic and real-world data, but the scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear theoretical-to-practical framework, with motivation leading seamlessly to conclusions.
- Value: ⭐⭐⭐⭐ — Important progress in the direction of AI and democracy, though some distance remains from actual deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] LLM Social Simulations Are a Promising Research Method](llm_social_simulations_are_a_promising_research_method.md)
- [\[ACL 2025\] Explicit and Implicit Data Augmentation for Social Event Detection](../../ACL2025/llm_nlp/explicit_and_implicit_data_augmentation_for_social_event_detection.md)
- [\[ACL 2025\] Which of These Best Describes Multiple Choice Evaluation with LLMs?](../../ACL2025/llm_nlp/multiple_choice_eval.md)
- [\[ACL 2026\] Generative Interfaces for Language Models](../../ACL2026/llm_nlp/generative_interfaces_for_language_models.md)
- [\[ACL 2025\] Temporal Reasoning for Timeline Summarisation in Social Media](../../ACL2025/llm_nlp/temporal_reasoning_for_timeline_summarisation_in_social_media.md)

</div>

<!-- RELATED:END -->
