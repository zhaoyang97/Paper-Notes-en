---
title: >-
  [Paper Note] Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents
description: >-
  [AAAI 2026][LLM Agent][personalized data selection] This paper formalizes structured constraints in LLM agent personalization—comprising logical dependencies and hierarchical quotas—as laminar matroids…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "personalized data selection"
  - "submodular optimization"
  - "matroid constraints"
  - "knowledge graph compilation"
  - "data minimization"
date: 2026-05-08
content_hash: a7ae21f92e3669c5
---

# Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents

**Conference**: AAAI 2026
**arXiv**: [2512.11907](https://arxiv.org/abs/2512.11907)  
**Code**: None  
**Area**: LLM Agent / Personalization Optimization
**Keywords**: personalized data selection, submodular optimization, matroid constraints, knowledge graph compilation, data minimization

## TL;DR

This paper formalizes structured constraints in LLM agent personalization—comprising logical dependencies and hierarchical quotas—as laminar matroids, proves that greedy algorithms retain constant-factor approximation guarantees under such constraints, and addresses the data-minimization selection problem with dependency relations and hierarchical limits.

## Background & Motivation

**Background**: Personalization of LLM agents is typically achieved by injecting user-specific data into the context. More data generally improves task performance, but also incurs higher token costs and privacy leakage risks.

**Limitations of Prior Work**: Under ideal conditions, personalization utility is submodular (diminishing returns), and simple greedy selection yields near-optimal solutions. In practice, however, user data exhibits two types of structured constraints: (a) **logical dependencies**—selecting fact A may require simultaneously selecting fact B (e.g., selecting "the user is responsible for ProjectPinnacle" requires selecting "ProjectPinnacle is a legacy system migration project," otherwise the agent may produce erroneous inferences); and (b) **hierarchical quotas**—nested cardinality limits (e.g., at most 3 hobbies, of which at most 1 may be a water sport).

**Key Challenge**: These structured constraints violate the independence assumptions of standard subset selection algorithms, invalidating the near-optimality guarantees of greedy methods. Yet existing personalization approaches either ignore these constraints or handle them with ad hoc heuristics.

**Goal**: To provide a mathematically rigorous framework for modeling these complex constraints, enabling optimization algorithms with provable guarantees to be applied in real-world personalization scenarios.

**Key Insight**: Drawing from combinatorial optimization theory, the paper leverages matroids—an algebraic structure—to characterize the properties of the feasible solution space.

**Core Idea**: A two-stage compilation process: first, strongly connected components (SCCs) are used to compress logically dependent facets into macro-facets (resolving dependencies); then the hierarchical quota constraints over macro-facets are shown to form a laminar matroid, reducing the problem to the classical problem of maximizing a submodular function subject to matroid constraints.

## Method

### Overall Architecture

The pipeline consists of three steps: **(1) Modeling**—the user identity is represented as a knowledge graph (chronicle), with logical implication relations between facets; **(2) Compilation**—SCC condensation compresses dependent facets into macro-facets, resolving the dependency structure; **(3) Constraint Modeling**—the hierarchical quota constraints over macro-facets are shown to form a laminar matroid, and a greedy algorithm is applied to obtain a $1/2$ approximation (or the continuous greedy achieves a $(1-1/e)$ approximation).

### Key Designs

**Module 1: Chronicle and Facet Dependency Compilation**

- **Function**: Compiles logically inseparable combinations of facets in the user identity knowledge graph into "macro-facets."
- **Mechanism**: Facets in the user identity $\mathcal{C}_u = \{e_1, \dots, e_n\}$ are connected via a directed implication graph $G = (\mathcal{C}_u, E)$, where edge $(e_i, e_j)$ indicates that selecting $e_i$ logically requires $e_j$. Any valid personalization filter must be a closed set (containing all transitively implied facets). **Key operation**: Compute the SCCs of the implication graph $G$; each SCC constitutes a macro-facet. Selecting a macro-facet automatically includes its entire closure $\text{cl}_G(m)$, satisfying all logical dependencies. The utility function is redefined over the new ground set as $U'(S_\mathcal{M}) \triangleq U(\text{Exp}(S_\mathcal{M}))$.
- **Design Motivation**: Encoding the constraint satisfaction problem into the ground set itself ensures that selection operations inherently respect dependency relations, simplifying subsequent optimization.

**Module 2: Laminar Matroid Constraint Proof**

- **Function**: Proves that hierarchical quota constraints over the macro-facet set form a laminar matroid.
- **Mechanism**: In a laminar family $\mathcal{F} \subseteq 2^\mathcal{M}$, any two sets are either nested or disjoint. A quota $q_A$ is imposed on each set $A \in \mathcal{F}$, and the independent sets are defined as those satisfying $|S_\mathcal{M} \cap A| \leq q_A$ for all $A$. The matroid axioms (empty set, hereditary property, augmentation) are verified to confirm that $(\mathcal{M}, \mathcal{I})$ is a matroid. The augmentation proof crucially exploits the nesting/disjointness property of the laminar family: if no augmenting element exists, the nesting structure of tight constraints leads to $|A| \geq |B|$, a contradiction. **Important convention**: Quotas are checked only against the selected macro-facet set (pre-closure); expanded elements do not count toward quotas.
- **Design Motivation**: Matroid structure is the key algebraic condition enabling greedy algorithms to achieve near-optimality guarantees. Partition matroids (simple categorical quotas) are a special case.

**Module 3: Greedy Selection Algorithm and Independence Oracle**

- **Function**: Implements an efficient greedy selection algorithm that iteratively selects the macro-facet with the highest marginal utility while maintaining independence.
- **Mechanism**: The algorithm initializes $S = \emptyset$ and at each step finds $m^* = \arg\max_{m \in C} [U'(S \cup \{m\}) - U'(S)]$, adding $m^*$ if $S \cup \{m^*\} \in \mathcal{I}$. **Independence Oracle** implementation: the laminar family is represented as a rooted forest; for each macro-facet, its ancestor chain $P(m)$ is precomputed (by laminarity, this is a root-to-leaf path), and independence is checked by verifying $\text{cnt}[A] < q_A$ for all $A \in P(m)$. Complexity is only $O(h)$ (where $h$ is the tree height), making constraint checking negligible.
- **Design Motivation**: The greedy algorithm achieves a $1/2$ approximation guarantee (Fisher et al.), while the continuous greedy achieves $(1-1/e)$ (Calinescu et al.). This reflects a neuro-symbolic division of labor—LLMs estimate utility (semantic), while matroids enforce feasibility (symbolic).

### Loss & Training

The utility function $U: 2^{\mathcal{C}_u} \to \mathbb{R}_{\geq 0}$ must satisfy normalization ($U(\emptyset)=0$), monotonicity, and submodularity. In practice, it can be estimated empirically by measuring downstream task performance on a benchmark using LLM-as-Judge. Weighted set coverage functions are used as canonical submodular functions in the simulation.

## Key Experimental Results

### Main Results

Simulation over 5,000 random problem instances (14 macro-facets, 4 partition matroid groups):

| Metric | Value |
|--------|-------|
| Mean greedy approximation ratio | 0.996 (95% CI: 0.995, 0.996) |
| Minimum approximation ratio | 0.911 |
| 5th percentile | 0.975 |
| Theoretical lower bound | 0.5 |

### Ablation Study

- The greedy algorithm maintains at least 97.5% of optimal utility in 95% of instances.
- The observed worst case (0.911) exceeds the theoretical lower bound (0.5) by more than 82%.
- The distribution is heavily left-skewed, with the vast majority of results concentrated near 1.0.

### Key Findings

1. **Greedy performance far exceeds theoretical guarantees in practice**: The mean approximation ratio of 0.996 is nearly equivalent to the optimal solution obtained by brute-force search.
2. **Laminar matroid modeling is effective**: Common hierarchical constraints and mutual exclusion rules can be elegantly unified within the matroid framework.
3. **Partition matroid is an important special case**: Simple categorical quotas ("select $k$ from each category") are automatically subsumed within the laminar matroid framework.
4. **The neuro-symbolic division of labor is necessary**: LLMs excel at semantic evaluation but are unreliable for strict logical constraint enforcement (potentially exhibiting "hallucinated compliance"); delegating hard constraints to the symbolic layer is the correct design choice.

## Highlights & Insights

- **Theoretical elegance**: The derivation chain from SCC condensation → laminar matroid → submodular maximization is clear and complete, with explicit mathematical guarantees at each step.
- **Practical significance**: Provides a guaranteed solution for data minimization in LLM personalization—maximizing personalization effectiveness while satisfying privacy and quota constraints.
- **The treatment of "logical dependencies" is particularly elegant**: Constraints are absorbed into the ground set through compilation, allowing subsequent optimization to proceed entirely in a "clean" space.
- **The distinction between pre-closure and post-closure has practical value**: The paper explicitly identifies that matroid structure no longer holds in the post-closure regime, demarcating the applicable boundaries of the approach.

## Limitations & Future Work

1. **Purely theoretical work**: Only synthetic simulation experiments are provided; end-to-end validation on real LLM personalization tasks is absent.
2. **Small ground set size**: The simulation uses only 14 macro-facets, where brute-force search is feasible; scalability to large-scale scenarios is unverified.
3. **Strong utility function assumptions**: Whether submodularity and monotonicity strictly hold in LLM settings requires further empirical support.
4. **Post-closure quotas are left as future work**: The current framework does not apply when quotas must be checked against the count of actual facets after expansion.
5. **Static model**: Dynamic personalization scenarios involving temporally evolving user preferences or context-dependent interests are not considered.

## Related Work & Insights

- **Classical submodular optimization theory**: Nemhauser et al. (1978) greedy $1/2$ guarantee; Calinescu et al. (2011) continuous greedy $(1-1/e)$ guarantee—this paper instantiates these classical results in the LLM personalization setting.
- **SCC condensation**: Tarjan's (1972) classical algorithm is used for dependency compilation.
- **LLM personalization**: Existing methods mostly employ simple budgets or partition quotas, lacking the ability to handle hierarchical structures and explicit dependency closures.
- **Insight**: When facing data selection problems with complex constraints, it is worthwhile to investigate whether the underlying structure admits a matroid characterization—if so, the rich toolbox of combinatorial optimization can be directly applied.

## Rating

⭐⭐⭐⭐

The theoretical contribution is solid—formalizing practical constraints in LLM personalization as matroids is a valuable innovation, with clear and complete proofs. However, the experiments are limited to synthetic simulations, and the lack of real-world validation is the primary weakness. The theory-to-practice gap in this paper is substantial and requires follow-up work to bridge.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ShapeCraft: LLM Agents for Structured, Textured and Interactive 3D Modeling](../../NeurIPS2025/llm_agent/shapecraft_llm_agents_for_structured_textured_and_interactive_3d_modeling.md)
- [\[AAAI 2026\] BayesAgent: Bayesian Agentic Reasoning Under Uncertainty via Verbalized Probabilistic Graphical Modeling](bayesagent_bayesian_agentic_reasoning_under_uncertainty_via_.md)
- [\[ICML 2026\] A Minimal Agent for Automated Theorem Proving](../../ICML2026/llm_agent/a_minimal_agent_for_automated_theorem_proving.md)
- [\[ACL 2026\] CodeStruct: Code Agents over Structured Action Spaces](../../ACL2026/llm_agent/codestruct_code_agents_over_structured_action_spaces.md)
- [\[ACL 2026\] CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](../../ACL2026/llm_agent/coevolve_training_llm_agents_via_agent-data_mutual_evolution.md)

</div>

<!-- RELATED:END -->
