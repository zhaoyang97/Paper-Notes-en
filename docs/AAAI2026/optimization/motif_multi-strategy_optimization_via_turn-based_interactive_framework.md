---
title: >-
  [Paper Note] MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework
description: >-
  [AAAI 2026][Optimization][Automatic Heuristic Design] This paper proposes MOTIF, a framework that models solver design as a multi-strategy optimization problem. Through a turn-based competitive mechanism involving two LL…
tags:
  - "AAAI 2026"
  - "Optimization"
  - "Automatic Heuristic Design"
  - "LLM"
  - "Combinatorial Optimization"
  - "Monte Carlo Tree Search"
  - "Multi-strategy Optimization"
date: 2026-05-08
content_hash: 08a60d4da846f480
---

# MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework

**Conference**: AAAI 2026
**arXiv**: [2508.03929](https://arxiv.org/abs/2508.03929)  
**Code**: [https://github.com/HaiAu2501/MOTIF](https://github.com/HaiAu2501/MOTIF)  
**Area**: Optimization
**Keywords**: Automatic Heuristic Design, LLM, Combinatorial Optimization, Monte Carlo Tree Search, Multi-strategy Optimization

## TL;DR

This paper proposes MOTIF, a framework that models solver design as a multi-strategy optimization problem. Through a turn-based competitive mechanism involving two LLM agents guided by Monte Carlo Tree Search (MCTS), MOTIF jointly optimizes multiple interdependent algorithmic components within combinatorial optimization solvers, consistently outperforming existing methods across TSP, CVRP, BPP, and other combinatorial optimization domains.

## Background & Motivation

### Limitations of Automatic Heuristic Design (AHD)

Combinatorial optimization problems (COPs) such as TSP and CVRP are NP-hard, and traditional solvers rely on expert-designed heuristics that are costly to develop and lack cross-problem flexibility. In recent years, Language Hyper-Heuristics (LHH) have leveraged the code generation and reasoning capabilities of LLMs to automate heuristic design, achieving notable progress. Representative methods include:

- **EoH**: Combines evolutionary algorithms with LLMs to evolve code
- **ReEvo**: Integrates evolutionary search with self-reflection
- **HSEvo**: Introduces diversity metrics and harmony search
- **MCTS-AHD**: Employs MCTS with progressive expansion to explore the heuristic space

### Two Key Limitations of Existing Methods

**Single-strategy limitation**: Existing methods optimize only a single component of the solver (typically the heuristic scoring function), overlooking opportunities for innovation in other components. The combination of weaker components may yield better overall performance than optimizing a single strong component in isolation.

**Shallow feedback mechanisms**: Although self-reflection mechanisms exist (e.g., ReEvo), they typically rely on static comparisons or surface-level summaries, making it difficult to capture implicit incompatibilities or potential synergies between components. The feedback received by LLMs is insufficient to guide meaningful, deep-level modifications.

### Core Innovation

MOTIF extends solver design from "single heuristic function optimization" to "joint multi-strategy optimization," and introduces a turn-based self-play mechanism inspired by game theory. Two LLM agents alternately generate and refine strategies, with competitive pressure and emergent cooperation driving the discovery of superior solutions.

## Method

### Overall Architecture

MOTIF adopts a two-phase optimization pipeline:

1. **Component-wise Competition**: The solver is decomposed into $K$ independent strategies, each optimized separately.
2. **System-aware Refinement**: Each strategy is re-optimized with the full global system configuration visible, promoting inter-component synergy.

### Key Designs

#### 1. **Multi-strategy Optimization Problem Formulation**: Unifying solver design as a combinatorial optimization framework

**Function**: Jointly optimizes $K$ strategies $\Pi = (\pi_1, \pi_2, \ldots, \pi_K)$ within solver $s$ to minimize the expected optimization objective:

$$\Pi^* = \arg\min_\Pi \mathbb{E}_{\mathbf{x} \sim \mathcal{X}_d}[F_d(\mathbf{x} | \Pi)]$$

**Core definitions**:
- **Strategy**: A replaceable component within the solver, such as a heuristic scoring rule, construction strategy, neighborhood move, or penalty update mechanism.
- **Strategy space $\mathcal{S}_\pi$**: The set of all valid implementations of a given strategy class, encompassing functional variants and parameterization schemes.
- Strategies within the same strategy space share function signatures and optimization objectives, ensuring interoperability within the solver.

**Design Motivation**: This formulation is more globally informed than isolated single-heuristic optimization, enabling the unlocking of inter-component synergies.

#### 2. **Competitive Monte Carlo Tree Search (CMCTS)**: Turn-based dual-agent search architecture

**Function**: A competitive tree is constructed for each strategy, with two LLM agents taking turns to generate new strategy implementations via three competitive operators.

**Search procedure** (standard four-step MCTS):

**(1) Selection**: Traverses from the root along child nodes created by the current player, using the UCB formula to balance exploration and exploitation:
$$\text{UCB}(o, \pi) = \frac{Q(o, \pi)}{N(o, \pi)} + C_{in}\sqrt{\frac{\ln N(\pi)}{N(o, \pi)}}$$

**(2) Expansion**: Selects an operator and invokes the LLM to generate a new implementation:
$$\pi' \leftarrow \mathcal{L}(\text{Prompt}(o; \{\pi^{(p)}, \pi^{(\neg p)}\}; \{\mathcal{H}^{(p)}, \mathcal{H}^{(\neg p)}\}; \mathcal{B}))$$
The prompt contains: operator type, both agents' current implementations, both agents' histories, and the current baseline.

**(3) Evaluation**: The new implementation is inserted into the system to replace the corresponding strategy, and improvement over the baseline is computed.

**(4) Backpropagation**: The Q-value jointly considers absolute improvement and relative competitive gain:
$$Q^{(p)} = \lambda \cdot \sigma(I^{(p)}) + (1-\lambda) \cdot \sigma(I^{(p)} - I^{(\neg p)})$$

#### 3. **Three Competitive Operators**: Covering the full spectrum from adversarial to innovative behavior

| Operator | Strategy | Behavior |
|----------|----------|----------|
| **Counter** | Designs adversarial improvements targeting opponent weaknesses | Exploits opponent inefficiencies or limitations |
| **Learning** | Integrates strengths of the opponent's solution into one's own | Constructive integration |
| **Innovation** | Ignores prior solutions and proposes an entirely new approach | Pure exploratory invention |

Together, the three operators cover a rich behavioral spectrum ranging from adversarial exploitation to constructive integration and exploratory invention.

#### 4. **Outer Controller and Dynamic Baseline**

- **Outer Controller**: Uses a UCB rule to select which strategy tree to optimize in the next round:
$$k = \arg\max_{1 \le k \le K}\left(\frac{R_j}{N_j} + C_{out}\sqrt{\frac{\ln\sum_i N_i}{N_j}}\right)$$

- **Dynamic Baseline**: The initial baseline is a manually designed or warm-started optimal implementation. It is updated whenever an agent produces a strictly superior implementation. Each agent competes against the other player's most recent best solution.

#### 5. **System-aware Refinement**

In the second phase, each strategy is re-optimized with the full system visible. The prompt includes the complete system configuration, global baseline cost, and historical search trajectories:
$$\pi_k' \leftarrow \mathcal{L}(\text{Prompt}(\Pi; \{\pi_k^{(p)}, \pi_k^{(\neg p)}\}; \{\mathcal{H}^{(p)}, \mathcal{H}^{(\neg p)}\}))$$

Updates are accepted only when improvements exceed both the baseline and the opponent's best result.

### Loss & Training

- Uses `gpt-4o-mini-2024-07-18` as the LLM backend for both agents.
- Lightweight CoT: each response includes three fields — `reasoning` (≤5 sentences), `code` (Python), and `summary` (change description).
- Optimization is performed on a lightweight training set; final evaluation is conducted on a held-out test set.
- Structured output format is used to ensure reliable parsing.

## Key Experimental Results

### Main Results

#### Single-strategy Optimization: TSP under GLS Framework

| Method | TSP50 | TSP100 | TSP200 | TSP500 |
|--------|-------|--------|--------|--------|
| EoH | 0.0000 | 0.0012 | 0.0639 | 0.5796 |
| ReEvo | 0.0000 | 0.0335 | 0.2081 | 0.7918 |
| HSEvo | 0.0108 | 0.3095 | 1.1254 | 2.4593 |
| MCTS-AHD | 0.0000 | 0.0024 | 0.0652 | 0.5611 |
| **MOTIF** | **0.0000** | **0.0007** | **0.0610** | **0.5577** |

Even in the single-strategy setting, MOTIF substantially outperforms existing methods.

#### Multi-strategy Optimization: Deconstruction-then-Repair Framework

| Strategy Combination | TSP50 | TSP100 | TSP200 | CVRP50 | CVRP100 | CVRP200 | BPP100 | BPP200 | BPP300 |
|----------------------|-------|--------|--------|--------|---------|---------|--------|--------|--------|
| $\pi_1$ | 0.39 | 0.81 | 0.72 | 1.47 | 2.96 | 2.30 | 3.17 | 4.82 | 4.77 |
| $\pi_2$ | 3.13 | 6.24 | 8.18 | 4.27 | 7.04 | 6.85 | 23.19 | 24.42 | 25.00 |
| $\pi_3$ | 3.88 | 7.91 | 11.35 | 6.34 | 5.07 | 4.28 | 12.70 | 12.84 | 12.15 |
| $(\pi_1,\pi_2)$ | 2.58 | 5.04 | 5.84 | 6.06 | 7.94 | 8.85 | 23.15 | 24.40 | 24.98 |
| $(\pi_2,\pi_3)$ | 3.83 | 7.97 | 11.59 | 10.64 | 12.31 | 11.71 | 23.05 | 24.32 | 24.89 |
| **$(\pi_1,\pi_2,\pi_3)$** | **3.88** | **7.96** | **11.65** | **10.98** | **12.84** | **13.06** | **23.94** | **25.02** | **25.41** |

Joint optimization of three strategy components consistently outperforms single-strategy optimization across TSP, CVRP, and BPP, with a gain of up to 13.06% on CVRP200.

### Ablation Study

| Configuration | ACO-Train | ACO-Test | GLS-Train | GLS-Test |
|---------------|-----------|----------|-----------|----------|
| w/o Outer Controller | 0.74 | 5.61 | 0.81 | 2.88 |
| w/o Dynamic Baseline | 1.55 | **9.50** | 0.62 | 2.94 |
| w/o Final Round | 1.26 | 5.72 | 0.39 | 2.82 |
| w/o Reasoning | 1.63 | 7.88 | 0.80 | 3.05 |
| w/o Counter | 1.42 | 6.71 | 0.64 | 3.20 |
| w/o Learning | 0.91 | **9.59** | 0.62 | 3.00 |
| w/o Innovation | 1.73 | 6.65 | 0.42 | 2.86 |
| **MOTIF (original)** | **0.80** | **5.21** | **0.34** | **2.73** |

**Key Findings**:
- Removing the dynamic baseline causes the most severe performance degradation, particularly at test time.
- Removing reasoning also has a significant negative impact.
- All three operators contribute meaningfully; Learning has the greatest effect on ACO performance.

### Diversity Analysis

| Operator | Success Rate ↑ | Novelty ↑ | Silhouette Score ↑ |
|----------|---------------|-----------|-------------------|
| Counter | 93% | 0.0136 | 0.5121 |
| Learning | 92% | 0.0118 | **0.5325** |
| **Innovation** | **97%** | **0.0175** | 0.4793 |

Innovation achieves the highest novelty but the lowest consistency; Learning yields the lowest novelty but the highest consistency; Counter exhibits the most balanced behavior.

### Key Findings

1. Joint multi-strategy optimization consistently outperforms single-strategy optimization, with clear synergistic effects among components.
2. Competitive pressure drives both players to mutually improve, with their performance curves highly aligned.
3. LLMs are capable not only of tuning individual heuristics but also of co-designing complete algorithmic pipelines.
4. Even with the relatively limited coding capacity of `gpt-4o-mini`, structured prompting yields high-quality results.

## Highlights & Insights

1. **Novel problem formulation for multi-strategy optimization**: Extends solver design from single-component to joint multi-component optimization, establishing a new paradigm.
2. **Game-theoretically inspired search**: The turn-based dual-agent competition represents the first application of self-play to combinatorial optimization solver design, leveraging adversarial pressure and in-context learning to achieve emergent self-improvement.
3. **Three-operator design**: Counter/Learning/Innovation cover the complete behavioral spectrum, each with a clearly defined role.
4. **Strong practical applicability**: Validated across three distinct frameworks — ACO, GLS, and Deconstruction-Repair — covering five COPs including TSP, CVRP, and BPP.

## Limitations & Future Work

1. Performance depends on the code generation quality of the LLM; stronger models (e.g., GPT-4o) may yield further improvements.
2. The dual-agent interaction increases the number of LLM calls, resulting in higher computational cost.
3. The current number of strategies $K$ is small (2–3); scalability to larger numbers of strategies remains to be verified.
4. The system-aware refinement phase uses a fixed rather than dynamic baseline, which may limit update frequency.
5. Extensions to more than two agents (3+ players) are worth exploring.

## Related Work & Insights

- **ReEvo / MCTS-AHD**: Single-strategy LHH baselines; MOTIF generalizes upon these by extending to multi-strategy optimization.
- **SPAG / CDG**: Applications of LLM self-play to reasoning enhancement and deception detection; MOTIF is the first to apply this paradigm to algorithm design.
- **AlphaGo / AlphaZero**: Successful exemplars of MCTS combined with self-play; MOTIF transfers these ideas to the domain of code generation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The combination of multi-strategy optimization formulation and game-theoretic search is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three frameworks, five COPs, detailed ablation studies, and diversity analysis.
- Writing Quality: ⭐⭐⭐⭐ — The framework is clearly described, though the density of mathematical notation is high.
- Value: ⭐⭐⭐⭐⭐ — Opens a new multi-strategy direction for LLM-driven algorithm design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)
- [\[AAAI 2026\] Parametrized Multi-Agent Routing via Deep Attention Models](parametrized_multi-agent_routing_via_deep_attention_models.md)
- [\[NeurIPS 2025\] AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](../../NeurIPS2025/optimization/autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)
- [\[NeurIPS 2025\] MOBO-OSD: Batch Multi-Objective Bayesian Optimization via Orthogonal Search Directions](../../NeurIPS2025/optimization/mobo-osd_batch_multi-objective_bayesian_optimization_via_orthogonal_search_direc.md)
- [\[AAAI 2026\] Co-Layout: LLM-driven Co-optimization for Interior Layout](co-layout_llm-driven_co-optimization_for_interior_layout.md)

</div>

<!-- RELATED:END -->
