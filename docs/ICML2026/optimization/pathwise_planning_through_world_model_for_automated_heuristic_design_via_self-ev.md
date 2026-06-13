---
title: >-
  [Paper Note] PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs
description: >-
  [ICML 2026][Optimization][Automated Heuristic Design] PathWise reformulates LLM-based Automated Heuristic Design (AHD) as a sequential decision process unfolding on an "entailment graph." Coordinated by four LLM agents—P…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Automated Heuristic Design"
  - "Multi-agent LLM"
  - "Entailment Graph"
  - "World Model"
  - "Combinatorial Optimization"
date: 2026-05-08
content_hash: 732d45be9f4cb03a
---

# PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs

**Conference**: ICML 2026  
**arXiv**: [2601.20539](https://arxiv.org/abs/2601.20539)  
**Code**: To be confirmed  
**Area**: Optimization / LLMs for Combinatorial Optimization / Automated Heuristic Design  
**Keywords**: Automated Heuristic Design, Multi-agent LLM, Entailment Graph, World Model, Combinatorial Optimization

## TL;DR
PathWise reformulates LLM-based Automated Heuristic Design (AHD) as a sequential decision process unfolding on an "entailment graph." Coordinated by four LLM agents—Policy, World Model, and dual Critics—the framework replaces gradient updates with reflections. It outperforms mainstream baselines such as FunSearch, EoH, ReEvo, HSEvo, and MCTS-AHD on TSP, CVRP, KP, and Bin Packing using only 50% of the evaluation budget.

## Background & Motivation

**Background**: The goal of Automated Heuristic Design (AHD) is to automatically discover high-quality heuristic programs $h:\mathcal{X}\to\mathcal{S}$ for Combinatorial Optimization Problems (COPs). Traditional approaches rely on Genetic Programming (GP) with manual operators on syntax trees. Recent works like FunSearch, EoH, ReEvo, and HSEvo embed LLMs as "high-level operators" in evolutionary loops, while MCTS-AHD organizes the generation into a Monte Carlo tree.

**Limitations of Prior Work**: Population-based methods use fixed selection/replacement rules, often prematurely discarding promising solutions and converging early. Tree-based methods, though hierarchical, rely on pure performance statistics like UCT for node selection and expansion, failing to capture the semantic relationships between heuristics. Both routes treat generation steps as independent or weakly coupled samples with static prompt templates, leading to "superficial rediscovery," repeated sampling of similar heuristics, and significant waste of LLM budgets.

**Key Challenge**: Search is inherently a reasoning process with memory. Semantic information about a heuristic's quality, its evolutionary lineage, and effective edit patterns should be explicitly modeled and reused. However, existing frameworks lack such state representations, fail to decouple "high-level strategy" from "low-level code synthesis," and do not route critic feedback to specific roles.

**Goal**: (1) Design a structured state representation for LLM-AHD to compress derivation history into search states; (2) Decouple and coordinate roles for strategic planning, code synthesis, and reflective evaluation; (3) Introduce diversity at the prompt level to allow critics to observe "contrasts."

**Key Insight**: Borrowing the concept of textual entailment graphs, each candidate heuristic is treated as a node, and the derivation of a child node from parents via natural language reasoning is treated as a labeled directed edge. This graph serves as both a compact memory of search trajectories and a readable context for LLMs. Combined with the "LLM as a World Model" perspective (Hao et al., 2023), an RL-like interface (Policy/World Model/Critic) can be implemented without parameter updates, iterating solely through natural language reflection.

**Core Idea**: Replace populations or MCTS trees with an entailment graph as the "State." Use a Policy LLM to sample high-level evolutionary actions (parent selection + derivation rationale), a World Model LLM to ground actions into code, and two Critic LLMs to route results back to their respective roles for "gradient-free" self-evolution.

## Method

PathWise treats heuristic discovery as an MDP $(\mathbb{S},\mathbb{A},\mathbb{T},\mathbb{R})$: state $s_t$ is the current entailment graph plus the frontier; action $a_t=(S,\kappa)$ consists of selected parent nodes $S\subseteq s_t$ and a natural language derivation rationale $\kappa$; transition $\mathbb{T}$ involves the World Model compiling $(S,\kappa)$ into a new heuristic and updating the graph; reward $\mathbb{R}$ is the negative cost of the heuristic on dataset $\mathcal{D}$: $P(h;\mathcal{D})=\mathbb{E}_{x\sim\mathcal{D}}[-f(h(x))]$. This MDP serves as a structural backbone where reinforcement is achieved via natural language reflections from critics.

### Overall Architecture

PathWise operates on two timescales: an outer iteration $r$ maintaining a root population $\mathcal{P}_r$ of size $\mathcal{N}_p$, and an inner iteration $t$ that expands a local entailment graph $G_t=(V_t,E_t)$ rooted at $\mathcal{P}_r$. An inner step involves four LLM roles: the Policy agent $\boldsymbol{\pi}_p$ samples $N_a$ candidate actions; the World Model agent $\boldsymbol{\pi}_{wm}$ generates $N_w$ code rollouts for each action; the Evaluator runs these rollouts to select the best new node $v_\star = \arg\max_{i,j} P(\hat{h}^{(i,j)};\mathcal{D})$; finally, the Policy Critic $\boldsymbol{\pi}_{p\_critic}$ and World Model Critic $\boldsymbol{\pi}_{wm\_critic}$ generate routed reflections for future steps. The state update rule $s_{t+1}=(s_t\cup\{v_\star\})\setminus(S^{(i_\star)}\setminus\{v^\star\})$ prunes used parents while retaining the global best $v^\star$ to balance exploration and exploitation.

Each node is a quintuple $(h,\kappa,d,P(h;\mathcal{D}),\mathrm{PM})$: code $h$, derivation rationale $\kappa$, natural language description $d$, performance $P$, and parent metadata $\mathrm{PM} = \{(d_k,P(h_k;\mathcal{D})) \mid v_k \in S\}$. Using metadata instead of full parent code prevents prompt context explosion.

### Key Designs

1.  **Entailment Graph as Stateful Search Memory**:
    *   **Function**: Compresses the search trajectory into a readable graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$, allowing the LLM to perceive how solutions evolved from ancestors and compare performance across branches.
    *   **Mechanism**: Each edge $S\xRightarrow{\kappa}v_\star$ encodes both parents and rationale, turning evolutionary steps into logic-like entailment steps. Pruning ensures the frontier remains manageable while preserving the optima.
    *   **Design Motivation**: Unlike populations that lose intermediates or MCTS trees that lack semantics, entailment graphs provide both "compression" and "semantics," basing future decisions on derivation path effectiveness rather than simple visit counts.

2.  **Bi-level Policy/World Model Decomposition**:
    *   **Function**: Separates "high-level strategy" from "low-level synthesis" to avoid overloading a single prompt with both logical tricks and syntax requirements.
    *   **Mechanism**: The Policy agent samples $N_a$ actions $a_t^{(i)}=(S^{(i)},\kappa^{(i)})$ based on state and current reflections. The World Model generates $N_w$ rollouts $(\hat h^{(i,j)},\hat d^{(i,j)})$ to implement these actions independently.
    *   **Design Motivation**: Fixed operator templates limit creativity, while monolithic prompts often degrade code quality due to length. Decoupling allows for "semantic-level editing" and "implementation-level grounding," enabling separate diagnostics for failure modes.

3.  **Routed Reflection + Diversity Perturbation**:
    *   **Function**: Routes feedback to the appropriate role and prevents mode collapse within inner steps.
    *   **Mechanism**: The Policy Critic aggregates average rewards $R_p(a_t^{(i)})=\frac{1}{N_w}\sum_{j=1}^{N_w}P(\hat h^{(i,j)};\mathcal{D})$ to critique strategies. The World Model Critic compares the best and worst rollouts to refine implementation. Diversity is introduced via a decaying prompt perturbation library $\Phi$ with rate $\varepsilon(\ell) = \varepsilon^{init} + (\varepsilon^{final}-\varepsilon^{init})\cdot\ell/n_e$, and random state shuffling to eliminate position bias.
    *   **Design Motivation**: Mixed feedback confuses roles. Diversity perturbations are more precise than temperature sampling, adding noise to "exploration directions" while maintaining task precision.

### Loss & Training

PathWise is a training-free framework. It utilizes an outer iteration limit $\mathcal{N}_r$ and an inner evaluation budget $n_e$. PathWise achieves results superior to baselines using $n_e=1000$ while using only $n_e=500$.

## Key Experimental Results

### Main Results

Evaluated across 7 COPs including TSP (constructive), CVRP (ACO/Neural), KP, BPP, and JSSP. Comparisons were made against FunSearch, EoH, ReEvo, HSEvo, and MCTS-AHD using GPT-4o-mini and GPT-5-nano.

| Task / Test Set | Metric | LKH-3 / Optimal | MCTS-AHD ($n_e=1000$) | HSEvo ($n_e=1000$) | **PathWise ($n_e=500$, GPT-4o-mini)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TSP $N=50$ | Obj↓ / Gap | 5.687 / – | 6.358 / 11.80% | 6.429 / 13.05% | **6.245 / 9.81%** |
| TSP $N=100$ | Obj↓ / Gap | 7.767 / – | 8.839 / 13.80% | 8.903 / 14.63% | **8.758 / 12.76%** |
| TSP $N=200$ | Obj↓ / Gap | 10.709 / – | 12.403 / 15.82% | 12.359 / 15.41% | **12.276 / 14.63%** |
| KP $N=200,W=25$ | Obj↑ / Gap | 57.132 / – | 57.020 / 0.20% | – | **57.082 / 0.09%** |
| KP $N=500,W=25$ | Obj↑ / Gap | 90.763 / – | 89.061 / 1.88% | – | **90.719 / 0.05%** |

### Ablation Study

| Configuration | Observation | Explanation |
| :--- | :--- | :--- |
| Full PathWise | Fastest convergence, lowest variance | Integrated roles + routed reflection + perturbations |
| w/o Routed Reflection | Significant curve jitter, gap increases | Feedback confusion leads to unstable edits |
| w/o Diversity Perturbation | Policy selects same parents, rollouts collapse | Critic lacks contrast; reflections become repetitive |
| w/o State Shuffling | Bias towards early nodes in prompt | Confirms LLM position bias issues |

### Key Findings
*   PathWise consistently outperforms MCTS-AHD/HSEvo with half the budget, proving that stateful memory in entailment graphs is more efficient than UCT or population replacement.
*   Reduced variance in convergence curves indicates that role decomposition and routed reflection make the process resilient to initialization and sampling randomness.
*   The framework scales well across different LLM backbones, showing that it does not rely on specific prompting styles of a single model.

## Highlights & Insights
*   **Entailment Graphs for AHD**: Adapts a tool from NLP reasoning into evolutionary search, providing an interpretable and semantic search trajectory that aids critic agents.
*   **Training-free Actor-Critic Analogy**: Replaces RL components with LLMs and natural language, providing a template for iterative search in any "black-box" scenario where fine-tuning is impossible.
*   **Engineering Nuance**: Simple tricks like prompt perturbation decay and state shuffling solve significant "hidden bugs" like mode collapse and position bias with zero computational overhead.

## Limitations & Future Work
*   **Context Pressure**: The entailment graph grows linearly; though mitigated by metadata compression and pruning, context limits remain a challenge for extremely long searches.
*   **Evaluation Costs**: While COPs are cheap to evaluate, tasks with expensive evaluation steps might find the budget requirements prohibitive.
*   **LLM Reason Capability**: The quality of reflections depends on the backbone; weaker models may produce superficial critiques that narrow PathWise's performance lead.

## Related Work & Insights
*   **vs FunSearch / EoH**: Moves from fixed templates to self-generated derivation rationales $\kappa$ and explicit historical encoding.
*   **vs ReEvo**: Improves upon global shared reflection by introducing role-specific routed reflections.
*   **vs MCTS-AHD**: Replaces statistical tree expansion with semantic graph expansion, reducing per-step costs and improving convergence speed.

## Rating
*   Novelty: ⭐⭐⭐⭐ Structural innovation by combining entailment graphs and World Models.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of COPs and backbones with solid ablations.
*   Writing Quality: ⭐⭐⭐⭐ Clear mapping between MDP formalisms and agent coordination.
*   Value: ⭐⭐⭐⭐ Provides a reusable "Graph State + Role Splitting + Routed Reflection" template for training-free LLM search.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Automated Algorithm Design via Nevanlinna-Pick Interpolation](../../NeurIPS2025/optimization/automated_algorithm_design_via_nevanlinna-pick_interpolation.md)
- [\[ICML 2026\] Memory-Efficient LLM Pretraining via Minimalist Optimizer Design](memory-efficient_llm_pretraining_via_minimalist_optimizer_design.md)
- [\[ICML 2026\] Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](learning_a_zeroth-order_optimizer_for_fine-tuning_llms.md)
- [\[ICLR 2026\] FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](../../ICLR2026/optimization/frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)
- [\[ICML 2026\] TPV: Parameter Perturbations Through the Lens of Test Prediction Variance](tpv_parameter_perturbations_through_the_lens_of_test_prediction_variance.md)

</div>

<!-- RELATED:END -->
