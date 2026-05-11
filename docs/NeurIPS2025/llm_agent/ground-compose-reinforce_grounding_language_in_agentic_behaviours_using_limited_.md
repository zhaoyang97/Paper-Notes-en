---
title: >-
  [Paper Note] Ground-Compose-Reinforce: Grounding Language in Agentic Behaviours using Limited Data
description: >-
  [NeurIPS 2025][LLM Agent][Language Grounding] This paper proposes Ground-Compose-Reinforce (GCR), an end-to-end neuro-symbolic framework that learns the grounding semantics of atomic propositions from a small number of a…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "Language Grounding"
  - "Reward Machine"
  - "Compositional Generalization"
  - "Neuro-Symbolic"
  - "Reward Shaping"
date: 2026-05-08
content_hash: 720630bf4b2e35e0
---

# Ground-Compose-Reinforce: Grounding Language in Agentic Behaviours using Limited Data

**Conference**: NeurIPS 2025
**arXiv**: [2507.10741](https://arxiv.org/abs/2507.10741)
**Code**: [github](https://github.com/andrewli77/ground-compose-reinforce)
**Area**: LLM Agent / Reinforcement Learning
**Keywords**: Language Grounding, Reward Machine, Compositional Generalization, Neuro-Symbolic, Reward Shaping

## TL;DR
This paper proposes Ground-Compose-Reinforce (GCR), an end-to-end neuro-symbolic framework that learns the grounding semantics of atomic propositions from a small number of annotated trajectories (only 350), composes them into complex task specifications via Reward Machines, and trains an RL agent using self-generated dense rewards — eliciting out-of-distribution complex behaviours without any hand-crafted reward functions.

## Background & Motivation

**Background**: Enabling agents to execute tasks via language instructions requires solving the "language grounding" problem — associating language with perception and action. Two main paradigms exist: (a) hand-engineering domain-specific reward functions or success detectors; (b) training on massive language–trajectory paired data (e.g., π0, RT-2).

**Limitations of Prior Work**:
   - Hand-engineering approaches are difficult to scale in complex or non-simulated settings and generalize poorly.
   - Data-driven approaches require enormous annotated trajectory datasets (e.g., hundreds of thousands of trajectories) and tend to fail on complex or OOD tasks in data-scarce settings (e.g., robotic manipulation).
   - Non-compositional methods cannot exploit the compositionality of language — learning "pick up the red block" and "open the drawer" does not automatically yield "place the red block in the drawer."

**Key Challenge**: How can one achieve an end-to-end mapping from high-level task specifications to executable behaviours with very limited annotated data, while generalizing to compositional tasks never seen during training?

**Goal**: To learn language grounding from a small number of annotated trajectories — without hand-crafted reward functions or large-scale data — and achieve OOD generalization through composition.

**Key Insight**: The paper exploits the natural compositional structure of Reward Machines (automaton-based task specification languages) — first learning the semantics of atomic propositions, then expressing arbitrarily complex tasks via logical composition, and finally applying RL with self-generated rewards.

**Core Idea**: Decompose language grounding into three steps — "learn atomic concepts + logical composition + RL fine-tuning" — leveraging compositionality rather than massive data to achieve generalization.

## Method

### Overall Architecture
GCR consists of two phases: pre-training and behaviour elicitation.
- **Pre-training (Ground)**: Learn a labelling function $\hat{\mathcal{L}}(s)$ (mapping environment states to proposition truth values) and Primitive Value Functions (PVFs, progress signals for each atomic proposition) from a small annotated dataset $\mathcal{D}$.
- **Behaviour Elicitation (Compose + Reinforce)**: Given a Reward Machine task specification $\mathcal{R}$, automatically generate reward signals and RM state tracking using learned $\hat{\mathcal{L}}$, compose PVFs for reward shaping, and train the policy with PPO.

### Key Designs

1. **Atomic Proposition Grounding (Ground)**:

    - **Function**: Ground abstract propositional symbols (e.g., "the robot is holding the red block") to environment states.
    - **Mechanism**: Train a binary classifier $\hat{\mathcal{L}}: \mathcal{S} \to 2^{\mathcal{AP}}$ on annotated trajectories $\mathcal{D}$ to predict the truth value of each proposition at each state. Once trained, the agent can freely query this function to self-assess task progress.
    - **Design Motivation**: Reformulating grounding as a standard classification problem drastically reduces data requirements (350 trajectories suffice).

2. **Compositional Reward Shaping (Compose)**:

    - **Function**: Address the *propositional sparsity* problem — when target propositions are rarely satisfied under random exploration (e.g., "pick up the block"), the agent receives almost no reward signal.
    - **Mechanism**:
        - During pre-training, learn $2|\mathcal{AP}|$ **Primitive Value Functions (PVFs)**, each estimating the optimal value function $V^*_{\Diamond x}(s)$ for satisfying a single proposition (or its negation).
        - For any given Reward Machine task, decompose it into RM transitions → logical sub-tasks → DNF → atomic proposition chains.
        - Compose PVFs using fuzzy logic semantics: $\max$ for disjunction, $\min$ for conjunction.
        - Obtain an approximate OVF for any RM task: $V^*_{\mathcal{R}}(s,u) \approx \max_{\langle u,u',\varphi,r\rangle} [V^*_{\Diamond\varphi}(s) \cdot (r + \gamma v^*_{\mathcal{R}}(u'))]$
        - Apply this OVF as potential-based reward shaping.
    - **Design Motivation**: Starting from $2|\mathcal{AP}|$ PVFs, the approach can compose value function estimates for an exponential number ($2^{2^n}$) of logical tasks and infinitely many RM tasks, achieving both efficiency and generalization.

3. **RL Fine-Tuning (Reinforce)**:

    - **Function**: Train the policy for a given RM task using self-generated reward signals and reward shaping.
    - **Mechanism**: Embed the learned $\hat{\mathcal{L}}$ into the RL loop — at each timestep, predict proposition truth values to drive RM state transitions and generate rewards. The policy takes $(s, u)$ as input (where $u$ is the RM state) and is optimized with PPO.
    - **Design Motivation**: Self-supervised RL requires no external oracle; the RM state compactly encodes history, avoiding the complexity of history-conditioned policies.

### Loss & Training
- Labelling function $\hat{\mathcal{L}}$: Standard binary cross-entropy classification loss.
- PVFs: Offline RL training, optimizing for discounted return to proposition-satisfying states.
- Policy: PPO with potential-based reward shaping (which preserves the set of optimal policies).

## Key Experimental Results

### Main Results

| Task | GCR (Ours) | LTL-BC | Bespoke RM | Bespoke BC | Max Achievable |
|------|-----------|--------|-----------|-----------|---------------|
| GeoGrid-Sequence | **1.00±0.00** | 0.04 | 0 | 0.05 | 1 |
| GeoGrid-Loop | **5.36±0.08** | 0.03 | 0 | 0.04 | 5.36 |
| GeoGrid-Logic | **0.94±0.01** | 0 | 0 | 0 | 1 |
| GeoGrid-Safety | **1.00±0.00** | -0.84 | -0.14 | -0.85 | 1 |
| DrawerWorld-Hold-Red-Box | **1538±130** | 0 | 0 | 0 | 1538 |
| DrawerWorld-Pickup-Each-Box | **1.00±0.00** | 0 | 0 | 0 | 1 |
| DrawerWorld-Show-Green-Box | **0.61±0.06** | 0 | 0 | 0 | — |

### Ablation Study (Reward Shaping)

| Configuration | Hold-Red-Box | Pickup-Each-Box | Show-Green-Box |
|---------------|-------------|----------------|---------------|
| GCR (full, Ours) | **1538** | **1.00** | **0.61** |
| GCR No RS | 0 | 0 | 0 |
| GCR High-Level RS | 0 | 0 | 0 |

### Key Findings
- **Compositionality is critical**: GCR substantially outperforms all non-compositional baselines on every task, including privileged Bespoke methods. Non-compositional methods fail completely under limited data (returning 0 on all DrawerWorld tasks).
- **350 trajectories are sufficient**: DrawerWorld uses only 350 manually collected trajectories, with pre-training data covering only single-object manipulation, yet the approach generalizes to multi-object compositional tasks.
- **Reward shaping is decisive**: In DrawerWorld, GCR without compositional reward shaping completely fails to learn (return = 0), as propositional sparsity renders RL signals nearly absent.
- Bespoke Reward Models exhibit severe reward hacking — the learned reward model is inconsistent with the true reward.

## Highlights & Insights
- The **"bottom-up" compositional grounding strategy** is remarkably data-efficient: learning the semantics of atomic concepts first, then expressing complex tasks via logical composition, is orders of magnitude more data-efficient than end-to-end learning of language-to-behaviour mappings. The gap between 350 trajectories and the millions required by VLA models is striking.
- The technique of **composing value functions with fuzzy logic** is elegant: $\max$ for disjunction, $\min$ for conjunction, composing $2|\mathcal{AP}|$ PVFs into value function estimates for $2^{2^n}$ logical tasks. This idea is transferable to any setting requiring estimation of compositional task value from atomic skill estimates.
- **Reward Machines as a bridge between LLMs and RL**: natural language → (LLM auto-formalization) → Reward Machine → RL. This pipeline combines LLMs' language understanding with RL's environment interaction capacity.
- **Propositional sparsity** is an underappreciated problem: when propositions such as "pick up the object" are nearly impossible to satisfy under random exploration, conventional RM-based RL methods fail entirely. The paper's compositional reward shaping is currently the only effective solution.

## Limitations & Future Work
- **Proposition set must be predefined**: The current framework requires $\mathcal{AP}$ to be fixed in advance with annotated data, and cannot dynamically extend the proposition set. Future work could explore using natural language to describe propositions and leveraging VLMs for automatic truth-value assessment.
- **DrawerWorld is limited in scale**: The experimental environment contains only 2 drawers and 3 blocks, yielding a proposition space of 11. Error accumulation from compositional approximation under larger, more complex real-world settings (with potentially hundreds of propositions) remains to be validated.
- **PVF approximation error**: Fuzzy logic composition is only approximate; $\min$ and $\max$ operations may severely underestimate or overestimate value in non-convex value landscapes. The paper does not quantify the impact of this error.
- **Auto-formalization validated only on GeoGrid**: The pipeline for converting natural language to RM via LLM is validated only on a simple gridworld; formalization accuracy in more complex settings warrants further investigation.
- **Future directions**: Replacing the labelling function $\hat{\mathcal{L}}$ with a VLM could enable a zero-annotation end-to-end pipeline; hierarchical RM structures could be explored for longer-horizon tasks.

## Related Work & Insights
- **vs. π0 / RT-2 (VLA models)**: VLA models require massive language–trajectory pairs; this paper requires only 350 task-unannotated trajectories. The trade-off is a predefined proposition set and RM specification, but the advantage in data-scarce settings is significant.
- **vs. BabyAI**: BabyAI relies on hand-crafted reward functions; GCR learns grounding from data and generalizes to OOD tasks.
- **vs. Nangue Tasse et al. (zero-shot composition)**: Their approach assumes all tasks can be captured by a finite set of goal states; GCR handles more general RM task structures and does not require goal states to be known in advance.

## Rating
- Novelty: ⭐⭐⭐⭐ The Ground-Compose-Reinforce three-step pipeline is clean and elegant; the compositional reward shaping technique has practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two domains, multiple tasks, thorough ablations, though environments are limited in scale.
- Writing Quality: ⭐⭐⭐⭐⭐ Writing is clear and fluent, formalization is rigorous, and a running example is woven throughout.
- Value: ⭐⭐⭐⭐ The compositional generalization approach for data-scarce settings provides strong practical inspiration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](../../CVPR2026/llm_agent/towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[NeurIPS 2025\] LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers](lc-opt_benchmarking_reinforcement_learning_and_agentic_ai_for_end-to-end_liquid_.md)
- [\[NeurIPS 2025\] Enhancing Demand-Oriented Regionalization with Agentic AI and Local Heterogeneous Data for Adaptation Planning](enhancing_demand-oriented_regionalization_with_agentic_ai_and_local_heterogeneou.md)
- [\[NeurIPS 2025\] R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)
- [\[NeurIPS 2025\] Are Large Language Models Sensitive to the Motives Behind Communication?](are_large_language_models_sensitive_to_the_motives_behind_communication.md)

</div>

<!-- RELATED:END -->
