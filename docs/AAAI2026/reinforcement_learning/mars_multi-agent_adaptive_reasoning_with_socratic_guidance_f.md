---
title: >-
  [Paper Note] MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization
description: >-
  [AAAI 2026][Reinforcement Learning][Automated Prompt Optimization] This paper proposes MARS, a five-agent framework for automated prompt optimization (APO): a Planner generates task-specific optimization trajectories…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Automated Prompt Optimization"
  - "Socratic Dialogue"
  - "POMDP"
  - "Teacher-Critic-Student"
  - "Pseudo-Gradient"
date: 2026-05-08
content_hash: 5e02f92f72821c5e
---

# MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization

**Conference**: AAAI 2026  
**arXiv**: [2503.16874](https://arxiv.org/abs/2503.16874)  
**Code**: [https://github.com/exoskeletonzj/MARS](https://github.com/exoskeletonzj/MARS)  
**Area**: Reinforcement Learning  
**Keywords**: Automated Prompt Optimization, Socratic Dialogue, POMDP, Teacher-Critic-Student, Pseudo-Gradient  

## TL;DR
This paper proposes MARS, a five-agent framework for automated prompt optimization (APO): a Planner generates task-specific optimization trajectories; a Teacher-Critic-Student triad conducts Socratic dialogue for iterative prompt refinement (simulating pseudo-gradient descent in text space); and a Target agent executes the prompt and provides feedback. The entire process is modeled as a POMDP. MARS outperforms the previous SOTA (PE2) by 6.04% on general tasks and 6.42% on domain-specific tasks across 17 datasets, requiring only 1-shot training data.

## Background & Motivation

**Background**: Automated prompt optimization (APO) aims to overcome the cognitive biases inherent in manually crafted prompts by automatically exploring superior prompt design spaces. Existing approaches fall into two categories: generation-and-search methods (APE/ProTeGi/PoisonedRAG: generate candidate prompts then search for the optimal one) and meta-prompt methods (OPRO/PE2: design elaborate meta-prompts to guide optimization).

**Limitations of Prior Work**: (a) **Template rigidity**: fixed meta-prompt templates cannot dynamically adapt to diverse task requirements, making it difficult to capture task-specific optimization directions; (b) **Inefficient exploration**: generation-and-search methods perform only local search near initial candidates, risking premature convergence or missing better regions of the prompt space.

**Key Challenge**: The search space for prompt optimization is discrete, high-dimensional, and non-differentiable, precluding direct gradient descent; yet gradient-like directional guidance is needed to avoid blind search.

**Key Insight**: Inspired by the Socratic teaching method—guiding students to discover answers through questioning rather than direct instruction—the paper models the prompt optimization process as a POMDP and employs multi-agent collaboration to simulate gradient-style iterative refinement.

**Core Idea**: A five-agent POMDP framework—a Planner charts the optimization path; Teacher-Critic-Student conduct Socratic dialogue for pseudo-gradient refinement; and a Target agent evaluates and provides feedback.

## Method

### Overall Architecture
Five LLM agents collaborate within a POMDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \mathcal{O} \rangle$:
- **Planner**: decomposes the optimization objective into a sub-goal sequence $\mathbf{ST} = [st_1, \ldots, st_n]$
- **Teacher**: formulates Socratic questions $q_i$ based on the current sub-goal and the previous prompt version
- **Critic**: evaluates the quality and directional validity of each question, producing feedback $c_i$
- **Student**: integrates the question and critique to update its internal state and generate a new prompt version $p_i$
- **Target**: executes the prompt on the downstream task and returns a performance reward $\mathcal{R}$

### Key Designs

1. **Planner — Optimization Trajectory Planning**:

    - **Function**: Decomposes the abstract objective of "optimizing a prompt" into a concrete, ordered sequence of sub-goals.
    - **Mechanism**: $\mathbf{ST} = \pi_{\text{plan}}(g, x, p_0)$; a latent variable $z$ is introduced to model task semantics, and a structured plan is generated via $\arg\max_{\mathbf{ST}} \mathbb{E}_{z \sim q(z|g,x)}[\log P(\mathbf{ST}|z, p_0)]$.
    - **Design Motivation**: Static meta-prompts apply a one-size-fits-all approach; the Planner tailors the optimization path to each individual task, enabling adaptive optimization.

2. **Teacher-Critic-Student Socratic Dialogue**:

    - **Function**: Refines the prompt through an iterative question–critique–revision cycle.
    - **Mechanism**: At each step $i$—the Teacher poses a question $q_i = \pi_t(st_i, p_{i-1}, \mathcal{H}_{<i})$ to steer the Student's reasoning toward a specific direction; the Critic evaluates $c_i = \pi_c(q_i, \mathcal{H}_{<i})$ to verify question quality and directional correctness; the Student updates the prompt $p_i = \pi_s((q_i, c_i), p_{i-1}, \mathcal{H}_{<i})$. All agents have full access to the dialogue history $\mathcal{H}_{<i}$.
    - **Design Motivation**: This simulates "pseudo-gradient descent" in discrete prompt space—the Teacher's question corresponds to the gradient direction, the Critic ensures directional correctness, and the Student executes the update step. Proposition 1 formally proves that the cumulative improvement is lower-bounded by $\geq \sum_i (\bar{A}_i - \sigma^2/2\lambda)$.

3. **Adaptive Termination**:

    - **Function**: Automatically determines when to stop optimization based on marginal returns.
    - **Mechanism**: Optimization continues when $\Delta\mathcal{R}^{(t)} = \mathcal{R}^{(t)} - \mathcal{R}^{(t-1)} > \delta$ and $t < I$. Proposition 2 proves that under Lipschitz conditions the reward change is bounded, guaranteeing convergence under small step sizes.
    - **Design Motivation**: Prevents over-refinement and unnecessary computational expenditure.

### Training Efficiency Highlights
Only **1 training sample** is required for optimization. The Planner can infer task structure and semantics from a single example, because the core of APO is understanding "what the task is" rather than memorizing "what the data is."

## Key Experimental Results

### Main Results — General Tasks (BBH + MMLU, 6+6=12 tasks)

| Method | BBH Avg. | MMLU Avg. | Overall Avg. |
|--------|----------|-----------|--------------|
| Origin (original prompt) | 53.71 | 76.39 | 64.95 |
| CoT (Zero-Shot) | 61.40 | 78.20 | 69.79 |
| PE2 (Prev. SOTA) | 69.45 | 88.44 | 78.81 |
| **MARS** | **79.52** | **90.94** | **85.11** |

### Main Results — Domain Tasks (C-Eval + LSAT + GSM8K, 5 tasks)

| Method | C-Eval | GSM8K | LSAT-AR | Avg. |
|--------|--------|-------|---------|------|
| PE2 | 66.47 | 83.46 | 34.50 | 69.39 |
| **MARS** | **77.13** | **89.22** | **38.42** | **75.81** |

### Ablation Study

| Configuration | BBH Avg. | MMLU Avg. | Change |
|---------------|----------|-----------|--------|
| MARS (full) | 79.52 | 90.94 | — |
| w/o Socratic | 68.28 | — | **−11.31** |
| w/o Planner | 72.82 | — | −6.77 |
| w/o Critic | 76.04 | — | −3.55 |

### Key Findings
- **MARS achieves state-of-the-art across all 17 datasets**: surpassing Prev. SOTA (PE2) by 6.04% on general tasks and 6.42% on domain-specific tasks.
- **The Socratic dialogue mechanism contributes the most**: removing it causes an average drop of 11.31%, far exceeding the impact of removing the Planner (−6.77%) or the Critic (−3.55%).
- **1-shot training is sufficient**: 0-shot achieves 77.77%, 1-shot achieves 79.59%, and 3-shot achieves 79.81%—the marginal gain from additional training data is negligible.
- **Fast convergence**: optimization typically converges within 5 rounds (vs. OPRO, which has not converged after 10 rounds), substantially reducing inference cost.
- **Cross-model generalization**: prompts optimized on DeepSeek-V2.5 transfer directly to GPT-4o without performance degradation, indicating that the optimized prompts are model-agnostic.
- **Inference-time scaling law**: MARS achieves the highest performance under an equivalent token budget, and reaches a given performance level with the lowest token cost.

## Highlights & Insights
- **Modeling APO as a POMDP** is the key theoretical innovation—the Student's internal reasoning state serves as the hidden state, Teacher/Critic interactions are actions, the prompt is the observation, and task performance is the reward, forming a complete mathematical framework.
- **The analogy between Socratic teaching and pseudo-gradient descent** is particularly elegant—the Teacher's question corresponds to the gradient direction, the Critic's evaluation to gradient correction, and the Student's update to a parameter step—with a formal proof of the cumulative improvement lower bound.
- **Requiring only 1 training sample** is a striking result, suggesting that the essence of APO is "understanding task specifications" rather than "fitting training data," and that the Planner's task comprehension capability is the core driver.
- **The appendix provides the final optimized prompts for all 17 tasks**, which are directly reusable.

## Limitations & Future Work
- The five-agent architecture incurs substantial inference overhead (5 LLM calls per round), making it sensitive to computational budgets.
- The framework relies on DeepSeek-V2.5 / GPT-4o as agent backbones; smaller models may be inadequate for the Teacher and Planner roles.
- The POMDP hidden-state transition $\mathcal{T}$ is implicitly realized by the LLM rather than being precisely modeled; the theoretical assumptions (Lipschitz continuity, bounded variance) may not hold strictly in practice.
- Evaluation is limited to text classification, QA, and mathematical reasoning tasks; generative tasks (e.g., summarization, translation) are not assessed.
- Whether the Planner-generated sub-goal sequences are genuinely superior to manually designed ones lacks comparison against human expert prompt engineers.

## Related Work & Insights
- **vs. OPRO**: OPRO uses a meta-prompt to directly instruct an LLM to generate an optimized prompt—a single-agent approach. MARS employs multi-agent Socratic dialogue for iterative refinement, converging faster (5 rounds vs. 10+) with higher performance.
- **vs. PE2**: PE2 is the previous SOTA meta-prompt method; MARS surpasses it by over 6% on all tasks with greater computational efficiency.
- **Implications for agent research**: The Teacher-Critic-Student triad pattern is transferable to other agent tasks requiring iterative refinement, such as code debugging, text revision, and solution optimization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ POMDP modeling combined with a Socratic five-agent framework represents a paradigm-level innovation in APO, with both rigorous theory and complete methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 17 datasets, ablation studies, convergence analysis, cross-model validation, 1-shot analysis, and inference-time scaling law—comprehensively evaluated.
- Writing Quality: ⭐⭐⭐⭐⭐ POMDP formalization is rigorous, Propositions are fully proven, and the appendix is exceptionally detailed (all prompts and complete optimization trajectories included).
- Value: ⭐⭐⭐⭐⭐ An APO method achieving SOTA with only 1-shot training data has high practical value; the multi-agent Socratic paradigm is broadly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)
- [\[ACL 2026\] LANG: Reinforcement Learning for Multilingual Reasoning with Language-Adaptive Hint Guidance](../../ACL2026/reinforcement_learning/lang_reinforcement_learning_for_multilingual_reasoning_with_language-adaptive_hi.md)
- [\[AAAI 2026\] HCPO: Hierarchical Conductor-Based Policy Optimization in Multi-Agent Reinforcement Learning](hcpo_hierarchical_conductor-based_policy_optimization_in_multi-agent_reinforceme.md)
- [\[AAAI 2026\] InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization](infigui-g1_advancing_gui_grounding_with_adaptive_exploration_policy_optimization.md)
- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](explaining_decentralized_multi-agent_reinforcement_learning_policies.md)

</div>

<!-- RELATED:END -->
