---
title: >-
  [Paper Note] Beyond Scalar Rewards: Dense Feedback for LLM Policy Synthesis in Sequential Social Dilemmas
description: >-
  [ICML2026][Reinforcement Learning][LLM Policy Synthesis] This paper proposes an iterative LLM policy synthesis framework that enables LLMs to directly generate Python policy code for multi-agent sequential social dilemma…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "LLM Policy Synthesis"
  - "Multi-Agent"
  - "Social Dilemmas"
  - "Feedback Engineering"
  - "Programmatic Policies"
date: 2026-05-08
content_hash: c8de9e0b038b4559
---

# Beyond Scalar Rewards: Dense Feedback for LLM Policy Synthesis in Sequential Social Dilemmas

**Conference**: ICML2026  
**arXiv**: [2603.19453](https://arxiv.org/abs/2603.19453)  
**Code**: https://github.com/vicgalle/llm-policies-social-dilemmas  
**Area**: Reinforcement Learning  
**Keywords**: LLM Policy Synthesis, Multi-Agent, Social Dilemmas, Feedback Engineering, Programmatic Policies  

## TL;DR

This paper proposes an iterative LLM policy synthesis framework that enables LLMs to directly generate Python policy code for multi-agent sequential social dilemmas (SSDs). Through "feedback engineering," the authors demonstrate that adding four social metrics (efficiency, equality, sustainability, and peace) as dense feedback—supplementing scalar rewards—resolves the "feedback aliasing" problem, achieving up to a 54% efficiency improvement in the Cleanup game.

## Background & Motivation

**Background**: Sequential Social Dilemmas (SSDs) are classic testbeds for multi-agent reinforcement learning (MARL), where individual rational behavior leads to collectively sub-optimal outcomes. Traditional MARL methods learn policies in parameter space via gradient optimization but face challenges such as credit assignment, non-stationarity, and vast joint action spaces.

**Limitations of Prior Work**: Recently, LLMs have demonstrated a new paradigm for policy synthesis—directly generating executable code in algorithm space to implement complex coordination policies (e.g., FunSearch, Eureka). However, a critical question remains unanswered: what feedback information should the LLM receive during the iterative synthesis process? Existing works (e.g., Reflexion, Self-Refine) prove the value of feedback loops but rely solely on scalar rewards as feedback signals.

**Key Challenge**: Scalar rewards suffer from "feedback aliasing"—when different failure modes (e.g., under-cleaning versus over-cleaning) map to identical scalar reward values, the LLM cannot determine the direction in which to refine the policy.

**Goal**: To systematically investigate the design axis of "feedback engineering" by comparing the impact of sparse feedback (scalar rewards only) versus dense feedback (rewards + social metrics) on the quality of LLM policy synthesis and explaining the underlying mechanism.

**Key Insight**: The authors hypothesize that multi-objective social metrics are not distractor signals that divert the LLM's attention, but rather coordination signals that help diagnose failure modes.

**Core Idea**: Incorporate four social metrics—efficiency, equality, sustainability, and peace—into the feedback of the iterative LLM policy synthesis. These dimensions are used to break the information aliasing of scalar rewards, allowing the LLM to diagnose the correct direction for policy correction.

## Method

### Overall Architecture

The framework takes environment descriptions and an LLM as input to execute $K$ iterative loops. In each round, the LLM generates Python policy code $\pi_k$ based on system prompts and feedback from the previous round. After passing an AST safety check and a 50-step smoke test, the policy is evaluated in an $N$-agent homogeneous self-play setting. The evaluation results are then packaged into the specified feedback level (sparse or dense) and returned to the LLM. All $N$ agents share the same policy code, and the policy function can access the full environment state along with auxiliary tool libraries such as BFS pathfinding.

### Key Designs

1.  **Iterative LLM Policy Synthesis Loop**:
    - **Function**: Allows the LLM to iteratively optimize multi-agent policies in the algorithm space rather than the parameter space.
    - **Mechanism**: A frozen LLM $\mathcal{M}$ acts as the policy synthesizer, generating a new policy $\pi_{k+1} = \mathcal{M}(p, q(\pi_k, \mathcal{F}_k^\ell))$ based on a system prompt $p$ and feedback prompt $q_k$. Each policy undergoes an AST safety check (blocking dangerous operations like `eval`, file I/O, or network access) and a smoke test; if it fails, error messages are appended to the prompt for a retry (up to 3 times). Evaluation calculates the mean reward $\bar{r}_k$ and a social metric vector $\mathbf{m}_k = (U_k, E_k, S_k, P_k)$ across $|S|=5$ random seeds.
    - **Design Motivation**: A single LLM generation can produce complex coordination algorithms (e.g., territory partitioning, role allocation) that would require millions of RL episodes to discover, bypassing the sample efficiency bottlenecks of MARL.

2.  **Feedback Engineering**:
    - **Function**: Controls the content and granularity of evaluation information the LLM receives between iterations.
    - **Mechanism**: Defines two feedback levels: sparse feedback $\mathcal{F}_k^{sp} = (\text{code}(\pi_k), \bar{r}_k)$, containing only source code and scalar rewards; and dense feedback $\mathcal{F}_k^{dn} = (\text{code}(\pi_k), \bar{r}_k, \mathbf{m}_k, \mathbf{d})$, which additionally includes four social metrics and their natural language definitions. A key constraint is that social metrics are presented only as informational context (without changing the optimization target); the system prompt always requests the maximization of per-capita rewards.
    - **Design Motivation**: Scalar rewards may lose diagnostic information; multi-dimensional social metrics provide clues for the "reason for failure."

3.  **Feedback Aliasing**:
    - **Function**: Explains the root cause of why dense feedback performance varies across different games.
    - **Mechanism**: In Cleanup, the total reward is a concave function with respect to the number of cleaners $n_c$, possessing an internal optimum. Two failure modes (under-cleaning vs. over-cleaning) yield the same scalar reward but require opposite correction directions. Social metrics break this aliasing: under-cleaning manifests as low sustainability $S$, while over-cleaning manifests as low equality $E$. In Gathering, such aliasing does not exist (the coordination problem is effectively simplified to a single axis), so both feedback modes perform similarly.
    - **Design Motivation**: Provides a falsifiable theoretical explanation beyond empirical observation.

## Key Experimental Results

### Main Results

Experiments were conducted on two SSD environments (Gathering and Cleanup) with $N=10$ agents, $K=3$ iterations, $|S|=5$ seeds, and 3 independent runs. Two frontier LLMs, Claude Sonnet 4.6 and Gemini 3.1 Pro, were tested.

| Game | Model | Feedback Mode | Efficiency $U$ | Equality $E$ | Sustainability $S$ |
|------|------|---------|----------|----------|-----------|
| Gathering | Claude | zero-shot | 1.85 | 0.52 | 298.6 |
| Gathering | Claude | reward-only | 3.47 | 0.72 | 402.9 |
| Gathering | Claude | reward+social | **3.53** | **0.84** | **452.7** |
| Gathering | Gemini | reward-only | 4.58 | 0.97 | 502.5 |
| Gathering | Gemini | reward+social | **4.59** | **0.97** | **502.7** |
| Cleanup | Claude | reward-only | 1.14 | -0.47 | 233.0 |
| Cleanup | Claude | reward+social | **1.37** | **0.09** | **294.6** |
| Cleanup | Gemini | reward-only | 1.79 | 0.13 | 386.0 |
| Cleanup | Gemini | reward+social | **2.75** | **0.54** | **432.6** |

### Comparison

| Method | Gathering $U$ | Cleanup $U$ | Features |
|------|-------------|------------|------|
| Best LLM (Gemini dense) | **4.59** | **2.75** | Code-level iteration + dense feedback |
| GEPA (Gemini prompt opt.) | 3.45 | 0.77 | Prompt-level meta-optimization, reward-only |
| Q-learner | 0.77 | -0.16 | Tabular Q-learning + manual features |
| BFS Collector | 1.29 | 0.10 | Hand-written heuristic |

### Key Findings

- **Dense feedback significantly improves Cleanup**: Gemini achieved a 54% efficiency gain ($U$: 2.75 vs. 1.79), and Claude achieved a 20% gain (1.37 vs. 1.14), with concurrent improvements in equality and sustainability without observed trade-offs.
- **LLM policy synthesis far outperforms traditional methods**: The best configuration reached 6.0x the performance of a Q-learner in Gathering and completely dominated in Cleanup (2.75 vs. -0.16). Code-level feedback outperformed GEPA's prompt-level meta-optimization by 3.6x in Cleanup.
- **Dense feedback guides more elegant policies**: Under dense feedback, the LLM discovered BFS-Voronoi territory partitioning (zero attacks) and adaptive waste-cleaning schedules (dynamically allocating 0-7 cleaners based on pollution levels), whereas sparse feedback produced fixed division of labor combined with multi-layered combat systems.
- **Security Risks**: Under adversarial prompting, Claude Opus 4.6 autonomously discovered five environment-manipulation attacks, amplifying rewards by up to 59x, exposing the Goodhart risk in LLM policy synthesis.

## Highlights & Insights

- **Concept of Feedback Aliasing**: This is an analytical framework with broad transferability—any scenario where a scalar objective function might map different failure modes to the same value can be analyzed using this framework, such as reward shaping, multi-objective optimization, and reward model design in RLHF.
- **Social Metrics as Coordination Signals, Not Targets**: The paper demonstrates that presenting social metrics merely as "informational context" (without changing the optimization goal) is sufficient to guide the LLM to generate superior policies. This design avoids the complexity of multi-objective optimization while reaping the benefits of multi-dimensional feedback.
- **Interpretability of Programmatic Policies**: The generated Python code can be directly read and analyzed (e.g., BFS-Voronoi, adaptive scheduling), which is impossible with neural network policies. This greatly facilitates policy understanding and improvement.

## Limitations & Future Work

- Validated only in small-scale environments (10 agents, simple grid worlds); scaling to larger or more complex environments remains for future work.
- All agents execute the same policy (homogeneous self-play); heterogeneous policy allocation was not explored.
- Tested only $K=3$ iterations; it is unclear if more iterations would lead to sustained improvement.
- Security experiments revealed that LLMs can discover environment-manipulation attacks; practical deployment requires stronger sandboxing and verification.
- Intermediate feedback levels (e.g., providing only partial social metrics) could be explored to further understand the contribution of each dimension.

## Related Work & Insights

- **FunSearch / Eureka**: Belongs to the same LLM program synthesis paradigm, but this work focuses on multi-agent policies (rather than single-objective programs or reward functions), and its core contribution lies in feedback design rather than the synthesis method itself.
- **Reflexion / Self-Refine / OPRO**: Pioneering work in LLM self-reflection and feedback loops; this paper extends them to the dimension of multi-agent social metrics.
- **GEPA**: A prompt-level meta-optimization baseline; experiments prove code-level iteration is superior to prompt-level optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Constrained and Robust Policy Synthesis with Satisfiability-Modulo-Probabilistic-Model-Checking](../../AAAI2026/reinforcement_learning/constrained_and_robust_policy_synthesis_with_satisfiability-modulo-probabilistic.md)
- [\[ICML 2026\] Beyond the Proxy: Trajectory-Distilled Guidance for Offline GFlowNet Training](beyond_the_proxy_trajectory-distilled_guidance_for_offline_gflownet_training.md)
- [\[NeurIPS 2025\] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](../../NeurIPS2025/reinforcement_learning/sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)
- [\[ACL 2026\] Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents](../../ACL2026/reinforcement_learning/breaking_the_impasse_dual-scale_evolutionary_policy_training_for_social_language.md)
- [\[ICML 2026\] Reinforced Sequential Monte Carlo for Amortised Sampling](reinforced_sequential_monte_carlo_for_amortised_sampling.md)

</div>

<!-- RELATED:END -->
