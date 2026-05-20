---
title: >-
  [Paper Note] Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents
description: >-
  [ICLR 2026][LLM Agent][active reasoning] This paper proposes T³ (Truncating Belief-Trapped Trajectories), which leverages POMDP theory to analyze the "belief trap" phenomenon in multi-turn active reasoning of LLM agents.…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "active reasoning"
  - "reinforcement-learning"
  - "belief tracking"
  - "POMDP"
  - "credit assignment"
date: 2026-05-08
content_hash: 1e833936bed2c649
---

# Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents

**Conference**: ICLR 2026
**arXiv**: [2510.12264](https://arxiv.org/abs/2510.12264)  
**Code**: [https://github.com/unimpor/T3](https://github.com/unimpor/T3)  
**Area**: LLM Agent
**Keywords**: active reasoning, reinforcement-learning, LLM agent, belief tracking, POMDP, credit assignment

## TL;DR

This paper proposes T³ (Truncating Belief-Trapped Trajectories), which leverages POMDP theory to analyze the "belief trap" phenomenon in multi-turn active reasoning of LLM agents. By detecting belief deviation and truncating uninformative trajectory suffixes, T³ corrects credit assignment errors during RL training, achieving performance gains of up to 30 points across 5 challenging tasks while reducing token consumption by 34%.

## Background & Motivation

1. **Core challenge of active reasoning**: LLM agents must strategically query and actively acquire information over multi-turn interactions, demanding precise belief tracking—maintaining accurate representations of underlying states and uncertainties.
2. **Belief deviation**: Due to limited LLM reasoning capacity, internal beliefs drift away from the true problem state, leading to loss of state awareness and uninformative or repetitive actions—a phenomenon termed "belief deviation."
3. **Vicious cycle in RL training**: Uninformative trajectory suffixes caused by belief deviation corrupt credit assignment in reinforcement learning, causing early valuable exploration actions to be incorrectly penalized, and advantage estimates may even be reversed.
4. **Multi-turn dilemma for LLM agents**: In practice, LLM agents frequently generate redundant, irrelevant, or uninformative actions during multi-turn reasoning, sometimes entering futile loops—problems that RL training alone does not fully resolve.
5. **Imperfect belief updates in POMDPs**: Classical POMDPs assume perfect Bayesian belief updates, but LLM agent belief updates are inherently imperfect and error-prone, leading to cumulative deviation.
6. **Limitations of existing RL methods**: Standard policy optimization methods (PPO, GRPO, etc.) do not account for belief-trap dynamics, leaving learned policies brittle in out-of-distribution scenarios with insufficient generalization.

## Method

### Theoretical Foundation: POMDP Formulation

Active reasoning is formalized as a POMDP $(S, A, O, T, O, R, \gamma)$. The agent selects actions $a_t$ based on belief state $b_t \in \Delta(S)$, and the environment returns observation $o_t$. A ground-truth-anchored potential function $\Psi(b) = -\log b(s^*)$ is introduced to measure task progress: $\Psi = 0$ indicates task completion, and smaller values indicate higher confidence in the true state $s^*$.

### Core Concept: Belief-Trapped Region (BTR)

**Theorem 1** (informal): Under assumptions of non-degenerate observations, Lipschitz policies, and growing belief update errors, the belief trajectory of an LLM agent will enter an absorbing region (BTR) within a finite number of steps, within which the expected task progress becomes non-positive: $\mathbb{E}[\Psi_{t+1} | b_t] \geq \Psi_t$.

**Assumption 1 (Growing Update Error)**: There exists a constant $m_\theta > 0$ such that in high-uncertainty regions, the LLM's belief update error grows at least linearly with the deviation itself. Intuitively, larger deviation makes correction harder, forming a positive-feedback amplification loop.

### Credit Assignment Failure Mechanism

**Theorem 2** (informal): Once a trajectory enters the BTR, the uninformative suffix corrupts the Generalized Advantage Estimation (GAE) of early exploration actions. When the suffix is sufficiently long, the negative drift can dominate the positive contribution, rendering the advantage estimate of early actions negative and reversing the gradient direction—penalizing valuable early exploration that should instead be encouraged.

**Corollary 1 (Value of Truncation)**: Truncating the trajectory at the BTR entry point eliminates the adverse effects of the uninformative suffix, yielding less biased gradient estimates.

### T³ Method Design

**T³ Condition (Definition 2)**: If within window $[t-k, t)$, the hypothesis space refinement metric $d(H_\tau, H_{\tau+1}) \leq \Delta_{\min}$ holds continuously for all steps, the trajectory is truncated at step $t$.

Task-specific instantiations:

- **GuessNumbers (GN)**: $H_t$ is the set of consistent candidate numbers; $d = |H_\tau| - |H_{\tau+1}|$; truncation triggered when a guess falls outside the candidate set ($k=1$).
- **SituationPuzzles (SP)**: The judge's "unknown" feedback serves as a proxy for no refinement; truncation after $k=5$ consecutive steps.
- **CircuitDecoding (CD)**: Similar to GN; truncation if the candidate set does not shrink for $k=3$ consecutive steps.
- **PreferenceEstimation (PE) / MovieRecommendation (MR)**: Monitors similarity changes between the estimated vector and true preferences; truncation after $k=2$ consecutive steps of decline.

### Key Design Properties

T³ operates as a meta-wrapper that integrates seamlessly into standard policy optimization frameworks including PPO, GRPO, and GSPO without modifying the underlying algorithms, making it a plug-and-play solution.

## Key Experimental Results

### Main Results (5 Tasks, 3 RL Algorithms)

| Method | CD (EM) | SP (F1-word) | GN (EM) | PE (Binary Sim) | MR (EM) | Avg. Rank |
|--------|---------|-------------|---------|-----------------|---------|-----------|
| o3-mini | 92.67 | 20.64 | 95.28 | 44.67 | 83.33 | 4.67 |
| Gemini-2.5-Pro | 92.23 | 24.12 | 90.84 | 16.67 | 83.00 | 5.67 |
| PPO | 61.67 | 28.77 | 91.62 | 42.00 | 24.33 | 6.50 |
| **PPO + T³** | **77.83 (+16.2)** | **36.85 (+8.1)** | **93.98 (+2.4)** | **49.00 (+7.0)** | **38.00 (+13.6)** | **4.50** |
| GRPO | 79.33 | 36.46 | 61.26 | 51.67 | 12.00 | 5.50 |
| **GRPO + T³** | **81.33 (+2.0)** | **39.45 (+3.0)** | **91.36 (+30.1)** | **52.33 (+0.7)** | **32.67 (+20.7)** | **3.17** |
| GSPO | 77.67 | 36.63 | 96.07 | 59.00 | 14.67 | 4.33 |
| **GSPO + T³** | **81.00 (+3.3)** | **36.96 (+0.3)** | **99.74 (+3.7)** | **62.00 (+3.0)** | **55.67 (+41.0)** | **2.50** |

T³ achieves non-marginal improvements on 14 out of 18 metrics. Largest gains: GSPO+T³ on MR (+41.0), GRPO+T³ on GN (+30.1). GSPO+T³ approaches near-perfect performance on GN (99.74).

### Out-of-Distribution (OOD) Generalization

| PE Task (PPO) | Vanilla | + T³ | CD Task (PPO) | Vanilla | + T³ |
|---------------|---------|------|---------------|---------|------|
| Reference set S=5 | 40.0 | 44.3 (+4.3) | Candidate set S=10 | 67.8 | 86.3 (+18.5) |
| S=10 | 42.0 | 49.0 (+7.0) | S=15 | 61.7 | 74.7 (+13.0) |
| S=20 | 41.0 | 53.7 (+12.7) | S=20 | 48.2 | 55.8 (+7.7) |
| S=30 | 42.3 | 46.3 (+4.0) | S=30 | 31.5 | 35.7 (+4.2) |

T³ consistently improves performance across all OOD settings, demonstrating that the method learns generalizable active reasoning strategies.

### Training Efficiency

T³ reduces the average token count per rollout through early truncation, enabling higher training efficiency. On PPO+CD, reaching reward 0.65 requires only 66.4% of the tokens needed by the baseline; on GSPO+GN, reaching 0.96 requires only 76.3%. Training curves are more stable, with rewards increasing monotonically or near-monotonically with fewer sharp drops.

## Highlights & Insights

- **Theory-driven**: The belief-trap and credit assignment failure mechanisms are rigorously derived from POMDP theory, with a complete theorem–assumption–corollary chain.
- **Plug-and-play**: T³ integrates into PPO/GRPO/GSPO without modifying the underlying RL algorithm, offering strong practical utility.
- **Multi-dimensional improvement**: Simultaneously improves final performance (up to +41 points), training stability, token efficiency (34% savings), and OOD robustness.
- **Empirical validation of theory**: Key theoretical claims (Assumption 1 on growing update error, Theorem 2 on advantage drift) are empirically validated.
- **Implications for frontier models**: On tasks with unbounded hypothesis spaces (SP, PE), a 7B model trained with RL+T³ can surpass o3-mini and Gemini-2.5-Pro.

## Limitations & Future Work

- **Task-specific proxy signals**: The T³ condition requires designing observable proxy signals (hypothesis space refinement metrics) for each task, limiting generalizability.
- **Hypothesis space construction**: Precisely constructing $H_t$ and the metric $d(\cdot, \cdot)$ remains challenging for tasks with continuous or unbounded hypothesis spaces.
- **Theoretical assumption limitations**: Assumption 1 (linear growth of update error) may only hold approximately in practice, and the threshold $U$ cannot be directly measured.
- **Evaluation scope**: Validation is primarily conducted on information-acquisition reasoning tasks; applicability to more complex open-ended agent scenarios (e.g., web browsing, code generation) remains to be verified.

## Related Work & Insights

### vs. Standard RL for LLMs (GRPO / PPO without truncation)
Standard RL methods do not account for belief-trap dynamics, allowing uninformative trajectory suffixes to participate in training and systematically corrupting credit assignment. T³ preserves correct credit attribution for informative prefixes by truncating trajectories upon BTR entry. Experiments show that GRPO achieves only 61.26 on GN, rising to 91.36 (+30.1) with T³, demonstrating the substantive improvement in gradient quality from truncation.

### vs. Frontier Reasoning Models (o3-mini, Gemini-2.5-Pro)
Frontier reasoning models perform strongly on tasks with finite, enumerable hypothesis spaces (GN: 95.28, CD: 92.67), but degrade significantly on tasks with large, continuous, or unbounded hypothesis spaces (SP: 20.64, PE: 16.67). This suggests that pure scale-up RL with outcome rewards is insufficient for active reasoning, and mechanisms like T³ that explicitly address credit assignment provide complementary gains.

### vs. Self-Correction / Reflection Methods (Self-Refine, Reflexion)
Self-correction methods rely on internal LLM reflection to improve reasoning trajectories, but cannot address the root cause of belief deviation—the model itself lacks the capacity to detect imperfect belief updates. T³ intervenes at the training level, detecting belief traps via external observable signals and truncating harmful trajectories at the data level, representing a methodologically distinct and complementary approach.

## Rating

- ⭐⭐⭐⭐⭐ **Novelty**: Derives a complete chain from POMDP theory—belief traps → credit assignment failure → truncation solution—with novel concepts and rigorous theoretical grounding.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Covers 5 tasks, 3 RL algorithms, OOD analysis, ablation studies, and theory validation experiments.
- ⭐⭐⭐⭐ **Value**: Plug-and-play design enables direct integration into existing RL training pipelines; token savings carry practical engineering value.
- ⭐⭐⭐⭐ **Writing Quality**: Theoretical derivations connect naturally to practical design; task-specific instantiations of proxy signals are described clearly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](../../AAAI2026/llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)
- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](../../ACL2026/llm_agent/hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)
- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [\[ICLR 2026\] CoMind: Towards Community-Driven Agents for Machine Learning Engineering](comind_towards_community-driven_agents_for_machine_learning_engineering.md)
- [\[ICLR 2026\] REMem: Reasoning with Episodic Memory in Language Agents](remem_reasoning_with_episodic_memory_in_language_agent.md)

</div>

<!-- RELATED:END -->
