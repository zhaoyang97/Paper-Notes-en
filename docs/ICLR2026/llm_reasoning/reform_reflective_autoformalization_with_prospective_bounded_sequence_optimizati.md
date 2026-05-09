---
title: >-
  [Paper Note] ReForm: Reflective Autoformalization with Prospective Bounded Sequence Optimization
description: >-
  [ICLR 2026][LLM Reasoning][autoformalization] This paper proposes ReForm, a reflective autoformalization paradigm that transforms the process of converting natural-language mathematics problems into Lean formal statements from single-pass generation into an iterative "generate → semantic self-verify → correct" loop. It further introduces the PBSO algorithm to optimize heterogeneous reward signals, achieving an average improvement of 22.6 percentage points over the strongest baselines across four benchmarks.
tags:
  - ICLR 2026
  - LLM Reasoning
  - autoformalization
  - Lean
  - semantic consistency
  - self-correction
  - reinforcement-learning
  - heterogeneous reward
  - PBSO
date: 2026-05-08
content_hash: b4b15d78576411bd
---

# ReForm: Reflective Autoformalization with Prospective Bounded Sequence Optimization

**Conference**: ICLR 2026
**arXiv**: [2510.24592](https://arxiv.org/abs/2510.24592)
**Code**: [GitHub](https://github.com/RUCAIBox/ReForm) (with models and benchmarks)
**Area**: LLM Reasoning
**Keywords**: autoformalization, Lean, semantic consistency, self-correction, reinforcement-learning, heterogeneous reward, PBSO

## TL;DR

This paper proposes ReForm, a reflective autoformalization paradigm that transforms the process of converting natural-language mathematics problems into Lean formal statements from single-pass generation into an iterative "generate → semantic self-verify → correct" loop. It further introduces the PBSO algorithm to optimize heterogeneous reward signals, achieving an average improvement of 22.6 percentage points over the strongest baselines across four benchmarks.

## Background & Motivation

Autoformalization — translating natural-language mathematics problems into machine-verifiable formal statements (e.g., in the Lean language) — represents a critical bottleneck in formal mathematical reasoning. The following core tensions exist in prior work:

**Syntactic correctness ≠ Semantic correctness**: LLMs can generate syntactically correct statements that pass the Lean compiler, yet frequently fail to faithfully preserve the semantic intent of the original problem (e.g., misinterpreted quantifier scopes, omitted implicit constraints, incorrect boundary conditions).

**Limitations of single-pass generation**: Existing methods (including Goedel-V2 and Kimina) treat formalization as a straightforward translation task and lack self-reflection or iterative error-correction mechanisms.

**Even human experts make mistakes**: 16.4% of manually formalized statements in miniF2F and 38.5% in ProofNet contain semantic errors, underscoring the inherent difficulty of the problem.

The core idea is to emulate the "review-and-revise" iterative process of human experts, enabling the model to detect and correct its own semantic errors during generation.

## Method

### Overall Architecture

ReForm reframes autoformalization as a reflective iterative process, interleaving formal statement generation with semantic self-verification.

Given a natural-language problem $Q$, at iteration $t$:
1. **Formalization generation**: $S_t = \pi(Q, \mathcal{H}_t)$, where $\mathcal{H}_t = \{(S_1, C_1), \ldots, (S_{t-1}, C_{t-1})\}$ is the history.
2. **Semantic self-verification**: $C_t = \pi(Q, \mathcal{H}_t, S_t)$, assessing the semantic consistency of $S_t$ with respect to $Q$.
3. The loop continues until verification passes.

Critically, the entire process is implemented as a **single autoregressive generation**, requiring no multiple model calls.

### Key Designs: Heterogeneous Reward Mechanism

Two complementary reward signals are designed:

**Task reward (terminal)**:
$$r_{\text{task}}(Q, \text{Ans}) = \begin{cases} 1 & \text{if PassesLean(Ans)} \land \text{IsConsistent}(Q, \text{Ans}) \\ 0 & \text{otherwise} \end{cases}$$

**Auxiliary reward (intermediate steps)**:
$$r_{\text{aux}}^t(Q, S_t, C_t) = \begin{cases} 1 & \text{if IsFaithfulCritique}(Q, S_t, C_t) \\ 0 & \text{otherwise} \end{cases}$$

The auxiliary reward evaluates whether the critique at each step accurately diagnoses the semantic relationship, penalizing false positives, false negatives, and premature termination.

### Prospective Bounded Sequence Optimization (PBSO)

The core contribution: resolving the optimization challenge posed by heterogeneous rewards at different sequence positions.

**Prospective bounded return**: For each step in the trajectory, the future cumulative reward is computed as:
$$G_t = \text{clip}(r_t + \gamma \cdot G_{t+1}, r_{\min}, r_{\max})$$

where $\gamma \in (0,1]$ is the discount factor, $G_{T+1} = 0$, and the clip operation constrains returns within the reward function's range to prevent unbounded accumulation from causing gradient instability.

**Position-specific advantage function**: For $N$ sampled trajectories, advantages at each iteration step are computed via joint normalization:
$$\hat{A}_t^j = \frac{G_t^j - \text{mean}(\mathcal{G})}{\text{std}(\mathcal{G})}, \quad \mathcal{G} = \bigcup_{j=1}^N \{G_t^j : t=1,...,T_j+1\}$$

Different iteration steps within the same trajectory can receive different advantage values, enabling fine-grained credit assignment. Standard GRPO policy updates are then applied.

### Loss & Training

Standard GRPO loss with position-specific advantages $\hat{A}_t^j$, jointly optimizing formalization accuracy and self-verification quality.

## Key Experimental Results

### Main Results

| Model | miniF2F sem | ProofNet sem | Putnam sem | AIME2025 sem | AVG sem |
|------|-----------|------------|----------|------------|---------|
| GPT-5 | 66.0 | 44.6 | 45.8 | 13.3 | 42.4 |
| Goedel-V2-8B | 81.1 | 47.3 | 42.9 | 26.7 | 49.5 |
| Goedel-V2-32B | 82.0 | 50.5 | 41.4 | 26.7 | 50.1 |
| **ReForm-8B** | **87.7** | **65.6** | **57.3** | **46.7** | **64.3** |
| **ReForm-32B** | **91.4** | **70.4** | **62.3** | **66.7** | **72.7** |

ReForm-8B improves average semantic consistency over Goedel-V2-8B by **+14.8pp**, even surpassing the 4× larger Goedel-V2-32B (+14.2pp). ReForm-32B achieves an average semantic consistency of **72.7%**, surpassing the strongest baseline by **+22.6pp**.

Gains are more pronounced on harder datasets: ProofNet +19.9pp, PutnamBench +20.9pp, AIME2025 +40.0pp (32B).

### Ablation Study

| Variant | miniF2F | ProofNet | Putnam | AIME25 |
|------|---------|----------|--------|--------|
| ReForm (full) | 87.7 | 65.6 | 57.3 | 46.7 |
| w/o clip | 84.0 | 59.6 | 48.9 | 26.7 |
| w/o $r_{\text{aux}}$ | 87.7 | 65.6 | 52.1 | 40.0 |
| w/o RL | 85.2 | 62.3 | 49.4 | 30.0 |
| One-pass | 82.7 | 59.1 | 40.8 | 16.7 |

Key findings:
- **Removing clip** causes severe degradation (AIME25 drops from 46.7 to 26.7), confirming that bounded returns are critical for heterogeneous reward optimization.
- **Auxiliary reward** $r_{\text{aux}}$ has greater impact on harder problems (Putnam −5.2, AIME25 −6.7).
- **One-pass vs. ReForm**: the gap widens with problem difficulty (30pp on AIME25), validating the necessity of the reflective paradigm for hard problems.

### Key Findings

1. **Training dynamics**: Response length grows naturally from 2,300 to 4,800 tokens (with no length reward), indicating the model spontaneously learns more thorough self-inspection behavior.
2. **ConsistencyCheck benchmark**: Among 859 expert-annotated instances, frontier LLMs as evaluators achieve approximately 85.8% accuracy; however, ReForm's improvement (+22.6pp) far exceeds evaluation noise.
3. **Human experts also make mistakes**: 16.4% of miniF2F and 38.5% of ProofNet manually formalized statements contain semantic errors.

## Highlights & Insights

1. **Paradigm shift**: Transitioning from single-pass translation to iterative reflection, and from a single terminal reward to heterogeneous rewards, represent two independent yet complementary innovations.
2. **Remarkable parameter efficiency**: The 8B model outperforms the 32B baseline, demonstrating that architectural innovation in the reflective framework outweighs parameter scale.
3. **Generalizability of PBSO**: Prospective bounded returns are not limited to formalization tasks and offer broader inspiration for multi-objective sequential decision-making problems.
4. **Training stability**: Reward curves rise smoothly with narrowing confidence bands, confirming PBSO's effective balancing of heterogeneous objectives.
5. **Revealing evaluation limitations**: The construction of the ConsistencyCheck benchmark not only validates evaluation reliability but also quantifies the challenge of formalization.

## Limitations & Future Work

1. Training data is sourced from multiple open-source repositories; despite deduplication, data quality may still constrain the performance ceiling.
2. Inference-time token consumption increases by 2.1×, imposing a moderate efficiency cost.
3. Evaluation relies on LLM-as-judge (85.8% accuracy), introducing approximately 14% judgment noise.
4. PBSO introduces additional hyperparameters, including the discount factor $\gamma$ and the clip range.

## Related Work & Insights

- **Relationship to Goedel/Kimina**: These methods improve semantic consistency through high-quality data but remain single-pass generation; ReForm introduces a reflective loop at the methodological level.
- **Relationship to general RL for LLMs**: Methods such as GRPO and DAPO rely solely on terminal rewards without supervising intermediate steps; PBSO's prospective bounded returns provide explicit supervision signals for intermediate steps.
- **Implications for automated theorem proving**: More accurate formalization directly raises the performance ceiling for downstream ATP systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Dual innovations of the reflective paradigm and PBSO address the core semantic problem in autoformalization.
- **Practicality**: ⭐⭐⭐⭐ — The formalization task is specialized, but the methodological ideas have broad transfer potential.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four benchmarks, comprehensive ablations, training dynamics analysis, and evaluation reliability verification.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is clear, method derivation is rigorous, and experimental analysis is thorough.
- **Overall**: ⭐⭐⭐⭐⭐ — Achieves a qualitative leap in the formalization domain with methodological contributions of broad applicability.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](../../ACL2026/llm_reasoning/jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)
- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [\[ICLR 2026\] FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)
- [\[NeurIPS 2025\] AbbIE: Autoregressive Block-Based Iterative Encoder for Efficient Sequence Modeling](../../NeurIPS2025/llm_reasoning/abbie_autoregressive_block-based_iterative_encoder_for_efficient_sequence_modeli.md)

<!-- RELATED:END -->
