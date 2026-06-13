---
title: >-
  [Paper Note] SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks
description: >-
  [ACL 2026][LLM Reasoning][Sequence-Level PPO] SPPO reformulates RLVR in long-chain CoT reasoning from a token-level MDP into a sequence-level contextual bandit. By using a scalar critic that solely observes the prompt to…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Sequence-Level PPO"
  - "Long-Horizon Reasoning"
  - "RLVR"
  - "Scalar Value Function"
  - "Contextual Bandit"
date: 2026-05-08
content_hash: 7b8f2b7e01e0af97
---

# SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks

**Conference**: ACL 2026  
**arXiv**: [2604.08865](https://arxiv.org/abs/2604.08865)  
**Code**: https://github.com/sustech-nlp/SPPO  
**Area**: LLM Reasoning / Reinforcement Learning / RLVR  
**Keywords**: Sequence-Level PPO, Long-Horizon Reasoning, RLVR, Scalar Value Function, Contextual Bandit

## TL;DR
SPPO reformulates RLVR in long-chain CoT reasoning from a token-level MDP into a sequence-level contextual bandit. By using a scalar critic that solely observes the prompt to estimate problem solvability, it achieves stability and performance comparable to or exceeding GRPO with single-sample PPO, while delivering approximately 5.9x training acceleration and lower memory consumption.

## Background & Motivation
**Background**: Mathematical reasoning, code reasoning, and verifiable QA tasks frequently employ RLVR to enhance large models, where rewards are typically based on final answer correctness. Standard PPO utilizes a token-level critic and GAE to propagate final rewards token-by-token along long CoT sequences; GRPO eliminates the critic, estimating a baseline through the relative performance of multiple samples for the same prompt.

**Limitations of Prior Work**: Standard PPO is unstable given long-chain sparse rewards, as the critic often perceives answer clues only at the sequence tail, causing advantage signals to vanish or become misaligned during the reasoning process. Although GRPO bypasses token-level critics, it requires multiple samples per prompt to estimate a group baseline, which limits training throughput.

**Key Challenge**: The reward for long-chain reasoning is binary (success or failure of the whole reasoning), but token-level PPO forces this into credit assignment per timestep. Conversely, group-based methods treat sequences as a whole but exchange stability for high-cost multi-sampling.

**Goal**: The authors aim to maintain the single-sample efficiency of PPO while achieving the stability of "sequence-level updates" similar to GRPO, specifically targeting verifiable mathematical reasoning tasks such as AIME, AMC, MATH500, and Minerva Math.

**Key Insight**: The paper reinterprets the success of GRPO: the critical factor is not the absence of a critic, but rather the implicit treatment of the reasoning process as a sequence-level contextual bandit—where the prompt is the context, the entire response is a single action, and the final reward is the action return.

**Core Idea**: Explicitly adopt a sequence-level bandit perspective using a scalar value model to estimate prompt success probability, then feed $A=R-V_\phi(s_p)$ back into PPO as a shared advantage signal for all tokens in the response.

## Method
The core of SPPO is not a mere change in the loss function name, but a shift in the semantics of the value function. While a standard PPO critic attempts to determine "future rewards given the token at step $t$," the SPPO critic answers "how likely is the current policy to solve this prompt." This task is closer to problem difficulty estimation and is significantly simpler than token-by-token reasoning state valuation.

### Overall Architecture
Given a prompt $s_p$, the policy samples a full response sequence $a_{seq}=(y_1,\dots,y_T)$, and an external verifier returns a binary reward $R\in\{0,1\}$. The value model $V_\phi(s_p)$ outputs a prompt-level success probability. SPPO constructs a sequence-level advantage using $R-V_\phi(s_p)$ and distributes this identical advantage to every token in the sequence within the PPO clipped objective.

### Key Designs
1.  **From token-level MDP to sequence-level contextual bandit**:
    - Function: Eliminates temporal credit assignment noise from sparse rewards in long CoT.
    - Mechanism: Conceptually compresses the horizon to 1, where the prompt is a static context, the full response is an atomic action, and the reward solely evaluates whether the correct answer was reached.
    - Design Motivation: Mathematical verifiers usually only judge final answers. Forcing a critic to value intermediate tokens introduces positional bias. Sequence-level modeling aligns with the actual reward granularity.

2.  **Scalar value function and advantage estimation**:
    - Function: Lowers the variance of single-sample returns using a learned prompt baseline.
    - Mechanism: The value model fits binary outcomes via BCE with the objective $L_V=-E[R\log V_\phi(s_p)+(1-R)\log(1-V_\phi(s_p))]$; the policy uses $A(s_p,a)=R-V_\phi(s_p)$. Rare successes on difficult problems yield strong positive advantages, while failures on easy problems yield strong negative advantages.
    - Design Motivation: GRPO's group baseline requires multi-sampling for stability. SPPO uses a calibratable scalar critic to directly approximate solvability, avoiding multi-sampling overhead.

3.  **Sequence-level PPO and decoupled critic**:
    - Function: Retains PPO's stable update mechanism while reducing memory pressure during LLM RL training.
    - Mechanism: The clipped probability ratio is still calculated per token, but the advantage is invariant across the sequence tokens. The authors also validate a configuration using a 1.5B critic for a 7B policy, as "estimating problem difficulty" is easier than "generating reasoning chains."
    - Design Motivation: Leverage mature PPO engineering and clipping stability while avoiding the GAE tail effect under sparse rewards.

### Loss & Training
Experiments utilize DeepSeek-R1-Distill-Qwen-1.5B and 7B, fine-tuned on DeepScaleR and DAPO-17K respectively. Rewards are 1 if the boxed answer is correct and 0 otherwise. Learning rates are 1e-6 for the actor and 5e-6 for the critic. PPO parameters $\gamma=1, \lambda=1$ are used to match sparse terminal rewards. 1.5B experiments were conducted on 4×A100, and 7B on 4×H100.

## Key Experimental Results

### Main Results

| Model Scale | Method | AIME24 | AIME25 | AMC23 | MATH500 | Minerva | Avg |
|--------|------|------|------|------|------|------|------|
| 1.5B | Base | 27.50 | 21.67 | 71.56 | 83.73 | 20.35 | 44.96 |
| 1.5B | PPO | 27.50 | 20.83 | 70.63 | 81.38 | 19.89 | 44.06 |
| 1.5B | GRPO N=8 | 30.00 | 26.25 | 73.13 | 83.88 | 22.15 | 47.08 |
| 1.5B | SPPO | 34.17 | 25.83 | 74.38 | 83.78 | 22.15 | 48.06 |
| 7B | PPO | 45.20 | 35.42 | 85.31 | 88.48 | 27.80 | 56.44 |
| 7B | GRPO N=8 | 47.08 | 35.00 | 86.25 | 90.15 | 28.74 | 57.44 |
| 7B | SPPO | 50.83 | 35.00 | 86.25 | 90.13 | 28.35 | 58.11 |
| 7B | SPPO + 1.5B critic | 52.29 | 34.58 | 87.19 | 89.88 | 28.86 | 58.56 |

### Ablation Study

| Analysis Item | Key Metric | Description |
|------|------|------|
| PPO + BCE | Performance collapse around 500 steps | Simply adding BCE loss to token-level PPO does not replicate SPPO, indicating gains come from the sequence-level bandit formulation. |
| Training Efficiency | ~22 hours for 7B to reach ~58 avg | Single-sample updates converge faster than GRPO/RLOO multi-sample baselines. |
| Value Calibration | Pearson 0.642, Spearman 0.664 | Prompt-level critic distinguishes problem difficulty; although predictions are conservative, it serves as an effective baseline. |
| Memory Efficiency | Decoupled critic reduces memory by ~12.8% | 1.5B critic with 7B policy still achieves the highest average score. |

### Key Findings
- SPPO outperforms the average score of GRPO at both 1.5B and 7B scales while requiring only single-sample updates, suggesting "sequence-level advantage" is a more fundamental source of stability than "multi-sample normalization."
- A small critic does not hinder the 7B policy; rather, it achieved the highest Avg (58.56), supporting the hypothesis that prompt solvability estimation is simpler than generative reasoning.
- In sparse binary control tasks (Precision CartPole, MountainCar, etc.), SPPO is more stable than standard PPO, indicating the findings are not solely due to verl engineering optimizations.

## Highlights & Insights
- The most valuable contribution is the re-interpretation of GRPO: its success may stem from "treating the response as a holistic action" rather than just the absence of a critic. This links the strengths and weaknesses of PPO and GRPO.
- SPPO does not abandon PPO entirely but shifts the advantage granularity to the sequence level, making it easier to integrate into existing RLHF/RLVR frameworks.
- Small critic insights: LLM RL does not strictly require same-scale actors and critics. If the critic's role is difficulty estimation, a smaller model suffices, lowering training barriers.

## Limitations & Future Work
- SPPO depends on verifiable outcomes to train the value model, making it ideal for math, code, and logical tasks; open-ended writing and dialogue lack objective verifiers, making transfer non-trivial.
- Sequence-level advantage reinforces/punishes the entire reasoning chain, still failing to isolate which specific steps within a sequence contributed to the correct answer.
- Value model calibration is vital. The paper indicates good correlation but conservative distribution; future research could focus on stronger calibration or uncertainty estimation.
- Experiments focused on the DeepSeek-R1-Distill-Qwen series and math tasks; more model families, code tasks, and multi-turn agent tasks require further verification.

## Related Work & Insights
- **vs Standard PPO**: Standard PPO uses token-level value and GAE for long-range credit assignment; SPPO uses prompt-level scalar values to avoid tail effects and improve stability.
- **vs GRPO**: GRPO constructs group baselines via N=8 multi-sampling; SPPO replaces this with a learned critic, enabling higher throughput.
- **vs ReMax / RLOO**: These REINFORCE variants also focus on whole-sequence rewards, but SPPO retains PPO clipping and uses value baselines to reduce variance.
- **vs DAPO / Dr.GRPO**: These methods focus on group-relative sampling or gradient patching; SPPO targets the underlying modeling granularity by rewriting the environment as a sequence-level bandit.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Not just parameter tuning, but a clear reconstruction of RLVR credit assignment granularity.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers math benchmarks, efficiency, value calibration, and control tasks; open-ended tasks are still missing.
- Writing Quality: ⭐⭐⭐⭐☆ Problem definition, intuition, and empirical evidence are clearly linked; formulas and diagrams are well-integrated.
- Value: ⭐⭐⭐⭐⭐ Highly practical for reasoning model teams aiming to reduce RLVR training costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents](fs-researcher_test-time_scaling_for_long-horizon_research_tasks_with_file-system.md)
- [\[ICLR 2026\] Segment-Level Attribution for Selective Learning of Long Reasoning Traces](../../ICLR2026/llm_reasoning/segment-level_attribution_for_selective_learning_of_long_reasoning_traces.md)
- [\[ACL 2026\] Evo-Attacker: Memory-Augmented Reinforcement Learning for Long-Horizon Tool Attacks on LLM-MAS](evo-attacker_memory-augmented_reinforcement_learning_for_long-horizon_tool_attac.md)
- [\[ICLR 2026\] The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](../../ICLR2026/llm_reasoning/the_illusion_of_diminishing_returns_measuring_long_horizon_execution_in_llms.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](../../ICML2026/llm_reasoning/toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)

</div>

<!-- RELATED:END -->
