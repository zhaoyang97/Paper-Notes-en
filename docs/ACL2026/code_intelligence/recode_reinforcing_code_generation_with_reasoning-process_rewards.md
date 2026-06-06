---
title: >-
  [Paper Note] ReCode: Reinforcing Code Generation with Reasoning-Process Rewards
description: >-
  [ACL 2026][Code Intelligence][Code Generation] ReCode trains a reward model capable of evaluating the quality of code reasoning processes through CRPL…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Code Generation"
  - "Process Reward"
  - "GRPO"
  - "Reward Model"
  - "Reasoning-Process"
date: 2026-05-08
content_hash: 5650e251f3be8c5a
---

# ReCode: Reinforcing Code Generation with Reasoning-Process Rewards

**Conference**: ACL 2026  
**arXiv**: [2508.05170](https://arxiv.org/abs/2508.05170)  
**Code**: https://github.com/ZJU-CTAG/ReCode  
**Area**: Code Intelligence / Reinforcement Learning / Reasoning-Process Reward  
**Keywords**: Code Generation, Process Reward, GRPO, Reward Model, Reasoning-Process

## TL;DR
ReCode trains a reward model capable of evaluating the quality of code reasoning processes through CRPL, and utilizes CG-GRPO to activate process rewards only when code execution is correct, thereby improving the Pass@1 of code generation models while avoiding reward hacking.

## Background & Motivation
**Background**: Code generation provides natural execution signals. Recently, many RL methods have directly used unit test pass rates as outcome rewards to train models, improving Pass@1 on benchmarks such as HumanEval, MBPP, and LiveCodeBench.

**Limitations of Prior Work**: Focusing solely on final test results ignores "why the model wrote this code." When two programs both pass tests, one reasoning process might be rigorous while the other is accidental; when both fail, one might have correct logic but flawed implementation details. Pure outcome rewards provide no fine-grained supervision for these differences.

**Key Challenge**: While reasoning process quality affects code correctness, directly incorporating a neural process reward into RL is prone to reward hacking. Models may learn to generate reasoning text that appears high-quality while the actual code remains incorrect.

**Goal**: To construct scalable reasoning-process preference data to train a reliable reasoning-process reward model, and to design a safe RL fusion method where process rewards supplement rather than replace execution correctness.

**Key Insight**: The reasoning process is treated as an intermediate product in code generation. Contrastive preferences are constructed using optimized/degraded reasoning variants, and execution results are used as a hard gate to ensure process rewards only function for correct code.

**Core Idea**: Process rewards are only credible when the result is correct; therefore, execution correctness should act as a "gate," allowing the reasoning reward to distinguish reasoning quality only among correct solutions.

## Method
ReCode consists of two core components. The first is Contrastive Reasoning-Process Reward Learning (CRPL), which trains a reward model using synthetic contrastive data. The second is Consistency-Gated GRPO (CG-GRPO), which integrates the reward model into RL while using test pass results to control whether the process reward is applied. This provides finer signals than binary test results while preventing the neural reward from being over-optimized on incorrect code.

### Overall Architecture
During training, the policy model generates outputs with a `<think>` and `<answer>` structure. The `<think>` section contains the reasoning process, and the `<answer>` section contains the code answer. Unit tests provide the outcome reward, format checks provide the format reward, and the CRPL reward model provides the process reward.

Unlike standard GRPO, the total reward in CG-GRPO is not a simple summation but is defined as $R = R_{fmt} + R_{out} + I(R_{out}=1) \times R_{proc}$. The reasoning process score only participates in the reward calculation when the generated code passes all tests. Thus, when multiple answers in a sampling group are correct, the process reward still provides a non-zero advantage; however, incorrect answers cannot receive extra rewards for elegant reasoning.

### Key Designs
1.  **CRPL Contrastive Reasoning-Process Reward Learning**:
    - **Function**: Training a reward model capable of distinguishing the quality of reasoning processes.
    - **Mechanism**: Qwen2.5-Coder-32B-Instruct first generates base reasoning processes for code problems, then generates optimized and degraded versions across three dimensions: factual accuracy, logical rigor, and logical coherence. This constructs three types of preference pairs: strong contrast, optimization, and degradation.
    - **Design Motivation**: Direct LLM scoring for reasoning often lacks calibration, whereas relative preferences are more stable. Optimized/degraded variants provide clear quality differences, helping the reward model learn fine-grained reasoning features.

2.  **LCB-RB Reasoning-Process Reward Benchmark**:
    - **Function**: Evaluating whether the reward model can truly judge reasoning-process quality.
    - **Mechanism**: 50 reasoning-solution pairs per problem are generated from LiveCodeBench v6, filtered by execution results, then checked for logical correctness and implementation consistency via GPT-4o. Finally, 219 preference pairs were obtained through manual review by two authors.
    - **Design Motivation**: Existing RewardBench focuses more on final answer quality and cannot specifically measure reasoning-process discrimination. LCB-RB fills this evaluation gap.

3.  **CG-GRPO Consistency-Gated**:
    - **Function**: Safely integrating process rewards into code RL.
    - **Mechanism**: The reward is composed of format, test results, and a gated process reward. If the code fails the test, the process reward is set to zero; if the code is correct, the process reward differentiates the reasoning quality of different correct solutions.
    - **Design Motivation**: Code tasks have strict execution signals, which should be treated as a hard constraint. Otherwise, the model might optimize the text preferred by the reward model rather than improving runnable code quality.

### Loss & Training
The CRPL reward model uses the Bradley-Terry pairwise loss: for each $(problem, preferred \ reasoning, rejected \ reasoning)$, it increases the difference between the score of the preferred reasoning and the rejected reasoning. Policy optimization is based on GRPO using group-relative advantage. The uniqueness of ReCode lies in the reward composition: the process reward is not added as a constant term but is hard-gated by the outcome reward. This provides a discriminative signal even when the entire group is correct, while avoiding reward hacking on incorrect samples.

## Key Experimental Results

### Main Results
On Qwen2.5-Coder-7B-Instruct, ReCode achieved an average improvement of 16.1% over the base model, a 6.7% relative improvement over outcome-only GRPO, and approached the average performance of GPT-4-Turbo.

| Model | HE | HE+ | MBPP | MBPP+ | LCB Easy | LCB Medium | LCB Hard | BigCode Full | BigCode Hard | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4-Turbo | 90.2 | 86.0 | 85.7 | 73.3 | 68.5 | 24.2 | 4.6 | 58.2 | 35.1 | 58.4 |
| Qwen2.5-Coder-14B | 89.6 | 87.2 | 86.2 | 72.8 | 61.0 | 11.3 | 2.8 | 48.4 | 22.2 | 53.5 |
| Qwen2.5-Coder-7B | 88.4 | 84.1 | 83.5 | 71.7 | 56.1 | 3.8 | 6.9 | 41.0 | 18.2 | 50.4 |
| +SFT | 66.2 | 57.3 | 73.3 | 63.5 | 34.1 | 3.8 | 0.0 | 39.9 | 13.5 | 39.1 |
| +GRPO | 85.9 | 81.1 | 86.7 | 75.1 | 58.5 | 15.1 | 9.7 | 52.0 | 29.7 | 54.9 |
| +ReCode | 90.9 | 86.0 | 87.0 | 76.2 | 68.3 | 20.8 | 9.7 | 54.0 | 33.8 | 58.5 |

The CRPL reward model performed strongly on LCB-RB and RewardBench reasoning subsets, indicating that synthetic contrastive preferences indeed learned transferable signals for process quality discrimination.

| Reward Model | Size | LCB-RB | RewardBench Code | RewardBench Math | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DeepSeek-V3 | 671B | 66.9 | 98.5 | 78.5 | 81.3 |
| GPT-4-Turbo | - | 63.7 | 98.1 | 67.3 | 76.4 |
| EURUS-RM | 7B | 57.0 | 92.8 | 79.9 | 76.5 |
| Qwen2.5-Coder-7B | 7B | 53.8 | 43.9 | 65.8 | 54.5 |
| +Score | 7B | 57.7 | 80.2 | 71.8 | 69.9 |
| +CRPL | 7B | 62.6 | 88.6 | 99.8 | 83.7 |

### Ablation Study
ReCode can migrate to mathematical tasks, transfer to Qwen3-4B, and complement compiler-based supervision.

| Setting | Metric | Baseline | +GRPO / Process Method | +ReCode | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-Math-7B | Avg on MATH500/Minerva/AIME24 | 24.5 | 48.0 | 51.5 | Process reward effective for Math |
| Qwen3-4B-Instruct | LiveCodeBench Avg | 30.7 | 32.5 | 36.1 | Transferable across model families |
| Compiler-based reward | LiveCodeBench Avg | 18.1 | 24.1 | 25.3 | ReCode outperforms compiler process rewards |
| ReCode + Compiler | LiveCodeBench Avg | 18.1 | 24.1 | 27.1 | Two types of signals are complementary |

Generation efficiency experiments show that ReCode wins not by generating longer reasoning, but by being shorter and more effective.

| Difficulty | GRPO Pass@1 | GRPO Avg Tokens | ReCode Pass@1 | ReCode Avg Tokens |
| :--- | :--- | :--- | :--- | :--- |
| Easy | 58.5 | 427.3 | 68.3 | 324.1 |
| Medium | 15.1 | 568.2 | 20.8 | 441.7 |
| Hard | 9.7 | 813.6 | 9.7 | 619.8 |

### Key Findings
- The improvement from ReCode stems from more rigorous reasoning rather than longer reasoning. On LiveCodeBench, the average generated tokens decreased by 23.4% while Pass@1 increased.
- Directly adding the process reward to the total reward leads to reward hacking; process scores quickly approach 1.0 while downstream performance plateaus.
- The consistency gate allows the process reward to compare quality only across correct programs, retaining fine-grained signals while preventing incorrect programs from benefiting from text quality.
- CRPL is stronger than score-based reward models, indicating that relative preference is more suitable for training process quality discriminators than scalar scoring.
- Preference data generated by a single strong generator is superior to that from mixed generators; the authors suggest this is due to a higher signal-to-noise ratio under a fixed budget.

## Highlights & Insights
- The paper transforms "reasoning process quality" from a slogan into a component that can be trained, evaluated, and integrated into RL. CRPL and LCB-RB form a complete closed loop.
- The consistency gate is a critical engineering judgment. Since code generation already has execution signals, neural rewards should yield to execution correctness rather than being added with equal weight.
- Results show that process supervision can improve token efficiency, which is vital for practical code models: shorter reasoning with higher accuracy implies lower inference costs.
- The effectiveness of ReCode on mathematical tasks suggests that reasoning-process rewards are not a code-specific trick but a transferable training paradigm.

## Limitations & Future Work
- Training output length is limited to 4K; effectiveness in long-context reasoning processes exceeding 30K tokens has not been verified.
- LCB-RB contains only 219 high-quality human-verified preference pairs; it is reliable but limited in coverage, requiring expansion to more problem types and languages.
- The policy model was only evaluated up to the 7B scale; it remains unclear if the marginal returns of process rewards hold for larger models.
- CRPL relies on strong code models to generate optimized/degraded reasoning; the quality of the generator affects the upper bound of the reward model.
- Process rewards may still learn stylistic preferences; although the execution gate reduces risk, the reward model itself requires more fine-grained auditing.

## Related Work & Insights
- **vs outcome-only GRPO**: Standard GRPO only considers test passes, resulting in sparse rewards that cannot distinguish between multiple correct solutions. ReCode continues to optimize reasoning quality within correct solutions.
- **vs StepCoder / PRLCoder**: These methods lean more toward implementation-level or compiler/test signals. ReCode focuses on the logical quality of the reasoning process and complements compiler signals.
- **vs General RewardBench reward models**: General reward models do not specifically examine code reasoning processes. CRPL, trained with optimized/degraded reasoning, is stronger on LCB-RB and RewardBench reasoning subsets.
- **Insights**: For verifiable tasks, neural process rewards should ideally serve as "ranking signals after verification" rather than replacing verification itself.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Process rewards are not a brand-new concept, but the combination of CRPL + consistency-gated RL is solid.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers code, reward models, math transfer, cross-model analysis, compiler supervision, and efficiency analysis; the chain of evidence is complete.
- Writing Quality: ⭐⭐⭐⭐☆ Methodology logic is clear, and experimental tables are extensive; some appendix details are dense and require careful tracking.
- Value: ⭐⭐⭐⭐⭐ High practical value for code model RL, process reward design, and prevention of reward hacking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)
- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](../../AAAI2026/code_intelligence/recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ICLR 2026\] Learning to Reason without External Rewards](../../ICLR2026/code_intelligence/learning_to_reason_without_external_rewards.md)
- [\[ACL 2026\] MARS²: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi_agent_tree_search_via_reinforcement_learning_for_code_genera.md)

</div>

<!-- RELATED:END -->
