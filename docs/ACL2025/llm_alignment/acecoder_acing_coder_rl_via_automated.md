---
title: >-
  [Paper Note] AceCoder: Acing Coder RL via Automated Test-Case Synthesis
description: >-
  [ACL 2025][LLM Alignment][Code Generation] The study constructs AceCode-87K (87K coding problems + 1.38M automatically synthesized test cases) to train a code-specific Reward Model (the 7B model outperforms the 340B Nemotron). Best-of-N sampling improves Llama-3.1-8B by 8.9 points on average. Direct R1-style RL from a base model for only 80 steps improves HumanEval+ by 22.5%.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Code Generation"
  - "Reinforcement Learning"
  - "Reward Model"
  - "test case synthesis"
  - "R1-style training"
date: 2026-05-08
content_hash: 88d7ae84e3e08382
---

# AceCoder: Acing Coder RL via Automated Test-Case Synthesis

**Conference**: ACL 2025  
**arXiv**: [2502.01718](https://arxiv.org/abs/2502.01718)  
**Code**: [https://tiger-ai-lab.github.io/AceCoder](https://tiger-ai-lab.github.io/AceCoder)  
**Area**: Other  
**Keywords**: Code Generation, Reinforcement Learning, Reward Model, test case synthesis, R1-style training

## TL;DR
The study constructs AceCode-87K (87K coding problems + 1.38M automatically synthesized test cases) to train a code-specific Reward Model (the 7B model outperforms the 340B Nemotron). Best-of-N sampling improves Llama-3.1-8B by 8.9 points on average. Direct R1-style RL from a base model for only 80 steps improves HumanEval+ by 22.5%.

## Background & Motivation
**Background**: Code generation models primarily rely on SFT for improvements, while the potential of RL remains under-explored. The success of RL in mathematics (such as DeepSeek-R1) suggests that the coding domain can also benefit.

**Limitations of Prior Work**: (1) Code evaluation requires executing test cases (unlike mathematics, which can rely on direct string matching), leading to a lack of reliable reward signals; (2) Large-scale coding datasets with test cases are scarce—APPS/TACO rely on manual annotation, and previous RL works were limited to <5K samples.

**Key Challenge**: Existing general reward models (such as Skywork) fail to generalize to code evaluation, whereas manual annotation of test cases is too costly to scale.

**Goal**: Establish an automated pipeline to generate large-scale, reliable test cases from code datasets → train a code-specific RM → utilize the RM for Best-of-N sampling or RL.

**Key Insight**: Batch "imagining" test cases using GPT-4o-mini combined with execution filtering by a strong code model yields low-cost, high-quality, and large-scale test data.

**Core Idea**: Fully automated test case synthesis + execution filtering → code RM training → RL/Best-of-N, scaling RL for code generation to 87K samples.

## Method

### Overall Architecture
A five-phase pipeline: (1) Generating LeetCode-style problems and ~20 test cases from a seed code dataset using GPT-4o-mini; (2) Using Qwen2.5-Coder-32B to generate solutions and filtering out test cases that fail to execute; (3) Constructing preference pairs from the filtered data (utilizing Bradley-Terry loss); (4) Training AceCode-RM (7B/32B); (5) Employing the RM for Best-of-N sampling or Reinforcement++ RL.

### Key Designs

1. **Automated Test-Case Synthesis + Filtering (AceCode-87K)**:

    - **Function**: Starting from 124K Python functions (Magicoder-Evol/OSS/StackPy), GPT-4o-mini rewrites them into LeetCode-style problems and generates ~20 test cases.
    - **Mechanism**: Using Qwen2.5-Coder-32B to generate solutions for each problem and executing the test cases, **removing test cases failed by the solutions and discarding problems with fewer than 5 remaining test cases**.
    - **Final Scale**: 87,149 problems and 1,380,000 cleaned test cases (an average of 15.87 per problem).
    - **Quality Verification**: Only 3 invalid cases out of 200 manually inspected samples (1.5% error rate).

2. **Preference Pair Construction and RM Training**:

    - **Function**: Sampling 16 programs per problem, sorting them by test pass rate, and constructing preference pairs.
    - **Mechanism**: Selective pairing—pairing is performed only when $s_i > s_j + 0.4$, $s_i > 0.8$, and $s_j > 0$ to ensure high-quality preference signals.
    - **Training**: Bradley-Terry loss with Qwen2.5-Coder-7B-Instruct as the backbone, 1 epoch, running on 8×A100 GPUs for 24 hours.
    - **Results**: 307,509 valid preference pairs generated from 46,618 problems.

3. **RL Training (Reinforcement++)**:

    - **Function**: Conducting RL with either AceCode-RM or binary rule-based rewards (all tests passed = 1, otherwise 0).
    - **Mechanism**: Reinforcement++ algorithm (more efficient than PPO, requiring no value model); training exclusively on the top 25% hardest problems in AceCode-87K.
    - **R1-style Experiment**: Direct RL starting from Qwen2.5-Coder-7B-**base** (without SFT) for only 80 steps.
    - **Key Findings**: **Rule-based rewards are more effective than RM rewards**, as RM training suffers from reward hacking during the RL process.

### Loss & Training
RM Training: Standard Bradley-Terry loss. RL Training: Reinforcement++ algorithm, rollout batch=256, 8 samples/question, lr=5e-7, 1 episode, 6 hours on 8×H100 GPUs.

## Key Experimental Results

### Best-of-N Sampling Results (Llama-3.1-8B-Instruct)

| Method | HumanEval | MBPP | BigCodeBench-C | LiveCodeBench | Average |
|------|-----------|------|----------------|---------------|------|
| Greedy | 68.9 | 67.2 | 38.5 | 18.0 | 40.9 |
| AceCode-RM-7B (Best-of-64) | 81.7 | 74.6 | 47.8 | 27.6 | 49.3 |
| **AceCode-RM-32B (Best-of-64)** | **85.4** | **72.0** | **48.5** | **31.0** | **49.8** |

### R1-style RL Training (Direct RL from Base Model, 80 Steps)

| Configuration | HumanEval+ | MBPP+ | BigCodeBench-I |
|------|-----------|-------|----------------|
| Qwen2.5-Coder-7B-Base | 61.6 | 76.9 | 40.2 |
| + AceCoder-Rule (80 Steps) | **84.1 (+22.5)** | **82.3 (+5.4)** | **43.2 (+3.0)** |
| + AceCoder-RM (80 Steps) | 83.5 (+21.9) | 80.2 (+3.3) | 36.8 (-3.4) |

### Ablation Study

| Configuration | HumanEval | MBPP | BigCodeBench-H | Average |
|------|-----------|------|----------------|------|
| RM w/o Filter | 73.8 | 73.3 | 17.6 | 45.2 |
| **RM w/ Filter** | **77.4** | **76.5** | **20.3** | **47.7** |

### Key Findings
- AceCode-RM-7B outperforms the 340B Nemotron-Reward by 7.5 points (66.9% vs. 54.5%) on the RM Bench coding category, achieving a level where a 7B model surpasses a 50× larger general-purpose RM.
- Best-of-N sampling provides a massive boost for weaker models (Mistral +13 points) but shows limited improvements on stronger models (Qwen-Coder +4 points).
- R1-style training achieves astonishing efficiency, boosting HumanEval+ by 22.5% in just 80 steps (48 H100-GPU-hours).
- **Rule-based rewards > RM rewards** for RL training, because RM suffers from reward hacking during the RL process.
- Test case filtering yields an average improvement of 2.5 points, especially on hard problems (BigCodeBench-Hard +2.7).
- The choice of RM backbone is crucial: the Qwen2.5-Coder backbone outperforms the Llama-3.1 backbone by 11.6 points (HumanEval).

## Highlights & Insights
- The **fully automated test case synthesis pipeline** is the core contribution: GPT-4o-mini generation + strong model execution filtering = low cost, large scale, and high quality (with only a 1.5% error rate). This pipeline can be transferred to any code RL scenario that requires test execution.
- The efficiency of **R1-style training** is remarkably high: practicing direct RL starting from a base model (bypassing SFT) reaches a level close to SFT + RL in only 80 steps, challenging the conventional paradigm where SFT is considered a prerequisite for RL.
- The finding that **rule-based rewards outperform RM rewards** is thought-provoking: it indicates that RMs are easily exploited (reward hacking) in RL loops, making simple yet precise reward signals (binary pass/fail) more effective.

## Limitations & Future Work
- Synthetic test cases still contain noise: passing all tests does not guarantee program correctness, as edge cases might be missed.
- RL shows limited average gain (+0.7 points) on already strong models (e.g., Qwen2.5-Coder-7B-Instruct).
- Test synthesis relies solely on GPT-4o-mini; using stronger models (such as GPT-4) could further improve quality.
- Only one training episode was conducted, leaving further RL scaling strategies unexplored.

## Related Work & Insights
- **vs. Skywork-Reward**: General-purpose RMs fall significantly short on code tasks, making code-specific RMs highly necessary and effective.
- **vs. DeepSeek-R1**: Adapts its "direct RL from base" philosophy and proves its viability in the coding domain.
- **vs. APPS/TACO**: While predecessor works rely on manual test-case annotations, AceCoder utilizes fully automated synthesis to scale the dataset size by over 10×.
- Insight: The "synthesis + filtering" paradigm of test cases can be extended to any task with executable validation (e.g., SQL, data analysis).

## Rating
- Novelty: ⭐⭐⭐⭐ Valuable contribution with the automated test synthesis pipeline and R1-style code RL.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-model × Multi-benchmark + Best-of-N + RL + R1 + detailed ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the pipeline and comprehensive empirical data.
- Value: ⭐⭐⭐⭐⭐ First to scale code RL to 87K samples, making it highly practical.

## Highlights & Insights
- The **fully automated test synthesis pipeline** resolves the reward signal bottleneck in code RL—using test pass rates as preference signals is more objective than manual annotations.
- The **success of direct R1-style RL from base** suggests that coding is an ideal domain for RL due to its natural execution-based verifiers.
- The **AceCode-87K dataset** itself is a significant contribution—comprising 87K problems and 1.38M cleaned test cases.

## Limitations & Future Work
- Covers only Python. Residual issues remain after test case filtering. RL training is highly sensitive to hyperparameters.

## Rating
- Novelty: ⭐⭐⭐⭐ First fully automated test synthesis → RM → RL pipeline for code.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Best-of-N + RL + R1-style + multi-benchmark.
- Writing Quality: ⭐⭐⭐⭐ Complete pipeline description and rigorous experimental design.
- Value: ⭐⭐⭐⭐⭐ Unlocks the immense potential of RL for code generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TableDreamer: Progressive and Weakness-Guided Data Synthesis from Scratch for Table Instruction Tuning](tabledreamer_progressive_and_weakness-guided_data_synthesis_from_scratch_for_tab.md)
- [\[ACL 2026\] AgentV-RL: Scaling Reward Modeling with Agentic Verifier](../../ACL2026/llm_alignment/agentv-rl_scaling_reward_modeling_with_agentic_verifier.md)
- [\[CVPR 2025\] Jailbreaking the Non-Transferable Barrier via Test-Time Data Disguising](../../CVPR2025/llm_alignment/jailbreaking_the_non-transferable_barrier_via_test-time_data_disguising.md)
- [\[NeurIPS 2025\] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](../../NeurIPS2025/llm_alignment/limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)
- [\[ICLR 2026\] Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study](../../ICLR2026/llm_alignment/safety_subspaces_are_not_linearly_distinct_a_fine-tuning_case_study.md)

</div>

<!-- RELATED:END -->
