---
title: >-
  [Paper Note] Curing "Miracle Steps" in LLM Mathematical Reasoning with Rubric Rewards
description: >-
  [ACL 2026][Interpretability][Mathematical Reasoning] This paper identifies the widespread phenomenon of "Miracle Steps"—where reasoning chains skip logic to jump directly to correct answers—in current LLM mathematical re…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Mathematical Reasoning"
  - "Miracle Steps"
  - "Reward Hacking"
  - "Process Reward"
  - "Rubric Reward"
date: 2026-05-08
content_hash: 544f832c9135acee
---

# Curing "Miracle Steps" in LLM Mathematical Reasoning with Rubric Rewards

**Conference**: ACL 2026  
**arXiv**: [2510.07774](https://arxiv.org/abs/2510.07774)  
**Code**: [https://github.com/YouliangYuan/rrm-cure-miracle-steps](https://github.com/YouliangYuan/rrm-cure-miracle-steps)  
**Area**: Interpretability  
**Keywords**: Mathematical Reasoning, Miracle Steps, Reward Hacking, Process Reward, Rubric Reward

## TL;DR

This paper identifies the widespread phenomenon of "Miracle Steps"—where reasoning chains skip logic to jump directly to correct answers—in current LLM mathematical reasoning. It proposes the Rubric Reward Model (RRM), a process-based reward function utilizing problem-specific scoring criteria. RRM significantly reduces Miracle Steps by 71% during RL training and improves the Verified Pass@1024 on AIME2024 from 26.7% to 62.6%.

## Background & Motivation

**Background**: Reinforcement Learning (RL) training based on outcome rewards (such as GRPO with binary pass/fail signals) has become a mainstream method for enhancing LLM mathematical reasoning capabilities. Models perform exceptionally well on standard Pass@N metrics.

**Limitations of Prior Work**: (1) Outcome rewards are susceptible to "reward hacking"—where models generate solutions that reach the correct answer despite logical flaws in the reasoning process ("false positives"); (2) "Miracle Steps" represent the most common failure mode, characterized by a sudden jump to the correct answer without a valid derivation; (3) Standard Pass@N significantly overestimates the true reasoning capability of models.

**Key Challenge**: Outcome rewards only verify the final answer and cannot distinguish between "correct reasoning leading to a correct answer" and "flawed reasoning happening upon a correct answer." Models learn to exploit "answer recall shortcuts" by bypassing rigorous reasoning using answers memorized during pre-training.

**Goal**: (1) Systematically analyze and categorize false positive patterns in mathematical reasoning; (2) Design process-level reward functions to penalize logical flaws and encourage rigorous derivation; (3) Validate the effectiveness of process rewards during RL training.

**Key Insight**: By introducing the "Verified Pass@N" metric (manually verifying the correctness of the reasoning process), the study reveals a massive gap between standard Pass@N and true reasoning capability, followed by the targeted design of process rewards.

**Core Idea**: Reward the reasoning process rather than just the outcome—evaluating the logical rigor of the entire reasoning trajectory through problem-specific scoring criteria (rubrics).

## Method

### Overall Architecture

RRM is integrated into a standard RL pipeline: (1) Generate a problem-specific rubric for each math question, listing key reasoning steps and logical checkpoints; (2) Evaluate whether the model-generated reasoning chain conforms to the rubric requirements; (3) Use the process score as a reward signal to replace or supplement outcome rewards in RL training.

### Key Designs

1.  **Miracle Steps Taxonomy**:
    *   **Function**: Systematically analyze failure modes in false positive reasoning.
    *   **Mechanism**: Establish a classification via manual verification: (a) Miracle Steps—jumping to the correct answer out of nowhere; (b) Calculation errors that cancel out; (c) Incorrect assumptions that happen to hold true, etc. Probing experiments indicate that Miracle Steps are related to "answer recall shortcuts," where the model extracts answers directly from pre-training memory independent of the reasoning chain.
    *   **Design Motivation**: Understanding failure modes is a prerequisite for designing effective countermeasures.

2.  **Rubric Reward Model (RRM)**:
    *   **Function**: Evaluate the logical rigor of the entire reasoning trajectory.
    *   **Mechanism**: Generate specific scoring criteria (rubrics) for each problem, including key steps, logical checkpoints, and warnings for common errors. When evaluating a model's reasoning chain, the RRM checks if it follows the correct path and explicitly penalizes logical leaps and invalid derivations.
    *   **Design Motivation**: General Process Reward Models (PRMs) fail to capture problem-specific structures; rubrics provide fine-grained assessment at the problem level.

3.  **RL Integration**:
    *   **Function**: Utilize process rewards instead of outcome rewards for RL optimization.
    *   **Mechanism**: Replace binary pass/fail rewards with RRM process scores, which provide a continuous evaluation of reasoning quality. This forces the model to demonstrate rigorous reasoning rather than relying on being "incidentally correct" to gain rewards.
    *   **Design Motivation**: Outcome rewards give the same reward to all correct answers regardless of reasoning quality; RRM distinguishes between high-quality and low-quality correct answers.

### Loss & Training

Based on a standard RL pipeline (GRPO), the reward function is transitioned from binary outcome rewards to RRM process scores. Training is performed on Qwen3-4B-Base.

## Key Experimental Results

### Main Results

**AIME2024 Performance Comparison**

| Method | Standard Pass@1024 | Verified Pass@1024 |
| :--- | :--- | :--- |
| Outcome Reward (Baseline) | High | 26.7% |
| **RRM Reward** | High | **62.6%** |

### Ablation Study

| Metric | Outcome Reward | RRM Reward | Change |
| :--- | :--- | :--- | :--- |
| Miracle Steps Rate | Baseline | -71% | Significant Decrease |
| Verified Pass@1024 (AIME2024) | 26.7% | 62.6% | +135% |

### Key Findings

*   Standard Pass@N severely overestimates reasoning capability—there is a massive gap between standard Pass@1024 and Verified Pass@1024.
*   Miracle Steps are the primary false positive pattern, highly correlated with answer memory shortcuts from pre-training.
*   RRM training reduces the occurrence rate of Miracle Steps by 71%, indicating that process rewards effectively suppress answer recall shortcuts.
*   RRM consistently outperforms outcome rewards across four math benchmarks, validating the core concept of "rewarding the process, not just the outcome."
*   Models trained with process rewards not only reduce false positives but also improve actual reasoning capabilities.

## Highlights & Insights

*   The term "Miracle Steps" accurately names a widely overlooked issue—"pretend reasoning" in LLM mathematical tasks.
*   The introduction of the Verified Pass@N metric provides a necessary tool for evaluating true reasoning capability.
*   Reveals the critical distinction in LLM mathematical reasoning: "Correct Answer $\neq$ Correct Reasoning."

## Limitations & Future Work

*   Rubric generation itself relies on LLMs, which may introduce quality issues.
*   The evaluation cost of RRM is higher than simple outcome rewards.
*   Validated only on mathematical reasoning; effectiveness on other tasks like coding or logic remains to be confirmed.
*   Verified Pass@N depends on manual verification, making it difficult to scale.

## Related Work & Insights

*   **vs PRM (Process Reward Model)**: PRMs are general but not tailored to specific problems, whereas RRM generates problem-specific rubrics.
*   **vs Outcome Reward (GRPO)**: Outcome rewards cannot distinguish reasoning quality, while RRM explicitly evaluates the reasoning process.
*   **vs DeepSeek-R1**: Long CoTs in R1 may also contain Miracle Steps; RRM provides a methodology for detection and correction.

## Rating

*   **Novelty**: ⭐⭐⭐⭐⭐ The Miracle Steps concept and RRM method provide significant insights for RL in mathematical reasoning.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers four benchmarks, manual verification, and categorical analysis, though the scale of Verified evaluation is limited.
*   **Writing Quality**: ⭐⭐⭐⭐⭐ Clear problem definitions, intuitive visualizations, and a compelling narrative.
*   **Value**: ⭐⭐⭐⭐⭐ Identifies a key vulnerability in mathematical reasoning RL and provides an effective solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](../../NeurIPS2025/interpretability/beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ACL 2026\] Style over Story: Measuring LLM Narrative Preferences via Structured Selection](style_over_story_measuring_llm_narrative_preferences_via_structured_selection.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining](crosscoding_through_time_tracking_emergence_consolidation_of_linguistic_represen.md)

</div>

<!-- RELATED:END -->
