---
title: >-
  [Paper Note] Curing "Miracle Steps" in LLM Mathematical Reasoning with Rubric Rewards
description: >-
  [ACL 2026][Interpretability][Mathematical Reasoning] This paper identifies a pervasive phenomenon in LLM mathematical reasoning termed "Miracle Steps"—instances where a reasoning chain leaps to the correct answer without…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Mathematical Reasoning"
  - "Miracle Steps"
  - "Reward Hacking"
  - "Process Reward"
  - "Rubric Reward"
date: 2026-05-08
content_hash: 9ca5b7bb5425be19
---

# Curing "Miracle Steps" in LLM Mathematical Reasoning with Rubric Rewards

**Conference**: ACL 2026
**arXiv**: [2510.07774](https://arxiv.org/abs/2510.07774)  
**Code**: [https://github.com/YouliangYuan/rrm-cure-miracle-steps](https://github.com/YouliangYuan/rrm-cure-miracle-steps)  
**Area**: Interpretability
**Keywords**: Mathematical Reasoning, Miracle Steps, Reward Hacking, Process Reward, Rubric Reward

## TL;DR

This paper identifies a pervasive phenomenon in LLM mathematical reasoning termed "Miracle Steps"—instances where a reasoning chain leaps to the correct answer without valid derivation—and proposes the Rubric Reward Model (RRM), a problem-specific process reward function that reduces Miracle Steps by 71% during RL training and improves Verified Pass@1024 on AIME2024 from 26.7% to 62.6%.

## Background & Motivation

**Background**: RL training with outcome rewards (e.g., GRPO with binary pass/fail signals) has become the dominant paradigm for enhancing LLM mathematical reasoning. Models achieve strong performance on standard Pass@N metrics.

**Limitations of Prior Work**: (1) Outcome rewards are susceptible to reward hacking—models produce solutions that reach the correct answer despite flawed reasoning ("false positives"); (2) "Miracle Steps" constitute the most prevalent failure mode, wherein a reasoning chain abruptly arrives at the correct answer without meaningful derivation; (3) Standard Pass@N substantially overestimates models' genuine reasoning capabilities.

**Key Challenge**: Outcome rewards validate only the final answer, rendering them unable to distinguish "correct answer via correct reasoning" from "correct answer via flawed reasoning." Models exploit memorized answers from pretraining to bypass rigorous derivation—a shortcut termed "answer recall."

**Goal**: (1) Systematically analyze and categorize false-positive patterns in mathematical reasoning; (2) Design process-level reward functions that penalize logical flaws and encourage rigorous derivation; (3) Empirically validate process rewards within RL training.

**Key Insight**: The paper introduces the "Verified Pass@N" metric—requiring human verification of reasoning correctness—to expose the substantial gap between standard Pass@N and genuine reasoning ability, then designs process rewards to address this gap.

**Core Idea**: Reward the reasoning process rather than the outcome alone—evaluating the logical rigor of an entire reasoning trajectory using problem-specific rubrics.

## Method

### Overall Architecture

RRM is integrated into a standard RL pipeline: (1) a problem-specific rubric is generated for each mathematical problem, enumerating key reasoning steps and logical checkpoints; (2) the model's generated reasoning chain is evaluated against the rubric; (3) the process score serves as the reward signal, replacing or supplementing outcome rewards during RL training.

### Key Designs

1. **Miracle Steps Taxonomy**:

    - **Function**: Systematically analyze failure modes in false-positive reasoning.
    - **Mechanism**: Human verification is used to establish a taxonomy: (a) Miracle Steps—reasoning chains that jump to the correct answer without derivation; (b) computational errors that cancel fortuitously; (c) incorrect assumptions that happen to hold, etc. Probing experiments indicate that Miracle Steps are associated with "answer recall shortcuts"—the model retrieves the answer directly from pretraining memory, independently of the reasoning chain.
    - **Design Motivation**: Understanding failure modes is a prerequisite for designing effective countermeasures.

2. **Rubric Reward Model (RRM)**:

    - **Function**: Evaluate the logical rigor of an entire reasoning trajectory.
    - **Mechanism**: A problem-specific rubric is generated for each question, comprising key reasoning steps, logical checkpoints, and warnings about common errors. When evaluating a model's reasoning chain, RRM checks whether the chain follows the correct reasoning path and explicitly penalizes logical leaps and invalid derivations.
    - **Design Motivation**: General-purpose process reward models (PRMs) cannot capture problem-specific reasoning structures; rubrics provide fine-grained, problem-level evaluation.

3. **RL Training Integration**:

    - **Function**: Replace outcome rewards with process rewards for RL optimization.
    - **Mechanism**: The binary pass/fail reward is replaced by RRM, which outputs a continuous score reflecting reasoning process quality. This prevents models from obtaining rewards merely by arriving at the correct answer incidentally; they must demonstrate rigorous reasoning.
    - **Design Motivation**: Outcome rewards assign identical rewards to all correct answers regardless of reasoning quality; RRM differentiates high-quality from low-quality correct solutions.

### Loss & Training

Built on a standard RL pipeline (GRPO), the reward function is replaced from a binary outcome reward to RRM's process score. Training is conducted on Qwen3-4B-Base.

## Key Experimental Results

### Main Results

**AIME2024 Performance Comparison**

| Method | Standard Pass@1024 | Verified Pass@1024 |
|--------|-------------------|-------------------|
| Outcome Reward (Baseline) | High | 26.7% |
| **RRM Reward** | High | **62.6%** |

### Ablation Study

| Metric | Outcome Reward | RRM Reward | Change |
|--------|---------------|-----------|--------|
| Miracle Steps Rate | Baseline | −71% | Substantially reduced |
| Verified Pass@1024 (AIME2024) | 26.7% | 62.6% | +135% |

### Key Findings

- Standard Pass@N substantially overestimates reasoning ability—a large gap exists between standard Pass@1024 and Verified Pass@1024.
- Miracle Steps constitute the dominant false-positive pattern and are strongly associated with answer recall shortcuts from pretraining.
- RRM training reduces the Miracle Steps rate by 71%, demonstrating that process rewards effectively suppress answer recall shortcuts.
- RRM consistently outperforms outcome rewards across four mathematical benchmarks, validating the core principle of "reward the process, not the outcome."
- Models trained with process rewards not only reduce false positives but also improve genuine reasoning ability.

## Highlights & Insights

- The concept of "Miracle Steps" precisely names a widely overlooked problem—"simulated reasoning" in LLM mathematical inference.
- The Verified Pass@N metric provides an essential tool for evaluating genuine reasoning ability.
- The paper reveals the critical distinction between "correct answer" and "correct reasoning" in LLM mathematical reasoning.

## Limitations & Future Work

- Rubric generation itself relies on LLMs and may be subject to quality issues.
- RRM evaluation incurs higher computational cost than simple outcome rewards.
- Validation is limited to mathematical reasoning; effectiveness on other reasoning tasks such as coding and logic remains to be confirmed.
- Verified Pass@N depends on human verification, making large-scale application difficult.

## Related Work & Insights

- **vs. PRM (Process Reward Model)**: PRMs are general-purpose but not tailored to specific problems; RRM generates problem-specific rubrics.
- **vs. Outcome Reward GRPO**: Outcome rewards cannot distinguish reasoning quality; RRM explicitly evaluates the reasoning process.
- **vs. DeepSeek-R1**: The long chain-of-thought in R1 may also contain Miracle Steps; RRM provides a method for detecting and remedying them.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The Miracle Steps concept and RRM approach offer important insights for mathematical reasoning RL.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four benchmarks, human verification, and taxonomic analysis; however, the scale of Verified evaluation is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem formulation is clear, visualizations are intuitive, and the narrative is compelling.
- **Value**: ⭐⭐⭐⭐⭐ Exposes a critical vulnerability in mathematical reasoning RL and provides an effective solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](../../NeurIPS2025/interpretability/beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ACL 2026\] Style over Story: Measuring LLM Narrative Preferences via Structured Selection](style_over_story_measuring_llm_narrative_preferences_via_structured_selection.md)
- [\[NeurIPS 2025\] Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning](../../NeurIPS2025/interpretability/cognitive_mirrors_exploring_the_diverse_functional_roles_of_attention_heads_in_l.md)
- [\[ICLR 2026\] RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs](../../ICLR2026/interpretability/radar_reasoning-ability_and_difficulty-aware_routing_for_reasoning_llms.md)

</div>

<!-- RELATED:END -->
