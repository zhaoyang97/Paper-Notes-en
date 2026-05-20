---
title: >-
  [Paper Note] Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs
description: >-
  [NeurIPS 2025][Information Retrieval & RAG][Process Reward Model] This study systematically demonstrates that pure RL training (without explicit PRM supervision) implicitly induces strong process judgment capability…
tags:
  - "NeurIPS 2025"
  - "Information Retrieval & RAG"
  - "Process Reward Model"
  - "Reinforcement Learning"
  - "Self-PRM"
  - "Reasoning Verification"
  - "Introspective Evaluation"
date: 2026-05-08
content_hash: 1ab27a126bc8ac83
---

# Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2505.11227](https://arxiv.org/abs/2505.11227)  
**Code**: None (uses public models DeepSeek-R1, QwQ-32B)  
**Area**: Information Retrieval
**Keywords**: Process Reward Model, Reinforcement Learning, Self-PRM, Reasoning Verification, Introspective Evaluation

## TL;DR

This study systematically demonstrates that pure RL training (without explicit PRM supervision) implicitly induces strong process judgment capability; existing PRMs are even less effective than simple majority voting on strong reasoning models such as DeepSeek-R1 and QwQ-32B. The paper proposes Self-PRM, which allows a model to rerank its outputs using its own internal reward signal, consistently outperforming external PRMs.

## Background & Motivation

**Background**: Two main approaches exist for improving LLM reasoning: RL training (e.g., DeepSeek-R1 and QwQ-32B trained via GRPO/DAPO with final-answer correctness as the reward signal) and Process Reward Models (PRMs, which score each reasoning step, e.g., PRM800K, Math-Shepherd-PRM). Both directions have strong advocates, yet the relationship between them has rarely been studied systematically.

**Limitations of Prior Work**: PRMs face three fundamental limitations: (1) the granularity of reasoning steps is ambiguous—there is no unified standard for what constitutes "one step"; (2) high-quality step-level annotation is extremely costly, requiring mathematical experts to evaluate reasoning correctness step by step; (3) PRMs are systematically susceptible to reward hacking—models learn to produce reasoning steps that appear correct but are actually flawed in order to obtain high scores. The DeepSeek-R1 technical report explicitly notes that PRMs did not yield significant gains during training.

**Key Challenge**: On one hand, the PRM community invests substantial resources in constructing step-level supervision data and training dedicated reward models; on the other hand, the strongest reasoning models (DeepSeek-R1, QwQ-32B) are trained purely with RL. The central question is whether explicit process supervision is truly necessary, or whether RL training itself already implicitly endows models with process judgment capability.

**Goal**: (1) Systematically validate the relationship between RL training and PRM capability; (2) Evaluate the practical utility of existing PRMs on strong reasoning models; (3) Explore better ways to leverage a model's own reasoning capability for solution verification.

**Key Insight**: If RL training teaches a model what constitutes good reasoning steps during problem solving, that understanding should also be applicable to evaluating others' reasoning steps—i.e., problem-solving ability and judgment ability are two sides of the same coin.

**Core Idea**: Problem-solving capability and process judgment capability co-evolve during pure RL training, rendering external PRMs redundant for already strong reasoning models.

## Method

### Overall Architecture

This paper is a hybrid empirical study and method proposal. It first systematically evaluates PRM capability across multiple model families on ProcessBench, then analyzes capability co-evolution along the RL training curve, and finally proposes two practical frameworks: Self-PRM and Self-REF.

### Key Designs

1. **Systematic Evaluation of PRM Capability**:

    - **Function**: Reveal the impact of different training paradigms on process judgment capability.
    - **Mechanism**: Three categories of models are evaluated on ProcessBench—(1) pure RL-trained reasoning models (DeepSeek-R1, QwQ-32B) used as Generative PRMs via prompting; (2) Discriminative PRMs explicitly trained on PRM data (e.g., Skywork-PRM-7B, Math-Shepherd-PRM-7B); (3) instruction-tuned models (Qwen2.5 series). Key finding: pure RL models achieve F1 scores (DeepSeek-R1: 83.5, QwQ-32B: 83.7) that comprehensively surpass all explicitly trained PRMs (the strongest, Qwen2.5-Math-PRM-72B, reaches only 78.3), even outperforming many PRMs with far larger parameter counts.
    - **Design Motivation**: Verify the core hypothesis that "PRM capability can emerge naturally from RL training."

2. **Self-REF (Self-Reference Enhanced PRM)**:

    - **Function**: Leverage the model's own generated solutions as reference signals to enhance PRM judgment.
    - **Mechanism**: When evaluating reasoning correctness, the model first generates a reference solution of its own, then judges the correctness of the target solution based on that reference. For instruction-tuned models without RL training (e.g., Qwen2.5-32B-Instruct), Self-REF yields substantial gains (F1: 45.1→63.9, +18.8). For RL-trained models that already possess strong PRM capability (QwQ-32B: 83.7→83.0), the effect is neutral or slightly negative—indicating that RL models have already internalized sufficient process understanding.
    - **Design Motivation**: Explore the utility of self-generated reasoning as a weak supervision signal for models at different training stages.

3. **Self-PRM (Self-as-PRM Reranking Strategy)**:

    - **Function**: Enable strong reasoning models to replace external PRMs with their own reward signal for Best-of-N selection.
    - **Mechanism**: Given $k$ sampled outputs, instead of scoring with an external PRM (e.g., Qwen2.5-Math-PRM-72B), the model evaluates each output itself and selects the one with the highest self-assigned score as the final answer. On AIME24/25 and CNMO24, Self-PRM consistently outperforms external PRMs and majority voting at large sampling budgets ($k=32/64$).
    - **Design Motivation**: Since a model's internal process understanding is better aligned with its own reasoning behavior, self-evaluation should be more accurate than external evaluation.

### Co-Evolution of PRM Capability During RL Training

Using Qwen2.5-7B-Base + DAPO trained from scratch, PRM F1 on ProcessBench is evaluated every 10 steps. F1 and problem-solving accuracy rise nearly in tandem, with F1 improvement often leading accuracy improvement—especially in early training—suggesting that models first learn to "recognize good reasoning steps" before they learn to "generate good reasoning steps." Chi-squared tests significantly reject the null hypothesis that problem-solving ability and judgment ability are independent across all tested models.

## Key Experimental Results

### ProcessBench Evaluation (Average F1)

| Model Type | Model | Avg. F1 |
|---------|------|---------|
| Discriminative PRM | Math-Shepherd-PRM-7B | 31.5 |
| Discriminative PRM | Skywork-PRM-7B | 42.1 |
| Discriminative PRM | Qwen2.5-Math-PRM-7B | 73.5 |
| Discriminative PRM | Qwen2.5-Math-PRM-72B | 78.3 |
| Generative PRM (RL) | **DeepSeek-R1** | **83.5** |
| Generative PRM (RL) | **QwQ-32B** | **83.7** |

### Self-PRM vs. External PRM vs. Majority Voting (QwQ-32B, k=32)

| Strategy | AIME24 | AIME25 | CNMO24 |
|------|--------|--------|--------|
| Majority Voting | 86.7 | 76.7 | 83.3 |
| BoN w/ PRM (72B) | 86.7 | 76.7 | 83.3 |
| **BoN w/ Self-PRM** | **90.0** | **80.0** | **88.9** |

### Self-REF Results

| Model | Original F1 | +Self-REF F1 | Change |
|------|---------|-------------|------|
| Qwen2.5-32B-Instruct | 45.1 | 63.9 | +18.8 |
| R1-Distill-Qwen-32B | 76.7 | 79.3 | +2.6 |
| QwQ-32B | 83.7 | 83.0 | -0.7 |
| DeepSeek-R1 | 83.5 | 81.1 | -2.4 |

### Key Findings

- Pure RL-trained models comprehensively outperform all specialized PRMs on PRM tasks, including large PRMs with 72B parameters.
- External PRMs (Qwen2.5-Math-PRM-72B) provide Best-of-N reranking performance on DeepSeek-R1/QwQ-32B that is on par with or worse than simple majority voting—deploying a 72B PRM is entirely wasteful.
- Self-PRM begins to show advantages at $k \geq 16$ and achieves optimal performance at $k=32$ (AIME24: 90.0 vs. 86.7).
- Self-REF greatly benefits instruction-tuned models (+18.8 F1) but is ineffective or slightly harmful for RL-trained models.
- Self-PRM's critical weakness: accuracy on hard problems is extremely low (<10%), and it frequently misidentifies incorrect solutions as correct.

## Highlights & Insights

- **Challenges the necessity assumption of PRMs**: This is the first work to systematically argue that pure RL training can implicitly produce PRM capability, raising important questions about the value of the substantial resources invested in the PRM research line. The core message is: rather than expending great effort annotating step-level data to train PRMs, it is more effective to simply perform more RL training.
- **The finding that F1 leads accuracy is particularly interesting**: Models first learn to "judge quality" before they learn to "perform well themselves," suggesting that process understanding may be a prerequisite for generative capability, rather than the other way around.
- **Practical value of Self-PRM**: Simply allowing a model to self-evaluate surpasses a 72B external PRM, providing direct engineering guidance—no additional PRM model is needed at deployment time.

## Limitations & Future Work

- Self-PRM achieves very low accuracy on hard problems, possibly because the model's reward alignment is insufficient—when the model's own problem-solving success rate is low, its self-evaluation is also unreliable.
- The analysis is limited to mathematical reasoning; whether the findings hold for code generation, logical reasoning, and other domains remains unverified.
- The paper does not explore how to actively reinforce PRM capability during RL training; the current findings only document its natural emergence.
- The slight negative effect of Self-REF on RL models suggests that self-generated references may introduce noise, requiring more intelligent reference selection mechanisms.

## Related Work & Insights

- **vs. PRM800K/ProcessBench**: These works invest substantial human effort in step-level annotation to train PRMs; this paper demonstrates that pure RL can match or exceed such results, potentially redirecting research priorities in the field.
- **vs. PRIME**: PRIME introduces implicit reward mechanisms to bypass step-level annotation, consistent with this paper's findings—explicit step-level supervision is not necessary.
- **vs. GenPRM**: GenPRM uses code verification to improve PRMs, representing an alternative path to avoiding manual annotation that is complementary to the Self-PRM approach.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically challenges the widely accepted assumption of PRM necessity; the Self-PRM design is intuitively well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive model comparisons, statistical tests, training curve analysis, and honest analysis of Self-PRM's limitations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rigorous argumentation, and transparent acknowledgment of Self-PRM's limitations.
- Value: ⭐⭐⭐⭐⭐ Provides important strategic insights for the PRM field—the entire technical direction may warrant reconsideration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning](symrtlo_enhancing_rtl_code_optimization_with_llms_and_neuron-inspired_symbolic_r.md)
- [\[ICLR 2026\] Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement](../../ICLR2026/information_retrieval/judges_verdict_a_comprehensive_analysis_of_llm_judge_capability_through_human_ag.md)
- [\[ACL 2026\] ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs](../../ACL2026/information_retrieval/chairo_contextual_hierarchical_analogical_induction_and_reasoning_optimization_f.md)
- [\[ACL 2026\] An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs](../../ACL2026/information_retrieval/an_iterative_utility_judgment_framework_inspired_by_philosophical_relevance_via_.md)
- [\[ACL 2026\] To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs](../../ACL2026/information_retrieval/to_lie_or_not_to_lie_investigating_the_biased_spread_of_global_lies_by_llms.md)

</div>

<!-- RELATED:END -->
