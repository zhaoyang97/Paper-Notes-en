---
title: >-
  [Paper Note] RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios
description: >-
  [ACL 2025 (Long Paper)][LLM Evaluation][rule-guided reasoning] This paper proposes RuleArena—a benchmark based on three real-world scenarios: airline baggage fees, NBA transaction rules, and tax regulations—to evaluate the ability of LLMs to perform reasoning while following complex natural language rules. Experiments reveal that even the strongest model (o1-preview) achieves less than 50% accuracy on the most difficult tasks, exposing systematic deficiencies of LLMs in rule…
tags:
  - "ACL 2025 (Long Paper)"
  - "LLM Evaluation"
  - "rule-guided reasoning"
  - "real-world benchmark"
  - "complex instruction following"
  - "tool augmentation"
date: 2026-05-08
content_hash: bbacdf5d2f379d2a
---

# RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios

**Conference**: ACL 2025 (Long Paper)  
**arXiv**: [2412.08972](https://arxiv.org/abs/2412.08972)  
**Code**: [https://github.com/skyriver-2000/rulearena](https://github.com/skyriver-2000/rulearena)  
**Area**: LLM Reasoning / Benchmark  
**Keywords**: rule-guided reasoning, LLM evaluation, real-world benchmark, complex instruction following, tool augmentation

## TL;DR
This paper proposes RuleArena—a benchmark based on three real-world scenarios: airline baggage fees, NBA transaction rules, and tax regulations—to evaluate the ability of LLMs to perform reasoning while following complex natural language rules. Experiments reveal that even the strongest model (o1-preview) achieves less than 50% accuracy on the most difficult tasks, exposing systematic deficiencies of LLMs in rule recall, rule discrimination, and mathematical computation.

## Background & Motivation
LLMs have been widely applied in real-world business scenarios, but the limitations of their domain knowledge often lead to erroneous outputs. For instance, Air Canada was ordered to pay compensation because its chatbot provided an incorrect refund policy. Existing instruction-following studies mainly focus on format and style (surface constraints such as length, topic, etc.) while ignoring scenarios where **rules act as logical constraints**—that is, instructions define the logical format of the reasoning process and the method of deriving the answer. On the other hand, most existing logical reasoning benchmarks are limited to first-order logic or simplified synthetic rules, failing to reflect the natural language complexity of real-world rules (nested conditions, multiple rules in parallel, dependencies between rules, etc.). Therefore, there is an urgent need for a benchmark grounded in real-world scenarios to evaluate the **rule-guided reasoning** capability of LLMs.

## Core Problem
Can LLMs accurately identify relevant rules, correctly apply them (including mathematical operations), and derive the correct answers when given complex natural language rules from the real world? This question directly relates to the reliable deployment of LLMs in high-risk scenarios such as law, finance, and policy enforcement.

## Method

### Overall Architecture
RuleArena is not a model-based method, but an **evaluation framework**. The overall workflow is as follows:
1. **Rule Collection**: Collect 95 rules from three real-world domains (10 for airline, 54 for NBA, and 31 for tax), with each rule averaging approximately 400 tokens.
2. **Question Construction**: Generate test questions of three difficulty levels for each domain, resulting in a total of 816 data points (300 for airline, 216 for NBA, and 300 for tax).
3. **Evaluation**: Provide the LLM with task instructions + complete domain rules + user instances, requiring the LLM to perform reasoning and calculation.
4. **Multi-dimensional Metrics**: Use GPT-4o to parse the LLM outputs into structured data, evaluating at both the problem level and the rule level.

### Key Designs
1. **Selection of Three Real-World Domains**:
    - **Airline Baggage Fees** (American Airlines Policy): Compute fees based on cabin class, route, and bag count/size/weight, with rules presented as Markdown tables.
    - **NBA Transactions** (2023 CBA Agreement): Determine whether team trades are compliant, involving numerous rules that are similar but have different application conditions, such as the salary cap and Bird Rights.
    - **Taxation** (IRS Forms): Calculate individual income tax, requiring the chaining of complex calculation processes across multiple tax forms.
   
2. **Difficulty Leveling Mechanism**:
    - Airline: Controlled by increasing the number of bags.
    - NBA: Controlled by increasing the number of teams, players, and transactions.
    - Taxation: Controlled by introducing more tax forms/regulations.

3. **Fine-Grained Evaluation Metric System** (Two groups, seven metrics in total):
    - **Problem-wise**: Precision $P(t)$ (whether only relevant rules are used), Recall $R(t)$ (whether all necessary rules are recalled), Application Correctness $AC(t)$ (whether the rule application/calculation is correct), and Accuracy $Acc(t)$ (whether the final answer is correct).
    - **Rule-wise**: $R(r)$ (whether a rule is invoked by the LLM when needed), $AC(r)$ (whether the calculation is correct when a rule is invoked), and $P(r)$ (whether an invoked rule is actually needed).
   
   This metric system can pinpoint exactly which step of the LLM failed (wrong rule selection? missed rules? or calculation error?), which is much more useful than traditional evaluations that only look at the final answer accuracy.

## Key Experimental Results

**Problem-wise Main Results (0-shot, Level 1 / Level 3 Accuracy):**

| Model | Airline L1 Acc | Airline L3 Acc | NBA L1 Acc | NBA L3 Acc | Tax L1 Acc | Tax L3 Acc |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Llama-3.1 70B | 0.01 | 0.00 | 0.40 | 0.22 | 0.01 | 0.00 |
| Qwen-2.5 72B | 0.01 | 0.00 | 0.44 | 0.30 | 0.10 | 0.00 |
| Llama-3.1 405B | 0.03 | 0.01 | 0.49 | 0.28 | 0.16 | 0.00 |
| Claude-3.5 Sonnet | 0.04 | 0.01 | 0.38 | 0.28 | 0.32 | 0.00 |
| GPT-4o | 0.02 | 0.00 | 0.40 | 0.24 | 0.42 | 0.00 |
| o1-preview | 0.54 | 0.21 | 0.44 | 0.24 | 0.72 | 0.19 |

**1-shot + Tool Augmentation (Airline Domain, Acc):**

| Model | 1-shot Default L1 | Tool Aug. L1 | 1-shot Default L2 | Tool Aug. L2 |
|------|:---:|:---:|:---:|:---:|
| Qwen 72B | 0.19 | 0.42 | 0.10 | 0.26 |
| GPT-4o | 0.32 | 0.44 | 0.16 | 0.33 |

### Ablation Study
- **Recall is highly linearly correlated with Accuracy**: $R(t)$ is the most critical factor affecting the final $Acc(t)$. Even if $AC(t)$ is high, missing a single rule can lead to an incorrect answer.
- **Non-essential rules are most easily missed**: Rules with the lowest recall are "non-essential" rules triggered under specific conditions (e.g., overweight baggage fees, special Bird Rights), whereas basic rules (e.g., base baggage fees) are almost never missed.
- **Compositional rules are most prone to calculation errors**: Rules with the lowest $AC(r)$ are "compositional" rules that require aggregating multiple intermediate results.
- **Similar rules are easily confused**: The precision in the NBA domain is the lowest because multiple rules appear similar but have different application conditions (e.g., different types of Mid-Level Exceptions).
- **Distractor rules significantly degrade performance**: In the taxation domain, introducing irrelevant but plausible-looking rules causes a notable drop in LLM performance; conversely, an equal amount of meaningless placeholders has almost no impact, demonstrating that the bottleneck lies in susceptibility to irrelevant rule distraction rather than context length.
- **Rule representation format has minimal impact**: Converting tabular rules into text "if-then" statements improves $R(r)$ slightly, but has little effect on the final $Acc$.
- **Tool augmentation is effective but insufficient**: Using a Python interpreter as a computation tool significantly improves accuracy (e.g., GPT-4o from 0.32 to 0.44), but is still far from perfect, indicating that the bottleneck lies not only in calculation but also in rule understanding and recall.

## Highlights & Insights
- **Highly realistic**: Rules in all three domains are extracted directly from actual policy documents (American Airlines policy, NBA CBA agreement, IRS tax forms) rather than being manually crafted toy tasks.
- **Exquisite evaluation metric design**: The "rule-following" process is decomposed into three dimensions: selection ($P$), recall ($R$), and correct application ($AC$), analyzed at both the problem and rule levels to precisely diagnose the failure modes of LLMs.
- **Practically meaningful findings**: (1) LLMs easily miss conditionally-triggered rules; (2) similar rules are easily confused; (3) mathematical calculations are unreliable even when the rules are correctly selected. These findings provide direct guidance for deploying LLMs in high-risk scenarios.
- **Transferable ideas**: The concept of fine-grained, rule-wise evaluation can be transferred to other structured reasoning tasks, such as legal reasoning and compliance checks.

## Limitations & Future Work
- **Limited domain coverage**: The benchmark includes only three domains (airlines, NBA, taxation) and has not yet covered other rule-dense fields such as law or medicine.
- **Static evaluation**: All rules are provided at once, failing to consider more realistic scenarios involving dynamic rule retrieval.
- **Automatic evaluation dependency on GPT-4o**: GPT-4o is used to parse LLM outputs into structured JSON, which might introduce errors of its own.
- **Lack of fine-tuning experiments**: The paper does not explore the performance of LLMs on these tasks when enhanced by fine-tuning or RAG.
- **Subjectivity in NBA annotations**: Questions in the NBA domain were manually constructed by annotators familiar with the rules, which may introduce minor bias, and detailed intermediate reasoning annotations are lacking.

## Related Work & Insights
- **vs. IFEval / FollowBench**: These benchmarks focus on instruction following at the format/style level (e.g., "answer in under 100 words"), whereas RuleArena focuses on the ability to follow **rules as reasoning constraints**, which is far more complex than stylistic constraints.
- **vs. FOLIO / RuleBench**: These benchmarks are based on first-order logic or formal rule representations, whereas rules in RuleArena are presented in natural language, representing real-world complexities like implicit conditions, tabular information, and multi-step calculations.
- **vs. GSM8K / MATH**: Mathematical reasoning benchmarks focus on calculation ability itself, whereas the main challenge of RuleArena lies in identifying and composing the correct rules from a large pool, with calculations being only the final step.

## Related Work & Insights
- The "rule recall" challenge exposed by this benchmark is highly related to retrieval accuracy issues in the RAG domain. Future work can explore architectures that decouple rule retrieval from LLM reasoning.
- The design concept of fine-grained evaluation metrics can be borrowed by benchmarks for other structured reasoning tasks.
- The effectiveness of tool augmentation suggests that for rule-intensive tasks, a hybrid system combining LLMs with symbolic reasoners might be a more reliable solution.

## Rating
- Novelty: ⭐⭐⭐⭐ The real-world scenarios and fine-grained rule evaluation are highlights, though the methodological innovation is relatively limited as a benchmark paper.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough analysis with 6 models × 3 domains × 3 difficulty levels × multiple ablations.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured and well-organized analysis, though some tables have highly dense data.
- Value: ⭐⭐⭐⭐ Provides direct cautionary insights for deploying LLMs in high-risk scenarios, and the evaluation metric system holds significant reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TripTailor: A Real-World Benchmark for Personalized Travel Planning](triptailor_a_real-world_benchmark_for_personalized_travel_planning.md)
- [\[ICCV 2025\] PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing](../../ICCV2025/llm_evaluation/phatnet_a_physics-guided_haze_transfer_network_for_domain-adaptive_real-world_im.md)
- [\[ACL 2025\] Com2: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models](com2_causal_commonsense.md)
- [\[ICCV 2025\] A Real-world Display Inverse Rendering Dataset](../../ICCV2025/llm_evaluation/a_real-world_display_inverse_rendering_dataset.md)
- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](../../ACL2026/llm_evaluation/taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)

</div>

<!-- RELATED:END -->
