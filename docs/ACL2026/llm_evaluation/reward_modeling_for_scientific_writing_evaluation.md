---
title: >-
  [Paper Note] Reward Modeling for Scientific Writing Evaluation
description: >-
  [ACL 2026][LLM Evaluation][Reward Model] This paper proposes SciRM and SciRM-Ref, two open-source reward models tailored for scientific writing evaluation. By optimizing evaluation preferences and reasoning capabilities…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Reward Model"
  - "Scientific Writing Evaluation"
  - "GRPO"
  - "Multi-aspect Evaluation"
  - "Reasoning Enhancement"
date: 2026-05-08
content_hash: 04faa7cb1ce7af5c
---

# Reward Modeling for Scientific Writing Evaluation

**Conference**: ACL 2026  
**arXiv**: [2601.11374](https://arxiv.org/abs/2601.11374)  
**Code**: [https://github.com/UKPLab/acl2026-expert-rm](https://github.com/UKPLab/acl2026-expert-rm)  
**Area**: LLM Alignment / Scientific Writing Evaluation  
**Keywords**: Reward Model, Scientific Writing Evaluation, GRPO, Multi-aspect Evaluation, Reasoning Enhancement

## TL;DR

This paper proposes SciRM and SciRM-Ref, two open-source reward models tailored for scientific writing evaluation. By optimizing evaluation preferences and reasoning capabilities through two-stage Reinforcement Learning (GRPO), these models achieve fine-grained multi-aspect evaluation across various scientific writing tasks and generalize effectively to unseen evaluation tasks and criteria.

## Background & Motivation

**Background**: LLMs are widely utilized in scientific text generation (e.g., related work synthesis, peer review generation, manuscript revision). However, evaluating these outputs remains an open challenge. Currently, the most prevalent method is LLM-as-a-judge, which employs LLMs to provide scores directly.

**Limitations of Prior Work**: (1) General-purpose LLM judges struggle to reason with domain-specific knowledge and task-specific preferences, often resulting in self-contradictory evaluations; (2) existing reward models are optimized for general benchmarks (math, code, helpfulness, etc.) and are ill-suited for the nuances of scientific writing; (3) most reward models utilize pairwise comparisons and cannot perform independent assessments based on explicit criteria; (4) existing models are optimized for fixed scoring rubrics, leading to performance degradation when criteria change.

**Key Challenge**: Scientific writing evaluation requires dynamic adaptation to diverse tasks, aspects, and scoring criteria (where different aspects of the same text may even involve conflicting standards). Existing models solidify evaluation preferences during training, lacking the flexibility for adaptation at inference time.

**Goal**: To construct open-source reward models for scientific writing evaluation that can dynamically adapt to an explicit constitution (evaluation criteria + scoring rules + examples) during inference.

**Key Insight**: Evaluation is treated as a conditional generation task where the model accepts a constitution as context and explains and follows evaluation criteria through a reasoning process. A two-stage training approach teaches the model to "score according to criteria" and "reflect on criteria to correct its own reasoning."

**Core Idea**: Train a reward reasoning model using two-stage GRPO: the first stage focuses on learning to evaluate following a constitution, while the second stage enhances reflection and self-correction. Multi-task joint training is utilized to improve cross-task generalization.

## Method

### Overall Architecture

The input consists of three components: a task query $q$ (the scientific text to be evaluated), evaluation criteria $c$ (the constitution, containing scoring rules and descriptions), and scoring examples $e$. The model outputs a reasoning process $j$ (wrapped in `<reasoning>` tags) and a final score $s$ (wrapped in `<score>` tags). Training data encompasses multiple tasks including related work evaluation (binary labels) and peer review quality assessment (1-5 scale).

### Key Designs

1.  **Stage 1: Evaluation Preference Optimization**:
    *   **Function**: Teaches the model to perform accurate scientific writing evaluation based on a given constitution.
    *   **Mechanism**: Optimized using the GRPO algorithm. The reward function is designed hierarchically: -0.5 for format errors (no `<score>` tag), 0 for non-numeric outputs, 0.25 for numeric outputs outside the legal range, 0.5 for legal but incorrect scores, and 1.5 for correct scores. An additional length penalty function $f(L,T)$ is introduced to apply a quadratic penalty when outputs are too short or too long, preventing reward hacking.
    *   **Design Motivation**: The hierarchical reward design distinguishes between different error types (format vs. semantic), guiding the model toward incremental improvement. The length penalty addresses speculative behavior where the model might skip reasoning to output only the score.

2.  **Stage 2: Reasoning Enhancement (Self-Reflection)**:
    *   **Function**: Strengthens the model's self-reflection and correction capabilities, enabling it to re-examine criteria when uncertain.
    *   **Mechanism**: The model takes its own output from Stage 1, strips the score while retaining the reasoning, and is prompted to re-examine the criteria before providing a final score. The reward function considers the initial score $s_i$ and the final score $s_f$: self-correction ($s_i \neq s^*$ and $s_f = s^*$) receives the highest reward of 1.0, while degradation ($s_i = s^*$ and $s_f \neq s^*$) incurs the heaviest penalty of -1.0.
    *   **Design Motivation**: Encourages the model to actively correct errors during reasoning while penalizing unstable behavior where it shifts from a correct to an incorrect answer. This addresses the issue where constitutional AI might internalize rules and fail to adapt dynamically to new standards.

3.  **Multi-task Joint Training**:
    *   **Function**: Enhances the model's generalization capabilities across different scoring criteria and evaluation dimensions.
    *   **Mechanism**: Training data includes various scientific writing tasks (consistency, citation type, and grounding in related works; actionability, justification, verifiability, and helpfulness in peer reviews) across different scoring scales (binary and 1-5).
    *   **Design Motivation**: Single-task training tends to overfit specific criteria. Joint training enables the model to learn the "meta-capability of evaluation" rather than memorizing specific patterns.

### Loss & Training

The models are fine-tuned using LoRA based on Qwen2.5-7B. Both stages utilize the GRPO algorithm. Inference parameters include a temperature of 1.0 and top-p of 0.95. Mean and standard deviation are reported across 5 runs. The Stage 1 model is referred to as SciRM, and the two-stage model is SciRM-Ref.

## Key Experimental Results

### Main Results

| Task | SciRM-Ref | Qwen2.5-7B | Qwen3-8B | GPT-5.2 | Prometheus |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Review-Actionability | **Best** | Low | Med | High | Low |
| Review-Verifiability | **Best** | Med | Med | High | Med |
| Related Work-Consistency | **Best** | Low | Med | Med | Low |
| Related Work-Grounding | **Near Perfect** | Med | Med | High | Low |

### Ablation Study (Unseen Aspect/Task Generalization)

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| SciRM-Masked (2 aspects removed) | Outperforms most baselines on unseen aspects | Demonstrates generalization rather than overfitting |
| Unseen Task-Novelty Eval | 0.71+ alignment acc | Generalization to entirely unseen tasks |
| Unseen Task-Revision Eval | Exceeds most baselines | Effective cross-task transfer |

### Key Findings

*   Two-stage training consistently improves performance, with the second stage (reflection) providing the most benefit for tasks requiring strong reasoning.
*   SciRM-Masked still exceeds most baselines on unseen evaluation aspects, proving that the model learns the general structure of evaluation rather than overfitting to specific aspects.
*   On completely unseen novelty evaluation and manuscript revision tasks, SciRM still outperforms general baselines, demonstrating robust generalization.
*   Reasoning models such as Qwen3 and o3-mini perform exceptionally well on specific aspects (e.g., Grounding), likely due to their inherent reasoning capabilities.

## Highlights & Insights

*   The "Constitution-conditioned evaluation" design is highly valuable—it avoids internalizing evaluation criteria into the weights and instead treats them as explicit conditions for reasoning. This allows the same model to evaluate different tasks using different standards, significantly enhancing utility.
*   The reflection reward design in Stage 2 is ingenious: it considers not only final correctness but also whether the model corrected an error (reward 1.0) or degraded from a correct answer (penalty -1.0), effectively encouraging stable self-correction behavior.
*   The hierarchical reward function design can be transferred to other RLHF tasks requiring structured output—applying different penalties based on the severity and type of error.

## Limitations & Future Work

*   The study is based only on 7B models; larger models might exhibit different scaling behavior.
*   Training data is primarily focused on scientific literature in NLP/ML domains; generalization to other disciplines (e.g., biology, physics) has not yet been verified.
*   The quality of the constitution directly impacts evaluation performance; low-quality criteria may mislead the model.
*   Hyperparameter $k$ for the length penalty requires manual tuning; an adaptive scheme might be preferable.

## Related Work & Insights

*   **vs. Prometheus/Selene**: General LLM-as-a-judge models that are not optimized for scientific writing. SciRM significantly outperforms them through domain-specific training data and constitution-conditioned design.
*   **vs. DeepSeek-GRM**: General reward models that typically use pairwise evaluation and cannot perform pointwise assessment based on explicit criteria. SciRM's multi-aspect independent evaluation is better suited for scientific writing scenarios.

## Rating

*   Novelty: ⭐⭐⭐⭐ First application of reward reasoning models specifically for scientific writing evaluation, with a novel two-stage training design.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers seen/unseen aspects and tasks, includes multiple baselines and metrics, and provides detailed analysis.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure with well-argued motivation.
*   Value: ⭐⭐⭐⭐ Provides a practical, open-source solution for the automated evaluation of scientific writing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing](howtobench_holistic_evaluation_for_llms_capability_in_human-level_writing_using_.md)
- [\[ACL 2026\] SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models](scicustom_a_framework_for_custom_evaluation_of_scientific_capabilities_in_large_.md)
- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)
- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[ACL 2026\] SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark](sessionintentbench_a_multi-task_inter-session_intention-shift_modeling_benchmark.md)

</div>

<!-- RELATED:END -->
