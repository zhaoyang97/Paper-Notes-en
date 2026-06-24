---
title: >-
  [Paper Note] Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options
description: >-
  [ACL 2025][LLM (Other)][reflective judgment] Proposes the concept of "Reflective Judgment" to measure the ability of LLMs to reject choosing when all multiple-choice options are incorrect. It reveals that aligned models (such as GPT-4o) tend to blindly follow instructions to select incorrect options, whereas base models often perform better, and this ability emerges as model scale increases.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "reflective judgment"
  - "LLM alignment"
  - "multiple-choice"
  - "instruction following"
  - "critical reasoning"
date: 2026-05-08
content_hash: 8a94827d51d6cb1b
---

# Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options

**Conference**: ACL 2025  
**arXiv**: [2409.00113](https://arxiv.org/abs/2409.00113)  
**Code**: [GitHub](https://github.com/GracjanGoral/When-All-Options-Are-Wrong)  
**Area**: LLM/NLP  
**Keywords**: reflective judgment, LLM alignment, multiple-choice, instruction following, critical reasoning

## TL;DR
Proposes the concept of "Reflective Judgment" to measure the ability of LLMs to reject choosing when all multiple-choice options are incorrect. It reveals that aligned models (such as GPT-4o) tend to blindly follow instructions to select incorrect options, whereas base models often perform better, and this ability emerges as model scale increases.

## Background & Motivation
**Background**: Alignment techniques like RLHF/DPO aim to make LLMs more "helpful" and compliant with user instructions.

**Limitations of Prior Work**: Over-optimizing "helpfulness" can lead models to blindly follow erroneous instructions—when all options are wrong, the model still forces itself to select one instead of pointing out the issue.

**Key Challenge**: A fundamental tension exists between helpfulness (following instructions to select) and critical reasoning (rejecting incorrect options); alignment techniques inadvertently undermine the latter.

**Goal**: Quantitatively evaluate the "Reflective Judgment" of LLMs—the ability to identify and reject making a choice when no correct option is available.

**Key Insight**: Designing multiple-choice questions where all options are incorrect to test models under three difficulty conditions (Easy: explicitly informed that there might be no correct option; Standard: no hint; Hard: forced to choose one).

**Core Idea**: Alignment training makes models more "compliant" but less capable of saying "no"; reflective judgment is a more fundamental capability than helpfulness.

## Method

### Overall Architecture
Three datasets are graded and constructed: BAD (basic arithmetic, three difficulty levels), a subset of MMLU (400 cross-domain knowledge questions), and MedMCQA (200 high-risk medical decision questions). Each question provides only two incorrect options, tested under three reflection conditions. The Reflective Judgment Score is defined as the number of reflective actions divided by the total number of questions.

### Key Designs
1. **Three Reflection Conditions**:

    - Easy: Explicitly informed that "the correct answer might not be among the options."
    - Standard: No additional hint.
    - Hard: Required to choose one from A/B.
    - Design Motivation: To test the critical reasoning ability of models under different levels of instruction pressure.

2. **Reflective Judgment Score (RJ Score)**:

    - Function: Measures the percentage of times the model rejects incorrect options or provides the correct answer on its own.
    - Mechanism: RJ = number of reflective actions / total questions, where reflective actions include pointing out that no correct option exists or providing the correct answer not listed in the options.

3. **Nonsense Options Experiment**:

    - Function: Replaces options with completely unrelated nouns (e.g., "chair," "apple") to test whether the model still follows instructions in extreme cases.
    - Results: GPT-4o-mini and Claude 3 Haiku still select nonsense options, whereas Llama-405B and Qwen2-Math-7B achieve 100% rejection.

## Key Experimental Results

### Main Results (BAD Dataset, Standard Condition)

| Model | RJ Score | Baseline Accuracy | Type |
|------|----------|-----------|------|
| GPT-4o | 0.00% | 100% | RLHF |
| Claude 3 Sonnet | 0.00% | 90.9% | RLHF |
| Qwen2-Math-7B Base | 99.0% | 100% | Base |
| DeepSeekMath-7B RLHF | 100% | 100% | RLHF |
| Llama 3.1-405B | 42.5% | 94.5% | RLHF |
| Qwen2.5-32B Base | 90.9% | 100% | Base |

### Impact of Alignment

| Model Family | Base RJ | Instruct RJ | RLHF RJ |
|----------|---------|-------------|---------|
| Qwen2-Math-7B | 99% | - | 16% (Significant decrease) |
| Qwen2.5-7B | 40.9% | 0% (Decrease) | 0% |
| DeepSeekMath-7B | 92% | 12% (Decrease) | 100% (Recovered) |

### Key Findings
- **Alignment Undermines Reflective Judgment**: The RJ of Qwen2-Math-7B Base is 99%, which drops to 16% in the RLHF version.
- **Scaling Emergence**: Llama 3.1 goes from 8B (0%) → 70B (60%) → 405B (42.5%-100%), and Qwen2.5 from 7B (40.9%) → 32B (90.9%).
- **High-Risk does not imply More Cautious**: RJ on medical questions (MedMCQA) is similar to that on simple arithmetic; high-risk scenarios do not trigger more reflective behavior.
- **CoT Drastically Boosts RJ**: Using Chain-of-Thought improves RJ by more than 85%.
- **Humans Share the Same Issue**: Over 80% of 50 human participants still selected an incorrect option when no correct option was present.
- **HH-RLHF Dataset**: A random audit of 50 cases revealed that over 40% of the "preferred" answers were incorrect.

## Highlights & Insights
- Reveals a fundamental contradiction between alignment optimization and critical reasoning capability—being "more helpful" does not equate to being "more correct."
- Human annotators also tend to blindly follow instructions, and this bias propagates to models through RLHF datasets, forming a "feedback loop of bias."
- Reflective judgment is positively correlated with safety: models with high RJ are also better at rejecting harmful options (Llama-405B shows an 82% rejection rate in the Hard condition vs. 60% for GPT-4o-mini).
- DeepSeekMath's alignment instead restores RJ, demonstrating that carefully designed alignment can balance helpfulness and critical thinking.

## Limitations & Future Work
- Employs only a binary-choice format, failing to cover cases with more options.
- BAD is a synthetically constructed simple arithmetic dataset, which deviates from real-world scenarios.
- The human experiment consists of only 50 participants, representing a limited sample size.
- Fails to investigate in depth why certain alignment methods (e.g., DeepSeek RLHF) are able to maintain RJ.

## Related Work & Insights
- **vs "None of the above" Studies**: Prior work added a "none of the above" option; this paper is more radical, providing absolute zero exit options.
- **vs Alignment Safety Studies**: Most studies focus on rejecting harmful requests, whereas this work focuses on rejecting "harmless but incorrect" instructions.
- **vs Model Calibration Studies**: Calibration focuses on confidence, whereas this work focuses on the ability to identify and reject false premises.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of "Reflective Judgment" is novel, revealing the fundamental tension between alignment and reasoning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 20+ models, 3 datasets, and multiple ablations, though dataset scales are somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ Logically clear, with human experiments adding to the persuasiveness.
- Value: ⭐⭐⭐⭐⭐ Provides profound insights into LLM alignment and safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Which of These Best Describes Multiple Choice Evaluation with LLMs?](multiple_choice_eval.md)
- [\[NeurIPS 2025\] Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options](../../NeurIPS2025/llm_nlp/preference-based_reinforcement_learning_beyond_pairwise_comparisons_benefits_of_.md)
- [\[ACL 2025\] ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)
- [\[ACL 2025\] Exploring Explanations Improves the Robustness of In-Context Learning](exploring_explanations_improves_the_robustness_of_in-context_learning.md)
- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)

</div>

<!-- RELATED:END -->
