---
title: >-
  [Paper Note] Spotting Out-of-Character Behavior: Atomic-Level Evaluation of Persona Fidelity in Open-Ended Generation
description: >-
  [ACL 2025][persona fidelity] An atomic-level (sentence-level) evaluation framework is proposed to detect fine-grained Out-of-Character (OOC) behaviors of Large Language Models (LLMs) in open-ended generation through three metrics (ACC_atom, IC_atom, RC_atom). This addresses the issue where traditional holistic scoring approaches fail to capture subtle personality inconsistencies in long texts.
tags:
  - "ACL 2025"
  - "persona fidelity"
  - "LLM evaluation"
  - "personality"
  - "out-of-character"
  - "atomic evaluation"
date: 2026-05-08
content_hash: ac9a2b16c6862b61
---

# Spotting Out-of-Character Behavior: Atomic-Level Evaluation of Persona Fidelity in Open-Ended Generation

**Conference**: ACL 2025  
**arXiv**: [2506.19352](https://arxiv.org/abs/2506.19352)  
**Code**: None  
**Area**: Others  
**Keywords**: persona fidelity, LLM evaluation, personality, out-of-character, atomic evaluation

## TL;DR

An atomic-level (sentence-level) evaluation framework is proposed to detect fine-grained Out-of-Character (OOC) behaviors of Large Language Models (LLMs) in open-ended generation through three metrics (ACC_atom, IC_atom, RC_atom). This addresses the issue where traditional holistic scoring approaches fail to capture subtle personality inconsistencies in long texts.

## Background & Motivation

### Background

- Assigning a persona to LLMs is a core requirement for scenarios such as role-playing, social simulation, and dialogue systems.
- LLMs often exhibit **Out-of-Character (OOC)** behaviors during long-text generation: the generated content deviates from the assigned persona, leading to inconsistencies.
- For example: when a model is assigned a persona that is "neither extroverted nor introverted", it might act extroverted at times and introverted at other times within the same essay.

### Limitations of Prior Work

- Existing evaluation methods typically assign a **single response-level score**, failing to capture subtle personality deviations inside long texts.
- When the overall average alignment score is correct but the internal sentence-level behaviors fluctuate wildly, traditional methods mistakenly judge it as "well-aligned persona".
- Prior studies have mostly focused on **multiple-choice questions or closed-ended QA**, paying insufficient attention to open-ended generation scenarios.

### Motivation

- An **atomic-level** evaluation method is required to decompose text into minimal semantic units (sentences) and evaluate persona alignment sentence by sentence.
- Borrowing the concept of FActScore, the fine-grained factual verification method is transferred to persona fidelity evaluation.

## Method

### Overall Architecture

1. **Atomic Unit Segmentation**: The generated long text G is split into sentence-level atomic units using NLTK's `sent_tokenize`.
2. **Trait Scoring**: GPT-4o is employed as the grader model to assign a personality trait score in $[1, 5]$ for each atomic sentence.
3. **Invalid Sentence Filtering**: Sentences without personality indicators (e.g., purely factual statements) are filtered out.
4. **Metric Calculation**: Three complementary metrics are calculated based on the atomic-level scores.

### Key Designs

| Metric | Definition | Computation | Value Range |
|------|------|----------|----------|
| ACC_atom | Atomic Accuracy | The average rate of atomic units matching the target personality interval | $[0, 1]$ |
| IC_atom | Internal Consistency | The inverse of the standard deviation of trait scores within a single generation | $[0, 1]$ |
| RC_atom | Test-Retest Consistency | The normalized average Earth Mover's Distance of distributions across multiple generations | $[-1, 1]$ |

**ACC_atom (Atomic Accuracy)**: Partitions the scale $[1, 5]$ into three equal intervals (low/medium/high) and checks whether the score of each atomic sentence falls into the target interval, taking the average. This detects the proportion of "out-of-character" sentences.

**IC_atom (Internal Consistency)**: Measures the consistency of persona expression within a single generation. A lower standard deviation indicates higher internal consistency; higher inverse values are better. This detects scenarios where the "overall average score is correct but the internal fluctuation is wild."

**RC_atom (Test-Retest Consistency)**: Employs Earth Mover's Distance (EMD) to measure the distribution difference of scores across multiple generations. This is better at capturing distribution-level differences compared to traditional standard deviation metrics.

### Persona Setting

- Based on the **Big Five Personality Model** (OCEAN): Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism.
- Highly, moderately, and lowly levels are set for each dimension, resulting in 15 persona settings in total.
- Evaluates the model's performance in three open-ended tasks: questionnaire interviews, essay writing, and social media posting.

## Key Experimental Results

### Experimental Setup

- **Models**: 12 LLMs, including 3 base models and 9 instruction-tuned/RLHF models.
- **Tasks**: 3 types of open-ended generation tasks (questionnaire interviews, essays, social media posts).
- **Evaluation Scale**: Each model $\times$ each persona $\times$ each task is run 30 times.

### Human Validation Results

| Personality Dimension | Kendall's tau | Fleiss' kappa |
|----------|---------------|---------------|
| O (Openness) | $0.69^{***}$ | 0.90 |
| C (Conscientiousness) | $0.76^{***}$ | 0.96 |
| E (Extraversion) | $0.67^{***}$ | 0.80 |
| A (Agreeableness) | $0.72^{***}$ | 0.84 |
| N (Neuroticism) | $0.69^{***}$ | 0.74 |

The scores from GPT-4o show high agreement with human judgments (all $p < .001$), validating the reliability of automatic evaluation.

### Correlation with Traditional Metrics

| Traditional Metric | ACC_atom | RC_atom | IC_atom |
|----------|----------|---------|---------|
| ACC | 0.91 | 0.51 | 0.40 |
| RC | 0.48 | 0.98 | 0.37 |

ACC is highly correlated with ACC_atom, yet critical differences remain: some lower-tier persona models have high ACC but low ACC_atom, indicating that the overall score masks internal inconsistency. IC_atom correlates poorly with traditional metrics ($r = 0.37 - 0.40$), proving that it captures a completely new dimension.

### Overall Model Performance Comparison

| Model | Instruction-tuned | RLHF | ACC_atom | IC_atom | RC_atom |
|------|----------|------|----------|---------|---------|
| Davinci-002 | Yes | | 0.39 | 0.64 | 0.56 |
| GPT-3.5-turbo | Yes | Yes | 0.60 | 0.75 | 0.79 |
| GPT-4o | Yes | Yes | 0.61 | 0.74 | 0.78 |
| LLaMA-3-8B | | | 0.41 | 0.60 | 0.64 |
| LLaMA-3-8B-Instruct | Yes | Yes | **0.65** | 0.70 | **0.82** |
| Mistral-7B | | | 0.41 | 0.59 | 0.67 |
| Mistral-7B-Instruct | Yes | | 0.58 | 0.69 | 0.80 |
| Claude-3-haiku | Yes | Yes | 0.59 | 0.71 | 0.69 |

### Key Findings

- **Structured tasks perform better**: ACC_atom for questionnaire tasks (0.73) > essays (0.58) > social posts (0.52).
- **High-level personas are easiest to remain loyal to**: High-level personas score an average ACC_atom of 0.95, mid-level scores only 0.27, and low-level scores 0.62.
- **Social Desirability Bias**: Models show the lowest fidelity towards socially undesirable personalities (e.g., "closed-mindedness" or "carelessness"), presumably due to RLHF training alignment preferences.
- **Tuned models systematically outperform base models**: Instruction-tuned + RLHF models obtain significant improvements across all three metrics.

## Highlights & Insights

1. **Innovative Atomic-Level Evaluation Paradigm**: The FActScore-style fine-grained verification is transferred to persona fidelity evaluation for the first time, characterizing alignment accuracy, internal consistency, and cross-generation consistency through three complementary metrics.
2. **Discovery of Social Desirability Bias**: Reveals that RLHF training implicitly biases models towards socially favorable personas, resulting in systematic fidelity degradation for less favorable personas.
3. **Clever Application of Earth Mover's Distance**: Uses distribution distance instead of traditional standard deviation to measure test-retest consistency, capturing distributional shifts missed by traditional metrics.
4. **Rigorous Human Validation**: Human evaluation on 250 sentence pairs confirms the reliability of the automatic scoring (Kendall's tau between 0.67 and 0.76).

## Limitations & Future Work

1. **Limited to Personality Domain**: Only validated on the Big Five personality dimensions, leaving other dimensions like social values and political stances unexplored.
2. **Dependence on GPT-4o for Scoring**: The accuracy of the automated evaluation is inherently limited by the bias of the grading model itself.
3. **Sentence-level Granularity May Lack Precision**: Intra-sentence inconsistencies (such as self-contradictions within a single sentence) cannot be captured by the current framework.
4. **No Mitigation Method Proposed**: It only serves as an evaluation framework, leaving concrete strategies to alleviate OOC behaviors unaddressed.
5. **Limited Generation Length**: Each generation in the experiments is only 100-300 words. Whether persona drift becomes worse in longer texts remains uninvestigated.

## Related Work & Insights

- **Persona Assignment to LLMs**: Zhang et al. (2018) on conversational personas, Safdari et al. (2023) on personality measurements, and Park et al. (2023) on role-playing.
- **Persona Fidelity Evaluation**: Response-level ACC/RC by Wang et al. (2024), and prompt format sensitivity analysis by Shu et al. (2024).
- **Open-ended Generation Evaluation**: Atomic factual verification derived from FActScore (Min et al., 2023).
- **Long-text Consistency**: Research on the decay of Transformer long-text coherence by Sun et al. (2021) and Krishna et al. (2022).

## Rating

| Dimension | Rating (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | Atomic-level persona evaluation is an innovative angle, and the discovery of social desirability bias is insightful. |
| Technical Depth | 3 | The methodology is straightforward (sentence splitting + grading + statistics), but the metric design is solid. |
| Experimental Thoroughness | 5 | Scaled across 12 models $\times$ 15 personas $\times$ 3 tasks $\times$ 30 runs, complemented by rigorous human validation. |
| Writing Quality | 4 | The paper is clearly structured, with intuitive case studies and thorough quantitative analysis. |
| Value | 4 | Establishes a new standard for LLM persona fidelity evaluation, and the findings can guide future alignment studies. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions](confetti_conversational_function-calling_evaluation_through_turn-level_interacti.md)
- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)
- [\[ACL 2025\] SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection](seoe_semantic_eval.md)
- [\[CVPR 2025\] Open Set Label Shift with Test Time Out-of-Distribution Reference](../../CVPR2025/others/open_set_label_shift_with_test_time_out-of-distribution_reference.md)
- [\[ACL 2025\] A Multi-Persona Framework for Argument Quality Assessment](a_multi-persona_framework_for_argument_quality_assessment.md)

</div>

<!-- RELATED:END -->
