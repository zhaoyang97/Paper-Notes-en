---
title: >-
  [Paper Note] Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning
description: >-
  [ACL 2025][LLM (Other)][Veracity Bias] This paper reveals that LLMs exhibit a "Veracity Bias" in reasoning tasks. Despite explicit alignment against stereotypes, LLMs systematically attribute correct answers to specific ethnic groups (attribution bias) and evaluate the same solution differently depending on the "author's" race (evaluation bias). This bias is prevalent across mathematics, coding, commonsense reasoning, and writing tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Veracity Bias"
  - "Demographic Bias"
  - "Fairness in Reasoning"
  - "LLM Evaluation Bias"
  - "Attribution Bias"
date: 2026-05-08
content_hash: d6d56ab7eb78b2e5
---

# Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning

**Conference**: ACL 2025  
**arXiv**: [2505.16128](https://arxiv.org/abs/2505.16128)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Veracity Bias, Demographic Bias, Fairness in Reasoning, LLM Evaluation Bias, Attribution Bias  

## TL;DR
This paper reveals that LLMs exhibit a "Veracity Bias" in reasoning tasks. Despite explicit alignment against stereotypes, LLMs systematically attribute correct answers to specific ethnic groups (attribution bias) and evaluate the same solution differently depending on the "author's" race (evaluation bias). This bias is prevalent across mathematics, coding, commonsense reasoning, and writing tasks.

## Background & Motivation
**Background**: LLMs have undergone post-training alignment to prevent demographic biases and stereotypes. When directly asked about intelligence differences among different races, models refuse to answer or explicitly state that such stereotypes are inappropriate.

**Limitations of Prior Work**: Prior research has found that this alignment is superficial. Hidden biases can be exposed through role-playing (persona setting) or social context provocation. However, these methods require specific prompt designs to "induce" bias.

**Key Challenge**: LLMs are inevitably exposed to societal biases in their pre-training data. As they develop increasingly stronger reasoning capabilities, have they internally associated "answer correctness" with "demographic characteristics"?

**Goal**: Instead of relying on social context provocation, this study directly tests whether LLMs possess implicit biases that associate solution correctness with race or gender in problem-solving (reasoning tasks).

**Key Insight**: Designing two types of experiments: attribution (given correct/incorrect solutions, asking the model which group authored them) and evaluation (labeling the same solution with different groups and observing variations in scoring).

**Core Idea**: There is a systematic association between veracity judgments and demographic characteristics in LLM reasoning tasks. This "Veracity Bias" transcends superficial alignment and is deeply embedded in the models' reasoning processes.

## Method

### Overall Architecture
Two types of bias detection experiments are designed:
- **Attribution Bias**: Given a pair of solutions (one correct, one incorrect), the LLM is required to attribute them to different demographic groups.
- **Evaluation Bias**: Given the same solution but labeled as being authored by different groups, the LLM's evaluations are observed for variations.

### Key Designs

1. **Attribution Experiment**:

    - **Function**: Presents [Problem][Correct Solution][Incorrect Solution] and requires the LLM to select the "author's" demographic group for each solution from the options.
    - Two modes: (a) Direct labels—"an Asian student, a Black student, a White student"; (b) Name proxies—using the most common names from each group.
    - Checking items: Selecting only "solvable" problems where the LLMs can reliably judge correctness under different temperatures, eliminating interference from model uncertainty.
    - **Design Motivation**: If the model is unbiased, the attribution of correct/incorrect solutions should be uniformly distributed across groups.

2. **Evaluation Experiment**:

    - **Function**: Presents [Problem][Solution] and labels the "author's" identity (e.g., "Camila, Ethnicity: Hispanic"), requiring the LLM to evaluate the solution's correctness.
    - Control group: Randomized neutral placeholders (e.g., XXXXX, [NAME]) used to distinguish demographic-driven bias from the model's inherent randomness.
    - Covered areas: Binary grading is used for math and coding; a 1-6 scale is used for writing.

3. **Metrics**:

    - $AB_{cor}$: Correct attribution bias—the probability of which group is most likely to be assigned the correct answer.
    - $AB_{inc}$: Incorrect attribution bias—the probability of which group is most likely to be assigned the incorrect answer.
    - $EI$: Evaluation inconsistency—the proportion of times the same solution is evaluated differently across different groups.
    - $EP$: Evaluation preference—the maximum pairwise evaluation bias probability.

4. **Impact of Reasoning on Bias**:

    - Three output formats are tested: No Reasoning (NR), Short Reasoning (SR, <100 words), and Long Reasoning (LR, >200 words).
    - Finding: Reasoning can reduce attribution bias but does not reduce evaluation bias, and the reasoning process may be inconsistent with the final attribution decision.

### Experimental Setup
- 5 models: GPT-3.5-turbo, GPT-4o, Claude-3 Sonnet, Gemini-1.5-Pro, LLaMA-3-8B
- 6 datasets: GSM8K, MATH, HumanEval, CommonsenseQA, ARC-Easy, ASAP-AES
- Temperature = 0 to ensure approximately deterministic output.
- 100 solvable problems per benchmark.

## Key Experimental Results

### Attribution Bias Main Experiment (Race Dimension)

| Model | Math - Correct Bias | Math - Incorrect Bias | Coding - Correct Bias | Coding - Incorrect Bias |
|------|-------------|-------------|-------------|-------------|
| GPT-4o | 14% (Asian) | 21% (Black) | 57% (Asian) | 46% (White) |
| GPT-3.5 | 60% (White) | 36% (Black) | 10% (Asian) | 10% (Black) |
| Claude-3 | 14% (White) | 20% (Black) | 18% (Asian) | 28% (Black) |
| Gemini | 22% (White) | 28% (Black) | 28% (Asian) | 42% (Black) |
| LLaMA-3 | 36% (White) | 38% (Black) | 22% (Asian) | 16% (White) |

### Key Statistics

| Finding | Data |
|------|------|
| GPT-3.5 in GSM8K | 82% of correct answers attributed to the White group |
| Black group | Lowest correct answer attribution rate across all models in math and coding |
| In writing evaluation | Essays by Hispanic authors scored higher than those by Asian authors |
| Gender bias | Generally weaker than racial bias; Claude has almost zero gender bias |
| GPT-4o refusal rate | 14% in math, 28% in coding refused to answer (the only model exhibiting refusal behavior) |

### Key Findings
- **Black Group Worst Consistency**: In math and coding, all models tend to attribute fewer correct answers and more incorrect answers to the Black group.
- **Domain Specificity**: Whites are preferred in math, and Asians are preferred in coding, reflecting societal stereotypes.
- **Name Proxies Equally Effective**: Using race-associated names instead of direct labels yields similar bias patterns, with no model refusing to answer.
- **Inconsistent Reasoning**: Models may explicitly state in their reasoning processes that "one should not be biased," yet their final attributions remain biased.
- **Color Stereotypes**: When asked to write visualization code, LLMs automatically assign stereotypical colors to different races (e.g., Black $\rightarrow$ black tones).

## Highlights & Insights
- **A New Dimension of Bias Detection**: Detecting bias directly from the perspective of problem-solving/reasoning capabilities without relying on social context provocation is an entirely new perspective. Biases may be deeper than expected—embedded in the models' judgment of "what is correct."
- **Findings on Reasoning Exacerbating Inconsistency**: Models demonstrate that they "know they shouldn't be biased" during their reasoning process but still behave biasedly. This suggests that alignment only affects surface-level outputs rather than internal beliefs.
- **Warning for Educational Applications**: If LLMs are used for grading or feedback, this bias could systematically disadvantage specific demographic groups.

## Limitations & Future Work
- It only covers race (three groups) and gender (binary), without exploring more fine-grained demographic dimensions.
- Experiments are conducted solely in an English context.
- It primarily uses model versions from early 2024, and the effectiveness of the latest alignment techniques remains unknown.
- It lacks in-depth analysis of the source of bias at the attention or representation levels.

## Related Work & Insights
- **vs BBQ (Parrish et al. 2022)**: BBQ tests bias through ambiguous scenarios, whereas this paper tests it through veracity judgments, which are closer to reasoning scenarios.
- **vs Gupta et al. 2024**: They expose bias through persona settings, whereas this paper demonstrates that reasoning bias can be observed without personas.
- **Relationship with RLHF Alignment**: This indicates that current alignment techniques mainly eliminate surface-level output bias but fail to eliminate implicit associations during the reasoning process.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces the concept of "Veracity Bias," detecting bias from the perspective of reasoning correctness—a completely fresh angle.
- Experimental Thoroughness: ⭐⭐⭐⭐ Spans 5 models, 6 datasets, and multiple settings, but lacks comparison with the most recent models.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, with systematic presentation of findings.
- Value: ⭐⭐⭐⭐⭐ Provides crucial warnings for deploying LLMs in educational/evaluation scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [\[ACL 2025\] Mitigate Position Bias in LLMs via Scaling a Single Hidden States Channel](mitigate_position_bias_in_large_language_models_via_scaling_a_single_dimension.md)
- [\[ACL 2025\] ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion](mathfusion_instruction_fusion.md)

</div>

<!-- RELATED:END -->
