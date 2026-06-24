---
title: >-
  [Paper Note] Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering
description: >-
  [ACL 2025][Reasoning][Test-time scaling] This paper presents the first evaluation of test-time scaling models (DeepSeek R1, S1) in selective question answering scenarios (where abstaining from answering is allowed). It finds that increasing test-time compute not only improves accuracy but also enhances the model's confidence in correct answers. The authors propose a selection function based on a confidence threshold and a "Jeopardy Odds" utility function to evaluate test-time…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Test-time scaling"
  - "selective question answering"
  - "confidence calibration"
  - "compute budget"
  - "abstention"
date: 2026-05-08
content_hash: ed9dc9517b4628dd
---

# Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering

**Conference**: ACL 2025  
**arXiv**: [2502.13962](https://arxiv.org/abs/2502.13962)  
**Code**: [https://github.com/wjurayj/final_answer](https://github.com/wjurayj/final_answer)  
**Area**: LLM Inference  
**Keywords**: Test-time scaling, selective question answering, confidence calibration, compute budget, abstention

## TL;DR
This paper presents the first evaluation of test-time scaling models (DeepSeek R1, S1) in selective question answering scenarios (where abstaining from answering is allowed). It finds that increasing test-time compute not only improves accuracy but also enhances the model's confidence in correct answers. The authors propose a selection function based on a confidence threshold and a "Jeopardy Odds" utility function to evaluate test-time scaling performance under non-zero penalties for incorrect answers.

## Background & Motivation
**Background**: Test-time scaling (e.g., DeepSeek R1, S1) has achieved breakthrough progress on mathematical reasoning benchmarks by extending the reasoning chain length. However, all existing evaluations are conducted under a "zero-risk" setting (where the model is forced to answer and incorrect answers receive no penalty).

**Limitations of Prior Work**:
   - In real-world scenarios, incorrect answers often carry measurable costs (e.g., medical diagnosis, legal consultation, autonomous driving).
   - Whether model "guessing" holds value when uncertain remains unclear; scenarios where "abstention" is preferred over "wrong answers" are completely neglected.
   - The differences in confidence calibration capabilities among different test-time scaling models have not been revealed.

**Key Challenge**: Prior test-time scaling research only reports accuracy, neglecting the model's capability of "knowing whether it is correct."

**Goal**: To evaluate test-time scaling in scenarios where abstention is allowed and to propose a standardized evaluation framework.

**Key Insight**: Utilizing the token log-probability at the end of the reasoning chain as a confidence metric, combined with utility functions under varying penalties for incorrect answers.

**Core Idea**: Increasing test-time compute not only retrieves more correct answers but also helps the model distinguish correct from incorrect answers with higher confidence, though calibration capabilities vary significantly across different models.

## Method

### Overall Architecture
The evaluation framework defines two dimensions: (1) **Compute Budget** (the number of inference tokens, ranging from 500 to 8000), strictly controlled via budget forcing; (2) **Confidence Threshold** (threshold ∈ {0.0, 0.5, 0.95}), where answers below the threshold are rejected. Evaluations are conducted across three utility settings.

### Key Designs

1. **Confidence Extraction**:

    - **Function**: Quantifying the model's certainty regarding its answers.
    - **Mechanism**: Summing the log-probabilities of the answer tokens to serve as the confidence score. All answers are formatted uniformly (e.g., 3-digit numbers) to ensure a consistent number of tokens.
    - **Design Motivation**: Simple, straightforward, and derived directly from internal model signals without requiring additional confidence estimation modules.

2. **Selection Function**:

    - **Function**: Deciding whether to output an answer based on the confidence score.
    - **Mechanism**: Accepting only answers with confidence above the threshold, otherwise abstaining. The evaluated thresholds are threshold ∈ {0.0, 0.5, 0.95}.
    - **Difference from Standard Evaluation**: A threshold of 0.0 is equivalent to standard evaluation (always answering), whereas a threshold > 0.0 allows abstention.

3. **Utility Function**:

    - **Function**: Measuring the model's value under different penalties for incorrect answers.
    - **Mechanism**: $f(\mathcal{M}, x) = \{1 \text{ if correct}, 0 \text{ if abstain}, r_t \text{ if wrong}\}$
    - **Three Settings**: Exam Odds ($r_t=0$, no penalty for incorrect answers), **Jeopardy Odds** ($r_t=-1$, equal deduction penalty for incorrect answers), and High-Stakes ($r_t=-20$, 20x deduction penalty for incorrect answers).
    - **Design Motivation**: Exam Odds is the current standard but unrealistic; Jeopardy Odds is closer to real-world scenarios where giving a wrong answer is worse than abstaining.

## Key Experimental Results

### Main Results

**AIME 2024+2025, DeepSeek R1-32B**:

| Threshold | Budget 500 Acc | Budget 4000 Acc | Budget 8000 Acc | Response Rate |
|:---------:|:-------------:|:---------------:|:---------------:|:------------:|
| 0.0 | ~30% | ~55% | ~60% | 100% |
| 0.5 | ~40% | ~65% | ~70% | ~70% |
| 0.95 | ~80% | ~85% | ~80% | ~20-40% |

At a high threshold, the model gives fewer answers but with extremely high accuracy; increasing compute mainly increases the number of answered questions (coverage).

**Jeopardy Odds Utility (R1-32B vs S1-32B)**:

| Model | Threshold=0 Budget=8K | Threshold=0.95 Budget=8K |
|------|:--------------------:|:------------------------:|
| R1-32B | ~0.20 | **~0.45** |
| S1-32B | ~0.18 | ~0.25 |

Under Jeopardy Odds, R1-32B shows significantly better confidence calibration than S1-32B—a distinction that is indiscernible under standard evaluations.

### Key Findings
- **Increasing test-time compute improves confidence, not just accuracy**: As the compute budget increases, the confidence of correct answers rises while the confidence of incorrect answers remains unchanged or even decreases (clearly visualized in Figure 3).
- **Calibration capability varies heavily across different models**: R1-32B is capable of distinguishing confidence between correct and incorrect answers, whereas S1-32B is not. While both show comparable performance under Exam Odds, R1 dramatically outperforms S1 under Jeopardy Odds.
- **Accuracy may drop at a high threshold with high budget**: Because newly found answers have a lower correctness rate, pulling down the overall accuracy of answered questions (yet the utility still rises).
- **Budget forcing (enforcing extended reasoning) can be harmful**: Abruptly truncating or strictly forcing the extension of reasoning chains may cause the model's behavior to deviate from the training distribution.

## Highlights & Insights
- **Reveals the "hidden dimension" of test-time scaling**: Current evaluations only focus on accuracy (accuracy surface). This work unveils the comprehensive 3D accuracy-coverage-confidence surface, which is crucial for understanding the true value of test-time scaling.
- **Jeopardy Odds proposed as a standard evaluation protocol**: A highly recommended metric—$r_t=-1$ serves as the most natural trade-off point between "incorrect" and "abstain", being simple and intuitively consistent.
- **Underlying differences between R1 and S1**: Two models that perform comparably on standard benchmarks show massive gaps in confidence calibration. This discovery offers significant guidance for model selection.

## Limitations & Future Work
- **Simple confidence estimation method**: Relying solely on token log-probability as confidence. Advanced methods (such as confidence based on reasoning chain consistency) might yield better performance.
- **Evaluation restricted to AIME (math competition)**: Generalizability to other tasks (e.g., coding, natural language understanding) remains unverified.
- **Side effects of budget forcing**: Enforced extension of reasoning chains might not be the optimal way to control compute allocation.
- **Compute cost itself is ignored**: Computation overhead cost is not factored into the utility function.

## Related Work & Insights
- **vs. Standard Test-Time Scaling Evaluations**: Standard evaluations rely on Exam Odds (threshold=0), whereas this work complements them with Jeopardy/High-Stakes dimensions.
- **vs. Selective Classification (Geifman & El-Yaniv)**: An LLM reasoning adaptation of the classical selective classification framework, applied to the test-time scaling paradigm for the first time.
- **vs. SQuAD 2.0 Abstention Mechanism**: While SQuAD 2.0 allows abstention without penalizing incorrect answers, this work introduces error penalties to mirror more realistic scenarios.
- Future work could utilize confidence as a signal for test-time compute allocation—terminating reasoning when confidence is high, and continuing when it is low.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces selective question answering evaluation into test-time scaling for the first time, offering a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐ Only evaluated two models on AIME, but provides deep analytical insights.
- Writing Quality: ⭐⭐⭐⭐⭐ Features beautiful visualizations (3D surface plots) and clear argumentation.
- Value: ⭐⭐⭐⭐ Contributes significantly to the evaluation methodology of test-time scaling; "Jeopardy Odds" is worth promoting widely.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](mclm_multilingual_test_time_scaling.md)
- [\[ACL 2025\] Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?](revisiting_the_test-time_scaling_of_o1-like_models_do_they_truly_possess_test-ti.md)
- [\[ACL 2025\] Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory](rethinking_the_role_of_prompting_strategies_in_llm_test-time_scaling_a_perspecti.md)
- [\[ICCV 2025\] Video-T1: Test-Time Scaling for Video Generation](../../ICCV2025/llm_reasoning/video-t1_test-time_scaling_for_video_generation.md)
- [\[NeurIPS 2025\] Atom of Thoughts for Markov LLM Test-Time Scaling](../../NeurIPS2025/llm_reasoning/atom_of_thoughts_for_markov_llm_testtime_scaling.md)

</div>

<!-- RELATED:END -->
