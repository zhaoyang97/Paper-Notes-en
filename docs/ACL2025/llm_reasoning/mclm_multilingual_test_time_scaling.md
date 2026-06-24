---
title: >-
  [Paper Note] Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning
description: >-
  [ACL 2025][Reasoning][Test-Time Scaling] This work proposes MCLM (a competition-level mathematical benchmark in 55 languages) and reveals that while three test-time scaling methods (ORM/PRM/Budget Forcing) yield significant improvements in English (e.g., +20 points on AIME), they yield an average gain of only 1.94 points in other languages, demonstrating a severe bottleneck in the multilingual generalization capability of test-time scaling.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Test-Time Scaling"
  - "Multilingual Reasoning"
  - "Mathematical Reasoning"
  - "Budget Forcing"
  - "Reward Model"
date: 2026-05-08
content_hash: dd58db035d8eecec
---

# Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning

**Conference**: ACL 2025  
**arXiv**: [2502.17407](https://arxiv.org/abs/2502.17407)  
**Code**: [https://github.com/gauss5930/MCLM](https://github.com/gauss5930/MCLM)  
**Area**: LLM Reasoning  
**Keywords**: Test-Time Scaling, Multilingual Reasoning, Mathematical Reasoning, Budget Forcing, Reward Model

## TL;DR
This work proposes MCLM (a competition-level mathematical benchmark in 55 languages) and reveals that while three test-time scaling methods (ORM/PRM/Budget Forcing) yield significant improvements in English (e.g., +20 points on AIME), they yield an average gain of only 1.94 points in other languages, demonstrating a severe bottleneck in the multilingual generalization capability of test-time scaling.

## Background & Motivation
**Background**: Compute scaling during the pre-training phase has been shown to naturally bring multilingual capabilities—as long as the model is large enough, the "curse of multilinguality" tends to disappear. Test-time scaling (such as Self-Consistency, PRM, and Budget Forcing/R1-style thinking) has recently emerged as a key direction for boosting reasoning performance.

**Limitations of Prior Work**: Existing multilingual mathematical benchmarks (e.g., MGSM) have saturated (with multiple models reaching 87–89%), failing to effectively evaluate frontier models. Meanwhile, the multilingual generalizability of test-time scaling has remained largely under-researched.

**Key Challenge**: While pre-training scaling naturally brings multilingual capacity, does test-time scaling possess the same linguistic generalization properties? Intuitively, longer reasoning chains might amplify error propagation, making models more sensitive to linguistic variations.

**Goal**: (1) To construct a highly challenging multilingual mathematical reasoning benchmark; (2) To systematically evaluate the multilingual generalizability of three test-time scaling methods.

**Key Insight**: Concurrently and horizontally compare the performance of ORM, PRM, and Budget Forcing across 55 languages under controlled equivalent inference FLOPs.

**Core Idea**: While test-time scaling is highly effective for English mathematical reasoning, these gains can hardly transfer to other languages—multilingual generalization must be resolved during pre-training or fine-tuning, rather than the inference stage.

## Method

### Overall Architecture
This work constructs the MCLM benchmark, compares three test-time scaling methods, and trains the MR1-1.5B model:
- Input: Competition-level mathematical problems in 55 languages
- Evaluation: Accuracy and cross-lingual consistency (Fleiss' kappa)
- Method Comparison: Compare three reasoning scaling strategies under equivalent inference FLOPs

### Key Designs

1. **MCLM Benchmark (Competition-level Math in 55 Languages)**:

    - **Function**: Contains 4 subsets—MT-MATH100 (100 problems from MATH-500 translated into 55 languages), MT-AIME2024 (30 AIME problems translated into 55 languages), M-IMO (manually translated IMO problems in 38 languages), and M-MO (original math Olympiad problems from various countries/regions in 11 languages).
    - **Mechanism**: A hybrid of machine translation and manual translation, covering difficulty levels from intermediate to extremely hard. Numerical answers are evaluated using a rule-based verifier, while complex answers are evaluated using LLM-as-a-Judge.
    - **Design Motivation**: Since MGSM is saturated, a more difficult benchmark is required; moreover, relying solely on machine translation might introduce translation artifacts, hence the inclusion of manually translated IMO/MO data.

2. **Unified Inference FLOPs Comparison**:

    - **Function**: Unifies the inference cost of the three methods into FLOPs for a fair comparison.
    - **Mechanism**: Generator cost $\approx 2N_G D$ (where $N_G$ is the parameter count, and $D$ is the number of generated tokens), validator cost $\approx 4N_V$ (accounting for doubled inference overhead). An ORM with $k=2$ corresponds to a PRM with $(S=3, c=3)$ and BF with 2048 tokens.
    - **Design Motivation**: Computation costs vary drastically across different methods, making fair comparison impossible without unification.

3. **MR1-1.5B: Multilingual Thinking Model**:

    - **Function**: Performs SFT on Deepseek-R1-1.5B using translated thinking trajectories.
    - **Mechanism**: Translates 100K thinking trajectories of R1 into 14 languages, keeping the reasoning process in English (as the pivot language) while only translating the question and answer parts. It is trained for only 0.5 epochs to prevent overfitting.
    - **Design Motivation**: Leverage the existing extended reasoning capabilities of R1 to improve cross-lingual generalization through multilingual fine-tuning.

4. **Cross-Lingual Consistency Metric (Fleiss' kappa)**:

    - **Function**: Treats each language as an "annotator" and uses Fleiss' kappa to measure whether the model consistently solves or fails the same problem across different languages.
    - **Design Motivation**: Relying solely on average accuracy is insufficient—the model might solve different problems correctly in different languages, rather than consistently solving the same set of problems.

### Loss & Training
- Base models: Qwen2.5-Math-1.5B/7B-Instruct and Deepseek-R1-1.5B
- External validators: Qwen2.5-Math-72B-RM (ORM/PRM)
- MR1 Training: 0.5 epoch SFT on translated R1 thinking trajectories

## Key Experimental Results

### Main Results: Performance on MCLM Benchmark

| Model | MT-MATH100 | MT-AIME2024 | M-IMO | M-MO | Average |
|------|-----------|-------------|-------|------|------|
| Qwen2.5-Math-1.5B | 42.32±8.61 | 16.36±6.89 | 12.23±6.02 | 25.00±19.10 | 23.98 |
| Deepseek-R1-1.5B | 49.40±8.84 | 17.21±6.69 | 21.94±6.75 | 26.77±19.83 | 28.83 |
| GPT-4o-Mini | 70.30±3.68 | 20.18±6.83 | 13.33±5.36 | 30.81±15.80 | 33.66 |
| **MR1-1.5B** | **55.61±10.93** | **19.94±8.10** | **19.20±6.24** | **28.97±16.64** | **30.93** |
| o3-Mini | 84.89±2.80 | 45.33±5.35 | 29.75±6.86 | 51.42±16.94 | 52.85 |

### Multilingual Generalization of Test-Time Scaling

| Method | English AIME Gain | Average Gain in Other Languages | Note |
|------|---------------|----------------|------|
| ORM (k=8) | +20 pts (1.5B) | Limited / Unstable | No obvious improvement for non-English on AIME |
| PRM (72B RM) | Accuracy improves with FLOPs | No improvement in consistency | No monotonic trend in Fleiss' kappa |
| Budget Forcing | +20 pts (English AIME) | **+1.94 pts Avg.** | Near-linear improvement in English, almost ineffective in other languages |

### Key Findings
- **Severe English bias in test-time scaling**: Budget Forcing improves the English score by 20 points on AIME, but improves it by only 1.94 points on average across 54 other languages.
- **Equivalent performance across three methods under the same FLOPs**: After controlling for inference computational cost, there is no essential difference between ORM, PRM, and BF, showing that "thinking LLMs" possess no obvious advantage.
- **ORM outperforms PRM**: Under equivalent FLOP budgets, ORM generally outperforms PRM, and PRM requires repeated queries to the validator, leading to higher latency.
- **Increasing inference budgets may reduce cross-lingual consistency**: Fleiss' kappa and standard deviation do not improve, and even deteriorate, when reasoning is scaled.
- **Multilingual translation SFT is effective but limited**: MR1-1.5B yields an average gain of 2.1% via translation SFT, which is far from sufficient to resolve the multilingual gap.

## Highlights & Insights
- **Systematically refutes a common assumption**: Many believe that the gains from test-time scaling will naturally generalize to multiple languages, much like pre-training scaling. This work disproves this through experiments across 55 languages.
- **FLOPs normalization methodology**: The method of unifying inference costs across ORM/PRM/BF is highly valuable, providing a standard framework for the fair comparison of test-time scaling.
- **Using Fleiss' kappa for evaluating cross-lingual consistency** is an ingenious adaptation—originally meant for inter-annotator agreement, it is used here by treating languages as "annotators".

## Limitations & Future Work
- Experiments are mainly conducted on 1.5B and 7B models. Larger models (70B+) might exhibit different behaviors—as the multilingual curse of pre-training scaling disappears in larger models, the same might hold true for test-time scaling.
- Only mathematical tasks are evaluated. The gap might be even wider for other tasks requiring cultural or domain-specific knowledge.
- Translation data only covers 14 languages, and the reasoning process remains in English without exploring reasoning in target languages.

## Related Work & Insights
- **vs MGSM (Shi et al. 2022)**: MGSM consists of simple math and is already saturated; MCLM is competition-level and can effectively differentiate frontier models.
- **vs s1 (Muennighoff et al. 2025)**: s1 proves that Budget Forcing is effective in English, whereas this work demonstrates its lack of multilingual generalization.
- **vs DeepSeek-R1**: The "aha moment" type of self-correction in R1 is highly effective in English, but this work suggests that such capability is difficult to transfer across languages.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically investigate the multilingual generalization of test-time scaling, delivering important and counter-intuitive findings.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely rigorous experimental design featuring 55 languages, 4 subsets, 3 methods, and unified FLOPs comparisons.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with rich illustrations and clear conclusions.
- Value: ⭐⭐⭐⭐ Provides important takeaways for both multilingual reasoning and test-time scaling research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?](revisiting_the_test-time_scaling_of_o1-like_models_do_they_truly_possess_test-ti.md)
- [\[ACL 2025\] Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering](test_time_scaling_selective_qa.md)
- [\[ACL 2025\] Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory](rethinking_the_role_of_prompting_strategies_in_llm_test-time_scaling_a_perspecti.md)
- [\[ACL 2025\] ProcessBench: Identifying Process Errors in Mathematical Reasoning](processbench_identifying_process_errors_in_mathematical_reasoning.md)
- [\[ICCV 2025\] Video-T1: Test-Time Scaling for Video Generation](../../ICCV2025/llm_reasoning/video-t1_test-time_scaling_for_video_generation.md)

</div>

<!-- RELATED:END -->
