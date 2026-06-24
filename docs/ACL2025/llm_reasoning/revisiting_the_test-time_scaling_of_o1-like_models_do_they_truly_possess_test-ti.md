---
title: >-
  [Paper Note] Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?
description: >-
  [ACL2025][Reasoning][test-time scaling] This paper systematically reveals that o1-like models, such as QwQ, DeepSeek-R1, and LIMO, do not possess true sequential scaling capabilities at test time—longer Chain-of-Thought (CoT) sequences do not yield higher accuracy, primarily due to insufficient self-revision capabilities. Based on this finding, the authors propose an alternative parallel scaling method called Shortest Majority Vote, which significantly outperforms traditional…
tags:
  - "ACL2025"
  - "Reasoning"
  - "test-time scaling"
  - "o1-like models"
  - "self-revision"
  - "majority vote"
  - "chain-of-thought"
date: 2026-05-08
content_hash: 550ed627733246a5
---

# Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?

**Conference**: ACL2025  
**arXiv**: [2502.12215](https://arxiv.org/abs/2502.12215)  
**Code**: [GitHub](https://github.com/ZhiYuanZeng/test-time-scaling-eval)  
**Area**: LLM Reasoning  
**Keywords**: test-time scaling, o1-like models, self-revision, majority vote, chain-of-thought

## TL;DR

This paper systematically reveals that o1-like models, such as QwQ, DeepSeek-R1, and LIMO, do not possess true sequential scaling capabilities at test time—longer Chain-of-Thought (CoT) sequences do not yield higher accuracy, primarily due to insufficient self-revision capabilities. Based on this finding, the authors propose an alternative parallel scaling method called Shortest Majority Vote, which significantly outperforms traditional majority voting.

## Background & Motivation

The OpenAI o1 series pioneered the "test-time scaling" paradigm, which continuously improves reasoning capabilities by allocating more computational resources during the inference phase. The success of o1 has spurred open-source replicas such as QwQ, DeepSeek-R1, and LIMO. While these models can generate exceptionally long CoTs, a key question remains unverified: **do these models truly possess test-time scaling capabilities?** That is, does a longer reasoning chain inevitably lead to better performance?

The authors observe a counterintuitive phenomenon: for the same problem, the average length of correct solutions is actually **shorter than** that of incorrect solutions. This motivates them to delve deeply into the test-time scaling mechanisms of o1-like models, distinguishing the actual efficacy of sequential scaling (prolonging the CoT) from that of parallel scaling (sampling multiple times and selecting the best).

## Method

### 1. Failure Analysis of Sequential Scaling

**Experimental Design**: For each question, five solutions are sampled and sorted by length into five groups. The average length and accuracy of each group are computed.

**Core Findings**:
- The average length of the longest solutions is approximately twice that of the shortest solutions, yet accuracy displays no significant improvement, and even exhibits inverse scaling on challenging benchmarks such as AIME and Omni-MATH.
- For the same problem, the average length of correct solutions is consistently shorter than that of incorrect solutions, with weaker models (e.g., QwQ, R1-Distill-1.5B) exhibiting an even wider gap.

### 2. Root Cause of Failure: Insufficient Self-Revision

**Discrepancy Analysis Between Long and Short CoTs**: Longer CoTs tend to contain more self-revision markers (such as "Wait" and "Alternatively"), with solution length showing a strong linear correlation with the frequency of "Wait".

**Active Self-Revision Triggering Experiment**:
- The "final answer" portion at the end of a solution is removed, and the model is prompted with "Wait" or "Alternatively" to continue reasoning for an additional 40 steps.
- The accuracy of QwQ and R1-Distill-1.5B declines continuously as the number of revision steps increases.
- R1-Distill-32B/14B and LIMO show slight initial improvements, followed by oscillations without further progress.

**Detailed Analysis of Revision Behaviors**:
- The successful-revision rate (correcting an incorrect answer to a correct one) is extremely low, consistently remaining below 10%.
- For QwQ and R1-Distill-1.5B, the failed-revision rate (changing a correct answer to an incorrect one) is even higher than the successful-revision rate.
- When the initial answer is wrong, R1-Distill-32B/14B retains the incorrect answer in over 70% of the cases.

### 3. Parallel Scaling vs. Sequential Scaling

Comparing the two strategies under the same token budget:
- **Coverage (pass@k)**: The coverage of sampling 10 solutions in parallel is significantly higher than that of sequentially revising for 40 steps.
- **Accuracy**: The scalability of Majority Voting (parallel) outperforms that of sequential revision.
- Sequential scaling also incurs higher computational cost due to the attention computed over longer context lengths.

### 4. Shortest Majority Vote

Based on the insight that "correct solutions are shorter," an improved parallel scaling algorithm is proposed:

For the $i$-th answer class, let $c_i$ represent the number of solutions in this class, and $l_i$ denote their average length. The score is computed as:

$$s_i = \frac{c_i}{\log(l_i)}$$

The category with the highest score is selected as the final answer.

**Design Motivation**: Correct answers are more likely to reside in answer classes that boast both higher counts and shorter lengths. When there are only two candidate solutions (rendering traditional majority voting ineffective), the solution length can serve as an informative auxiliary signal.

## Key Experimental Results

### Table 1: Ratio of Retaining the Original Incorrect Answer During Revision

| Model | Ratio of Retaining Original Incorrect Answer |
|------|---------------------|
| R1-Distill-32b | 72% |
| R1-Distill-14b | 70% |
| R1-Distill-1.5b | 58% |
| QwQ | 32% |
| LIMO | 54% |

During self-revision, the models remain "stubborn" in most cases, refusing to alter their output even when the answer is incorrect, which severely undermines the efficacy of sequential scaling.

### Table 2: Shortest Majority Vote vs. Majority Vote (16 solutions)

| Model | AIME MV | AIME Shortest MV | GPQA MV | GPQA Shortest MV |
|------|---------|-------------------|---------|-------------------|
| R1-Distill-32b | 72.88 | **73.77** | 63.33 | **63.53** |
| R1-Distill-14b | 71.77 | 71.55 | 56.16 | 56.46 |
| R1-Distill-1.5b | 40.00 | **42.22** | 29.59 | **30.20** |
| QwQ | 51.33 | 50.88 | 62.25 | 62.25 |
| LIMO | 68.88 | **70.00** | 55.58 | 55.89 |

Shortest Majority Vote outperforms traditional Majority Vote across most models and datasets, yielding particularly substantial gains on AIME.

## Highlights & Insights

- Systematically refutes the intuitive assumption that "longer CoTs equal better reasoning" through large-scale empirical experiments.
- Pinpoints the root cause of sequential scaling failure to insufficient self-revision capabilities (with a successful-revision rate <10% and a high probability of preserving incorrect answers).
- The Shortest Majority Vote method is simple and elegant: it leverages solution length solely as an auxiliary signal without requiring the training of additional reward models.
- Evaluation spans the entire QwQ and DeepSeek-R1-Distill series, as well as LIMO, across 4 benchmarks (MATH-500, AIME, Omni-MATH, GPQA), demonstrating the robustness of the findings.

## Limitations & Future Work

- Full evaluation of the 671B DeepSeek-R1 model was constrained by computational costs and conducted only in a subset of experiments.
- The experiments are based on static model checkpoints, without considering the dynamic evolution of test-time scaling behaviors during reinforcement learning training.
- Shortest Majority Vote may yield limited benefits for models possessing strong sequential scaling abilities (where a "Longest Majority Vote" might conversely be needed).
- Analysis of the root cause behind insufficient self-revision remains at a phenomenological level, lacking a deeper exploration of the impact of model architectures or training methodologies.

## Related Work & Insights

- **Parallel Scaling Methods**: Best-of-N Search (Cobbe et al., 2021), Majority Voting (Wang et al., 2023), Tree Search (Beam Search, MCTS).
- **Sequential Scaling / Self-Revision**: Self-Refine (Madaan et al., 2023), CRITIC (Gou et al., 2024); debates surrounding the efficacy of self-revision (Huang et al., 2024 vs. Kumar et al., 2024).
- **Analysis on o1-like Models**: Wang et al. (2025) rationalize similar phenomena from an "underthinking" perspective; Chen et al. (2024) and Arora & Zanette (2025) demonstrate that shortening CoTs does not degrade performance.
- **Differences**: This work provides the first systematic comparison between the sequential and parallel scaling capacities of o1-like models, and proposes an improved voting method based on length signals.

## Rating

- Novelty: ⭐⭐⭐⭐ — Empirically calls into question the mainstream assumptions of test-time scaling, delivering valuable insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Demonstrates comprehensive coverage across multiple models and benchmarks, with step-by-step rigorous analysis.
- Writing Quality: ⭐⭐⭐⭐ — Transparent structure, rich with illustrations and tables, featuring logical argumentation.
- Value: ⭐⭐⭐⭐ — Holds significant reference value for understanding and enhancing reasoning-scaling strategies in o1-like models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](mclm_multilingual_test_time_scaling.md)
- [\[ACL 2025\] Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory](rethinking_the_role_of_prompting_strategies_in_llm_test-time_scaling_a_perspecti.md)
- [\[NeurIPS 2025\] LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](../../NeurIPS2025/llm_reasoning/limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)
- [\[ACL 2025\] Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering](test_time_scaling_selective_qa.md)
- [\[ICCV 2025\] Video-T1: Test-Time Scaling for Video Generation](../../ICCV2025/llm_reasoning/video-t1_test-time_scaling_for_video_generation.md)

</div>

<!-- RELATED:END -->
