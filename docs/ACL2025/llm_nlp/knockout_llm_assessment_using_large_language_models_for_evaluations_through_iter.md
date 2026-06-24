---
title: >-
  [Paper Note] Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons
description: >-
  [ACL 2025][LLM (Other)][LLM-as-a-Judge] Proposes Knockout Assessment, an iterative pairwise comparison LLM-as-a-Judge method based on a knockout tournament system. By allowing responses to be compared repeatedly across multiple tournament rounds to establish a global ranking perspective, it achieves an average improvement of 0.07 in Pearson correlation coefficient over individual assessment methods on science exam scoring and machine translation evaluation.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM-as-a-Judge"
  - "pairwise comparison"
  - "knockout tournament"
  - "evaluation"
  - "scoring accuracy"
date: 2026-05-08
content_hash: 32152873740a96d1
---

# Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons

**Conference**: ACL 2025  
**arXiv**: [2506.03785](https://arxiv.org/abs/2506.03785)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: LLM-as-a-Judge, pairwise comparison, knockout tournament, evaluation, scoring accuracy

## TL;DR

Proposes Knockout Assessment, an iterative pairwise comparison LLM-as-a-Judge method based on a knockout tournament system. By allowing responses to be compared repeatedly across multiple tournament rounds to establish a global ranking perspective, it achieves an average improvement of 0.07 in Pearson correlation coefficient over individual assessment methods on science exam scoring and machine translation evaluation.

## Background & Motivation

**Background**: LLM-as-a-Judge has become the dominant paradigm for automated evaluation. Common approaches include individual assessment (scoring each response independently) and pairwise assessment (comparing two responses at a time). Chatbot Arena (Zheng et al. 2023) utilizes the Elo rating system to evaluate all possible response pairs, but its computational complexity is $O(N^2)$.

**Limitations of Prior Work**: (1) Individual assessment lacks a global perspective—models evaluate each response independently without knowing the quality of other responses, leading to imprecise scoring; (2) Single-round pairwise assessment provides comparative benchmarks but still fails to establish a global understanding of all responses; (3) Packing all responses into a single prompt is infeasible due to context window constraints.

**Key Challenge**: A global ranking perspective is needed to improve scoring accuracy, but the $O(N^2)$ computational overhead cannot be afforded, nor can all responses be crammed into a single prompt.

**Goal**: To progressively establish a global ranking perspective for LLM judges within an $O(N \log N)$ computational complexity, thereby increasing scoring alignment with human experts.

**Key Insight**: Drawing from sports knockout tournament formats, responses progressively "advance" through multiple rounds of iterative pairwise comparisons, allowing strong responses to compete with other strong responses in subsequent rounds, thereby incrementally establishing a global perspective.

**Core Idea**: To perform iterative pairwise comparisons using a knockout tournament system, allowing each response to accumulate scores from multiple rounds and utilizing the average as the final score.

## Method

### Overall Architecture

The core process of Knockout Assessment consists of: (1) randomly pairing $N$ responses; (2) performing a pairwise comparison (question-level match) for each pair, where the LLM judge scores both responses simultaneously; (3) advancing the higher-scoring response to the next round; (4) repeating this until only one response remains; (5) computing the final score of each response as the average of the scores it received across all matches it participated in.

### Key Designs

1. **Question-Level Match (Pairwise Evaluation Unit)**:

    - **Function**: To have the LLM evaluate two responses simultaneously in each pairing.
    - **Mechanism**: Given a question and two responses, the LLM utilizes a pairwise ranking prompt (similar to Liusie et al. 2024) to output a score for each response. The higher-scoring response advances, and the scores are recorded in the respective lists of scores for each response.
    - **Design Motivation**: Pairwise comparisons provide direct quality contrast, making it easier to determine relative quality than individual assessments.

2. **Knockout Tournament System**:

    - **Function**: To establish a global ranking through multiple tournament rounds.
    - **Mechanism**: Starting with $N$ responses, the first round consists of $N/2$ pairs. Winners advance to the second round, yielding $N/4$ pairs, and so forth, until a final single response is left. If $N$ is odd, one response receives a bye and directly advances. The computational complexity is $O(N \log N)$ (vs. $O(N^2)$ for Chatbot Arena).
    - **Design Motivation**: Responses that advance to later rounds are compared with other strong responses to progressively refine their scores, while weak responses are eliminated early, avoiding wasted computational resources.

3. **Debiasing**:

    - **Function**: To eliminate position bias regarding response order in pairwise comparisons.
    - **Mechanism**: For each pair of responses, evaluations are conducted twice using orders A-B and B-A respectively, and the average of the two scores is taken. Although this doubles the computation, it significantly improves evaluation fairness.
    - **Design Motivation**: LLMs exhibit well-known position biases in pairwise evaluations (Resnik 2024), tending to favor the first or last response.

4. **Final Score Calculation**:

    - **Function**: To aggregate the multi-round scores of each response across the tournament.
    - **Mechanism**: The final score is the arithmetic mean of the scores a response received across all matches it participated in. Responses advancing to later rounds accumulate more scores, basing their final score on more comparative information.
    - **Design Motivation**: Cumulative multi-round scores are more stable than a single score, mitigating the noise from individual comparisons.

## Key Experimental Results

### Main Results (Pearson Correlation Coefficient vs. Human Expert Scores)

| Method | SciEx (Question-level) | SciEx (Exam-level) | WMT Dataset | Overall Average |
|------|-------------|-------------|-----------|---------|
| Individual Assessment | 0.460 | 0.545 | 0.211 | 0.405 |
| KO Assessment (no debiasing) | 0.532 | 0.611 | 0.181 | 0.441 |
| KO Assessment (debiased) | 0.550 | 0.636 | 0.205 | **0.475** |

(Data represents the average across three models: Llama 3.2-1B/3B and Llama 3.1-70B)

### Ablation Study (Pearson Comparison: First-round vs. Later-round Eliminated)

| Dataset | First-round Eliminated (1 comparison) | Later-round Eliminated (multiple comparisons) | Difference |
|--------|---------------------|--------------------|----|
| SciEx (debiased, overall) | 0.4808 | 0.5634 | +0.0826 |
| WMT (debiased, overall) | 0.2309 | 0.1490 | -0.0819 |

### By Difficulty Level (SciEx)

| Difficulty | Individual Assessment | KO Assessment (debiased) | Gain Trend |
|------|---------|---------------------|---------|
| Easy | Low | Significant Gain | Global perspective compensates for insufficient model knowledge |
| Medium | Moderate | Gain | Stable Gain |
| Hard | High | Further Gain | Stable Gain |

### Key Findings
- Knockout outperforms individual assessment across all datasets and models (averaging +0.07 Pearson improvement after debiasing).
- The gain is more prominent on complex tasks (SciEx Science Exam), whereas performance degradation is observed for the 70B model on simple tasks (WMT translation).
- Multi-round comparisons are indeed more effective than single-round comparisons: on SciEx, responses eliminated in later rounds achieve a Pearson score 0.08 higher than those eliminated in the first round.
- On WMT, the Pearson correlation is lower for later-round responses, indicating that iterative comparisons might introduce noise to simple tasks.
- Debiasing consistently improves results and is recommended.
- The largest improvement is observed on easy questions (Easy)—the global perspective helps the model compensate for deficiencies in its own domain knowledge.

## Highlights & Insights
- Simple and elegant methodology—integrating sports tournament intuition into LLM evaluation, resulting in low implementation overhead, high effectiveness, and ease of deployment. It requires no training and no special model architecture.
- Key findings have practical guiding significance: Knockout performs well on complex evaluation tasks but may underperform compared to individual evaluation on simple tasks. Practitioners should select strategies based on task complexity.

## Limitations & Future Work
- Performance degradation occurred on the 70B model on WMT machine translation, indicating that the method is not universally applicable to simple tasks.
- Only Llama series models (1B/3B/70B) were tested, omitting closed-source models such as GPT-4 or Claude.
- Random pairing in the tournament may introduce variance—different initial pairings could lead to different final rankings.
- It has not been directly compared with more efficient sorting methods (such as the Heapsort method by Qin et al. 2024).

## Related Work & Insights
- **vs Zheng et al. 2023 (Chatbot Arena)**: Also based on pairwise comparisons, but the Elo rating system requires $O(N^2)$ comparisons; Knockout only requires $O(N \log N)$.
- **vs Qin et al. 2024**: Proposes sorting methods based on Heapsort and Bubble Sort ($O(N \log N)$ and $O(N)$), but has not been compared on the same datasets.
- **vs Liusie et al. 2024**: Proposes pairwise ranking prompts and debiasing methods; Knockout adds the iterative tournament mechanism on top of this.

## Rating
- Novelty: ⭐⭐⭐⭐ The tournament idea is intuitive but not deeply profound; in essence, it adds multi-round iteration to pairwise comparisons.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets + three models, but lacks comparison with more baselines and validation on larger-scale closed-source models.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear, concise, and well-described methodology.
- Value: ⭐⭐⭐⭐ Provides a simple and effective strategy for improving LLM evaluation, though its applicability scope is limited (primarily effective for complex tasks).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Theory of Mind in Large Language Models: Assessment and Enhancement](theory_of_mind_llm.md)
- [\[NeurIPS 2025\] Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options](../../NeurIPS2025/llm_nlp/preference-based_reinforcement_learning_beyond_pairwise_comparisons_benefits_of_.md)
- [\[ACL 2025\] Clue Guided Re-Assessment to Improve Reasoning in Large Language Models](clue_guided_re-assessment_to_improve_reasoning_in_large_language_models.md)
- [\[ACL 2025\] LLM as a Broken Telephone: Iterative Generation Distorts Information](llm_broken_telephone.md)
- [\[ACL 2025\] GAMEBoT: Transparent Assessment of LLM Reasoning in Games](gamebot_transparent_assessment_of_llm_reasoning_in_games.md)

</div>

<!-- RELATED:END -->
