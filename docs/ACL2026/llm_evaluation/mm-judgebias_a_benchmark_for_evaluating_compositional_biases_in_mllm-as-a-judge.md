---
title: >-
  [Paper Note] MM-JudgeBias: A Benchmark for Evaluating Compositional Biases in MLLM-as-a-Judge
description: >-
  [ACL 2026][LLM Evaluation][MLLM-as-a-Judge] The authors formalize whether MLLM-as-a-Judge truly integrates images, queries, and responses as "Compositional Bias" and construct MM-JudgeBias—a diagnostic set containing 9 t…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "MLLM-as-a-Judge"
  - "compositional bias"
  - "modality bias"
  - "Bias-Deviation"
  - "Bias-Conformity"
date: 2026-05-08
content_hash: 60c46facec2f9bb9
---

# MM-JudgeBias: A Benchmark for Evaluating Compositional Biases in MLLM-as-a-Judge

**Conference**: ACL 2026  
**arXiv**: [2604.18164](https://arxiv.org/abs/2604.18164)  
**Code**: Project page + GitHub (as indicated in the abstract)  
**Area**: Multimodal Evaluation / MLLM-as-a-Judge / Bias Benchmark  
**Keywords**: MLLM-as-a-Judge, compositional bias, modality bias, Bias-Deviation, Bias-Conformity

## TL;DR
The authors formalize whether MLLM-as-a-Judge truly integrates images, queries, and responses as "Compositional Bias" and construct MM-JudgeBias—a diagnostic set containing 9 types of bias and 1,804 samples from 29 source benchmarks. Using two complementary metrics, Bias-Deviation (BD; score should drop after semantic destruction but does not) and Bias-Conformity (BC; score should remain stable when semantics are preserved but does not), they find that 26 SOTA MLLM judges (including Gemini-3 Pro, GPT-5.1, and Claude Opus 4.5) exhibit severe modality neglect.

## Background & Motivation

**Background**: MLLM-as-a-Judge has become the mainstream automatic evaluation paradigm for multimodal generation (captioning, VQA, visual reasoning), evolving from early direct prompting of GPT-4o to specialized fine-tuned critic models like Prometheus-Vision and LLaVA-Critic.

**Limitations of Prior Work**: While 12 types of biases in LLM-as-a-Judge have been systematically studied, research on the reliability of MLLM judges remains at shallow dimensions directly borrowed from LLMs, such as "position bias / verbosity / length." No systematic study has asked: "When a judge lacks the image, or the image and response are misaligned, or an irrelevant caption is added, will the judge still assign incorrect scores?" These are failure modes unique to multimodal judges.

**Key Challenge**: MLLM judges are frequently found to "assign the same score regardless of whether they see the image." This is not merely a lack of capability but a failure of verification integrity: the judge's primary duty is conditional verification, yet the model degrades it into unconditional prediction, awarding full marks based on the superficial fluency of the response.

**Goal**: (1) Provide a formalized framework for multimodal judge bias; (2) Construct a diagnostic set with controlled perturbations; (3) Quantify the severity of these issues across 26 MLLMs.

**Key Insight**: The behavior expected of a "reliable judge" is decomposed into three categories: **Integrality** (scores should drop if components are missing), **Congruity** (scores should drop if components contradict each other), and **Robustness** (scores should not be affected by perturbations that preserve semantics). The former two are measured by Bias-Deviation (how much the score drops when it should), while the latter is measured by Bias-Conformity (how stable the score is when it should be).

**Core Idea**: Systematize and measure "compositional bias" using 9 types of controlled perturbations across 1,804 data points and two complementary metrics.

## Method

### Overall Architecture
The construction of MM-JudgeBias consists of four stages: (a) Sampling from 29 source benchmarks (COCO, MathVista, DocVQA, ChartQAPro, etc.) across 4 task types and 12 domains; (b) Using Gemini-2.5-Pro to generate 3 queries per sample, followed by model+human dual-review to select the "best-Q" ensuring the query truly requires multimodal information; (c) Parallel construction of a text-only query set (for unnecessary-image bias); (d) Balanced response generation using multiple models (GPT-5 mini / Gemini-2.0-Flash-Lite / Qwen2.5-VL-7B) to ensure diverse score distributions; (e) Perturbing original triplets $(Q, I, R)$ according to 9 bias types to obtain $(Q', I', R')$; (f-g) Evaluating both versions using 26 MLLM judges with scores from 1-10, quantified by BD and BC. The final dataset includes 1,804 samples covering 9 biases, 4 tasks, and 12 domains.

### Key Designs

1.  **Taxonomy of 9 Bias Types Across Three Dimensions**:
    - **Function**: Decomposes "judge failure" into 9 fine-grained types that can be independently measured.
    - **Mechanism**: **Integrality** (3 types) = Text-Dominance / Image-Dominance / Response-Dominance, replacing components with null image, null query, or both; **Congruity** (2 types) = Instruction-Misalignment / Image-Misalignment, replacing components with random irrelevant queries/images; **Robustness** (4 types) = Detail-Description (appending image captions to query) / Unnecessary-Image (adding irrelevant images to text tasks) / Visual-Transformation (semantics-preserving image augmentation) / Texture-Insertion (overlaying query keywords onto the image).
    - **Design Motivation**: The first two dimensions (Integrality + Congruity) test sensitivity when scores *should* drop, while the latter (Robustness) tests stability when scores *should* stay. This "sensitivity-stability dual evaluation" prevents a single metric from misinterpreting trivial behaviors (like always scoring 0 or 10) as high performance.

2.  **Bias-Deviation (BD) and Bias-Conformity (BC) Metrics**:
    - **Function**: Quantifies expectations for when scores "should drop" vs. "should not drop."
    - **Mechanism**: BD is used for Integrality/Congruity types, defined as $\text{BD} = \mathbb{E}_{(y, \hat{y}) \sim D}[(y - \hat{y})_+ / (y - 1) | y > 1]$. It normalizes the "actual drop after perturbation" against the "maximum possible drop"; higher is better. BC is used for Robustness, defined as $\text{BC} = \mathbb{E}_{(y, \hat{y}) \sim D}[1 - |y - \hat{y}| / \max(y-1, S-y)]$; closer to 1 is better. These are used selectively based on bias type.
    - **Design Motivation**: BC alone can be cheated by a "constant score for all samples" behavior (zero discriminative power); the paper reports inter-sample variance to detect "trivial constant judges." BD excludes $y=1$ samples since they have no room to drop further, avoiding boundary effect contamination.

3.  **Human-in-the-loop High-Quality Data Synthesis**:
    - **Function**: Ensures the query truly requires joint image-text reasoning, preventing models from guessing based on one modality.
    - **Mechanism**: Gemini-2.5-Pro generates 3 candidate queries $\rightarrow$ model self-check $\rightarrow$ human selection; semantics-preserving augmentations (VisualTrans) use preset pipelines, while destructive perturbations (Texture insertion) are model-generated. Each bias type has three difficulty levels (easy / mod / hard), stratified by task and domain.
    - **Design Motivation**: If a query can be answered using only the image or text, a judge not dropping points after perturbation would be reasonable rather than biased. Human review is an essential safeguard against this trap.

### Loss & Training
MM-JudgeBias is a benchmark, not a model; no training is involved. All judge models used `max_tokens=16384` during evaluation, with reasoning effort set to "high" (for models supporting thinking modes like Gemini-2.5, o3, Claude Opus 4.5). Other hyperparameters remained at defaults. Results were averaged over three independent runs, and inter-run/inter-sample variances were reported.

## Key Experimental Results

### Main Results: BD/BC of 26 MLLMs across 9 Bias Types (Selected, Higher is Better)

| Model (Thinking) | Integrality Avg BD | Congruity Avg BD | Robustness Avg BC | Overall |
|:---|:---:|:---:|:---:|:---:|
| Gemini-3-Pro (high) ✓ | 0.726 | 0.969 | 0.926 | **0.869** |
| Claude-Opus-4.5 ✓ | 0.729 | 0.973 | 0.897 | 0.858 |
| Gemini-2.5-Pro ✓ | 0.755 | 0.952 | 0.913 | 0.869 |
| Gemini-2.5-Flash ✓ | 0.486 | 0.879 | 0.908 | 0.761 |
| o3 (high) ✓ | 0.409 | 0.661 | 0.880 | 0.675 |
| GPT-5.1 (high) ✓ | 0.201 | 0.648 | 0.912 | 0.616 |
| GPT-5 mini (high) ✓ | 0.228 | 0.588 | 0.914 | 0.613 |
| Qwen3-VL-30B-Thinking ✓ | 0.481 | 0.730 | 0.879 | 0.713 |
| Qwen2.5-VL-72B-Instruct | 0.154 | 0.598 | 0.858 | 0.566 |
| LLaVA-Critic-72B | 0.214 | 0.620 | 0.943 | 0.628 |
| Prometheus-Vision-13B | 0.288 | 0.528 | 0.785 | 0.563 |
| **All 26 Models Average** | **0.384** | **0.746** | **0.868** | **0.679** |

### Ablation Study: Prompt-Level Intervention (3 Representative Judges)

| Model | Original | +Score Guide | +Modality Constraint | +Modality Reasoning |
|:---|:---:|:---:|:---:|:---:|
| GPT-5 mini | 0.612 | 0.609 | 0.644 | 0.645 |
| Qwen3-VL-8B-Thinking | 0.660 | 0.674 | 0.694 | **0.726** |
| LLaVA-Critic-7B | 0.647 | 0.631 | 0.600 | 0.653 |

Another ablation involving N/A abstention-aware evaluation showed limited abstention rates (GPT-5 17.2%, Qwen3-VL 26.1%, LLaVA-Critic 6.5%). Overall scores remained stable after exclusion, verifying the benchmark does not inflate bias by "forcing scores on invalid samples."

### Key Findings
-   **Integrality is the Weakest Link**: The average Text-Dominance BD across all 26 models was only 0.287, and Image-Dominance was only 0.317. Even when the entire image is replaced with a black frame, judges only slightly lower the score, indicating they do not treat the image as a necessary condition for judgment.
-   **Shocking Response-Dominance**: When both query and image are nullified, leaving only the response, the average BD is still only 0.547—meaning that in almost half the cases, the judge awards high scores based solely on the fluency of the response.
-   **Reasoning Enhances Consistency, but is Not a Panacea**: Gemini 2.5 Pro and Claude Opus 4.5 with thinking enabled significantly outperformed their non-thinking counterparts. However, reasoning in o3/GPT-5 mini did not solve modality neglect. Qwen3-VL-8B even performed worse with thinking, suggesting training recipes matter more than raw reasoning capability.
-   **Critic Models are Not Necessarily More Reliable**: LLaVA-Critic-72B had a lower overall score than the 7B version (0.628 vs. 0.647), indicating that specialized critic SFT data does not resolve underlying modality integration issues.
-   **Scale $\neq$ Reliability**: Increasing parameters within the same family does not consistently improve BD/BC, proving judgment reliability is an orthogonal dimension to general capability.
-   **Persistent Classical Biases**: Post-hoc analysis shows verbosity bias is more severe than position bias, and self-enhancement preferences remain unmitigated.

## Highlights & Insights
-   **"Compositional Bias" is an Apt Framework**: Framing modality bias as a failure of "verification integrity" rather than mere "task-solving capability" explains why judges must be tested more rigorously than solvers.
-   **BD/BC Dual Metrics Prevent Metric Gaming**: BC prevents "constant-score" cheating, while BD prevents "over-sensitivity" cheating. Using them selectively based on bias type with inter-sample variance as a safeguard is a robust design pattern.
-   **9 Bias Types Corresponding to 9 Perturbation Recipes**: Null image/null query/random Q/random I/detail caption/overlay text. Each category is minimalist, reproducible, and easily extensible—a best-practice template for diagnostic benchmarks.
-   **Structural Prompts Offer Partial Mitigation**: Modality reasoning prompts increased Qwen3-VL-8B's overall score from 0.660 to 0.726, suggesting room for inference-time engineering, though results vary by model.

## Limitations & Future Work
-   The authors acknowledge: (1) Coverage is limited to vision+language, excluding video/audio; (2) Only pointwise scoring was used, without pairwise or ranking analysis; (3) Social and cultural biases were not covered.
-   Personal observation: Pointwise 1-10 scoring has upper-bound effects. While excluding $y=1$ helps BD, results may still be influenced by default score distributions (e.g., LLMs favoring "7").
-   Improvements: Expanding to pairwise comparisons ("No image vs. Image") could further control for baseline effects. Calibration-conditioned BD (normalizing against a model's default distribution) could enable fairer comparisons.

## Related Work & Insights
-   **vs. MLLM-as-a-Judge (Chen et al. 2024a)**: They cover 14 tasks but focus on LLM-inherited biases (position/length); this work targets multimodal-specific compositional bias.
-   **vs. Hwang et al. 2025 (Visual Biases)**: They focus narrowly on visual transformation biases in T2I evaluation; this work is more systematic across three dimensions and dual-modality perturbations.
-   **vs. Ye et al. 2025 (12 LLM-as-Judge Biases)**: This serves as a multimodal extension, providing a similarly systematic framework focused on multimodal failure modes.
-   **Insight**: Any work using MLLM-as-a-Judge should undergo a BD/BC check. Reward model training could use "assigning low scores to null-modality inputs" as an auxiliary loss to fix integrality bias.

## Rating
-   Novelty: ⭐⭐⭐⭐ The framing of "Compositional Bias" and the dual-metric approach are fresh, though individual perturbation recipes are adapted from LLM judge research.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 26 models across 9 biases and 1,804 samples, plus multiple prompt interventions and abstention analysis.
-   Writing Quality: ⭐⭐⭐⭐⭐ Clear taxonomy, rigorous metric definitions, and high-density information in Table 2.
-   Value: ⭐⭐⭐⭐⭐ Provides a reproducible diagnostic suite and reliability rankings for 26 SOTA models, serving as a critical tool for alignment and leaderboard construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases](reasoning_model_is_superior_llm-judge_yet_suffers_from_biases.md)
- [\[ACL 2026\] RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity](roleconflictbench_a_benchmark_of_role_conflict_scenarios_for_evaluating_llms39_c.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)

</div>

<!-- RELATED:END -->
