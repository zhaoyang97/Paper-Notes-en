---
title: >-
  [Paper Note] Is LLM an Overconfident Judge? Unveiling the Capabilities of LLMs in Detecting Offensive Language with Annotation Disagreement
description: >-
  [ACL 2025][Social Computing][Offensive language detection] This study systematically evaluates the performance of multiple LLMs in offensive language detection when faced with annotation disagreement, finding that LLMs perform exceptionally well on samples with high annotator agreement (GPT-4o F1 85.24%) but drop sharply to 57.06% on low-agreement samples. Moreover, the models exhibit severe overconfidence on uncertain samples. Further experiments with few-shot learning and i…
tags:
  - "ACL 2025"
  - "Social Computing"
  - "Offensive language detection"
  - "annotation disagreement"
  - "LLM-as-a-judge"
  - "overconfidence"
  - "human-AI alignment"
  - "confidence calibration"
date: 2026-05-08
content_hash: af45f8637c7221ff
---

# Is LLM an Overconfident Judge? Unveiling the Capabilities of LLMs in Detecting Offensive Language with Annotation Disagreement

**Conference**: ACL 2025  
**arXiv**: [2502.06207](https://arxiv.org/abs/2502.06207)  
**Code**: [github.com/DUT-lujunyu/Disagreement](https://github.com/DUT-lujunyu/Disagreement)  
**Area**: Social Computing  
**Keywords**: Offensive language detection, annotation disagreement, LLM-as-a-judge, overconfidence, human-AI alignment, confidence calibration

## TL;DR

This study systematically evaluates the performance of multiple LLMs in offensive language detection when faced with annotation disagreement, finding that LLMs perform exceptionally well on samples with high annotator agreement (GPT-4o F1 85.24%) but drop sharply to 57.06% on low-agreement samples. Moreover, the models exhibit severe overconfidence on uncertain samples. Further experiments with few-shot learning and instruction tuning demonstrate that incorporating disagreement samples during training can simultaneously improve detection accuracy and human-AI alignment.

## Background & Motivation

**Background**: The LLM-as-a-judge paradigm is increasingly popular, with a growing number of studies and systems utilizing LLMs instead of human annotators to judge text content. In the field of offensive language detection, LLMs have been widely applied for automated content moderation.

**Limitations of Prior Work**: Existing evaluations almost strictly adopt a binary classification setup—aggregating annotations via majority voting to obtain a "gold standard" label, and then evaluating LLMs based on it. This approach overlooks a core fact: human annotators often disagree on the same text, especially in subjective tasks involving cultural backgrounds, linguistic ambiguity, and differences in personal perception. Majority voting discards this crucial information about annotation disagreement.

**Key Challenge**: While LLM outputs are deterministic (especially when $\text{temperature}=0$), human perception of offensive language is inherently subjective and uncertain. An ideal judge model should make highly confident decisions when annotators agree, and express uncertainty when there is high annotator disagreement. However, whether LLMs possess this ability to "know what they do not know" has not been systematically investigated.

**Goal**: (RQ1) What is the detection accuracy of LLMs across different levels of annotation agreement? Is there systemic overconfidence? (RQ2) How do disagreement samples affect the decision-making behavior of models in few-shot learning and instruction tuning?

**Key Insight**: By stratifying the MD-Agreement dataset (10,753 tweets, 5 annotators per tweet) according to the level of annotation agreement (A++ full agreement / A+ mild disagreement / A0 weak agreement), the classification accuracy and confidence of LLMs can be evaluated hierarchically on the same data, thereby revealing the systematic relationship between "agreement and performance."

**Core Idea**: To analyze the capabilities of LLMs as judges through stratification based on annotation agreement, thereby exposing their overconfidence defects on ambiguous samples, and demonstrating that disagreement-aware training can effectively mitigate this issue.

## Method

### Overall Architecture

Rather than proposing a new model, this paper designs a systematic evaluation framework to answer two research questions. The entire study is divided into three stages:

1. **Zero-shot Classification Performance Evaluation**: Run zero-shot offensive language detection on 12 LLMs (6 closed-source + 6 open-source) and report accuracy and F1 score stratified by annotation agreement level.
2. **Confidence-Agreement Alignment Analysis**: Estimate LLM confidence using the self-consistency method (averaging multiple samples under 5 different temperature settings), and measure alignment with human annotation agreement using MSE and Spearman correlation coefficient.
3. **Impact of Disagreement Samples on Learning**: Systematically compare the impact of training samples with different agreement levels on model performance and calibration in few-shot learning (GPT-4o) and instruction tuning (LLaMa3-8B), respectively.

### Key Designs

#### Key Design 1: Hierarchical Evaluation Framework Based on Annotation Agreement
- **Function**: Categorize the MD-Agreement dataset into three levels based on the agreement among 5 annotators: A++ (5/5 agreement), A+ (4/5 agreement), and A0 (3/5 agreement). Each level is further subdivided into offensive/non-offensive categories: O++/O+/O0 and N++/N+/N0.
- **Mechanism**: Traditional evaluation only reports an overall F1 score, which masks the performance variations of models on samples of varying difficulty. Hierarchical evaluation introduces "annotation certainty" as a new dimension, making the performance degradation patterns quantifiable and comparable. Meanwhile, soft labels (the average of the 5 hard labels, in the range of $[0,1]$) are introduced as a continuous measure of human uncertainty.
- **Design Motivation**: Human annotation disagreement is not noise but an inherent property of the samples (which has been independently verified in SemEval-2023 Task 11). Therefore, when evaluating LLMs, it is crucial to distinguish between "easy cases" and "hard cases."

#### Key Design 2: Self-Consistency Confidence Estimation
- **Function**: For each sample, sample the model outputs multiple times under $\text{temperature} \in \{0, 0.25, 0.5, 0.75, 1\}$, and average the hard predictions (0/1) across the 5 samples to obtain a confidence score in the range of $[0,1]$.
- **Mechanism**: At $\text{temperature}=0$, the LLM always gives the same deterministic answer, making it impossible to observe its internal uncertainty. Introducing temperature perturbation allows the model to "expose" its degree of certainty on different samples—it is confident if it returns the same answer across all temperatures, and uncertain if the answers flip with temperature changes.
- **Design Motivation**: For black-box closed-source models (such as GPT-4o), internal logits or token probabilities are inaccessible. Self-consistency is the most direct and feasible method for uncertainty estimation. Two complementary metrics, MSE (measuring absolute deviation) and Spearman's $\rho$ (measuring rank correlation), are selected to comprehensively evaluate alignment.

#### Key Design 3: Disagreement-Aware Training Experiments
- **Function**: Systematically compare the effects of using training samples with different agreement levels—single levels (A++/A+/A0) vs. mixed levels (A++/+, A++/0, A+/0, A++/+/0)—under both few-shot learning and instruction-tuning paradigms.
- **Mechanism**: In the few-shot experiments, a pair of positive and negative samples (one offensive + one non-offensive) is constructed for each agreement level as prompt examples. In instruction tuning, an equal number (1800) of samples are drawn from each level for fine-tuning. By comparing the accuracy and MSE of different configurations on the test set, the optimal training data combination strategy is determined.
- **Design Motivation**: If the overconfidence of LLMs stems from a lack of ambiguous samples in their training data, introducing disagreement samples during learning should help the model construct finer decision boundaries.

## Key Experimental Results

### Main Results: Zero-shot Classification Performance (Selected from Table 3)

| Model | Type | Overall Acc | A++ F1 | A+ F1 | A0 F1 | A++ $\rightarrow$ A0 Drop |
|------|------|------------|--------|-------|-------|------------|
| GPT-4o | Closed | 80.36 | 85.24 | 74.60 | 57.06 | -28.18 |
| GPT-o1 | Closed | 78.35 | 81.29 | 72.03 | 58.63 | -22.66 |
| Claude-3.5 | Closed | 78.56 | 83.13 | 72.02 | 62.61 | -20.52 |
| LLaMa3-70B | Open | 76.93 | 81.36 | 71.96 | 64.37 | -16.99 |
| LLaMa3-8B | Open | 71.82 | 70.72 | 66.06 | 61.26 | -9.46 |
| Qwen2.5-72B | Open | 72.08 | 70.92 | 67.36 | 63.76 | -7.16 |

### Confidence-Agreement Alignment (Selected from Table 4)

| Model | A++ MSE $\downarrow$ | A++ $\rho$ $\uparrow$ | A0 MSE $\downarrow$ | A0 $\rho$ $\uparrow$ |
|------|----------|--------|---------|-------|
| GPT-4o | 0.051 | 0.810 | 0.193 | 0.233 |
| Claude-3.5 | 0.066 | 0.759 | 0.202 | 0.238 |
| LLaMa3-70B | 0.075 | 0.763 | 0.207 | 0.237 |

### Ablation Study: Instruction Tuning (Table 6, LLaMa3-8B)

| Training Config | Test Acc | A++ Acc | A0 Acc | A0 MSE $\downarrow$ |
|---------|----------|---------|--------|---------|
| Zero-shot | 70.92 | 85.22 | 54.09 | 0.229 |
| w/ A++ | 75.79 | 89.01 | 58.18 | 0.237 |
| w/ A+ | 77.04 | 90.40 | 59.70 | 0.229 |
| w/ A0 | 73.99 | 86.53 | 56.31 | **0.167** |
| w/ A+/0 | 82.53 | 95.20 | 61.80 | 0.198 |
| w/ A++/+/0 | **84.23** | **95.98** | **64.37** | 0.215 |

### Key Findings
- **LLMs do not know what they do not know**: On A0 samples, all models still show Cohen's $\kappa > 0.75$ (high self-consistency), but their decision directions deviate severely from human judgment—which is the operational definition of "overconfidence."
- **Scaling up models offers limited help for ambiguous cases**: Scaling from LLaMa3-8B to 70B improves F1 on A++ by 10.64%, but only by 3.11% on A0.
- **LLMs skew toward classifying uncertain samples as offensive**: For low-agreement non-offensive samples (N0), the accuracy is only 45.77%, indicating that when models are uncertain, they tend to err on the side of classifying text as offensive.
- **Disagreement samples are the most valuable training data**: Fine-tuning with A0 samples reduces the MSE from 0.229 to 0.167 (best calibration), and mixed training across all three levels achieves the best overall performance of 84.23%.
- **A+ is the optimal choice among single levels**: It achieves the best balance between accuracy and calibration during fine-tuning.

## Highlights & Insights

- **The methodological value of hierarchical evaluation exceeds the specific numbers**: Evaluating LLM-as-a-judge stratified by annotation agreement is a general framework that can be applied to all subjective labeling tasks such as sentiment analysis and humor detection. Future LLM judge evaluations should report stratified results rather than a single aggregated score.
- **Precise characterization of "overconfidence"**: Through self-consistency and confusion matrices, the study intuitively exhibits that GPT-4o still yields extreme confidence values even in the ambiguous region of soft labels ($0.4 \sim 0.6$). This is a training paradigm issue rather than a model size issue.
- **The "curriculum learning" effect of disagreement samples**: Mixing training data from different agreement levels is analogous to curriculum learning—learning simple cases first to establish a basic boundary, then introducing hard samples to refine the decision surface. This strategy can be generalized to other tasks with high annotator uncertainty.

## Limitations & Future Work

- **Single dataset and single task**: Experiments are only conducted on MD-Agreement, without verifying generalizability to other subjective tasks.
- **Basic confidence estimation method**: Self-consistency relies on temperature sampling; logit-based methods could provide more precise calibration for open-source models.
- **Limited number of annotators**: With only 5 annotators, the resolution of agreement levels is limited (only three bins: 3/5, 4/5, 5/5). More annotators would provide a more fine-grained spectrum of disagreement.

## Related Work & Insights

- **vs. Traditional Offensive Language Detection**: Traditional methods train and evaluate on majority-voted labels, entirely ignoring annotation disagreement. This paper introduces annotation disagreement as a core evaluation dimension in LLM evaluation for the first time.
- **vs. LLM-as-a-Judge Research**: Existing LLM judge evaluations focus on ranking consistency between models, whereas this work focuses on alignment between the model and human uncertainty—a deeper aspect of reliability.
- **vs. Uncertainty Estimation Literature**: LLM uncertainty research mostly focuses on factual question answering, while this work focuses on uncertainty in subjective tasks where the "correct answer" itself is ambiguous.

## Rating
- Novelty: ⭐⭐⭐⭐ Evaluating LLM judges through the lens of annotation disagreement is an overlooked and important angle. The hierarchical evaluation framework has general value.
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 LLMs covering closed-source and open-source models across three paradigms (zero-shot, few-shot, and fine-tuning), validated with multiple metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear research questions, with findings unfolding progressively.
- Value: ⭐⭐⭐⭐ Provides a significant warning for LLM-as-a-judge practices, and the hierarchical evaluation framework is highly generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] K/DA: Automated Data Generation Pipeline for Detoxifying Implicitly Offensive Language in Korean](kda_automated_data_generation_pipeline_for_detoxifying_implicitly_offensive_lang.md)
- [\[ACL 2026\] Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews](../../ACL2026/social_computing/justice_in_judgment_unveiling_hidden_bias_in_llm-assisted_peer_reviews.md)
- [\[NeurIPS 2025\] Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector](../../NeurIPS2025/social_computing/any_large_language_model_can_be_a_reliable_judge_debiasing_w.md)
- [\[ACL 2025\] Exploring the Impact of Instruction-Tuning on LLMs' Susceptibility to Misinformation](exploring_the_impact_of_instruction-tuning_on_llms_susceptibility_to_misinformat.md)
- [\[ACL 2025\] BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla](banstereoset_a_dataset_to_measure_stereotypical_social_biases_in_llms_for_bangla.md)

</div>

<!-- RELATED:END -->
