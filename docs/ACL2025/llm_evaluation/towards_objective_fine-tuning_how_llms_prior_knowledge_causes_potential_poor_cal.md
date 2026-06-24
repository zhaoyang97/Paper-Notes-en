---
title: >-
  [Paper Note] Towards Objective Fine-tuning: How LLMs' Prior Knowledge Causes Potential Poor Calibration?
description: >-
  [ACL 2025][LLM Evaluation][Calibration] This paper reveals that the prior knowledge of LLMs leads to calibration degradation during fine-tuning (known data triggers overconfidence, whereas unknown data actually benefits calibration). It proposes CogCalib, a cognition-aware calibration framework that dynamically applies different learning strategies based on knowledge bias during training, reducing the Expected Calibration Error (ECE) by an average of 57% while maintaining tas…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Calibration"
  - "Prior Knowledge"
  - "Fine-tuning"
  - "Knowledge Bias"
  - "Overconfidence"
date: 2026-05-08
content_hash: b9ffa8efd422b088
---

# Towards Objective Fine-tuning: How LLMs' Prior Knowledge Causes Potential Poor Calibration?

**Conference**: ACL 2025  
**arXiv**: [2505.20903](https://arxiv.org/abs/2505.20903)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Calibration, Prior Knowledge, Fine-tuning, Knowledge Bias, Overconfidence

## TL;DR

This paper reveals that the prior knowledge of LLMs leads to calibration degradation during fine-tuning (known data triggers overconfidence, whereas unknown data actually benefits calibration). It proposes CogCalib, a cognition-aware calibration framework that dynamically applies different learning strategies based on knowledge bias during training, reducing the Expected Calibration Error (ECE) by an average of 57% while maintaining task performance.

## Background & Motivation

Fine-tuned LLMs often exhibit poor calibration, meaning there is a discrepancy between the model's confidence scores and its actual accuracy, primarily manifested as overconfidence. This is particularly dangerous in high-risk scenarios such as medical diagnosis and safety-critical applications.

Limitations of existing calibration research:
- **Traditional research focuses on training from scratch**: In simpler models like ResNet, Negative Log-Likelihood (NLL) overfitting is considered a key factor in calibration degradation, but these studies do not account for the model's prior knowledge.
- **Peculiarity of LLM fine-tuning**: Fine-tuning data typically contains known knowledge that overlaps with the pre-training corpus as well as new domain knowledge. The impact of such knowledge bias on calibration has not yet been investigated.
- **Limitations of post-hoc calibration methods**: Existing methods (e.g., Bayesian LoRA, Temperature Scaling) introduce additional modules post-fine-tuning to reconstruct the mapping between outputs and probabilities, which increases deployment overhead and does not address the root cause.

The core finding of this paper is that **the prior knowledge of LLMs is the root cause of calibration degradation**—data consistent with prior knowledge (known data) induces overconfidence, whereas new knowledge data contributes to better calibration. This occurs because after the model quickly digests known data, its confidence continues to grow while its accuracy has already saturated, causing an asynchronous fitting between accuracy and confidence.

## Method

### Overall Architecture

CogCalib (Cognition-aware Calibration) is a real-time fine-tuning calibration framework consisting of three core components:
1. Knowledge Bias Evaluation
2. Adaptive Learning Strategy
3. Style Adaptation

### Key Designs

1. **Knowledge Bias Evaluation**:

    - Core idea: NLL (Negative Log-Likelihood) is utilized as an online metric for knowledge bias. Known data aligns with pre-training priors and yields low NLL, whereas unknown data deviates from prior knowledge and yields high NLL.
    - A threshold $t$ determines whether data is known: if NLL $\le t$, it is classified as known data ($I = 1$); otherwise, it is classified as unknown data ($I = 0$).
    - **Adaptive Threshold Update**: The model's knowledge distribution continuously changes during training, requiring dynamic threshold adjustment. A calibration set is used to find the optimal threshold $t^*$ that maximizes TPR + TNR via grid search.
    - The accuracy of NLL-based classification reaches 98-99% on multiple-choice tasks (e.g., 99.44% on OBQA) and 83-84% on open-ended tasks (e.g., 83.64% on HotpotQA).

2. **Adaptive Learning Strategy**:

    - For known data (low bias): Apply calibration regularization terms to suppress confidence overfitting.
    - For unknown data (high bias): Utilize standard cross-entropy loss to maintain normal learning dynamics.
    - The calibration term can be flexibly selected: Label Smoothing (CoLS), Margin-based Label Smoothing (CoMbLS), or Entropy Confidence Penalty (CoECP).

3. **Style Adaptation**:

    - Problem: Because the LLM's language style and label format differ from those of downstream tasks, directly using NLL cannot accurately evaluate knowledge bias.
    - Solution: Prior to formal training, the model undergoes a short style adaptation process to rapidly adjust to the syntactic patterns of the downstream task, after which the initial threshold $t_0$ is calculated based on the adapted model.

### Loss & Training

Total loss:
$$\mathcal{L} = \mathcal{L}_{CE} + I(p,q) \cdot \alpha \cdot \mathcal{L}_{cal}$$

where $I(p,q) = 1$ denotes known data, triggering the calibration regularization term, and $I(p,q) = 0$ indicates unknown data, using only cross-entropy. $\alpha$ controls the regularization intensity.

## Key Experimental Results

### Main Results

In-distribution (ID) performance of Llama3-8B on 5 multiple-choice datasets (LoRA fine-tuning):

| Dataset | Metric | Vanilla SFT | TS | CoLS ($\Delta$TS) | CoMbLS ($\Delta$TS) | CoECP ($\Delta$TS) |
|--------|------|------------|-----|------------|-------------|-------------|
| OBQA | ECE↓ | 11.20 | 9.90 | **2.50** (-7.4) | 3.70 (-6.2) | 7.30 (-2.6) |
| OBQA | ACC↑ | 84.80 | 84.80 | 85.60 (+0.8) | **86.20** (+1.4) | **86.20** (+1.4) |
| ARC-C | ECE↓ | 16.50 | 12.30 | 4.80 (-7.5) | **4.20** (-8.1) | 7.40 (-4.9) |
| WG-M | ECE↓ | 14.80 | 11.50 | 4.10 (-7.4) | 3.10 (-8.4) | **1.00** (-10.5) |
| BoolQ | ECE↓ | 9.54 | 7.70 | **1.97** (-5.7) | 2.36 (-5.3) | 7.68 (0.0) |

### Ablation Study

Knowledge bias ratio experiment (OBQA, ratio of unknown:known data):

| Ratio (unknown:known) | Key Metric | Description |
|---------------------|---------|------|
| 5:0 (pure unknown) | Best calibration | Calibration is optimal when there is no known data |
| 4:1 | Calibration starts to degrade | A small amount of known data is sufficient to trigger degradation |
| 0:5 (pure known) | Worst calibration | Completely known data leads to the most severe overconfidence |

OOD generalization (OBQA training $\to$ MMLU testing):

| Method | Business ECE | Culture ECE | History ECE | Psychology ECE |
|------|-------------|-------------|-------------|----------------|
| Vanilla SFT | 18.40 | 17.61 | 19.22 | 23.38 |
| TS | 16.10 | 16.70 | 17.40 | 21.30 |
| CoECP | **3.80** (-12.3) | **3.46** (-13.2) | **6.27** (-11.1) | **9.51** (-11.8) |

### Key Findings

- **Known data is the root of overconfidence**: Even with only a small amount of known data (ratio of 4:1), calibration starts to degrade. Accuracy on known data saturates in 200 steps, but confidence continues to climb.
- **Dual beneficial effects of unknown data**: Unknown data not only improves calibration but also has a positive impact on accuracy (consistent with findings that learning new knowledge enhances task performance).
- **NLL is an effective metric for knowledge bias**: The discrimination accuracy exceeds 99% on multiple-choice questions and reaches 83%+ on open-ended tasks.
- **CogCalib is comprehensively effective**: It significantly improves calibration and maintains or enhances task performance across 7 tasks and 3 LLM families (Llama3-8B, Llama2-13B, Mistral-7B, Qwen2.5-7B).
- **Strong OOD generalization**: On MMLU topics with large distribution shifts, ECE decreases by 11-13 percentage points, showing excellent generalization capability.
- **Zero extra deployment overhead**: Unlike post-hoc calibration methods, CogCalib is completed during the training phase, incurring no extra computational cost during inference.

## Highlights & Insights

- **Revealing a neglected causal relationship**: This study systematically reveals for the first time the mechanism of the negative impact of LLM prior knowledge on fine-tuning calibration—specifically, the asynchronous fitting of accuracy and confidence.
- **From observation to solution**: After observing the different fitting dynamic characteristics of known/unknown data, targeted learning strategies are naturally derived.
- **Clever reuse of NLL**: Since NLL is already calculated during the training process, using it as a knowledge bias metric incurs almost zero extra cost.
- **A general interface for calibration methods**: CogCalib is not a new calibration method in itself, but rather a framework that renders existing calibration regularization terms cognition-aware, offering flexibility and extensibility.

## Limitations & Future Work

- The length and strategy of the style adaptation step may vary across tasks, and there is a lack of an automated tuning mechanism.
- The accuracy of NLL-based knowledge bias identification on open-ended QA tasks (~83%) is lower than on multiple-choice questions (~99%), leaving room for improvement in open-ended scenarios.
- The construction of the calibration set and the frequency of threshold updates (e.g., every epoch) may affect performance, but a sufficient sensitivity analysis has not been conducted.
- A simple baseline of purely removing known data yields inconsistent results across different datasets (OBQA improves while ARC-C degrades), indicating that the problem of knowledge selection is more complex.
- Future work could explore applying CogCalib to training phases such as RLHF or preference alignment.

## Related Work & Insights

This work extends the calibration problem from the traditional "model architecture/learning rate" perspective to a "knowledge bias" perspective, establishing a causal chain between the LLM's pre-training knowledge and fine-tuning calibration. While the SliCK framework by Gekhman et al. provided a foundation for knowledge categorization, CogCalib advances this to real-time application during training. For all application scenarios involving LLM fine-tuning (especially high-risk fields such as medicine and law), the method proposed in this paper holds direct application value.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ This work systematically reveals for the first time the causal mechanism of prior knowledge on LLMs calibration, providing deep insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive coverage with 7 tasks, 4 models, and multiple fine-tuning methods.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from phenomenon observation to analysis and solution is highly clear, decorated with exquisite figures.
- Value: ⭐⭐⭐⭐⭐ Direct guiding significance for LLM fine-tuning practices, especially in high-risk application scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] On the Robustness Tradeoff in Fine-Tuning](../../ICCV2025/llm_evaluation/on_the_robustness_tradeoff_in_fine-tuning.md)
- [\[ACL 2025\] EvoWiki: Evaluating LLMs on Evolving Knowledge](evowiki_evaluating_llms_on_evolving_knowledge.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)
- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](../../NeurIPS2025/llm_evaluation/hyperbolic_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] GRACE: A Granular Benchmark for Evaluating Model Calibration Against Human Calibration](grace_a_granular_benchmark_for_evaluating_model_calibration_against_human_calibr.md)

</div>

<!-- RELATED:END -->
