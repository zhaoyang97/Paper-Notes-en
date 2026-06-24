---
title: >-
  [Paper Note] UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models
description: >-
  [ACL2025][LLM Safety][Uncertainty Estimation] This work proposes the UAlign framework, which leverages two uncertainty estimations—confidence score and semantic entropy—to explicitly model the knowledge boundary of LLMs. By incorporating these estimations as input features into PPO alignment training, the model is guided to answer known questions confidently and refuse unknown ones firmly, significantly improving reliability and generalization across multiple factual QA datas…
tags:
  - "ACL2025"
  - "LLM Safety"
  - "Uncertainty Estimation"
  - "Factuality Alignment"
  - "Knowledge Boundary"
  - "PPO"
  - "Semantic Entropy"
  - "Confidence"
date: 2026-05-08
content_hash: c787dae3b9618012
---

# UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models

**Conference**: ACL2025  
**arXiv**: [2412.11803](https://arxiv.org/abs/2412.11803)  
**Code**: [AmourWaltz/UAlign](https://github.com/AmourWaltz/UAlign)  
**Area**: LLM Safety  
**Keywords**: Uncertainty Estimation, Factuality Alignment, Knowledge Boundary, PPO, Semantic Entropy, Confidence

## TL;DR

This work proposes the UAlign framework, which leverages two uncertainty estimations—confidence score and semantic entropy—to explicitly model the knowledge boundary of LLMs. By incorporating these estimations as input features into PPO alignment training, the model is guided to answer known questions confidently and refuse unknown ones firmly, significantly improving reliability and generalization across multiple factual QA datasets.

## Background & Motivation

LLMs acquire a vast amount of knowledge during the pre-training phase, but often fail to accurately express the factual knowledge they possess in downstream tasks.
The core problem lies in the **fuzzy knowledge boundary** of LLMs, which manifests at three levels:

**Discarding weakly-known knowledge**: The model actually "knows" but is uncertain about certain questions, resulting in correct answers in only a fraction of multiple rollouts. Previous methods like R-Tuning directly label these questions as "unknown" and train the model to refuse them, wasting knowledge that could have been answered correctly.

**Overconfidently answering unknown knowledge**: Models generate plausible-looking answers even for completely unfamiliar questions, causing severe hallucination issues and undermining user trust.

**Limitations of existing alignment methods**: Prior work on factual alignment (e.g., R-Tuning, RLKF, RL-DPO) does not explicitly exploit knowledge boundary information. They either formulate it as a simple binary classification of known/unknown or estimate it indirectly via knowledge probing, without directly feeding uncertainty metrics into the model as inputs.

The core insight of UAlign is that explicitly quantifying the LLM's uncertainty for each question and integrating this information as additional input features into alignment training enable the model to better understand its own knowledge boundary.
This is equivalent to incorporating "confidence level" and "answer dispersion" prompts to help the model make more prudent decisions—answering known questions boldly and refusing unknown ones decisively.

## Method

### Overall Architecture

UAlign consists of two main stages:

- **Stage 1: Dataset Preparation** — Performs multiple sampling on the knowledge QA dataset to calculate confidence and semantic entropy.
- **Stage 2: UAlign Training** — First trains the uncertainty estimation model and the reward model via SFT, and then aligns the policy model using PPO.

### Stage 1: Dataset Preparation

**Multi-query Sampling Strategy**:
For each question in the dataset, generation is repeated using $K=10$ different 1-shot prompt templates and a sampling temperature $T=0.2$.
Each sample yields a candidate answer, whose correctness is labeled by comparing it to the ground-truth answer.
If all $K$ samples are incorrect, the question is categorized as "unknown", and the ground-truth answer is rewritten as a refusal response: "Sorry, I don't know."

**Uncertainty Metric 1 — Confidence Score**:
This is defined as the proportion of correct answers among the $K$ samples, reflecting the model's "probability of answering correctly" for the given question.
Intuitively, a higher confidence score indicates that the LLM is more certain about this piece of knowledge.

**Uncertainty Metric 2 — Semantic Entropy**:
An NLI model is first used to cluster semantically equivalent responses into the same semantic set, and then the entropy of the cluster distribution is calculated.
Semantic entropy measures the dispersion of generated answers at the semantic level. Even if confidence is low, the entropy will remain low if all answers collapse into a few semantic concepts.

**Complementarity of the Two Metrics**:
Confidence measures "how likely the model is to answer correctly," while semantic entropy measures "how semantically dispersed the model's responses are."
Key scenario: A question has a confidence score of only 40% (low correctness rate) but high semantic entropy (other answers are highly dispersed). In this case, although the correct answer is not dominant, it remains the most concentrated cluster. The model should be guided to output this answer instead of refusing it.

### Stage 2: UAlign Training

**SFT Sub-stage — Training the Estimation Model and Reward Model**:

- **Uncertainty Estimation Model** (predicting confidence and semantic entropy): Built on a vanilla LLM base and fine-tuned with LoRA (rank=4). The input is only the question, and the target is to predict the corresponding confidence score or semantic entropy value.
- **Reward Model**: Also built on the LLM base with LoRA (rank=4). The input consists of the question + the two predicted uncertainty values + the candidate answer, while the output is the probability of correctness, trained using binary cross-entropy loss.

Key design: The input to the reward model explicitly includes uncertainty estimations, allowing it to leverage knowledge boundary information to judge answer quality more accurately.

**PPO Sub-stage — Policy Model Alignment**:

- The input to the policy model is: question + predicted confidence + predicted semantic entropy.
- The input to the reference model is: only the question (without uncertainty information).
- The reward function consists of two parts: the score signal from the reward model and a KL divergence penalty.
- PPO is used to maximize this reward, guiding the policy model to generate more factual responses based on the knowledge boundary information.

All LLMs are fine-tuned using LoRA (rank=16) and trained on 4x NVIDIA A100-40GB GPUs.

## Key Experimental Results

### Experimental Setup

- **Models**: Llama-3-8B, Mistral-7B
- **Training Sets**: Three factual QA datasets: TriviaQA (TVQA), SciQ, and NQ-Open.
- **Test Sets**: Validation/test sets of the three datasets above (In-Domain, ID) + LSQA multilingual QA dataset (Out-of-Domain, OOD).
- **Evaluation Metrics**: Precision (the proportion of correctly answered questions among known ones), Truthfulness (the overall proportion of correctly answering known questions + correctly refusing unknown ones).

### Main Results (Table 1, Llama-3-8B)

| Method | TVQA Prec. | TVQA Truth. | SciQ Prec. | NQ Prec. | Avg ID Prec. | LSQA OOD Prec. |
|------|-----------|------------|-----------|---------|-------------|---------------|
| ICL | 76.15 | 56.55 | 70.43 | 50.28 | 65.62 | 77.35 |
| R-Tuning | 72.93 | 55.44 | 71.38 | 47.81 | 64.04 | 71.54 |
| RL-PPO | 76.32 | 55.19 | 75.70 | 54.07 | 68.03 | 72.18 |
| RLKF | 77.12 | 56.07 | 72.36 | 54.86 | 68.11 | 74.95 |
| **UAlign** | **79.14** | **57.04** | **76.44** | **56.60** | **70.72** | **79.56** |

On Mistral-7B, UAlign achieves a Prec. of 82.10 and Truth. of 59.05 on TVQA, also leading by a wide margin.
Notably, while most training methods experience a degradation in performance on the OOD dataset, UAlign still outperforms all methods (including prompt-based baselines) on LSQA.

### Ablation Study: Impact of Uncertainty Metrics on Reward Model Accuracy (Table 2)

| Confidence | Semantic Entropy | TVQA | SciQ | NQ-Open | LSQA (OOD) |
|--------|--------|------|------|---------|-----------|
| x | x | 82.31 | 79.00 | 67.45 | 70.12 |
| o | x | 85.41 | 84.30 | 70.37 | 75.09 |
| x | o | 82.05 | 77.90 | 67.85 | 70.40 |
| o | o | **86.73** | **86.40** | **72.00** | **74.59** |

The above shows the results for Llama-3-8B. Key Findings:

- Confidence contributes the most; adding it alone improves the reward model accuracy by 3-5 percentage points.
- Semantic entropy is unstable when used alone, even slightly decreasing performance on some datasets.
- Using both combined achieves the best performance in most settings.

### Impact of Number of Samples $K$

As $K$ increases from 1 to 4, 7, and 10, both Prec. and Truth. continuously improve, but with diminishing gains.
Performance largely converges at $K=10$, with marginal benefits from further increasing $K$.
The time cost of performing $K=10$ sampling for 10,000 QA instances on 4x A100 GPUs is controllable, as the answers are entity-level short texts.

## Highlights & Insights

- **Novel Explicit Knowledge Boundary Modeling**: First to incorporate uncertainty estimation as an explicit prompt input into the RLHF alignment workflow; the approach is intuitive and highly effective.
- **Exquisite Co-design of Complementary Metrics**: Combining confidence ("probability of correctness") and semantic entropy ("answer dispersion") helps redeem weakly-known knowledge that is correct but has low confidence.
- **Outstanding OOD Generalization**: UAlign is the only training method that consistently outperforms prompt-based baselines on LSQA (OOD).
- **Connection to Test-Time Scaling**: The pipeline of computing uncertainty after multiple rollouts to guide alignment echoes the trend of test-time compute scaling.

## Limitations & Future Work

- **Narrow Task Scope**: Only validated on short-answer knowledge QA, without extension to open-ended generation, long-text writing, or reasoning tasks.
- **Reliance on Ground-Truth Answers**: Confidence calculation requires ground-truth labels to evaluate the correctness of the sampled answers, making it challenging to directly transfer to unlabeled scenarios.
- **Linear Growth in Computation Cost**: Constructing the dataset requires $K$ samplings, scaling linearly with the size of $K$ and the dataset.
- **Instability of Semantic Entropy**: Ablation studies show that semantic entropy underperforms or fluctuates when used in isolation, even slightly degrading performance on certain datasets.

## Related Work & Insights

- **R-Tuning**: Formulates SFT after classifying known/unknown via sampling, without using RL or explicit knowledge boundary inputs.
- **RLKF**: Trains a reward model with knowledge probing and consistency checks before PPO, leaving knowledge boundary information only implicitly in the reward signal.
- **RL-DPO**: Constructs factual preference pairs for DPO alignment without involving uncertainty estimation.
- **ITI**: Intervenes in attention head activations at inference time; training-free but with limited effectiveness.
- **UAlign**: Explicitly passes confidence and semantic entropy as prompt inputs to both the reward and policy models, representing the core innovation that distinguishes it from all prior methods.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of explicitly integrating uncertainty estimation into the alignment workflow is novel, though the individual components (confidence, semantic entropy, PPO) are combinations of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 2 models, 4 datasets, various baselines, and detailed ablations, but lacks validation on open-ended generation tasks.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic, abundant figures and tables, with intuitive visual explanations of the knowledge boundary.
- Value: ⭐⭐⭐⭐ — Provides a new perspective for factuality alignment, but practical applications are restricted to short-answer QA scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models](chinese_simpleqa_a_chinese_factuality_evaluation_for_large_language_models.md)
- [\[ACL 2026\] FAITH: Factuality Alignment through Integrating Trustworthiness and Honestness](../../ACL2026/llm_safety/faith_factuality_alignment_through_integrating_trustworthiness_and_honestness.md)
- [\[ACL 2025\] ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty](comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)
- [\[ACL 2026\] Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](../../ACL2026/llm_safety/learning_uncertainty_from_sequential_internal_dispersion_in_large_language_model.md)
- [\[ACL 2025\] Unveiling and Addressing Pseudo Forgetting in Large Language Models](unveiling_and_addressing_pseudo_forgetting_in_large_language_models.md)

</div>

<!-- RELATED:END -->
