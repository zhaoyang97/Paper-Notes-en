---
title: >-
  [Paper Note] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models
description: >-
  [ACL 2025][Social Computing][Bias Detection] BiasGuard is proposed to detect LLM output bias by explicitly reasoning about fairness specifications. In the first stage, a teacher model generates reasoning trajectories for SFT initialization; in the second stage, DPO is utilized to enhance reasoning quality. The method outperforms classifiers and LLM-as-the-Judge approaches across 5 datasets while reducing over-fairness false positives.
tags:
  - "ACL 2025"
  - "Social Computing"
  - "Bias Detection"
  - "Reasoning Enhancement"
  - "Fairness Specifications"
  - "SFT"
  - "DPO"
  - "CoT reasoning"
date: 2026-05-08
content_hash: 6ce1e2db1eb7a8d4
---

# BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2504.21299](https://arxiv.org/abs/2504.21299)  
**Code**: None  
**Area**: Social Computing  
**Keywords**: Bias Detection, Reasoning Enhancement, Fairness Specifications, SFT, DPO, CoT reasoning

## TL;DR
BiasGuard is proposed to detect LLM output bias by explicitly reasoning about fairness specifications. In the first stage, a teacher model generates reasoning trajectories for SFT initialization; in the second stage, DPO is utilized to enhance reasoning quality. The method outperforms classifiers and LLM-as-the-Judge approaches across 5 datasets while reducing over-fairness false positives.

## Background & Motivation
**Background**: Detecting LLM output bias is a prerequisite for ensuring fairness. Existing approaches fall into two categories: (a) training bias classifiers (e.g., Toxigen RoBERTa, Llama-Guard-3) for binary classification; (b) directly utilizing LLMs as evaluators (LLM-as-Judge).

**Limitations of Prior Work**: Classifiers rely heavily on pattern matching. While effective for explicit bias, they fail to grasp the deeper underlying intents of implicit bias (e.g., the Toxigen classifier achieves only 41.3% accuracy on the Implicit Toxicity dataset). Conversely, LLM-as-the-Judge lacks clear fairness criteria, leading to over-sensitivity and high false positive rates (i.e., the "over-fairness" problem where everything is judged as biased).

**Key Challenge**: Bias detection requires deep semantic understanding (treating it as a reasoning task rather than a classification task) while strictly adhering to human-defined fairness specifications (preventing the LLM from hallucinating its own standards).

**Goal**: To enable the bias detection model to explicitly reason about fairness specifications before making a judgment—specifically, by first analyzing intent, then verifying against rules, and finally drawing a conclusion.

**Key Insight**: Compiling bias definitions and quantitative evaluation rules from sociology literature as fairness specifications, and employing a two-stage training scheme to teach the model how to perform CoT reasoning based on these specifications.

**Core Idea**: Analyze sentence structures and intents $\to$ verify against fairness specifications $\to$ make the final judgment, instead of performing direct end-to-end classification.

## Method

### Overall Architecture
Given an LLM-generated text $\mathbf{x}$, BiasGuard $\pi_\theta$ outputs $(\text{CoT}, \mathbf{y} | \mathbf{s}, \mathbf{x})$—that is, it first performs chain-of-thought reasoning based on the fairness specifications $\mathbf{s}$, and then produces a "biased/unbiased" judgment.

### Key Designs

1. **Fairness Specifications**:

    - **Function**: Compiling definitions and quantitative judgment rules for various types of biases (e.g., gender, race, age) from sociological literature.
    - **Mechanism**: Referring to sociological research such as Burgess & Borgida (1999) (gender bias) and Balibar et al. (2007) (racial bias), combined with the quantitative evaluation criteria of Hammersley & Gomm (1997).
    - **Design Motivation**: Providing explicit "legal statutes" to the LLM—specifications that guide the model to systematically analyze the sentence structure, interpret intent and attitude, and judge based on rules rather than classifying by raw intuition.

2. **Stage 1 — SFT Initialization of Reasoning Ability**:

    - **Function**: Enabling the base model to learn diverse reasoning following the specifications.
    - **Mechanism**: Utilizing a teacher model (DeepSeek-R1-32B) to generate $k=4$ reasoning trajectories per sample, filtering those with correct conclusions to serve as SFT data. The reasoning format is "Step 1: Analyze Intent $\to$ Step 2: Check Specifications $\to$ Step 3: Make Judgment".
    - **Design Motivation**: Initializing the model's reasoning path distribution so it can generate diverse yet structured reasoning processes.

3. **Stage 2 — DPO to Enhance Reasoning Discriminability**:

    - **Function**: Further promoting reasoning quality through preference optimization.
    - **Mechanism**: Sampling $N=8$ reasoning trajectories using a high temperature ($\tau=1.2$) from the SFT model, and pairing correct/incorrect responses to construct DPO training data. The optimization objective is: $\mathcal{L}(\pi_\theta; \pi_{\text{SFT}}) = -\log\sigma(\beta \log\frac{\pi_\theta(\text{CoT}_w, \mathbf{y}_w|\mathbf{x})}{\pi_{\text{SFT}}(\text{CoT}_w, \mathbf{y}_w|\mathbf{x})} - \beta \log\frac{\pi_\theta(\text{CoT}_l, \mathbf{y}_l|\mathbf{x})}{\pi_{\text{SFT}}(\text{CoT}_l, \mathbf{y}_l|\mathbf{x})})$
    - **Design Motivation**: While SFT instills the reasoning path, its discriminative capability is limited. DPO enhances the model's ability to distinguish between good and bad reasoning through exploration and exploitation.

### Training Details
- Base: DeepSeek-R1-Distill-Qwen-14B
- Teacher: DeepSeek-R1-Distill-Qwen-32B
- Training Data: A subset of RedditBias + Toxigen

## Key Experimental Results

### Main Results: Bias Detection on 5 Datasets (Accuracy / Over-Fairness Rate OF↓)

| Method | Toxigen | Implicit Toxi. | SBIC | GabHate | Reddit |
|------|---------|----------------|------|---------|--------|
| Toxigen Classifier | 90.3/0.25 | 41.3/4.35 | 55.6/38.4 | 60.3/4.85 | 53.5/15.1 |
| Llama-Guard-3 | 49.3/9.40 | 34.6/0.25 | 58.4/22.0 | 49.1/2.65 | 57.5/11.6 |
| GPT-4o (Vanilla) | 66.8/10.3 | 54.3/5.00 | 58.0/40.4 | 62.1/16.1 | 53.9/16.7 |
| GPT-4o + Specs | 68.4/8.45 | **75.0**/5.60 | 80.8/5.60 | 70.9/16.5 | 75.0/10.0 |
| **BiasGuard** | 73.2/8.00 | 81.0/1.25 | 74.0/13.2 | 71.3/12.5 | **79.3**/8.90 |

### Ablation Study

| Configuration | Average Accuracy | Description |
|---|---|---|
| Base (Vanilla LLM) | ~50% | Direct prompting |
| w. Rule (with specifications) | ~65% | Specification guidance provides significant improvement |
| Instruction Tuning | ~68% | Fine-tuning is helpful |
| CoT SFT (Stage 1) | ~72% | Reasoning initialization is effective |
| **CoT DPO (Stage 2)** | **~76%** | DPO yields significant enhancement |

### Key Findings
- **BiasGuard achieves optimal performance on 3/5 datasets** while maintaining a low over-fairness rate across all of them—avoiding the issue of "judging everything as biased".
- **Classifiers exhibit poor generalizability**: The Toxigen classifier reaches 90.3% on its own dataset but drops to only 41.3% on Implicit Toxicity, whereas BiasGuard shows more balanced performance.
- **Specification guidance is effective for all LLMs**: GPT-4o with specifications jumps from 53-66% to 68-80%, suggesting that clear standards are crucial.
- **Reasoning ability scales with model size**: The performance continuously improves as the base size increases from 1.5B $\to$ 7B $\to$ 14B $\to$ 32B, showcasing stable scaling properties for reasoning enhancement.

## Highlights & Insights
- **Bias Detection = Reasoning Task**: This task requires understanding target intent in context and adhering to established specifications, rather than relying on surface patterns. This framework substantively redefines the problem.
- **Sociology-grounded fairness specifications** are theoretically well-founded—by providing the LLM with definitive judgment standards, they address the "lack of standardized criteria" issue inherent to LLM-as-the-Judge.
- **Complementary two-stage design**: SFT teaches the model reasoning paths (*how to reason*), while DPO teaches it discrimination (*which reasoning is better*)—a pipeline that could be extended to other specification-bounded reasoning tasks.

## Limitations & Future Work
- **Compiling specifications requires manual review of sociological literature**, which incurs high costs when scaling to new bias categories—could LLMs be utilized to assist in specification generation?
- **English-only scenarios**: Definitions of bias vary significantly across different cultures, and these specifications cannot be directly transferred.
- **The 14B base model has cognitive upper bounds on complex implicit bias**: Larger models may yield further improvements.
- **BiasGuard underperforms DeepSeek-R1-32B + Rules (93.6%) on the SBIC dataset**: This is likely because the bias categories in SBIC align poorly with the training distribution.

## Related Work & Insights
- **vs Classifiers (Toxigen/Llama-Guard)**: Classifiers exhibit poor generalizability—while achieving high accuracy on specific datasets, their performance drops substantially across others, whereas BiasGuard remains more balanced.
- **vs LLM-as-the-Judge (Vanilla GPT-4o)**: The lack of concrete standards leads to over-sensitivity (with an over-fairness rate of up to 40%), which is significantly mitigated by introducing specifications.
- **vs DeepSeek-R1**: While R1 inherently possesses strong reasoning capabilities, BiasGuard achieves comparable performance using a smaller model (14B vs. 32B), demonstrating the effectiveness of the training scheme.
- This work holds straightforward practical value for content moderation and LLM safety evaluation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The "reasoning + specifications" paradigm for bias detection is novel, effectively redefining bias detection from a classification task to a reasoning task.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Incorporates 5 datasets (including in-domain and out-of-domain), multiple baselines, comprehensive ablation studies, and scaling experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Formulates the problem clearly, with Figure 1 intuitively showcasing the reasoning pipeline.
- **Value**: ⭐⭐⭐⭐ Acts as a plug-and-play bias detection tool with high practical utility for fairness evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Translate With Care: Addressing Gender Bias, Neutrality, and Reasoning in Large Language Model Translations](translate_with_care_addressing_gender_bias_neutrality_and_reasoning_in_large_lan.md)
- [\[NeurIPS 2025\] Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector](../../NeurIPS2025/social_computing/any_large_language_model_can_be_a_reliable_judge_debiasing_w.md)
- [\[ACL 2025\] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language](exploring_gender_bias_in_large_language_models_an_in-depth_dive_into_the_german_.md)
- [\[ACL 2025\] Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection](explicit_vs_implicit_investigating_social_bias_in_large_language_models_through_.md)
- [\[ACL 2025\] A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models](a_survey_on_proactive_defense_strategies_against_misinformation_in_large_languag.md)

</div>

<!-- RELATED:END -->
