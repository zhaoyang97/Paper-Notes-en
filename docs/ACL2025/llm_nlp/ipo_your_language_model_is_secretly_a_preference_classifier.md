---
title: >-
  [Paper Note] IPO: Your Language Model is Secretly a Preference Classifier
description: >-
  [ACL 2025][LLM (Other)][Preference Optimization] Proposes Implicit Preference Optimization (IPO), which leverages the generative LLM itself as a preference classifier (via the probability of "Yes/No" tokens) instead of an external reward model to obtain preference signals, achieving low-cost self-alignment training.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Preference Optimization"
  - "Reward Modeling"
  - "Self-Improvement"
  - "DPO"
  - "RLHF"
date: 2026-05-08
content_hash: fde255ee7c9f9650
---

# IPO: Your Language Model is Secretly a Preference Classifier

**Conference**: ACL 2025  
**arXiv**: [2502.16182](https://arxiv.org/abs/2502.16182)  
**Code**: [github](https://github.com/shivank21/Implicit_Preference_Optimization)  
**Area**: LLM/NLP  
**Keywords**: Preference Optimization, Reward Modeling, Self-Improvement, DPO, RLHF

## TL;DR

Proposes Implicit Preference Optimization (IPO), which leverages the generative LLM itself as a preference classifier (via the probability of "Yes/No" tokens) instead of an external reward model to obtain preference signals, achieving low-cost self-alignment training.

## Background & Motivation

RLHF has become the mainstream method for aligning LLMs with human preferences, but it relies on training external reward models or human-annotated preference data, which is computationally and financially expensive. Existing Self-Rewarding methods, although using the LLM itself for scoring, typically employ discrete 1-5 ratings, which lack granularity and perform poorly on smaller models. Furthermore, these methods require large models to act as judges, and the judging capabilities of the models are not updated during training.

The core hypothesis of this work is that instead of using discrete prompt-based scoring, providing a continuous preference magnitude is superior. The authors find that LLMs inherently possess implicit preference-classification capabilities—by examining the model's probability of outputting "Yes" to the question "Is this response good?", fine-grained preference signals can be obtained.

## Method

### Overall Architecture

The IPO framework consists of two phases:
1. **Preference Classification**: Leverages the LLM itself as a preference classifier to score responses based on the probability of the "Yes" token.
2. **Self-Improvement Training**: Constructs preference datasets based on the scoring and performs training using DPO.

### Key Designs

1. **Probability-based Preference Classification**: Given an instruction and a response, a category-specific prompt (divided into Chat, Code, Math, and Safety) is used to ask the model "whether the response is appropriate". The logits of "Yes" and "No" in the first output token are extracted, which are then passed through softmax and normalized to obtain the preference score $p'_{yes} = \frac{p_{yes}}{p_{yes} + p_{no}}$. The response with the higher score is designated as chosen, while the lower score is designated as rejected.

2. **Category-Specific Prompt Design**: Tailored guide prompts are designed for four distinct task categories (Chat, Code, Math, and Safety), and an automated prompt selection pipeline is established. This is particularly crucial for smaller models, as they are highly sensitive to prompt phrasing.

3. **Preference Dataset Construction**: For each instruction, the SFT/Instruct model generates 4 candidate responses. The IPO method is used to score all responses, selecting the highest-scoring one as chosen and the lowest-scoring one as rejected, forming (Prompt, Chosen, Rejected) triplets.

### Loss & Training

A standard two-stage training strategy is adopted:
- **SFT Phase**: Supervised fine-tuning on the Dolly-15k dataset, optimizing the cross-entropy loss.
- **DPO Phase**: Performs preference alignment using the constructed preference dataset to optimize the DPO loss.

Using 4k instructions from UltraFeedback as input prompts, a Bart-Zero Shot classifier is employed to categorize them into four fields.

## Key Experimental Results

### Main Results

Preference classification accuracy on RewardBench:

| Model | Chat | Code | Math | Safety | Average |
|------|------|------|------|--------|------|
| Qwen-2.5-7B-Inst (IPO) | 78.26 | 83.13 | 56.24 | 93.24 | 77.71 |
| Mistral-7B-Inst (IPO) | 61.25 | 70.93 | 96.20 | 83.85 | 78.05 |
| Qwen-2.5-7B-Inst (Self-Rewarding) | 58.73 | 47.93 | 40.49 | 52.20 | 49.82 |
| Mistral-7B-Inst (Self-Rewarding) | 24.55 | 1.6 | 28.18 | 15.39 | 17.43 |

Downstream task performance after DPO training:

| Model | BBH | Arc-Easy | Alpaca-Eval | MMLU | IFEval | Average |
|------|-----|----------|-------------|------|--------|------|
| Mistral-7B-Ours | 34.60 | 82.20 | 78.20 | 37.60 | 39.19 | 54.35 |
| Mistral-7B-Reward | 30.20 | 85.20 | 77.40 | 41.00 | 31.69 | 53.10 |
| Mistral-7B-Self Rewarding | 31.20 | 77.00 | 69.60 | 33.00 | 29.31 | 48.02 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Base vs Instruct Models | Instruct is significantly better | Instruction tuning improves preference classification capability |
| Large vs Small Models | Larger models perform better | Qwen-7B >> Qwen-3B |
| Code Models | On par with reward models | Qwen-Coder exhibits strong performance |
| Math Models | Significantly worse | Possibly due to incompatibility between CoT training objectives and binary classification |

### Key Findings

1. IPO significantly outperforms the Self-Rewarding method across all subcategories, with a more pronounced performance gap on smaller models.
2. The discrete 1-5 rating of the Self-Rewarding method struggles to differentiate response quality, frequently assigning identical scores to both chosen and rejected samples.
3. Code-specific models inherently possess preference classification capability, while math-specific models do not (likely due to differences in training objectives).
4. Most models perform best in the Safety category, indicating that safety alignment has been widely integrated into training.
5. Mistral-7B trained with IPO matches or even surpasses the training based on the Skywork reward model across multiple benchmarks.

## Highlights & Insights

- **Extremely simple yet effective idea**: Eliminates the need to train reward models or use complex prompts to cast LLMs as judges. It only requires extracting the probability of a single token, incurring minimal computational cost.
- **Reveals implicit preference capabilities of LLMs**: Even base models exhibit a certain degree of preference classification ability, suggesting that quality judgment is already implicitly learned during pre-training.
- **Continuous vs. discrete signals**: Probability values provide significantly finer-grained preference signals than discrete scores (1-5), serving as a crucial improvement in preference representation.

## Limitations & Future Work

1. Requires pre-classification of datasets (Chat/Code/Math/Safety), though letting the model automatically generate category labels is a viable alternative.
2. Evaluated only at 1B and 7B scales, using a relatively small subset of 4k prompts.
3. Due to computational constraints, a single-turn DPO was used instead of iterative DPO (as configured in Self-Rewarding).
4. All experimental results are from a single run (seed=42), lacking comprehensive robustness validation.
5. Smaller models cannot generate high-quality instructions using simple prompting, requiring external datasets.

## Related Work & Insights

- This work shares a similar philosophy with VQA Score (Lin et al., 2025), where probabilities are leveraged as scoring signals.
- Compared to Meta-Rewarding, IPO is more lightweight as it does not require additional training for judging.
- IPO could potentially be combined with iterative training (e.g., SPIN) to allow the model's preference classification capability to co-evolve alongside training.
- It also holds direct application value for test-time scaling (such as best-of-n selection).

## Rating

- Novelty: ⭐⭐⭐⭐ The idea is simple and ingenious, but the core concept leans towards incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐ Wide model coverage, but DPO experiments are relatively small in scale.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with abundant data presentation.
- Value: ⭐⭐⭐⭐ Provides a cost-effective alternative for obtaining preference signals, possessing practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidence](selfelicit_evidence_highlighting.md)
- [\[ACL 2025\] Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)
- [\[ACL 2025\] Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](gapo_multi_objective_alignment.md)
- [\[ACL 2025\] Are Your LLMs Capable of Stable Reasoning?](are_your_llms_capable_of_stable_reasoning.md)
- [\[ACL 2025\] Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up](reversal_of_thought_enhancing_large_language.md)

</div>

<!-- RELATED:END -->
