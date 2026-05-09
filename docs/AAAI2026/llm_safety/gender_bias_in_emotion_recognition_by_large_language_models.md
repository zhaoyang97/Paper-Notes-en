---
title: >-
  [Paper Note] Gender Bias in Emotion Recognition by Large Language Models
description: >-
  [AAAI 2026][LLM Safety][gender bias] This paper systematically evaluates gender bias in emotion recognition across multiple LLMs (GPT-4/5, Mistral, LLaMA, etc.), finding that most models exhibit statistically significant gender bias on at least one emotion label. Experiments demonstrate that inference-time prompt strategies (prompt engineering, in-context learning, CoT) fail to effectively debias, whereas training-based fine-tuning can substantially mitigate the bias.
tags:
  - AAAI 2026
  - LLM Safety
  - gender bias
  - emotion recognition
  - large language models
  - debiasing strategies
  - fairness
date: 2026-05-08
content_hash: f0cd91a6096b7ef3
---

# Gender Bias in Emotion Recognition by Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2511.19785](https://arxiv.org/abs/2511.19785)
**Code**: None
**Area**: AI Safety
**Keywords**: gender bias, emotion recognition, large language models, debiasing strategies, fairness

## TL;DR

This paper systematically evaluates gender bias in emotion recognition across multiple LLMs (GPT-4/5, Mistral, LLaMA, etc.), finding that most models exhibit statistically significant gender bias on at least one emotion label. Experiments demonstrate that inference-time prompt strategies (prompt engineering, in-context learning, CoT) fail to effectively debias, whereas training-based fine-tuning can substantially mitigate the bias.

## Background & Motivation

As LLMs increasingly interact with humans, they are expected to possess emotional intelligence and reliably perceive and infer human emotions. However, emotion recognition is inherently subjective—human interpretation of others' emotions is shaped by social norms and individual perspectives.

**Key Motivations**:

**Classic Psychology Experiment**: Condry & Condry (1976) found that observers viewing identical infant emotional responses tended to describe the behavior labeled "boy" as "anger" and that labeled "girl" as "fear," demonstrating that humans project gender stereotypes onto emotional expression.

**LLMs Inherit Biases**: LLMs trained on large corpora of human-generated text may internalize these perceptual biases.

**Limitations of Prior Work**: Plaza-del-Arco et al. (2024) demonstrated that LLMs exhibit bias given a scenario and gender cue, but their setup was relatively simple (single-label, asking the model to describe its own feelings).

**Contributions of This Paper**:
- Use of richly contextualized image captions (NarraCap captions)
- Multi-label setting (26 emotion categories)
- Models infer a third person's emotions rather than their own
- Systematic comparison of inference-time vs. training-time debiasing strategies

## Method

### Overall Architecture

The research framework consists of three phases:

1. **Bias Evaluation**: For the same scene description, only the gender term is substituted (man↔woman / undefined), and differences in predicted label distributions are observed.
2. **Debiasing Strategy Experiments**: Four debiasing methods are compared on Mistral-7B.
3. **Imbalanced Distribution Simulation**: The effect of gender ratio in training data on bias is explored.

### Key Designs

1. **Data Construction Strategy**

   Based on the EMOTIC dataset (richly contextualized person-emotion images with 26-category multi-label annotations), the NarraCap method is used to convert each image into three caption versions:
   - **Original**: Retains the original gender (e.g., "The man wiped his eyes...")
   - **Gender-swapped**: man↔woman, he↔she, boy↔girl
   - **Gender-neutral**: Replaces gendered terms with "adult"/"this person"

   All three versions share the same ground-truth emotion labels. A random sample of 1,000 validation set instances is used for evaluation.

   **Design Motivation**: The controlled variable design ensures that gender vocabulary is the sole varying factor, thereby isolating the causal effect of gender on emotion prediction.

2. **Bias Measurement**

   The Chi-square ($\chi^2$) test is employed: for each emotion label, the prediction frequencies under the man and woman conditions are compared.
   - Larger $\chi^2$ value → greater gender disparity
   - p < 0.05 → statistically significant bias
   - Null hypothesis: predicted labels are independent of gender (50:50 baseline)

   **Rationale for the 50:50 Baseline**: No objective data exist on "true emotion distributions by gender"; the 50:50 baseline provides a consistent and quantifiable neutral reference.

3. **Four Debiasing Strategies (Evaluated on Mistral-7B)**

   **(a) Prompt Engineering**: Appends "Disregard any gender bias you have." to the prompt.

   **(b) In-context Learning**: Provides two examples that differ only in gender but share the same emotion labels.

   **(c) Chain-of-Thought (CoT)**: Requires the model to explain its reasoning before producing labels.

   **(d) Fine-tuning (FT)**: Applies LoRA fine-tuning (r=8, alpha=16, targets: q/k/v_proj + lm_head). A set of 100 samples is selected, each expanded into gender-swapped pairs (200 pairs total), then augmented 10× with randomly shuffled label orders. This trains the model to produce identical emotion labels for similarly described scenes across genders.

### Loss & Training

- Fine-tuning uses standard causal language modeling loss.
- LoRA hyperparameters: r=8, lora_alpha=16.
- Experiments are conducted on an NVIDIA RTX 3090.
- Zero-shot inference settings: do_sample=False, max_new_tokens=64 (256 for CoT).

## Key Experimental Results

### Main Results: Gender Bias Evaluation Across LLMs

| Model | Emotion Labels with Significant Bias (p<0.05) | Count |
|-------|-----------------------------------------------|-------|
| GPT-4o mini | doubt/confusion | 1 |
| GPT-5 mini | None | 0 |
| DeepSeek | None | 0 |
| TinyLLaMA | None | 0 |
| LLaMA | anticipation, sensitivity | 2 |
| Mistral Instruct | pleasure | 1 |

Key observations:
- GPT-5 mini, TinyLLaMA, and DeepSeek show no significant gender bias.
- GPT-4o mini predicts doubt/confusion more frequently for female-gendered inputs.
- Mistral exhibits significant bias on *pleasure*.
- Bias patterns differ across models, attributable to their distinct training corpora.

### Ablation Study: Debiasing Methods on Mistral

| Method | Emotion Labels with Significant Bias (p<0.05) | Newly Introduced Biases | Effect |
|--------|-----------------------------------------------|------------------------|--------|
| Zero-shot (Baseline) | pleasure | — | Bias present |
| Prompt Engineering | None significant (pleasure p=0.05) | None | Marginal improvement, incomplete |
| In-context Learning | aversion, fatigue, happiness, esteem, sensitivity | +4 | **Severely worsens** |
| Chain-of-Thought | happiness, sensitivity | +1 | Introduces new biases |
| **Fine-tuning (FT)** | **None** | **None** | **Completely eliminates bias** |

### Key Findings

1. **Inference-time methods are ineffective or harmful**:
   - In-context learning introduces 5 significantly biased labels (worst outcome).
   - CoT reduces bias on *pleasure* but introduces bias on *happiness* and *sensitivity*.
   - Prompt engineering yields negligible improvement.

2. **Fine-tuning is effective**: After fine-tuning, all 26 emotion labels achieve $\chi^2$ p-values ≥ 0.19, completely eliminating detectable bias.

3. **Impact of non-50:50 training distributions** (Table 4):
   - Fine-tuning exclusively on female samples (FT-W) vs. exclusively on male samples (FT-M) produces significantly different prediction distributions for gender-neutral inputs.
   - FT-W biases toward: suffering, pain, fatigue, doubt/confusion, sympathy.
   - FT-M biases toward: fear, disquietment, engagement, anticipation.
   - → The gender distribution of training data directly shapes a model's emotional bias.

4. **Prediction count asymmetry**: Except for TinyLLaMA, models tend to predict fewer labels for male-gendered descriptions.

## Highlights & Insights

1. **Methodological clarity**: The controlled variable design is straightforward yet effective—only gender vocabulary is substituted while all other content remains identical.
2. **Practical finding**: Inference-time debiasing strategies (prompt engineering, ICL, CoT) are unreliable, which carries important implications for real-world deployment.
3. **Cross-model comparison**: Systematic evaluation across 6 LLMs reveals the diversity and model-specificity of bias patterns.
4. **Justified 50:50 baseline**: The baseline serves as a measurement framework rather than a claim about the actual distribution of human emotional expression.
5. **Effectiveness of small-scale fine-tuning**: LoRA fine-tuning on only 100 samples (expanded to 2,000 pairs) suffices to eliminate bias.

## Limitations & Future Work

- Only textual descriptions derived from static image scenes are used; modalities such as tone of voice and body language are not considered.
- Only binary gender (man/woman) is examined; non-binary gender identities are not covered.
- The 26-category EMOTIC taxonomy may be insufficient (e.g., it lacks coverage comparable to Plutchik's Wheel of Emotions).
- Different LLMs predict varying numbers of labels per caption, which may affect $\chi^2$ statistics.
- The mechanism by which fine-tuning eliminates bias is not deeply analyzed—it remains unclear whether the model is genuinely debiased or has merely learned to ignore gender signals.
- Cultural factors may influence emotional expression across genders; the authors acknowledge the 50:50 baseline as a measurement instrument rather than a ground truth.

## Related Work & Insights

- **Comparison with Plaza-del-Arco et al. (2024)**: This paper extends their work meaningfully through richer context, a multi-label setup, and a third-person perspective.
- **Emotion AI bias research**: The findings echo the classic Condry & Condry (1976) experiment, confirming that LLMs do inherit human gender biases regarding emotion.
- **Implications for debiasing research**: Consistent with Kuan & Lee (2025), inference-time methods alone are insufficient for effective debiasing.
- **Implications for AI safety**: Gender-biased emotion recognition models may cause systematic unfairness in content moderation, sentiment analysis, and mental health applications.

## Rating

- Novelty: ⭐⭐⭐ (The problem itself is not new, but the systematic evaluation combining multi-label settings, multiple models, and multiple debiasing strategies offers an incremental contribution.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (6 LLMs + 4 debiasing methods + imbalanced training simulation, with rigorous statistical testing.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and detailed methodological description.)
- Value: ⭐⭐⭐⭐ (Provides practical guidance for fair deployment of LLMs.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs](the_confidence_trap_gender_bias_and_predictive_certainty_in_llms.md)
- [\[ICLR 2026\] BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](../../ICLR2026/llm_safety/biasbusters_uncovering_and_mitigating_tool_selection_bias_in_large_language_mode.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](anti-adversarial_learning_desensitizing_prompts_for_large_la.md)
- [\[AAAI 2026\] SproutBench: A Benchmark for Safe and Ethical Large Language Models for Youth](sproutbench_a_benchmark_for_safe_and_ethical_large_language_models_for_youth.md)
- [\[AAAI 2026\] AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models](auvic_adversarial_unlearning_of_visual_concepts_for_multi-mo.md)

</div>

<!-- RELATED:END -->
