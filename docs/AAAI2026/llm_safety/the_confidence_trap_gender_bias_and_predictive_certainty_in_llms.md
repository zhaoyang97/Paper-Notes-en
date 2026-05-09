---
title: >-
  [Paper Note] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs
description: >-
  [AAAI2026][LLM Safety][LLM Fairness] This paper proposes Gender-ECE, a metric for systematically evaluating the confidence calibration and alignment with human bias judgments of six open-source LLMs on gendered pronoun prediction tasks. The authors find that Gemma-2 exhibits the worst calibration and an extreme disparity between male and female pronoun calibration, whereas GPT-J-6B — trained on less filtered data — achieves the best calibration overall.
tags:
  - AAAI2026
  - LLM Safety
  - LLM Fairness
  - Gender Bias
  - Confidence Calibration
  - Expected Calibration Error
  - Coreference Resolution
  - Gender-ECE
date: 2026-05-08
content_hash: d543ef5f80296992
---

# The Confidence Trap: Gender Bias and Predictive Certainty in LLMs

**Conference**: AAAI2026
**arXiv**: [2601.07806](https://arxiv.org/abs/2601.07806)
**Authors**: Ahmed Sabir, Markus Kängsepp, Rajesh Sharma (University of Tartu)
**Code**: [GitHub](https://github.com/ahmedssabir/GECE)
**Area**: AI Safety
**Keywords**: LLM Fairness, Gender Bias, Confidence Calibration, Expected Calibration Error, Coreference Resolution, Gender-ECE

## TL;DR

This paper proposes Gender-ECE, a metric for systematically evaluating the confidence calibration and alignment with human bias judgments of six open-source LLMs on gendered pronoun prediction tasks. The authors find that Gemma-2 exhibits the worst calibration and an extreme disparity between male and female pronoun calibration, whereas GPT-J-6B — trained on less filtered data — achieves the best calibration overall.

## Background & Motivation

The widespread deployment of LLMs in high-stakes domains such as recruitment, healthcare, and law has made model trustworthiness an increasingly pressing concern. Models not only inherit gender biases from training data but may also amplify stereotypes. The critical issue lies not only in detecting bias, but in ensuring that users can reliably interpret model predictions — particularly when models exhibit uneven confidence distributions across gender groups.

Calibration is a core dimension of model trustworthiness: a well-calibrated model that assigns 80% confidence to a prediction should be correct approximately 80% of the time. However, existing research on LLM bias has rarely examined the calibration quality of biased predictions. If a model is overconfident in gender-biased scenarios yet frequently incorrect, deployment risks are substantial.

While a large body of work has studied bias and stereotypes in LLMs, almost no research has examined whether model prediction confidence aligns with human-annotated bias judgments. This paper fills that gap by focusing on the gendered pronoun resolution task, analyzing the probabilistic calibration of LLMs, and proposing Gender-ECE — a novel metric designed specifically to measure gender-based calibration disparities.

## Core Problem

**To what extent are LLMs well-calibrated in their predictive confidence on gendered pronoun resolution tasks?** Specifically: (1) Are predictions made with high confidence actually correct? (2) Do systematic calibration error differences exist between male and female pronouns? (3) Can calibration metrics capture fairness-relevant disparities?

## Method

### Overall Architecture

For a sentence containing a pronoun $S = (w_1, w_2, \ldots, w_T)$, the model probability at pronoun position $k$ is extracted as:

$$P(w_p \mid w_1, \ldots, w_{k-1}) = \frac{e^{z_{k-1, w_p}}}{\sum_{j=1}^{V} e^{z_{k-1, j}}}$$

where $z_{k-1, w_p}$ is the model's logit for pronoun $w_p$ at position $k$. By comparing the model's probability assignments for "him" versus "her", confidence behavior is evaluated in occupational gender-bias contexts (e.g., *nurse*, *developer*). The evaluation pipeline proceeds as follows: deterministic forward pass → pronoun token probability extraction → precise alignment via offset mapping → computation of multiple calibration metrics.

### Key Design: Gender-ECE

Standard ECE partitions predictions into $M$ bins $B_m$ and computes the weighted absolute difference between average confidence and average accuracy per bin:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

However, ECE cannot reveal differences in model behavior between male and female pronouns. This paper proposes Gender-ECE:

$$\text{Gender-ECE} = \frac{1}{2} \left( \text{ECE}_{\text{male}} + \text{ECE}_{\text{female}} \right)$$

where $\text{ECE}_{\text{male}}$ and $\text{ECE}_{\text{female}}$ are computed on subsets where the model's **predicted label** is male and female, respectively. Unlike MacroCE, which groups instances by whether predictions are correct or incorrect, Gender-ECE groups by predicted gender label, directly reflecting calibration quality for each gender. Unlike cc-ECE, which groups by ground-truth label, Gender-ECE focuses on model preference (predicted label), making it more sensitive to confidence-level bias.

### Other Calibration Metrics

- **ICE (Instance Calibration Error)**: $\text{ICE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{p}_i|$, computing per-instance confidence deviation from the true label
- **MacroCE**: Groups instances by correct/incorrect predictions, computes ICE for each group, then averages
- **Brier Score**: $\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2$, mean squared error of probabilistic predictions

## Key Experimental Results

### Datasets

- **WinoBias**: 3,160 Winograd-style sentences with pronouns embedded mid-sentence, testing occupational gender bias
- **Winogender**: 720 template sentences introducing a gender-neutral "someone" to avoid stereotyping
- **GenderLex**: 1,676 sentence pairs with pronouns placed at sentence end (last cloze), generated by ChatGPT and manually corrected

### Models: GPT-J-6B, Llama-3.1-8B, Gemma-2-9B, Qwen2.5-7B, Falcon3-7B, DeepSeek-8B

### Calibration Results on GenderLex (Table 1)

| Model | ECE↓ | MacroCE↓ | ICE↓ | Brier↓ | Gender-ECE(Group) | G-ECE(M) | G-ECE(F) | Human↑ |
|------|------|----------|------|--------|-------------------|----------|----------|--------|
| GPT-J-6B | **0.076** | 0.453 | 0.374 | 0.432 | **0.076** | 0.085 | 0.066 | 0.715 |
| Llama-3.1-8B | 0.111 | 0.466 | 0.371 | 0.446 | 0.111 | 0.112 | 0.109 | **0.727** |
| Gemma-2-9B | 0.327 | 0.493 | 0.390 | 0.559 | 0.267 | 0.330 | 0.204 | 0.617 |
| Qwen2.5-7B | 0.106 | 0.476 | 0.422 | **0.385** | 0.107 | 0.052 | 0.162 | 0.637 |
| Falcon3-7B | 0.161 | 0.491 | 0.449 | 0.356 | 0.149 | 0.081 | 0.217 | 0.605 |
| DeepSeek-8B | 0.085 | 0.461 | 0.369 | 0.470 | 0.090 | 0.074 | 0.106 | 0.686 |

### Per-Gender ECE (Table 3)

| Model | WinoBias-M | WinoBias-F | GenderLex-M | GenderLex-F |
|------|-----------|-----------|-------------|-------------|
| GPT-J-6B | 0.206 | 0.508 | 0.373 | 0.377 |
| Llama-3.1-8B | 0.197 | 0.559 | 0.396 | 0.333 |
| **Gemma-2-9B** | **0.067** | **0.895** | **0.056** | **0.901** |
| Qwen2.5-7B | 0.130 | 0.596 | 0.426 | 0.416 |
| Falcon3-7B | 0.215 | 0.502 | 0.505 | 0.363 |
| DeepSeek-8B | 0.158 | 0.606 | 0.303 | 0.469 |

Gemma-2-9B achieves an ECE of only 6–7% on male pronouns, yet reaches 89–90% on female pronouns — an extreme disparity.

### WinoQueer (LGBTQ+ Bias, Table 4)

| Model | Gay | Lesbian | Trans | Queer |
|------|-----|---------|-------|-------|
| GPT-J-6B | 0.121 | 0.790 | 0.816 | 0.700 |
| Qwen2.5-7B | 0.189 | **0.898** | **0.919** | **0.788** |
| Gemma-2-9B | **0.026** | 0.221 | 0.586 | 0.182 |
| DeepSeek-8B | 0.277 | 0.838 | 0.258 | 0.910 |

On the LGBTQ+ task, findings are reversed: Gemma-2-9B becomes the best-calibrated model, while Qwen2.5-7B performs worst.

### Improvement After Beta Calibration

| Model | Pre-calibration Accuracy | Post-calibration Accuracy |
|------|------------|------------|
| GPT-J-6B | 69.2% | 76.9% |
| Llama-3.1-8B | 65.8% | 74.9% |
| Qwen2.5-7B | 61.1% | 76.4% |
| Gemma-2-9B | 51.6% | 54.7% |
| DeepSeek-8B | 63.5% | 69.9% |

Beta post-hoc calibration reduces ECE by approximately threefold while improving accuracy, though it does not constitute a bias mitigation strategy.

### Effect of Model Scale (Table 6, WinoBias)

| Model | ECE | Gender-ECE(M) | Gender-ECE(F) |
|------|-----|---------------|---------------|
| Gemma-2-9B | 0.429 | 0.438 | 0.156 |
| Gemma-2-27B | 0.366 (↓14.7%) | 0.341 (↓22.1%) | 0.381 (↑144.2%) |

Scaling up the model improves calibration for male pronouns, but female pronoun calibration error increases by 144.2%.

## Highlights & Insights

- **Gender-ECE metric**: The first calibration metric to group ECE computation by predicted gender label, directly revealing confidence-level bias toward different genders — more targeted than standard ECE and MacroCE
- **Counterintuitive finding**: GPT-J-6B, trained on the least filtered data, achieves the best calibration, suggesting that data augmentation techniques such as gender swapping may in fact disrupt model confidence
- **Extreme gender disparity**: Gemma-2 exhibits an ECE of only 7% for male pronouns and 90% for female pronouns on WinoBias, quantitatively exposing severe gender calibration asymmetry
- **Bias propagation through distillation**: DeepSeek-8B (distilled from Llama-3.1-8B) shows higher calibration error and lower human alignment, indicating that distillation degrades calibration quality
- **Scale paradox**: Increasing model size improves male calibration but worsens female calibration (Gemma-2-27B female ECE ↑144%), challenging the "bigger is better" assumption

## Limitations & Future Work

- **English-only and binary gender pronouns**: Non-binary pronouns (they/them) and multilingual settings are not addressed
- **Template-only evaluation**: Winograd-style controlled sentences diverge substantially from natural language; Table 8 shows that ECE differences vanish on free-text captions
- **Calibration ≠ bias mitigation**: Beta calibration improves confidence reliability but does not eliminate underlying bias
- **Sample size sensitivity**: Table 7 shows that with 50 samples, ECE standard deviation reaches 0.038, making calibration evaluation unreliable on small datasets
- **Closed-source models excluded**: Only open-source models are evaluated; comparisons with GPT-4, Claude, and similar systems are absent
- **Moderate inter-annotator agreement**: GenderLex annotators achieve a Cohen's $\kappa = 0.51$, indicating only moderate agreement

## Related Work & Insights

- **Kadavath et al. (2022)**: Assesses LLM self-confidence via prompting using $P(\text{true})$; this paper directly uses token logit probabilities, offering a more low-level and reproducible approach
- **Kapoor et al. (2024)**: Calibrates QA confidence through fine-tuning; this paper targets calibration in bias-sensitive scenarios without additional training
- **Zhao et al. (2018) WinoBias**: Introduces a gender bias benchmark without examining calibration quality; this paper extends that foundation by incorporating a calibration dimension
- **MacroCE (Si et al. 2022)**: Instance-wise calibration grouped by correctness; Gender-ECE instead groups by predicted gender and adopts bin-wise computation, yielding greater stability and interpretability
- **Cheng et al. (2023)**: Evaluates LLM stereotypes through generated persona descriptions — a more open-ended approach that is difficult to quantify comparatively

### Broader Implications

- **Confidence auditing as a deployment gate**: Prior to deploying LLMs in high-stakes applications, calibration disparities across demographic subgroups should be examined; Gender-ECE can serve as a standard checklist item
- **The double-edged sword of data filtering**: The finding that GPT-J achieves the best calibration suggests that aggressive data cleaning or augmentation may introduce new calibration issues worthy of attention in data engineering
- **Calibration-aware distillation**: Distillation should preserve not only task performance but also calibration quality; incorporating calibration-aware regularization terms into the distillation loss warrants exploration
- **Integration with AI safety evaluation**: Gender-ECE can be generalized into a Group-ECE framework, enabling calibration fairness evaluation across arbitrary demographic attributes such as race, age, or nationality

## Rating

- Novelty: ⭐⭐⭐⭐ — Gender-ECE is a meaningful contribution; the cross-disciplinary perspective linking calibration with fairness is relatively novel, though technical depth is limited (essentially a simple variant of ECE)
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three benchmarks + WinoQueer + scale ablation + post-hoc calibration experiments provide broad coverage; closed-source models and non-English evaluation are absent
- Writing Quality: ⭐⭐⭐⭐ — Structure is clear and research-question-driven, with rich tables and reliability diagrams; some analyses are repetitive
- Value: ⭐⭐⭐⭐ — Highly practical; Gender-ECE can be directly applied to pre-deployment fairness auditing of LLMs, and the extreme findings on Gemma-2 carry important cautionary implications

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](gender_bias_in_emotion_recognition_by_large_language_models.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](../../NeurIPS2025/llm_safety/on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[AAAI 2026\] Can Editing LLMs Inject Harm?](can_editing_llms_inject_harm.md)
- [\[AAAI 2026\] Uncovering Bias Paths with LLM-guided Causal Discovery: An Active Learning and Dynamic Scoring Approach](uncovering_bias_paths_with_llm-guided_causal_discovery_an_active_learning_and_dy.md)
- [\[AAAI 2026\] From Single to Societal: Analyzing Persona-Induced Bias in Multi-Agent Interactions](from_single_to_societal_analyzing_persona-induced_bias_in_multi-agent_interactio.md)

<!-- RELATED:END -->
