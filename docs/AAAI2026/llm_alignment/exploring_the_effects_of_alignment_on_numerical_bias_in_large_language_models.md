---
title: >-
  [Paper Note] Exploring the Effects of Alignment on Numerical Bias in Large Language Models
description: >-
  [AAAI 2026][LLM Alignment][Numerical Bias] This paper systematically demonstrates that the LLM alignment process (instruction tuning + preference tuning) is the root cause of numerical bias in LLM evaluators…
tags:
  - "AAAI 2026"
  - "LLM Alignment"
  - "Numerical Bias"
  - "LLM-as-a-Judge"
  - "Alignment Side Effects"
  - "Kurtosis Analysis"
  - "Evaluation Robustness"
  - "Score Range Adjustment"
date: 2026-05-08
content_hash: 0a30e5a9e7316f2e
---

# Exploring the Effects of Alignment on Numerical Bias in Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2601.16444](https://arxiv.org/abs/2601.16444)  
**Code**: Not available  
**Area**: LLM Alignment / LLM Evaluation
**Keywords**: Numerical Bias, LLM-as-a-Judge, Alignment Side Effects, Kurtosis Analysis, Evaluation Robustness, Score Range Adjustment

## TL;DR

This paper systematically demonstrates that the LLM alignment process (instruction tuning + preference tuning) is the root cause of numerical bias in LLM evaluators, and validates that score range adjustment is the most effective mitigation strategy.

## Background & Motivation

**Background**: The LLM-as-a-Judge paradigm has demonstrated effectiveness across various tasks, including machine translation quality estimation and grammatical error correction evaluation, and is increasingly accepted as an alternative to human annotation. However, researchers have observed a pervasive phenomenon in LLM evaluators—numerical bias—whereby certain score values are generated at disproportionately high frequencies, leading to severely skewed scoring distributions.

**Limitations of Prior Work**: Although prior work (e.g., Stureborg et al.) identified biases in LLM evaluators—such as preference for low-perplexity text and reduced accuracy when score ranges are expanded—none traced the root cause of numerical bias. These efforts addressed symptoms rather than the underlying cause.

**Key Challenge**: Alignment is a prerequisite for LLMs to function as competent evaluators—without alignment, models cannot correctly interpret evaluation instructions. Yet alignment itself compresses output diversity, which may be the very source of numerical bias. This creates a dilemma: no alignment → inability to evaluate; alignment → biased evaluation.

**Goal**: The paper investigates two core research questions—RQ1: How does alignment affect numerical bias? RQ2: How can numerical bias in aligned models be mitigated?

**Key Insight**: The study employs kurtosis as a statistical measure to quantify the concentration of scoring distributions. By comparing pre- and post-alignment versions of the same model families across three regression evaluation tasks, it establishes a causal link between alignment and bias.

**Core Idea**: The excessive conformity introduced by alignment is the root cause of numerical bias, and score range should be treated as a tunable hyperparameter rather than a fixed design choice.

## Method

### Overall Architecture

The study proceeds in two phases. The first addresses alignment impact analysis (RQ1), comparing pre- and post-alignment versions of four model families on MTQE, GECQE, and LCP regression evaluation tasks in terms of scoring distributions and evaluation performance. The second phase addresses bias mitigation (RQ2), exploring three strategies: temperature scaling, distributional calibration, and score range adjustment.

The models used include Gemma-7B, Mistral-7B-v0.1, Llama-3-8B, and Qwen2-7B, each available in both base (pre-alignment) and instruct (post-alignment) variants.

### Key Designs

**Score Generation Pipeline**: For each input sample, the LLM generates 10 scores using a fixed prompt template (max_token=5). Non-numeric outputs are filtered out, out-of-range values are clipped, and the final score is taken as the mean of valid outputs.

**Kurtosis as Bias Metric**: Kurtosis is chosen over variance or interquartile range because it better captures local concentration of a distribution around specific values. Higher kurtosis indicates greater concentration of scores at particular values, reflecting more severe bias.

**Distributional Calibration**: Drawing on the generative calibration method of Jiang et al., the approach reweights scores to correct the marginal scoring distribution. Specifically, the model distribution $p(y)$ is estimated via 1,000 samples, the target distribution $q(y)$ is fitted using a Beta distribution, and the final score for each sample is computed as a weighted average of 10 draws: $\hat{y} = \sum_{i=1}^{n} w_i y_i$, where $w_i \propto q(y_i)/p(y_i)$.

**Score Range Adjustment**: By modifying the scoring range in the prompt (1–5, 0–9, or 1–100), the set of possible output tokens is directly altered, thereby influencing both the degree of bias and evaluation performance.

### Loss & Training

This paper involves no model training; all core methods are applied at inference time. The temperature parameter is fixed at 0.7 for RQ1 and varied over {0.4, 0.7, 1.0, 1.3} in the temperature scaling experiments for RQ2.

## Key Experimental Results

### Main Results: Pre- vs. Post-Alignment Kurtosis and Pearson Correlation (MTQE Task)

| Language Pair | Model | Gold Kurtosis | Pre-Align Kurtosis | Post-Align Kurtosis | Pre-Align $r$ | Post-Align $r$ |
|---|---|---|---|---|---|---|
| En-De | Gemma | 1.48 | 0.27 | **128.17** | 0.33 | 0.08 |
| En-De | Mistral | 1.48 | 0.41 | **52.92** | 0.11 | 0.20 |
| En-De | Llama | 1.48 | 2.53 | **21.05** | 0.33 | 0.35 |
| En-De | Qwen | 1.48 | 0.53 | **21.05** | 0.33 | 0.38 |
| Ne-En | Gemma | 0.97 | −0.34 | 0.45 | 0.21 | 0.37 |
| Ne-En | Llama | 0.97 | −0.64 | −0.95 | 0.32 | 0.45 |
| Si-En | Gemma | −0.72 | −0.37 | 1.97 | 0.24 | 0.38 |
| Si-En | Qwen | −0.72 | 0.48 | −0.85 | 0.45 | 0.48 |

### Ablation Study: Comparison of Three Mitigation Strategies (MTQE En-De)

| Model | Orig. Kurtosis | Calib. Kurtosis | Range 1–5 Kurtosis | Range 1–100 Kurtosis | Orig. $r$ | Calib. $r$ | Range 1–5 $r$ | Range 1–100 $r$ |
|---|---|---|---|---|---|---|---|---|
| Gemma | 128.17 | 47.93 | 71.79 | 87.45 | 0.08 | 0.08 | 0.05 | **0.14** |
| Mistral | 52.92 | 3.39 | 11.95 | 42.00 | 0.20 | 0.15 | **0.22** | 0.21 |
| Llama | 21.05 | 11.89 | 10.19 | 23.96 | **0.35** | 0.38 | 0.27 | 0.33 |
| Qwen | 4.21 | 2.70 | 6.87 | 5.28 | **0.38** | 0.38 | 0.34 | 0.37 |

### Key Findings

- **Alignment substantially amplifies numerical bias**: Kurtosis increases sharply after alignment for nearly all models. Gemma's kurtosis on En-De rises from 0.27 to 128.17—86 times that of the gold distribution. Post-alignment scores are heavily concentrated around 8 regardless of input content.
- **Bias severity varies by model**: Gemma exhibits the most severe bias, followed by Mistral, while Llama and Qwen are comparatively less affected—suggesting that different alignment procedures have varying impacts on numerical bias.
- **High-resource language pairs show stronger bias**: Bias is markedly more pronounced for high-resource pairs such as En-De and En-Zh than for low-resource pairs such as Ne-En and Si-En, presumably because low-resource languages appear less frequently in alignment training data and are thus less affected by alignment side effects.
- **Kurtosis negatively correlates with evaluation accuracy**: Across MTQE and GECQE, the correlation between kurtosis and Pearson correlation coefficient is −0.60, indicating that stronger bias corresponds to lower evaluation accuracy.
- **Score range adjustment is most effective**: Among the three mitigation strategies, temperature scaling can reduce kurtosis but does not reliably improve correlation; distributional calibration yields inconsistent results; score range adjustment can simultaneously reduce bias and improve accuracy in some settings (e.g., Gemma MTQE improves from 0.08 to 0.14).
- **Input characteristics influence bias severity**: Higher fluency (lower perplexity) sentences are more prone to bias; in GECQE, larger edit distances (lower lexical overlap) are associated with stronger bias.

## Highlights & Insights

- **First causal link established**: Rather than merely observing bias, the paper uses controlled experiments—comparing pre- and post-alignment versions of the same model family—to establish a causal relationship between alignment and numerical bias.
- **Practical implications**: Treating score range as a tunable hyperparameter offers direct guidance for practitioners using LLM-as-a-Judge—the default 0–10 range should not be assumed optimal; the range should be explored per task and model.
- **Kurtosis as a proxy for model selection**: In the absence of gold labels, kurtosis can serve as a screening criterion for evaluator selection—models with higher kurtosis tend to exhibit lower evaluation accuracy.

## Limitations & Future Work

- The alignment procedures and data of the open-source models examined are opaque, precluding deeper analysis of which specific alignment operation (SFT vs. DPO vs. RLHF) is the primary source of bias.
- Only numerical scoring tasks are studied; the effects of alignment on natural language label evaluation and ranking-based evaluation remain unexplored.
- Score range adjustment remains a heuristic approach; the optimal range depends on the task and model, and no theoretical guidance is provided.
- Experiments are limited to models in the 7–8B parameter range; bias characteristics of larger models (70B+) may differ.
- Methods for directly mitigating bias during the alignment training phase (e.g., modifying the distribution of alignment datasets) are not explored.

## Related Work & Insights

- **Stureborg et al. (2024)**: Identified three biases in LLM evaluation (low-perplexity preference, range sensitivity, and influence of prior evaluations) but did not trace the root cause of numerical bias.
- **Santurkar et al. (2023)**: Found that alignment reduces diversity in model outputs, providing a theoretical basis for the paper's hypothesis.
- **Jiang et al. (2023)**: Proposed the generative calibration method adopted in this paper as the distributional calibration strategy.
- **Insights**: This work highlights that while alignment is necessary, its side effects warrant systematic investigation. Similar approaches could be extended to other alignment side effects, such as reduced creativity and opinion homogenization.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic quantification of the relationship between alignment and numerical bias
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 model families × 3 task types × 7 language pairs, providing broad coverage
- Writing Quality: ⭐⭐⭐⭐ Clear research questions, rigorous experimental design, and coherent analytical logic
- Value: ⭐⭐⭐⭐ Offers direct practical guidance for users of LLM-as-a-Judge

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search](w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_.md)
- [\[AAAI 2026\] Align to Structure: Aligning Large Language Models with Structural Information](align_to_structure_aligning_large_language_models_with_struc.md)
- [\[AAAI 2026\] BiasJailbreak: Analyzing Ethical Biases and Jailbreak Vulnerabilities in Large Language Models](biasjailbreakanalyzing_ethical_biases_and_jailbreak_vulnerabilities_in_large_lan.md)
- [\[AAAI 2026\] EASE: Practical and Efficient Safety Alignment for Small Language Models](ease_practical_and_efficient_safety_alignment_for_small_language_models.md)
- [\[NeurIPS 2025\] Alignment of Large Language Models with Constrained Learning](../../NeurIPS2025/llm_alignment/alignment_of_large_language_models_with_constrained_learning.md)

</div>

<!-- RELATED:END -->
