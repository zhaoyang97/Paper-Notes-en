---
title: >-
  [Paper Note] Empirical Privacy Variance
description: >-
  [ICML2025][LLM Safety][Differential Privacy] Reveals that under the same $(ε,δ)$-DP guarantee, language models trained with different DP-SGD hyperparameter configurations exhibit significant variations in empirical privacy (degree of memorization), and proposes a hyperparameter selection heuristic that balances empirical privacy.
tags:
  - "ICML2025"
  - "LLM Safety"
  - "Differential Privacy"
  - "DP-SGD"
  - "Empirical Privacy"
  - "Memorization"
  - "Hyperparameter Selection"
  - "Language Model Fine-tuning"
date: 2026-05-08
content_hash: b71190ed5c4d5536
---

# Empirical Privacy Variance

**Conference**: ICML2025  
**arXiv**: [2503.12314](https://arxiv.org/abs/2503.12314)  
**Code**: [empvv/empirical-privacy-variance](https://github.com/empvv/empirical-privacy-variance)  
**Area**: Privacy / AI Safety  
**Keywords**: Differential Privacy, DP-SGD, Empirical Privacy, Memorization, Hyperparameter Selection, Language Model Fine-tuning

## TL;DR

Reveals that under the same $(ε,δ)$-DP guarantee, language models trained with different DP-SGD hyperparameter configurations exhibit significant variations in empirical privacy (degree of memorization), and proposes a hyperparameter selection heuristic that balances empirical privacy.

## Background & Motivation

Differential Privacy (DP) represents the mainstream standard for protecting training data privacy. DP-SGD satisfies $(ε,δ)$-DP guarantees by gradient clipping and Gaussian noise injection. However, a significant gap remains between the theoretical guarantees of DP and the privacy risks actually perceived by users:

- **Theory vs. Empirics**: DP provides a worst-case mathematical guarantee, whereas users are more concerned with whether the model leaks sensitive information (such as phone numbers, emails, etc.) during interactions.
- **Core Problem**: Are the empirical privacy levels consistent across models calibrated to the same $(ε,δ)$-DP guarantee?
- **Ours Findings**: The answer is **no**. Different hyperparameter configurations (batch size $b$, number of iterations $T$, learning rate $η$) yield models with chest-and-shoulders differences in memorization behavior even under the identical DP guarantee. The authors term this phenomenon **Empirical Privacy Variance**.

## Method

### 1. DP-SGD Recapitulation

DP-SGD updates the model at each step $t$ by clipping gradients and injecting Gaussian noise:

$$\bar{g}_t = \frac{1}{|S_t|} \left( \sum_{x \in S_t} \frac{\nabla_{w_t} \ell(w_t; x)}{\max\left(1, \frac{\|\nabla_{w_t} \ell(w_t; x)\|}{c}\right)} + \mathcal{N}(0, \sigma^2 c^2 I) \right)$$

where the noise multiplier $\sigma$ is computed by a PRV accountant to satisfy the target $(ε,δ)$-DP. Key hyperparameters include batch size $b$, iteration count $T$, learning rate $η$, and clipping norm $c$.

### 2. Empirical Privacy Metrics

The authors define three empirical privacy metrics based on memorization:

- **ACR (Adversarial Compression Ratio)**: Measures the efficiency with which secret information is stored in model weights, $\text{ACR}(s) = |s| / |p^*|$, where $p^*$ is the shortest prompt capable of inducing the model to output the secret $s$.
- **VMR (Verbatim Memorization Rate)**: Given a secret prefix $s_1$, whether the model can generate the corresponding suffix $s_2$.
- **AIR (Attribute Inference Rate)**: Whether the model can answer queries about specific attributes (e.g., "What genre does author X write?").

Higher scores $\rightarrow$ stronger memorization $\rightarrow$ poorer empirical privacy.

### 3. Regression Analysis of Hyperparameter Effects

In log-space, a multiple regression is performed on $(\log b, \log T, \log η)$ with empirical privacy scores as the target variable:

| Variable | Enron Coefficient | TOFU Coefficient | Interpretation |
|------|-----------|-----------|------|
| $\log b$ (batch size) | 0.13*** | 0.029*** | Smallest positive effect |
| $\log T$ (iterations) | 0.37*** | 0.048*** | Moderate positive effect |
| $\log η$ (learning rate) | 0.51*** | 0.068*** | Largest positive effect |

All coefficients are significantly positive ($p < 0.001$), implying that increasing any hyperparameter deteriorates empirical privacy.

### 4. Composite Hyperparameters

Defining **Compute** $C = b \cdot T$ and **Updates** $U = C \cdot η$ forms a hierarchy:

- When fixing $C$, increasing $b$ (decreasing $T$) $\rightarrow$ improves empirical privacy.
- When fixing $U$, decreasing $η$ (increasing $C$) $\rightarrow$ improves empirical privacy.

### 5. Hyperparameter Selection Heuristic

Three heuristic rules are proposed:

1. **Updates Heuristic**: Select the smallest $η$ under the same $U$.
2. **Compute Heuristic**: Select the largest $b$ under the same $(U, C)$.
3. **Individual Heuristic**: Eliminate configurations dominated by other configurations across all three dimensions of $(b, T, η)$.

Ultimately, **Algorithm 1** hierarchically applies these three rules in sequence, and selects the configuration with the worst utility (utility-privacy trade-off) among the remaining points.

## Key Experimental Results

### Experimental Setup

| Setting | Model | Dataset | Secret Type | No. of Configs |
|------|------|--------|----------|--------|
| 1 | GPT-2-S/L | Enron Email (33k) | Phone numbers/Emails, etc. | 23/15 |
| 2 | Llama-2-7b/13b | TOFU | Author-genre attributes | 60 |

$ε \in \{1, 2, 4, 8, 16\}$, $δ = n^{-1.1}$, utilizing LoRA fine-tuning + DP-Adam.

### Key Findings

- **Ubiquity of Variance**: For Llama-2-7b under TOFU-4 with $ε=8$, the AIR can vary from near 0 to over 0.8.
- **Variance Grows With**: Larger models, larger datasets, higher secret density, and larger $ε$.
- **No-Free-Lunch**: Existing best practices (large batch, high learning rate, more iterations) improve utility but deteriorate empirical privacy.

### Effectiveness of Heuristics

| Setting | Heuristic Accuracy | Random Baseline | Relative Privacy Risk Reduction |
|------|-------------|----------|----------------|
| GPT-2-S Enron | 70-90% | 50% | Significantly defeats the best-utility choice |
| Llama-2-7b TOFU-4 | 65-85% | 50% | Consistently outperforms the baseline across all $ε$ |

### Privacy Auditing Results

Utilizing SOTA black-box auditing methods to obtain $\hat{ε}$:

- $\hat{ε}$ indeed varies across configurations (supporting the first part of Hypothesis 1).
- $\hat{ε}$ exhibits extremely low correlation with empirical privacy (Spearman $ρ = -0.13$).
- $\hat{ε}$ is strongly and negatively correlated with utility ($ρ = -0.71$), suggesting that loss-based auditing methods are entangled with utility.

## Highlights & Insights

1. **Outstanding Conceptual Contribution**: First to systematically define and investigate "empirical privacy variance," highlighting the incompleteness of DP guarantees in practice.
2. **Profound No-Free-Lunch Conclusion**: Exposes a long-ignored issue in the DP-SGD community—hyperparameter tuning that optimizes utility silently sacrifices empirical privacy.
3. **Actionable Heuristics**: Allows for better hyperparameter selection without the need to actually measure empirical privacy.
4. **A Cautionary Tale for Standardization**: $ε$ cannot serve as a certification tool; regulatory bodies formulating standards solely based on $ε$ will face unforeseen risks.
5. **Limitations of Privacy Auditing**: Reveals the fundamental deficiency of loss-based auditing methods for measuring empirical privacy.
6. **Two Valuable Hypotheses**: The differential "true $ε$" hypothesis and the privacy profile divergence hypothesis outline promising directions for future research.

## Limitations & Future Work

1. **Oversimplified Regression Model**: Linear models may fail to capture the complex, non-linear relationships between hyperparameters and empirical privacy.
2. **Limited Scope of Datasets and Models**: Only validated on two datasets (Enron and TOFU) across GPT-2 and Llama-2.
3. **Sampler Mismatch in DP Guarantees**: Reports DP guarantees for Poisson subsampling while the actual training employs shuffled batches.
4. **Limitations of Empirical Privacy Metrics**: ACR, VMR, and AIR are task-specific metrics that may not encompass all privacy risks.
5. **Fixed Clipping Norm $c$**: Fails to deeply explore the influence of $c$ on empirical privacy.
6. **Unresolved Causality**: Both hypotheses remain preliminary explorations that lack rigorous causal analysis.
7. **Scalability**: Whether the findings generalize to other algorithms such as diffusion models and DP-FTRL remains unverified.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Empirical privacy variance is a brand-new concept that reveals fundamental issues neglected in DP practices.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multidimensional validation, regression analysis, and empirical selection evaluations are provided, though the diversity of datasets/models could be improved.
- Writing Quality: ⭐⭐⭐⭐⭐ — Highly structured, progressively advancing from phenomenon to analysis, proposed solution, and theoretical exploration.
- Value: ⭐⭐⭐⭐⭐ — Offers significant insights for both the DP community and privacy policy makers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Benchmarking Empirical Privacy Protection for Adaptations of Large Language Models](../../ICLR2026/llm_safety/benchmarking_empirical_privacy_protection_for_adaptations_of_large_language_mode.md)
- [\[ICML 2025\] Cape: Context-Aware Prompt Perturbation Mechanism with Differential Privacy](cape_context-aware_prompt_perturbation_mechanism_with_differential_privacy.md)
- [\[NeurIPS 2025\] On the Empirical Power of Goodness-of-Fit Tests in Watermark Detection](../../NeurIPS2025/llm_safety/on_the_empirical_power_of_goodness-of-fit_tests_in_watermark_detection.md)
- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](../../NeurIPS2025/llm_safety/a_reliable_cryptographic_framework_for_empirical_machine_unl.md)
- [\[ICML 2025\] The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text](the_canarys_echo_auditing_privacy_risks_of_llm-generated_synthetic_text.md)

</div>

<!-- RELATED:END -->
