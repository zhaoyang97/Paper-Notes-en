---
title: >-
  [Paper Note] Prior-based Noisy Text Data Filtering: Fast and Strong Alternative for Perplexity
description: >-
  [ICLR 2026][AI Safety][Data Filtering] This paper proposes a text data filtering method based on token priors (token frequency statistics)…
tags:
  - "ICLR 2026"
  - "AI Safety"
  - "Data Filtering"
  - "Pretraining"
  - "Perplexity"
  - "Token Frequency Prior"
  - "Data Quality"
date: 2026-05-08
content_hash: 4134062ce81754d4
---

# Prior-based Noisy Text Data Filtering: Fast and Strong Alternative for Perplexity

**Conference**: ICLR 2026
**arXiv**: [2509.18577](https://arxiv.org/abs/2509.18577)  
**Code**: [GitHub](https://github.com/ybseo-ac/prior_filter)  
**Area**: Multilingual Translation
**Keywords**: Data Filtering, Pretraining, Perplexity, Token Frequency Prior, Data Quality

## TL;DR

This paper proposes a text data filtering method based on token priors (token frequency statistics), using the mean and standard deviation of in-document token priors as a proxy for perplexity (PPL). The method achieves the highest average performance across 20 downstream benchmarks while being over 1000× faster than PPL-based filtering.

## Background & Motivation

### Importance of Pretraining Data Quality

Large language models rely on massive web corpora for pretraining, yet web data is extremely noisy. Two major challenges arise: (1) the sheer scale of data demands efficient filtering to conserve computational resources; and (2) noisy data degrades model performance.

### Limitations of Prior Work

PPL-based filtering is the current state-of-the-art approach but has two inherent drawbacks:

- **Computational cost**: Requires training a reference model (137M parameters) and running PPL inference over the entire corpus. For a 6B-token corpus, this takes 216 GPU hours.
- **Reliability issues**: Model PPL estimates are unreliable for out-of-distribution (OOD) samples such as noisy data; small models are especially prone to assigning low PPL (i.e., falsely high quality) to repetitive or patterned noise.

### Linguistic Inspiration

The paper draws inspiration from the 8th-century linguist Al-Kindi's cryptanalysis method: **analyzing word frequencies can reveal linguistic structure**.

Two key linguistic observations:
1. **Token frequency is a one-dimensional representation of token role**: high-frequency tokens are function words (e.g., "the", "is"), while low-frequency tokens are content words (e.g., "president", "algorithm").
2. **Well-formed sentences exhibit stable lexical density**: the ratio of function words to content words remains relatively stable across different documents.

## Method

### Overall Architecture

The core idea is that PPL can be decomposed via Bayes' rule into a likelihood term and a prior term. The paper approximates the prior term using token frequency statistics, bypassing the likelihood term that requires model inference.

### Token Prior Estimation

Given a corpus $D$ and vocabulary $V$, the prior probability of token $x$ is estimated from its frequency:

$$p_{\text{prior}}(x) = \frac{f_D(x)}{\sum_{x' \in V} f_D(x')}$$

where $f_D(x)$ is the count of token $x$ in the corpus.

### Key Designs: Dual-Metric Filtering

Two statistics are defined for each document $\texttt{d}$:

**Prior mean** $\mu_{\texttt{d}}$:

$$\mu_{\texttt{d}} = \mathbb{E}_{x_i \in \texttt{d}} [\log p_{\text{prior}}(x_i)]$$

This reflects the token composition of the document — the balance between high- and low-prior tokens.

**Prior standard deviation** $\sigma_{\texttt{d}}$:

$$\sigma_{\texttt{d}} = \text{std}_{x_i \in \texttt{d}} [p_{\text{prior}}(x_i)]$$

This reflects the distributional structure of token priors within the document — its diversity or uniformity.

### Outlier Detection

Using corpus-level medians $M_\mu = \text{median}(\mu_{\texttt{d}})$ and $M_\sigma = \text{median}(\sigma_{\texttt{d}})$ as reference centers, deviation distances are computed to quantify anomaly:

$$\delta_\mu(\texttt{d}) = |\mu_{\texttt{d}} - M_\mu|, \quad \delta_\sigma(\texttt{d}) = |\sigma_{\texttt{d}} - M_\sigma|$$

Samples with the largest $\delta$ values are discarded, with the constraint $|F_\mu| = |F_\sigma|$, until the remaining subset reaches the target size.

### Theoretical Connection to PPL

$$\log \text{PPL}(\texttt{d}) \propto \underbrace{\sum_i \log p_\theta(x_{<i}|x_i)}_{\pi_{\text{likelihood}}} + \underbrace{\sum_i \log p_\theta(x_i)}_{\pi_{\text{prior}}}$$

- $\mu_{\texttt{d}}$ is exactly equivalent to the $\pi_{\text{prior}}$ term.
- $\sigma_{\texttt{d}}$ approximately captures the regularity of inter-token relationships reflected by $\pi_{\text{likelihood}}$.
- Together, they serve as a reasonable proxy for PPL.

### Design Motivation: Unique Advantages of the Prior

The prior is not merely an approximation of PPL — it is superior in certain respects:

- Small models struggle to learn the likelihood accurately, whereas token frequency statistics are simple and stable.
- Model likelihood estimates for OOD noisy data are unreliable, while the prior is unaffected.
- PPL tends to misclassify repetitive or patterned noise as high-quality text, whereas prior-based filtering does not.

## Key Experimental Results

### Main Results: Downstream Task Performance on Dolma Corpus

GPT-2 architecture, 1.5B and 137M models, trained for 40K steps (~6B tokens), evaluated on 20 downstream benchmarks.

| Method | Type | Time | Avg. | World Knowledge | Commonsense Reasoning | Language Understanding | Symbolic Reasoning | Reading Comprehension |
|--------|------|------|------|-----------------|----------------------|----------------------|-------------------|----------------------|
| No-filter | Rule | - | 5.78 | 5.52 | 0.44 | 6.14 | 13.22 | 3.59 |
| FastText | Classifier | 3.6h | 7.09 | 6.71 | 6.11 | 6.89 | 11.93 | 3.82 |
| DSIR | n-gram | 4h | 7.56 | 7.03 | 6.84 | 7.31 | 12.67 | 3.97 |
| PPL-based | Model | **216 GPU h** | 8.22 | 9.98 | 11.91 | 7.34 | 7.91 | 3.96 |
| **Prior-based** | Statistical | **0.25h** | **9.20** | 9.53 | 11.27 | **10.31** | 11.13 | 3.79 |

**Key finding**: Prior-based filtering achieves higher average performance than PPL (9.20 vs. 8.22) at only 0.1% of the computational cost.

### Symbolic Language Experiment: Pile-github

| Method | Time | Avg. | CS | Dyck | Arithmetic | Elementary Math | GSM | SVAMP |
|--------|------|------|-----|------|------------|-----------------|-----|-------|
| No-filter | - | 9.51 | 35.75 | 12.30 | 5.71 | 1.15 | 0.15 | 2.00 |
| PPL-based | 224 GPU h | 11.21 | 37.42 | 20.60 | 7.14 | 2.09 | 0.00 | 0.00 |
| **Prior-based** | 0.26h | **12.03** | 38.86 | 21.30 | **9.04** | 1.17 | 0.15 | 1.67 |

Prior-based filtering also outperforms PPL filtering on symbolic languages such as code and mathematics.

### Ablation Study

**Large-scale consistency** (Qwen2.5-3B and 1.5B models trained on 12B tokens): Prior-based filtering consistently outperforms PPL filtering.

**Subsampling efficiency**: Computing token priors on only 1% of the corpus yields nearly identical filtering results, reducing runtime from ~30 minutes to ~70 seconds.

**PPL overlap analysis**: At filtering ratio $e=0.10$, the overlap between $F_\mu$ and $F_{\text{ppl}}$ approaches 50%, confirming that prior-based filtering does indeed approximate PPL-based filtering.

### Key Findings

1. **PPL performs worst on symbolic reasoning**: PPL tends to filter out small but meaningful code/math snippets.
2. **$\mu_{\texttt{d}}$ outliers** are predominantly documents with extremely high- or low-prior tokens (e.g., newline accumulations, non-English text).
3. **$\sigma_{\texttt{d}}$ outliers** are predominantly unstructured noun lists — containing content words but lacking syntactic structure.
4. **Multilingual adaptivity**: When Chinese data constitutes <1% of an English corpus, it is automatically filtered as noise; when it exceeds 20%, it is recognized as a learnable language.

## Highlights & Insights

1. **Victory of minimalism**: Token frequency statistics alone surpass PPL methods that require model training and inference.
2. **Solid linguistic foundations**: Every design choice is grounded in linguistics, from Al-Kindi's cryptanalysis to lexical density theory.
3. **Overwhelming speed advantage**: 0.25 hours vs. 216 GPU hours (~1000× speedup), with the gap widening as web corpora continue to grow.
4. **Adaptive multilingual handling**: No need to manually specify reference datasets; filtering/retention decisions are made automatically based on language proportion.
5. **Complementary dual metrics**: $\mu_{\texttt{d}}$ captures token composition while $\sigma_{\texttt{d}}$ captures distributional structure, covering distinct types of noise.

## Limitations & Future Work

1. The method relies on linguistic properties and is not applicable to non-textual modalities (images, audio, etc.).
2. As an approximation of PPL, prior-based filtering is less effective than PPL at identifying noise that is superficially well-formed but semantically meaningless.
3. Experiments primarily use the GPT-2 architecture; validation on more modern architectures (e.g., Llama) is limited.
4. For training objectives heavily skewed toward a specific data type (e.g., pure mathematics), manual tuning may be required.

## Related Work & Insights

- **Ankner et al. 2024 (PPL filtering)**: The primary baseline this paper targets; the paper demonstrates that prior-based filtering is both better and faster than PPL.
- **DSIR (Xie et al. 2023)**: Requires manually specified reference datasets, whereas prior-based filtering operates automatically.
- **FastText classifier**: Requires human-annotated reference data; prior-based filtering is entirely unsupervised.
- **Insight**: Data filtering does not require complex model inference — returning to statistical fundamentals may be the superior approach.

## Rating

- **Novelty**: ⭐⭐⭐⭐☆ — Extremely concise and elegant idea grounded in linguistic principles.
- **Theoretical Depth**: ⭐⭐⭐⭐ — Thorough Bayesian decomposition analysis of the PPL approximation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 20 benchmarks + symbolic language + large-scale validation + multilingual analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — 1000× speedup with superior performance; directly applicable to industrial data pipelines.
- **Value**: ⭐⭐⭐⭐☆ — A simple yet effective methodological contribution with significant practical value for pretraining data curation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Alternative Fairness and Accuracy Optimization in Criminal Justice](../../AAAI2026/ai_safety/alternative_fairness_and_accuracy_optimization_in_criminal_j.md)
- [\[AAAI 2026\] An Information Theoretic Evaluation Metric for Strong Unlearning](../../AAAI2026/ai_safety/an_information_theoretic_evaluation_metric_for_strong_unlearning.md)
- [\[AAAI 2026\] Enhancing DPSGD via Per-Sample Momentum and Low-Pass Filtering](../../AAAI2026/ai_safety/enhancing_dpsgd_via_per-sample_momentum_and_low-pass_filtering.md)
- [\[ICML 2026\] SORA: Free Second-Order Attacks in Fast Adversarial Training](../../ICML2026/ai_safety/sora_free_second-order_attacks_in_fast_adversarial_training.md)
- [\[ICCV 2025\] Backdooring Self-Supervised Contrastive Learning by Noisy Alignment](../../ICCV2025/ai_safety/backdooring_self-supervised_contrastive_learning_by_noisy_alignment.md)

</div>

<!-- RELATED:END -->
