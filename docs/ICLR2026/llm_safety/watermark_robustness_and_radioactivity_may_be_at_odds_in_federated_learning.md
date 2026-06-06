---
title: >-
  [Paper Note] Robustness and Radioactivity of Watermarks in Federated Learning May Be at Odds
description: >-
  [ICLR 2026][LLM Safety][Federated Learning] This work presents the first study on LLM watermark-based data provenance in federated learning (FL). It demonstrates that watermarks are radioactive (detectable) in FL…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "Federated Learning"
  - "LLM Watermarking"
  - "Data Provenance"
  - "Robust Aggregation"
  - "Radioactivity Detection"
date: 2026-05-08
content_hash: bad5d08decb24700
---

# Robustness and Radioactivity of Watermarks in Federated Learning May Be at Odds

**Conference**: ICLR 2026
**arXiv**: [2510.17033](https://arxiv.org/abs/2510.17033)  
**Area**: Video Understanding
**Keywords**: Federated Learning, LLM Watermarking, Data Provenance, Robust Aggregation, Radioactivity Detection

## TL;DR

This work presents the first study on LLM watermark-based data provenance in federated learning (FL). It demonstrates that watermarks are radioactive (detectable) in FL, yet a malicious server can suppress watermark signals by employing strong robust aggregation algorithms to filter watermarked updates, revealing a fundamental trilemma among radioactivity, robustness, and model utility.

## Background & Motivation

As LLM-generated synthetic data becomes increasingly prevalent in FL, data provenance has become a critical concern:

**Proliferation of Synthetic Data**: FL clients increasingly rely on LLM-generated synthetic datasets for local training.

**Provenance Requirements**: Without a provenance mechanism, malicious use of synthetic data for fine-tuning cannot be attributed or held accountable.

**Regulatory Demands**: Regulations such as the EU AI Act explicitly require transparency and traceability in AI systems.

**Core Problem**: Are LLM watermarks still effective in FL environments? Can a malicious server remove watermark signals while preserving model utility?

### Key Findings

t-SNE visualization reveals that model updates from watermarked clients appear as **outliers** in high-dimensional space, clearly separated from the update distribution of clean clients (Figure 1), thereby providing a handle for adversarial servers to mount attacks.

## Method

### Overall Architecture

The study considers two FL scenarios:

- **VanillaFL**: A benign server that naively averages all client updates.
- **ActiveFL**: A malicious server that employs a robust aggregator to filter watermarked updates.

### Key Definitions

**Radioactivity**: A dataset $D^w$ is $\alpha$-radioactive with respect to statistical test $T$ if $T$ can reject the null hypothesis $H_0$ (that the model was not trained on $D^w$) with a p-value below $\alpha$.

**FL Robustness**: Dataset $D^w_i$ is non-robust against adversary $\mathcal{A}$ if there exists $\mathcal{A}$ such that:
1. $\mathcal{A}(U_\Delta, \theta^{t_\mathcal{A}}) \approx_\mathcal{E} \mathcal{T}(C_\Delta, \theta^{t_\mathcal{A}})$ (utility preserved)
2. $\text{Detect}^{\mathcal{M}_{\theta^{t_\mathcal{A}+1}},\mathcal{A}}_s(D^w_i) \rightarrow \text{False}$ (watermark undetectable)

### Attack Mechanism

The malicious server replaces simple averaging with a **Byzantine-robust aggregator** (e.g., RandEigen), which guarantees an upper bound on aggregation bias:

$$\text{bias} = \|\text{Fil}(U_\Delta) - \mu_C\|_2 \leq \beta \cdot \|\Sigma_C\|_2^{1/2}$$

Strong robust aggregators guarantee $\beta = O(1)$, independent of the parameter dimensionality $d$, making them well-suited for the high-dimensional parameter spaces of LLMs.

### Evaluation Metrics

- **Escape Rate (ER)**: Fraction of watermarked clients whose updates survive aggregation without being filtered.
- **Over-Filtering Rate (OFR)**: Fraction of filtered updates that belong to non-watermarked clients.

## Key Experimental Results

### Radioactivity Detection (VanillaFL, ε=6.6%)

| Dataset | Watermark | Model | p-value (pre-FT) | p-value (post-FT) |
|---------|-----------|-------|------------------|-------------------|
| C4 | KGW+ | 160M | 0.397 | $1.27\times10^{-3}$ |
| C4 | KGW+ | 410M | 0.877 | $2.41\times10^{-8}$ |
| Alpaca | KGW+ | 410M | 0.302 | $4.96\times10^{-24}$ |
| C4 | KTH+ | All | ~0.5 | ~0.5 |

### Robustness Detection (ActiveFL vs. VanillaFL, ε=6.6%)

| Dataset | Model | VanillaFL p-value | ActiveFL p-value |
|---------|-------|-------------------|------------------|
| C4 | 160M | $1.27\times10^{-3}$ | 0.550 |
| C4 | 410M | $2.41\times10^{-8}$ | 0.613 |
| Alpaca | 160M | $1.59\times10^{-11}$ | 0.231 |
| Alpaca | 410M | $4.96\times10^{-24}$ | 0.282 |

### Key Findings

1. **KGW+ watermarks exhibit strong radioactivity in FL**: Even with only 6.6% watermarked data, p-values can reach as low as $10^{-24}$.
2. **KTH+ watermarks are not radioactive in FL**: Their detectors cannot accumulate statistical signals across prompts.
3. **RandEigen effectively removes all watermarks**: Under ActiveFL, p-values for all radioactive watermarks recover to ~0.5.
4. **Larger δ increases radioactivity but reduces robustness**: ER drops from 60.2% (δ=0) to 0.7% (δ=5).
5. **Trilemma**: Increasing ε simultaneously improves radioactivity and robustness but degrades model utility.

## Highlights & Insights

1. **First federated data provenance study**: Extends watermark detection from centralized to distributed FL settings.
2. **Adversarial insight**: The distributional shift introduced by watermarking causes updates to appear as outliers, which are precisely the targets of aggregators originally designed to defend against Byzantine attacks.
3. **Revelation of a fundamental trilemma**: Radioactivity (detectability), robustness (adversarial resistance), and utility (model quality) cannot be simultaneously satisfied.
4. **Practical threat model**: A malicious server only needs to replace the aggregation function to suppress watermarks, without any knowledge of the watermarking scheme or its parameters.

## Limitations & Future Work

1. Evaluation is limited to the Pythia model family (70M–410M); validation on larger models is absent.
2. Only two watermarking schemes (KGW+, KTH+) are considered, limiting coverage.
3. The malicious server is assumed to have no knowledge of the watermark key or scheme; in practice, such information may leak.
4. No effective defense against robust aggregation attacks is proposed.
5. The watermarked client ratio ε is kept small (at most 30%); more extreme scenarios remain unexplored.

## Rating ⭐⭐⭐⭐

The problem formulation is novel and the experimental design is rigorous, uncovering a fundamental tension in FL watermarking. Although no solution is proposed, the work charts a clear direction for future research. The connection to video understanding is tenuous; the paper is more squarely situated at the intersection of security and federated learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning](sabre-fl_selective_and_accurate_backdoor_rejection_for_federated_prompt_learning.md)
- [\[ICML 2026\] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning](../../ICML2026/llm_safety/decoupled_training_with_local_reinforcement_fine-tuning_in_federated_learning.md)
- [\[AAAI 2026\] FedP²EFT: Federated Learning to Personalize PEFT for Multilingual LLMs](../../AAAI2026/llm_safety/fedp2eft_federated_learning_to_personalize_peft_for_multilingual_llms.md)
- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](../../NeurIPS2025/llm_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](../../ACL2026/llm_safety/topic-based_watermarks_for_large_language_models.md)

</div>

<!-- RELATED:END -->
