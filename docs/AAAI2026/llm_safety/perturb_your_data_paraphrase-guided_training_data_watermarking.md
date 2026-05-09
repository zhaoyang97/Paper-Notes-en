---
title: >-
  [Paper Note] Perturb Your Data: Paraphrase-Guided Training Data Watermarking
description: >-
  [AAAI2026][LLM Safety][Training data watermarking] This paper proposes SPECTRA — a paraphrase-sampling-based training data watermarking method. It generates paraphrases via an LLM and uses Min-K%++ scoring to select paraphrases with scores close to the original text as watermarks. Even when watermarked data constitutes as little as 0.001% of the training corpus, the p-value gap between members and non-members consistently exceeds 9 orders of magnitude.
tags:
  - AAAI2026
  - LLM Safety
  - Training data watermarking
  - membership inference attack
  - data copyright protection
  - LLM
  - paraphrasing
  - Min-K%++
date: 2026-05-08
content_hash: cc784b795504cf52
---

# Perturb Your Data: Paraphrase-Guided Training Data Watermarking

**Conference**: AAAI2026
**arXiv**: [2512.17075](https://arxiv.org/abs/2512.17075)
**Authors**: Pranav Shetty, Mirazul Haque, Petr Babkin, Zhiqiang Ma, Xiaomo Liu, Manuela Veloso (JPMorgan AI Research)
**Code**: Not available
**Area**: AI Safety
**Keywords**: Training data watermarking, membership inference attack, data copyright protection, LLM, paraphrasing, Min-K%++

## TL;DR

This paper proposes SPECTRA — a paraphrase-sampling-based training data watermarking method. It generates paraphrases via an LLM and uses Min-K%++ scoring to select paraphrases with scores close to the original text as watermarks. Even when watermarked data constitutes as little as 0.001% of the training corpus, the p-value gap between members and non-members consistently exceeds 9 orders of magnitude.

## Background & Motivation

### The Data Copyright Crisis

Modern LLMs rely on large-scale web-crawled text for pretraining, which frequently includes unauthorized copyrighted content. Recent lawsuits (e.g., The New York Times v. OpenAI) have highlighted the legal questions surrounding the unauthorized use of proprietary content for model training. As more content is consumed through LLMs such as ChatGPT, current data collection practices risk creating an "extractive dead end" — undermining creators' incentives to produce new content if fair compensation is not ensured.

### Limitations of Prior Work

Membership Inference Attack (MIA) is the dominant approach for detecting training data, based on the principle that training alters a model's output probabilities for training samples. However, MIA is highly sensitive to small distributional shifts between member and non-member data; when both are drawn from homogeneous distributions (after removing temporal artifacts), all MIA methods degrade to random chance. The STAMP method addresses this by rewriting text multiple times using the KGW watermarking scheme, but it requires storing large numbers of private paraphrase versions, access to the LLM's decoding layer (which is GPU-intensive), and achieves a member vs. non-member p-value gap of at most 3 orders of magnitude — insufficient for contexts with significant legal consequences.

### Core Insight

Existing MIA scores aim to measure changes in the loss surface before and after training, yet access to a model's pre-training state is rarely available in practice. SPECTRA's key insight is to use a scoring model known not to have been trained on the target data as a proxy for the pre-training state, thereby reliably detecting training-induced changes. A carefully designed paraphrase sampling strategy further ensures that the watermark itself introduces no false-positive signal.

## Core Problem

How to design a watermarking process $W$ and statistical test $T$ such that: (1) if the target model $M$ was indeed trained on watermarked data $D'$, $T$ can detect this with high confidence; and (2) if $M$ was not trained on $D'$, $T$ produces no false positives. The entire process requires only grey-box access (token log probabilities), and the watermarked data may constitute as little as 0.001% of the training corpus.

## Method

### Overall Architecture

SPECTRA operates in two phases: a **watermarking phase** (before release) and a **verification phase** (at detection time).

**Watermarking phase**: For each document in dataset $D$, an LLM generates multiple paraphrases. A scoring model computes Min-K%++ scores, and a carefully designed sampling strategy selects the paraphrase whose score is closest to that of the original document as the watermarked version, forming $D' = W(D)$. The content creator retains the original $D$ and publishes only $D'$.

**Verification phase**: Min-K%++ score ratios between original and watermarked documents are computed on both the scoring model $M_S$ and the target model $M_T$; a paired t-test then determines whether $M_T$ was trained on $D'$.

### Key Design 1: Min-K%++ Scoring

Given an autoregressive model $M$ and token sequence $x = (x_1, \ldots, x_n)$, the normalized token log probability is defined as:

$$z(x_t; M) = \frac{\log P(x_t \mid x_{<t}; M) - \mu_{x_{<t}}}{\sigma_{x_{<t}}}$$

where $\mu_{x_{<t}} = \mathbb{E}_{z \sim P(\cdot|x_{<t}; M)}[\log P(z|x_{<t}; M)]$ is the expected log probability and $\sigma_{x_{<t}}$ is the corresponding standard deviation. The Min-K%++ score averages the K% lowest values (highest surprisal):

$$f_{\text{Min-K\%++}}(x; M) = \frac{1}{|\text{min-K}(x)|} \sum_{x_t \in \text{min-K}(x)} z(x_t; M)$$

Theoretically, Min-K%++ corresponds to the negative Hessian trace of the loss landscape. Training directly reduces curvature at training samples via maximum likelihood, causing Min-K%++ scores to increase.

### Key Design 2: Paraphrase Sampling Strategy

The core objective is to select paraphrases whose Min-K%++ scores are close to that of the original document, avoiding systematic shifts that would cause false positives. For each document $i$, the ratio $r_{ij} = s_{ij} / s_i^{(0)}$ of paraphrase score to original score is computed, and paraphrases are grouped into upper and lower sides:

- Weight definition: $w_{ij} = \exp(-\alpha |r_{ij} - 1|)$, with $\alpha = 100$ controlling distribution sharpness.
- Global balance: Documents for which all paraphrases score above the original (set $\mathcal{B}$) and below the original (set $\mathcal{A}$) are counted; the upper side is selected with probability $\pi_+ = |\mathcal{A}| / (|\mathcal{A}| + |\mathcal{B}|)$ to achieve global balance.
- Categorical sampling is performed from the selected side using normalized weights.

This design ensures that the score distribution of the watermarked dataset remains consistent with that of the original data, so that a detectable signal only emerges after the model has actually been trained on the watermarked data.

### Key Design 3: Statistical Test

Define the score ratio: $r(x, x'; M) = f_{\text{Min-K\%++}}(x'; M) / f_{\text{Min-K\%++}}(x; M)$

Null hypothesis $H_0$: $\mathbb{E}[r(x, x'; M_T)] = \mathbb{E}[r(x, x'; M_S)]$ ($M_T$ was not trained on $D'$)

Alternative hypothesis $H_1$: $\mathbb{E}[r(x, x'; M_T)] < \mathbb{E}[r(x, x'; M_S)]$ ($M_T$ was trained on $D'$)

A one-sided paired t-test yields a p-value; a sufficiently low p-value rejects $H_0$, indicating that $M_T$ was trained on the watermarked data.

## Key Experimental Results

### Experimental Setup

- **Paraphraser**: Llama 3.1-405b, generating 10 paraphrases per document
- **Scoring model**: Pythia 2.8b-deduped (Pile dataset) / OLMo-1b (PeS2o dataset)
- **Target model**: Pythia 410m with 5 billion tokens of continued pretraining
- **Watermarked data**: 500 samples per dataset (≤512 tokens each), constituting <0.001% of the training corpus

### Main Results: p-value Comparison

| Method | PubMed M | PubMed NM | Wiki M | Wiki NM | HN M | HN NM | PeS2o M | PeS2o NM |
|--------|----------|-----------|--------|---------|------|-------|---------|----------|
| LLM-DI | 0.06 | 0.48 | 0.02 | 0.44 | 0.49 | 0.35 | 0.02 | 0.17 |
| STAMP | 0.01 | 0.48 | 0.17 | 0.03 | 7E-4 | 0.15 | 0.15 | 0.46 |
| Maximum | 0.03 | 1.00 | 1.00 | 1.00 | 3E-6 | 1.00 | 0.95 | 1.00 |
| Random | 1E-7 | 8E-4 | 5E-9 | 2E-5 | 4E-27 | 0.10 | 1E-3 | 0.11 |
| **SPECTRA** | **1E-17** | 0.02 | **4E-19** | 0.02 | **3E-60** | 0.59 | **2E-12** | 3E-3 |

Under the strict threshold $p < 10^{-4}$, SPECTRA is the only method that correctly detects membership across all four datasets without producing false positives. The p-value gap between members and non-members consistently exceeds $10^9$ orders of magnitude.

### MIA Baseline Performance (500M vs. 5B Tokens)

| Method | Wiki 500M | HN 500M | PubMed 500M | Wiki 5B | HN 5B | PubMed 5B |
|--------|-----------|---------|-------------|---------|-------|-----------|
| Loss | 0.71 | 0.73 | 0.63 | 0.55 | 0.54 | 0.52 |
| Min-K% | 0.76 | 0.79 | 0.65 | 0.56 | 0.55 | 0.52 |
| Min-K%++ | 0.85 | 0.84 | 0.72 | 0.55 | 0.55 | 0.51 |

Min-K%++ performs best at 500M tokens (AUC 0.72–0.85), but degrades to random chance at 5B tokens (AUC ≈ 0.5) — precisely the problem SPECTRA addresses.

### Paraphrase Quality

| Dataset | PubMed | Wiki | HN | PeS2o |
|---------|--------|------|----|-------|
| P-SP | 0.88 | 0.93 | 0.76 | 0.93 |

Human evaluation (54 documents, 4 annotators) shows average scores exceeding 4 out of 5 across three dimensions — meaning preservation, structural preservation, and authorial tone preservation — with structural preservation slightly lower for conversational text such as Hackernews posts.

### Scoring Model Robustness

| Scoring Model | Spearman $\rho$ | Kendall $\tau$ |
|--------------|----------------|----------------|
| OLMo-7b | 0.826 | 0.639 |
| Pythia-2.8b | 0.824 | 0.635 |
| Pythia-160m | 0.699 | 0.514 |
| Pythia-6.9b | 0.818 | 0.631 |

Models with ≥2.8B parameters achieve ranking correlations of $\rho > 0.8$, indicating that SPECTRA is robust to the choice of scoring model.

## Highlights & Insights

- **Exceptionally strong statistical signal**: The p-value gap between members and non-members consistently exceeds 9 orders of magnitude, far surpassing STAMP (3 orders of magnitude) and all other baselines.
- **No decoding-layer access required**: Only grey-box access (token log probabilities) is needed, unlike STAMP which requires modifying the LLM decoding process.
- **No non-member dataset required**: Detection is based on comparing score ratios between the scoring model and the target model, without the need for in-domain held-out data.
- **Elegant sampling strategy**: Global side-balancing combined with exponentially decaying weights preserves the score distribution while maintaining post-training detectability.
- **Cross-architecture effectiveness**: The PeS2o experiment demonstrates effectiveness even when the scoring model (OLMo-1b) and the target model (Pythia 410m) have different architectures.

## Limitations & Future Work

- **Validated only on continued pretraining**: Due to computational constraints, the method has not been validated in from-scratch training scenarios; larger-scale validation is needed for practical deployment.
- **Requires grey-box access**: Closed-source commercial models typically do not expose token log probabilities, necessitating a third-party arbitration mechanism.
- **Applicable only to unpublished data**: Watermarks must be embedded before release; already-published content cannot be retroactively protected.
- **Reduced effectiveness on structured/conversational text**: The P-SP score for Hackernews is only 0.76, reflecting lower paraphraser fidelity on non-standard text styles.
- **Dependence on a powerful paraphraser**: Paraphrase generation relies on Llama 3.1-405b, which is costly; the effectiveness of smaller models remains insufficiently explored.
- **Practical implications of the 500-sample requirement**: At least 100–150 samples are needed to reach statistical significance, which may be prohibitive for small-scale content creators.

## Related Work & Insights

- **vs. STAMP**: STAMP requires modifying the LLM decoding layer via the KGW watermarking scheme and storing large numbers of private paraphrase versions, achieving a p-value gap of only 3 orders of magnitude. SPECTRA requires no decoding-layer access and achieves a gap exceeding 9 orders of magnitude.
- **vs. MIA (Min-K%++, etc.)**: Conventional MIA degrades to random chance under large-scale training (5B tokens) and depends on non-member data. SPECTRA creates a reliable detection signal through active watermarking.
- **vs. LLM-DI**: Dataset Inference fails to reliably detect membership under strict thresholds; SPECTRA achieves statistical significance across all datasets.
- **vs. Maximum sampling**: Greedily selecting the paraphrase with the highest Min-K%++ score introduces excessive distributional shift during pretraining, making post-training signals indistinguishable. SPECTRA's balanced sampling strategy avoids this issue.
- **vs. backdoor watermarking** (e.g., Winter Soldier): Backdoor methods insert special tokens that degrade text readability; SPECTRA generates natural paraphrases whose semantic quality is verifiable.

The **proxy model** paradigm has broad applicability: using a related or similar pretrained model as a baseline for differential detection — when the model's pre-training state is unavailable — generalizes naturally to other model auditing settings. The **score distribution preservation** design philosophy is equally instructive: the key to watermarking is not to maximize the current signal but to keep the pre-training distribution unchanged so that only training induces a detectable shift — analogous to the notion of semantic security in cryptography. This method could further integrate with **data attribution** and **model provenance** research to form a more complete AI data governance toolchain. The minimum requirement of 100–150 samples suggests the feasibility of a **collaborative watermarking framework** tailored for small-scale creator coalitions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Cleverly combines paraphrase sampling with Min-K%++ scoring for data watermarking; the side-balance design in the sampling strategy is original, though the core components (Min-K%++ and paraphrasing) are existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four datasets, multiple baselines, ablation studies (sample size, scoring model selection), and human evaluation are fairly comprehensive, though validation is limited to continued pretraining.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear and the method is described intuitively, though some notation definitions are scattered across the paper, requiring frequent cross-referencing.
- **Value**: ⭐⭐⭐⭐ — Addresses a practical pain point in LLM training data copyright protection; the 9-order-of-magnitude p-value gap provides compelling statistical evidence for legal contexts.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](../../NeurIPS2025/llm_safety/virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)
- [\[AAAI 2026\] Principles2Plan: LLM-Guided System for Operationalising Ethical Principles into Plans](principles2plan_llm-guided_system_for_operationalising_ethical_principles_into_p.md)
- [\[ACL 2026\] Synthia: Scalable Grounded Persona Generation from Social Media Data](../../ACL2026/llm_safety/synthia_scalable_grounded_persona_generation_from_social_media_data.md)
- [\[ICLR 2026\] Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers](../../ICLR2026/llm_safety/perturbation-induced_linearization_constructing_unlearnable_data_with_solely_lin.md)
- [\[ICLR 2026\] Redirection for Erasing Memory (REM): Towards a Universal Unlearning Method for Corrupted Data](../../ICLR2026/llm_safety/redirection_for_erasing_memory_rem_towards_a_universal_unlearning_method_for_cor.md)

<!-- RELATED:END -->
