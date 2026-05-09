---
title: >-
  [Paper Note] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy
description: >-
  [NeurIPS 2025][LLM Safety][differential privacy] This paper proposes InvisibleInk, a framework that reduces the computational cost of differentially private long-text generation by more than 8× through two innovations—differential clipping (DClip) for isolating sensitive information and Top-$k^+$ truncated sampling—achieving, for the first time, high-quality private text generation with only 4–8× overhead over non-private generation.
tags:
  - NeurIPS 2025
  - LLM Safety
  - differential privacy
  - text generation
  - exponential mechanism
  - LLM decoding
  - privacy-preserving inference
date: 2026-05-08
content_hash: c3aeddb730bd64ef
---

# InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy

**Conference**: NeurIPS 2025
**arXiv**: [2507.02974](https://arxiv.org/abs/2507.02974)
**Code**: [cerai-iitm/invisibleink](https://github.com/cerai-iitm/invisibleink)
**Area**: AI Safety
**Keywords**: differential privacy, text generation, exponential mechanism, LLM decoding, privacy-preserving inference

## TL;DR
This paper proposes InvisibleInk, a framework that reduces the computational cost of differentially private long-text generation by more than 8× through two innovations—differential clipping (DClip) for isolating sensitive information and Top-$k^+$ truncated sampling—achieving, for the first time, high-quality private text generation with only 4–8× overhead over non-private generation.

## Background & Motivation

**Background**: LLMs demonstrate strong long-text generation capabilities in paradigms such as RAG and inference-time scaling, but generation inevitably involves privacy-sensitive data (e.g., medical records, legal documents, user conversations), creating risks of privacy leakage through model outputs.

**Limitations of Prior Work**: Existing differentially private (DP) text generation methods suffer from severe practical limitations. The SOTA method of Amin et al. [2024] interprets LLM next-token sampling as an exponential mechanism but requires more than 100× the computational overhead of non-private generation to produce non-degenerate text, and is completely non-functional at small batch sizes.

**Key Challenge**: DP text generation faces a three-way tradeoff among privacy, utility, and computation. Conventional clipping applies to the entire logit vector, yet over 95% of the information in logits constitutes the language model's public prior (grammar, commonsense, etc.); clipping this non-sensitive information wastes privacy budget. Simultaneously, DP noise causes full-vocabulary sampling to yield low-quality, degenerate text.

**Goal**: (i) Reduce the per-token privacy cost to decrease computational requirements; (ii) improve text quality through truncated decoding under DP constraints.

**Key Insight**: The key observation is that the distribution range of the difference between private logits $\phi_i$ and public logits $\phi_{\text{pub}}$ (logits computed without sensitive references) is approximately 10× smaller than that of the raw logits. This implies that sensitive information contributes only a small fraction of the logit values and can be handled with a much smaller clipping threshold.

**Core Idea**: Clip only the increment of the logits relative to the public model (i.e., the sensitive component), and combine this with truncated sampling over a top-$k$ superset derived from the public logits, substantially reducing privacy costs and improving text quality.

## Method

### Overall Architecture
Given a query $\boldsymbol{q}$ and $B$ sensitive reference texts $\boldsymbol{R} = \{\boldsymbol{r}_1, \ldots, \boldsymbol{r}_B\}$, InvisibleInk generates private text token by token. At each step, $B$ private logits $\phi_i = \phi(\cdot|\boldsymbol{q}, \boldsymbol{r}_i, \boldsymbol{x}_{<t})$ and one public logit $\phi_{\text{pub}} = \phi(\cdot|\boldsymbol{q}, \boldsymbol{x}_{<t})$ are obtained, aggregated via DClip, and the next token is sampled using the exponential mechanism over a Top-$k^+$ vocabulary subset.

### Key Designs

1. **DClip (Differential Clipping)**:

    - **Function**: Clips only the sensitive increment of the logits rather than the full logit vector.
    - **Mechanism**: $\text{DClip}_C(\phi_i, \phi_{\text{pub}}) := \phi_{\text{pub}} + \text{clip}_C(\phi_i - \phi_{\text{pub}})$. The difference between each private logit and the public logit is clipped to $[-C, C]$, yielding an aggregated sensitivity of $C/B$ under replace-by-null adjacency.
    - **Design Motivation**: Empirical observation shows that the value range of $\phi_i - \phi_{\text{pub}}$ is approximately 10× smaller than that of $\phi_i$, with over 95% of vocabulary tokens unaffected at $C \approx 1$. Conventional methods require $C \approx 10$ to avoid severe distortion, whereas DClip operates at $C \approx 1$, directly yielding an 8× computational saving (since temperature $\tau \propto C/B$, a smaller $C$ requires a smaller $B$).

2. **Top-$k^+$ Sampling**:

    - **Function**: Performs truncated sampling over a vocabulary set determined by the public logits at no additional privacy cost.
    - **Mechanism**: $V_k^+ = \{y \in V: \phi_{\text{pub}}(y) \geq \ell - 2C/B\}$, where $\ell$ is the top-$k$ threshold of the public logit. This set is a superset of all private top-$k$ vocabularies, yet contains only approximately 10 more tokens than $V_k(\phi_{\text{pub}})$.
    - **Design Motivation**: Sampling from the full vocabulary ($|V| \sim 10^5$) produces degenerate text; directly using the public top-$k$ misses domain-specific tokens that are frequent in sensitive data but rare in the public model. Top-$k^+$ balances both concerns—truncating low-probability noisy tokens while retaining useful signals from private data.

3. **Non-Adaptive Privacy Accounting**:

    - **Function**: Provides a closed-form privacy budget formula.
    - **Mechanism**: Under adaptive composition over $T$ tokens, the total privacy budget is $\rho_{\text{seq}} = TC^2/(2B^2\tau^2)$ (zCDP). Given $\rho_{\text{seq}}$ and $T$, the clipping threshold can be derived directly as $C = B\tau\sqrt{2\rho_{\text{seq}}/T}$.
    - **Design Motivation**: Prior methods with adaptive DP guarantees require grid search to calibrate hyperparameters, which itself incurs additional privacy costs that previous work overlooked. InvisibleInk provides a user-friendly, non-adaptive guarantee that works out of the box.

### Graceful Degradation Property
When the privacy or computational budget is extremely small, $C$ approaches 0 and $\bar{\phi}_i \approx \phi_{\text{pub}}$, causing the system to smoothly degrade to public model generation. Prior methods either fail entirely (Amin et al.) or can only produce very short texts (AdaPMixED) when the budget is insufficient.

## Key Experimental Results

### Main Results: Privacy–Utility–Computation Tradeoff (MIMIC Dataset, TinyLLaMA 1.1B)

| Method | Batch Size $B$ | MAUVE(%) (ε=10) | MAUVE(%) (ε=1) | Notes |
|--------|---------------|----------------|----------------|-------|
| InvisibleInk (k=100) | 3 | 68.4 | 68.3 | Functional at minimal compute |
| InvisibleInk (k=100) | 15 | 73.7 | 67.9 | |
| InvisibleInk (k=100) | 31 | **75.0** | **69.2** | SOTA |
| Amin et al. | 63 | 68.9 | INF | Completely fails at $B \leq 31$ |
| Amin et al. | 127 | 70.1 | INF | Requires 8×+ compute |
| AdaPMixED | 31 | 58.9 | 56.8 | Worst performance |
| Non-private (ε=0) | - | 68.56 | - | Baseline |

### Ablation Study: Contributions of DClip and Top-$k^+$ (MIMIC, ε=10, B=7)

| Configuration | MAUVE (TinyLLaMA) | MAUVE (LLaMA3.2-1B) | Notes |
|---------------|------------------|--------------------|----|
| InvisibleInk (k=100) | 72% | 76% | Full model |
| InvisibleInk (k=\|V\|, no Top-$k^+$) | 65% | 62% | Removing truncated sampling drops 7–14% |
| Amin et al. (same $B$=7) | ~56% | - | Conventional clipping far inferior to DClip |

### Downstream Task (Yelp Classification, ε=10)

| Method | Batch Size | 50-class Accuracy | Top-5 Acc. | Category Acc. | Score L₁ |
|--------|-----------|------------------|-----------|--------------|---------|
| InvisibleInk | 7 | **32.98%** | **72.16%** | **64.90%** | **0.652** |
| Amin et al. | 127 | 29.44% | 64.56% | 60.18% | 0.748 |
| AdaPMixED | 31 | 7.44% | 34.72% | 56.82% | 1.858 |

### Key Findings
- DClip accounts for the 8× computational efficiency gain: the value range of $\phi_i - \phi_{\text{pub}}$ is approximately 10× smaller than that of $\phi_i$, permitting a much smaller $C$.
- Top-$k^+$ yields greater benefits for large-vocabulary models: the gain for LLaMA3.2 (128K vocabulary) is larger than for TinyLLaMA (32K)—14% vs. 7%.
- Under the strict privacy budget of $\epsilon = 1$, Amin et al. completely fails to generate text at $B \leq 127$, whereas InvisibleInk achieves MAUVE 68.3% at $B = 3$.
- Temperature $\tau \in [1.0, 1.1]$ and $k \approx 100$ are near-optimal across settings, reducing hyperparameter tuning requirements.
- Medical text generated by InvisibleInk contains more medical named entities, indicating better preservation of domain-specific information.

## Highlights & Insights
- **The core insight of DClip is elegant**: the majority of information in logits reflects language priors (grammar, commonsense) that carry no private information and should not consume privacy budget. This observation reduces sensitivity by nearly an order of magnitude.
- **Top-$k^+$ cleverly balances privacy and utility**: directly truncating to the private top-$k$ would leak privacy (a token's probability dropping from positive to zero upon leaving the top-$k$ is observable), whereas a superset constructed from public logits consumes zero privacy budget and adds only approximately 10 candidate tokens—negligible overhead.
- **Graceful degradation has practical engineering value**: the system naturally falls back to public model generation rather than producing incoherent output when the budget is insufficient, which is critical for deployment.
- **A pip-installable package `invink` is provided**, enabling direct generation of DP text and substantially lowering the barrier to adoption.

## Limitations & Future Work
- **Trust model constraint**: Requires a trusted central aggregator with white-box access to raw sensitive data and model weights.
- **Inference-stage only**: Does not protect against privacy leakage during training or fine-tuning, nor does it cover data poisoning or backdoor attacks.
- **Sample-level DP only**: User-level privacy protection (multiple documents from the same user) remains unaddressed.
- **8× overhead leaves room for improvement**: For very large models and extremely strict privacy budgets, an 8× computational overhead remains significant; the linear dependence of the privacy budget on the number of tokens $T$ may potentially be improved through regularization.
- Experiments primarily use 1B-scale models, with 8B-scale models only receiving limited validation; performance on larger models is unknown.

## Related Work & Insights
- **vs. Amin et al. [2024] (EMNLP Findings)**: Also based on the exponential mechanism, but clipping the full logit vector leads to excessive sensitivity, requiring $B \geq 50$ to function. InvisibleInk reduces the required $B$ to single digits via DClip.
- **vs. AdaPMixED [Flemings et al.]**: Adaptive DP budget is rapidly exhausted during long-text generation (generating only ~10 tokens at $B = 8$), and data-dependent privacy guarantees are difficult to calibrate in advance.
- **vs. AugPE [API-access]**: API access constraints make privatization inherently more difficult; InvisibleInk outperforms comprehensively across all settings. However, AugPE remains competitive on datasets well-covered by pretraining data (e.g., Yelp).
- **vs. DP fine-tuning methods**: DP fine-tuning of large models incurs extremely high overhead; inference-time privatization as in InvisibleInk is substantially more cost-effective.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The "clip only the sensitive increment" idea underlying DClip is concise yet powerful; the superset construction in Top-$k^+$ elegantly resolves the tension between truncated sampling and DP.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three privacy-sensitive domain datasets, multiple model scales, and comprehensive ablation and downstream evaluations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous, motivation is clearly articulated, and figures are intuitive (the logit distribution comparison in Fig. 3 is immediately illuminating).
- **Value**: ⭐⭐⭐⭐⭐ Reduces DP text generation overhead from 100× to 4–8×, making it feasible at practical scale for the first time; open-source code further amplifies its impact.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](../../ICLR2026/llm_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [\[NeurIPS 2025\] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)
- [\[NeurIPS 2025\] MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction](masksql_safeguarding_privacy_for_llm-based_text-to-sql_via_abstraction.md)
- [\[NeurIPS 2025\] CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing](cryptomoe_privacy-preserving_and_scalable_mixture_of_experts_inference_via_balan.md)
- [\[NeurIPS 2025\] Demystifying Language Model Forgetting with Low-Rank Example Associations](demystifying_language_model_forgetting_with_low-rank_example_associations.md)

<!-- RELATED:END -->
