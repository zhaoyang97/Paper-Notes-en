---
title: >-
  [Paper Note] What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty
description: >-
  [ACL 2026][Recommender Systems][Quote recommendation] NOVELQR proposes a novelty-driven quote recommendation framework that constructs a deep semantic knowledge base via a generative label agent to enable semantically rational retrieval, and employs a token-level novelty estimator to mitigate autoregressive continuation bias, achieving significant improvements on a bilingual benchmark.
tags:
  - ACL 2026
  - Recommender Systems
  - Quote recommendation
  - novelty estimation
  - defamiliarization theory
  - deep semantic retrieval
  - continuation bias
date: 2026-05-08
content_hash: 77b7b9595f95cde8
---

# What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty

**Conference**: ACL 2026
**arXiv**: [2602.22220](https://arxiv.org/abs/2602.22220)
**Code**: None
**Area**: Recommender Systems / Natural Language Generation
**Keywords**: Quote recommendation, novelty estimation, defamiliarization theory, deep semantic retrieval, continuation bias

## TL;DR

NOVELQR proposes a novelty-driven quote recommendation framework that constructs a deep semantic knowledge base via a generative label agent to enable semantically rational retrieval, and employs a token-level novelty estimator to mitigate autoregressive continuation bias, achieving significant improvements on a bilingual benchmark.

## Background & Motivation

**State of the Field**: Quote recommendation systems aim to suggest appropriate quotations given a writing context. Existing systems (e.g., QuoteR, QUILL) primarily optimize for semantic relevance through text embedding matching.

**Limitations of Prior Work**: Two critical issues arise: (1) existing systems focus solely on surface-level semantic matching while ignoring the aesthetic value and novelty of quotations, recommending "correct but stale" quotes (e.g., "Failure is the mother of success") rather than "unexpected yet rational" gems (e.g., Dante's "Beauty awakens the soul to act"); (2) LLMs struggle to grasp the deep meaning of a quotation from its text alone, and logit-based novelty metrics (e.g., surprisal) suffer from autoregressive continuation bias—once the beginning of a common phrase is predicted, the model "inertially" completes it, distorting novelty estimation.

**Root Cause**: An ideal quotation should be "unexpected yet rational"—a reader may be puzzled at first sight, but upon connecting it to context, gains sudden clarity. Existing systems perform reasonably well on "rationality" but entirely neglect the "unexpectedness" dimension.

**Paper Goals**: (1) Retrieve in a deep semantic space to ensure the rationality of recommended quotes; (2) estimate quotation novelty without introducing continuation bias.

**Starting Point**: Grounded in defamiliarization theory ("art aims to make the familiar strange") and a large-scale user study (964-participant survey + controlled experiment), the paper confirms that users genuinely prefer "unexpected yet rational" quotes. Building on this, label augmentation compensates for LLMs' deficiency in quote comprehension, while token-level novelty focusing on "novel tokens" mitigates continuation bias.

**Core Idea**: First map quotations into a deep semantic space via a label agent to ensure "rationality"; then rerank by token-level novelty to ensure "unexpectedness"—two steps collaborating to achieve "unexpected yet rational" recommendations.

## Method

### Overall Architecture

NOVELQR operates in three stages: (1) **Label Augmentation**—a generative label agent generates deep semantic interpretations and multi-dimensional labels for both the quote knowledge base and user context; (2) **Rational Retrieval**—candidates are retrieved in the deep semantic embedding space and filtered via label-based hard constraints to ensure semantic consistency; (3) **Novelty Reranking**—a token-level novelty estimator reranks candidates, combining semantic matching and popularity signals to produce the final recommendation.

### Key Designs

1. **Generative Label Agent**:

    - **Function**: Maps quotations from surface text to an interpretable deep semantic space.
    - **Mechanism**: Built on Qwen3-8B, the agent executes a four-step pipeline: (a) comprehensive analysis—examining the author's background, historical and cultural context, and emotional connotations; (b) deep meaning generation—distilling a semantic summary of no more than 50 words ("expresses..."); (c) multi-round revision—up to 3 rounds of self-critique and correction checking for superficialization, over-interpretation, and logical gaps, with approximately 4.6% of outputs rejected; (d) semantic label extraction—extracting structured labels across five dimensions: core domain, insights, values, target audience, and emotional tone.
    - **Design Motivation**: Experiments show that LLMs perform poorly at understanding the deep meaning of quotations when given only the text (GPT-4o scores below the high-quality comprehension threshold even on the EASY subset), but improve substantially when auxiliary information is provided (approaching 9.0 on the HARD subset).

2. **Label-enhanced Retrieval**:

    - **Function**: Retrieves semantically rational candidate quotations in the deep semantic space.
    - **Mechanism**: Embeddings are computed over the deep meanings (rather than raw text) of quotations; Top-N (=50) candidates are retrieved by embedding similarity, then hard-filtered using label similarity on the "core domain/values/insights" dimensions (threshold T=0.7) to remove semantically inconsistent candidates. Manual verification confirms that the label distortion rate is below 3%.
    - **Design Motivation**: Retrieval based on raw text captures only surface-level relevance, while deep-meaning retrieval simulates the human cognitive process of "first understanding context, then selecting a quote."

3. **Token-level Novelty Estimator**:

    - **Function**: Estimates the novelty of a quotation in a given context while mitigating continuation bias.
    - **Mechanism**: Token-level novelty is defined as the difference between context-free and context-conditioned log probabilities: $R_t = \log p_{\text{prior}}(x_t) - \log p_{\text{cond}}(x_t)$. The key innovation lies in identifying "novel tokens": second-order differences of the self-perplexity sequence $|\delta_2(t)|$ locate "inflection points" indicating semantic shifts within the quotation. These novel tokens receive high weights, while smoothly continuing segments (the primary source of continuation bias) are down-weighted. The final novelty score is $S_N = \sum_t \tilde{w}_t R_t$, where $\tilde{w}_t$ is determined by normalized inflection weights.
    - **Design Motivation**: Standard surprisal or KL divergence aggregated at the token or quotation level is severely distorted by continuation bias—for example, the opening of "Genius is one percent inspiration..." is hard to predict, but "...ninety-nine percent perspiration" becomes an inevitable continuation.

### Loss & Training

The final reranking score is $S_{\text{final}} = \lambda_1 \cdot S_N + \lambda_2 \cdot S_P + \lambda_3 \cdot S_M$, where $S_N$ denotes novelty, $S_P$ is a popularity signal based on Bing search frequency (to avoid recommending overly obscure quotes), and $S_M$ is the cosine similarity of deep meanings. Weights are set to $\lambda_1=0.70$, $\lambda_2=0.20$, $\lambda_3=0.10$.

## Key Experimental Results

### Main Results

**Quote Recommendation Quality Comparison (NOVELQR-BENCH)**

| Method | Novelty | Match | HR@5 | nDCG@5 |
|--------|---------|-------|------|--------|
| QR + No Rerank | 3.14 | 3.99 | 0.35 | 0.26 |
| QUILL | 3.08 | 4.15 | 0.15 | 0.12 |
| LR + No Rerank | 3.40 | **4.55** | 0.55 | 0.44 |
| LR + GPT Rerank | 3.75 | 4.50 | 0.66 | 0.47 |
| **LR + Ours** | **3.81** | 4.50 | **0.70** | **0.51** |

### Ablation Study

| Configuration | Novelty | Match | HR@5 | Note |
|---------------|---------|-------|------|------|
| Self-BLEU | 3.55 | 4.48 | 0.50 | Lexical novelty insufficient |
| Surprisal | 3.66 | 4.31 | 0.55 | Affected by continuation bias |
| + Novelty-token | **3.73** | **4.39** | **0.62** | Bias mitigation effective |

### Key Findings

- Switching from QR to LR (label-enhanced retrieval) substantially improves Match from 3.99 to 4.55, validating the advantage of deep semantic retrieval.
- The novel-token mechanism improves Surprisal's HR@5 from 0.55 to 0.62, directly validating the effectiveness of continuation bias mitigation.
- In a human multiple-choice study, 78% of participants preferred recommendations from the NOVELQR system.
- Removing the popularity signal leads to degraded consistency, confirming its role as a necessary regularizer against overly obscure recommendations.

## Highlights & Insights

- Leveraging defamiliarization theory and a large-scale user study to demonstrate that "users genuinely want novel quotes" operationalizes a subjective aesthetic demand into an engineering objective—a methodologically compelling approach.
- The identification of continuation bias and the token-level mitigation strategy are particularly elegant: using second-order differences of self-perplexity to locate "semantic inflection points" yields a signal transferable to any scenario requiring text novelty estimation.
- The four-step pipeline of the label agent (analysis → generation → revision → extraction) offers a general paradigm for "LLM comprehension of difficult texts."

## Limitations & Future Work

- The label agent relies on auxiliary information (author, source); performance may degrade for anonymous or unattributed quotations.
- The novelty definition is oriented toward "semantic unexpectedness" and does not account for aesthetic effects arising from rhetorical devices such as irony or wordplay.
- The popularity signal depends on search engines; cross-lingual and cross-cultural portability remains to be validated.
- Each test set contains only 100 instances, which is a relatively small scale.

## Related Work & Insights

- **vs. QuoteR/QUILL**: These systems optimize for semantic relevance; NOVELQR additionally optimizes for novelty.
- **vs. Surprisal/KL-divergence**: These standard novelty metrics are distorted by continuation bias; NOVELQR's token-level approach explicitly mitigates this problem.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Theory (defamiliarization), user study, and technical implementation form a complete closed loop; the identification of continuation bias constitutes an independent academic contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers bilingual, multi-domain settings with human evaluation, though test set size is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The problem framing ("unexpected yet rational") is compelling, and the narrative is cohesive.
- **Value**: ⭐⭐⭐⭐ — Makes a significant contribution to quote recommendation and offers findings on continuation bias transferable to broader scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context](what_makes_llms_effective_sequential_recommenders_a_study_on_preference_intensit.md)
- [\[ACL 2026\] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)
- [\[NeurIPS 2025\] Measuring What Matters: Construct Validity in Large Language Model Benchmarks](../../NeurIPS2025/recommender/measuring_what_matters_construct_validity_in_large_language_model_benchmarks.md)
- [\[ACL 2026\] Content Fuzzing for Escaping Information Cocoons on Social Media](content_fuzzing_for_escaping_information_cocoons_on_digital_social_media.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)

</div>

<!-- RELATED:END -->
