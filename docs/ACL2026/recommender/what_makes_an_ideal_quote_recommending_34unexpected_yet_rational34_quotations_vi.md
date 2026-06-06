---
title: >-
  [Paper Note] What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty
description: >-
  [ACL 2026][Recommender Systems][Quote Recommendation] NOVELQR proposes a novelty-driven quote recommendation framework that constructs a deep semantic knowledge base through generative label agents for rational semantic…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Quote Recommendation"
  - "Novelty Estimation"
  - "Defamiliarization Theory"
  - "Deep Semantic Retrieval"
  - "Continuation Bias"
date: 2026-05-08
content_hash: 6151ad7d84d6fefc
---

# What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty

**Conference**: ACL 2026  
**arXiv**: [2602.22220](https://arxiv.org/abs/2602.22220)  
**Code**: None  
**Area**: Recommender Systems / Natural Language Generation  
**Keywords**: Quote Recommendation, Novelty Estimation, Defamiliarization Theory, Deep Semantic Retrieval, Continuation Bias

## TL;DR

NOVELQR proposes a novelty-driven quote recommendation framework that constructs a deep semantic knowledge base through generative label agents for rational semantic retrieval and utilizes a token-level novelty estimator to mitigate autoregressive continuation bias, significantly improving recommendation quality on bilingual benchmarks.

## Background & Motivation

**Background**: Quote recommendation systems aim to recommend suitable famous quotes or aphorisms for a given writing context. Existing systems (such as QuoteR and QUILL) primarily optimize semantic relevance through text embedding matching for retrieval.

**Limitations of Prior Work**: Two key issues persist: (1) existing systems focus solely on surface semantic matching while ignoring the aesthetic value and novelty of quotes, resulting in the recommendation of "correct but stale" quotes (e.g., "Failure is the mother of success") rather than "unexpected yet rational" ones (e.g., Dante's "Beauty awakens the soul to act"); (2) LLMs struggle to understand the deep meaning of quotes when only provided with the quote text, and logit-based novelty metrics (such as surprisal) suffer from autoregressive continuation bias—common phrases, once the beginning is predicted, are completed by "inertia," leading to distorted novelty estimation.

**Key Challenge**: Ideal quotes should be "unexpected yet rational"—potentially confusing at first glance but becoming clear upon connecting with the context. Existing systems perform well on "rationality" but completely ignore the dimension of being "unexpected."

**Goal**: (1) To retrieve within a deep semantic space to ensure the rationality of quotes; (2) To estimate quote novelty without introducing continuation bias.

**Key Insight**: Based on defamiliarization theory ("Art aims to make the familiar strange") and a large-scale user survey (964 questionnaires + controlled experiments), it is confirmed that users indeed prefer "unexpected yet rational" quotes. On this basis, label augmentation is used to compensate for LLM deficiencies in quote comprehension, and token-level novelty focuses on "novel tokens" to mitigate continuation bias.

**Core Idea**: First, use a label agent to map quotes to a deep semantic space to ensure they are "rational," then use token-level novelty reranking to ensure they are "unexpected." These two steps collaborate to achieve an "unexpected yet rational" recommendation.

## Method

### Overall Architecture

NOVELQR consists of three steps: (1) Label Augmentation—using a generative label agent to generate deep semantic explanations and multi-dimensional labels for the quote knowledge base and user context; (2) Rational Retrieval—retrieving candidates in the deep semantic embedding space and using label hard-filtering to ensure semantic consistency; (3) Novelty Reranking—using a token-level novelty estimator to rerank candidates, combining semantic matching and popularity signals for the final recommendation.

### Key Designs

1.  **Generative Label Agent**:
    *   **Function**: Maps quotes from surface text to an interpretable deep semantic space.
    *   **Mechanism**: Based on Qwen3-8B, it executes a four-step process: (a) Comprehensive analysis—analyzing quotes from the perspectives of author background, historical/cultural context, and emotional connotation; (b) Deep meaning generation—refining a semantic summary of no more than 50 words ("Expresses..."); (c) Multi-round refinement—up to 3 rounds of self-criticism and correction to check for oversimplification, overinterpretation, and logical flaws, where approximately 4.6% of outputs are rejected; (d) Semantic label extraction—extracting structured labels across five dimensions: core field, insight, values, target audience, and emotional tone.
    *   **Design Motivation**: Experiments show that LLM ability to understand deep meaning is poor when only given quote text (GPT-4o scores below the high-quality threshold even on the EASY subset) but improves significantly after providing auxiliary information (approaching 9.0 points on HARD).

2.  **Label-enhanced Retrieval**:
    *   **Function**: Retrieves semantically rational candidate quotes in the deep semantic space.
    *   **Mechanism**: Encodes embeddings for the deep meaning of quotes (rather than the raw text), retrieves Top-$N$ ($N=50$) candidates via embedding similarity, and then applies hard filtering based on label similarity in the "core field/value/insight" dimensions (threshold $T=0.7$) to eliminate semantically irrational candidates. Human verification shows the distortion rate of generated labels is below 3%.
    *   **Design Motivation**: Retrieval based on raw text only captures surface relevance; deep semantic retrieval simulates the human thought process of "understanding context before selecting a quote."

3.  **Token-level Novelty Estimator**:
    *   **Function**: Estimates the degree of novelty of a quote in a given context while mitigating continuation bias.
    *   **Mechanism**: Defines token-level novelty as the difference between unconditioned and conditioned logits: $R_t = \log p_{\text{prior}}(x_t) - \log p_{\text{cond}}(x_t)$. The key innovation is identifying "novel tokens": finding "mutation points" (indicating a semantic turn within the quote) through the second-order difference of the self-perplexity sequence $|\delta_2(t)|$. High weights are assigned to these novel tokens, while smooth continuation segments (the primary source of continuation bias) are downweighted. The final novelty score is $S_N = \sum_t \tilde{w}_t R_t$, where $\tilde{w}_t$ is determined by normalized mutation weights.
    *   **Design Motivation**: Standard surprisal or KL divergence at the token or quote level is heavily distorted by continuation bias—for example, the beginning of "Genius is one percent inspiration..." is hard to predict, but "ninety-nine percent perspiration" becomes an inevitable continuation.

### Loss & Training

The final reranking score is $S_{\text{final}} = \lambda_1 \cdot S_N + \lambda_2 \cdot S_P + \lambda_3 \cdot S_M$, where $S_N$ is novelty, $S_P$ is a popularity signal based on Bing search frequency (to avoid recommending overly obscure quotes), and $S_M$ is the cosine similarity of deep meanings. The weights are $\lambda_1=0.70, \lambda_2=0.20, \lambda_3=0.10$.

## Key Experimental Results

### Main Results

**Quote Recommendation Quality Comparison (NOVELQR-BENCH)**

| Method | Novelty | Match | HR@5 | nDCG@5 |
| :--- | :--- | :--- | :--- | :--- |
| QR + No Rerank | 3.14 | 3.99 | 0.35 | 0.26 |
| QUILL | 3.08 | 4.15 | 0.15 | 0.12 |
| LR + No Rerank | 3.40 | **4.55** | 0.55 | 0.44 |
| LR + GPT Rerank | 3.75 | 4.50 | 0.66 | 0.47 |
| **LR + Ours** | **3.81** | 4.50 | **0.70** | **0.51** |

### Ablation Study

| Configuration | Novelty | Match | HR@5 | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| Self-BLEU | 3.55 | 4.48 | 0.50 | Lexical novelty is insufficient |
| Surprisal | 3.66 | 4.31 | 0.55 | Presence of continuation bias |
| + Novelty-token | **3.73** | **4.39** | **0.62** | Mitigating continuation bias is effective |

### Key Findings

*   Switching from QR to LR (Label-enhanced Retrieval) resulted in a significant increase in Match from 3.99 to 4.55, validating the advantage of deep semantic retrieval.
*   The novel token mechanism improved the HR@5 of Surprisal from 0.55 to 0.62, directly validating the effect of mitigating continuation bias.
*   In human multi-choice studies, 78% of preferences favored recommendations from the NOVELQR system.
*   The removal of the popularity signal led to a decrease in consistency, suggesting its necessity as a regularizer to avoid "overly obscure" recommendations.

## Highlights & Insights

*   The use of defamiliarization theory combined with large-scale user surveys to prove that "users indeed want novel quotes" transforms a subjective aesthetic need into an operationalized engineering goal, which is methodologically persuasive.
*   The discovery of continuation bias and the token-level mitigation strategy are ingenious—identifying "semantic turning points" through the second-order difference of self-perplexity is a signal that can be transferred to any scenario requiring text novelty estimation.
*   The four-step processing flow of the label agent (Analysis → Generation → Refinement → Extraction) provides a general paradigm for "LLMs understanding difficult text."

## Limitations & Future Work

*   The label agent relies on auxiliary information (author, source); performance may decline for anonymous quotes or those with unknown origins.
*   The definition of novelty leans toward "semantic unexpectedness" and does not account for aesthetic effects brought by rhetorical devices (e.g., irony, puns).
*   The popularity signal relies on search engines, and its portability across languages and cultures remains to be verified.
*   The scale of the test set is relatively small, with only 100 entries per dataset.

## Related Work & Insights

*   **vs QuoteR/QUILL**: These systems optimize semantic relevance, whereas NOVELQR additionally optimizes novelty.
*   **vs Surprisal/KL-divergence**: These standard novelty metrics are distorted by continuation bias; NOVELQR's token-level method explicitly mitigates this issue.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ Formulates a complete closed loop from theory (defamiliarization) to user research to technical implementation; the discovery of continuation bias is an independent academic contribution.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers bilingual and multi-domain scenarios with human evaluation, though the test set size is small.
*   Writing Quality: ⭐⭐⭐⭐⭐ Excellent problem definition ("unexpected yet rational") and smooth narrative.
*   Value: ⭐⭐⭐⭐ Makes a significant contribution to the field of quote recommendation; the discovery of continuation bias can be transferred to broader contexts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context](what_makes_llms_effective_sequential_recommenders_a_study_on_preference_intensit.md)
- [\[ACL 2026\] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)
- [\[NeurIPS 2025\] Measuring What Matters: Construct Validity in Large Language Model Benchmarks](../../NeurIPS2025/recommender/measuring_what_matters_construct_validity_in_large_language_model_benchmarks.md)
- [\[ACL 2026\] Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation](mirroring_users_towards_building_preference-aligned_user_simulator_with_user_fee.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)

</div>

<!-- RELATED:END -->
