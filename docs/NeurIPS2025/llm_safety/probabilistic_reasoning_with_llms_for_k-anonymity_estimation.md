---
title: >-
  [Paper Note] Probabilistic Reasoning with LLMs for K-Anonymity Estimation
description: >-
  [NeurIPS 2025][LLM Safety][privacy risk estimation] This paper proposes Branch, a framework that leverages large language models to model personal information disclosed in user-generated text as a joint probability distr…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "privacy risk estimation"
  - "k-anonymity"
  - "Bayesian networks"
  - "probabilistic reasoning"
  - "large language models"
date: 2026-05-08
content_hash: d589263730547ba6
---

# Probabilistic Reasoning with LLMs for K-Anonymity Estimation

**Conference**: NeurIPS 2025
**arXiv**: [2503.09674](https://arxiv.org/abs/2503.09674)  
**Code**: None  
**Area**: AI Safety / Privacy Protection
**Keywords**: privacy risk estimation, k-anonymity, Bayesian networks, probabilistic reasoning, large language models

## TL;DR
This paper proposes Branch, a framework that leverages large language models to model personal information disclosed in user-generated text as a joint probability distribution over a Bayesian network. By estimating conditional probabilities for individual attributes and composing them to compute k-anonymity values (i.e., the number of individuals globally matching a given profile), Branch achieves 73% accuracy on privacy risk estimation, outperforming o3-mini chain-of-thought reasoning by 13%.

## Background & Motivation
As users share increasing amounts of personal information on platforms such as Reddit and ChatGPT, quantifying the privacy risks associated with such textual disclosures has become increasingly important. Traditional k-anonymity research focuses on anonymizing datasets from the perspective of the data holder, whereas work on quantifying privacy risk from the perspective of the data contributor (e.g., social media users) remains largely unexplored.

**Limitations of Prior Work**:
- When prompted to reason about privacy risk using chain-of-thought (CoT), LLMs tend to commit three categories of errors: incorrect independence assumptions (ignoring conditional dependencies among attributes), order-of-magnitude estimation errors, and arithmetic errors in probability computation.
- For example, when a user mentions being Italian, 26 years old, on the autism spectrum, and experiencing social anxiety, CoT independently estimates the probability of each attribute and multiplies them, entirely ignoring the strong conditional dependence between autism spectrum condition and social anxiety.

**Key Challenge**: Can LLMs effectively perform numerical reasoning under uncertainty? How can the demographic knowledge internalized by LLMs be leveraged to estimate joint probabilities over multiple interrelated attributes?

**Key Insight**: Personal information disclosures in text are modeled as random variables; LLMs implicitly construct a Bayesian network to factorize the joint probability distribution, estimating conditional probabilities sequentially before composing them into a final result.

## Method

### Overall Architecture
Branch (Bayesian network Reasoning for k-ANonymity using Conditional Hierarchies) operates in four steps: (1) extracting personal attributes from text as random variables; (2) having the LLM determine variable ordering and conditional dependencies to implicitly construct a Bayesian network; (3) translating conditional probabilities into textual queries for the LLM to estimate individually; and (4) composing probabilities according to the Bayesian network structure to compute the final k value.

### Key Designs

1. **Structure Elicitation**:

    - The LLM first selects an ordering of the disclosed attributes, guided by the principle of choosing the conditioning direction for which statistical data are more readily available. For instance, $P(\text{woman} \mid \text{work in Tech})$ is easier to estimate than $P(\text{work in Tech} \mid \text{woman})$, because gender breakdowns within occupations are more commonly reported than occupational distributions within gender groups.
    - The LLM then determines the conditional dependencies for each attribute. Not all attributes need to be fully connected; independent attributes can simplify the network. For example, "gender" and "home ownership" may be treated as conditionally independent given "occupation" and "location."
    - This factorization decomposes the estimation of a high-dimensional joint probability into multiple lower-dimensional conditional probability estimations.

2. **Query Generation and Sub-query Decomposition**:

    - Conditional probabilities are translated into natural language queries. For example, $P(\text{maternity leave} \mid \text{work in Tech, Townsbridge})$ is reformulated as "the proportion of mothers among tech workers in Townsbridge."
    - The first disclosed attribute is combined with the overall population to generate a population-specific query (e.g., "population of Townsbridge"), reducing variance in the k value estimate.
    - Complex queries can be further decomposed into sub-queries; for instance, $P(\text{no landlords} \mid \text{Townsbridge})$ is decomposed into "proportion who own property" and "proportion living with parents."

3. **Query Estimation and Confidence Calibration**:

    - The LLM draws on internalized demographic knowledge to estimate numerical values for each query.
    - A key innovation is that the LLM reports a confidence level for each estimate; low-confidence queries are simplified (e.g., by dropping a conditional dependency) and re-estimated.
    - Final arithmetic computations are executed by a Python interpreter to avoid LLM arithmetic errors.
    - Retrieval-augmented generation (RAG) can optionally be incorporated to improve estimation accuracy.

### Loss & Training
- The fine-tuned variant of Branch uses LLaMA-3.1-Instruct-8B fine-tuned via LoRA on a manually annotated training set.
- Fine-tuning is conducted separately for five components: disclosure selection, variable ordering, conditional dependency determination, query generation, and statistical estimation.
- At inference time, sampling decoding with temperature 0.7 is used, with 3 demonstrations per post.

## Key Experimental Results

### Main Results

| Method | Model | Spearman ρ↑ | Log Error↓ | Range%↑ (a=5) |
|--------|-------|-------------|-----------|---------------|
| Branch | o3-mini | **0.878** | **1.83** | **72.61%** |
| Branch | GPT-4o | 0.834 | 2.33 | 66.96% |
| CoT | o3-mini | 0.766 | 2.81 | 59.13% |
| CoT | o1-preview | 0.636 | 3.23 | 52.17% |
| CoT | DeepSeek-R1 | 0.684 | 3.36 | 53.91% |
| Few-Shot | GPT-4o | 0.565 | 3.94 | 42.17% |
| Branch | LLaMA-8B (FT) | 0.807 | 2.19 | 69.57% |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---------------|-----------|---------|
| Documents with 4+ attributes | Branch significantly outperforms CoT | More attributes amplify CoT's independence assumption errors |
| Documents with 1–3 attributes | Gap between Branch and CoT narrows | Joint probability complexity is low with few attributes |
| High-variance predictions | Accuracy drops by 37.47% | LLM uncertainty serves as a reliable indicator of estimation accuracy |
| Single-attribute estimation | Low demographic percentage error | Branch's fundamental module performs reliably |

### Key Findings
- Branch outperforms all baselines on documents from both Reddit and ShareGPT, demonstrating cross-domain generalization.
- The most common error in chain-of-thought reasoning is the failure to account for conditional dependencies among attributes (accounting for 50%+ of errors).
- Fine-tuned LLaMA-8B Branch approaches the performance of GPT-4o Branch, suggesting that the Branch framework itself matters more than the underlying model.
- LLM uncertainty (consistency across multiple generations) is a reliable indicator of estimation accuracy.

## Highlights & Insights
- **Novel problem formulation**: This work is the first to reframe k-anonymity from the data holder's perspective to that of the data contributor, providing users with interpretable quantification of privacy risk.
- **Elegant Bayesian factorization strategy**: Decomposing high-dimensional joint probabilities into multiple lower-dimensional conditional probabilities circumvents the difficulty of direct joint probability estimation by LLMs.
- **Information-theoretic perspective**: Evaluating k-value errors on a logarithmic scale naturally corresponds to information-theoretic uncertainty measures — small deviations at low k values are more consequential than large deviations at high k values.
- The dataset construction methodology is rigorous: dual annotation, synthetic data generation, and validation against census records, with annotation quality comparable to official census error rates.
- Strong practical utility: the framework could serve as an online privacy tool analogous to a "password strength meter."

## Limitations & Future Work
- Demographic knowledge internalized by LLMs may be outdated or biased due to training data cutoff limitations.
- The dataset of 220 documents is relatively small and may not cover all privacy scenarios.
- The work focuses exclusively on English-language text and demographics primarily relevant to Western countries.
- The threat model assumes the adversary has access to all disclosed context across posts, which may be overly conservative.
- Multi-post correlation analysis and temporal tracking are not considered.

## Related Work & Insights
- **vs. Traditional k-anonymity**: Conventional approaches require full database access to anonymize data; Branch requires no database and instead leverages LLM knowledge for estimation.
- **vs. LLM mathematical reasoning**: This task goes beyond standard mathematical reasoning benchmarks by requiring LLMs to perform probabilistic reasoning under uncertainty while integrating real-world knowledge.
- **Insight**: The Bayesian factorization strategy underlying Branch is generalizable to other tasks that require LLMs to perform probabilistic reasoning, such as risk assessment and decision support.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Novel problem formulation; quantifying privacy risk via k-anonymity from the user's perspective is original, and the Branch framework is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive multi-model comparisons, ablation analyses, and census-based validation, though the dataset size is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, intuitive figures, and thorough error analysis.
- Value: ⭐⭐⭐⭐ — Offers both academic value (a new benchmark for LLM probabilistic reasoning) and practical value (a user-facing privacy protection tool).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Contextual Integrity in LLMs via Reasoning and Reinforcement Learning](contextual_integrity_in_llms_via_reasoning_and_reinforcement_learning.md)
- [\[ACL 2026\] ADVICE: Answer-Dependent Verbalized Confidence Estimation](../../ACL2026/llm_safety/advice_answer-dependent_verbalized_confidence_estimation.md)
- [\[NeurIPS 2025\] Evaluation of Vision-LLMs in Surveillance Video](evaluation_of_vision-llms_in_surveillance_video.md)
- [\[NeurIPS 2025\] DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning](dragon_guard_llm_unlearning_in_context_via_negative_detection_and_reasoning.md)
- [\[NeurIPS 2025\] One Token Embedding Is Enough to Deadlock Your Large Reasoning Model](one_token_embedding_is_enough_to_deadlock_your_large_reasoning_model.md)

</div>

<!-- RELATED:END -->
