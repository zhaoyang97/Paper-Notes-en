---
title: >-
  [Paper Note] DeFine: Decision-Making with Analogical Reasoning over Factor Profiles
description: >-
  [ACL 2025][Reasoning][Analogical Reasoning] This paper proposes the DeFine framework, which constructs probabilistic factor profiles from spoken transcripts in complex scenarios such as earnings calls. By combining the Bradley-Terry model to identify key factors and using KL divergence between factor profiles for analogical reasoning, DeFine assists LLMs in making investment decisions under uncertainty, outperforming baselines in both accuracy and F1 score.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Analogical Reasoning"
  - "Decision-Making"
  - "Factor Profiles"
  - "Bradley-Terry Model"
  - "Earnings Call"
date: 2026-05-08
content_hash: c0cf97cd8cbe84b8
---

# DeFine: Decision-Making with Analogical Reasoning over Factor Profiles

**Conference**: ACL 2025  
**arXiv**: [2410.01772](https://arxiv.org/abs/2410.01772)  
**Code**: To be confirmed  
**Area**: LLM Reasoning  
**Keywords**: Analogical Reasoning, Decision-Making, Factor Profiles, Bradley-Terry Model, Earnings Call

## TL;DR
This paper proposes the DeFine framework, which constructs probabilistic factor profiles from spoken transcripts in complex scenarios such as earnings calls. By combining the Bradley-Terry model to identify key factors and using KL divergence between factor profiles for analogical reasoning, DeFine assists LLMs in making investment decisions under uncertainty, outperforming baselines in both accuracy and F1 score.

## Background & Motivation

**Background**: LLMs have been widely used for reasoning and decision-making tasks. However, dealing with long real-world documents (such as earnings call transcripts, averaging around 10K tokens) poses significant challenges—recency bias, hallucinations, and numerical inconsistency affect decision reliability.

**Limitations of Prior Work**: (a) Corporate executives tend to convey positive information in earnings calls to reassure investors, despite substantial actual uncertainty; (b) LLMs perform poorly when directly processing lengthy transcripts; (c) Existing approaches lack precise and quantitative descriptions of key decision factors and do not systematically integrate uncertainty into decision-making.

**Key Challenge**: LLMs can generate chains of thought, but their explanations are often ambiguous or unfaithful, and they cannot quantify the contribution weights of different factors to the final decision.

**Goal**: How to extract structured decision factors and their uncertainties from lengthy and ambiguous conference transcripts, and utilize historical analogous cases to assist the current decision?

**Key Insight**: Compress information into probabilistic factor profiles (where each factor has multiple outcomes and their probabilities), retrieve analogous cases using the similarity of factor profiles (rather than textual similarity), and let LLMs make decisions by referencing similar historical cases.

**Core Idea**: Structure complex scenarios into probabilistic factor profiles, identify key factors using the BT model, and retrieve analogous cases via KL divergence to assist LLM decision-making.

## Method

### Overall Architecture
The input is the transcript of an earnings call, and the output is one of five investment decisions (strong buy / buy / hold / sell / strong sell). The pipeline includes: (1) extracting probabilistic profiles for 15 factors from the transcripts; (2) performing factor-wise pairwise comparisons using the Bradley-Terry model to quantify the influence of each factor on decisions; (3) retrieving similar factor profiles from historical cases using KL divergence as analogical exemplars; (4) feeding the current factor profile along with the Top-K analogical cases into the LLM for the final decision.

### Key Designs

1. **Probabilistic Factor Profile Construction**:

    - **Function**: Compresses the transcript into 15 factors (categorized into macroeconomics, company dynamics, and historical financial metrics) and their probability distributions.
    - **Mechanism**: Leveraging the structured output capability of GPT-4o, a brief summary of each factor is first generated from the transcript, verbal probabilities are assigned to its possible outcomes (very unlikely $\rightarrow$ very likely, mapped to 1-6), and they are finally normalized into a probability $P(O_{ij}|X) = \frac{P_{i,j}}{\sum_k P_{i,k}}$.
    - **Design Motivation**: Factor profiles not only capture explicitly mentioned content but also label "unknown/uncertain" to reflect missing information in the text—something traditional text summarization cannot achieve.

2. **Bradley-Terry Model for Identifying Key Factors**:

    - **Function**: Quantifies the relative influence of each factor on the investment decision.
    - **Mechanism**: Pairwise comparisons are performed on transcript pairs with different labels in the training set to construct a factor-level preference matrix $W$, and the EM algorithm is used to estimate the strength coefficient $p_x = e^{\beta_x}$ for each factor-outcome pair. The comparison weights are the product of the probabilities of the corresponding factor outcomes in the two transcripts, i.e., $P(O_{ij}|X^{(A)}) \times P(O_{ij}|X^{(B)})$.
    - **Design Motivation**: Key factors differ across industries (e.g., the technology sector is more sensitive to "uncertainty" factors, while the consumer defensive sector is more sensitive to regulatory changes). The BT model can automatically discover these in a data-driven manner.

3. **Analogical Reasoning over Factor Profiles**:

    - **Function**: Retrieves the Top-K historical cases whose factor profiles are most similar to the current transcript.
    - **Mechanism**: KL divergence is used to measure the similarity between two factor profiles: $D_{KL}(P||Q) = \sum_{i,j} P(O_{ij}|X) \log \frac{P(O_{ij}|X)}{Q(O_{ij}|X_c)}$. The Top-K cases with the minimum KL divergence are selected as analogical exemplars, which are fed into the LLM along with the current factor profile for decision-making.
    - **Design Motivation**: Searching based on factor profiles rather than the full text focuses on the similarity of market drivers, avoiding interference from irrelevant details. For instance, while the full texts of Google and Broadcom transcripts may differ significantly, their factor profiles might be highly similar.

### Loss & Training
- No training required; it is an inference-only framework.
- BT model parameters are estimated on the training set using the EM algorithm.
- LLM decision-making is implemented via in-context learning.

## Key Experimental Results

### Main Results

| Method | Recall | Precision | F1 | Accuracy |
|------|--------|-----------|-----|----------|
| LLM+CoT+Trans | 21.56 | 33.66 | 13.52 | 19.59 |
| LLM+CoT+Summ | 22.77 | 16.17 | 14.12 | 20.61 |
| LLM+CoT+Factors | 24.38 | 28.58 | 17.26 | 22.32 |
| DeLLMa | 38.30 | 23.14 | 16.68 | 22.35 |
| **DeFine** | **26.15** | **27.67** | **23.73** | **29.64** |

DeFine significantly outperforms other models in both F1 (23.73) and Accuracy (29.64), with its F1 score being 7.05 percentage points higher than the best baseline, DeLLMa.

### Ablation Study (DeFine-BT Variants)

| Configuration | Description |
|------|------|
| DeFine-BT-Same Sector | Pairwise comparison within the same industry, exhibiting stable performance |
| DeFine-BT-Cross Sectors | Cross-industry comparison, still effective but showing different factor preferences |
| DeFine-BT-Same Company | Comparison with the same company's history, leveraging the company's own trends |
| All BT Variants | All outperform the random baseline and DeLLMa |

### Key Findings
- **Factor Profiles Outperform Full Text and Summaries**: LLM+CoT+Factors (22.32%) > LLM+CoT+Trans (19.59%), indicating that structured factors are more beneficial for decision-making than lengthy documents.
- **Analogical Reasoning is the Key to Improvement**: DeFine (29.64%) vs LLM+CoT+Factors (22.32%), suggesting that analogical cases provide concrete historical references.
- **Significant Industry Differences**: Bullish factors in the tech industry are dominated by "uncertainty" (where economic conditions, market sentiment, etc., are "unknown"), reflecting the market's high growth expectations for tech companies. In contrast, the consumer defensive industry is most affected by regulatory changes and black swan events.
- **Weakest Performance on "Strong Sell" Predictions**: This is because executives tend to put a positive spin on things during earnings calls, causing the transcript text itself to possess a positive bias.

## Highlights & Insights
- **Broad Transferability of the Probabilistic Factor Profile Concept**: Not limited to finance, this paradigm of "extracting structured probabilistic factors from long documents" can be applied to other decision-making scenarios under uncertainty, such as political debates, consulting, and risk assessment.
- **Using the BT Model for Factor Importance Ranking**: The classic ranking model is elegantly applied in this scenario, converting pairwise comparison preference data into factor strength coefficients.
- **Innovation in Retrieval Paradigm**: Retrieval based on the KL divergence of factor profiles is more focused on decision-relevant information than semantic retrieval based on text embeddings.

## Limitations & Future Work
- **Absolute accuracy remains low** (29.64%, compared to a 20% random baseline for a 5-class classification), indicating that stock prediction itself is extremely difficult, and factor profiles can only capture partial information.
- **Factor selection is fixed to 15**: Manual selection may miss important factors, and the optimal factor set might vary across different industries.
- **Reliance on GPT-4o for factor extraction**: Extraction quality is constrained by the capability of the LLM, and it may introduce the LLM's own biases.
- **30-day stock price movements are affected by too many external factors**: Earnings calls are only one of many signals.
- **Potential improvements**: (a) The factor set could be adaptively adjusted (e.g., using LLMs to generate industry-specific factor sets); (b) A shorter prediction window (e.g., 1–3 days) might better reflect the direct impact of earnings call information.

## Related Work & Insights
- **vs DeLLMa (Liu et al., 2024)**: DeLLMa uses decision theory to rank state-action pairs but does not utilize analogical reasoning. DeFine provides a richer basis for decision-making through similar historical cases.
- **vs LLM+CoT+Trans**: Directly processing full transcripts yields the worst performance, indicating that LLMs still have limited reasoning capabilities over extremely long texts, and structured preprocessing is highly necessary.
- **vs Traditional Bayesian Inference**: Traditional approaches usually lack direct associations with historical cases, a limitation that DeFine's analogical reasoning remedies.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of probabilistic factor profiles, BT model, and analogical reasoning is novel in financial NLP.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation on 11,950 real-world earnings documents, cross-industry analysis, and multiple baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition with well-described modular methods.
- Value: ⭐⭐⭐ Low stock prediction accuracy limits practical utility, but the framework design offers valuable insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models](../../NeurIPS2025/llm_reasoning/i-raven-x_benchmarking_generalization_and_robustness_of_analogical_and_mathemati.md)
- [\[ICML 2026\] Reasoning Can Be Restored by Correcting a Few Decision Tokens](../../ICML2026/llm_reasoning/reasoning_can_be_restored_by_correcting_a_few_decision_tokens.md)
- [\[ACL 2026\] ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs](../../ACL2026/llm_reasoning/chairo_contextual_hierarchical_analogical_induction_and_reasoning_optimization_f.md)
- [\[ICLR 2026\] Making, Not Taking, the Best of N](../../ICLR2026/llm_reasoning/making_not_taking_the_best_of_n.md)
- [\[AAAI 2026\] The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models](../../AAAI2026/llm_reasoning/the_curious_case_of_analogies_investigating_analogical_reasoning_in_large_langua.md)

</div>

<!-- RELATED:END -->
