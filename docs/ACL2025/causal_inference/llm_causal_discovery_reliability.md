---
title: >-
  [Paper Note] On the Reliability of Large Language Models for Causal Discovery
description: >-
  [ACL 2025][Causal Inference][Causal Discovery] Using pre-training corpora accessible via open-source LLMs (OLMo, BLOOM), this study empirically validates the "Causal Parrot" hypothesis—that an LLM's ability to identify causal relationships is highly correlated with the frequency of that relationship in the pre-training data (Spearman $r=0.9$), and that the presence of erroneous causal relationships and changes in context significantly affect prediction reliability.
tags:
  - "ACL 2025"
  - "Causal Inference"
  - "Causal Discovery"
  - "LLM Reliability"
  - "Memorization"
  - "Pre-training Data"
  - "Contextual Influence"
date: 2026-05-08
content_hash: eaae9ac6e8f2fd0a
---

# On the Reliability of Large Language Models for Causal Discovery

**Conference**: ACL 2025  
**arXiv**: [2407.19638](https://arxiv.org/abs/2407.19638)  
**Code**: [https://github.com/WilliamsToTo/causality_llm](https://github.com/WilliamsToTo/causality_llm)  
**Area**: Causal Reasoning  
**Keywords**: Causal Discovery, LLM Reliability, Memorization, Pre-training Data, Contextual Influence  

## TL;DR
Using pre-training corpora accessible via open-source LLMs (OLMo, BLOOM), this study empirically validates the "Causal Parrot" hypothesis—that an LLM's ability to identify causal relationships is highly correlated with the frequency of that relationship in the pre-training data (Spearman $r=0.9$), and that the presence of erroneous causal relationships and changes in context significantly affect prediction reliability.

## Background & Motivation
**Background**: LLMs perform exceptionally well on causal discovery benchmarks, with GPT-4 even outperforming traditional statistical methods. However, do these models truly understand causal relationships, or do they merely "recall" causal relations seen in their pre-training data?

**Limitations of Prior Work**: Zečević et al. proposed the "Causal Parrot" hypothesis, but prior studies could not provide empirical evidence because they used closed-source LLMs or models without accessible pre-training corpora. Meanwhile, the impacts of erroneous causal information and context variations in the pre-training data on predictions have not been quantitatively studied.

**Key Challenge**: If LLMs rely on memorization, their predictions for novel or rare causal relationships will be unreliable. This severely limits the application of LLMs in real-world causal discovery scenarios.

**Goal**: This paper aims to address: (1) Under what conditions can LLMs reliably predict causal relationships? (2) How do erroneous causal relationships in the pre-training data affect performance? (3) How does contextual information influence causal judgments?

**Key Insight**: By utilizing OLMo (whose pre-training corpus, Dolma, has the search tool WIMBD) and BLOOM (whose ROOTS corpus has a search tool), one can directly query the frequency of causal relations in the pre-training data, establishing a frequency-performance correlation analysis.

**Core Idea**: Systematically validate the reliability boundaries of LLM causal discovery by searching the occurrence frequency of causal relationships in pre-training corpora, performing correlation analysis with model prediction accuracy, and conducting controlled experiments on synthetic data.

## Method

### Overall Architecture
Three research questions correspond to three sets of experiments: RQ1 validates the frequency-performance correlation using real and synthetic data; RQ2 investigates the impact of reversed/negated causal relationships on confidence; RQ3 studies the impact of positive/negative contexts on predictions. All experiments are conducted across six LLMs.

### Key Designs

1. **Frequency-Performance Correlation Analysis (RQ1)**:

    - **Function**: Quantify the correlation between the occurrence frequency of causal relationships in the pre-training data and the prediction accuracy of LLMs.
    - **Mechanism**: Use pre-training corpus search tools (WIMBD/ROOTS Search) to query the number of occurrences of templates like "X causes Y". Group causal relationships into frequency buckets, calculate the F1/accuracy within each bucket, and then compute Spearman/Pearson correlation coefficients.
    - **Design Motivation**: Direct empirical evidence—if high-frequency relations are predicted well while low-frequency ones are predicted poorly, it supports the "Causal Parrot" hypothesis. Controlled experiments using dummy words (such as blaonge, goloneke) on synthetic data further eliminate confounding factors.

2. **Erroneous Causal Relationships Impact Experiment (RQ2)**:

    - **Function**: Quantify the impact of reversed/negated causal relationships on the confidence of correct causal relationships.
    - **Mechanism**: Define confidence as the proportion of positive answers in 10 samplings. Compute the ratio of "occurrences of erroneous relations / occurrences of correct relations" and analyze the correlation between this ratio and confidence. Conduct controlled experiments via Reverse Relation Scaling and Negated Relation Scaling on synthetic data.
    - **Design Motivation**: Erroneous causal information (e.g., "lung cancer causes smoking") inevitably exists in pre-training data, necessitating a quantitative assessment of its "poisoning" effect on models.

3. **Contextual Influence Experiment (RQ3)**:

    - **Function**: Evaluate how positive/negative contexts alter LLM causal judgments.
    - **Mechanism**: Generate 5 positive and 5 negative contexts for each causal relationship using GPT-4o. Test the causal judgment accuracy of 6 LLMs under different contextual conditions.
    - **Design Motivation**: The validity of a causal relationship is context-dependent ("rain causes flooding" holds in poorly drained cities but not in well-drained ones), and LLMs should be able to distinguish this.

## Key Experimental Results

### Main Results (RQ1 Frequency-Performance Correlation)

| Model | Task | Spearman $r$ | Highest Frequency Bin F1 | Lowest Frequency Bin F1 |
|------|------|-----------|----------------|----------------|
| OLMo-7b | Full Causal Discovery | 0.90* | 0.88 | 0.20 |
| OLMo-7b | Direction Identification (ConceptNet) | 0.83* | 0.93 | 0.35 |
| BLOOM-7b | Full Causal Discovery | 0.90* | ~0.8 | ~0.3 |
| Synthetic Data (OLMo) | Direction Identification | 1.00* | ~0.95 | ~0.3 |

### Contextual Influence (RQ3)

| Condition | Full Causal Discovery F1 (Avg. of 6 Models) |
|------|--------------------------|
| No Context | 0.66 |
| Positive Context | **0.83** (+26%) |
| Negative Context | **0.33** (-50%) |

### Key Findings
- **RQ1: Strong empirical support for the "Causal Parrot" hypothesis**: The occurrence frequency of causal relationships in the pre-training corpus is highly positively correlated with prediction accuracy (Spearman $r=0.83\text{--}1.0$), and the F1 of low-frequency relationships can be as low as 1/4 of that of high-frequency ones.
- **RQ2: Erroneous causal relations significantly reduce confidence**: The occurrence ratio of reversed causal relationships is significantly negatively correlated with the confidence of correct relationships (Pearson $r=-0.83 \text{ to } -0.98$). Even if the correct relationship occurs 1,000 times, if the reversed relationship also occurs 1,000 times, the confidence drops substantially.
- **RQ3: Context can double or halve accuracy**: Positive contexts increase F1 from 0.66 to 0.83, while negative contexts drop it to 0.33. This shows that LLM causal discovery without context is inherently unreliable.
- GPT-4o achieves an F1 of 0.92 under positive contexts but similarly plummets to 0.27 under negative contexts.

## Highlights & Insights
- **First direct empirical validation of the "Causal Parrot" hypothesis using pre-training corpus search**: While previously only conjectured, this paper provides quantitative evidence. The combination of open OLMo/BLOOM corpora and search tools served as key enablers.
- **Exquisitely designed synthetic data control experiments**: Using fictitious terms eliminated the confounding factor of "models potentially learning causal knowledge from other sources," achieving true causal inference.
- **The immense impact of context is an important warning**: The outcomes of the same causal pair differ dramatically under different contexts, showing that context-free LLM causal discovery results are untrustworthy. Future work should treat context as a necessary input.

## Limitations & Future Work
- The search tools cannot cover all ways causal relationships are mentioned in the pre-training corpus (e.g., implicit causal expressions), meaning the observed frequency is an underestimate.
- Only 7B open-source models were tested; larger-scale models may exhibit different memorization characteristics.
- Synthetic data experiments used LoRA fine-tuning instead of full pre-training, which might not fully simulate learning behaviors during the pre-training phase.
- Mitigation strategies for memorization issues in practice (e.g., integration with statistical methods) were not explored.

## Related Work & Insights
- **vs Kıcıman et al. (2023)**: Previously claiming GPT-4 outperforms statistical methods in causal discovery, this work shows this is mainly because the causal relationships in the test sets occurred with high frequency in the pre-training data, suggesting limited generalization ability.
- **vs Zečević et al. (2023)**: Proposed the "Causal Parrot" conjecture without empirical proof; this paper provides quantitative validation.
- **vs Traditional Statistical Causal Discovery Methods**: Although LLMs perform well on known relationships, their ability to discover novel relationships is limited and cannot replace statistical methods. Synergy between the two could be a promising future direction.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First empirical validation of the causal parrot hypothesis using pre-training dataset search, filling an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Both real and synthetic data experiments for each of the three RQs, across multiple models and datasets, with rigorous correlation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-organized by research questions, and rich in tables and figures.
- **Value**: ⭐⭐⭐⭐ Provides crucial insights into the limitations of LLM causal reasoning capabilities, warning against blindly trusting LLMs for causal discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](../../NeurIPS2025/causal_inference/revealing_multimodal_causality_with_large_language_models.md)
- [\[ACL 2025\] Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models](counterfactual-consistency_prompting_for_relative_temporal_understanding_in_larg.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[ACL 2026\] Evaluating Counterfactual Strategic Reasoning in Large Language Models](../../ACL2026/causal_inference/evaluating_counterfactual_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2025\] IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery](iris_an_iterative_and_integrated_framework.md)

</div>

<!-- RELATED:END -->
