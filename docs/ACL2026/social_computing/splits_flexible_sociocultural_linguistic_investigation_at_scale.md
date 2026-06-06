---
title: >-
  [Paper Note] Splits! Flexible Sociocultural Linguistic Investigation at Scale
description: >-
  [ACL 2026][Social Computing][Sociocultural linguistic phenomena] This paper proposes a method to construct a sociolinguistic "sandbox" by building the Splits! dataset from Reddit…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Sociocultural linguistic phenomena"
  - "Reddit dataset"
  - "hypothesis filtering"
  - "lexical analysis"
  - "demographics"
date: 2026-05-08
content_hash: 6114fccd1478672b
---

# Splits! Flexible Sociocultural Linguistic Investigation at Scale

**Conference**: ACL 2026  
**arXiv**: [2504.04640](https://arxiv.org/abs/2504.04640)  
**Code**: [GitHub](https://github.com/ecaplan/splits) (Code + Data + Demo)  
**Area**: Sociolinguistics / Computational Social Science  
**Keywords**: Sociocultural linguistic phenomena, Reddit dataset, hypothesis filtering, lexical analysis, demographics

## TL;DR
This paper proposes a method to construct a sociolinguistic "sandbox" by building the Splits! dataset from Reddit, containing 9.7 million posts split by demographic groups and discussion topics. It designs a two-stage filtering pipeline based on lift and triviality to efficiently select noteworthy sociocultural linguistic phenomena from 23,000 LLM-generated candidate hypotheses.

## Background & Motivation

**Background**: Computational social science studies language usage differences across groups (e.g., AAVE code-switching, Yiddish lexicons in Jewish English) using social media data. However, such research typically requires customized data collection and experimental designs for specific groups/topics, making it costly and difficult to prototype rapidly.

**Limitations of Prior Work**: Researchers must invest significant effort to validate a single sociolinguistic hypothesis. While automated hypothesis generation (e.g., via LLMs) can produce many candidates, the critical bottleneck is efficiently identifying those truly worthy of in-depth study among thousands of machine-generated options. Many statistically significant hypotheses are actually trivial (e.g., "Jewish people mention Judaism more frequently").

**Key Challenge**: Statistical significance $\neq$ research value. Many trivial hypotheses achieve high significance through data validation but offer no insight into social science. An automated method is needed to distinguish between "statistically valid" and "academically interesting."

**Goal**: (1) Construct a flexible sociolinguistic exploration "sandbox" dataset; (2) design an automated filtering pipeline to distinguish between interesting and trivial hypotheses.

**Key Insight**: Sociocultural linguistic phenomena (SLP) are formalized as "Group A uses lexicon L more than Group B when discussing topic t." BM25 retrieval and lift metrics quantify statistical validity, while semantic similarity quantifies triviality.

**Core Idea**: Construct the sandbox using a dual split of demographics $\times$ topics, and filter non-trivial valid hypotheses from large candidate sets using a two-stage lift + triviality process.

## Method

### Overall Architecture
The framework consists of two main parts: (1) Dataset construction—building the Splits! dataset through a pipeline of seed subreddit $\rightarrow$ seed user $\rightarrow$ site-wide post collection $\rightarrow$ topic labeling, split by 6 demographic groups $\times$ 89 topics; (2) Hypothesis filtering—calculating lift (data support) and triviality (obviousness) for candidate hypotheses to filter for valuable ones.

### Key Designs

1.  **Group-ness Metric and Demographic Verification**:
    - **Function**: Ensures collected posts originate from the target demographic groups.
    - **Mechanism**: Defines $\text{group-ness}(u) = \sum_{s \in SD} \log(1 + c_{u,s})$, where $c_{u,s}$ is the number of posts by user $u$ in seed subreddit $s$. This metric rewards both total post count and diversity across subreddits. High group-ness users are verified using self-identification phrases (e.g., "I am Catholic").
    - **Design Motivation**: Inferring group identity solely based on subreddit membership introduces noise; quantitative metrics are needed to select high-confidence users.

2.  **Lift Metric for Quantifying Hypothesis Validity**:
    - **Function**: Measures the statistical power of a lexicon L in distinguishing two groups.
    - **Mechanism**: Posts from two groups under the same topic are merged into a BM25 index. Lexicon L is used as a query to rerank posts, calculating $\text{lift}@p\% = \frac{\#A \text{ posts}@p\% / \# \text{posts}@p\%}{\#A \text{ posts overall} / \# \text{posts overall}}$. A Lift > 1 indicates the lexicon successfully ranks the target group higher. Hypergeometric tests ensure statistical significance.
    - **Design Motivation**: Lift is a mature association measure in data mining, more robust than simple frequency comparisons, and can capture phenomena at different granularities by adjusting $p\%$.

3.  **Triviality Metric for Filtering Obvious Hypotheses**:
    - **Function**: Automatically determines if a hypothesis is trivial (e.g., "Jews mentioning Judaism").
    - **Mechanism**: A small "defining lexicon" $\ell_A$ (5-10 words) is manually curated for each group. The subspace recall $R_{subspace}(L, \ell_A)$ between the hypothesis lexicon L and $\ell_A$ is calculated. Higher scores indicate the lexicon is closer to the group definition and thus more trivial. The Spearman correlation with human "unexpectedness" ratings is $\rho = -0.38$, validating the metric.
    - **Design Motivation**: Statistical significance correlates positively with triviality (Spearman 0.32), meaning reliance on statistical tests alone leads to being overwhelmed by trivial hypotheses.

### Loss & Training
No model training is involved. Dataset construction utilizes the ColBERT retrieval model and LLM-assisted topic classification. Hypothesis filtering uses BM25 and semantic similarity in BERT embedding space.

## Key Experimental Results

### Main Results
Validation of known sociolinguistic phenomena:

| Phenomenon | Target Group | Topic | Usage (Target) | Usage (Control) | p-value |
|------------|--------------|-------|----------------|-----------------|---------|
| AAVE usage | Black | Hip-Hop | 3.16% | 2.00% | <0.001 |
| AAVE code-switching | Black | Job→Hip-Hop | 0.33%→3.16% | - | <0.001 |
| Yiddish usage | Jewish | Judaism | 0.19% | 0.07% | <0.001 |
| Dance identity | Hindu/Sikh/Jain | Cultural Identity | 0.44% | 0.36% | <0.001 |

### Ablation Study
Analysis of two-stage filtering efficiency:

| Triviality Percentile Threshold | Precision | Recall | F1 | Efficiency Gain |
|-------------------------------|-----------|--------|----|-----------------|
| Baseline (p-value only) | 0.270 | 1.000 | 0.425 | 1.00× |
| 0.3 | 0.447 | 0.496 | 0.470 | 1.65× |
| 0.5 | 0.398 | 0.741 | 0.518 | 1.47× |

### Key Findings
- Two-stage filtering achieves a total efficiency gain of 15-18×: stage one (statistical filtering) reduces candidates by 10×, and stage two (triviality filtering) reduces them further by 1.5-1.8×.
- Hypotheses from academic literature show significantly lower triviality distributions (mean 0.585 vs. 0.810 for LLM-generated ones), validating the metric's alignment with "academic interest."
- An interesting non-trivial discovery: Jewish users use words related to "preventative care" and "early detection" more frequently when discussing health, potentially reflecting cultural values emphasizing worldly concerns.

## Highlights & Insights
- The insight that "statistical significance $\neq$ research value" is intuitively obvious, but this paper is the first to quantitatively prove the positive correlation between the two (Spearman 0.32) and provide a systematic solution. This issue is pervasive in computational social science.
- The "sandbox" concept is highly practical—researchers can test new hypotheses at zero marginal cost; providing a lexicon yields immediate cross-group and cross-topic analysis results.
- The methodology of Group-ness metrics + self-identification verification provides a reproducible paradigm for demographic inference on social media.

## Limitations & Future Work
- Data is limited to Reddit (2012-2018), entailing platform and temporal biases.
- Only covers 6 demographic groups and is primarily English-focused.
- The Group-ness method favors users highly active in identity communities, which may amplify linguistic differences between groups.
- Analysis is limited to lexical differences, excluding syntax, semantic frames, or pragmatic features.
- Intersectional identities (e.g., Black and Catholic) were not analyzed.

## Related Work & Insights
- **vs. Traditional Sociolinguistics**: Traditional methods rely on fieldwork and deep ethnography, which are costly but offer deep insights; Splits! provides large-scale hypothesis screening as a complement.
- **vs. LLM Hypothesis Generation**: Works like Yang et al. (2024) focus on LLM hypothesis generation but lack mechanisms to filter for valuable ones; this paper fills that gap.

## Rating
- Novelty: ⭐⭐⭐⭐ The sandbox concept and triviality filtering are innovative contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Replication of 5 known phenomena + filtering 23k candidate hypotheses + manual annotation verification.
- Writing Quality: ⭐⭐⭐⭐ Methodology is clearly described, though the paper is long and requires careful reading.
- Value: ⭐⭐⭐⭐ Provides reproducible methodology and data resources for computational social science.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] As Language Models Scale, Low-order Linear Depth Dynamics Emerge](../../CVPR2026/social_computing/as_language_models_scale_low-order_linear_depth_dynamics_emerge.md)
- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)
- [\[ACL 2026\] Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation](beyond_the_crowd_llm-augmented_community_notes_for_governing_health_misinformati.md)
- [\[ACL 2026\] Explain the Flag: Contextualizing Hate Speech Beyond Censorship](explain_the_flag_contextualizing_hate_speech_beyond_censorship.md)
- [\[ACL 2026\] Is this chart lying to me? Automating the detection of misleading visualizations](is_this_chart_lying_to_me_automating_the_detection_of_misleading_visualizations.md)

</div>

<!-- RELATED:END -->
