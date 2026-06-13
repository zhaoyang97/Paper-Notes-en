---
title: >-
  [Paper Note] Among Us: Language of Conspiracy Theorists on Mainstream Reddit
description: >-
  [ACL 2026][Social Computing][Conspiracy theories] By analyzing 10 years of longitudinal data encompassing 500 million Reddit comments, this study finds that users active in conspiracy communities exhibit detectable and d…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Conspiracy theories"
  - "linguistic features"
  - "Reddit analysis"
  - "psycholinguistics"
  - "community adaptation"
date: 2026-05-08
content_hash: 150870615b6d41f2
---

# Among Us: Language of Conspiracy Theorists on Mainstream Reddit

**Conference**: ACL 2026  
**arXiv**: [2506.05086](https://arxiv.org/abs/2506.05086)  
**Code**: None  
**Area**: Social Computing / Computational Linguistics  
**Keywords**: Conspiracy theories, linguistic features, Reddit analysis, psycholinguistics, community adaptation

## TL;DR

By analyzing 10 years of longitudinal data encompassing 500 million Reddit comments, this study finds that users active in conspiracy communities exhibit detectable and distinct linguistic patterns even within mainstream communities (average classification accuracy of 87%). However, these patterns are highly dependent on the community context, with community-specific models outperforming global models by up to 17 percentage points.

## Background & Motivation

**Background**: Conspiracy theories are more than just fringe beliefs—they are linked to vaccine hesitancy, public health risks, and even threats to democratic institutions (e.g., the January 6 Capitol riot). Existing research primarily focuses on the detection of conspiracy content but overlooks the linguistic behavior of conspiracy believers within mainstream spaces.

**Limitations of Prior Work**: (1) While conspiracy theorists are known to use specific rhetorical styles and vocabularies, it remains unclear whether these patterns are confined to conspiracy-themed spaces or if they permeate mainstream communication. (2) Existing detection methods mostly focus on the content level (e.g., topic keywords), ignoring linguistic style features that are independent of the discussion topic.

**Key Challenge**: Do conspiracy theorists possess a "monological worldview" that infiltrates all their communications, or can they fully adapt to the linguistic norms of different communities?

**Goal**: To systematically examine the linguistic distinguishability of conspiracy community users in mainstream spaces using large-scale longitudinal data.

**Key Insight**: Build linguistic profiles of users using LIWC-22 psycholinguistic features (rather than topic-specific words) and train individual classifiers for 22 mainstream communities.

**Core Idea**: The language of conspiracy users is indeed distinguishable, but the patterns of distinction are highly community-dependent—no single global model can capture these patterns, necessitating community-specific analysis.

## Method

### Overall Architecture

Data Collection (All r/conspiracy comments + 22 mainstream communities) → LIWC-22 Feature Extraction (110 dimensions) → User-level Feature Aggregation → Training Random Forest Classifiers for each community → Feature Importance Analysis (SHAP values) → Cross-community Similarity Analysis.

### Key Designs

1.  **Large-scale Longitudinal Data Construction**:
    *   **Function**: Provides reliable data on long-term linguistic behavior.
    *   **Mechanism**: Approximately 510 million comments from 2013-2023 were extracted from the Pushshift Reddit dataset, covering 980,000 users from r/conspiracy and 22 mainstream communities. Bots and low-activity users (<20 comments) were excluded.
    *   **Design Motivation**: A sufficient volume of comments is required to build stable user linguistic profiles; short-term or sparse data might be dominated by noise.

2.  **Community-Specific Classification Experiments**:
    *   **Function**: Tests whether linguistic distinguishability is consistent across communities.
    *   **Mechanism**: A Random Forest classifier was trained independently for each mainstream community. The positive class consisted of users who had commented in r/conspiracy, while the negative class consisted of an equal number of randomly sampled regular users. Random sampling was repeated 5 times to reduce variance, and permutation tests were used for statistical significance.
    *   **Design Motivation**: The classifier is not an end in itself but serves as a proxy tool to quantify linguistic distinguishability.

3.  **SHAP Feature Importance and Cross-community Clustering**:
    *   **Function**: Reveals which linguistic features are most discriminative and whether patterns are similar across communities.
    *   **Mechanism**: SHAP values were calculated for each community model to generate 110-dimensional feature importance vectors, followed by analysis using cosine similarity and hierarchical clustering.
    *   **Design Motivation**: If all communities were distinguished by the same features, it would suggest a global "conspiracy language." Distinct features across communities would indicate that linguistic expression is context-adaptive.

### Loss & Training

Random Forest models were tuned using grid search and 5-fold cross-validation, with an 80/20 train-test split. Feature normalization was performed on the training data only. Statistical significance was evaluated using 100 permutation tests.

## Key Experimental Results

### Main Results

| Metric | Value | Description |
| :--- | :--- | :--- |
| Avg. Classification Acc | 87% | Binary classification across 20+ communities |
| Community-specific vs. Global | Up to +17pp | Community-specific models significantly outperform global ones |
| Statistical Significance | $p < 0.01$ | Permutation tests were significant for all communities |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Activity Threshold | Better performance with high activity | More comments $\rightarrow$ more stable linguistic profiles |
| r/AskReddit Positive Class | Acc $\approx$ Random | General community users are indistinguishable (Negative control) |
| r/MensRights Positive Class | Moderate Accuracy | Ideological communities also show some level of distinguishability |

### Key Findings
*   The language of conspiracy users is indeed detectable in mainstream spaces—the average 87% accuracy far exceeds random chance.
*   However, no single global model can capture these patterns—community-specific models perform up to 17 percentage points better.
*   This suggests that the linguistic expression of conspiracy users is dynamically adaptive—while unique features exist, they are adjusted based on community norms.
*   The inability to distinguish r/AskReddit users (negative control) validates the specificity of the observed effects.

## Highlights & Insights
*   The "distinguishable yet context-dependent" finding is nuanced—it supports the existence of a "monological worldview" while showing it is not a simple global label.
*   The results have direct implications for content moderation strategies—uniform detection models are insufficient, and community-tailored approaches are necessary.
*   The use of LIWC psycholinguistic features (rather than topic words) ensures that the analysis captures linguistic style rather than mere discussion content.

## Limitations & Future Work
*   Equating "having commented in r/conspiracy" with being a "conspiracy believer" might be overly broad.
*   The dictionary-based approach of LIWC might miss emerging linguistic patterns.
*   The analysis is limited to Reddit; patterns on other social media platforms might differ.
*   Future research could combine content and stylistic analysis for more fine-grained studies.

## Related Work & Insights
*   **vs. Content Detection Methods**: Focuses on linguistic style rather than content, revealing deeper cognitive traits.
*   **vs. User Pathway Research**: Instead of tracking how users enter conspiracy communities, it analyzes their behavior within mainstream spaces.
*   **vs. Community Detection**: Reveals behavioral adaptability across communities, supplementing research on community boundaries.

## Rating
*   **Novelty**: ⭐⭐⭐⭐ Investigates cross-community behavior of conspiracy users from a linguistic style perspective.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 500M comments, 10-year longitudinal scope, 22 communities, and rigorous statistical testing.
*   **Writing Quality**: ⭐⭐⭐⭐ Rigorous research design with well-implemented negative controls.
*   **Value**: ⭐⭐⭐⭐ Provides practical guidance for social media governance and conspiracy theory research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Inertia in Moral and Value Judgments of Large Language Models](inertia_in_moral_and_value_judgments_of_large_language_models.md)
- [\[ACL 2026\] Bayesian Social Deduction with Graph-Informed Language Models](bayesian_social_deduction_with_graph-informed_language_models.md)
- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)
- [\[ACL 2026\] Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation](probing_multimodal_large_language_models_on_cognitive_biases_in_chinese_short-vi.md)
- [\[NeurIPS 2025\] SLAyiNG: Towards Queer Language Processing](../../NeurIPS2025/social_computing/slaying_towards_queer_language_processing.md)

</div>

<!-- RELATED:END -->
