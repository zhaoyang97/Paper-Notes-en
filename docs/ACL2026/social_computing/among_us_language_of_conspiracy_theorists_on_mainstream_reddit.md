---
title: >-
  [Paper Note] Among Us: Language of Conspiracy Theorists on Mainstream Reddit
description: >-
  [ACL 2026][Social Computing][conspiracy theories] Analyzing 500 million Reddit comments over 10 years of longitudinal data, this study finds that users active in conspiracy communities exhibit detectable unique linguistic patterns in mainstream communities (average 87% classification accuracy), but these patterns are highly context-dependent—community-specific models outperform global models by up to 17 percentage points.
tags:
  - ACL 2026
  - Social Computing
  - conspiracy theories
  - linguistic features
  - Reddit analysis
  - psycholinguistics
  - community adaptability
date: 2026-05-08
content_hash: d1e601467bf457c4
---

# Among Us: Language of Conspiracy Theorists on Mainstream Reddit

**Conference**: ACL 2026  
**arXiv**: [2506.05086](https://arxiv.org/abs/2506.05086)  
**Code**: N/A  
**Area**: Social Computing  
**Keywords**: conspiracy theory, language features, Reddit analysis, psycholinguistics, community adaptation

## TL;DR

Analyzing 500 million Reddit comments over 10 years of longitudinal data, this study finds that users active in conspiracy theory communities exhibit detectable unique language patterns in mainstream communities (average 87% classification accuracy), but these patterns are highly context-dependent, with community-specific models outperforming global models by up to 17 percentage points.

## Background & Motivation

**State of the Field**: Conspiracy theories are not merely fringe beliefs—they are associated with vaccine hesitancy, public health risks, and even threats to democratic institutions (e.g., the 2021 Capitol incident). Existing research primarily focuses on detecting conspiracy theory content while neglecting how conspiracy theory believers express themselves in mainstream spaces.

**Limitations of Prior Work**: (1) It is known that conspiracy theorists use specific rhetorical styles and vocabulary, but it remains unclear whether these patterns are confined to conspiracy spaces or permeate mainstream communication; (2) Existing detection methods mostly focus on content-level features (e.g., topic words), neglecting language style features independent of discussion topics.

**Root Cause**: Do conspiracy theorists possess a "conspiracy mindset" (monological worldview) that pervades all their communication, or can they fully adapt to different communities' linguistic norms?

**Paper Goals**: Systematically examine the linguistic distinguishability of conspiracy community users in mainstream spaces using large-scale longitudinal data.

**Starting Point**: Use LIWC-22 psycholinguistic features (rather than topic words) to construct user language profiles, training separate classifiers for each of 22 mainstream communities.

**Core Idea**: Conspiracy theory users' language is indeed distinguishable, but the distinguishing patterns are highly community-dependent—no single global model can capture these patterns, requiring community-specific analysis.

## Method

### Overall Architecture

Data collection (all r/conspiracy comments + 22 mainstream communities) → LIWC-22 feature extraction (110 dimensions) → user-level feature aggregation → Random Forest classifier trained per community → feature importance analysis (SHAP values) → cross-community similarity analysis.

### Key Designs

1. **Large-Scale Longitudinal Data Construction**:

    - Function: Provide reliable long-term language behavior data
    - Mechanism: Extract approximately 510 million comments from 2013–2023 from the Pushshift Reddit dataset, covering 980,000 users of r/conspiracy and 22 mainstream communities. Exclude bots and low-activity users (<20 comments)
    - Design Motivation: Sufficient comment volume is needed to construct stable user language profiles; short-term or sparse data may be dominated by noise

2. **Community-Specific Classification Experiments**:

    - Function: Test whether linguistic distinguishability is consistent across communities
    - Mechanism: Train an independent Random Forest for each mainstream community; positive class = users who have commented in r/conspiracy, negative class = randomly sampled equal-sized ordinary users. Repeat 5 times with random sampling to reduce variance. Permutation tests verify statistical significance
    - Design Motivation: Classifiers are not the goal itself but serve as proxy tools for quantifying linguistic distinguishability

3. **SHAP Feature Importance Analysis and Cross-Community Clustering**:

    - Function: Reveal which language features are most discriminative and whether distinguishing patterns are similar across communities
    - Mechanism: Compute SHAP values for each community model to obtain 110-dimensional feature importance vectors, then use cosine similarity + hierarchical clustering to analyze cross-community pattern similarity
    - Design Motivation: If all communities use the same features for discrimination, there exists a global "conspiracy language"; if features vary by community, language expression is context-adaptive

### Loss & Training

Random Forest with grid search and 5-fold cross-validation for hyperparameter tuning, 80/20 train-test split. Feature normalization performed only on training data. 100 permutation tests evaluate statistical significance.

## Key Experimental Results

### Main Results

| Metric | Value | Note |
|--------|-------|------|
| Average classification accuracy | 87% | Binary classification across 20+ communities |
| Community-specific vs. global | Up to +17pp | Community-specific models significantly outperform global models |
| Statistical significance | p<0.01 | Permutation tests significant for all communities |

### Ablation Study

| Config | Key Metric | Note |
|--------|-----------|------|
| Activity threshold | Higher activity performs better | More comments → more stable language profiles |
| r/AskReddit as positive | Accuracy ~random | General community users cannot be distinguished (negative control) |
| r/MensRights as positive | Medium accuracy | Ideological communities also show partial discriminability |

### Key Findings

- Conspiracy theory users' language is indeed detectable in mainstream spaces—average 87% accuracy far exceeds random
- But no single global model captures these patterns—community-specific models outperform global models by up to 17 percentage points
- This indicates conspiracy theory users' language expression is dynamically adaptive—while they have distinctive features, they adjust according to community norms
- r/AskReddit users (negative control) cannot be distinguished, validating effect specificity

## Highlights & Insights

- "Distinguishable but context-dependent" is an elegant finding—supporting the existence of a "conspiracy mindset" while showing it is not a simple global label
- Direct implications for content moderation strategies—unified detection models are insufficient; community-customized approaches are needed
- Using LIWC psycholinguistic features (rather than topic words) ensures analysis targets language style rather than discussion content

## Limitations & Future Work

- Equating "having commented in r/conspiracy" with "conspiracy theory believer" may be overly broad
- LIWC's dictionary-based approach may miss emerging language patterns
- Only analyzes the Reddit platform; patterns on other social media may differ
- Future work could combine content analysis and style analysis for more fine-grained research

## Related Work & Insights

- **vs Content detection methods**: Focuses on language style rather than content, revealing deeper cognitive traits
- **vs User pathway studies**: Rather than tracking how users enter conspiracy communities, analyzes their behavior in mainstream spaces
- **vs Community detection**: Reveals cross-community behavioral adaptability, complementing community boundary research

## Rating

- Novelty: ⭐⭐⭐⭐ Studies conspiracy theory users' cross-community behavior from a language style perspective
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 500 million comments, 10-year longitudinal data, 22 communities, statistical tests
- Writing Quality: ⭐⭐⭐⭐ Rigorous research design with comprehensive negative controls
- Value: ⭐⭐⭐⭐ Practical guidance for social media governance and conspiracy theory research

## Related Papers

- [\[ACL 2025\] Conspiracy Theories and Where to Find Them on TikTok](../../ACL2025/social_computing/conspiracy_theories_and_where_to_find_them_on_tiktok.md)
- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)
- [\[ACL 2026\] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects](how_language_models_conflate_logical_validity_with_plausibility_a_representation.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](../../ICLR2026/social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)
---
title: >-
  [Paper Note] Among Us: Language of Conspiracy Theorists on Mainstream Reddit
description: >-
  [ACL 2026][Social Computing] Analyzing 500 million Reddit comments over 10 years of longitudinal data, this study finds that users active in conspiracy communities exhibit detectable unique linguistic patterns in mainstream communities (average 87% classification accuracy), but these patterns are highly context-dependent—community-specific models outperform global models by up to 17 percentage points.
tags:
  - ACL 2026
  - Social Computing
  - Conspiracy Theories
  - Linguistic Features
  - Reddit Analysis
  - Psycholinguistics
date: 2026-05-08
content_hash: d1e601467bf457c4
---

# Among Us: Language of Conspiracy Theorists on Mainstream Reddit

**Conference**: ACL 2026  
**arXiv**: [2506.05086](https://arxiv.org/abs/2506.05086)  
**Code**: N/A  
**Area**: Social Computing  
**Keywords**: conspiracy theories, linguistic features, Reddit analysis, psycholinguistics, community adaptability

## TL;DR

Analyzing 500 million Reddit comments over 10 years of longitudinal data, this study finds that users active in conspiracy communities exhibit detectable unique linguistic patterns in mainstream communities (average 87% classification accuracy), but these patterns are highly context-dependent—community-specific models outperform global models by up to 17 percentage points.

## Background & Motivation

**State of the Field**: Conspiracy theories are not merely fringe beliefs—they are linked to vaccine hesitancy, public health risks, and even threats to democratic institutions (e.g., the 2021 Capitol incident). Existing research primarily focuses on detecting conspiracy content but ignores how conspiracy believers express themselves in mainstream spaces.

**Limitations of Prior Work**: (1) It is known that conspiracy theorists use specific rhetorical styles and vocabulary, but it is unclear whether these patterns are confined to conspiracy spaces or permeate mainstream communication; (2) Existing detection methods mostly focus on content-level features (e.g., topic words) while ignoring stylistic linguistic features independent of discussion topics.

**Root Cause**: Do conspiracy theorists possess a "conspiratorial mindset" (monological worldview) that permeates all their communication, or can they fully adapt to different communities' linguistic norms?

**Paper Goals**: Use large-scale longitudinal data to systematically test the linguistic distinguishability of conspiracy community users in mainstream spaces.

**Starting Point**: Use LIWC-22 psycholinguistic features (rather than topic words) to build user linguistic profiles, training separate classifiers on 22 mainstream communities.

**Core Idea**: Conspiracy users' language is indeed distinguishable, but the distinguishing patterns are highly community-dependent—no single global model can capture these patterns; community-specific analysis is required.

## Method

### Overall Architecture

Data collection (all r/conspiracy comments + 22 mainstream communities) → LIWC-22 feature extraction (110 dimensions) → User-level feature aggregation → Random Forest classifier training per community → Feature importance analysis (SHAP values) → Cross-community similarity analysis.

### Key Designs

1. **Large-Scale Longitudinal Data Construction**:

    - Function: Provide reliable long-term linguistic behavior data
    - Mechanism: Extract approximately 510 million comments from the Pushshift Reddit dataset spanning 2013–2023, covering 980K users of r/conspiracy and 22 mainstream communities. Bots and low-activity users (<20 comments) are excluded
    - Design Motivation: Sufficient comment volume is needed to build stable user linguistic profiles; short-term or sparse data may be dominated by noise

2. **Community-Specific Classification Experiments**:

    - Function: Test whether linguistic distinguishability is consistent across communities
    - Mechanism: Train independent Random Forest classifiers for each mainstream community, with positive class = users who have commented in r/conspiracy, negative class = randomly sampled equal-sized ordinary users. Repeat 5 times with random sampling to reduce variance. Permutation tests validate statistical significance
    - Design Motivation: Classifiers are not the end goal but rather proxy tools to quantify linguistic distinguishability

3. **SHAP Feature Importance Analysis and Cross-Community Clustering**:

    - Function: Reveal which linguistic features are most discriminative and whether discrimination patterns are similar across communities
    - Mechanism: Compute SHAP values for each community model to obtain 110-dimensional feature importance vectors, then analyze cross-community pattern similarity using cosine similarity + hierarchical clustering
    - Design Motivation: If all communities use the same features for discrimination, there exists a global "conspiracy language"; if features vary by community, linguistic expression is context-adaptive

### Loss & Training

Random Forest with grid search and 5-fold cross-validation for hyperparameter tuning, 80/20 train-test split. Feature normalization performed only on training data. 100 permutation tests evaluate statistical significance.

## Key Experimental Results

### Main Results

| Metric | Value | Note |
|--------|-------|------|
| Average classification accuracy | 87% | Binary classification across 20+ communities |
| Community-specific vs. global | Up to +17pp | Community-specific models significantly outperform global models |
| Statistical significance | p<0.01 | Permutation tests significant for all communities |

### Ablation Study

| Config | Key Metric | Note |
|--------|-----------|------|
| Activity threshold | Higher activity → better results | More comments → more stable linguistic profiles |
| r/AskReddit as positive | Accuracy ~random | General community users cannot be distinguished (negative control) |
| r/MensRights as positive | Medium accuracy | Ideological communities also have some discriminative power |

### Key Findings

- Conspiracy users' language is indeed detectable in mainstream spaces—87% average accuracy far exceeds random
- No single global model can capture these patterns—community-specific models outperform global models by up to 17 percentage points
- This indicates conspiracy users' linguistic expression is dynamically adaptive—while exhibiting distinctive features, they adjust according to community norms
- r/AskReddit users (negative control) cannot be distinguished, validating the specificity of the effect

## Highlights & Insights

- "Distinguishable but context-dependent" is a nuanced finding—it both supports the existence of a "conspiratorial mindset" and shows it is not a simple global label
- Direct implications for content moderation strategies—unified detection models are insufficient; community-customized approaches are needed
- Using LIWC psycholinguistic features (rather than topic words) ensures analysis targets linguistic style rather than discussion content

## Limitations & Future Work

- Equating "having commented in r/conspiracy" with "being a conspiracy believer" may be overly broad
- LIWC's dictionary-based approach may miss emerging linguistic patterns
- Only one platform (Reddit) is analyzed; patterns on other social media may differ
- Future work can combine content analysis and style analysis for more fine-grained research

## Related Work & Insights

- **vs Content detection methods**: Focuses on linguistic style rather than content, revealing deeper cognitive characteristics
- **vs User pathway research**: Does not track how users enter conspiracy communities, but analyzes their behavior in mainstream spaces
- **vs Community detection**: Reveals cross-community behavioral adaptability, complementing community boundary research

## Rating

- Novelty: ⭐⭐⭐⭐ Studies conspiracy users' cross-community behavior from a linguistic style perspective
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 500 million comments, 10-year longitudinal data, 22 communities, statistical testing
- Writing Quality: ⭐⭐⭐⭐ Rigorous research design with thorough negative controls
- Value: ⭐⭐⭐⭐ Practical guidance for social media governance and conspiracy theory research

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)
- [\[ACL 2026\] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects](how_language_models_conflate_logical_validity_with_plausibility_a_representation.md)
- [\[NeurIPS 2025\] SLAyiNG: Towards Queer Language Processing](../../NeurIPS2025/social_computing/slaying_towards_queer_language_processing.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](../../ICLR2026/social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)

<!-- RELATED:END -->
