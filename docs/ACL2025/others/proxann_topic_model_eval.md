---
title: >-
  [Paper Note] ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering
description: >-
  [ACL 2025][topic model evaluation] This work proposes ProxAnn, a use-oriented evaluation protocol for topic models. By combining a scalable human evaluation pipeline with LLM proxy annotators, the study finds that the best LLM proxies are statistically indistinguishable from human annotators, serving as a reasonable alternative for automated evaluation.
tags:
  - "ACL 2025"
  - "topic model evaluation"
  - "proxy annotator"
  - "LLM evaluation"
  - "document clustering"
  - "qualitative content analysis"
date: 2026-05-08
content_hash: aa9569615753f074
---

# ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering

**Conference**: ACL 2025  
**arXiv**: [2507.00828](https://arxiv.org/abs/2507.00828)  
**Code**: [github.com/ahoho/proxann](https://github.com/ahoho/proxann)  
**Area**: Other  
**Keywords**: topic model evaluation, proxy annotator, LLM evaluation, document clustering, qualitative content analysis  

## TL;DR

This work proposes ProxAnn, a use-oriented evaluation protocol for topic models. By combining a scalable human evaluation pipeline with LLM proxy annotators, the study finds that the best LLM proxies are statistically indistinguishable from human annotators, serving as a reasonable alternative for automated evaluation.

## Background & Motivation

**Problem Definition:** The evaluation of topic models and document clustering methods either relies on automated metrics (such as NPMI) that align poorly with human preferences or is dependent on expert annotations that are difficult to scale. There is a need for an evaluation method that both reflects actual use cases and scales automatically.

**Limitations of Prior Work:** Currently, mainstream evaluations focus on semantic topic coherence. However, analyzing only the top words is insufficient to verify whether the model outputs are effective—highly coherent topic words do not guarantee that the document-topic assignments are reasonable. Automated metrics like NPMI have been shown to correlate poorly with human judgment (Hoyle et al., 2021). Human-annotation-based evaluation methods (e.g., Ying et al., 2022) are expensive and difficult to replicate.

**Design Motivation:** Effective evaluation should approximate real-use scenarios. In qualitative content analysis (QCA), a core use case of topic models, practitioners first inductively derive categories from the data and then apply these categories to new documents. The evaluation should simulate this process, and LLMs show potential as a scalable alternative to human annotators.

## Method

### Overall Architecture

The evaluation protocol consists of three steps, with two parallel implementation schemes designed for humans and LLMs:

1. **Label Step (Category Identification):** Annotators examine sample documents and keywords from a specific topic/cluster to infer a semantic category label for that group.
2. **Fit Step (Relevance Assessment):** Annotators rate 7 additional evaluation documents one by one (1-5 scale) to judge their alignment with the inferred category.
3. **Rank Step (Representativeness Ranking):** Annotators rank the evaluation documents based on their representativeness of the category.

LLMs execute the same tasks using condensed prompts and identical sample documents. The Fit Step uses token-probability-weighted means, and the Rank Step uses pairwise comparisons combined with the Bradley-Terry model.

### Key Designs

1. **Stratified Document Sampling:** Evaluation documents are stratified-sampled from the document-topic probability distribution $\theta_k^{(r)}$ instead of only obtaining the top documents, ensuring coverage across high, medium, and low probability intervals. Each set includes a near-zero probability document as a control.
2. **Alternative Annotator Test:** Using the alt-test statistical test by Calderon et al. (2025), the LLM's probability of superiority $\rho$ as a "proxy annotator" is calculated to determine whether the LLM is statistically indistinguishable from human annotators. This compares LLM-human agreement against human-human agreement using a leave-one-out approach.
3. **Multi-Model Coverage:** Human evaluation covers three methods (Mallet/LDA, CTM, BERTopic). LLM proxies include GPT-4o, Llama-3.1-8B/3.3-70B, Qwen-2.5-72B, and Qwen-3-8B/32B.

### Evaluation Metrics

- **Human-Human Agreement:** Krippendorff's $\alpha$ (ordinal-weighted)
- **Model-Annotator Correlation:** Kendall's $\tau$ (document-topic probability vs. human rating/ranking)
- **LLM Proxy Validation:** Probability of superiority $\rho$ + one-sided t-test / Wilcoxon signed-rank test

## Key Experimental Results

### Human-Human Agreement (Krippendorff's $\alpha$)

| Dataset | Model | Fit Step | Rank Step |
|--------|------|----------|-----------|
| Wiki | Mallet (LDA) | 0.71 | 0.74 |
| Wiki | CTM | 0.55 | 0.45 |
| Wiki | BERTopic | 0.57 | 0.44 |
| Bills | Mallet (LDA) | 0.31 | 0.49 |
| - | Label-Derived (Upper Bound) | 0.80 | 0.86 |

### LLM Proxy vs Human (Probability of Superiority $\rho$, document-level, Fit/Rank)

| LLM | Wiki Fit | Wiki Rank | Bills Fit | Bills Rank |
|-----|----------|-----------|-----------|------------|
| GPT-4o | 0.56† | 0.68† | 0.65† | 0.71† |
| Llama-3.1-8B | 0.22 | 0.36 | 0.30 | 0.53† |
| Llama-3.3-70B | 0.57† | 0.67† | 0.66† | 0.67† |
| Qwen-3-32B | 0.55† | 0.63† | 0.67† | 0.68† |
| Qwen-2.5-72B | 0.52† | 0.68† | 0.61† | 0.71† |

† indicates LLM probability of superiority is significantly $\ge 0.5$

### Key Findings

1. **Classical LDA Performs Best:** In human evaluations, the 20-year-old Mallet (LDA) consistently outperforms modern CTM and BERTopic in human-human agreement and human-model correlation, challenging the "newer is better" assumption.
2. **Large LLMs Can Substitute Human Annotators:** Models with $\ge 32\text{B}$ parameters, such as GPT-4o, Llama-3.3-70B, and Qwen-2.5-72B, exhibit a probability of superiority significantly $\ge 0.5$ in document-level alt-tests, making them statistically non-inferior to random human annotators.
3. **Small LLMs Are Unreliable:** Llama-3.1-8B is highly inconsistent with humans on most tasks.
4. **NPMI is Uncorrelated with Human Judgments:** The Kendall's $\tau$ between NPMI and human evaluation metrics is close to 0 or even negative, confirming its unreliability as a metric for topic quality.
5. **ProxAnn Evaluation vs. Human Evaluation Rankings are Consistent:** Topic rankings based on ProxAnn with larger LLMs achieve a level of agreement with human rankings that is comparable to leave-one-out human agreement.

## Highlights & Insights

- Anchors the evaluation in real-world use cases (qualitative content analysis) rather than unrealistic topic coherence.
- Features the largest human evaluation scale of its kind, covering three models $\times$ two datasets $\times$ $\ge 4$ annotators per topic.
- Validates LLM proxies through rigorous statistical testing (alt-test) rather than simple correlation comparisons.
- Open-sources all human and LLM annotation data along with the evaluation toolkit.
- Provides deep qualitative analyses, dissecting the causes of low-coherence topics (overly broad vs. mixed topics) and presenting specific case studies of LLM-human disagreements.

## Limitations & Future Work

- Human evaluation only covers 8 out of 50 topics (per model); the limited sample size restricts statistical power.
- Uses only two English datasets (Wiki and Bills); generalizability to other languages and domains remains unknown.
- LLM prompts were tuned on the Wiki pilot data, resulting in slightly worse performance on Bills (a more specialized domain).
- The LLM adaptation for the ranking task (pairwise comparison + Bradley-Terry) introduces additional approximation errors.
- Does not evaluate LLM-based topic models (e.g., Pham et al., 2024) due to their lack of a standard document-topic distribution.

## Related Work & Insights

- **Topic Coherence Evaluation:** NPMI (Lau et al., 2014), word intrusion (Chang et al., 2009), Ying et al. (2022) label validation.
- **LLM-based Evaluation:** Stammbach et al. (2023) and Rahimi et al. (2024) simulate word intrusion via LLMs; Yang et al. (2024) perform LLM keyword alignment.
- **Topic Models:** LDA/Mallet (Blei et al., 2003), CTM (Bianchi et al., 2021), BERTopic (Grootendorst, 2022).
- **Interactive Topic Modeling:** Poursabzi-Sangdeh et al. (2016), Li et al. (2024) evaluate under content analysis contexts.

## Rating

| Metric | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Using Shapley Interactions to Understand How Models Use Structure](using_shapley_interactions_to_understand_how_models_use_structure.md)
- [\[ACL 2025\] S2WTM: Spherical Sliced-Wasserstein Autoencoder for Topic Modeling](s2wtm_spherical_sliced-wasserstein_autoencoder_for_topic_modeling.md)
- [\[ACL 2025\] Understanding Cross-Domain Adaptation in Low-Resource Topic Modeling](understanding_cross-domain_adaptation_in_low-resource_topic_modeling.md)
- [\[ACL 2025\] SoRFT: Issue Resolving with Subtask-oriented Reinforced Fine-Tuning](sorft_issue_resolving_with_subtask-oriented_reinforced_fine-tuning.md)
- [\[ACL 2025\] Persistent Homology of Topic Networks for the Prediction of Reader Curiosity](persistent_homology_of_topic_networks_for_the_prediction_of_reader_curiosity.md)

</div>

<!-- RELATED:END -->
