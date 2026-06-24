---
title: >-
  [Paper Note] The Anatomy of Evidence: An Investigation Into Explainable ICD Coding
description: >-
  [ACL 2025][Interpretability][ICD Coding] This paper conducts an in-depth, application-oriented analysis of the MDACE dataset and current explainable ICD coding systems, revealing the overlap patterns between human-annotated evidence and code descriptions, the distributional characteristics of evidence within documents, and proposes new matching metrics to evaluate the utility of model explanations.
tags:
  - "ACL 2025"
  - "Interpretability"
  - "ICD Coding"
  - "Explainability"
  - "Evidence Extraction"
  - "MDACE Dataset"
  - "Feature Attribution"
date: 2026-05-08
content_hash: 3c87f99eb5dd69e9
---

# The Anatomy of Evidence: An Investigation Into Explainable ICD Coding

**Conference**: ACL 2025  
**arXiv**: [2507.01802](https://arxiv.org/abs/2507.01802)  
**Code**: Yes ([https://github.com/lamarr-xai-group/anatomy-of-evidence](https://github.com/lamarr-xai-group/anatomy-of-evidence))  
**Area**: Explainability  
**Keywords**: ICD Coding, Explainability, Evidence Extraction, MDACE Dataset, Feature Attribution

## TL;DR

This paper conducts an in-depth, application-oriented analysis of the MDACE dataset and current explainable ICD coding systems, revealing the overlap patterns between human-annotated evidence and code descriptions, the distributional characteristics of evidence within documents, and proposes new matching metrics to evaluate the utility of model explanations.

## Background & Motivation

Automated clinical coding is a key technology for streamlining documentation and billing processes. ICD (International Classification of Diseases) coding is the most critical coding system in hospital settings, directly determining reimbursement rates. With the application of deep learning, automated coding systems have reached practical readiness. However, these models, with billions of parameters, **lack transparency**—which reduces acceptance among clinical coders and poses obstacles in regulatory assessments.

Key challenges in explainability research:

**Data Scarcity**: Word-level annotations in the clinical domain are extremely costly and require domain experts.

**Limited Evaluation**: Most prior works are restricted to short texts and binary classification tasks on social media or product reviews.

**Lack of Application Perspective**: Prior research is technology-centric, lacking deep analysis from the perspective of data understanding and clinical adoption.

The MDACE dataset released by Cheng et al. (2023) is the first to provide textual evidence annotations for ICD codes in clinical records, opening a new direction for explainable ICD coding research. However, guidelines for utilizing this dataset and utility evaluations of existing methods are still lacking.

## Method

### Overall Architecture

This study revolves around two main axes: **data analysis** and **model explanation evaluation**:

**Data Analysis (RQ1-RQ3):**
- RQ1: What is the **positional distribution** of evidence in the document?
- RQ2: What is the **degree of overlap** between evidence and ICD code descriptions?
- RQ3: Is the sufficient annotation (Inpatient) a **subset** of the complete annotation (Profee)?

**Model Explanation Evaluation (RQ4-RQ6):**
- RQ4: What is the **relationship** between explanation length and classification performance?
- RQ5: How well do model explanations **match** human annotations?
- RQ6: What is the **consistency** of evidence across different modeling methods?

### Key Designs

1. **Evidence Position Analysis (RQ1)**

    - Analyzed the relative positional distribution of evidence in discharge summaries and physician notes.
    - Results: Inpatient evidence mainly appears at the beginning and end of documents; Profee evidence is more evenly distributed across discharge summaries.
    - Key finding: **Sufficient evidence does not predominantly appear at the beginning of documents**, overturning the intuitive assumption of a "top-to-bottom scan."

2. **Code Description Overlap Analysis (RQ2)**

    - Lemmatized evidence and code descriptions, removing stop words and punctuation.
    - Calculated the ratio of intersection over the description word set for all evidence-description pairs of each code.
    - Results: Distribution categorized into three types—almost no overlap, partial overlap, and strong overlap.
    - Examples: R06.83 "snoring" and I31.3 "pericardial effusion" show extremely high overlap.
    - Insight: **Codes with high overlap can be efficiently handled using rule-based systems**, freeing up neural model capacity for more challenging codes.

3. **New Matching Metrics (Core Contribution of RQ5)**

    - **Empty**: No evidence was generated because attribution scores fell below the threshold.
    - **Exact match**: Model evidence matches the annotated evidence exactly.
    - **Proximate match**: At least one token of all annotated sequences is matched, and unmatched tokens are within a context window of $k=10$.
    - **Partial match**: At least one annotated sequence has no match, or has tokens outside the context window.
    - **No match**: Absolutely zero intersection in token IDs.
    - Motivation: Traditional F1/IOU metrics are not intuitive enough; in practical scenarios, simply "guiding coders to the correct context window" is highly effective.

4. **Model Experimental Setup**

    - 50 models ($5 \text{ training strategies} \times 10 \text{ seeds}$): supervised, unsupervised, gradient regularization, projected gradient descent, and token masking.
    - Explanation method: AttInGrad (Attention $\times$ Input $\times$ Grad, $L_2$ norm).
    - Selected the best supervised and best unsupervised models for in-depth analysis.

## Key Experimental Results

### Dataset Analysis Results

| Analysis Item | Conclusion |
|--------|------|
| Average evidence length | Inpatient 2.18 tokens, Profee 1.96 tokens |
| Average number of labels/document | Inpatient 11.3, Profee 31.4 |
| Inpatient/Profee common subset | Only 118 common out of 470 unique note IDs |
| Subset ratio in common codes | Only 55 strict subsets out of 331 identical codes |

### Model Explanation Evaluation (Figure 6 Summary)

| Metric | Supervised Model | Unsupervised Model (IGR) |
|------|---------|----------------|
| Exact match | ~49 | ~49 |
| Proximate match | High | Higher |
| Partial match | High | Lower |
| No match | Low | Higher |
| Empty | Very Low | Low |

### Key Findings

1. **High overlap between model explanations and human annotations**: The best supervised model identifies at least one correct token for approximately 80% of the test cases.
2. **About 46-53% of "No match" cases are actually semantic matches**: For instance, 'obesity' vs 'obese', indicating that raw token ID matching underestimates model performance.
3. **Explanation length positively correlates with classification performance**: Recall for ICD codes is higher when the model extracts more evidence tokens.
4. **Supervised and unsupervised models share the same match type in 74% of the cases**, demonstrating good evidence consistency.
5. **Low-probability predictions generate almost no evidence (empty)**, which can be safely ignored in practical inference.
6. **Alignment between Inpatient and Profee is far less than expected**: Different coding regulations lead to significant discrepancies in the code sets.

### Evidence Diversity Analysis

| ICD Code | Occurrences/Unique | Evidence Examples |
|----------|-------------|---------|
| I10 (Hypertension) | 133/8 | 'hypertension', 'HTN', 'hypertensive' |
| Z87.891 (History of tobacco use) | 20/19 | 'smoking history', 'former smoker', 'the distant past' |

$\rightarrow$ Evidence for some codes is highly consistent (I10), while for others it is highly diverse (Z87.891), making the latter potentially harder to learn.

## Highlights & Insights

1. **Application-Oriented Deep Analysis**: Unlike purely technical papers, this work evaluates explainability directly from clinical requirements.
2. **Practicality of the New Matching Metric**: The concept of "Proximate match" (being within the context window) aligns well with the actual workflow of clinical coders.
3. **Data-Driven Strategic Recommendations**: High-overlap codes can be handled by rule-based systems, and the evidence diversity analysis can guide training strategies.
4. **Cross-Annotation Scheme Comparison**: The first in-depth analysis of the relationship between Inpatient (sufficient) and Profee (complete) annotations.

## Limitations & Future Work

- Experiments are limited to English and the MIMIC-III dataset; different national coding schemes and languages might lead to different patterns.
- The context window $k=10$ in the matching metric is empirically set and lacks a sensitivity analysis.
- Only AttInGrad was evaluated as the explanation method, leaving other types (such as SHAP or LIME) uncompared.
- The small size of the dataset (302 admissions) limits the granularity of the statistical analysis.

## Related Work & Insights

- The pioneering work of Edin et al. (2024) provides trained models and faithfulness/plausibility evaluations.
- The creation of the MDACE dataset by Cheng et al. (2023) serves as a foundational contribution to this field.
- The label-wise attention architecture of PLM-ICD (Huang et al., 2022) forms the basis for modeling.
- Jacovi and Goldberg (2020) categorized faithfulness and plausibility as two dimensions of explainability evaluation.

## Rating

- **Novelty**: ⭐⭐⭐ — The main contributions lie in the analysis and insights, with limited methodological novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage across data analysis, model evaluation, and qualitative analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear research questions, elegant structure, and findings that possess practical guiding value.
- **Value**: ⭐⭐⭐⭐ — Provides critical practical recommendations for the development and deployment of explainable clinical coding systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations](expert_an_explainable_image_captioning_evaluation_metric_with_structured_explana.md)
- [\[ACL 2025\] Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework](towards_explainable_temporal_reasoning_in_large_language_models_a_structure-awar.md)
- [\[ICLR 2026\] Evidence for Limited Metacognition in LLMs](../../ICLR2026/interpretability/evidence_for_limited_metacognition_in_llms.md)
- [\[NeurIPS 2025\] SpEx: A Spectral Approach to Explainable Clustering](../../NeurIPS2025/interpretability/spex_a_spectral_approach_to_explainable_clustering.md)
- [\[ICML 2025\] Rethinking Explainable Machine Learning as Applied Statistics](../../ICML2025/interpretability/rethinking_explainable_machine_learning_as_applied_statistics.md)

</div>

<!-- RELATED:END -->
