---
title: >-
  [Paper Note] Internal and External Impacts of Natural Language Processing Papers
description: >-
  [ACL 2025][LLM (Other)][scientometrics] This work systematically analyzes the impact of ACL/EMNLP/NAACL papers from 1979 to 2024 across both internal (academic citations) and external (patents, media, policy documents) dimensions. It finds that the language modeling topic has the broadest impact, the ethics/fairness topic has prominent impact in policy documents despite low academic citations, and multi-dimensional external impact can effectively predict highly-cited papers i…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "scientometrics"
  - "impact measurement"
  - "NLP research"
  - "citation analysis"
  - "external impact"
date: 2026-05-08
content_hash: 8e2d8e9a1ea86e56
---

# Internal and External Impacts of Natural Language Processing Papers

**Conference**: ACL 2025  
**arXiv**: [2505.16061](https://arxiv.org/abs/2505.16061)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: scientometrics, impact measurement, NLP research, citation analysis, external impact

## TL;DR

This work systematically analyzes the impact of ACL/EMNLP/NAACL papers from 1979 to 2024 across both internal (academic citations) and external (patents, media, policy documents) dimensions. It finds that the language modeling topic has the broadest impact, the ethics/fairness topic has prominent impact in policy documents despite low academic citations, and multi-dimensional external impact can effectively predict highly-cited papers internally.

## Background & Motivation

**Background**: The NLP field is rapidly evolving, and top venues (ACL, EMNLP, NAACL) publish a large number of papers annually. However, there is a lack of systematic research on how the impact of these papers is perceived both within academia and by the external public.

**Limitations of Prior Work**: Traditional scientometrics primarily focuses on academic citations (internal impact), neglecting the consumption of papers in external channels such as patents, media, and policy documents. Existing studies (e.g., Yin et al. 2022) only use binarized metrics (whether cited or not), failing to quantify differences in citation intensity.

**Key Challenge**: The NLP community is transitioning from linguistic foundations to large language models. However, it remains unclear which research topics are truly widely consumed, both within academia and across technological, societal, and policy domains.

**Goal**: To systematically quantify the differences in the impact of various NLP research topics in internal and external domains, revealing the correlation and complementarity of cross-domain impacts.

**Key Insight**: Utilizing four data sources—OpenAlex, Reliance on Science, Altmetric, and Overton—to construct multi-dimensional impact metrics covering citations, patents, media, and policy documents.

**Core Idea**: To expand the impact of NLP papers from single academic citations to four dimensions (including patents, media, and policy documents), and to reveal differences in consumption patterns across different topics and domains using normalized impact metrics.

## Method

### Overall Architecture

This paper is a large-scale scientometric analysis study. The overall workflow consists of: (1) collecting 24,821 papers from 1979 to 2024 from the ACL Anthology; (2) linking them to OpenAlex to obtain citation data (successfully mapping 21,104 papers); (3) retrieving external citation data from Reliance on Science (patents), Altmetric (media), and Overton (policy documents); (4) labeling papers into 25 topic categories using GPT-4o; and (5) calculating normalized impact metrics for each topic in each domain and performing correlation analysis.

### Key Designs

1. **Normalized Impact Metric Impact(t|d)**:

    - **Function**: Quantifies the impact of a specific topic $t$ in a specific domain $d$.
    - **Mechanism**: $\text{Impact}(t|d) = \frac{\sum_{p \in \mathcal{P}_t} \#\text{citation}(p|d) / |\mathcal{P}_t|}{\sum_{p \in \mathcal{P}} \#\text{citation}(p|d) / |\mathcal{P}|}$, i.e., the average number of citations of the specific topic divided by the average number of citations of all papers.
    - **Design Motivation**: Eliminates differences in absolute citation volumes across different domains via normalization, enabling cross-domain comparisons. Compared to the binarization method of Yin et al. 2022, this approach preserves citation intensity information.

2. **Multi-Source Data Fusion**:

    - **Function**: Integrates four data sources to cover internal and external impacts.
    - **Mechanism**: Utilizes OpenAlex for internal academic citations and Reliance on Science (patents, 20,218 links), Altmetric (media, 18,586 links), and Overton (policy documents, 1,223 links) for external citations.
    - **Design Motivation**: Different external domains reflect different types of public consumption demands—patents focus on practical technologies, media focuses on model behavior, and policy documents focus on societal impact.

3. **GPT-4o Topic Labeling and Quality Verification**:

    - **Function**: Classifies each paper into one of the 25 submission topics in the ACL 2025 CFP.
    - **Mechanism**: Uses GPT-4o to predict the most relevant topic $t_p \in \mathcal{T}$ for each paper, followed by human evaluation to verify quality, achieving "substantial agreement" (Fleiss' kappa) among human evaluators.
    - **Design Motivation**: Leverage LLMs for large-scale automatic labeling while ensuring reliability through human evaluation.

4. **Highly-Cited Prediction Experiment**:

    - **Function**: Verifies the positive correlation between external and internal impacts through experiments.
    - **Mechanism**: Examines whether papers cited by external domains (at least once) are more likely to be top-1% highly-cited papers. The random baseline hit rate is 1%. If cited by patents, it is 5.46%; if cited by media, 9.26%; if cited by policy documents, 18.29%; and if cited by all three simultaneously, it reaches 71.88%.
    - **Design Motivation**: Provides quantitative evidence demonstrating that external impact can serve as a predictive signal for internal impact.

## Key Experimental Results

### Main Results

| Topic | Citation Impact | Patent Impact | Media Impact | Policy Document Impact |
|------|-----------------|---------------|--------------|------------------------|
| Language Modeling | >1 (Highest) | >1 (Highest) | >1 (Highest) | >1 (Second) |
| Ethics, Bias, Fairness | <1 | Lowest | >1 | >1 (Highest) |
| Linguistic Foundations | <1 | <1 | <1 | <1 |

### Cross-Domain Correlation Analysis

| External Domain | Pearson Correlation with Citation |
|-----------------|-----------------------------------|
| Patent | 0.654 |
| Media | 0.725 |
| PolicyDocument | 0.247 (0.599 after removing Ethics) |

### Highly-Cited Prediction Hit Rate

| External Domain Combination | Hit Rate |
|-----------------------------|----------|
| No external signal (baseline) | 1.00% |
| {Patent} | 5.46% |
| {Media} | 9.26% |
| {PolicyDocument} | 18.29% |
| {Patent, Media} | 26.72% |
| {Patent, Media, PolicyDocument} | 71.88% |

### Key Findings
- Language modeling is the only topic that is overrepresented (Impact > 1) across all internal and external domains.
- The ethics/fairness topic has the highest impact in policy documents but ranks last in patents, forming a prominent "external dimension divergence."
- Linguistic foundations (phonology, morphology, psycholinguistics) perform poorly across all dimensions.
- Patents prefer practical NLP technologies (IR, MT, Speech), whereas media and policy focus more on model behavior and societal impact.
- There is complementarity rather than substitution among different external domains (the Pearson correlation between Patent and PolicyDocument is -0.140).

## Highlights & Insights
- For the first time, this work systematically evaluates NLP papers from a multi-dimensional external impact perspective, revealing impact divergence phenomena that cannot be captured by citation counts alone, thereby providing data support for the strategic direction of the NLP community.
- The highly-cited prediction experiment is highly convincing—combining three-dimensional external signals increases the random hit rate from 1% to 71.88%, offering a new avenue for early prediction of paper impact.

## Limitations & Future Work
- External data sources are incomplete and do not cover all public channels that NLP papers might impact.
- GPT-4o topic labeling may introduce errors, although human evaluation demonstrates reliable quality.
- There is a lack of causal explanation—it remains unclear whether academia leads the public or public demand guides academia.
- Temporal analysis cannot be performed because some external data sources do not provide citation timestamps.

## Related Work & Insights
- **vs Yin et al. 2022**: Similarly studies the public consumption of scientific papers, but this work utilizes actual citation counts instead of a binarized metric and focuses on fine-grained topic analysis within the NLP field.
- **vs Cao et al. 2023**: Focuses only on patent citations of HCI papers, whereas this work expands to three external domains: patents, media, and policy documents.
- **vs Zhang et al. 2024**: While that survey focuses on the application of scientific LLMs, this work analyzes the impact of NLP papers themselves from a scientometric perspective.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-dimensional external impact analysis perspective is novel, but the methodology itself (normalized citation count + correlation analysis) is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large data scale (24K papers, four data sources) and comprehensive analysis, but lacks a discussion of causal mechanisms.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, progressive presentation of 7 observation points, and intuitive charts.
- Value: ⭐⭐⭐⭐ Provides valuable data support for the self-reflection and strategic planning of the NLP community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge](enhancing_transformation_from_natural_language_to_signal_temporal_logic_using_ll.md)
- [\[ACL 2025\] Natural Language Processing in Support of Evidence-based Medicine: A Scoping Review](natural_language_processing_in_support_of_evidence-based_medicine_a_scoping_revi.md)
- [\[ACL 2025\] Cooperating and Competing Through Natural Language](cooperating_and_competing_through_natural_language.md)
- [\[ACL 2025\] LLM×MapReduce: Simplified Long-Sequence Processing using Large Language Models](llm_mapreduce_simplified_long_sequence_processing.md)
- [\[ACL 2025\] What Makes a Good Natural Language Prompt?](good_natural_language_prompt.md)

</div>

<!-- RELATED:END -->
