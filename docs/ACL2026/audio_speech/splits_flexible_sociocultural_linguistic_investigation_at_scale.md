---
title: >-
  [Paper Note] Splits! Flexible Sociocultural Linguistic Investigation at Scale
description: >-
  [ACL 2026][Audio & Speech][sociocultural linguistic phenomena] This paper proposes a methodology for constructing a sociolinguistic "sandbox," building Splits!—a 9.7 million post dataset from Reddit partitioned along two axes (demographic group × discussion topic) across 6 groups and 89 topics—and designing a two-stage filtering pipeline based on lift and triviality to efficiently identify non-trivial, research-worthy sociocultural linguistic phenomena from 23,000 LLM-generated candidate hypotheses.
tags:
  - ACL 2026
  - "Audio & Speech"
  - sociocultural linguistic phenomena
  - Reddit dataset
  - hypothesis filtering
  - lexical analysis
  - demographics
date: 2026-05-08
content_hash: 318745b029e007da
---

# Splits! Flexible Sociocultural Linguistic Investigation at Scale

**Conference**: ACL 2026
**arXiv**: [2504.04640](https://arxiv.org/abs/2504.04640)
**Code**: [GitHub](https://github.com/ecaplan/splits) (code + data + demo)
**Area**: Sociolinguistics / Computational Social Science
**Keywords**: sociocultural linguistic phenomena, Reddit dataset, hypothesis filtering, lexical analysis, demographics

## TL;DR
This paper proposes a methodology for constructing a sociolinguistic "sandbox," building Splits!—a 9.7 million post dataset from Reddit partitioned along two axes (demographic group × discussion topic) across 6 groups and 89 topics—and designing a two-stage filtering pipeline based on lift and triviality to efficiently identify non-trivial, research-worthy sociocultural linguistic phenomena from 23,000 LLM-generated candidate hypotheses.

## Background & Motivation

**Background**: Computational social science leverages social media data to study linguistic variation across communities (e.g., AAVE code-switching, Yiddish vocabulary in Jewish English), but such studies typically require customized data collection and experimental design tailored to specific groups and topics, making rapid prototyping costly and labor-intensive.

**Limitations of Prior Work**: Validating a sociolinguistic hypothesis demands substantial upfront investment. While automated hypothesis generation (e.g., via LLMs) can produce large numbers of candidate hypotheses, the critical bottleneck lies in efficiently identifying which among thousands of machine-generated candidates are genuinely worth pursuing. Many statistically significant hypotheses are in practice trivial (e.g., "Jewish users mention Judaism more frequently").

**Key Challenge**: Statistical significance ≠ research value. Trivial hypotheses can achieve high significance on data, yet offer no scientific insight to social science. An automated mechanism is needed to distinguish "statistically valid" from "academically interesting."

**Goal**: (1) Construct a flexible sociolinguistic exploration sandbox dataset; (2) design an automated filtering pipeline to distinguish interesting hypotheses from trivial ones.

**Key Insight**: Formalize sociocultural linguistic phenomena (SLPs) as "group A uses lexical set L more than group B when discussing topic t," quantify statistical validity via BM25 retrieval + lift metric, and quantify triviality via semantic similarity.

**Core Idea**: Build a sandbox through demographic × topic dual partitioning; apply a lift + triviality two-stage filter to distill non-trivial, statistically supported hypotheses from a large candidate pool.

## Method

### Overall Architecture
The framework consists of two components: (1) **Dataset construction**—a pipeline of seed subreddit → seed user → site-wide post collection → topic annotation to build the Splits! dataset, partitioned across 6 demographic groups × 89 topics; (2) **Hypothesis filtering**—computing lift (data support) and triviality for each candidate hypothesis, with dual filtering to surface high-value hypotheses.

### Key Designs

1. **Group-ness Metric and Demographic Validation**:

    - Function: Ensures collected posts genuinely originate from the target demographic group.
    - Mechanism: Defines $\text{group-ness}(u) = \sum_{s \in SD} \log(1 + c_{u,s})$, where $c_{u,s}$ is the number of posts by user $u$ in seed subreddit $s$. This metric rewards both total posting volume and cross-subreddit diversity. High group-ness users are validated against the target group via self-identification phrases (e.g., "I am Catholic").
    - Design Motivation: Inferring group identity solely from subreddit membership introduces substantial noise; a quantitative metric is necessary to select high-confidence users.

2. **Lift Metric for Hypothesis Validity**:

    - Function: Measures the statistical efficacy of a lexical set $L$ in distinguishing two groups.
    - Mechanism: Posts from both groups on the same topic are pooled into a BM25 index; $L$ serves as a query for re-ranking. The metric is computed as $\text{lift}@p\% = \frac{\#A \text{ posts}@p\% / \# \text{posts}@p\%}{\#A \text{ posts overall} / \# \text{posts overall}}$. Lift > 1 indicates that $L$ successfully ranks target-group posts higher. A hypergeometric test is applied to ensure statistical significance.
    - Design Motivation: Lift is a well-established association measure in data mining, more robust than simple frequency comparison, and can capture phenomena at varying granularities by adjusting $p\%$.

3. **Triviality Metric for Filtering Trivial Hypotheses**:

    - Function: Automatically identifies whether a hypothesis is trivial (e.g., "Jewish users mention Judaism").
    - Mechanism: A small "defining lexicon" $\ell_A$ (5–10 words) is manually constructed for each group; the subspace recall $R_{subspace}(L, \ell_A)$ between hypothesis lexical set $L$ and $\ell_A$ is computed. A higher score indicates closer proximity to group-defining vocabulary, hence greater triviality. The metric yields a Spearman correlation of $\rho = -0.38$ with human "surprisingness" ratings, validating its effectiveness.
    - Design Motivation: Statistical significance is positively correlated with triviality (Spearman 0.32), demonstrating that reliance on statistical testing alone causes trivial hypotheses to dominate.

### Loss & Training
No model training is involved. Dataset construction uses the ColBERT retrieval model and LLM-assisted topic classification. Hypothesis filtering employs BM25 and semantic similarity computed in BERT embedding space.

## Key Experimental Results

### Main Results
Replication of known sociocultural linguistic phenomena:

| Phenomenon | Target Group | Topic | Usage Rate (Target) | Usage Rate (Control) | p-value |
|---|---|---|---|---|---|
| AAVE usage | Black | Hip-Hop | 3.16% | 2.00% | <0.001 |
| AAVE code-switching | Black | Professional→Hip-Hop | 0.33%→3.16% | — | <0.001 |
| Yiddish usage | Jewish | Judaism | 0.19% | 0.07% | <0.001 |
| Dance identity | Hindu/Sikh/Jain | Cultural identity | 0.44% | 0.36% | <0.001 |

### Ablation Study
Two-stage filtering efficiency analysis:

| Triviality Percentile Threshold | Precision | Recall | F1 | Efficiency Gain |
|---|---|---|---|---|
| Baseline (p-value only) | 0.270 | 1.000 | 0.425 | 1.00× |
| 0.3 | 0.447 | 0.496 | 0.470 | 1.65× |
| 0.5 | 0.398 | 0.741 | 0.518 | 1.47× |

### Key Findings
- The two-stage filtering achieves an overall 15–18× efficiency gain: the first-stage statistical filter reduces the candidate set by 10×, and the second-stage triviality filter yields an additional 1.5–1.8× reduction.
- Hypotheses sourced from the academic literature exhibit significantly lower triviality scores than LLM-generated ones (mean 0.585 vs. 0.810), validating the metric's alignment with "academic interestingness."
- One notable non-trivial finding: Jewish users discussing healthcare topics use vocabulary related to "preventive care" and "early detection" more frequently, potentially reflecting a culturally embedded emphasis on this-worldly concerns.

## Highlights & Insights
- The insight that "statistical significance ≠ research value" may be intuitively obvious, yet this paper is the first to empirically quantify their positive correlation (Spearman 0.32) and provide a systematic solution—a problem that pervades computational social science.
- The sandbox design is highly practical: researchers can test new hypotheses at zero marginal cost by supplying a lexical set and immediately obtaining cross-group, cross-topic analyses.
- The group-ness metric combined with self-identification validation establishes a reproducible paradigm for demographic inference from social media data.

## Limitations & Future Work
- Data are drawn exclusively from Reddit (2012–2018), introducing platform and temporal biases.
- Coverage is limited to 6 demographic groups, predominantly English-speaking.
- The group-ness approach favors users who are highly active in identity-focused communities, potentially amplifying between-group linguistic differences.
- Analysis is restricted to the lexical level, leaving syntactic, semantic frame, and pragmatic features unexamined.
- Intersectional identities (e.g., individuals who are both Black and Catholic) are not analyzed.

## Related Work & Insights
- **vs. Traditional Sociolinguistics**: Conventional approaches rely on fieldwork and deep ethnography—high cost but rich insight; Splits! provides large-scale hypothesis screening as a complementary tool.
- **vs. LLM Hypothesis Generation**: Yang et al. (2024) and related work focus on using LLMs to generate hypotheses but lack mechanisms for filtering high-value ones; this paper fills that gap.

## Rating
- Novelty: ⭐⭐⭐⭐ The sandbox concept and triviality filtering constitute genuine innovative contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Replication of 5 known phenomena + filtering of 23,000 candidate hypotheses + human annotation validation.
- Writing Quality: ⭐⭐⭐⭐ Methodology is clearly described, though the paper's length requires patience.
- Value: ⭐⭐⭐⭐ Provides reusable methodology and data resources for computational social science.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models](../../NeurIPS2025/audio_speech/data-juicer_20_cloud-scale_adaptive_data_processing_for_and_with_foundation_mode.md)
- [\[ICLR 2026\] Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention](../../ICLR2026/audio_speech/efficient_audio-visual_speech_separation_with_discrete_lip_semantics_and_multi-s.md)
- [\[ACL 2026\] When Misinformation Speaks and Converses: Rethinking Fact-Checking in Audio Platforms](when_misinformation_speaks_and_converses_rethinking_fact-checking_in_audio_platf.md)
- [\[ACL 2026\] MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus](mcga_a_multi-task_classical_chinese_literary_genre_audio_corpus.md)
- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)

<!-- RELATED:END -->
