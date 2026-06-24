---
title: >-
  [Paper Note] Data Caricatures: On the Representation of African American Language in Pretraining Corpora
description: >-
  [ACL 2025][LLM Pretraining][African American Language] Combining quantitative experiments, human judgment, and qualitative analysis, this work systematically evaluates the quantity and quality of African American Language (AAL) across 12 open-source pretraining corpora. It finds that AAL constitutes only 0.007%–0.18% of the documents (far below its population representation). In C4, 28.9% of AAL texts are judged inappropriate for LLM generation…
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "African American Language"
  - "pretraining corpora"
  - "data quality"
  - "representation bias"
  - "automated filtering"
  - "sociolinguistic equity"
date: 2026-05-08
content_hash: 062affc212c63baa
---

# Data Caricatures: On the Representation of African American Language in Pretraining Corpora

**Conference**: ACL 2025  
**arXiv**: [2503.10789](https://arxiv.org/abs/2503.10789)  
**Code**: [NickDeas/DataCaricatures](https://github.com/NickDeas/DataCaricatures)  
**Area**: NLP Fairness / Pretraining Data Analysis  
**Keywords**: African American Language, pretraining corpora, data quality, representation bias, automated filtering, sociolinguistic equity

## TL;DR

Combining quantitative experiments, human judgment, and qualitative analysis, this work systematically evaluates the quantity and quality of African American Language (AAL) across 12 open-source pretraining corpora. It finds that AAL constitutes only 0.007%–0.18% of the documents (far below its population representation). In C4, 28.9% of AAL texts are judged inappropriate for LLM generation, and 24.5% reinforce harmful stereotypes. Furthermore, 13 out of 16 automated filters systematically favor retaining White Mainstream English (WME) over AAL.

## Background & Motivation

**Background**: The distribution of LLM pretraining data directly determines the model's ability to understand different language varieties. AAL is one of the most widely spoken English dialects in the United States—used by approximately 80% of African Americans (about 10% of the US population). However, prior to this work, only Dodge et al. (2021) had conducted a limited analysis on C4.

**Limitations of Prior Work**:
   - **Unknown Quantity**: Outside of C4, the proportion of AAL in other major pretraining corpora has never been systematically quantified.
   - **Unvetted Quality**: Even when AAL text is present, how much of it represents authentic everyday language use, versus hip-hop lyrics, corporate marketing, or stereotypical mimicry by non-native speakers?
   - **Filtering Bias**: Do standard data cleaning pipelines (quality filtering, toxicity filtering, deduplication) exhibit systematic bias against AAL?

**Key Challenge**: If AAL is underrepresented or of poor quality in pretraining data (skewed toward stereotypical caricatures rather than natural language), LLMs will not only fail to understand and generate AAL correctly, but may also reinforce discriminatory behavior against AAL speakers. Previous research has already observed that LLMs show bias against AAL in toxicity detection, stereotyping, and dialogue generation.

**Goal**: Conduct a comprehensive audit of the representation of AAL in pretraining data, structured around three research questions (RQs):
   - **RQ1**: How much AAL is in pretraining corpora? What is the distribution of specific AAL morphosyntactic features?
   - **RQ2**: What is the quality of the included AAL texts (source diversity, authenticity, harmfulness)?
   - **RQ3**: What is the impact of modern data quality filtering strategies on AAL representation?

**Key Insight**: Instead of focusing only on "how much" (quantity), this work places a strong emphasis on "what kind" (quality)—introducing the concept of "Data Caricatures," where AAL in pretraining data is a distorted caricature rather than a faithful representation of real-world language use.

**Core Idea**: A mixed-methods approach (quantitative + human + qualitative) to systematically audit 12 open-source corpora, revealing issues in AAL representation across three dimensions: quantity, quality, and filtering.

## Method

### Overall Architecture

The research is designed as a three-stage audit pipeline corresponding to the three RQs:
1. **Quantitative Audit (RQ1)**: Extract AAL subsets from 12 corpora using a demographically-aligned classifier to calculate proportions and analyze the distribution of 17 morphosyntactic features.
2. **Quality Audit (RQ2)**: Perform human annotation on an AAL subset of C4 (1,054 texts), cross-reference with hip-hop lyrics overlap detection, and analyze use by non-native speakers.
3. **Filtering Audit (RQ3)**: Evaluate the differential behavior of 16 automated filters on AAL vs. WME in RedPajama-v2, alongside controlled experiments on three AAL sources (dialogue, lyrics, social media).

### Key Design 1: AAL Extraction and Feature Analysis

- **Function**: Identifying AAL-containing documents from 12 corpora and analyzing the distribution of their grammatical features.
- **Mechanism**: Leveraging the mixed-membership demographically-aligned classifier from Blodgett et al. (2016) (trained on Twitter data) and selecting documents with the highest AAL probabilities as the AAL subset. Furthermore, a human-in-the-loop framework utilizing the CGEdit model (Masis et al., 2022) is used to automatically identify 17 AAL morphosyntactic features (e.g., habitual *be*, copula deletion, negative concord, etc.).
- **Design Motivation**:
    - Using a threshold of 0.3 (instead of the common 0.8) to obtain a more conservative estimate of feature prevalence while keeping the corpus size manageable.
    - Adopting a 250 GB random sampling analysis for four ultra-large scale corpora (>3 billion documents) and reporting 99% confidence intervals.
    - Human validation: Recruiting three native AAL speaker annotators to judge human similarity and language match for 1,054 texts in C4 ($\kappa = 0.581$ and $0.747$).

### Key Design 2: Multidimensional Evaluation of AAL Text Quality

- **Function**: Evaluating AAL text quality across three dimensions: source diversity, authenticity, and non-harmfulness.
- **Mechanism**:
    - **Hip-hop lyrics detection**: Employing the deduplication method of Brown et al. (2020) using 8-13 token n-gram overlap detection to identify hip-hop/rap lyrics embedded in C4.
    - **Native speaker judgment**: Annotators evaluate whether a text was authored by a native AAL speaker (Native Speaker dimension, $\kappa = 0.619$).
    - **Stereotypes and appropriateness**: Annotators assess whether a text reinforces harmful stereotypes and its suitability for LLM generation (Appropriateness) using a 4-point Likert scale.
- **Design Motivation**: Although hip-hop lyrics contain AAL features, they do not represent everyday language use. Similarly, corporate social media mimicry of AAL tends to exaggerate linguistic features. These factors distort the LLM's learning of AAL. Thus, it is crucial to distinguish between "authentic representation" and "caricatured representation."

### Key Design 3: Evaluation of Automated Filter Bias

- **Function**: Evaluating the impact of 16 automated filtering strategies (language filters, toxicity filters, quality filters) on AAL.
- **Mechanism**:
    - **Natural distribution experiments**: Extracting approximately 235k documents each for AAL ($p \ge 0.8$) and WME ($p \ge 0.8$) subsets from RedPajama-v2, and comparing the z-score normalized scores output by the filters.
    - **Controlled source experiments**: Evaluating across three different AAL sources—CORAAL conversational transcriptions (natural language), hip-hop lyrics, and TwitterAAE social media texts.
    - Applying two-tailed t-tests to evaluate statistically significant differences ($p < 0.01$).
- **Design Motivation**: Prior work only established that the C4 blocklist was biased against AAL. However, it was unclear whether modern model-driven filters (such as Wikipedia-based quality classifiers or LLM-as-a-judge) exhibit a similar bias.

## Key Experimental Results

### Main Results: AAL Proportion in 12 Corpora (Table 1)

| Corpus | Documents | AAL Doc % | Common Crawl % |
|--------|--------|-------------|-----------------|
| OpenWebText | 8M | 0.01% | 0% |
| The Pile | 140M | 0.08% | 3% |
| Dolmino (Dolmino-mix) | 165M | 0.03% | 83% |
| C4 | 365M | 0.07% | 100% |
| C4.NoBlockList | 395M | 0.11% | 100% |
| RefinedWeb | 968M | 0.12% | 100% |
| RedPajama | 968M | **0.007%** (lowest) | 88% |
| FineWeb-Edu | 1.8B | **0.0009%** | 100% |
| Dolma | 2.5B | 0.12% | 78% |
| RedPajama-v2 (sampled) | 20.8B | **0.18%** (highest) | 100% |

### Human Annotation Validation (Table 2)

| AAL Classifier Probability Interval | C4 Docs | % Judged to Contain AAL Features |
|-------------------|----------|---------------------|
| 0.5 ≤ p ≤ 0.6 | 41,930 | 44.7% |
| 0.6 ≤ p ≤ 0.7 | 12,913 | 36.3% |
| 0.7 ≤ p ≤ 0.8 | 4,319 | 36.7% |
| 0.8 ≤ p ≤ 0.9 | 922 | 30.9% |
| 0.9 ≤ p | 120 | 23.0% |

### Quality Evaluation Key Data (Figure 5)

| Dimension | Negative Proportion in C4.en | Description |
|------|----------------|------|
| Inappropriateness | **28.9%** | Unsuitable for LLM generation |
| Stereotype | **24.5%** | Reinforces harmful stereotypes |
| Written by non-native speakers | **51%** | Judged to contain AAL features but not written by native AAL speakers in C4.en |
| Hip-hop lyrics overlap | ~12% (C4.en) / ~15% (C4.en.noBlocklist) | 8-gram overlap detection |

### Filter Bias Analysis (Figure 6)

| Metric | Result |
|------|------|
| Filters favoring WME retention | **13/16 filters** (81.3%) |
| Filters favoring AAL retention | Only 3/16 filters (including 2 using Wikipedia as a high-quality reference) |
| Filter preference for AAL sources | 11/16 filters prefer conversational transcriptions > social media > lyrics |
| Cross-corpus AAL document duplication rate | **17%** of AAL documents are duplicated in at least one other corpus |

### Key Findings

1. **Extreme Underrepresentation of AAL**: The proportion of AAL across all corpora is far below the demographics of AAL speakers in the US (approximately 10%), dropping as low as 0.0009%.
2. **Concerning Quality—"Data Caricatures"**: Nearly one-third of AAL text in C4 is evaluated as inappropriate or stereotype-reinforcing. 51% of AAL texts are not written by native speakers, and roughly 12%–15% consist of hip-hop lyrics rather than everyday speech.
3. **Insufficient Diversity**: 17% of AAL documents are duplicated across corpora, and the distribution of AAL features is heavily skewed by filtering (e.g., the frequency of Zero Copula drops drastically after filtering).
4. **Systemic Filter Bias**: Most modern filters (including model-based quality and toxicity filters) disproportionately remove AAL content.
5. **The Paradox**: Filters prefer natural, conversational AAL (CORAAL), yet such resources are virtually non-existent on the web; consequently, the actually retained AAL texts are mostly low-quality web texts.

## Highlights & Insights

- **The conceptualization of "Data Caricatures"** is highly insightful: AAL in pretraining data is not just scarce, but is a distorted caricature of authentic language use—consisting of hip-hop lyrics, corporate mimicry, and exaggerated non-native speaker use. This represents a more insidious problem than simple underrepresentation.
- **Rigorous Mixed-Methods Design**: The tri-fold combination of quantitative statistics (across 12 corpora), human annotation (1,054 texts annotated by 3 native AAL speakers), and qualitative analysis provides a highly robust evidentiary chain.
- **Comprehensive Filter Evaluation and Controlled Experiments**: Beyond evaluating the natural distribution on 16 filters, controlled source experiments (conversations vs. lyrics vs. social media) uncover the root of the bias: it is not a design flaw of the filters themselves, but rather that the benchmark or definition of "high quality" is inherently skewed toward WME.
- **The discovery of corporate social media AAL mimicry** is a novel observation. Texts like "...this will get you where you need to be. Ball out by clicking the link below." exaggerate AAL characteristics, worsening the model's acquisition of stereotypes.
- **Clear Downstream Implications**: This study provides a direct explanation for why LLMs show bias against AAL in toxicity detection (Sap et al., 2019a) and perform poorly when understanding and generating AAL (Deas et al., 2023).

## Limitations & Future Work

1. **Classifier generalization is limited**: The demographically-aligned classifier is trained on Twitter data, which may lead to varying accuracy in other domains (such as forums or news comments). In higher probability intervals, the proportion of texts judged to actually contain AAL features was paradoxically lower, showing that the classifier is sensitive to false positives like slang or stage names.
2. **Annotator underrepresentation**: The 3 annotators all have academic backgrounds in linguistics or computational linguistics, which may not represent the views of the broader AAL community. Agreement on Stereotype and Appropriateness dimensions was extremely low ($\kappa = -0.021$ and $0.188$), reflecting high subjectivity.
3. **Only covers open-source corpora**: Closed-source models like GPT-4o and Llama-3 have training data that cannot be analyzed, limiting the generalizability of some findings.
4. **Does not address other minority language varieties**: Varieties like Chicano English, Asian American English, or Indigenous dialects are not covered, though the framework can be extended to them.
5. **Lacks downstream impact validation**: The paper solely audits the dataset itself, rather than verifying whether the sparsity/distortion of AAL in the pretraining data directly results in biased model behavior.
6. **Future directions**:
    - Developing more accurate cross-domain AAL detectors.
    - Collecting natural conversational AAL data (e.g., expanding CORAAL) for data augmentation.
    - Designing "dialect-aware" data filtering strategies to avoid systemic deletion of minority languages.
    - Extending the auditing framework to benchmarks (such as HellaSwag or MMLU), which are similarly devoid of AAL, indirectly driving biased curation of pretraining data.

## Related Work & Insights

- **Relationship to Dodge et al. (2021)**: That work first established that C4's blocklist disproportionately filters AAL. Ours significantly extends the scope (12 corpora vs. just C4) and depth (quality auditing + 16 filters) of that analysis.
- **Relationship to Hofmann et al. (2024) and Fleisig et al. (2024)**: These works found that LLMs exhibit stereotypical behavior toward AAL. Our work explains the source of this issue—the representation of AAL in the pretraining data itself is highly stereotyped and caricatured.
- **Insights**: Any audit of pretraining data should focus on both "quantity" and "quality" dimensions. The definition of "quality" in automated filters inherently encodes cultural bias (using Wikipedia as a "high quality" gold standard). Collecting dialectal and minority language data requires participatory design involving the respective communities.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The "Data Caricatures" concept is highly innovative, and extending the audit to quantity, quality, and filtering dimensions is a first, even if the general paradigm of auditing pretraining data has prior precedent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — The paper offers exceptionally strong empirical evidence, covering 12 corpora, 16 filters, 1,054 human annotations with 3 native annotators, and combining quantitative and qualitative methodologies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The structure is highly clear (with three logical RQs driving the paper), includes deep and responsible ethical discussions, and provides thoughtful reflections on AAL representation.
- **Value**: ⭐⭐⭐⭐ — Highly significant for understanding the data-level roots of LLM bias, but lacks direct mitigation techniques ("fixing" the data), which limits its immediate actionable impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation](emergent_abilities_continued_pt.md)
- [\[ICML 2025\] Chameleon: A Flexible Data-mixing Framework for Language Model Pretraining and Finetuning](../../ICML2025/llm_pretraining/chameleon_a_flexible_data-mixing_framework_for_language_model_pretraining_and_fi.md)
- [\[ICLR 2026\] Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining](../../ICLR2026/llm_pretraining/scaling_laws_revisited_modeling_the_role_of_data_quality_in_language_model_pretr.md)
- [\[ICLR 2026\] Reformulation for Pretraining Data Augmentation](../../ICLR2026/llm_pretraining/reformulation_for_pretraining_data_augmentation.md)
- [\[ACL 2025\] Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset](nemotron_cc_pretraining_data.md)

</div>

<!-- RELATED:END -->
