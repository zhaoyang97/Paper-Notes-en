---
title: >-
  [Paper Note] The Nature of NLP: Analyzing Contributions in NLP Papers
description: >-
  [ACL2025][LLM (Other)][NLP scientometrics] Proposes a taxonomy of NLP paper contributions (Knowledge/Artifact $\times$ 8 subcategories), builds the manually annotated dataset NLPContributions ($\approx 2\text{k}$ papers), trains SciBERT to automatically identify contribution statements, and conducts a 50-year longitudinal trend analysis on $\approx 29\text{k}$ ACL Anthology papers, revealing the evolutionary trajectory of NLP research from being linguistics-oriented to method…
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "NLP scientometrics"
  - "contribution taxonomy"
  - "research trend analysis"
  - "multi-label classification"
  - "SciBERT"
date: 2026-05-08
content_hash: b8d25dd72e206f2b
---

# The Nature of NLP: Analyzing Contributions in NLP Papers

**Conference**: ACL2025  
**arXiv**: [2409.19505](https://arxiv.org/abs/2409.19505)  
**Code**: [UKPLab/acl25-nlp-contributions](https://github.com/UKPLab/acl25-nlp-contributions)  
**Area**: LLM/NLP  
**Keywords**: NLP scientometrics, contribution taxonomy, research trend analysis, multi-label classification, SciBERT

## TL;DR

Proposes a taxonomy of NLP paper contributions (Knowledge/Artifact $\times$ 8 subcategories), builds the manually annotated dataset NLPContributions ($\approx 2\text{k}$ papers), trains SciBERT to automatically identify contribution statements, and conducts a 50-year longitudinal trend analysis on $\approx 29\text{k}$ ACL Anthology papers, revealing the evolutionary trajectory of NLP research from being linguistics-oriented to method/model-dominant, and recently returning to concerns about humanity and language.

## Background & Motivation

1. **Debate on the nature of NLP research**: "What is NLP research?" has been highly debated—is it algorithm-oriented, linguistics-oriented, or a broader computational-linguistic intersection? This paper proposes to objectively answer this question through a quantitative analysis of contribution statements.
2. **Contribution statements as a window to research nature**: Self-reported contribution statements by authors are the most direct signals for understanding the essence of research, but there is still no systematic framework for extraction and classification.
3. **Lack of annotated data**: Existing NLP scientometric efforts mostly focus on metadata (citation networks, topic models), lacking fine-grained annotated corpora for the actual contribution content of papers.
4. **Explosive growth of literature**: The number of NLP papers has grown drastically in recent years, making it difficult for researchers to track field trends and emerging directions, thereby intensifying the need for automated tools.
5. **Limited scope of existing work**: Prior works like the NLP Contribution Graph are confined to information unit extraction for predefined tasks, failing to cover knowledge-based contributions (e.g., new findings about language or humans).
6. **Gap in longitudinal analysis**: To date, no work has systematically quantified the evolution of contribution types across a 50-year span in the NLP field, especially lacking a comparative analysis of knowledge vs. artifact contributions.

## Method

### Overall Architecture

Constructs a full pipeline of "taxonomy definition $\rightarrow$ data annotation $\rightarrow$ automatic classifier training $\rightarrow$ large-scale application $\rightarrow$ trend analysis". The core lies in first establishing a taxonomy of contribution types, and under this guidance, annotating data, training models, and analyzing 50 years of papers.

### Key Designs

#### 1. Contribution Taxonomy

- **Function**: Classifies NLP paper contributions into 2 major categories and 8 subcategories.
    - **Knowledge (k)**: k-dataset (new knowledge from dataset analysis), k-language (new knowledge about language), k-method (method/model analysis), k-people (new knowledge about people/society), k-task (new knowledge about tasks)
    - **Artifact (a)**: a-dataset (new dataset), a-method (new method/model), a-task (new task)
- **Design Motivation**: Aligns with the ACL'23 call for papers (analyzing analysis-based vs. resource-based contributions), and covers the five core entities in NLP research (methods, datasets, tasks, language, people).
- **Mechanism**: Iteratively defined based on the authors' NLP research experience and a synthesis of several existing papers, coupled with ontology-oriented annotation guidelines.

#### 2. NLPContributions Dataset

- **Function**: Manually annotates contribution statements in the abstracts of $1,995$ ACL Anthology papers, producing $5,890$ labeled contribution sentences.
- **Design Motivation**: Abstracts are paragraphs where contribution statements are most concentrated, offering high annotation efficiency and strong representativeness; full-text annotation is prohibitively expensive.
- **Mechanism**: Completed using a main annotator (6 years of NLP research experience) and an auxiliary annotator (4 years of experience) on Label Studio. IAA was calculated on $100$ double-annotated papers (Fleiss' $\kappa = 0.71$), with the remaining annotated by the main annotator and quality-checked by a senior author. $57.6\%$ of the contribution sentences were assigned multiple labels.

#### 3. Automatic Contribution Classification Model

- **Function**: Models contribution statement detection and classification as a multi-label classification task—given a sentence, determines whether it is a contribution sentence, and if so, assigns one or more contribution type labels.
- **Design Motivation**: Automation is required to scale up to the large-scale analysis of $\approx 29\text{k}$ papers.
- **Mechanism**: Adopts a binary relevance strategy (independent binary classification for each label), comparing fine-tuned PLMs (BERT, RoBERTa, SciBERT, BiomedBERT, Flan-T5) with prompted LLMs (GPT-3.5-Turbo, GPT-4-Turbo, LLaMA-3-8B). SciBERT was ultimately selected (F1 $= 0.80$, on par with GPT-4-Turbo, but more economical and environmentally friendly).

#### 4. Large-Scale Trend Analysis

- **Function**: Applies the trained SciBERT to $28,937$ ACL Anthology papers (1974–2024) to build the NLPContributions-Auto corpus, analyzing the temporal evolution of contribution types, venue differences, and citation impacts.
- **Design Motivation**: Answers the core question of "how NLP research has evolved over time" and provides data-driven insights to the community.
- **Mechanism**: Calculates the proportion of each contribution type by year, compares distributions across venues, and analyzes the relationship between citation counts and contribution types for ACL'18 papers.

## Key Experimental Results

### Table 1: Performance Comparison of Automatic Classification Models

| Setup | Model | Precision | Recall | F1 |
|------|------|-----------|--------|-----|
| Finetuning | BERT | 0.31 | 0.50 | 0.38 |
| Finetuning | BiomedBERT | 0.64 | 0.59 | 0.60 |
| Finetuning | **SciBERT** | **0.81** | **0.80** | **0.80** |
| Finetuning | Flan-T5 | 0.79 | 0.78 | 0.78 |
| Prompting | GPT-3.5-Turbo | 0.75 | 0.71 | 0.73 |
| Prompting | GPT-4-Turbo | 0.80 | 0.80 | 0.80 |
| Prompting | LLaMA-3-8B | 0.60 | 0.56 | 0.53 |

SciBERT’s F1 score reaches $0.80$, which is on par with GPT-4-Turbo while being more cost-effective, and was thus selected for subsequent large-scale analyses.

### Table 2: Citations of Different Contribution Types for ACL'18 Papers (352 papers, with $\ge 5$ years of publication history)

| Contribution Type | Number of Papers | Mean Citations | Median Citations |
|----------|--------|----------|----------|
| a-dataset | 154 | 137.7 | 64.0 |
| k-method | 280 | 127.8 | 56.0 |
| a-method | 310 | 122.2 | 58.0 |
| k-dataset | 219 | 121.1 | 56.0 |
| a-task | 270 | 116.0 | 56.0 |
| k-task | 328 | 115.7 | 55.0 |
| k-people | 119 | 109.5 | 54.0 |
| k-language | 193 | 107.1 | 53.0 |

Papers introducing new datasets (a-dataset) received the highest citation volume (average of $137.7$), while language knowledge contributions (k-language) received the lowest.

### Key Findings

- In the 1970s and 1980s, NLP was dominated by linguistics and humanities research (k-language accounted for $\approx 80\%$); this percentage fell sharply to $\approx 40\%$ after the rise of statistical methods in the 1990s.
- Method-based artifact contributions (a-method) have risen sharply since the 1990s and have remained consistently high.
- Language and humanities contributions have rebounded since 2020, reflecting the rise of computational social science and NLP ethics.
- Currently, the contribution types of NLP papers are more diverse than during any previous historical period.

## Highlights & Insights

- **Elegant taxonomy design**: The Knowledge/Artifact $\times$ 5/3 subcategory taxonomy aligns with ACL submission guidelines and offers a fine grain of granularity to distinguish method analysis from method design.
- **50-year longitudinal perspective**: Covers $\approx 29\text{k}$ papers from 1974 to 2024, representing the largest longitudinal analysis of NLP contribution types to date.
- **Insightful findings**: The discovery that the "shift to method-dominated research started in the 90s rather than the Transformer era" challenges common narratives.
- **High utility**: The NLPContributions-Auto corpus can be directly utilized for automatic survey generation, semantic search, and research trend tracking.

## Limitations & Future Work

- **Limited coverage**: Only covers the ACL Anthology, excluding a large volume of NLP-related papers from top AI conferences (e.g., NeurIPS, ICML) and preprint servers.
- **Abstract-only analysis**: Paper bodies may contain unique contributions not mentioned in the abstracts; full-text analysis is a necessary next step.
- **Limited model accuracy**: SciBERT's F1 of $0.80$ implies that $\approx 20\%$ of errors propagate to large-scale analyses; while macro trends remain reliable, fine-grained conclusions must be treated with caution.
- **Taxonomy subjectivity**: The taxonomy of 8 subcategories is based on the authors' experience; other researchers might propose alternative taxonomies.

## Related Work & Insights

### vs NLP Contribution Graph (D'Souza & Auer, 2020)

NLP Contribution Graph extracts information units (models, datasets, baselines) associated with predefined NLP tasks, which **do not necessarily represent the original contributions of the paper**, and is limited to specific tasks. This work directly extracts contribution statements from author self-reports, covering both knowledge and artifact categories, yielding a broader scope and finer granularity.

### vs Traditional NLP Scientometrics (Mohammad, 2020; Jurgens et al., 2018)

Traditional methods mainly analyze metadata (citation networks, co-authorship, topic modeling), falling under "external statistics". This work dives into paper contents to **directly analyze the text of the authors' contribution statements**, providing an "internal semantic" perspective with stronger explanatory power for trends.

### vs Citation Intent Analysis (Teufel et al., 2006)

Citation intent analysis understands a paper from the **citing author's perspective** (e.g., background, contrast, usage), whereas this work extracts contribution statements from the **author's own perspective**. The two views are complementary but distinct.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Automatic extraction and classification of contribution statements is a new task; the taxonomy design is original, and the 50-year longitudinal analysis offers a unique perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage with multi-model comparisons, IAA validation, multi-dimensional trend analysis, and citation impact analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Question-driven narrative structure (Q1-Q5), rich figures and tables, and deep insightful discussions.
- **Value**: ⭐⭐⭐⭐ — Highly valuable for understanding the evolution of the NLP field; the datasets and tools can directly support subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [\[ACL 2025\] Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)
- [\[ACL 2025\] Attribution Methods in NLP: Navigating a Fragmented Landscape](attribution_methods_in_nlp_navigating_a_fragmented_landscape.md)
- [\[ACL 2025\] Unveiling Dual Quality in Product Reviews: An NLP-Based Approach](unveiling_dual_quality_in_product_reviews_an_nlp-based_approach.md)
- [\[ACL 2025\] LazyReview: A Dataset for Uncovering Lazy Thinking in NLP Peer Reviews](lazyreview_peer_review.md)

</div>

<!-- RELATED:END -->
