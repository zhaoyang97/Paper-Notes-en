---
title: >-
  [Paper Note] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models
description: >-
  [Medical Imaging] This position paper systematically reviews the current state of LLM-assisted thematic analysis (TA) on unstructured clinical transcripts…
tags:
  - "Medical Imaging"
date: 2026-05-08
content_hash: 7726086b30b67c59
---

# Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models

## Metadata
- **Conference**: NeurIPS 2025
- **arXiv**: [2509.14597](https://arxiv.org/abs/2509.14597)
- **Code**: Not available
- **Area**: Medical Imaging
- **Keywords**: Thematic Analysis, Large Language Models, Clinical Transcripts, Evaluation Framework, Qualitative Research

## TL;DR
This position paper systematically reviews the current state of LLM-assisted thematic analysis (TA) on unstructured clinical transcripts, identifies highly fragmented evaluation practices across the literature, and proposes a standardized evaluation framework centered on three dimensions: Validity, Reliability, and Interpretability.

## Background & Motivation

Thematic Analysis (TA) is one of the most widely used methods in qualitative data analysis, commonly applied in clinical settings to extract meaningful patterns from patient interview transcripts. However, manual TA faces severe scalability bottlenecks:

**Substantial labor costs**: Over 900,000 medical interviews in the United States annually require TA, the majority of which involve unstructured data necessitating Inductive Thematic Analysis (ITA). Manual ITA demands more than 6.1 million person-hours per year, equivalent to 3,000 full-time positions at a cost of $305.4 million.

**Efficiency gap**: Human analysts require 5–8 hours per transcript, whereas LLMs can complete preliminary coding and theme generation within 10 minutes.

**Evaluation fragmentation**: Across 56 identified studies, there is substantial inconsistency in analysis types, model selection, prompting strategies, and evaluation methods, which impedes cross-study comparison and reproducibility.

**Core argument**: Establishing standardized evaluation practices is critical to advancing LLM-assisted TA; the current fragmentation of evaluation methods constitutes the primary obstacle.

## Method

### Overall Architecture

A mixed-methods approach combining a systematic literature review with expert interviews is adopted. Studies are identified via arXiv metadata search (up to August 15, 2025) supplemented by platforms such as Elicit, resulting in 56 LLM-assisted TA studies published over the past three years, analyzed along five dimensions. A 2-hour in-depth interview with a cardiac surgeon is also incorporated.

### Key Designs

1. **Five-dimensional systematic analysis**:

    - **TA type**: Inductive (64%) is dominant, followed by hybrid (22%), with purely deductive comprising only 9%. The two types require different evaluation approaches, representing a major source of evaluation inconsistency.
    - **Model distribution**: The GPT series (58%) accounts for an overwhelming majority, with Claude (13%), LLaMA (11%), and Gemini (11%) as secondary options. A small number of studies employ fine-tuned or specialized deployments.
    - **Data domain**: Social media (25%), education (21%), software engineering (20%), and healthcare/clinical (16%). Clinical applications are underrepresented.
    - **Prompting strategy**: Zero-shot (35%) is the most common, followed by few-shot (16%), chain-of-thought (13%), and self-consistency/reflection (15%). Zero-shot dominates due to ease of use and natural alignment with inductive TA.
    - **Evaluation methods**: Human qualitative review (40%), automated text metrics (27%), task-oriented evaluation (13%), and hybrid approaches (20%).

2. **In-depth analysis of evaluation fragmentation**:

    - Even when identical metrics are used, differences in underlying embedding models undermine comparability.
    - Many studies do not release complete ground truth data, often citing privacy or IRB restrictions.
    - The lack of one-to-one mapping between human-generated and LLM-generated themes inherently limits direct similarity comparisons.

3. **Three-dimensional evaluation framework (core contribution)**:

   **Validity**: Maximum weighted bipartite matching is computed over a similarity matrix $\mathbf{S}$, where $S_{ij}$ denotes the similarity between human theme $i$ and model theme $j$. Both lexical overlap (Jaccard, ROUGE) and semantic similarity (cosine, BERTScore) are employed. Reported metrics include: Precision/Recall/F1@match, Coverage@$\tau$ (proportion of human themes matched above a threshold), Redundancy (mean internal similarity among LLM themes), and Novelty rate (proportion of LLM themes without a matching human theme).

   **Reliability**: Krippendorff $\alpha$ or Cohen/Fleiss $\kappa$ serve as diagnostic tools. The pipeline is re-run with different random seeds or bootstrap samples to compute the Adjusted Rand Index (ARI) or Variation of Information (VI) for stability assessment. Theme-level alignment is achieved by comparing overlaps in supporting text segments. Confirmability assessment examines whether themes are data-driven or influenced by intrinsic LLM biases.

   **Interpretability**: Embedding similarity is used to assess coherence (mean similarity of citations within the same theme) and distinctiveness (distance between theme centroids). Per-theme passage coverage and participant representativeness are reported. Human-in-the-loop validation is incorporated to ensure domain-specific credibility.

### Cost Analysis

LLM inference costs have decreased substantially to approximately \$0.15–\$2.50 per million input tokens, far below the \$200,000+ typically required for manual TA.

## Key Experimental Results

### Literature Distribution Statistics

| Dimension | Category | Count/Proportion |
|-----------|----------|-----------------|
| TA type | Inductive | 36/64% |
| TA type | Hybrid | 12/22% |
| Model | GPT series | 32/58% |
| Model | Claude | 7/13% |
| Data domain | Social media | 14/25% |
| Data domain | Healthcare/Clinical | 9/16% |
| Prompting strategy | Zero-shot | 19/35% |
| Evaluation | Human qualitative review | 22/40% |
| Evaluation | Automated text metrics | 16/27% |

### Comparative Analysis of Evaluation Methods

| Evaluation Approach | Strengths | Limitations |
|--------------------|-----------|-------------|
| Human qualitative review (40%) | Flexible; captures nuance | Poor reproducibility; subjective; time-consuming |
| Automated text metrics (27%) | Quantifiable; efficient | Incomparable across embedding models; ignores semantic depth |
| Task-oriented (13%) | Assesses practical utility | Indirect measurement; difficult to standardize |
| Hybrid evaluation (20%) | Balances qualitative and quantitative | Complex design; inconsistent standards |

### Key Findings

1. **LLM-assisted TA is an emerging but rapidly growing research direction**: The first work appeared after the release of GPT-3.5 (November 2022), with exponential growth in publications thereafter.
2. **Clinical domain applications are severely underrepresented**: Only 16% of studies involve healthcare/clinical data, misaligned with the substantial practical demand.
3. **Step 1 (familiarization) is the primary bottleneck**: Full transcript review consumes the most time, and most existing approaches still require complete human-in-the-loop review.
4. **Evaluation amounts to "shooting in the dark"**: Clinical expert interviews directly highlighted that the absence of clear ground truth in inductive TA makes evaluation inherently difficult.

## Highlights & Insights

1. **Identifies the core pain point**: It is not the methods themselves, but the absence of evaluation standards that impedes progress across the field.
2. **Elegant three-dimensional framework design**: Validity employs bipartite matching to address the non-one-to-one mapping problem; Reliability introduces cross-LLM confirmability checks; Interpretability quantifies coherence and distinctiveness via embedding spaces.
3. **Incorporation of clinical expert perspective**: A 2-hour in-depth interview provides the practitioner viewpoint frequently absent in methodological research.
4. **Cost analysis quantifies value**: The contrast between LLM inference costs and manual TA costs (<\$10 vs. \$200,000+) strongly supports the necessity of automation.

## Limitations & Future Work

- As a position paper, no empirical validation is conducted; the three-dimensional framework remains theoretical.
- Literature retrieval is primarily arXiv-based, potentially omitting important journal publications.
- The operationalizability and clinical deployment pathway of the proposed framework remain unclear.
- The paper does not thoroughly discuss how LLM-specific biases affect the diversity of inductively generated themes.
- Concrete implementation pathways for end-to-end automated TA pipelines are not provided.

## Related Work & Insights

- **Braun & Clarke's six-phase approach**: Familiarization → Initial coding → Searching for themes → Reviewing themes → Defining and naming → Writing up.
- **Auto-TA**: Scalable TA via multi-agent LLMs with reinforcement learning (prior work by the same team).
- **ProtoMed-LLM**: An automated LLM evaluation framework for medical protocol development.
- **TAMA**: Human-AI collaborative thematic analysis.
- **Insights**: Evaluation standardization is a challenge shared across all LLM-assisted qualitative research; the proposed framework is potentially generalizable to other qualitative methods such as content analysis and grounded theory.

## Rating
- Novelty: ⭐⭐⭐⭐☆ — First systematic review of LLM-TA evaluation challenges with a unified framework
- Experimental Thoroughness: ⭐⭐☆☆☆ — Position/survey paper; no empirical validation
- Writing Quality: ⭐⭐⭐⭐☆ — Clear structure; well-argued
- Value: ⭐⭐⭐⭐☆ — Charts a clear direction for evaluation in LLM-assisted qualitative research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/medical_imaging/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](../../AAAI2026/medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[ICLR 2026\] Protein as a Second Language for LLMs](../../ICLR2026/medical_imaging/protein_as_a_second_language_for_llms.md)
- [\[NeurIPS 2025\] RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification](raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica.md)
- [\[NeurIPS 2025\] LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation](llm-assisted_emergency_triage_benchmark_bridging_hospital-rich_and_mci-like_fiel.md)

</div>

<!-- RELATED:END -->
