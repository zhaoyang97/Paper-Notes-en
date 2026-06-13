---
title: >-
  [Paper Note] A Computational Method for Measuring "Open Codes" in Qualitative Analysis
description: >-
  [ACL 2026][NLP Understanding][Inductive coding] A theory-based computational method is proposed to systematically evaluate human and AI performance in inductive qualitative coding through an LLM-enhanced code merging alg…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Inductive coding"
  - "Qualitative analysis"
  - "LLM-assisted evaluation"
  - "Code space aggregation"
  - "Team-based evaluation"
date: 2026-05-08
content_hash: 6fcd0916ed23d5da
---

# A Computational Method for Measuring "Open Codes" in Qualitative Analysis

**Conference**: ACL 2026  
**arXiv**: [2411.12142](https://arxiv.org/abs/2411.12142)  
**Code**: [GitHub](https://github.com/) (Open-source package)  
**Area**: Model Compression  
**Keywords**: Inductive coding, Qualitative analysis, LLM-assisted evaluation, Code space aggregation, Team-based evaluation

## TL;DR

A theory-based computational method is proposed to systematically evaluate human and AI performance in inductive qualitative coding through an LLM-enhanced code merging algorithm and four ground-truth-free metrics (Coverage, Overlap, Novelty, and Divergence).

## Background & Motivation

**Background**: Qualitative analysis is a core method in social sciences for understanding human data. Within this, inductive coding (open coding) requires researchers to discover patterns and themes directly from the data rather than relying on preset frameworks. As generative AI is increasingly used to assist in coding tasks, reliable evaluation methods are urgently needed.

**Limitations of Prior Work**: Evaluation of inductive coding faces fundamental dilemmas: (1) Ground-truth-based metrics (e.g., inter-rater reliability) contradict the open-ended nature of inductive coding; (2) Clustering or topic consistency metrics focus on internal homogeneity rather than conceptual breadth; (3) Manual evaluation is costly and difficult to scale.

**Key Challenge**: Inductive coding pursues the "broad capture of novel insights" rather than "consistency with a standard answer," which existing evaluation methods fail to reflect.

**Goal**: Design a set of theory-driven, ground-truth-free computational metrics that can systematically measure the quality of contributions from both human and machine coders in inductive coding.

**Key Insight**: Borrowing from team-based approach in qualitative research, the results of multiple coders are aggregated into a shared analysis space, enabling collective-based relative evaluation.

**Core Idea**: Aggregate the codebooks of multiple coders into an Aggregated Code Space (ACS) using an LLM-enhanced hierarchical clustering algorithm, and then use four complementary metrics to measure each coder's contribution from different dimensions.

## Method

### Overall Architecture

The system works in two steps: (1) Aggregating the Code Spaces (CSP) of multiple coders into an Aggregated Code Space (ACS) via a four-stage merging algorithm; (2) Calculating four evaluation metrics based on the ACS.

### Key Designs

1.  **Four-Stage Code Space Merging Algorithm**:

    *   **Function**: Merges codes from different coders, which may use different wording for the same concept, into a unified ACS.
    *   **Mechanism**: Stage 1 involves naive label merging; Stage 2 uses hierarchical clustering by labels with strict thresholds; Stage 3 introduces LLM-generated definitions to merge based on both labels and definitions; Stage 4 utilizes dual-threshold iterative merging and adds a $penalty$ term based on example overlap and the number of unique examples.
    *   **Design Motivation**: A single threshold struggles to distinguish between different concepts. The dual-threshold and penalty mechanism prevents the erroneous merging of distinct concepts and avoids disproportionate influence from small codebooks.

2.  **Four Ground-Truth-Free Evaluation Metrics**:

    *   **Function**: Measures the quality of coder contributions across different dimensions.
    *   **Mechanism**: Coverage measures the breadth of the ACS covered by a coder (weighted); Overlap measures conceptual consistency with others; Novelty measures unique contributions (codes discovered only by oneself); Divergence uses Jensen-Shannon divergence to measure the degree of distribution deviation.
    *   **Design Motivation**: These dimensions are complementary—high Coverage combined with high Overlap indicates a reliable coder; high Novelty with low Overlap may suggest hallucinations. Combined interpretations offer more diagnostic value than single thresholds.

3.  **Coder Weight Normalization Mechanism**:

    *   **Function**: Prevents metric inflation caused by over-coding (flooding).
    *   **Mechanism**: The weight for each coder is $w_x = \frac{1}{\ln(size_x)}$, where $size_x$ represents their code count (with a lower bound at the median). A higher number of codes results in a lower weight per code.
    *   **Design Motivation**: If a coder produces a large number of redundant codes, the contribution of each individual code should be diluted to reflect true quality rather than quantitative dominance.

### Loss & Training

This work does not involve model training. The merging algorithm uses cosine distance as a measure of semantic similarity, with thresholds selected via interactive validation (strict=0.32, upper=0.55). Open-source local models (Gemma3-27B) and embedding models (mxbai-embed-large) are used to ensure data privacy.

## Key Experimental Results

### Main Results

| Configuration | Coverage Change | Overlap Change | Novelty Change | Divergence Change |
| :--- | :--- | :--- | :--- | :--- |
| Stage 2 vs 1 | +0.09% | -0.09% | +0.05% | +0.37% |
| Stage 3 vs 1 | +3.60% | +5.45% | +0.94% | -4.31% |
| Stage 4 vs 1 | +7.02% | +7.86% | -1.64% | -1.91% |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Cross-LLM Consistency | CoV < 0.1 | Extremely low coefficient of variation across 10 repeated runs |
| Model Explanatory Power | R² > 0.91 | Variations in metrics are highly explained by condition, model, and coder identity |
| Flooding Detection | Coverage=78.7%, Novelty=68.1% | Over-coding is effectively identified |
| Hallucination Detection | Overlap=15.6%, Divergence=75.7% | Hallucinated coding is effectively diagnosed |

### Key Findings
*   The four-stage merging algorithm significantly reduces the number of codes after merging ($p < 0.001$), while the ranking of the top 5 coders remains stable.
*   Three out of four LLMs (Gemma3, QwQ, GPT-4.1) produce highly similar metrics, with only Gemini-2.5-Pro showing significant deviation.
*   While "flooding" coders have high Coverage, their Novelty shows a diminishing effect; "hallucination" coders show a sharp drop in both Coverage and Overlap.

## Highlights & Insights
*   The combined diagnostic capability of the four metrics is powerful: a normal coder presents a healthy pattern of "moderate Coverage + reasonable Overlap + modest Novelty + low Divergence."
*   The method is entirely independent of ground truth, making it suitable for real-world exploratory analysis scenarios.
*   Stable results can be obtained even with small-scale open-source LLMs, making it friendly to data privacy (all processing can be done locally).

## Limitations & Future Work
*   Validation has currently been performed on only one dataset; testing across more domains and languages is required.
*   Threshold selection still requires manual interactive validation and has not yet achieved full automation.
*   In scenarios with very few coders (e.g., only 2), the statistical power of the metrics may be insufficient.
*   Future work could extend this to larger-scale multi-round iterative coding workflows.

## Related Work & Insights
*   **vs Ground-truth metrics**: This method does not require a preset correct answer, aligning better with the exploratory nature of inductive coding.
*   **vs Clustering consistency metrics**: It focuses not only on internal consistency but also on complementarity and conceptual coverage across coders.
*   **vs Manual evaluation**: The computational metrics are repeatable, scalable, and align with the diagnostic direction of manual evaluation.

## Rating
*   **Novelty**: ⭐⭐⭐⭐ First to propose systematic computational metrics for inductive coding independent of ground truth.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Sufficient validation through ablation, robustness, and boundary case detection.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear theoretical motivation and rigorous algorithm description.
*   **Value**: ⭐⭐⭐⭐ Provides practical guidance for the collaboration between qualitative analysis and AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DimABSA: Building Multilingual and Multidomain Datasets for Dimensional Aspect-Based Sentiment Analysis](dimabsa_building_multilingual_and_multidomain_datasets_for_dimensional_aspect-ba.md)
- [\[ACL 2026\] MSMO-ABSA: Multi-Scale and Multi-Objective Optimization for Cross-Lingual Aspect-Based Sentiment Analysis](msmo-absa_multi-scale_and_multi-objective_optimization_for_cross-lingual_aspect-.md)
- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](../../NeurIPS2025/nlp_understanding/generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)
- [\[ACL 2026\] Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering](beyond_chunking_discourse-aware_hierarchical_retrieval_for_long_document_questio.md)
- [\[ACL 2026\] MetFuse: Figurative Fusion between Metonymy and Metaphor](metfuse_figurative_fusion_between_metonymy_and_metaphor.md)

</div>

<!-- RELATED:END -->
