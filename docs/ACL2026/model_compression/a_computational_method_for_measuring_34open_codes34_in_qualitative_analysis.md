---
title: >-
  [Paper Note] A Computational Method for Measuring "Open Codes" in Qualitative Analysis
description: >-
  [ACL 2026][Model Compression][Inductive coding] This paper proposes a theoretically grounded computational framework that employs an LLM-augmented code merging algorithm alongside four ground-truth-free metrics (Coverage…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Inductive coding"
  - "qualitative analysis"
  - "LLM-assisted evaluation"
  - "code space aggregation"
  - "team-based coding evaluation"
date: 2026-05-08
content_hash: d704a04a1e2dc8c3
---

# A Computational Method for Measuring "Open Codes" in Qualitative Analysis

**Conference**: ACL 2026
**arXiv**: [2411.12142](https://arxiv.org/abs/2411.12142)
**Code**: [GitHub](https://github.com/) (open-source package)
**Area**: Model Compression
**Keywords**: Inductive coding, qualitative analysis, LLM-assisted evaluation, code space aggregation, team-based coding evaluation

## TL;DR

This paper proposes a theoretically grounded computational framework that employs an LLM-augmented code merging algorithm alongside four ground-truth-free metrics (Coverage, Overlap, Novelty, and Divergence) to systematically evaluate the performance of both human and AI coders in inductive qualitative coding.

## Background & Motivation

**Background**: Qualitative analysis is a cornerstone methodology in the social sciences for understanding human-generated data. Inductive coding (open coding) requires researchers to discover patterns and themes directly from data without relying on predefined frameworks. As generative AI is increasingly applied to assist coding tasks, reliable evaluation methods have become critically needed.

**Limitations of Prior Work**: Evaluation of inductive coding faces fundamental challenges: (1) ground-truth-dependent metrics (e.g., inter-rater reliability) are at odds with the open-ended nature of inductive coding; (2) clustering and topic coherence metrics emphasize internal homogeneity rather than conceptual breadth; and (3) manual evaluation is costly and difficult to scale.

**Key Challenge**: Inductive coding aims to broadly capture novel insights rather than align with a gold standard, yet existing evaluation methods fail to reflect this characteristic.

**Goal**: To design a theoretically driven, ground-truth-free suite of computational metrics capable of systematically measuring the contribution quality of both human and machine coders in inductive coding.

**Key Insight**: Drawing on team-based coding approaches, the method aggregates results from multiple coders into a shared analytic space, enabling collective-relative evaluation.

**Core Idea**: An LLM-augmented hierarchical clustering algorithm merges codebooks from multiple coders into an Aggregated Code Space (ACS), after which four complementary metrics assess each coder's contribution along distinct dimensions.

## Method

### Overall Architecture

The system operates in two stages: (1) multiple coders' Code Spaces (CSPs) are aggregated into an Aggregated Code Space (ACS) via a four-stage merging algorithm; (2) four evaluation metrics are computed based on the ACS.

### Key Designs

1. **Four-Stage Code Space Merging Algorithm**:

    - **Function**: Merges codes from different coders—potentially phrased differently yet expressing the same concept—into a unified ACS.
    - **Mechanism**: Stage 1 performs naive label merging; Stage 2 applies hierarchical clustering with strict thresholds for label-based merging; Stage 3 introduces LLM-generated definitions, combining labels and definitions for merging; Stage 4 employs dual-threshold iterative merging with a penalty term $penalty$ based on example overlap and unique example counts.
    - **Design Motivation**: A single threshold cannot distinguish between distinct concepts; the dual-threshold and penalty mechanism prevents erroneous merging of different concepts while also avoiding disproportionate influence from small codebooks.

2. **Four Ground-Truth-Free Evaluation Metrics**:

    - **Function**: Assess coder contribution quality along complementary dimensions.
    - **Mechanism**: Coverage measures the weighted breadth of a coder's coverage of the ACS; Overlap measures conceptual agreement with other coders; Novelty quantifies unique contributions (codes discovered by the coder alone); Divergence uses Jensen–Shannon divergence to measure distributional deviation.
    - **Design Motivation**: The dimensions are mutually complementary—high Coverage + high Overlap indicates a reliable coder; high Novelty + low Overlap may signal hallucination; combined interpretation offers greater diagnostic value than any single threshold.

3. **Coder Weight Normalization Mechanism**:

    - **Function**: Prevents metric inflation caused by flooding (excessive code generation).
    - **Mechanism**: Each coder's weight is defined as $w_x = \frac{1}{\ln(size_x)}$, where $size_x$ denotes the number of codes (lower-bounded by the median); coders with more codes receive lower weights.
    - **Design Motivation**: When a coder produces a large number of redundant codes, the contribution of each individual code should be diluted, thereby reflecting genuine quality rather than a quantitative advantage.

### Loss & Training

No model training is involved in this work. The merging algorithm uses cosine distance as the semantic similarity measure; thresholds are selected through interactive validation (strict = 0.32, upper = 0.55). An open-source local model (Gemma3-27B) and an embedding model (mxbai-embed-large) are used to ensure data privacy, with all processing performed locally.

## Key Experimental Results

### Main Results

| Configuration | Coverage Δ | Overlap Δ | Novelty Δ | Divergence Δ |
|---------------|-----------|-----------|-----------|--------------|
| Stage 2 vs. 1 | +0.09% | −0.09% | +0.05% | +0.37% |
| Stage 3 vs. 1 | +3.60% | +5.45% | +0.94% | −4.31% |
| Stage 4 vs. 1 | +7.02% | +7.86% | −1.64% | −1.91% |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| Cross-LLM consistency | CoV < 0.1 | Coefficient of variation across 10 repeated runs is extremely low |
| Model explanatory power | R² > 0.91 | Condition + model + coder identity highly explain metric variance |
| Flooding detection | Coverage = 78.7%, Novelty = 68.1% | Over-coding is effectively identified |
| Hallucination detection | Overlap = 15.6%, Divergence = 75.7% | Hallucinated coding is effectively diagnosed |

### Key Findings

- The four-stage merging algorithm significantly reduces the number of merged codes ($p < 0.001$), while the ranking of the top-5 coders remains stable.
- Three out of four LLMs (Gemma3, QwQ, GPT-4.1) produce highly similar metrics; only Gemini-2.5-Pro exhibits significant deviation.
- Flooding coders show high Coverage but a diminishing-returns effect on Novelty; hallucination coders exhibit sharp declines in both Coverage and Overlap.

## Highlights & Insights

- The combined diagnostic power of the four metrics is substantial: well-functioning coders exhibit a healthy profile of moderate Coverage, reasonable Overlap, appropriate Novelty, and low Divergence.
- The method is entirely ground-truth-free, making it suitable for genuinely exploratory analytic scenarios.
- Stable results are obtained even with small open-source LLMs, which is privacy-friendly as all processing can be performed locally.

## Limitations & Future Work

- Validation is currently limited to a single dataset; testing across more domains and languages is needed.
- Threshold selection still requires interactive human validation and has not yet been fully automated.
- In settings with very few coders (e.g., only two), the statistical power of the metrics may be insufficient.
- Future work may extend the framework to larger-scale multi-round iterative coding workflows.

## Related Work & Insights

- **vs. Ground-truth metrics**: The proposed method requires no predefined correct answers, better aligning with the exploratory nature of inductive coding.
- **vs. Clustering coherence metrics**: Rather than focusing solely on internal consistency, this approach also captures cross-coder complementarity and conceptual coverage breadth.
- **vs. Manual evaluation**: Computational metrics are reproducible, scalable, and diagnostically consistent with manual evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic computational metrics for inductive coding that do not rely on ground truth.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablation studies, robustness analyses, and boundary case detection are all thoroughly validated.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical motivation is clear; algorithmic descriptions are rigorous.
- Value: ⭐⭐⭐⭐ — Offers practical guidance for qualitative analysis and human–AI collaboration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[ACL 2026\] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)
- [\[ACL 2026\] Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis](analytical_ffn-to-moe_restructuring_via_activation_pattern_analysis.md)
- [\[NeurIPS 2025\] DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method](../../NeurIPS2025/model_compression/deltaflow_an_efficient_multi-frame_scene_flow_estimation_method.md)
- [\[NeurIPS 2025\] Towards Unsupervised Open-Set Graph Domain Adaptation via Dual Reprogramming](../../NeurIPS2025/model_compression/towards_unsupervised_open-set_graph_domain_adaptation_via_dual_reprogramming.md)

</div>

<!-- RELATED:END -->
