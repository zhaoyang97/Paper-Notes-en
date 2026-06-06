---
title: >-
  [Paper Note] Aggregate vs. Personalized Judges in Business Idea Evaluation: Evidence from Expert Disagreement
description: >-
  [ACL 2026][LLM Evaluation][LLM-as-a-Judge] Addressing the reality of systematic expert disagreement in business idea evaluation, this study constructs the PBIG-DATA dataset containing 3…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "LLM-as-a-Judge"
  - "Business idea evaluation"
  - "Expert disagreement"
  - "Personalized evaluation"
  - "Multi-dimensional scoring"
date: 2026-05-08
content_hash: 993da4515bb1df9a
---

# Aggregate vs. Personalized Judges in Business Idea Evaluation: Evidence from Expert Disagreement

**Conference**: ACL 2026  
**arXiv**: [2604.22517](https://arxiv.org/abs/2604.22517)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: LLM-as-a-Judge, Business idea evaluation, Expert disagreement, Personalized evaluation, Multi-dimensional scoring

## TL;DR
Addressing the reality of systematic expert disagreement in business idea evaluation, this study constructs the PBIG-DATA dataset containing 3,000 individual expert ratings. It empirically demonstrates that in this domain, a "personalized judge" (conditioned on the historical data of a target reviewer) aligns better with expert behavior than an "aggregate judge" (conditioned on a mixture of histories from multiple reviewers), challenging the common assumption of using pooled labels as the sole ground truth.

## Background & Motivation

**Background**: LLM-as-a-Judge has become the mainstream solution for large-scale evaluation of generation quality. A common practice is to pool labels from multiple reviewers to serve as a single ground truth, forcing the judge model to approximate this pooled signal.

**Limitations of Prior Work**: In scenarios like business idea evaluation that require multi-dimensional judgment (feasibility, novelty, differentiation, market potential, etc.), experts from different backgrounds (technical vs. business) often provide systematically different scores even when using the same rubric. Treating such disagreement as "label noise" and averaging them via pooling erases the genuine heterogeneity of standards.

**Key Challenge**: Standard LLM-as-a-Judge assumes that "an idea has a unique correct score." However, empirical findings show that no such single standard exists for business idea evaluation—the fine-grained ordinal agreement (Krippendorff's $\alpha$) between reviewers is near zero or even negative, yet they show higher consistency in the coarse-grained task of "selecting strong ideas."

**Goal**: To quantify the nature of disagreement using a real-world multi-reviewer dataset and to test which of the two opposing judge designs—aggregate vs. personalized—is better suited to reflect diverse expert judgments.

**Key Insight**: By viewing idea evaluation as a pluralistic assessment problem and acknowledging that "individual reviewers are internally consistent but mutually heterogeneous," one can conclude that "assigning a judge to each reviewer" may be more accurate than forcing convergence to a consensus.

**Core Idea**: The study uses PBIG-DATA to empirically characterize the structure of expert disagreement and compares three judge configurations—zero-shot, aggregate, and personalized—proving that personalized judges align more closely with corresponding reviewers across multiple model sizes.

## Method

### Overall Architecture

The work consists of two parts: (1) Data construction—300 LLM-generated product ideas based on patents covering NLP, CS, and MatChem domains. Each idea is rated by 4–12 domain experts across 6 dimensions using a staged screening protocol (downstream dimensions are not rated if specificity is insufficient), resulting in approximately 3,109 ratings. (2) Judge evaluation—Three judge configurations are run on the same dataset, comparing the alignment between judge predictions and corresponding expert annotations using Krippendorff's $\alpha$.

### Key Designs

1.  **Multi-dimensional + Staged Rating Protocol in PBIG-DATA**:
    -   **Function**: Captures the real-world workflow of "which dimensions to evaluate, what scale to use, and when to skip" in business idea evaluation.
    -   **Mechanism**: Different scales are used for 6 dimensions to match their natural granularity—1-4 for specificity/technical validity/competitive advantage; 1-5 for innovativeness (to distinguish "impressive but not disruptive"); and 0-3 for need validity/market size (where 0 indicates category exclusion for non-B2B products). Staged screening rules: specificity is rated first; technical validity is rated only if specificity exceeds a threshold; innovativeness and competitive advantage follow if technical validity passes; need validity and market size are rated only if specificity passes. "Missingness" is treated as part of the evaluation process rather than noise.
    -   **Design Motivation**: In business reviews, it is common for an idea to be "too vague to discuss feasibility." Forcing reviewers to rate all 6 dimensions for low-quality ideas only introduces noise.

2.  **Comparative Design of Three Judge Configurations**:
    -   **Function**: Isolates the impact of "target signal assumptions" on judge performance under a unified dataset and leave-one-out protocol.
    -   **Mechanism**: (a) Zero-shot judge—provides only rubrics and instructions without expert history; (b) Aggregate judge—few-shot examples are drawn from a mixture of "non-target reviewers" (same domain/dimension, different patents), representing the pooled-label assumption; (c) Personalized judge—few-shot examples come exclusively from the history of the "target reviewer," representing the pluralistic assumption. The only difference is the reviewer attribution of few-shot examples to ensure a fair comparison. Judges also output a confidence score (0-100); following Dong et al. (2024), predictions with confidence < 80 are discarded, and majority voting is used over three random seeds.
    -   **Design Motivation**: To strictly determine whether aggregate or personalized modeling is correct, one must align the number of examples, domains, and sampling logic, varying only the reviewer conditioning.

3.  **Two-tier Measurement for Disagreement Quantification (Fine vs. Coarse)**:
    -   **Function**: Analyzes reviewer consensus at two levels—fine-grained ordinal scores vs. coarse-grained selection of strong ideas.
    -   **Mechanism**: Fine-grained agreement is measured via Krippendorff's $\alpha$ for ordinal scores. Coarse agreement is measured using the Jaccard similarity between the "sets of ideas above each reviewer's median" (calculated only for reviewer pairs with $\ge 10$ co-rated ideas).
    -   **Design Motivation**: Measuring both granularities reveals whether disagreement is systematic structure or pure noise—high disagreement at the fine grain but existing consensus at the coarse grain suggests each reviewer has stable but distinct standards.

### Loss & Training

No training is involved. All judges utilize the Qwen3-Instruct series (4B / 30B-A3B / 30B-A3B-Thinking / 235B-A22B) and GPT-5 mini directly through few-shot prompting. The primary variables are the number of few-shot examples (0 / 1 / 2 / 5 / 10) and their source (target evaluator vs. non-target).

## Key Experimental Results

### Main Results (Expert Disagreement Structure)

| Dimension | Fine $\alpha$ (NLP) | Fine $\alpha$ (CS) | Fine $\alpha$ (Mat) | Coarse Jaccard (NLP) | Coarse Jaccard (Mat) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Specificity | 0.06 | -0.11 | 0.04 | 0.45 | 0.45 |
| Technical validity | -0.03 | -0.40 | -0.28 | 0.50 | 0.42 |
| Innovativeness | 0.33 | 0.47 | 0.46 | 0.71 | 0.54 |
| Competitive advantage | -0.08 | 0.24 | -0.02 | 0.71 | 0.46 |
| Need validity | -0.23 | 0.02 | 0.05 | – | 0.89 |
| Market size | 0.48 | -0.31 | 0.08 | – | 0.57 |

Fine-grained $\alpha$ values for most dimensions are near 0 or negative, while coarse-grained Jaccard scores are significantly higher (0.4-0.9), indicating that disagreement is a structural heterogeneity of "different scales but similar preference structures."

### Judge Alignment Comparison

Dataset: PBIG-DATA 6 dimensions $\times$ 3 domains; Metric: Krippendorff's $\alpha$ between judge predictions and corresponding expert annotations.

| Judge Configuration | Alignment with Target Evaluator | Trend with Few-shot Count |
| :--- | :--- | :--- |
| Zero-shot judge | Baseline (Low) | Unchanged |
| Aggregate judge | Moderate | Slight improvement then plateaus |
| **Personalized judge** | **Highest** | Monotonic improvement with shots |

The gap between Personalized and Aggregate judges becomes more pronounced as model size increases (expanding from 4B $\rightarrow$ 30B $\rightarrow$ 235B).

### Key Findings
-   Across all dimensions and model sizes, personalized judges align more closely with corresponding reviewers than aggregate judges, with the performance gap widening as model capacity increases.
-   Aggregate judges are generally better than zero-shot judges, suggesting that provided expert examples are beneficial; however, pooled examples provide a blurred "target signal" compared to examples from a single reviewer.
-   Agreement between reviewers correlates positively with the similarity of judge-generated reasoning only under personalized conditions—indirectly proving that personalized judges learn "reviewer-specific reasoning styles" rather than universal criteria.

## Highlights & Insights
-   The study explicitly questions the default assumption that "disagreement = noise" and provides a robust counterexample through its dataset and experiments—this problematizing methodology is a contribution in itself.
-   The two-tier agreement metric (fine + coarse) serves as an elegant diagnostic tool: it distinguishes "complete noise" (low at both tiers) from "structural disagreement" (low fine-grained, high coarse-grained), applicable to any subjective evaluation scenario.
-   The staged screening protocol effectively avoids noise from forcing reviewers to rate poor-quality ideas for the sake of completeness; missingness is treated as a signal rather than a flaw.
-   The experimental design using leave-one-out and strict variable control is solid—personalized and aggregate configurations are aligned across shot count, domain, dimension, and model, varying only the reviewer attribution, making the conclusions highly credible.

## Limitations & Future Work
-   The data scale is relatively small (300 ideas / 3,109 ratings); with only dozens of historical entries per reviewer, it is uncertain if LLMs truly "learn" a reviewer's style.
-   Evaluation was limited to the Qwen3 series and GPT-5 mini; transferability to models like Claude or Gemini remains unverified.
-   The study focuses specifically on idea evaluation; whether the conclusions generalize to other subjective tasks (translation, writing, ethical judgment) requires further testing.
-   Deploying personalized judges requires maintaining historical data for each reviewer, which involves non-negligible storage and privacy costs.

## Related Work & Insights
-   **vs. Dong et al. 2024 (personalized judging)**: That work first proposed the personalized judging approach; this paper rigorously validates it in extreme scenarios of systematic low agreement, finding the performance gap to be even more significant.
-   **vs. Hämäläinen & Alnajjar 2021**: Early NLG evaluation literature acknowledged low agreement but often treated it as an unavoidable limitation; this work transforms that heterogeneity into a specific judge design choice.
-   **vs. Si et al. 2025**: While that work documented disagreements among NLP researchers evaluating LLM ideas, this paper further investigates how a judge should model such disagreements.

## Rating
-   Novelty: ⭐⭐⭐⭐ Problematizes the "disagreement is noise" assumption with empirical refutation; a methodological contribution.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Rigorous control of variables across 6 dimensions $\times$ 3 domains $\times$ 4 model sizes.
-   Writing Quality: ⭐⭐⭐⭐ Clear progression of motivation; tight logical chain between disagreement diagnosis and judge design.
-   Value: ⭐⭐⭐⭐ Methodological implications for all multi-rater evaluation systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Stability vs. Manipulability: Evaluating Robustness Under Post-Decision Interaction in LLM Judges](stability_vs_manipulability_evaluating_robustness_under_post-decision_interactio.md)
- [\[ACL 2026\] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation](teaching_language_models_to_forecast_research_success_through_comparative_idea_e.md)
- [\[ACL 2026\] K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology](k-metbench_a_multi-dimensional_benchmark_for_fine-grained_evaluation_of_expert_r.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)

</div>

<!-- RELATED:END -->
