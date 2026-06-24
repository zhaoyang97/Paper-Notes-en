---
title: >-
  [Paper Note] Deprecating Benchmarks: Criteria and Framework
description: >-
  [ICML 2025][Recommender Systems][Benchmark Deprecation] Proposes a set of **7 criteria** to determine when an AI benchmark should be deprecated, alongside a three-phase **deprecation framework** (Assessment-Reporting-Notification), and provides an institutional implementation plan using the EU AI Office as a case study.
tags:
  - "ICML 2025"
  - "Recommender Systems"
  - "Benchmark Deprecation"
  - "Evaluation Lifecycle"
  - "AI Governance"
  - "Benchmark Saturation"
  - "Data Contamination"
date: 2026-05-08
content_hash: fe8a30c17481ccc9
---

# Deprecating Benchmarks: Criteria and Framework

**Conference**: ICML 2025  
**arXiv**: [2507.06434](https://arxiv.org/abs/2507.06434)  
**Code**: None  
**Area**: Recommendation Systems  
**Keywords**: Benchmark Deprecation, Evaluation Lifecycle, AI Governance, Benchmark Saturation, Data Contamination

## TL;DR

Proposes a set of **7 criteria** to determine when an AI benchmark should be deprecated, alongside a three-phase **deprecation framework** (Assessment-Reporting-Notification), and provides an institutional implementation plan using the EU AI Office as a case study.

## Background & Motivation

With the rapid advancement of frontier AI model capabilities, benchmarks serve as the primary means to evaluate and compare model performance, and are increasingly integrated into compliance requirements such as the EU AI Act. However, the current benchmarking ecosystem faces severe challenges:

**Benchmark Inertia**: Many benchmarks persist historically, despite being no longer valid. For instance, ImageNet became a de facto standard post-AlexNet, hindering the adoption of superior alternatives (the Benchmark Lottery problem).

**Distorted Commercial Incentives**: AI companies lack incentives to deprecate benchmarks that favor them. Meta's LLaMA 4 fine-tuning on conversational benchmarks (benchmark gaming) serves as a typical example.

**Safety-washing Risks**: Outdated or flawed benchmarks can inflate model capabilities and obscure safety vulnerabilities, conveying false signals to the public and regulators.

**Lack of Guidance**: Currently, there are no systematic criteria or processes to guide when and how to deprecate benchmarks.

The core argument of this work is that **outdated or flawed benchmarks must be proactively deprecated** to prevent distorted capability evaluations, waste of evaluation resources, and safety-washing.

## Method

### Overall Architecture

The contributions of this work are divided into two main parts: **Deprecation Criteria** (Criteria) and the **Deprecation Framework** (Framework).

```
Deprecation Criteria (Section 3)          Deprecation Framework (Section 4)
┌─────────────────┐                        ┌─────────────────────┐
│ 1. Saturation   │                        │ Phase 1: Assessment │
│ 2. Contamination│          →             │ Phase 2: Reporting  │
│ 3. Stat. Bias   │                        │ Phase 3: Notification│
│ 4. Label Errors │                        └─────────────────────┘
│ 5. Obsolescence │                                   ↓
│ 6. Invalid Assum│                        ┌─────────────────────┐
│ 7. Seman. Drift │                        │ Implementation (S5) │
└─────────────────┘                        │ EU AI Office Example│
                                           └─────────────────────┘
```

### Key Designs

#### Seven Deprecation Criteria

The authors classify the criteria into two major categories: **Quantitative Signals** and **Qualitative Issues**:

| # | Criterion | Category | Description | Typical Case |
|---|---|---|---|---|
| 1 | **Saturation** | Quantitative | Model performance approaches or reaches the upper limit of the evaluation, making further improvements indistinguishable | MMLU, GSM8K, HumanEval |
| 2 | **Contamination** | Quantitative | Models memorize benchmark content due to data leakage, so performance no longer reflects true generalization ability | Multiple LLM benchmark leakage incidents |
| 3 | **Statistical Bias** | Quantitative | Class imbalance leads models to exploit shortcuts in data distribution rather than demonstrating target capabilities | Majority of normal samples dominating in anomaly detection benchmarks |
| 4 | **High Label Error Rate** | Qualitative | Errors introduced by annotators compromise benchmark data quality | 57% of questions in the MMLU virology subset are incorrect |
| 5 | **Task Obsolescence** | Qualitative | The task itself no longer carries evaluation significance or has been solved | Word-from-letters task in BIG-bench |
| 6 | **Invalidated Assumptions** | Qualitative | Simplifying assumptions in benchmark design no longer hold | Single-fact retrieval in Needle-in-a-Haystack vs. multi-information reasoning in realistic RAG |
| 7 | **Semantic Drift** | Qualitative | Task meaning or label interpretation changes over time | Linguistic/cultural contexts frozen at dataset creation |

The authors emphasize that these criteria are **non-exhaustive, soft heuristics**, analogous to case law, evolving alongside the accumulation of community experience.

#### Three-Phase Deprecation Framework

**Phase 1: Assessment**

- Evaluate whether deprecation is needed using the aforementioned 7 criteria.
- Determine the deprecation tier: **Partial Deprecation** (updating/upgrading valid components) vs. **Total Deprecation**.
- Statistical bias, data contamination $\rightarrow$ Suitable for partial deprecation (resampling, removing leaked data).
- Task obsolescence, invalidated assumptions $\rightarrow$ Usually require total deprecation.
- Label errors $\rightarrow$ Depends on the scale and impact of the errors.
- Establish a formal **appeal process** allowing benchmark developers to challenge deprecation decisions.

**Phase 2: Reporting**

- Deprecation reports should contain: reasons for deprecation and evidence of risk, future usage guidelines (total/partial), implementation timeline, alternative benchmark recommendations, methodology for interpreting historical results, and authorized use terms (for research/archival purposes).
- Using **SWE-bench v1.0** and **SWE-bench Lite v1.0** as case studies, this paper provides two semi-fictional deprecation report templates.

**Phase 3: Notification**

- Publish deprecation notices on the original distribution channels.
- Visual markers similar to academic retraction notices.
- Direct notification to key users (e.g., evaluators of safety-critical systems).
- Utilize version control to distinguish between original and modified versions.

### Loss & Training

Since this is a position and framework paper, it does not involve model training. The core "strategy" is manifested in the institutional implementation plan, modeled after the **EU AI Office (AIO)**:

- **Assessment**: The AIO compiles and periodically reviews commonly used benchmarks for safety-critical tasks (such as CBRN capabilities), producing a deprecation list.
- **Reporting**: The AIO creates a deprecation report for each benchmark, detailing the deprecation decision, rationale, timeline, alternatives, and guidelines for interpreting historical results.
- **Notification**: The AIO contacts AI competent authorities in member states, requiring commercially deployed models to update their model cards and technical reports within a specified timeframe.

## Key Experimental Results

### Main Results

This is a framework paper with no traditional experiments. The core arguments are derived from quantitative evidence in the literature survey:

| Benchmark | Issue | Key Data | Source |
|---|---|---|---|
| MMLU (Full) | Label error | 6.49% of questions are erroneous | Gema et al. 2024 |
| MMLU Virology Subset | Label error | 57% of questions are erroneous | Gema et al. 2024 |
| SWE-bench v1.0 | Multiple defects | 68.3% of samples filtered due to issues | Chowdhury et al. 2024 |
| SWE-bench Lite v1.0 | Problem description quality | 4.3% contain complete ground truth; 10% lack key information; 5% contain misleading solutions | Xia et al. 2024 |
| MMLU / GSM8K / HumanEval | Saturation | Frontier models approaching upper limits | Maslej et al. 2025 |

### Ablation Study

Decision guide using deprecation tiers (Partial vs. Total) as the "ablation variable":

| Deprecation Criterion | Recommended Deprecation Tier | Description |
|---|---|---|
| Saturation | Case-by-case | Partially deprecate if difficulty can be scaled, otherwise totally deprecate |
| Data Contamination | Partial Deprecation | Remove leaked data, introduce random variables |
| Statistical Bias | Partial Deprecation | Resample to balance class distributions |
| High Label Error Rate (Minor) | Partial Deprecation | Correct labels and document changes |
| High Label Error Rate (Severe) | Total Deprecation | Too many errors to repair |
| Task Obsolescence | Total Deprecation | The task itself loses evaluation value |
| Invalidated Assumptions | Total Deprecation | Benchmark design is out of touch with reality |
| Semantic Drift | Case-by-case | Crucial to evaluate the degree and impact of drift |

### Key Findings

1. **Widespread Benchmark Issues**: Reuel et al. (2024) analyzed 24 commonly used benchmarks and found "significant defects even in mainstream benchmarks."
2. **Commercial Incentives Hinder Deprecation**: AI companies tend to retain benchmarks that favor their products, and there is no regulatory requirement to re-evaluate benchmark validity.
3. **Frontier Models Lower Deprecation Costs**: For frontier models, benchmarks are primarily test-time artifacts requiring no explicit training, allowing new benchmarks to be immediately applied to existing models, which reduces the cost of deprecation.
4. **Third-Party Deprecation is Crucial**: When benchmark developers are unable or unwilling to deprecate, governance bodies must assume this responsibility.

## Highlights & Insights

1. **Framing Benchmark Deprecation as a Governance Issue**: It is not merely a technical problem but an institutional issue involving regulatory compliance (EU AI Act), safety assessment, and public trust.
2. **The Concept of Partial Deprecation**: Provides a more flexible practical path than an "all-or-nothing" approach, allowing valuable components to be retained.
3. **Appeal Mechanism Design**: Draws on the due process principles of legal proceedings to ensure the fairness of deprecation decisions.
4. **SWE-bench Case Study Reports**: Showcases the practical preparation of deprecation reports through concrete, actionable templates.
5. **Analogy to Academic Retractions**: Analogizes benchmark deprecation notices to paper retraction notices, providing a clear reference for practice.

## Limitations & Future Work

1. **Lack of Quantitative Thresholds**: The descriptions of criteria remain qualitative without providing explicit numerical thresholds (e.g., how should "saturation" be defined in terms of score?).
2. **Focus Solely on Frontier Models**: Insufficient discussion on deprecating benchmarks for non-frontier models (e.g., domain-specific or smaller models).
3. **Questionable Enforceability**: The framework relies on the proactive intervention of governance bodies, but lacks corresponding enforcement mechanisms in practice.
4. **No Discussion on Benchmark Alternatives**: Focuses only on "when to deprecate" without systematically discussing "what to replace them with."
5. **Cross-Cultural Applicability**: The framework is primarily tailored to the EU context as its deployment scenario, leaving its applicability to other legal systems unaddressed.
6. **Lack of Empirical Validation**: The two SWE-bench deprecation reports are semi-fictional case studies and have not been validated in real-world governance processes.

## Related Work & Insights

- **Luccioni et al. (2022)**: The most direct precursor, which proposed a dataset deprecation framework. This work builds upon it by focusing specifically on benchmarks, adding the concept of partial deprecation, and offering governance-level recommendations.
- **Reuel et al. (2024) BetterBench**: Evaluated the quality of 24 benchmarks and revealed widespread flaws, providing an empirical foundation for this study.
- **Eriksson et al. (2025)**: Argued that current benchmarks are fragile risk assessment tools and raised the critical question of "which benchmarks to trust."
- **Raji et al. (2021)**: Criticized the context-detached, broad application of benchmarks, emphasizing the situated nature of evaluations.
- **Ren et al. (2024) SafetyWashing**: Defined and analyzed the phenomenon of safety-washing, providing a safety-centric argument for the urgency of deprecation.
- **Insights**: Can serve as a theoretical basis for discussing the rationality of evaluation protocols in papers/projects; the deprecation criteria checklist can be directly applied to scrutinize the benchmarks one uses.

## Rating

- **Novelty**: ★★★☆☆ — An incremental advancement over dataset deprecation (Luccioni 2022), focusing on benchmarks and the governance layer.
- **Utility**: ★★★★☆ — The criteria checklist and deprecation report templates offer direct reference value.
- **Rigor**: ★★★☆☆ — A framework paper lacking empirical validation and quantitative thresholds.
- **Clarity**: ★★★★☆ — Well-structured; the SWE-bench case study enhances actionability.
- **Impact**: ★★★★☆ — Potential policy-level influence in the context of growing attention to AI governance.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Measuring What Matters: Construct Validity in Large Language Model Benchmarks](../../NeurIPS2025/recommender/measuring_what_matters_construct_validity_in_large_language_model_benchmarks.md)
- [\[NeurIPS 2025\] FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens](../../NeurIPS2025/recommender/face_a_general_framework_for_mapping_collaborative_filtering_embeddings_into_llm.md)
- [\[ACL 2026\] SenseJudge: Human-Centric Preference-Driven Judgment Framework](../../ACL2026/recommender/sensejudge_human-centric_preference-driven_judgment_framework.md)
- [\[ACL 2026\] Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework](../../ACL2026/recommender/personalizing_llms_with_binary_feedback_a_preference-corrected_optimization_fram.md)
- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](../../AAAI2026/recommender/tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)

</div>

<!-- RELATED:END -->
