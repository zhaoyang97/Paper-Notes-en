---
title: >-
  [Paper Note] Evaluating LLMs for Police Decision-Making: A Framework Based on Police Action Scenarios
description: >-
  [AAAI 2026][Recommender Systems][LLM Evaluation] This paper proposes PAS (Police Action Scenarios), an LLM evaluation framework for policing contexts. The framework comprises five stages: scenario definition…
tags:
  - "AAAI 2026"
  - "Recommender Systems"
  - "LLM Evaluation"
  - "Police Decision-Making"
  - "Domain-Specific Evaluation"
  - "Scenario-Based Evaluation"
  - "LLM-as-Judge"
date: 2026-05-08
content_hash: b9a693a3512eacad
---

# Evaluating LLMs for Police Decision-Making: A Framework Based on Police Action Scenarios

**Conference**: AAAI 2026
**arXiv**: [2601.03553](https://arxiv.org/abs/2601.03553)  
**Code**: [https://github.com/Heedou/PASFramework](https://github.com/Heedou/PASFramework)  
**Area**: Recommender Systems
**Keywords**: LLM Evaluation, Police Decision-Making, Domain-Specific Evaluation, Scenario-Based Evaluation, LLM-as-Judge

## TL;DR
This paper proposes PAS (Police Action Scenarios), an LLM evaluation framework for policing contexts. The framework comprises five stages: scenario definition, reference answer construction, LLM response generation, core metric extraction, and performance interpretation. An evaluation dataset is constructed from 8,000+ official Korean police documents. The study finds that commercial LLMs (GPT-4, Gemini, Claude) perform significantly below reference answers on policing tasks, particularly in factual accuracy and logical correctness.

## Background & Motivation

LLMs are increasingly applied in law enforcement (traffic accident analysis, automated report generation, criminal investigation, etc.), yet a **critical validation gap** persists:

- **Limitations of existing evaluation**: Traditional evaluations focus solely on informational accuracy (e.g., statute matching, crime classification), failing to capture key policing requirements—situational adaptability, legal procedural compliance, and soundness of judgment.
- **Practical risks**: LLM-generated responses may not be legally "incorrect," yet direct deployment without validation can lead to serious consequences such as unlawful arrests and improper evidence collection.
- **Known bias issues**: LLM dispatch recommendations have been shown to exhibit geographic and racial biases.

**Key Challenge**: No systematic LLM evaluation framework specifically designed for the policing domain exists. Existing benchmarks (BLEU, ROUGE, multiple-choice) cannot assess reasoning quality or operational compliance in policing scenarios.

**Core Idea**: Establish PAS, a five-stage scalable evaluation framework that uses real policing scenarios as evaluation vehicles and employs a police expert-validated metric selection process to identify indicators that genuinely reflect task quality in law enforcement.

## Method

### Overall Architecture

The PAS framework is formalized as $E_{police} = f(S, R, G, M, P)$, comprising five stages:
1. $S$ (Scenario Definition) → 2. $R$ (Reference Answers) → 3. $G$ (LLM Response Generation) → 4. $M$ (Core Metrics) → 5. $P$ (Performance Interpretation)

### Key Designs

1. **Scenario Definition ($S$) — Dual-Dimensional Design of Situation × Action**:

    - **Function**: Defines the task vehicle for policing evaluation.
    - **Mechanism**: Scenarios are decomposed into Situation (department, role, task, incident type) and Action (legal judgment, case classification, report writing, citizen interaction, etc.).
    - **Design Motivation**: Grounded in legal definitions of police duties and policing model theory to ensure evaluation coverage across diverse real-world contexts.
    - In this work, scenarios are instantiated as **pre-shift mental training**—simulating officers mentally rehearsing responses to various incidents before duty.

2. **Two-Stage Metric Selection**:

    - **Function**: Filters 5 core metrics from 15 candidates that are genuinely relevant to policing task quality.
    - **Mechanism**:
      * **Stage 1 — Multiple Regression**: Which metrics significantly predict overall quality scores ($p < .05$)?
      * **Stage 2 — Expert Correlation Validation**: Are the filtered metrics positively correlated with judgments from human police experts (Spearman $\rho$, $p < .05$)?
    - **Design Motivation**: Statistical fit alone may select domain-irrelevant metrics; expert judgment serves as an essential second-pass validation.
    - **Final retained metrics**: Logical Correctness, Completeness, Factuality, Logical Efficiency, and Logical Robustness.

3. **Hybrid Evaluation (LLM-as-Judge + Human Experts)**:

    - **Function**: Balances the scalability of automated evaluation with domain insight from human experts.
    - **Mechanism**: 225 responses (75 questions × 3 LLMs) are each evaluated via 3 automated reviews and 2 expert reviews.
    - **Design Motivation**: Drawing on prior findings regarding LLM reviewer reliability, structured prompts (reference answers + reasoning explanations) are used to improve consistency.

### Loss & Training

This paper is not a training-based work. The data collection and construction pipeline is as follows:
- **Data Source**: 1,602 official police manuals (PDFs) from the Korean National Police Agency, covering investigation, law enforcement, traffic control, emergency response, and more.
- **Preprocessing**: Segmented by heading and formatted into {title, content, question} structures.
- **Quality Filtering**: Filtered under expert guidance, yielding 8,348 high-quality entries.
- **Evaluation Set**: 75 question-answer pairs spanning 7 categories and 3 procedural types.
- **LLM Testing**: Zero-shot, temperature = 0.8, with only brief scenario descriptions provided.

## Key Experimental Results

### Main Results

**Gap between LLMs and reference answers on core metrics (5-point scale)**:

| Metric | Reference | GPT-4 | Gemini | Claude | Avg. Gap |
|--------|-----------|-------|--------|--------|----------|
| Logical Correctness | 4.14 | 2.61 | 3.04 | 2.80 | **−1.32** |
| Factuality | 4.15 | 2.60 | 2.98 | 2.72 | **−1.38** |
| Completeness | 3.81 | 2.76 | 3.30 | 2.78 | **−0.86** |
| Logical Efficiency | 3.84 | 2.63 | 2.87 | 2.77 | **−1.09** |
| Logical Robustness | 3.81 | 2.72 | 3.09 | 2.84 | **−0.93** |
| Overall Quality | 3.88 | 2.69 | **3.06** | 2.83 | **−1.02** |

### Ablation Study (Metric Selection Process)

| Metric | Regression $\beta$ | Regression $p$ | Expert $\rho$ | Correlation $p$ | Retained |
|--------|-------------------|----------------|---------------|-----------------|----------|
| Logical Correctness | 0.116 | .018 | 0.560 | <.001 | ✓ |
| Completeness | 0.141 | .001 | 0.409 | <.001 | ✓ |
| Factuality | 0.247 | <.001 | 0.330 | .002 | ✓ |
| Logical Efficiency | 0.137 | .004 | 0.330 | .002 | ✓ |
| Logical Robustness | 0.167 | .002 | 0.315 | .003 | ✓ |
| Logical Explanation | 0.108 | .003 | 0.203 | .055 | ✗ (Expert correlation not significant) |

### Key Findings

- **All commercial LLMs score significantly below reference answers on core metrics** ($p < .001$), with an average gap of −1.115 (core metrics) vs. −0.828 (all metrics), indicating that the framework accurately captures critical capability shortfalls.
- **Gemini performs best overall** (3.06); GPT-4 scores lowest (2.69), though differences across the three models are modest.
- **LLMs excel at text quality** (readability, harmlessness) but underperform on **domain knowledge** (factuality, logical correctness).
- **By policing domain**: All LLMs score lowest in 112 emergency calls (2.37), traffic (2.55), and security (2.61).
- **The regression model achieves a high fit** ($R^2 = 0.856$), confirming that the 5 core metrics effectively predict overall quality.

## Highlights & Insights

- **Strong methodological contribution**: Rather than simply evaluating one domain, the paper proposes a **scalable evaluation framework methodology** applicable to any setting requiring domain-specific LLM assessment.
- **Rigor of the two-stage metric selection**: The dual threshold of statistical significance and expert validation avoids subjectivity in metric selection.
- **The exclusion of "Logical Explanation"** is a compelling case study—it passed regression testing but failed expert correlation validation, demonstrating that neither stage alone is sufficient.
- **Practical dataset construction**: 8,000+ real official documents → 75 high-quality question-answer pairs, illustrating a rigorous pipeline for domain dataset development.

## Limitations & Future Work

- **Limited dataset scale**: 75 questions may not capture the full complexity of all policing scenarios.
- **Language specificity**: The framework is grounded in Korean policing law and documents; direct transfer to other countries requires reconstructing reference answers.
- **Zero-shot setting**: Few-shot or fine-tuned LLM performance is untested and may yield substantial improvements.
- **Inherent bias of LLM-as-Judge**: Moderate correlation coefficients ($\rho = 0.315$–$0.560$) indicate a non-trivial gap between automated evaluation and expert judgment.
- **RAG augmentation unexplored**: Using police manuals as a retrieval knowledge base could substantially improve LLM factuality.
- **Evaluation set limited to "pre-shift preparation" scenarios**: Other contexts (e.g., on-scene response, interrogation, forensics) require separate validation.

## Related Work & Insights

- The FLASK framework provided a starting point for candidate metrics, but the two-stage selection method presented here represents an important methodological advance.
- Applying LLM-as-Judge in specialized domains requires a triple guarantee: structured prompts, reference answers, and scoring rubrics.
- The proposed framework design is directly transferable to **healthcare** (emergency triage), **law** (case analysis), and **finance** (compliance review).
- The paper surfaces an important insight: strong LLM performance on standardized examinations (e.g., bar exams) does not necessarily translate to practical utility in real-world professional scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems](rectom_a_benchmark_for_evaluating_machine_theory_of_mind_in_llm-based_conversati.md)
- [\[ACL 2026\] Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework](../../ACL2026/recommender/personalizing_llms_with_binary_feedback_a_preference-corrected_optimization_fram.md)
- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)
- [\[AAAI 2026\] From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)
- [\[ACL 2026\] SenseJudge: Human-Centric Preference-Driven Judgment Framework](../../ACL2026/recommender/sensejudge_human-centric_preference-driven_judgment_framework.md)

</div>

<!-- RELATED:END -->
