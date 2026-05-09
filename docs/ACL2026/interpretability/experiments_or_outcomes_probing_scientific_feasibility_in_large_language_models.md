---
title: >-
  [Paper Note] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models
description: >-
  [ACL 2026][scientific feasibility assessment] This work constructs a controlled knowledge framework to systematically study how LLMs leverage experimental descriptions and outcome evidence in scientific feasibility assessment. Results show that providing outcome evidence is more reliable than experimental descriptions, that partial experimental information frequently degrades performance below a parametric-knowledge-only baseline, and that LLM reasoning exhibits notable fragility under incomplete evidence.
tags:
  - ACL 2026
  - scientific feasibility assessment
  - controlled knowledge framework
  - evidence robustness
  - experiments vs. outcomes
  - LLM reasoning
date: 2026-05-08
content_hash: 6d6b7912bc37ab18
---

# Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.18786](https://arxiv.org/abs/2604.18786)
**Code**: [https://github.com/mohammadi-ali/scify](https://github.com/mohammadi-ali/scify)
**Area**: Interpretability
**Keywords**: scientific feasibility assessment, controlled knowledge framework, evidence robustness, experiments vs. outcomes, LLM reasoning

## TL;DR

This work constructs a controlled knowledge framework to systematically study how LLMs leverage experimental descriptions and outcome evidence in scientific feasibility assessment. Results show that providing outcome evidence is more reliable than experimental descriptions, that partial experimental information frequently degrades performance below a parametric-knowledge-only baseline, and that LLM reasoning exhibits notable fragility under incomplete evidence.

## Background & Motivation

**Background**: LLMs are increasingly employed in scientific workflows (literature review, hypothesis generation, experiment planning), yet their capacity to perform a fundamental scientific task—scientific feasibility assessment—remains poorly understood. Feasibility assessment requires judging whether a claim is consistent with established knowledge and whether experimental evidence supports or refutes it.

**Limitations of Prior Work**: Existing work either focuses on hypothesis generation rather than evaluation, conflates parametric knowledge with retrieved information without isolating their respective contributions, or examines compliance with external knowledge in non-scientific settings. Three critical questions remain unanswered: (RQ1) Can LLMs assess feasibility using parametric knowledge alone? (RQ2) How does providing experimental/outcome context alter judgments? (RQ3) How robust are these judgments under incomplete information?

**Key Challenge**: Intuitively, more evidence should improve judgment—yet partial or noisy evidence may in fact mislead. The question is whether LLMs can handle incomplete information gracefully.

**Goal**: To understand how evidence type affects LLM feasibility judgments by systematically controlling the visibility of experimental descriptions and outcomes.

**Key Insight**: Design four knowledge conditions (hypothesis only / +experiments / +outcomes / +both) and a stability analysis involving progressive removal of partial evidence.

**Core Idea**: Outcome evidence is generally more reliable than experimental descriptions; partial evidence frequently causes brittle collapse rather than graceful degradation.

## Method

### Overall Architecture

Given a scientific hypothesis $h$, LLM feasibility judgments are evaluated under four controlled knowledge conditions: H (hypothesis only), H+E (+experimental descriptions), H+O (+outcome summaries), and H+E+O (+both). The visibility proportions of experiments and outcomes are controlled via parameters $k_1, k_2 \in \{0, 0.5, 1.0\}$; each configuration is averaged over five random samples.

### Key Designs

1. **Controlled Knowledge Framework**:

    - Function: Isolate the effect of different evidence types on LLM feasibility judgments.
    - Mechanism: The prediction task is held strictly constant (output: feasible/infeasible + rationale); only the context accompanying the hypothesis varies: $x \in \{\emptyset, \mathcal{E}^*, \mathcal{O}^*, (\mathcal{E}^*, \mathcal{O}^*)\}$. Experimental descriptions and outcomes are extracted directly from source papers rather than retrieved, ensuring evidence quality. Any difference in predictions across conditions therefore reflects the influence of evidence alone, not task variation.
    - Design Motivation: Prior work conflated multiple information sources, making it impossible to distinguish which type of evidence is genuinely useful.

2. **Stability Analysis**:

    - Function: Test the degradation pattern of LLM judgments under incomplete evidence.
    - Mechanism: The proportions of experiments and/or outcomes are progressively removed ($k_1, k_2$ reduced from 1.0 to 0.5 to 0), and performance is observed to determine whether degradation is monotonic (graceful) or non-monotonic (brittle). A "below-baseline rate" is defined as the proportion of partial-evidence conditions in which performance falls below the zero-evidence (H) baseline.
    - Design Motivation: Real-world scientific reasoning frequently relies on incomplete evidence. If partial evidence misleads the model, this suggests the model is performing superficial alignment rather than deep reasoning.

3. **Multi-Dimensional Evaluation**:

    - Function: Comprehensively assess both the accuracy of feasibility judgments and the quality of explanations.
    - Mechanism: Accuracy, macro-F1, and MCC (more informative under class imbalance) are reported, alongside ROUGE lexical overlap between generated and reference explanations (used as a diagnostic signal only). Five frontier LLMs (GPT-5.1, GPT-4o, Gemini-2.5-Pro/Flash, Grok-4.1-fast) are evaluated on two datasets.
    - Design Motivation: MCC is more reliable than accuracy under imbalanced classification; multi-model evaluation ensures cross-platform generalizability of findings.

### Loss & Training

This is a purely evaluative study using zero-shot prompting. All models are given identical task instructions.

## Key Experimental Results

### Main Results

GPT-5.1 performance on the MoF dataset:

| Condition | Accuracy | F1_macro | MCC |
|-----------|----------|----------|-----|
| H (hypothesis only) | 0.68 | 0.67 | 0.42 |
| H+E (100% experiments) | 0.70 | 0.69 | 0.44 |
| H+O (100% outcomes) | 0.66 | 0.66 | 0.33 |
| H+E+O (full) | 0.66 | 0.66 | 0.33 |

### Ablation Study

GPT-5.1 on the Reasons dataset:

| Condition | Accuracy | Note |
|-----------|----------|------|
| H | 0.84 | Parametric knowledge baseline |
| H+E (50%) | 0.85 | Marginal improvement |
| H+O (100%) | 0.92 | Strong outcome evidence |
| H+E+O (100%) | 0.93 | Best overall |
| H+E (50%) + H+O (50%) | 0.90 | Partial evidence still useful |

### Key Findings

- Outcome evidence consistently improves feasibility judgments more reliably than experimental descriptions—on the Reasons dataset, H+O uniformly outperforms H+E.
- Experimental descriptions can be "brittle": partial experimental information ($k_1=0.5$) causes performance to fall below the hypothesis-only baseline across multiple models, suggesting models engage in surface-feature matching rather than genuine understanding of experimental design.
- Degradation is frequently non-monotonic—performance at $k_1=0.5$ can be worse than at $k_1=0$—indicating that models do not reason in a "use whatever information is available" manner.
- Gemini-2.5-Pro exhibits the greatest instability under experimental description conditions (dropping from 0.67 to 0.48), exposing severe surface alignment issues.
- Even for the strongest model, GPT-5.1, providing full experiments+outcomes does not consistently outperform providing outcomes alone (MCC is equal or lower on the MoF dataset).

## Highlights & Insights

- The finding that "partial evidence can be actively harmful" is a profound and sobering result: it reveals a fundamental fragility in LLM scientific reasoning—models behave more like pattern matchers than genuine reasoners over the logical structure of experiments. This carries important warnings for using LLMs in scientific peer review and decision-making.
- The controlled knowledge framework is methodologically elegant: by holding the task constant and varying only the context, it enables clean causal inference. This design can be transferred to other research evaluating how LLMs utilize different types of knowledge.
- The "outcomes > experiments" finding suggests that LLMs are better at processing declarative knowledge ("what happened") than procedural knowledge ("how it was done")—a pattern consistent with the nature of LLM training data.

## Limitations & Future Work

- Only zero-shot evaluation is conducted; fine-tuned or few-shot settings may yield different results.
- Feasibility judgment is reduced to binary classification, whereas real-world scientific feasibility typically exists on a spectrum.
- The quality of experiment and outcome extraction may influence conclusions—incomplete extraction itself could introduce the observed brittleness.
- Explanation quality is assessed solely via ROUGE lexical overlap, which cannot truly measure the logical correctness of scientific reasoning.
- Only commercial API models are evaluated; open-source models may behave differently.

## Related Work & Insights

- **vs. Qi et al. (2023) / Yang et al. (2024)**: Focus on hypothesis generation rather than evaluation; this paper fills the gap in feasibility judgment.
- **vs. Jansen et al. (2025)**: Mixes parametric knowledge and retrieval without isolating their contributions; the controlled framework here achieves a clean separation.
- **vs. Mohammadi et al. (2025)**: Studies LLM compliance with external knowledge but in non-scientific settings; this paper focuses on evidence utilization in scientific reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ The experimental design combining a controlled knowledge framework with stability analysis is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models × 2 datasets × 9 evidence conditions × 5 random seeds.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formalization is clear; experimental design is rigorous.
- Value: ⭐⭐⭐⭐ Meaningfully advances understanding of LLM scientific reasoning capabilities.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](../../NeurIPS2025/interpretability/table_as_a_modality_for_large_language_models.md)
- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](../../NeurIPS2025/interpretability/the_trilemma_of_truth_in_large_language_models.md)
- [\[ACL 2026\] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)

<!-- RELATED:END -->
