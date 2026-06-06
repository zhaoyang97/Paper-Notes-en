---
title: >-
  [Paper Note] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models
description: >-
  [ACL 2026][Interpretability][Scientific feasibility assessment] A controlled knowledge framework was constructed to systematically investigate how LLMs utilize experimental descriptions and outcome evidence in scientific…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Scientific feasibility assessment"
  - "controlled knowledge framework"
  - "evidentiary robustness"
  - "experiments vs. outcomes"
  - "LLM reasoning"
date: 2026-05-08
content_hash: ff83687894bb2500
---

# Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.18786](https://arxiv.org/abs/2604.18786)  
**Code**: [https://github.com/mohammadi-ali/scify](https://github.com/mohammadi-ali/scify)  
**Area**: Interpretability  
**Keywords**: Scientific feasibility assessment, controlled knowledge framework, evidentiary robustness, experiments vs. outcomes, LLM reasoning

## TL;DR

A controlled knowledge framework was constructed to systematically investigate how LLMs utilize experimental descriptions and outcome evidence in scientific feasibility assessment. It was discovered that providing outcome evidence is more reliable than experimental descriptions, and partial experimental information often leads to performance lower than the baseline using only parametric knowledge, revealing the fragility of LLM reasoning.

## Background & Motivation

**Background**: LLMs are increasingly utilized in scientific workflows (literature review, hypothesis generation, experimental planning), but their ability to execute basic scientific tasks—such as scientific feasibility assessment—remains unclear. Feasibility assessment requires determining whether a claim aligns with existing knowledge and whether experimental evidence supports or refutes it.

**Limitations of Prior Work**: Existing research either focuses on hypothesis generation rather than evaluation, mixes internal model knowledge with retrieval without isolating individual contributions, or tests adherence to external knowledge in non-scientific scenarios. Three key questions remain unanswered: (RQ1) Can LLMs assess feasibility using only parametric knowledge? (RQ2) How does providing experimental/outcome context change judgments? (RQ3) How robust are these judgments when information is incomplete?

**Key Challenge**: Intuitively, more evidence should aid judgment, but partial or noisy evidence might be misleading—can LLMs handle incomplete information gracefully?

**Goal**: To understand the impact of evidence types on LLM feasibility judgments through systematic control of the visibility of experiments and outcomes.

**Key Insight**: Design 4 knowledge conditions (Hypothesis only / +Experiment / +Outcome / +Both) and stability analysis (progressive removal of partial evidence).

**Core Idea**: Outcome evidence is generally more reliable than experimental descriptions, and partial evidence often leads to fragile collapse rather than graceful degradation.

## Method

### Overall Architecture

Given a scientific hypothesis $h$, the feasibility judgments of LLMs are evaluated under 4 controlled knowledge conditions: H (Hypothesis only), H+E (+Experimental description), H+O (+Outcome summary), and H+E+O (+Both). The visibility ratios of experiments and outcomes are controlled via parameters $k_1, k_2 \in \{0, 0.5, 1.0\}$, with each configuration averaged over 5 random samplings.

### Key Designs

1. **Controlled Knowledge Framework**:

    - **Function**: Isolates the impact of different evidence types on LLM feasibility judgments.
    - **Mechanism**: The prediction task remains identical (output feasible/infeasible + reasoning), while only the context accompanying the hypothesis is varied: $x \in \{\emptyset, \mathcal{E}^*, \mathcal{O}^*, (\mathcal{E}^*, \mathcal{O}^*)\}$. Experimental descriptions and outcomes are extracted from source papers rather than retrieved, ensuring evidence quality. Differences in predictions across conditions reflect the impact of evidence rather than task variation.
    - **Design Motivation**: Previous work mixed multiple information sources, making it impossible to distinguish which type of evidence was truly beneficial.

2. **Stability Analysis**:

    - **Function**: Tests the degradation patterns of LLM judgments when evidence is incomplete.
    - **Mechanism**: Gradually reduces the proportion of experiments and/or outcomes ($k_1, k_2$ from 1.0 to 0.5 to 0) to observe whether performance exhibits monotonic degradation (graceful) or non-monotonic collapse (fragile). The "below-baseline rate" is defined as the proportion of cases where performance under partial evidence is lower than the zero-evidence (H) baseline.
    - **Design Motivation**: Real-world scientific reasoning is often based on incomplete evidence; if partial evidence misleads the model, it indicates the model is performing surface alignment rather than deep reasoning.

3. **Multi-dimensional Evaluation**:

    - **Function**: Comprehensively evaluates the accuracy and explanation quality of feasibility judgments.
    - **Mechanism**: Evaluates Accuracy, macro-F1, MCC (more informative under class imbalance), and ROUGE lexical overlap between generated and reference explanations (as a diagnostic signal). The study covers 5 frontier LLMs (GPT-5.1, GPT-4o, Gemini-2.5-Pro/Flash, Grok-4.1-fast) across two datasets.
    - **Design Motivation**: MCC is more reliable than accuracy in unbalanced classification, and multi-model evaluation ensures cross-platform consistency of findings.

### Loss & Training

A pure evaluation study utilizing zero-shot prompting. All models use the same task instructions.

## Key Experimental Results

### Main Results

Performance of GPT-5.1 on the MoF dataset:

| Condition | Accuracy | F1_macro | MCC |
|------|----------|---------|-----|
| H (Hypothesis only) | 0.68 | 0.67 | 0.42 |
| H+E (100% Exp) | 0.70 | 0.69 | 0.44 |
| H+O (100% Out) | 0.66 | 0.66 | 0.33 |
| H+E+O (All) | 0.66 | 0.66 | 0.33 |

### Ablation Study

On the Reasons dataset (GPT-5.1):

| Condition | Accuracy | Description |
|------|----------|------|
| H | 0.84 | Parametric knowledge baseline |
| H+E (50%) | 0.85 | Slight improvement |
| H+O (100%) | 0.92 | Strong outcome evidence |
| H+E+O (100%) | 0.93 | Optimal |
| H+E (50%) + H+O (50%) | 0.90 | Partial evidence useful |

### Key Findings

- Outcome evidence (outcomes) generally improves feasibility judgments more than experimental descriptions (experiments)—on the Reasons dataset, H+O consistently outperforms H+E.
- Experimental descriptions can be "fragile": partial experimental information ($k_1=0.5$) causes performance to drop below the hypothesis-only baseline across multiple models, suggesting models perform surface feature matching rather than true understanding of experimental design.
- Degradation is often non-monotonic—performance at $k_1=0.5$ can be worse than at $k_1=0$—indicating that models are not reasoning by "using whatever information is available."
- Gemini-2.5-Pro exhibited most instability under experimental description conditions (dropping from 0.67 to 0.48), exposing significant surface alignment issues.
- Even for the strongest model, GPT-5.1, providing full experiments + outcomes is not necessarily better than providing outcomes alone (MCC is the same or lower on the MoF dataset).

## Highlights & Insights

- The finding that "partial evidence is harmful" is a profound and cautionary insight: it reveals a fundamental fragility in LLM scientific reasoning—models behave more like pattern matchers than entities that understand the logical structure of experiments. This serves as an important warning for using LLMs in scientific peer review and decision-making.
- The experimental design of the Controlled Knowledge Framework is elegant: by keeping the task constant and varying only the context, it achieves clean causal inference. This methodology can be transferred to other studies evaluating LLM knowledge utilization.
- The "Outcomes > Experiments" finding implies that LLMs are better at processing declarative knowledge ("what happened") than procedural knowledge ("how it was done")—consistent with the nature of LLM training data.

## Limitations & Future Work

- Only zero-shot evaluation was used; fine-tuning or few-shot settings might yield different results.
- Feasibility judgment was simplified into binary classification, whereas real scientific feasibility is often a spectrum.
- The quality of experiment and outcome extraction might influence conclusions—incomplete extraction itself could lead to "fragility."
- Explanation quality was only evaluated via ROUGE lexical overlap, which cannot truly measure the logical correctness of scientific reasoning.
- Only commercial API models were tested; open-source models may perform differently.

## Related Work & Insights

- **vs. Qi et al. (2023) / Yang et al. (2024)**: Focused on hypothesis generation rather than evaluation; this work fills the gap in feasibility judgment.
- **vs. Jansen et al. (2025)**: Mixed internal knowledge and retrieval without isolating contributions; the controlled framework in this work achieves clean separation.
- **vs. Mohammadi et al. (2025)**: Investigated LLM adherence to external knowledge in non-scientific scenarios; this work focuses on evidence utilization in scientific reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative experimental design with controlled knowledge framework and stability analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models × 2 datasets × 9 evidence conditions × 5 random seeds.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem formalization and rigorous experimental design.
- Value: ⭐⭐⭐⭐ Significantly advances the understanding of LLM scientific reasoning capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MINED: Probing and Updating with Multimodal Time-Sensitive Knowledge for Large Multimodal Models](mined_probing_and_updating_with_multimodal_time-sensitive_knowledge_for_large_mu.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)

</div>

<!-- RELATED:END -->
