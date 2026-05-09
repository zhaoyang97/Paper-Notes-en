---
title: >-
  [Paper Note] From If-Statements to ML Pipelines: Revisiting Bias in Code-Generation
description: >-
  [ACL 2026][code generation bias] This work demonstrates that existing bias evaluations of LLM code generation severely underestimate real-world risk: in ML pipeline generation, sensitive attributes appear in 87.7% of feature selection decisions (vs. 59.2% in conditional statements), and models correctly exclude irrelevant features yet consistently retain sensitive attributes such as race and gender, revealing systematic implicit discrimination.
tags:
  - ACL 2026
  - code generation bias
  - ML pipelines
  - feature selection
  - implicit discrimination
  - fairness evaluation
date: 2026-05-08
content_hash: 922b10caffe5c1c9
---

# From If-Statements to ML Pipelines: Revisiting Bias in Code-Generation

**Conference**: ACL 2026
**arXiv**: [2604.21716](https://arxiv.org/abs/2604.21716)
**Code**: [https://github.com/MinhDucBui/Code-Bias-ML-Pipelines](https://github.com/MinhDucBui/Code-Bias-ML-Pipelines)
**Area**: Code Generation / AI Fairness
**Keywords**: code generation bias, ML pipelines, feature selection, implicit discrimination, fairness evaluation

## TL;DR

This work demonstrates that existing bias evaluations of LLM code generation severely underestimate real-world risk: in ML pipeline generation, sensitive attributes appear in 87.7% of feature selection decisions (vs. 59.2% in conditional statements), and models correctly exclude irrelevant features yet consistently retain sensitive attributes such as race and gender, revealing systematic implicit discrimination.

## Background & Motivation

**State of the Field**: LLMs are increasingly applied to code generation, and bias in this context has attracted growing research attention. However, existing evaluations (e.g., CodeGenBias, FairCoder) almost exclusively measure bias through simple conditional statements (if-else logic), such as `if race == 'XX': deny_loan()`.

**Limitations of Prior Work**: Simple conditional statements can only capture explicit discrimination—code that directly maps protected attributes to outcomes. In practice, discrimination more commonly manifests through implicit mechanisms, particularly feature selection decisions in ML pipelines. Including race or nationality as predictive features violates the fundamental principle of "fairness through unawareness" in algorithmic fairness.

**Root Cause**: If implicit discrimination in LLM-generated ML pipelines is far more prevalent than explicit discrimination in conditional statements, then existing evaluation frameworks fundamentally underestimate bias risk.

**Paper Goals**: (RQ1) Do LLMs exhibit systematic bias when generating ML pipelines? (RQ2) How does the extent of such bias compare to that observed in conditional statements?

**Starting Point**: Extending evaluation from explicit conditional statements to the more realistic task of feature selection in ML pipelines.

**Core Idea**: Bias in LLM code generation is substantially more severe than previously recognized—implicit discrimination (ML pipeline feature selection) exceeds explicit discrimination (conditional statements) by approximately 30 percentage points.

## Method

### Overall Architecture

Ten LLMs are evaluated on seven fairness-sensitive datasets (credit scoring, recidivism prediction, etc.) with respect to their feature selection behavior when generating ML pipelines. Each dataset contains sensitive attributes (e.g., race, gender), non-sensitive attributes, and deliberately introduced irrelevant attributes (e.g., favorite color). The Code Bias Score (CBS) measures the proportion of cases in which sensitive attributes are included.

### Key Designs

1. **Dual-Track Evaluation of Explicit vs. Implicit Discrimination**:

    - **Function**: Compares the severity of two forms of discrimination in LLM-generated code.
    - **Mechanism**: For the same dataset, models are prompted to (a) solve a prediction task using conditional statements (explicit route) and (b) implement a complete ML pipeline (randomly selecting among MLP, Random Forest, SVM, Decision Tree, or Logistic Regression). Sensitive attribute inclusion rates are compared across both conditions.
    - **Design Motivation**: Higher bias in ML pipelines would indicate that safety mechanisms successfully detect and block explicit discrimination, yet fail to identify implicit discrimination introduced through feature selection.

2. **Irrelevant Attributes as a Control Group**:

    - **Function**: Verifies that attribute selection is deliberate rather than indiscriminate retention of all features.
    - **Mechanism**: Three clearly irrelevant attributes (e.g., "favorite color") are added to each dataset. The analysis examines whether models correctly exclude these attributes. If models exclude irrelevant attributes but retain sensitive ones, the issue reflects a judgment failure rather than a capability limitation.
    - **Design Motivation**: Distinguishes between "retaining all attributes" (laziness) and "selectively retaining sensitive attributes" (bias), with the latter being substantially more concerning.

3. **Multi-Dimensional Robustness Validation**:

    - **Function**: Rules out experimental artifacts.
    - **Mechanism**: Tests include (a) prompt-based mitigation strategies (explicit instructions to avoid sensitive attributes), (b) varying numbers of attributes, and (c) different pipeline complexity levels. Even at the lowest complexity level (feature selection only, without a full pipeline), sensitive attribute inclusion remains 16% higher than in conditional statements.
    - **Design Motivation**: Demonstrates that the bias stems from fundamentally different model behavior in ML pipeline contexts, rather than from task difficulty or prompt design.

### Loss & Training

This is an evaluation study; no training is involved. Greedy decoding is used, with 50 prompt variants per task (generated with GPT-4.1 assistance under human supervision).

## Key Experimental Results

### Main Results

Average bias across all models and datasets:

| Code Type | Avg. CBS | Statistically Significant Proportion |
|-----------|----------|---------------------------------------|
| Conditional Statements | 58.7% | Majority |
| **ML Pipelines** | **88.3%** | **98%** |

A representative case (Llama-3.3-70B on crime rate prediction): irrelevant attributes such as `favorite_color` are excluded, while `race` and `foreigners` are retained.

### Ablation Study

| Robustness Test | ML Pipeline Bias | Conditional Statement Bias | Gap |
|----------------|-----------------|---------------------------|-----|
| Standard prompt | 88.3% | 58.7% | +29.6% |
| With mitigation prompt | Still higher | Reduced | Persistent |
| Feature selection only | 74% | 58% | +16% |
| Varying attribute counts | Stable | Stable | Persistent |

### Key Findings

- Among 180 model–dataset–attribute combinations, 178 exhibit higher bias in ML pipelines; 165 reach statistical significance.
- Models include sensitive attributes as standard predictive features in 100% of cases, with no fairness-aware processing applied.
- Code-specialized models (DeepSeek Coder, Qwen Coder) exhibit bias levels comparable to general-purpose models.
- Even in the simplest "feature selection only" task, bias remains 16% higher than in conditional statements, indicating that task complexity is not the driving factor.

## Highlights & Insights

- The finding that "models exclude 'favorite color' but retain 'race'" is particularly striking: it demonstrates that LLMs are not unaware of which attributes should be avoided, but rather apply different judgment in ML contexts. This suggests that models may have internalized the pattern that "race is a useful predictive feature in ML"—a pattern prevalent in training data.
- The contrast between explicit and implicit discrimination reveals a blind spot in safety alignment: RLHF and safety training primarily target explicitly harmful outputs, yet are largely ineffective against implicit bias introduced through design decisions.
- This work carries direct policy implications for AI deployment: the EU AI Act encourages the collection of sensitive data for debiasing and auditing purposes, but if LLMs automatically incorporate such data as predictive features, the outcome may exacerbate rather than mitigate discrimination.

## Limitations & Future Work

- The CBS metric measures discrimination *risk* rather than actual discriminatory outcomes—inclusion of sensitive attributes does not necessarily lead to unfair predictions.
- The actual predictive bias of generated models (e.g., performance disparities across demographic groups) is not analyzed.
- Only greedy decoding is employed; results under alternative sampling strategies may differ.
- Mitigation strategies are evaluated only at the prompt level; the effectiveness of model-level interventions (e.g., targeted safety fine-tuning) remains unknown.

## Related Work & Insights

- **vs. Liu et al. (2023)**: First to identify bias in code generation but relying solely on conditional statements; the present work demonstrates that this approach severely underestimates actual risk.
- **vs. FairCoder (Du et al., 2025)**: Extends evaluation to more tasks but remains within the conditional statement paradigm; the present work fundamentally reframes the evaluation approach.
- **vs. Algorithmic Fairness Literature (COMPAS, Dutch Welfare)**: Real-world discrimination cases provide the motivation for this study, though the focus here is specifically on bias introduced through automated code generation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Fundamentally reframes the evaluation paradigm for bias in code generation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 models × 7 datasets × multiple control conditions × extensive robustness testing.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is precise and the findings are highly impactful.
- Value: ⭐⭐⭐⭐⭐ Significant implications for safety evaluation of LLM-based code generation.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)
- [\[ACL 2026\] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)
- [\[ACL 2026\] OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward](omnidiagram_advancing_unified_diagram_code_generation_via_visual_interrogation_r.md)

<!-- RELATED:END -->
