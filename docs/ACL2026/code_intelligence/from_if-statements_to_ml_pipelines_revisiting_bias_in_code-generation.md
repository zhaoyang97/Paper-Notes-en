---
title: >-
  [Paper Note] From If-Statements to ML Pipelines: Revisiting Bias in Code-Generation
description: >-
  [ACL 2026][Code Intelligence][Code generation bias] This paper reveals that bias evaluation in LLM code generation significantly underestimates actual risks: in ML pipeline generation…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Code generation bias"
  - "ML pipelines"
  - "feature selection"
  - "implicit discrimination"
  - "fairness evaluation"
date: 2026-05-08
content_hash: 025744cedc716f94
---

# From If-Statements to ML Pipelines: Revisiting Bias in Code-Generation

**Conference**: ACL 2026  
**arXiv**: [2604.21716](https://arxiv.org/abs/2604.21716)  
**Code**: [https://github.com/MinhDucBui/Code-Bias-ML-Pipelines](https://github.com/MinhDucBui/Code-Bias-ML-Pipelines)  
**Area**: Code Generation / AI Fairness  
**Keywords**: Code generation bias, ML pipelines, feature selection, implicit discrimination, fairness evaluation

## TL;DR

This paper reveals that bias evaluation in LLM code generation significantly underestimates actual risks: in ML pipeline generation, sensitive attributes appear in 87.7% of feature selections (vs. 59.2% in conditional statements). Furthermore, models correctly exclude irrelevant features while selectively retaining sensitive attributes like race and gender, demonstrating systemic implicit discrimination.

## Background & Motivation

**Background**: LLMs are increasingly applied in code generation, bringing research into bias to the forefront. However, existing evaluations (e.g., CodeGenBias, FairCoder) almost entirely measure bias through simple conditional statements (if-else logic), such as "if race == 'XX': deny_loan()".

**Limitations of Prior Work**: Simple conditional statements only capture explicit discrimination—code that directly maps protected attributes to outcomes. However, in the real world, discrimination more commonly occurs through implicit mechanisms, especially in feature selection decisions within ML pipelines. Including race or nationality as a predictive feature violates the fundamental principle of "fairness through unawareness" in algorithmic fairness.

**Key Challenge**: If implicit discrimination in LLM-generated ML pipelines is far more prevalent than explicit discrimination in conditional statements, then existing evaluation frameworks fundamentally underestimate the risk of bias.

**Goal**: (RQ1) Do LLMs exhibit systemic bias when generating ML pipelines? (RQ2) How does the degree of this bias compare to that in conditional statements?

**Key Insight**: The evaluation is extended from explicit conditional statements to more realistic ML pipeline feature selection tasks.

**Core Idea**: Bias in LLM code generation is far more severe than previously identified—implicit discrimination (ML pipeline feature selection) is 30 percentage points higher than explicit discrimination (conditional statements).

## Method

### Overall Architecture

The study evaluates the feature selection behavior of 10 LLMs across 7 fairness-sensitive datasets (e.g., credit scoring, recidivism prediction). Each dataset contains sensitive attributes (e.g., race, gender), non-sensitive attributes, and intentionally added irrelevant attributes (e.g., favorite color). Code Bias Score (CBS) is used to measure the proportion of sensitive attributes included.

### Key Designs

1.  **Dual-track evaluation of explicit vs. implicit discrimination**:
    - **Function**: Compares the severity of two forms of discrimination in LLM code generation.
    - **Mechanism**: For the same dataset, models are asked to (a) solve a prediction task using conditional statements (explicit route) and (b) implement a complete ML pipeline (randomly selecting one of MLP, Random Forest, SVM, Decision Tree, or Logistic Regression). The usage rates of sensitive attributes are then compared.
    - **Design Motivation**: A higher bias in ML pipelines suggests that while model safety mechanisms can identify and block explicit discrimination, they fail to detect implicit discrimination introduced through feature selection.

2.  **Irrelevant attributes as a control group**:
    - **Function**: Verifies that model attribute selection is selective rather than a blind retention of all attributes.
    - **Mechanism**: Three clearly irrelevant attributes (e.g., "favorite color") are added to each dataset to check if the model correctly excludes them. If the model excludes irrelevant attributes but retains sensitive ones, it indicates a problem of judgment rather than capability.
    - **Design Motivation**: This distinguishes "retaining all attributes" (laziness) from "selective retention of sensitive attributes" (bias); the latter is more concerning.

3.  **Multi-dimensional robustness verification**:
    - **Function**: Excludes experimental artifacts.
    - **Mechanism**: The study tests (a) prompt mitigation strategies (explicit instructions to avoid sensitive attributes), (b) variations in the number of attributes, and (c) different pipeline difficulty levels. Even at the simplest level (requiring only feature selection rather than a full pipeline), the sensitive attribute rate remains 16% higher than that of conditional statements.
    - **Design Motivation**: This proves that bias stems from the model's fundamentally different understanding of ML pipelines rather than task difficulty or prompt design.

### Loss & Training

This is an evaluation study and does not involve training. Greedy decoding is used, with 50 prompt variants per task (generated with GPT-5.1 assistance and human supervision).

## Key Experimental Results

### Main Results

Average bias across all models and datasets:

| Code Type | Average CBS | Statistically Significant Proportion |
| :--- | :--- | :--- |
| Conditional Statements | 58.7% | Majority |
| **ML Pipelines** | **88.3%** | **98%** |

Typical case (Llama-3.3-70B crime rate prediction): Irrelevant attributes like "favorite_color" were excluded, but "race" and "foreigners" were retained.

### Ablation Study

| Robustness Test | ML Pipeline Bias | Conditional Statement Bias | Gain |
| :--- | :--- | :--- | :--- |
| Standard Prompt | 88.3% | 58.7% | +29.6% |
| With Mitigation Prompt | Still Higher | Decreased | Persists |
| Feature Selection Only | 74% | 58% | +16% |
| Different Attribute Counts | Stable | Stable | Persists |

### Key Findings

- In 180 model-dataset-attribute combinations, 178 showed higher bias in ML pipelines, with 165 being statistically significant.
- Models used sensitive attributes as standard predictive features 100% of the time without any fairness processing.
- Code-specific models (DeepSeek Coder, Qwen Coder) exhibit bias as severe as general-purpose models.
- Even in the simplest "feature selection only" task, bias is 16% higher than in conditional statements—indicating that task complexity is not the primary factor.

## Highlights & Insights

- The discovery that "models can exclude 'favorite color' but retain 'race'" is highly impactful: it proves LLMs are not unaware of which attributes should be excluded, but rather they make different judgments in an ML context. This suggests models may have learned patterns prevalent in training data where race is treated as a useful predictive feature in ML.
- The comparison between explicit and implicit discrimination reveals a blind spot in safety alignment: RLHF and safety training mainly target explicitly harmful outputs but are nearly ineffective against implicit bias introduced through design decisions.
- This work has direct policy implications for AI deployment: While the EU AI Act encourages collecting sensitive data for debiasing and auditing, if LLMs automatically utilize this data as predictive features, it may inadvertently exacerbate discrimination.

## Limitations & Future Work

- The CBS metric measures the "risk" of discrimination rather than actual discriminatory outcomes—inclusion of a sensitive attribute does not necessarily lead to unfair output.
- Actual prediction bias of the generated models (e.g., predictive disparities between groups) was not analyzed.
- Only greedy decoding was used; results may vary under different sampling strategies.
- Mitigation strategies were only tested at the prompt level; the effectiveness of model-level interventions (e.g., specific safety fine-tuning) remains unknown.

## Related Work & Insights

- **vs. Liu et al. (2023)**: First identified bias in code generation but only used conditional statements; this paper proves that this approach significantly underestimates actual risks.
- **vs. FairCoder (Du et al., 2025)**: Expanded to more tasks but remained within the conditional statement paradigm; this paper fundamentally changes the evaluation paradigm.
- **vs. Algorithmic Fairness Literature (COMPAS, Dutch welfare)**: Real-world cases of system discrimination provided the motivation for this study, while this paper focuses on bias within the automation of code generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Fundamentally changes the evaluation paradigm for code generation bias.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 models × 7 datasets × multiple control conditions × multiple robustness tests.
- Writing Quality: ⭐⭐⭐⭐⭐ Sharp problem definition and highly impactful findings.
- Value: ⭐⭐⭐⭐⭐ Significant impact on the safety evaluation of LLM code generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReCode: Reinforcing Code Generation with Reasoning-Process Rewards](recode_reinforcing_code_generation_with_reasoning-process_rewards.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] Aligned Multi-View Scripts for Universal Chart-to-Code Generation](aligned_multi-view_scripts_for_universal_chart-to-code_generation.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)

</div>

<!-- RELATED:END -->
