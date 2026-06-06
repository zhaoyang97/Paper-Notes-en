---
title: >-
  [Paper Note] The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models
description: >-
  [ACL 2026][Knowledge Editing][Surface Compliance] The proposed SA-MCQ diagnostic framework reveals the "surface compliance" phenomenon in knowledge editing—editors achieve high scores on standard benchmarks but fail to t…
tags:
  - "ACL 2026"
  - "Knowledge Editing"
  - "Surface Compliance"
  - "Self-Assessment"
  - "Parametric Memory"
  - "In-Context Learning"
date: 2026-05-08
content_hash: 5fa8f04076c10b42
---

# The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.05995](https://arxiv.org/abs/2604.05995)  
**Code**: [XiaojieGu/SA-MCQ](https://github.com/XiaojieGu/SA-MCQ)  
**Area**: LLM Trustworthiness / Knowledge Editing  
**Keywords**: Knowledge Editing, Surface Compliance, Self-Assessment, Parametric Memory, In-Context Learning

## TL;DR

The proposed SA-MCQ diagnostic framework reveals the "surface compliance" phenomenon in knowledge editing—editors achieve high scores on standard benchmarks but fail to truly overwrite internal beliefs. Models revert to original parametric memory in discriminative self-assessment, and sequential editing accumulates representation residuals, leading to cognitive instability.

## Background & Motivation

**Background**: LLMs encode world knowledge within their parameters as parametric memory, but inevitably inherit obsolete or incorrect information from training corpora. Knowledge editing techniques aim to precisely modify specific internal memory states without retraining. Recently, editors have demonstrated high success rates on standard benchmarks.

**Limitations of Prior Work**: Existing evaluation frameworks mainly rely on Exact Match to assess editing success, checking only whether the model can reproduce target tokens under specific prompts. However, it remains unclear if this surface-level textual consistency truly reflects structural reconfiguration of internal memory. Teacher-forced evaluation further inflates success rates by guiding the output with correct token prefixes.

**Key Challenge**: High benchmark scores may merely represent "surface compliance"—where editors achieve high scores by mimicking target outputs without structurally overwriting the model's internal beliefs. When the evaluation shifts from generative to discriminative (forcing the model to choose between options), the modified memory may fail completely.

**Goal**: Design a diagnostic framework capable of distinguishing "true memory modification" from "surface compliance" to reveal the actual efficacy of knowledge editing.

**Key Insight**: Tasks should require the edited model to perform Multiple-Choice Questions (MCQ) instead of open generation. MCQs force the model to actively adjudicate between competing options, circumventing the rote memorization bias inherent in generative evaluations.

**Core Idea**: The Self-Assessment Multiple-Choice Question (SA-MCQ) framework forces models to perform discriminative self-assessment, systematically detecting the authenticity and robustness of edited memory under in-context learning settings.

## Method

### Overall Architecture

The SA-MCQ framework consists of: (1) converting knowledge editing triplets into a multiple-choice format, where options include the edited target answer, the original parametric answer, and an "uncertain" option; (2) requiring the model to answer "based on its own memory" via system prompts; (3) evaluating the stability of edited memory across different context conditions (no context, supportive evidence, irrelevant noise, and counterfactual conflict).

### Key Designs

1.  **Self-Assessment Multiple-Choice Questions (SA-MCQ)**:
    *   **Function**: Diagnoses whether the edited model has truly overwritten its internal beliefs.
    *   **Mechanism**: The edited model is presented with a multiple-choice question containing the target answer, the original answer, and an "uncertain" option. If the model has truly learned the new knowledge, it should choose the target answer; if it only exhibits surface compliance, it will revert to the original parametric memory in this discriminative setting. System prompts explicitly command the model to answer "based on its own memory" to exclude context-guiding effects.
    *   **Design Motivation**: In open generative evaluation, models can "guess" the correct answer from context clues. The MCQ format forces active comparison and adjudication, providing a more rigorous test of memory.

2.  **Multi-Condition Context Probing**:
    *   **Function**: Evaluates the robustness of edited memory under various contextual interferences.
    *   **Mechanism**: Four context conditions are designed—(a) No context: purely testing parametric memory; (b) Supportive context: providing evidence consistent with the edit to see if memory is reinforced; (c) Irrelevant noise: providing unrelated information to test stability; (d) Counterfactual conflict: providing information contradicting the edit to test interference resistance.
    *   **Design Motivation**: In real-world deployment, models always operate within a context (e.g., ICL, RAG). Edited memory must remain consistent across various contexts to be considered truly effective.

3.  **Sequential Editing Analysis**:
    *   **Function**: Evaluates the impact of continuous multi-round editing on memory reversibility.
    *   **Mechanism**: Multiple rounds of sequential editing are performed, with SA-MCQ evaluation after each round. This detects whether editing accumulates representation residuals—even if an edit is revoked, whether residual parametric perturbations permanently impair the model's ability to return to its original state.
    *   **Design Motivation**: Practical applications require continuous knowledge updates. If every edit leaves irreversible traces, the model will gradually degrade.

### Loss & Training

SA-MCQ is a pure evaluation framework and does not involve training. The tested editors include AlphaEdit (locate-then-edit), RLEdit (meta-learning), and UltraEdit (large-scale precise editing), representing three major editing paradigms. Experiments use the CounterFact and zsRE datasets.

## Key Experimental Results

### Main Results

| Editor | Traditional Efficacy (TF) | SA-MCQ Efficacy | Gap |
| :--- | :--- | :--- | :--- |
| AlphaEdit | ~99% | Significant decrease | Severe surface compliance |
| RLEdit | ~99% | Significant decrease | Severe surface compliance |
| UltraEdit | ~99% | Significant decrease | Severe surface compliance |
| Vanilla (Unedited) | - | Original Answer | Reference Baseline |

### Ablation Study

| Context Condition | Phenomenon | Description |
| :--- | :--- | :--- |
| No Context | Revert to original answer | Parametric memory not overwritten |
| Supportive Evidence | Partial target recovery | Relies on context prompts rather than true memory |
| Counterfactual Conflict | "Cognitive Deadlock" | External counterfactuals easily suppress editing effects |
| Sequential Editing | Permanent drop in reversibility | Accumulation of residuals causes cognitive instability |

### Key Findings

*   **Surface compliance is a universal phenomenon**: It exists across all three major editing paradigms. Near-perfect success rates under traditional evaluation drop significantly under SA-MCQ.
*   **Edited memory is hypersensitive to context**: Supportive contexts can "rescue" editing effects, while counterfactual contexts can easily "suppress" them. This indicates that editors may create a fragile context-dependency rather than modifying parametric memory.
*   **Sequential editing is irreversible**: Continuous editing accumulates representation residuals. Even if an edit is undone, the original memory state cannot be fully recovered, leading to permanent cognitive instability.
*   **Teacher-forced evaluation severely overestimates efficacy**: It produces falsely high success rates through prefix guidance.

## Highlights & Insights

*   **Introduction of "Surface Compliance"**: This concept accurately names a long-standing but under-recognized issue in knowledge editing—where editors "get the answer right" without "learning the knowledge." This serves as a warning to the knowledge editing community.
*   **Paradigm Shift in Evaluation**: Shifting from generative evaluation (can it say the right answer) to discriminative evaluation (can it judge correctly between options) is a simple but profound insight—the latter is a better test of "true understanding."
*   **Irreversibility of Sequential Editing**: This is a significant negative finding that poses a fundamental challenge to the vision of "sustainable knowledge updates."

## Limitations & Future Work

*   The "uncertain" option in SA-MCQ might introduce selection bias, as models may favor it as a safe choice.
*   Only three editors and two datasets were tested; validation across a broader range of methods and knowledge types is needed.
*   The study is limited to diagnosis and does not propose a solution for surface compliance.
*   Future work could explore: designing training objectives that incorporate discriminative evaluation to improve editors, and researching methods to clear representation residuals.

## Related Work & Insights

*   **vs. Traditional Evaluation (Exact Match + TF)**: Traditional methods assess generation consistency under specific prompts, while SA-MCQ tests discriminative beliefs. The discrepancy between the two exposes the surface compliance problem.
*   **vs. Memory Augmentation (SERAC, etc.)**: Memory augmentation methods store edits externally rather than modifying parameters. These are outside the scope of this analysis but may naturally avoid surface compliance issues.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ The "surface compliance" concept is novel and significant; the SA-MCQ evaluation paradigm shift is worth promoting.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive analysis across three editors, four context conditions, and sequential editing.
*   Writing Quality: ⭐⭐⭐⭐ Clear problem definition and persuasive experimental findings.
*   Value: ⭐⭐⭐⭐⭐ Provides a crucial warning for the knowledge editing field and pushes for more rigorous evaluation standards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Aligning Language Models with Real-time Knowledge Editing](aligning_language_models_with_real-time_knowledge_editing.md)
- [\[ICML 2026\] Reverse-Engineering Model Editing on Language Models](../../ICML2026/knowledge_editing/reverse-engineering_model_editing_on_language_models.md)
- [\[ICML 2026\] The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models](../../ICML2026/knowledge_editing/the_labyrinth_and_the_thread_rethinking_regularizations_in_sequential_knowledge_.md)
- [\[NeurIPS 2025\] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](../../NeurIPS2025/knowledge_editing/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [\[ICML 2026\] Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](../../ICML2026/knowledge_editing/revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)

</div>

<!-- RELATED:END -->
