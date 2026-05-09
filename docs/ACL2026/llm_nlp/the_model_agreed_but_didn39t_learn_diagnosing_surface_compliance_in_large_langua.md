---
title: >-
  [Paper Note] The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models
description: >-
  [ACL 2026][LLM/NLP][Knowledge Editing] This paper proposes the SA-MCQ diagnostic framework to reveal the phenomenon of "surface compliance" in knowledge editing — editors achieve high scores on standard benchmarks without genuinely overwriting internal beliefs, models revert to original parametric memory under discriminative self-assessment, and sequential editing accumulates representational residuals that lead to cognitive instability.
tags:
  - ACL 2026
  - LLM/NLP
  - Knowledge Editing
  - Surface Compliance
  - Self-Assessment
  - Parametric Memory
  - In-Context Learning
date: 2026-05-08
content_hash: a6e3cf7675f07720
---

# The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.05995](https://arxiv.org/abs/2604.05995)
**Code**: [XiaojieGu/SA-MCQ](https://github.com/XiaojieGu/SA-MCQ)
**Area**: LLM Trustworthiness / Knowledge Editing
**Keywords**: Knowledge Editing, Surface Compliance, Self-Assessment, Parametric Memory, In-Context Learning

## TL;DR

This paper proposes the SA-MCQ diagnostic framework to reveal the phenomenon of "surface compliance" in knowledge editing — editors achieve high scores on standard benchmarks without genuinely overwriting internal beliefs, models revert to original parametric memory under discriminative self-assessment, and sequential editing accumulates representational residuals that lead to cognitive instability.

## Background & Motivation

**Background**: LLMs encode world knowledge in their parameters as parametric memory, but inevitably inherit outdated and erroneous information from training corpora. Knowledge editing techniques aim to precisely modify specific internal memory states without retraining. Recent editors have demonstrated high success rates on standard benchmarks.

**Limitations of Prior Work**: Existing evaluation frameworks primarily rely on Exact Match to assess editing success, only checking whether the model can reproduce the target token under a specific prompt. Whether such surface-level textual consistency truly reflects internal memory reconfiguration remains unclear. Teacher forcing evaluation further inflates success rates by guiding outputs through correct token prefixes.

**Key Challenge**: High benchmark scores may represent mere "surface compliance" — editors achieve high scores by mimicking target outputs, while the model's internal beliefs are not structurally overwritten. When the evaluation paradigm shifts from generative to discriminative (forcing the model to choose among options), the modified memory may fail entirely.

**Goal**: Design a diagnostic framework capable of distinguishing "genuine memory modification" from "surface compliance," and reveal the true efficacy of knowledge editing.

**Key Insight**: Require edited models to answer multiple-choice questions (MCQs) rather than generate open-ended responses — MCQs force models to actively adjudicate among competing options, bypassing the rote-memorization bias inherent in generative evaluation.

**Core Idea**: Force models into discriminative self-assessment via Self-Assessment MCQ (SA-MCQ), and systematically detect the authenticity and robustness of edited memories under in-context learning settings.

## Method

### Overall Architecture

The SA-MCQ framework consists of: (1) converting knowledge editing triples into MCQ format, with options including the post-edit target answer, the original parametric answer, and an "uncertain" option; (2) instructing the model via system prompt to answer "based on its own memory"; (3) evaluating the stability of edited memories under varying contextual conditions (no context, supportive evidence, irrelevant noise, and counterfactual conflict).

### Key Designs

1. **Self-Assessment MCQ (SA-MCQ)**:

    - Function: Diagnose whether an edited model has genuinely overwritten its internal beliefs.
    - Mechanism: Present the edited model with a multiple-choice question whose options include the target answer, the original answer, and "uncertain." A model that has truly learned the new knowledge should select the target answer; a model exhibiting only surface compliance will revert to its original parametric memory under the discriminative setting. The system prompt explicitly instructs the model to answer "based on its own memory," eliminating contextual guidance effects.
    - Design Motivation: In open-ended generative evaluation, models can "guess" the correct answer from contextual cues. The MCQ format forces the model to actively compare and adjudicate among options, constituting a more rigorous test of memory.

2. **Multi-Condition Context Probing**:

    - Function: Evaluate the robustness of edited memories under varying contextual interference.
    - Mechanism: Four contextual conditions are designed — (a) no context: purely tests parametric memory; (b) supportive context: provides evidence consistent with the edit, testing whether memory can be reinforced; (c) irrelevant noise: provides unrelated information, testing memory stability; (d) counterfactual conflict: provides information contradicting the edit, testing resistance to interference.
    - Design Motivation: In real-world deployment, models always operate within contexts (e.g., ICL, RAG). Edited memories must remain consistent across various contextual conditions to be considered genuinely effective.

3. **Sequential Editing Analysis**:

    - Function: Evaluate the impact of successive editing rounds on the reversibility of model memory.
    - Mechanism: Perform multiple rounds of sequential editing and apply SA-MCQ after each round. The analysis detects whether edits accumulate representational residuals — i.e., whether residual parameter perturbations permanently impair the model's ability to revert to its original state, even after a given edit is undone.
    - Design Motivation: In practice, knowledge requires continuous updating. If each edit leaves irreversible traces, the model will progressively degrade.

### Loss & Training

SA-MCQ is a purely evaluative framework and does not involve training. The editors tested include AlphaEdit (locate-then-edit), RLEdit (meta-learning), and UltraEdit (large-scale precise editing), representing three mainstream editing paradigms. The CounterFact and zsRE datasets are used.

## Key Experimental Results

### Main Results

| Editor | Conventional Efficacy (TF) | SA-MCQ Efficacy | Gap |
|--------|------|------|----------|
| AlphaEdit | ~99% | Significant drop | Severe surface compliance |
| RLEdit | ~99% | Significant drop | Severe surface compliance |
| UltraEdit | ~99% | Significant drop | Severe surface compliance |
| Vanilla (unedited) | - | Original answer | Reference baseline |

### Ablation Study

| Context Condition | Phenomenon | Explanation |
|------|---------|------|
| No context | Reverts to original answer | Parametric memory not overwritten |
| Supportive evidence | Partial recovery of target answer | Relies on contextual cues rather than genuine memory |
| Counterfactual conflict | Falls into "cognitive deadlock" | External counterfactuals readily suppress editing effect |
| Sequential editing | Reversibility permanently reduced | Accumulated representational residuals cause cognitive instability |

### Key Findings

- **Surface compliance is pervasive**: All three mainstream editing paradigms exhibit this phenomenon — near-perfect editing success rates under conventional evaluation drop substantially under SA-MCQ.
- Edited memories are hypersensitive to context: supportive context can "rescue" editing effects, while counterfactual context can readily "suppress" them, indicating that editing creates a fragile context-dependent state rather than modifying parametric memory.
- Sequential editing is irreversible: successive edits accumulate representational residuals, and even undoing edits cannot restore the original memory state, leaving the model in permanent cognitive instability.
- Teacher forcing evaluation severely overestimates editing efficacy by producing spuriously high success rates through prefix guidance.

## Highlights & Insights

- **Introduction of the "surface compliance" concept**: This work precisely names a long-standing yet insufficiently recognized problem in the knowledge editing field — editors "answer questions correctly" without "genuinely learning the knowledge." This concept serves as an important warning to the broader knowledge editing community.
- **Paradigm shift in evaluation**: Moving from generative evaluation (can the model produce the correct answer?) to discriminative evaluation (can the model correctly judge among options?) is a simple yet profound insight — the latter is a more faithful test of genuine understanding.
- **The irreversibility of sequential editing** is an important negative finding that poses a fundamental challenge to the vision of "sustainable knowledge updating."

## Limitations & Future Work

- The "uncertain" option in SA-MCQ may introduce selection bias — models may prefer "uncertain" as a safe choice.
- Only three editors and two datasets are tested; broader coverage of editing methods and knowledge types remains to be validated.
- No solution to surface compliance is proposed; the work remains at the diagnostic level.
- Future directions include: designing training objectives that incorporate discriminative evaluation to improve editors, and investigating methods for clearing representational residuals.

## Related Work & Insights

- **vs. Conventional Evaluation (Exact Match + TF)**: Conventional methods evaluate generative consistency under specific prompts, while SA-MCQ tests discriminative belief — the gap between the two exposes the surface compliance problem.
- **vs. Memory-Augmented Methods (e.g., SERAC)**: Memory-augmented methods store edits externally rather than modifying parameters and fall outside the scope of this analysis, but may naturally avoid surface compliance.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The "surface compliance" concept is novel and important; the SA-MCQ evaluation paradigm shift merits broad adoption.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across three editors, four contextual conditions, and sequential editing analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and experimental findings are convincing.
- Value: ⭐⭐⭐⭐⭐ Provides an important warning to the knowledge editing field and promotes more rigorous evaluation standards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revisiting Non-Verbatim Memorization in Large Language Models: The Role of Entity Surface Forms](revisiting_non-verbatim_memorization_in_large_language_models_the_role_of_entity.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities](think_in_sentences_explicit_sentence_boundaries_enhance_language_model39s_capabi.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)
- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)

</div>

<!-- RELATED:END -->
