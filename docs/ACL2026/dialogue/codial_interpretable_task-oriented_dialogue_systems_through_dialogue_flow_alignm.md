---
title: >-
  [Paper Note] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment
description: >-
  [ACL 2026][Dialogue Systems][Task-oriented dialogue] This paper proposes CoDial, a framework that converts predefined dialogue flows (task schemas) into structured heterogeneous graphs and then automatically generates LL…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Task-oriented dialogue"
  - "LLM guardrails"
  - "Interpretability"
  - "Dialogue flow alignment"
  - "Zero-shot generalization"
date: 2026-05-08
content_hash: ca095e7436f4c335
---

# CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment

**Conference**: ACL 2026  
**arXiv**: [2506.02264](https://arxiv.org/abs/2506.02264)  
**Code**: [https://github.com/radinshayanfar/CoDial](https://github.com/radinshayanfar/CoDial)  
**Area**: Image Generation  
**Keywords**: Task-oriented dialogue, LLM guardrails, Interpretability, Dialogue flow alignment, Zero-shot generalization

## TL;DR

This paper proposes CoDial, a framework that converts predefined dialogue flows (task schemas) into structured heterogeneous graphs and then automatically generates LLM guardrail code (e.g., Colang). This achieves interpretable and controllable task-oriented dialogue policies during inference, reaching SOTA on the STAR benchmark without requiring training data.

## Background & Motivation

**Background**: Task-oriented dialogue (TOD) systems need to generalize across different tasks. Data-driven methods struggle to migrate to unseen tasks; schema-based methods enhance generalization by decoupling language understanding from task logic but rely on neural or generative models for schema parsing, lacking interpretability.

**Limitations of Prior Work**: (1) Neural-based schema methods are opaque, preventing users from understanding how the schema affects dialogue behavior; (2) Methods like AnyTOD achieve interpretability through programmatic implementation but require users to possess programming skills to manually write policy programs, increasing technical barriers; (3) Interpretability is crucial in high-risk fields such as law and medicine.

**Key Challenge**: Existing TOD systems struggle to achieve both generalization and interpretability—neural methods have generalization but lack interpretability, while programmatic methods are interpretable but require programming skills.

**Goal**: Design a TOD framework that requires no training data or manual programming, automatically converting dialogue flows into executable LLM guardrail programs to provide interpretable and controllable dialogue behavior during inference.

**Key Insight**: Reposition LLM guardrails as the foundation for defining TOD system behavior, leveraging LLM code generation capabilities to automatically convert dialogue flows into guardrail code.

**Core Idea**: Dialogue flow → Heterogeneous graph (CHIEF) → Guardrail code (Colang) → Executable TOD system. The entire process is automated and inherently interpretable.

## Method

### Overall Architecture

CoDial consists of three components: (1) CHIEF (Heterogeneous Dialogue Flow Representation)—defines task schemas as heterogeneous directed graphs, supporting various node types (Request/External Action/Inform/Confirm/Global/Fallback); (2) GCG (Guardrail Code Generation)—automatically converts the JSON representation of CHIEF into Colang guardrail code using an LLM; (3) CHF (Human Feedback Mechanism)—iteratively optimizes the generated guardrail code through human or LLM feedback.

### Key Designs

1. **CHIEF Heterogeneous Dialogue Flow Representation**:

    - **Function**: Defines rich task schemas in a structured manner.
    - **Mechanism**: Designs different node types (Request defines slots to track, External Action calls external functions, Inform/Confirm provides information and confirmation, Global/Fallback handles global and fallback actions), connecting nodes with conditional edges. The entire structure is encoded in JSON format.
    - **Design Motivation**: Prior work used homogeneous graphs (where all node types are the same), which cannot express complex task logic; heterogeneous graphs support different node types and metadata.

2. **Two Paradigms for Guardrail Code Generation (CoDialfree/CoDialstructured)**:

    - **Function**: Automatically converts dialogue flows into executable guardrail programs.
    - **Mechanism**: CoDialfree provides Colang syntax documentation to allow the LLM to freely design guardrail logic; CoDialstructured explicitly guides the LLM on how to model dialogue state, implement DST (Dialogue State Tracking), and NAP (Next Action Prediction), generating structured guardrail code.
    - **Design Motivation**: CoDialfree serves as an interpretable baseline to verify the feasibility of automatic code generation; CoDialstructured improves code quality and reliability through explicit structural constraints.

3. **CoDial Human Feedback Mechanism (CHF)**:

    - **Function**: Iteratively optimizes generated guardrail code.
    - **Mechanism**: Supports three feedback modes: (a) human experts direct code modification; (b) humans providing natural language suggestions executed by the LLM; (c) LLM automatically analyzing dialogue failures and generating suggestions (LLM-aided feedback).
    - **Design Motivation**: Automatically generated code may contain errors or omissions; the iterative feedback mechanism allows for continuous improvement.

### Loss & Training

CoDial is a zero-shot, training-free framework. All dialogue policies are executed via guardrail code during inference, requiring no gradient updates. Core computation comes from LLM code generation and dialogue inference.

## Key Experimental Results

### Main Results

**Performance on STAR benchmark (Task Success Rate %)**

| Method | Training Required | Interpretable | Success Rate |
|------|---------|---------|--------|
| SGD-LLM | Required | No | Low |
| AnyTOD | Required + Manual Programming | Yes | Medium |
| CoDialfree | Zero-shot | Yes | Competitive |
| CoDialstructured | Zero-shot | Yes | **SOTA** |
| CoDialstructured + CHF | Zero-shot + Feedback | Yes | **Further Improvement** |

### Ablation Study

| Feedback Strategy | Effect | Description |
|----------|------|------|
| No feedback | Baseline | Single-pass generation |
| Direct human modification | Optimal | Requires programming skills |
| Human + LLM execution | Near-optimal | Lowers technical barriers |
| LLM-aided feedback | Significant improvement | Fully automated |

### Key Findings

- CoDialstructured achieves SOTA on STAR and is on par with SOTA on MultiWOZ while being completely zero-shot.
- The structured code generation paradigm significantly outperforms the free generation paradigm, indicating that explicit structural constraints are crucial for code quality.
- Just 1-2 rounds of feedback can significantly improve dialogue success rates.
- User studies confirm that the code generated by CoDial is easier to understand and modify than neural methods.

## Highlights & Insights

- Repositioning LLM guardrails from the safety domain to a general foundation for TOD behavior definition offers a unique perspective.
- Heterogeneous graph representation is more expressive than homogeneous graphs, and JSON encoding naturally fits LLM inputs.
- The combination of zero-shot and interpretability is highly valuable for real-world deployment—no labeled data is required, and every decision is traceable to code logic.

## Limitations & Future Work

- Reliance on the Colang guardrail language; LLMs' familiarity with this language may be limited, potentially affecting code quality.
- Prompts for CoDialstructured are long and complex, increasing token consumption.
- Evaluated only on English datasets; effectiveness in multilingual scenarios remains to be verified.
- Simulation of external actions (API calls) may differ from real-world environments.

## Related Work & Insights

- **vs AnyTOD**: AnyTOD requires manual programming and training, whereas CoDial generates code automatically and is zero-shot.
- **vs SGD-LLM**: Neural schema-based methods are uninterpretable, while CoDial is inherently interpretable.
- **vs NeMo Guardrails**: CoDial is the first to extend guardrails from safety constraints to a general framework for TOD behavior definition.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to model TOD systems as automatically generated LLM guardrail programs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, multiple feedback strategies, and user studies, though the number of benchmarks is limited.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear, with detailed algorithmic logic.
- Value: ⭐⭐⭐⭐ Provides a practical zero-shot framework for interpretable TOD systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](template-assisted_contrastive_learning_of_task-oriented_dialogue_sentence_embedd.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)
- [\[ACL 2026\] STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems](stride-ed_a_strategy-grounded_stepwise_reasoning_framework_for_empathetic_dialog.md)
- [\[ACL 2026\] ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification](odutqa-mdc_a_task_for_open-domain_underspecified_tabular_qa_with_multi-turn_dial.md)
- [\[ACL 2026\] Reasoning Gets Harder for LLMs Inside A Dialogue](reasoning_gets_harder_for_llms_inside_a_dialogue.md)

</div>

<!-- RELATED:END -->
