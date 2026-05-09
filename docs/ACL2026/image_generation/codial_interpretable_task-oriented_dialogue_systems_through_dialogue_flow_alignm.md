---
title: >-
  [Paper Note] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment
description: >-
  [ACL 2026][Image Generation][Task-oriented dialogue] This paper proposes CoDial, a framework that converts predefined dialogue flows (task schemas) into structured heterogeneous graphs and automatically generates LLM guardrail code (e.g., Colang), achieving interpretable and controllable task-oriented dialogue policies at inference time. It reaches SOTA on the STAR benchmark without requiring training data.
tags:
  - ACL 2026
  - Image Generation
  - Task-oriented dialogue
  - LLM guardrails
  - Interpretability
  - Dialogue flow alignment
  - Zero-shot generalization
date: 2026-05-08
content_hash: 28fac02ec1e4f43c
---

# CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment

**Conference**: ACL 2026  
**arXiv**: [2506.02264](https://arxiv.org/abs/2506.02264)  
**Code**: [https://github.com/radinshayanfar/CoDial](https://github.com/radinshayanfar/CoDial)  
**Area**: Image Generation  
**Keywords**: Task-oriented dialogue, LLM guardrails, Interpretability, Dialogue flow alignment, Zero-shot generalization

## TL;DR

This paper proposes CoDial, a framework that converts predefined dialogue flows (task schemas) into structured heterogeneous graphs and automatically generates LLM guardrail code (e.g., Colang), achieving interpretable and controllable task-oriented dialogue policies at inference time. It reaches SOTA on the STAR benchmark without requiring training data.

## Background & Motivation

**State of the Field**: Task-oriented dialogue (TOD) systems need to generalize across different tasks. Data-driven approaches struggle to transfer to unseen tasks; schema-based methods improve generalization by decoupling language understanding from task logic, but rely on neural or generative models for schema parsing, lacking interpretability.

**Limitations of Prior Work**: (1) Neural network-based schema methods are opaque, preventing users from understanding how schemas influence dialogue behavior; (2) Methods like AnyTOD achieve interpretability through programmatic implementation but require users to possess programming skills to manually write policy programs, raising technical barriers; (3) Interpretability is particularly critical in high-risk domains such as legal and medical applications.

**Root Cause**: Existing TOD systems face a tradeoff between generalization capability and interpretability—neural methods have generalization but lack interpretability, while programmatic methods are interpretable but require programming expertise.

**Paper Goals**: Design a TOD framework that requires neither training data nor manual programming, automatically converting dialogue flows into executable LLM guardrail programs to provide interpretable and controllable dialogue behavior at inference time.

**Starting Point**: Reposition LLM guardrails as the foundation for defining TOD system behavior, leveraging LLM code generation capabilities to automatically convert dialogue flows into guardrail code.

**Core Idea**: Dialogue flow → Heterogeneous graph (CHIEF) → Guardrail code (Colang) → Executable TOD system, with the entire pipeline automated and inherently interpretable.

## Method

### Overall Architecture

CoDial consists of three components: (1) CHIEF (heterogeneous dialogue flow representation)—defines task schemas as heterogeneous directed graphs supporting different node types (request/external action/inform/confirm/global/fallback); (2) GCG (guardrail code generation)—uses LLMs to automatically convert CHIEF's JSON representation into Colang guardrail code; (3) CHF (human feedback mechanism)—iteratively optimizes generated guardrail code through human or LLM feedback.

### Key Designs

1. **CHIEF Heterogeneous Dialogue Flow Representation**:

    - Function: Define rich task schemas in a structured manner
    - Mechanism: Designs different node types (Request defines slots to track, External Action calls external functions, Inform/Confirm provide information and confirmation, Global/Fallback handle global and fallback actions), connects nodes with conditional edges. The entire structure is encoded in JSON format
    - Design Motivation: Prior work used homogeneous graphs (all nodes of the same type), unable to express rich task logic; heterogeneous graphs support different node types and metadata

2. **Two Paradigms for Guardrail Code Generation (CoDialfree/CoDialstructured)**:

    - Function: Automatically convert dialogue flows into executable guardrail programs
    - Mechanism: CoDialfree provides Colang syntax documentation for LLMs to freely design guardrail logic; CoDialstructured explicitly guides LLMs on how to model dialogue state, implement DST (dialogue state tracking) and NAP (next action prediction), generating structured guardrail code
    - Design Motivation: CoDialfree serves as an interpretable baseline to verify the feasibility of automatic code generation; CoDialstructured improves code quality and reliability through explicit structural constraints

3. **CoDial Human Feedback Mechanism (CHF)**:

    - Function: Iteratively optimize generated guardrail code
    - Mechanism: Supports three feedback modes: (a) human experts directly modify code; (b) humans provide natural language modification suggestions executed by LLMs; (c) LLMs automatically analyze dialogue failures and generate modification suggestions (LLM-aided feedback)
    - Design Motivation: Automatically generated code may contain errors or omissions; iterative feedback mechanisms allow continuous improvement

### Loss & Training

CoDial is a zero-shot, training-free framework. All dialogue policies are executed through guardrail code at inference time, requiring no gradient updates. Core computation comes from LLM code generation and dialogue inference.

## Key Experimental Results

### Main Results

**Performance on STAR Benchmark (Task Success Rate %)**

| Method | Training Required | Interpretability | Success Rate |
|--------|------------------|------------------|--------------|
| SGD-LLM | Requires training | No | Lower |
| AnyTOD | Requires training + manual programming | Yes | Medium |
| CoDialfree | Zero-shot | Yes | Competitive |
| CoDialstructured | Zero-shot | Yes | **SOTA** |
| CoDialstructured + CHF | Zero-shot + feedback | Yes | **Further improvement** |

### Ablation Study

| Feedback Strategy | Effect | Note |
|------------------|--------|------|
| No feedback | Baseline | Single generation |
| Direct human modification | Optimal | Requires programming skills |
| Human + LLM execution | Near optimal | Lowers technical barrier |
| LLM-aided feedback | Significant improvement | Fully automated |

### Key Findings

- CoDialstructured achieves SOTA on STAR, matches SOTA on MultiWOZ, and is completely zero-shot
- Structured code generation paradigm significantly outperforms free generation, indicating explicit structural constraints are critical for code quality
- Just 1-2 rounds of feedback can significantly improve dialogue success rate
- User studies confirm CoDial-generated code is easier to understand and modify than neural methods

## Highlights & Insights

- Repositions LLM guardrails from the safety domain as a general foundation for TOD behavior definition, offering a unique perspective
- Heterogeneous graph representation is more expressive than homogeneous graphs, and JSON encoding naturally fits LLM input
- The combination of zero-shot and interpretability has extremely high value in practical deployment—no annotation data required, and every decision is traceable to code logic

## Limitations & Future Work

- Relies on the Colang guardrail language, which LLMs have limited familiarity with, potentially affecting code quality
- CoDialstructured prompts are lengthy and complex, increasing token consumption
- Evaluated only on English datasets; effectiveness in multilingual scenarios remains to be verified
- Simulation of external actions (API calls) may differ from real-world environments

## Related Work & Insights

- **vs AnyTOD**: AnyTOD requires manual programming and training; CoDial automatically generates code and is zero-shot
- **vs SGD-LLM**: Neural schema-based methods are not interpretable; CoDial is inherently interpretable
- **vs NeMo Guardrails**: CoDial is the first to extend guardrails from safety constraints to a general framework for TOD behavior definition

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to model TOD systems as automatically generated LLM guardrail programs
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, multiple feedback strategies, user studies, but limited number of benchmarks
- Writing Quality: ⭐⭐⭐⭐ Clear framework description, detailed algorithmic pseudocode
- Value: ⭐⭐⭐⭐ Provides a practical zero-shot framework for interpretable TOD systems

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching](zipvoice-dialog_non-autoregressive_spoken_dialogue_generation_with_flow_matching.md)
- [\[CVPR 2026\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](../../CVPR2026/image_generation/codrawagents_a_multiagent_dialogue_framework_for_c.md)
- [\[ACL 2026\] Investigating Counterfactual Unfairness in LLMs towards Identities through Humor](investigating_counterfactual_unfairness_in_llms_towards_identities_through_humor.md)
- [\[CVPR 2026\] Interpretable and Steerable Concept Bottleneck Sparse Autoencoders](../../CVPR2026/image_generation/interpretable_and_steerable_concept_bottleneck_sparse_autoencoders.md)
- [\[ACL 2026\] Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models](follow_the_flow_on_information_flow_across_textual_tokens_in_text-to-image_model.md)

<!-- RELATED:END -->
