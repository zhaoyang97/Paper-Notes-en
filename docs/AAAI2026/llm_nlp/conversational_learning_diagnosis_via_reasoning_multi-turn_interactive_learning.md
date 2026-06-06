---
title: >-
  [Paper Note] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning
description: >-
  [AAAI 2026][LLM/NLP][Conversational Learning Diagnosis] This paper proposes ParLD (Preview-Analyze-Reason framework), which leverages multi-agent collaboration to achieve fine-grained…
tags:
  - "AAAI 2026"
  - "LLM/NLP"
  - "Conversational Learning Diagnosis"
  - "Knowledge Tracing"
  - "Multi-Agent Collaboration"
  - "Cognitive State"
  - "LLM"
date: 2026-05-08
content_hash: 5e0fa566ef7a3f97
---

# Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning

**Conference**: AAAI 2026  
**arXiv**: [2603.03236](https://arxiv.org/abs/2603.03236)  
**Code**: [https://github.com/fannazya/ParLD](https://github.com/fannazya/ParLD)  
**Area**: LLM/NLP  
**Keywords**: Conversational Learning Diagnosis, Knowledge Tracing, Multi-Agent Collaboration, Cognitive State, LLM

## TL;DR
This paper proposes ParLD (Preview-Analyze-Reason framework), which leverages multi-agent collaboration to achieve fine-grained, turn-level diagnosis of students' cognitive states during conversational learning. ParLD outperforms traditional knowledge tracing methods by 10% on performance prediction and substantially improves tutoring outcomes.

## Background & Motivation

**Background**: Conversational Learning is a pedagogical paradigm in which knowledge is conveyed through multi-turn teacher–student dialogues. Learning Diagnosis aims to continuously monitor students' cognitive states to help instructors adapt their teaching strategies.

**Limitations of Prior Work**: (a) Traditional Knowledge Tracing (KT) and Cognitive Diagnosis Model (CDM) methods rely on structured correct/incorrect labels and can only provide coarse-grained estimates, failing to capture the continuously evolving fine-grained cognitive changes within dialogues. (b) Student responses in conversations are open-ended text, and cognitive signals are distributed across multiple interaction turns, making it difficult to extract stable signals with label-based approaches. (c) Existing LLM applications primarily treat diagnosis as an intermediate step for downstream tasks, relying on direct prompting to analyze dialogue text without psychological grounding, and the diagnostic results are unverifiable.

**Key Challenge**: Cognitive state is an unobservable latent construct, and directly mapping high-dimensional unstructured dialogue text to discrete diagnostic labels constitutes an ill-posed problem that is prone to producing unreliable results.

**Goal**: Formally define the Conversational Learning Diagnosis (CLD) task and design a multi-agent diagnostic framework grounded in psychological theory that is capable of self-verification and self-correction.

**Key Insight**: Inspired by the Zone of Proximal Development (ZPD) theory, the framework first predicts plausible student behavior patterns, then infers cognitive states by comparing actual dialogue against the predictions, and finally employs performance prediction for self-verification.

**Core Idea**: A three-step Preview-Analyze-Reason chain implements a "predict-then-compare" diagnostic paradigm, complemented by a Chain Reflector for self-correction, forming a closed loop.

## Method

### Overall Architecture
ParLD is a multi-agent system comprising four modules: **Behavior Previewer**, **State Analyzer**, **Performance Reasoner**, and **Chain Reflector**. These modules execute iteratively at each dialogue turn: Preview → Analyze → Reason → (if deviation detected) Reflect & Correct.

Input: dialogue history up to turn $t$, $D_{ue}^t = \{d_1, \dots, d_t\}$, learning question $e$, and its associated knowledge concepts $K_e$.  
Output: the student's cognitive state $S_t$ at turn $t$ (mastery level per knowledge concept + textual explanation).

### Key Designs

1. **Behavior Previewer**:

    - **Function**: Before analyzing the actual dialogue, predicts the behavioral patterns the student is likely to exhibit in the current turn.
    - **Mechanism**: Based on ZPD theory, the student's potential behaviors are partitioned into three zones: **Mastered** (can perform independently), **Acquirable** (can perform with teacher guidance), and **Inaccessible** (cannot perform even with guidance). The LLM generates a ZPD-Behavior schema from the prior cognitive state $S_{t-1}$, question features, and knowledge concepts: $B_t = \text{LLM}(S_{t-1}, e, K_e, \mathcal{P}_b)$.
    - **Design Motivation**: Directly inferring cognitive states from dialogue is ill-posed. The ZPD schema provides a structured prior that constrains the diagnostic space from unconstrained text analysis to a bounded behavior-zone matching problem.

2. **State Analyzer**:

    - **Function**: Infers the mastery level of each knowledge concept by comparing the actual dialogue against the predicted behaviors.
    - **Mechanism**: The student's actual behavior at turn $t$ is mapped onto the ZPD-Behavior schema. If the observed behavior aligns with the Acquirable zone, the mastery of the corresponding knowledge concept is inferred to be improving. The structured cognitive state is output as: $S_t = \text{LLM}(S_{t-1}, B_t, d_t, e, \mathcal{P}_a)$, in the format `{"KC1": {"level": "Poor/Fair/Good", "explanation": "..."}}`.
    - **Design Motivation**: With the ZPD schema serving as a reference, the analyzer performs "expected vs. actual" comparative matching rather than interpreting dialogue from scratch, thereby reducing the difficulty of reasoning.

3. **Performance Reasoner + Chain Reflector**:

    - **Function**: Predicts the student's final learning performance based on the current cognitive state, and triggers chain-of-thought self-correction when predictions are incorrect.
    - **Mechanism**: The Reasoner predicts $y_t = \text{LLM}(S_t, e, \mathcal{P}_r)$, outputting both the prediction and its rationale. When the prediction diverges from the actual outcome, the Chain Reflector backtracks through the entire Preview-Analyze-Reason chain, auditing each step in sequence. Reflection results are stored in Conversation Memory for reference in subsequent turns. A max_num parameter limits the number of reflection iterations to control cost.
    - **Design Motivation**: Cognitive states are latent variables that cannot be directly verified. However, learning performance is observable; thus, performance prediction serves as a proxy signal to calibrate diagnostic results, enabling self-correction.

4. **Conversation Memory**:

    - **Function**: Maintains temporary memory for the current learning session, storing the complete operation record for each turn (dialogue, ZPD schema, cognitive state, and reflection results).
    - **Mechanism**: The turn trace $h_t$ produced at each turn contains $d_t, B_t, S_t$, and any reflection records, which are appended to memory. Subsequent modules can reference this historical information. Memory is cleared at the end of each session.

### Loss & Training
- No conventional training is involved. All modules are implemented via LLM APIs (GPT-4.1 and GPT-4o) with temperature=0 to ensure output stability.
- The maximum number of reflection iterations is set to 2 on MathDial and 1 on CoMTA to balance effectiveness and cost.

## Key Experimental Results

### Main Results

| Model | Dataset | ACC↑ | F1↑ | vs Best KT |
|-------|---------|------|-----|-----------|
| ParLD (GPT-4.1) | MathDial | **68.72** | **66.15** | +10.0% vs DKT |
| ParLD (GPT-4o) | MathDial | 65.08 | 64.04 | +6.36% vs DKT |
| ParLD (GPT-4.1) | CoMTA | **57.26** | **56.91** | +3.42% vs AKT |
| DKT | MathDial | 58.72 | 65.26 | - |
| AKT | CoMTA | 53.84 | 52.88 | - |

ParLD (GPT-4.1) significantly outperforms all traditional KT models on both datasets.

### Ablation Study

| Configuration | MathDial ACC | Notes |
|--------------|-------------|-------|
| Full ParLD | **68.72** | Complete framework |
| w/o Previewer | Lowest | Removing ZPD-Behavior schema causes the largest degradation |
| w/o Reflector | Sub-optimal | Without reflection, diagnostic results cannot be self-corrected |
| w/o P+R | Baseline | Only State Analyzer; worst performance |

### Tutoring Enhancement Experiment

| Setting | Correct Rate CR↑ | Avg. Turns Avg.T↓ |
|---------|-----------------|------------------|
| ParLD (GPT-4.1) | **72.22%** | 3.29 |
| Direct Analyze | 62.96% | 3.28 |
| Direct Respond | 56.48% | 3.25 |

ParLD enables 72.22% of students to successfully master the material (vs. 56.48% for Direct Respond). Case studies show that ParLD can guide students to the correct answer within 3 turns, whereas Direct Respond fails even after 10 turns.

### Key Findings
- **The Previewer is the core component**: Removing the ZPD-Behavior schema causes the greatest performance degradation, demonstrating that the "predict-then-compare" paradigm is more effective than direct dialogue analysis.
- **Reflection is effective but costly**: The Reflector improves diagnostic quality through self-verification at the expense of additional LLM calls.
- **Stronger LLMs yield better results**: GPT-4.1 consistently outperforms GPT-4o, indicating that the framework can fully leverage the capabilities of the underlying model.
- **Diagnostic quality directly affects tutoring outcomes**: More reliable cognitive state estimates lead to more targeted instructional guidance.

## Highlights & Insights
- **Psychologically grounded AI design**: Framing "cognitive state diagnosis" as "matching behaviors to developmental zones" via ZPD theory transforms an unconstrained problem into a structured one, providing stronger theoretical grounding and reliability than direct prompting. This "set expectations, then compare actuals" paradigm is transferable to other assessment scenarios.
- **Verifiable self-correction mechanism**: Calibrating unobservable cognitive states using observable learning performance elegantly resolves the challenge of latent variable verification. The Chain Reflector design constitutes a general-purpose self-correction pattern for agent systems.
- **Formalization of the CLD task**: The paper is the first to explicitly define the task formulation for Conversational Learning Diagnosis, establishing a benchmark for future research.

## Limitations & Future Work
- Reliance on LLM APIs results in high cost and latency, making real-time classroom deployment impractical.
- Validation is limited to the mathematics domain; applicability to other subjects such as language and science remains unknown.
- The CoMTA dataset is relatively small (116 dialogues), limiting the statistical significance of evaluations.
- Ground-truth cognitive states remain unavailable; validation can only be performed indirectly through proxy tasks.
- The convergence properties and optimal iteration count for the reflection mechanism lack theoretical analysis.

## Related Work & Insights
- **vs. Traditional KT (DKT, AKT)**: Conventional KT methods process sequences of discrete correct/incorrect labels and are ill-suited for open-ended dialogue scenarios. ParLD operates directly on natural language interactions and provides finer-grained, turn-level diagnosis.
- **vs. Direct LLM Prompting**: Existing LLM-based educational applications typically analyze dialogue with simple prompts, lacking structured reasoning. ParLD provides a more reliable analytical framework through ZPD schema integration and multi-agent collaboration.
- **vs. Multi-Agent Systems**: The well-defined division of labor among ParLD's agents (preview / analyze / reason / reflect) serves as a reference design pattern for decomposing complex reasoning tasks into agent pipelines.

## Rating
- Novelty: ⭐⭐⭐⭐ First to formalize the CLD task; ZPD-driven agent framework is original
- Experimental Thoroughness: ⭐⭐⭐ Two datasets + ablation + tutoring enhancement, but CoMTA is too small
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear; framework description is systematic
- Value: ⭐⭐⭐⭐ Valuable reference for educational AI and agent self-correction mechanisms

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Learning Spatial Decay for Vision Transformers](learning_spatial_decay_for_vision_transformers.md)
- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)
- [\[AAAI 2026\] Improving Sustainability of Adversarial Examples in Class-Incremental Learning](improving_sustainability_of_adversarial_examples_in_class-incremental_learning.md)
- [\[ICLR 2026\] Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions](../../ICLR2026/llm_nlp/unsupervised_evaluation_of_multi-turn_objective-driven_interactions.md)
- [\[NeurIPS 2025\] PluralisticBehaviorSuite: Stress-Testing Multi-Turn Adherence to Custom Behavioral Policies](../../NeurIPS2025/llm_nlp/pluralistic_behavior_suite_stress-testing_multi-turn_adherence_to_custom_behavio.md)

</div>

<!-- RELATED:END -->
