---
title: >-
  [Paper Note] Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling
description: >-
  [ACL 2025][Dialogue Systems][Dialogue State Tracking] This paper proposes an Accountability Model for task-oriented dialogue systems, which integrates an additional accountability head as a binary classifier into LLMs to predict the probability of each slot in dialogue states. This enables the detection and self-correction of false positive and false negative errors, improving JGA from 64.34 to 70.51 (↑9.6%) on MultiWOZ and achieving SOTA.
tags:
  - "ACL 2025"
  - "Dialogue Systems"
  - "Dialogue State Tracking"
  - "Explainability"
  - "User Overreliance"
  - "Self-Correction"
  - "Friction Turn"
date: 2026-05-08
content_hash: 02709065ae8cffc4
---

# Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling

**Conference**: ACL 2025  
**arXiv**: [2501.10316](https://arxiv.org/abs/2501.10316)  
**Code**: [Yes](https://github.com/uiuc-conversational-ai-lab/Accountable-DST)  
**Area**: NLP / Dialogue Systems / Dialogue State Tracking  
**Keywords**: Dialogue State Tracking, Explainability, User Overreliance, Self-Correction, Friction Turn

## TL;DR

This paper proposes an Accountability Model for task-oriented dialogue systems, which integrates an additional accountability head as a binary classifier into LLMs to predict the probability of each slot in dialogue states. This enables the detection and self-correction of false positive and false negative errors, improving JGA from 64.34 to 70.51 (↑9.6%) on MultiWOZ and achieving SOTA.

## Background & Motivation

Although LLM-based dialogue systems have made significant progress, they face two core challenges:

**Hallucination**: LLMs are prone to generating plausible-sounding but factually incorrect responses.

**User Overreliance**: Users tend to accept AI-generated suggestions, even when those suggestions are incorrect.

In Task-Oriented Dialogue Systems (TODS), these two issues are particularly critical. Dialogue State Tracking (DST) is a key component of TODS, responsible for tracking slot-value pairs representing user intent. DST mistakes fall into three categories:
- **False Positive (FP)**: Predicting a slot not mentioned in the dialogue.
- **False Negative (FN)**: Omitting a slot mentioned in the dialogue.
- **Value Error**: Correct slot, but incorrect value.

Even a single incorrect slot can significantly alter the direction of the conversation. For example, if a user says "I am looking for a park in the centre", and the model misses `attraction-area: centre` (a false negative), the system might recommend parks outside the city center, leading the user to book an inappropriate option due to overreliance on the system.

Existing LLM-based DST systems are purely generative, making it impossible to estimate confidence scores for **slots not present in the output** (thus failing to detect false negatives). The core idea of this work is to combine the classification advantages of traditional slot-filling with modern generative methods.

## Method

### Overall Architecture

An accountability head is added to the LLM backbone, forming a dual-head architecture: (1) a language modeling head that generates the dialogue state; and (2) an accountability head serving as a binary classifier to predict the probability of each slot. The two heads are jointly trained, and slot probabilities are subsequently utilized for self-correction.

### Key Designs

1. **Accountability Head Design**:

    - Extract the encoding of the last token in the dialogue context $C_t$, denoted as $\phi_t \in \mathbb{R}^d$.
    - Pass it through a linear layer and a sigmoid function to obtain the probability of each slot: $p = \sigma(\text{LIN}(\phi_t)) \in \mathbb{R}^{|S|}$.
    - Train the head using binary cross-entropy (BCE) loss.
    - Design Motivation: $\phi_t$ is optimized to minimize both BCE and LM losses, meaning it encodes slot relevance information, which back-propagates to improve generation accuracy.

2. **Joint Training Objective**:

    - $\mathcal{L}_{Account} = \mathcal{L}_{LM} + \lambda \cdot \mathcal{L}_{BCE}$
    - $\lambda \in [0, 1]$ controls the weight of the accountability head.
    - Optimal $\lambda$: 0.25 for MultiWOZ, 0.1-0.25 for Snips.
    - Design Motivation: The auxiliary loss provides prior information on slots, guiding more accurate dialogue state generation.

3. **Dialogue State Self-Correction Algorithm (Algorithm 1)**:

    - **Step 1 — Filter False Positives**: For each predicted slot-value pair, if $p_{slot} < \tau_{fp}$, remove it.
    - **Step 2 — Add False Negatives**: For slots not present in the prediction, if $p_{slot} \geq \tau_{fn}$, invoke `generateSlotValue()` to generate their values.
    - `generateSlotValue()` appends the slot name to the already generated dialogue state and allows the model decoder to continue generating.
    - Optimal thresholds are determined via grid search on the validation set.
    - Design Motivation: Leverage the classification capability of the accountability head to directly correct the generative outputs.

4. **Friction Turn Mechanism**:

    - An alternative to self-correction: confirm detected errors with the user by asking clarification questions.
    - For example, if the model detects a potential omission of `attraction-area`, it actively asks: "What area's park are you looking for?"
    - Design Motivation: Introduce beneficial "positive friction" to promote analytical thinking in users and mitigate overreliance.

### Training Details

- Backbone models: Llama 3.1 (8B), Mistral (7B), Gemma (7B), all instruction-tuned versions.
- Fine-tuned using LoRA (r=8, α=32, dropout=0.1).
- AdamW optimizer, learning rate 5e-5, trained for 4 epochs.
- Optimal thresholds: MultiWOZ $(τ_{fp}, τ_{fn}) = (0.1, 0.5)$; Snips $(0.05, 0.9)$.

## Key Experimental Results

### Main Results (MultiWOZ 2.4 + Snips)

| Backbone Model | Variant | MultiWOZ JGA↑ | MultiWOZ FNR↓ | Snips JGA↑ |
|---------|------|-------------|-------------|-----------|
| Llama | SFT Baseline | 64.34 | 23.72 | 92.43 |
| Llama | +AMD | 67.13 (↑4.3%) | 18.28 | 93.57 |
| Llama | +AMD+SC | **70.51 (↑9.6%)** | **14.44** | **93.71** |
| Mistral | SFT Baseline | 65.86 | 20.41 | 92.57 |
| Mistral | +AMD | 68.58 (↑4.1%) | 16.94 | 93.71 |
| Mistral | +AMD+SC | 69.84 (↑6.0%) | 14.19 | 94.00 |
| Gemma | SFT Baseline | 62.12 | 28.84 | 91.43 |
| Gemma | +AMD | 65.05 (↑4.7%) | 20.15 | 91.86 |
| Gemma | +AMD+SC | 66.27 (↑6.7%) | 15.08 | 92.00 |

### Ablation Study (Threshold Effects, Llama on MultiWOZ)

| $\tau_{fp}$ | $\tau_{fn}$ | JGA↑ | FPR↓ | FNR↓ | Generation Cost (%turns) |
|------------|------------|------|------|------|-----------------|
| 0 (Baseline) | 1 (Baseline) | 67.13 | 13.17 | 18.28 | 0 |
| 0.1 | 1 | 68.15 | 11.16 | 18.92 | 0 |
| 0 | 0.5 | **69.31** | 16.39 | **12.11** | 7.5 |
| 0 | 0.4 | 68.97 | 18.28 | 10.74 | 8.9 |

### Key Findings

1. **Consistent improvements from the Accountability head**: All three backbone models achieved ~3% absolute JGA improvements on MultiWOZ, primarily due to a significant reduction in FNR.
2. **Remarkable self-correction effect**: AMD+SC improved the JGA of Llama from 64.34 to 70.51, achieving SOTA.
3. **Optimal $\lambda$ value is 0.25**: Being too large (1.0) hurts generation quality, while being too small is ineffective.
4. **Trade-off in false negative correction**: Lowering $\tau_{fn}$ reduces FNR but increases FPR, requiring a balance.
5. **Friction turn is as effective as self-correction**: Error correction via user confirmation achieved similar performance gains, validating the practical feasibility of reducing overreliance.

## Highlights & Insights

- **Simple yet effective method**: Adding only a single linear layer (the accountability head) yields consistent and significant improvements across three backbone models.
- **Broad applicability of guiding generation via auxiliary classification loss**: The slot information encoded in $\phi_t$ feedback assists generative capabilities.
- **Framework shift in AI system design from human-AI collaboration**: Instead of purely scaling up accuracy, the system is designed to "know its mistakes," introducing friction turns to guide user analytical thinking.
- **Practical and efficient self-correction algorithm**: Zero cost for false positive filtering, and false negative correction affects only 7.5% of dialogue turns on average.

## Limitations & Future Work

1. Only focuses on the DST task, without extending to end-to-end dialogue systems (e.g., dialogue policy and response generation).
2. False negative correction in self-correction might introduce new false positives, posing a risk of error propagation.
3. No actual human-user study was conducted for friction turns; validations were based purely on simulation.
4. Methods trained on synthetic data such as STAR and ASSIST (JGA ~80%) were not considered, limiting fairness in broad comparisons.
5. The calibration of slot probabilities was not analyzed in depth.

## Related Work & Insights

- Combines the classification advantages of traditional slot-filling with the flexibility of modern generative methods.
- The concept of "friction turn" comes from the HCI domain (Mejtoft et al., 2019), and its application to dialogue systems is highly novel.
- Complementary to confidence estimation works (Sun et al., 2024): the accountability head can evaluate slots not present in the output.
- The auxiliary classification head concept can be transferred to other structured prediction tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of accountability head + self-correction is novel, and introducing friction turns has forward-looking value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers three backbone models, two datasets, detailed threshold ablations, and a comprehensive comparison with SOTA.
- **Writing Quality**: ⭐⭐⭐⭐ — Logical, intuitive illustrations, and standardized algorithm descriptions.
- **Value**: ⭐⭐⭐⭐⭐ — The method is simple, practical, highly effective (achieving SOTA), and offers a fresh perspective on reducing user overreliance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] PersonaLens: A Benchmark for Personalization Evaluation in Conversational AI Assistants](personalens_a_benchmark_for_personalization_evaluation_in_conversational_ai_assi.md)
- [\[ACL 2025\] Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles](know_you_first_and_be_you_better_modeling_human-like_user_simulators_via_implici.md)
- [\[ACL 2026\] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](../../ACL2026/dialogue/codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)
- [\[ACL 2025\] Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision Tree Branching](wizard_of_shopping_target-oriented_e-commerce_dialogue_generation_with_decision_.md)
- [\[ACL 2025\] DEMO: Reframing Dialogue Interaction with Fine-grained Element Modeling](demo_reframing_dialogue_interaction_with_fine-grained_element_modeling.md)

</div>

<!-- RELATED:END -->
