---
title: >-
  [Paper Note] From Misleading Queries to Accurate Answers: A Three-Stage Fine-Tuning Method for LLMs
description: >-
  [ACL 2025][LLM Safety][misleading queries] Proposes a three-stage fine-tuning method (misleading detection -> query correction -> accurate response) to enhance the capability of LLMs in processing inputs containing misleading information, significantly improving accuracy in misleading detection and QA tasks while reducing hallucination generation.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "misleading queries"
  - "hallucination"
  - "fine-tuning"
  - "query correction"
  - "robustness"
date: 2026-05-08
content_hash: 3e6996d75ebd3ea1
---

# From Misleading Queries to Accurate Answers: A Three-Stage Fine-Tuning Method for LLMs

**Conference**: ACL 2025  
**arXiv**: [2504.11277](https://arxiv.org/abs/2504.11277)  
**Code**: [https://github.com/cong03/FMQAA](https://github.com/cong03/FMQAA)  
**Area**: LLM Safety  
**Keywords**: misleading queries, hallucination, fine-tuning, query correction, robustness

## TL;DR
Proposes a three-stage fine-tuning method (misleading detection -> query correction -> accurate response) to enhance the capability of LLMs in processing inputs containing misleading information, significantly improving accuracy in misleading detection and QA tasks while reducing hallucination generation.

## Background & Motivation

### Background

**Background**: LLMs are highly sensitive to input quality and are prone to hallucinations when queries contain inaccurate or misleading information.

**Limitations of Prior Work**: Existing methods (such as RAG and self-correction) focus on correcting outputs, overlooking the potential of correcting inputs themselves. RAG requires an external knowledge base, leading to high latency, while self-correction relies heavily on the model's self-evaluation capabilities.

**Key Challenge**: How to enable LLMs to actively identify and correct misleading information in the input rather than passively accepting it and producing erroneous outputs?

**Goal**: Train LLMs to sequentially perform: misleading detection -> query correction -> response based on the corrected query.

**Key Insight**: Decompose "handling misleading inputs" into three independently trainable subtasks.

**Core Idea**: Teach the LLM to "check the question before answering"—enabling the model to actively sanitize inputs through three-stage fine-tuning.

## Method

### Overall Architecture
Utilizes Qwen2.5-72B to generate queries containing misleading information (quality filtering: edit distance similarity >0.8 and error rate >0.5) -> Three-stage fine-tuning: (1) detect whether the query contains misleading information, (2) correct the misleading query, and (3) generate the answer based on the corrected query.

### Key Designs

1. **Misleading Data Construction**

    - Uses LLMs to generate 3 variants for each original query, maintaining high similarity while introducing misleading information.
    - Double filtering: surface similarity $S_{sim}$ > 0.8 + answer error rate $E_{error}$ > 0.5.
    - Design Motivation: Ensure that the misleading query is sufficiently similar to the original query (hard to detect) and actually deceives the model.

2. **Three-Stage Training**

    - Stage 1 — Binary classification detection (YES/NO), cross-entropy loss.
    - Stage 2 — Generate corrected queries, which can be combined with internal or external knowledge.
    - Stage 3 — Generate accurate answers based on corrected queries.
    - Design Motivation: Decompose the task into independently optimizable subtasks, with each stage focusing on a specific capability.

## Key Experimental Results

### Main Results -- QA Accuracy under Misleading Queries

| Method | HaluEval-QAmis | CQAmis | Gain |
|------|---------------|--------|------|
| Vanilla LLM | ~45% | ~40% | Baseline |
| RAG-enhanced | ~55% | ~50% | +10% |
| Self-correction | ~50% | ~48% | +7% |
| **Three-stage method** | **~72%** | **~68%** | **+25%** |

### Ablation Study

| Configuration | Accuracy | Description |
|------|--------|------|
| Stage 1 only | +5% | Detection is helpful but insufficient |
| Stage 1+2 | +15% | Correction further improves performance |
| **Stage 1+2+3** | **+25%** | **Full pipeline is optimal** |

### Key Findings
- **The three-stage method significantly outperforms the baseline on misleading inputs** (+25%) and also improves performance on normal inputs.
- **Misleading queries also exist in standard datasets**: Once identified and corrected, the original model's performance also improves.
- **Balancing detection and answering**: Hallucination detection capabilities are improved simultaneously.
- **Significantly enhanced robustness to inputs**: Stable performance regardless of the presence of misleading information.

## Highlights & Insights
- **"Correcting input before answering"** is a simple yet effective paradigm—akin to humans questioning the premise of a question before answering it.
- The **misleading data construction method** can be transferred to other robustness studies.

## Limitations & Future Work
- Misleading data is generated by LLMs, which may not fully reflect real-world user misinformation.
- The three-stage sequential process increases inference latency.
- Future directions: End-to-end training, integration with RAG.

## Related Work & Insights
- **vs RAG**: RAG corrects outputs, whereas this work corrects inputs—they are complementary.
- **vs Self-correction (Madaan et al.)**: Self-correction relies on the model's self-evaluation, whereas this work enhances detection capability through specialized training.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-stage design of "correcting input before answering" is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets + ablation study + dual scenarios (misleading/normal).
- Writing Quality: ⭐⭐⭐⭐ Clear methodology.
- Value: ⭐⭐⭐⭐ Practical application value for LLM robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach](towards_context-robust_llms_a_gated_representation_fine-tuning_approach.md)
- [\[ICML 2025\] TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs](../../ICML2025/llm_safety/tuco_measuring_the_contribution_of_fine-tuning_to_individual_responses_of_llms.md)
- [\[ICLR 2026\] Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!](../../ICLR2026/llm_safety/be_careful_when_fine-tuning_on_open-source_llms_your_fine-tuning_data_could_be_s.md)
- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [\[ICLR 2026\] SafeMoE: Safe Fine-Tuning for MoE LLMs by Aligning Harmful Input Routing](../../ICLR2026/llm_safety/safemoe_safe_fine-tuning_for_moe_llms_by_aligning_harmful_input_routing.md)

</div>

<!-- RELATED:END -->
