---
title: >-
  [Paper Note] SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging
description: >-
  [ACL 2026][LLM Safety][Safety Alignment] This paper proposes SafeMERGE, a lightweight post-fine-tuning framework that identifies fine-tuned layers deviating from safe behavior via cosine similarity and merges only these…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Safety Alignment"
  - "Model Merging"
  - "LoRA Fine-Tuning"
  - "Post-Fine-Tuning Defense"
  - "Selective Layer Merging"
date: 2026-05-08
content_hash: dd70e883c8de6abe
---

# SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging

**Conference**: ACL 2026 Findings  
**arXiv**: [2503.17239](https://arxiv.org/abs/2503.17239)  
**Code**: [GitHub](https://github.com/aladinD/SafeMERGE)  
**Area**: LLM Alignment / Safety  
**Keywords**: Safety Alignment, Model Merging, LoRA Fine-Tuning, Post-Fine-Tuning Defense, Selective Layer Merging

## TL;DR

This paper proposes SafeMERGE, a lightweight post-fine-tuning framework that identifies fine-tuned layers deviating from safe behavior via cosine similarity and merges only these layers with corresponding layers of a safety model. It significantly reduces harmful outputs across four LLMs while maintaining or even enhancing task performance.

## Background & Motivation

**Background**: Fine-tuning LLMs for specific domains is a common practice, but research indicates that fine-tuning (even with harmless data) can erode safety alignment—just a few malicious samples can cause an aligned model to comply with harmful requests. Safety alignment has been proven to be "shallow" and easily compromised during fine-tuning.

**Limitations of Prior Work**: (1) Alignment-stage defenses require modifying the initial alignment pipeline, which is unfriendly to practitioners; (2) Fine-tuning-stage defenses require custom training algorithms, making them difficult to integrate with standard open-source libraries; (3) Simple post-fine-tuning defenses (such as full-layer merging like RESTA) often sacrifice task performance for safety.

**Key Challenge**: How to restore safety after fine-tuning without modifying existing training processes or compromising task performance?

**Goal**: Design a simple, plug-and-play post-fine-tuning framework that performs selective merging only when necessary (when layers deviate from safe behavior).

**Key Insight**: Utilize the weight difference between the aligned model and the base model to define a "safety alignment subspace" and detect whether fine-tuned LoRA layers deviate from this subspace using cosine similarity.

**Core Idea**: Merge only those layers that deviate from safe behavior while preserving the task performance of other layers—selectivity is superior to global merging.

## Method

### Overall Architecture

SafeMERGE involves three steps: (1) Train a safety LoRA model (using public safety datasets, reusable after one training); (2) Detect which layers of the fine-tuned model are "unsafe" via safety subspace projection; (3) Execute linear merging only for the unsafe layers with the corresponding layers of the safety model.

### Key Designs

1. **Safety Alignment Subspace and Layer Selection**:

    - Function: Automatically identify layers that deviate from safe behavior after fine-tuning.
    - Mechanism: Safety subspace $V^i = W_{aligned}^i - W_{unaligned}^i$ (the weight difference between the aligned model and its base version). Calculate the cosine similarity $\rho^i$ between the fine-tuned LoRA layer $\Delta W_f^i$ and its projection $C^i \Delta W_f^i$ on the safety subspace. If $\rho^i < \tau$ (threshold), the layer is marked as unsafe.
    - Design Motivation: SafeLoRA applies projection to all layers uniformly, which damages task performance; SafeMERGE intervenes only in the deviating layers, preserving the learning of other layers.

2. **Selective Layer Merging**:

    - Function: Perform safety restoration only for unsafe layers.
    - Mechanism: For layers marked as unsafe, perform linear merging $\Delta W_{merge}^i = \alpha \Delta W_f^i + (1-\alpha) \Delta W_s^i$, where $\Delta W_s^i$ is the corresponding layer of the safety model. $\alpha$ controls the trade-off between task performance and safety. Safe layers maintain the fine-tuned weights unchanged.
    - Design Motivation: Global merging (RESTA) applies safety correction to all layers, modifying even those that are already safe, which unnecessarily compromises task performance.

3. **Safety Model Construction**:

    - Function: Provide safety reference layers for merging.
    - Mechanism: Use public safety datasets (harmful prompt + safe response pairs) to LoRA fine-tune an aligned model. Test different data volumes (100/500/1000/2500 samples) and select the model with the lowest toxicity score. The safety model is task-agnostic and reusable across tasks after one training.
    - Design Motivation: The safety model provides a parameterized representation of "safe behavior," giving the merging process a clear target.

### Loss & Training

The safety model is fine-tuned using standard LoRA. SafeMERGE itself requires no training—it only involves computing cosine similarities and linear merging, which can run entirely on a CPU. Evaluation uses Llama-Guard-3-8B and ShieldGemma-9B for cross-verification.

## Key Experimental Results

### Main Results

| Method | Llama-3.1 GSM8K↑ | DirectHarm↓ | HexPhi↓ |
|------|-----------------|-------------|---------|
| Original Aligned Model | 73.80 | 11.30 | 7.90 |
| After Fine-Tuning | 78.24 | 28.30 | 14.70 |
| SafeInstruct | 77.40 | 12.50 | 7.20 |
| RESTA | 74.20 | 11.90 | 6.90 |
| SafeLoRA | 77.90 | 15.10 | 7.10 |
| **SafeMERGE** | **78.50** | **8.80** | **6.30** |

### Ablation Study

| Analysis Dimension | Result |
|----------|------|
| Merging Strategy (Linear vs DARE vs TIES) | Linear merging is sufficient |
| Threshold τ Sensitivity | Larger τ merges more layers, increasing safety but potentially decreasing task performance |
| Safety Data Volume | 500-1000 samples are usually optimal |
| Weighting Schemes | Uniform α generally performs well |

### Key Findings

- SafeMERGE consistently outperforms or matches baselines across all 4 LLMs × 2 task settings.
- On Llama-3.1, SafeMERGE even exceeds the task performance of the original aligned model (78.50 vs 73.80) while being safer (8.80 vs 11.30).
- Selective merging is superior to full-layer merging (RESTA)—RESTA shows a significant drop in task performance (74.20 vs 78.50).
- The safety model is reusable across tasks, eliminating the need for retraining for every new task.

## Highlights & Insights

- The intuition of "fixing only the layers that need fixing" is simple but highly effective—selective intervention is superior to global intervention.
- The ability to run entirely on a CPU without retraining makes it highly valuable for practical deployment.
- The design of a safety model that is reusable across tasks after a single training significantly reduces the cost of adoption.

## Limitations & Future Work

- The definition of the safety subspace depends on the availability of both the aligned and base models—not all models release their base versions.
- Validation was performed only on 7B-8B models; the layer selection characteristics of larger models might differ.
- The threshold $\tau$ requires tuning, and there is currently no automatic selection method.
- Only LoRA fine-tuning was considered; the applicability to full-parameter fine-tuning scenarios remains unknown.

## Related Work & Insights

- **vs SafeLoRA**: SafeLoRA uniformly projects all layers into the safety subspace, losing some task information; SafeMERGE selectively merges only unsafe layers.
- **vs RESTA**: RESTA globally subtracts a "harmful task vector" without distinguishing between safe and unsafe layers; SafeMERGE’s selective strategy is more granular.
- **vs SafeInstruct**: SafeInstruct mixes safety samples into the training data, requiring modification of the training process; SafeMERGE is an entirely post-processing approach.

## Rating

- Novelty: ⭐⭐⭐ The idea of selective merging is intuitive and effective, though technically it is a combination of existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models × 5 tasks, cross-verification, and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear and concise, with an intuitive description of the method.
- Value: ⭐⭐⭐⭐⭐ Extremely high practical value—simple, effective, and plug-and-play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models](sharedrequest_privacy-preserving_model-agnostic_inference_for_large_language_mod.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)
- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)
- [\[AAAI 2026\] SafeNlidb: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces](../../AAAI2026/llm_safety/safenlidb_a_privacy-preserving_safety_alignment_framework_for_llm-based_natural_.md)
- [\[ACL 2026\] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking](seeing_no_evil_blinding_large_vision-language_models_to_safety_instructions_via_.md)

</div>

<!-- RELATED:END -->
