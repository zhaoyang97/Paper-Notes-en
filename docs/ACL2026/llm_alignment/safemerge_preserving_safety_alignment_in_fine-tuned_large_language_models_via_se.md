---
title: >-
  [Paper Note] SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging
description: >-
  [ACL 2026][LLM Alignment][Safety Alignment] This paper proposes SafeMERGE, a lightweight post-fine-tuning framework that detects layers deviating from safe behavior via cosine similarity…
tags:
  - "ACL 2026"
  - "LLM Alignment"
  - "Safety Alignment"
  - "Model Merging"
  - "LoRA Fine-Tuning"
  - "Post-Fine-Tuning Defense"
  - "Layer-Selective Merging"
date: 2026-05-08
content_hash: da42fb6acc1d1691
---

# SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging

**Conference**: ACL 2026
**arXiv**: [2503.17239](https://arxiv.org/abs/2503.17239)
**Code**: [GitHub](https://github.com/aladinD/SafeMERGE)
**Area**: LLM Alignment / Safety
**Keywords**: Safety Alignment, Model Merging, LoRA Fine-Tuning, Post-Fine-Tuning Defense, Layer-Selective Merging

## TL;DR

This paper proposes SafeMERGE, a lightweight post-fine-tuning framework that detects layers deviating from safe behavior via cosine similarity, and selectively merges only those layers with their counterparts from a safety model. Across four LLMs, the method significantly reduces harmful outputs while maintaining or even improving task performance.

## Background & Motivation

**State of the Field**: Fine-tuning LLMs for domain-specific tasks is common practice, yet research has shown that fine-tuning—even on benign data—erodes safety alignment. As few as a handful of malicious samples can cause an aligned model to comply with harmful requests. Safety alignment has been shown to be "shallow" and easily disrupted during fine-tuning.

**Limitations of Prior Work**: (1) Alignment-stage defenses require modifying the initial alignment pipeline, which is practitioner-unfriendly; (2) Fine-tuning-stage defenses require custom training algorithms that are difficult to integrate with standard open-source libraries; (3) Naive post-fine-tuning defenses (e.g., full-layer merging as in RESTA) typically sacrifice task performance in exchange for safety.

**Root Cause**: How can safety be restored after fine-tuning without modifying the existing training pipeline, while simultaneously preserving task performance?

**Paper Goals**: Design a simple, plug-and-play post-fine-tuning framework that performs selective merging only when necessary—i.e., when a layer deviates from safe behavior.

**Starting Point**: The weight difference between the aligned model and the base model is used to define a "safety alignment subspace," and cosine similarity is employed to detect whether fine-tuned LoRA layers deviate from this subspace.

**Core Idea**: Merge only those layers that deviate from safe behavior, preserving task performance in the remaining layers—selective merging outperforms global merging.

## Method

### Overall Architecture

SafeMERGE proceeds in three steps: (1) train a safety LoRA model using a publicly available safety dataset (trained once and reused across tasks); (2) apply safety subspace projection to detect which layers of the fine-tuned model are "unsafe"; (3) perform linear merging with the safety model exclusively on the identified unsafe layers.

### Key Designs

1. **Safety Alignment Subspace and Layer Selection**:

    - Function: Automatically identify layers that deviate from safe behavior after fine-tuning.
    - Mechanism: The safety subspace is defined as $V^i = W_{aligned}^i - W_{unaligned}^i$ (the weight difference between the aligned and base models). The cosine similarity $\rho^i$ between the fine-tuned LoRA layer $\Delta W_f^i$ and its projection onto the safety subspace $C^i \Delta W_f^i$ is computed. If $\rho^i < \tau$ (a threshold), the layer is flagged as unsafe.
    - Design Motivation: SafeLoRA projects all layers uniformly onto the safety subspace, which degrades task performance. SafeMERGE intervenes only on deviating layers, preserving the learned representations of the remaining layers.

2. **Selective Layer Merging**:

    - Function: Apply safety restoration exclusively to unsafe layers.
    - Mechanism: For layers flagged as unsafe, linear merging is performed as $\Delta W_{merge}^i = \alpha \Delta W_f^i + (1-\alpha) \Delta W_s^i$, where $\Delta W_s^i$ is the corresponding layer from the safety model. $\alpha$ controls the trade-off between task performance and safety. Layers deemed safe retain their fine-tuned weights unchanged.
    - Design Motivation: Global merging (RESTA) applies safety correction to all layers—including those already safe—unnecessarily degrading task performance.

3. **Safety Model Construction**:

    - Function: Provide safe reference layers for merging.
    - Mechanism: A publicly available safety dataset (harmful prompts paired with safe responses) is used to LoRA fine-tune the aligned model. Different dataset sizes (100/500/1000/2500 samples) are evaluated, and the model with the lowest harmfulness score is selected. The safety model is task-agnostic and can be reused across tasks after a single training run.
    - Design Motivation: The safety model provides a parametric representation of "safe behavior," giving the merging process a well-defined target.

### Loss & Training

The safety model is trained using standard LoRA fine-tuning. SafeMERGE itself requires no training—only cosine similarity computation and linear merging, both of which can run entirely on CPU. Evaluation is conducted using Llama-Guard-3-8B and ShieldGemma-9B for cross-validation.

## Key Experimental Results

### Main Results

| Method | Llama-3.1 GSM8K↑ | DirectHarm↓ | HexPhi↓ |
|--------|------------------|-------------|---------|
| Original Aligned Model | 73.80 | 11.30 | 7.90 |
| After Fine-Tuning | 78.24 | 28.30 | 14.70 |
| SafeInstruct | 77.40 | 12.50 | 7.20 |
| RESTA | 74.20 | 11.90 | 6.90 |
| SafeLoRA | 77.90 | 15.10 | 7.10 |
| **SafeMERGE** | **78.50** | **8.80** | **6.30** |

### Ablation Study

| Analysis Dimension | Finding |
|--------------------|---------|
| Merging strategy (Linear vs. DARE vs. TIES) | Linear merging is sufficient |
| Threshold $\tau$ sensitivity | Larger $\tau$ merges more layers: safety↑ but task performance may↓ |
| Safety data size | 500–1000 samples are generally optimal |
| Different weighting schemes | Uniform $\alpha$ generally performs well |

### Key Findings

- SafeMERGE consistently outperforms or matches baselines across all 4 LLMs × 2 task settings.
- On Llama-3.1, SafeMERGE surpasses the original aligned model in task performance (78.50 vs. 73.80) while being safer (8.80 vs. 11.30).
- Selective merging outperforms full-layer merging (RESTA)—RESTA shows a notable drop in task performance (74.20 vs. 78.50).
- The safety model can be reused across tasks without retraining for each new task.

## Highlights & Insights

- The intuition of "only fixing the layers that need fixing" is simple yet highly effective—selective intervention outperforms global correction.
- The ability to run entirely on CPU without any retraining makes SafeMERGE highly practical for real-world deployment.
- The one-time training and cross-task reuse of the safety model substantially lowers the barrier to adoption.

## Limitations & Future Work

- The definition of the safety subspace requires access to both the aligned model and the base model—not all models have publicly available base versions.
- Validation is limited to 7B–8B models; layer selection characteristics may differ for larger models.
- The threshold $\tau$ requires manual tuning; an automatic selection method is currently absent.
- Only LoRA fine-tuning is considered; applicability to full-parameter fine-tuning settings remains unknown.

## Related Work & Insights

- **vs. SafeLoRA**: SafeLoRA uniformly projects all layers onto the safety subspace, incurring task information loss; SafeMERGE selectively merges only unsafe layers.
- **vs. RESTA**: RESTA globally subtracts a "harmful task vector" without distinguishing between safe and unsafe layers; SafeMERGE's selective strategy is more fine-grained.
- **vs. SafeInstruct**: SafeInstruct incorporates safety samples into the training data, requiring modification of the training pipeline; SafeMERGE is entirely post-hoc.

## Rating

- Novelty: ⭐⭐⭐ The selective merging idea is intuitive and effective, but technically represents a combination of existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four models × five tasks, cross-validation, and extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear and concise, with an intuitive method description.
- Value: ⭐⭐⭐⭐⭐ Extremely high practical value—simple, effective, and plug-and-play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SafeNlidb: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces](../../AAAI2026/llm_alignment/safenlidb_a_privacy-preserving_safety_alignment_framework_for_llm-based_natural_.md)
- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](../../ICLR2026/llm_alignment/guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)
- [\[AAAI 2026\] EASE: Practical and Efficient Safety Alignment for Small Language Models](../../AAAI2026/llm_alignment/ease_practical_and_efficient_safety_alignment_for_small_language_models.md)
- [\[ICLR 2026\] Towards Understanding Valuable Preference Data for Large Language Model Alignment](../../ICLR2026/llm_alignment/towards_understanding_valuable_preference_data_for_large_language_model_alignmen.md)
- [\[ICLR 2026\] A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models](../../ICLR2026/llm_alignment/a2d_any-order_any-step_safety_alignment_for_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
