---
title: >-
  [Paper Note] PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning
description: >-
  [NeurIPS 2025][LLM Safety][Machine Unlearning] This paper proposes the PULSE evaluation protocol, which assesses existing unlearning methods for large multimodal models (LMMs) along two practically motivated dimensions:…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Large Multimodal Models"
  - "Pretrained Knowledge"
  - "Sustainability"
  - "Evaluation Benchmark"
date: 2026-05-08
content_hash: a310ef55bc9ba7ff
---

# PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning

**Conference**: NeurIPS 2025
**arXiv**: [2507.01271](https://arxiv.org/abs/2507.01271)  
**Code**: None  
**Area**: AI Safety / Machine Unlearning Evaluation
**Keywords**: Machine Unlearning, Large Multimodal Models, Pretrained Knowledge, Sustainability, Evaluation Benchmark

## TL;DR

This paper proposes the PULSE evaluation protocol, which assesses existing unlearning methods for large multimodal models (LMMs) along two practically motivated dimensions: the forgetting of pretrained knowledge and the sustainability of repeated sequential unlearning. The findings reveal severe deficiencies in current methods—forgetting pretrained knowledge causes over 90% loss of general capability, and after five sequential unlearning operations, model generalization nearly collapses entirely.

## Background & Motivation

- **Background**: As large language models (LLMs) and LMMs become increasingly prevalent, training data may contain personal privacy information and copyright-protected content, making unlearning techniques a growing area of concern. Several unlearning methods have been proposed (e.g., GA, NPO, SIU), and evaluation benchmarks such as TOFU and MUSE exist for LLMs.
- **Limitations of Prior Work**: The LMM unlearning field lacks a practical evaluation framework. The only existing LMM unlearning benchmark, MLLMU-Bench, suffers from two critical limitations:
  1. **Only considers fine-tuning knowledge forgetting**: It evaluates only the forgetting of knowledge acquired during the most recent fine-tuning phase, without assessing forgetting of knowledge obtained during pretraining—yet in practice, information subject to deletion requests is very likely to have been learned during pretraining.
  2. **Only considers single-round unlearning**: In reality, unlearning requests arrive continuously (e.g., different users submitting data deletion requests over time), requiring multiple sequential unlearning operations on the same model.
- **Goal**: PULSE is designed specifically to address these two critical evaluation gaps.

## Method

### Overall Architecture

PULSE extends the conventional "fine-tune then unlearn" evaluation pipeline with two additional evaluation dimensions:

- **Pretrained Knowledge Unlearning**: Evaluates whether unlearning methods can effectively forget knowledge acquired during pretraining.
- **Sustainability Evaluation**: Divides the unlearning targets into multiple subsets and performs sequential unlearning operations, tracking model performance across rounds.

### Key Designs

1. **Problem Formalization**:
    - Data is partitioned into a forget set $\mathcal{D}_{\text{unlearn}}$ and a retain set $\mathcal{D}_{\text{retain}}$.
    - Evaluation covers two aspects: effectiveness (accuracy drop on $\mathcal{D}_{\text{unlearn}}$) and generalization (accuracy preservation on $\mathcal{D}_{\text{retain}}$ and MMBench).
    - Critical setting: **regardless of whether image inputs are provided, the model should not leak any information about the forget targets**—both multimodal and text-only tasks are evaluated.

2. **Pretrained Knowledge Unlearning Design**:
    - Rather than selecting forget targets from fine-tuning data, this protocol selects from knowledge the model already possesses from pretraining.
    - From 153 real celebrities in the MLLMU-Bench dataset, the 45 individuals on which LLaVA-v1.5-13B achieves the highest accuracy are filtered out.
    - 20 individuals form $\mathcal{D}_{\text{unlearn}}$ and 25 form $\mathcal{D}_{\text{retain}}$.
    - Each individual is associated with 10 QA pairs (5 multimodal + 5 text-only).

3. **Sustainability Evaluation Design**:
    - $\mathcal{D}_{\text{unlearn}}$ (50 individuals) is divided into 5 subsets of 10 individuals each.
    - Five sequential unlearning operations are performed on the model.
    - Effectiveness and generalization metrics are tracked after each operation.

### Loss & Training

Three unlearning methods are evaluated:
- **GA (Gradient Ascent)**: Updates parameters in the direction opposite to the gradient on $\mathcal{D}_{\text{unlearn}}$.
- **GA+KLR**: Augments GA with KL divergence regularization to keep the updated model close to the original.
- **NPO**: A preference optimization approach that treats forget data as negative examples.

LLaVA-v1.5-13B serves as the base model; both fine-tuning and unlearning are performed using LoRA.

## Key Experimental Results

### Main Results (Fine-tuned vs. Pretrained Knowledge Unlearning)

| Knowledge Type | Method | $\mathcal{D}_{\text{unlearn}}$ Forget Rate | $\mathcal{D}_{\text{retain}}$ Retention | MMBench Retention |
|---|---|---|---|---|
| Fine-tuned | GA | High (effective) | ~70% | ~90% |
| Fine-tuned | GA+KLR | Moderate | ~75% | ~92% |
| Fine-tuned | NPO | High | ~72% | ~91% |
| Pretrained | GA | High (effective) | Significant drop | **<10%** (catastrophic) |
| Pretrained | GA+KLR | Moderate | Drop | **<10%** |
| Pretrained | NPO | High | Drop | **<10%** |

### Ablation Study (Modality Gap & Parameter Update Target)

| Update Target | Method | $\mathcal{D}_{\text{unlearn}}$ Multi↓ | $\mathcal{D}_{\text{unlearn}}$ Text↓ | MMBench↑ |
|---|---|---|---|---|
| Before Unlearning | — | 78.0 | 76.8 | 75.1 |
| Proj+LLM | GA | 9.6 | 35.2 | 71.1 |
| LLM only | GA | 24.8 | 33.2 | 48.8 |

### Key Findings

- **Pretrained knowledge is extremely difficult to forget**: Although accuracy on $\mathcal{D}_{\text{unlearn}}$ does decrease, MMBench scores collapse by over 90%, indicating that forgetting pretrained knowledge comes at the cost of almost entirely destroying the model's general multimodal capability.
- **Sustainability is wholly inadequate**: After five sequential unlearning rounds, generalization metrics ($\mathcal{D}_{\text{retain}}$ and MMBench) for all methods approach zero, demonstrating that current methods are entirely incapable of handling continuous unlearning requests in real-world settings.
- **Cross-modal unlearning imbalance**: When updating Proj+LLM, multimodal task accuracy drops from 78.0% to 9.6%, while text-only task accuracy only drops to 35.2%—suggesting that existing methods may merely "break the alignment between images and knowledge" rather than truly forgetting the knowledge itself.
- **Parameter selection trade-off**: Updating only the LLM leads to a substantial MMBench drop (48.8%), whereas jointly updating Proj and LLM yields only a minor drop (71.1%). A plausible explanation is that allowing the projection matrix to be updated enables the model to "shortcut" forgetting by severing cross-modal connections.

## Highlights & Insights

- **Filling a critical evaluation gap**: PULSE is the first systematic evaluation protocol for LMM unlearning that covers both pretrained knowledge forgetting and sustainability.
- **Alarming empirical findings**: The discovery that pretrained knowledge unlearning causes 90%+ capability loss directly questions the practical viability of current unlearning methods.
- **Multimodal vs. text-only analysis**: The work reveals a previously overlooked issue in LMM unlearning—forgetting a multimodal task is not equivalent to forgetting the same knowledge in text-only tasks.
- **Practical evaluation design**: Forget targets are selected based on the model's actual behavior rather than requiring access to pretraining data, making the protocol more applicable to real deployment scenarios.

## Limitations & Future Work

- Only LLaVA-v1.5-13B is evaluated; the behavior of other LMMs (e.g., LLaVA-NeXT, InternVL) may differ.
- Only three relatively basic unlearning methods (GA, GA+KLR, NPO) are tested; more advanced approaches are not covered.
- Methods targeting only multimodal tasks, such as SIU, are excluded; while the rationale is sound, this limits the comprehensiveness of the method comparison.
- The dataset scale is small (20–50 individuals), which may be insufficient for high-confidence statistical conclusions.
- The sustainability experiments fix a specific configuration (10 individuals per round, 5 rounds); more variants (e.g., different batch sizes or more rounds) deserve exploration.
- No potential solutions (e.g., elastic weight consolidation, model distillation) are investigated; the paper solely diagnoses the problem.

## Related Work & Insights

- MUSE proposes a multi-dimensional evaluation for LLM unlearning that includes sustainability; PULSE extends this perspective to the multimodal domain.
- TOFU evaluates unlearning on LLMs using fictional persona data; MLLMU-Bench is the first LMM unlearning benchmark.
- Yao et al. (2024) explore pretrained knowledge unlearning in LLMs but require access to pretraining data.
- Core takeaway: **current unlearning techniques are far from practically viable for LMMs**, and fundamental methodological advances are needed.

## Rating

- Novelty: ⭐⭐⭐⭐ (The new evaluation perspective is valuable, though no algorithmic contribution is made)
- Experimental Thoroughness: ⭐⭐⭐ (Coverage of models and methods is relatively narrow)
- Writing Quality: ⭐⭐⭐⭐ (Problem motivation is clear; experimental design is sound)
- Value: ⭐⭐⭐⭐ (Provides important benchmarking significance for the LMM unlearning community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](a_reliable_cryptographic_framework_for_empirical_machine_unl.md)
- [\[NeurIPS 2025\] FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model](falcon_fine-grained_activation_manipulation_by_contrastive_orthogonal_unalignmen.md)
- [\[AAAI 2026\] Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](../../AAAI2026/llm_safety/cross-modal_unlearning_via_influential_neuron_path_editing_i.md)
- [\[ACL 2026\] MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models](../../ACL2026/llm_safety/muse_a_run-centric_platform_for_multimodal_unified_safety_evaluation_of_large_la.md)
- [\[NeurIPS 2025\] One Token Embedding Is Enough to Deadlock Your Large Reasoning Model](one_token_embedding_is_enough_to_deadlock_your_large_reasoning_model.md)

</div>

<!-- RELATED:END -->
