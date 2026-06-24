---
title: >-
  [Paper Note] Random Initialization of Gated Sparse Adapters (RIGSA)
description: >-
  [ICML 2025][Model Compression][sparse fine-tuning] Proposes RIGSA, a sparse fine-tuning method based on randomly initialized full-rank adapters + ReZero gating + iterative magnitude pruning, which retains source task performance better than QLoRA while learning new tasks.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "sparse fine-tuning"
  - "PEFT"
  - "lottery ticket hypothesis"
  - "catastrophic forgetting"
  - "LoRA"
date: 2026-05-08
content_hash: 451e7d5d8b18cbdb
---

# Random Initialization of Gated Sparse Adapters (RIGSA)

**Conference**: ICML 2025  
**arXiv**: [2511.01794](https://arxiv.org/abs/2511.01794)  
**Code**: -  
**Area**: Parameter-Efficient Fine-Tuning / Sparse Adaptation  
**Keywords**: sparse fine-tuning, PEFT, lottery ticket hypothesis, catastrophic forgetting, LoRA  

## TL;DR

Proposes RIGSA, a sparse fine-tuning method based on randomly initialized full-rank adapters + ReZero gating + iterative magnitude pruning, which retains source task performance better than QLoRA while learning new tasks.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: Fine-tuning LLMs on new tasks suffers from the **catastrophic forgetting** issue.

### Background

**Background**: LoRA achieves parameter-efficient fine-tuning via low-rank constraints, but this low-rank limitation degrades performance on complex tasks.

### Key Challenge

**Key Challenge**: Sparse fine-tuning offers an alternative that bypasses rank constraints, potentially enabling more expressive adaptation.

### Core Idea

**Core Idea**: The lottery ticket hypothesis (LTH) suggests that sparse subnetworks exist within dense networks that can match the performance of the full network.

### Additional Notes

**Additional Notes**: Existing sparse fine-tuning methods (such as LT-SFT) initialize the weight delta matrix $\Delta W$ to zero, failing to incorporate "lucky" initializations.

## Method

### Core Idea

Learning $W = W_0 + \alpha \Delta W$, where:
- $W_0$: Frozen pre-trained weights
- $\Delta W$: Randomly initialized full-rank weight delta matrix
- $\alpha$: ReZero-style learnable gating parameter, initialized to $10^{-6}$

**Key Designs**: The near-zero initialization of $\alpha$ ensures training starts near the pre-trained weights $W_0$, while the random initialization of $\Delta W$ allows deviation from pre-trained weights, and weight decay guides them back toward this region.

### Iterative Magnitude Pruning (IMP)

1. Train $W_0 + \alpha \Delta W$ for one epoch
2. Pruning: Among parameters that have not changed sign, keep the top 80% with the largest magnitude, reset the rest to their initial values, and freeze them.
3. Repeat 5 times → final sparsity of approximately 3.46%
4. Train with the final sparse mask to obtain the winning ticket

### Differences from Other Methods

| Method | Initialization | Rank Constraint | Pruning Strategy |
|------|--------|--------|----------|
| LoRA/QLoRA | B initialized to zero | Low-rank | None |
| LT-SFT | $\Delta W = 0$ | None | One-shot pruning |
| RoSA | Gradient accumulation | Joint low-rank + sparse | Gradient-based |
| **RIGSA** | Random + ReZero gating | None (full-rank → sparse) | IMP + sign preservation |

## Key Experimental Results

### Target Task: Textual MNIST

Converts MNIST images into numeric text matrices (each pixel quantized to 0-9) as a pure-text image classification task:
- SmolLM2-1.7B-Instruct zero-shot accuracy is only about 10% (random-guess level)
- Effectively learned after fine-tuning

### Main Results

| Method | Textual MNIST | PIQA | HellaSwag | GSM8k |
|------|--------------|------|-----------|-------|
| Baseline (un-finetuned) | ~10% | 75.4 | 51.7 | 43.7 |
| RIGSA (step 1, dense) | 99.05% | - | - | 40.7↓ |
| RIGSA (step 3, sparse) | 98.37% | ~75 | ~52 | **45.1↑** |
| QLoRA (rank=16) | **99.46%** | ~75 | ~51 | 14.18↓↓ |
| Random Mask | ~97% | - | - | ~43 |

### Key Findings

- **Target Task**: QLoRA is slightly superior to RIGSA (99.46% vs 98.37%).
- **Forgetting**: Forgetting in RIGSA on GSM8k is significantly less than that of QLoRA (decrease of ~0 vs ~6% drop).
- Sparse fine-tuning (including random masking) systematically outperforms QLoRA in retaining source task performance.
- High-rank QLoRA actually retains source task performance better, possibly because high-rank adaptation is more "natural".

## Highlights & Insights

- Innovatively applies LTH concepts to LLM adapters, employing ReZero gating to address the instability of random initialization.
- Proposes Textual MNIST as a standardized OOD visual-textual task.
- Reveals the systematic advantages of sparse fine-tuning in mitigating forgetting.
- The method is simple and bypasses the need for complex mask selection strategies.

## Textual MNIST Task Design

The proposed Textual MNIST serves as a uniquely valuable OOD evaluation benchmark:

- Quantizes each pixel of a 28×28 grayscale image into 0-9, forming a pure-text representation.
- SmolLM2 zero-shot/5-shot accuracy is around 10% (random-guess level).
- The task distribution is completely different from pre-training, clearly measuring the ability to learn new capabilities.
- Unlike BigBench's ASCII art approach, using single-character numeric encoding is more suitable for tokenizers.

This design effectively isolates the effect of transfer learning, making it an ideal testbed for evaluating adapter learning capabilities.

## Limitations & Future Work

- Experiments are restricted to 1.7 B-parameter models, leaving larger models unverified.
- Pruning 80% in each step is aggressive, potentially leading to suboptimal performance.
- Validation with only a single target task (Textual MNIST) is insufficient.
- Lacks statistical significance analysis over multiple experimental runs.
- Lacks direct comparison with stronger sparse fine-tuning baselines like RoSA.
- The choice of high weight decay (1.0) lacks theoretical support.

## Rating

⭐⭐⭐ — The idea is clear and interesting, combining ReZero gating with IMP in a novel way. However, the experimental scale and depth are insufficient to fully validate the advantages of the method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics](beyond_zero_initialization_investigating_the_impact_of_non-zero_initialization_o.md)
- [\[ICML 2025\] Neutral Residues: Revisiting Adapters for Model Extension](neutral_residues_revisiting_adapters_for_model_extension.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](../../NeurIPS2025/model_compression/gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[AAAI 2026\] Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge](../../AAAI2026/model_compression/put_the_space_of_lora_initialization_to_the_extreme_to_preserve_pre-trained_know.md)
- [\[NeurIPS 2025\] Beyond Random: Automatic Inner-Loop Optimization in Dataset Distillation](../../NeurIPS2025/model_compression/beyond_random_automatic_inner-loop_optimization_in_dataset_distillation.md)

</div>

<!-- RELATED:END -->
