---
title: >-
  [Paper Note] Neutral Residues: Revisiting Adapters for Model Extension
description: >-
  [ICML2025][Model Compression][Adapter] This paper proposes **Neutral Residues**, which introduces a ReLU gate, an $\ell_1$ sparse local loss, and low-variance initialization into adapters. This design forces the added residual blocks to output near-zero values on the original distribution, achieving the optimal trade-off between learning a new language and preserving English capabilities on Gemma-2B.
tags:
  - "ICML2025"
  - "Model Compression"
  - "Adapter"
  - "Catastrophic Forgetting"
  - "Model Extension"
  - "Gating Mechanism"
  - "Multilingual"
  - "LLM"
date: 2026-05-08
content_hash: 3bcd3b5032c453df
---

# Neutral Residues: Revisiting Adapters for Model Extension

**Conference**: ICML2025  
**arXiv**: [2410.02744](https://arxiv.org/abs/2410.02744)  
**Code**: Not open-sourced  
**Area**: Model Extension  
**Keywords**: Adapter, Catastrophic Forgetting, Model Extension, Gating Mechanism, Multilingual, LLM

## TL;DR

This paper proposes **Neutral Residues**, which introduces a ReLU gate, an $\ell_1$ sparse local loss, and low-variance initialization into adapters. This design forces the added residual blocks to output near-zero values on the original distribution, achieving the optimal trade-off between learning a new language and preserving English capabilities on Gemma-2B.

## Background & Motivation

Training Large Language Models from scratch is extremely expensive (e.g., Llama 3 is estimated to cost hundreds of millions of dollars). Consequently, **model extension** has become a critical need: adding new capabilities to an existing model without full retraining.

Limitations of existing methods:

- **Fine-tuning**: Does not increase model capacity; acquiring a large amount of new knowledge leads to severe catastrophic forgetting.
- **LoRA**: Low-rank updates also do not increase model capacity, yielding limited effectiveness under significant distribution shifts.
- **Vanilla Adapter**: While introducing additional parameters, it still suffers from substantial forgetting.

Key Challenge: The **trade-off between learning a new distribution and preserving original capabilities**. The authors address this challenge by jointly optimizing across three dimensions: data, architecture, and training.

## Method

### 1. Mixed Distribution Training

Retaining a small portion ($p = 10\%$) of approximate original distribution data (such as English Wikipedia) during training significantly alleviates forgetting without noticeably slowing down the learning of the new language:

| English Data Ratio $p$ | English PPL ↓ | French PPL ↓ |
|:-:|:-:|:-:|
| 0.00 | 0.720 | 0.810 |
| 0.01 | 0.707 | 0.810 |
| **0.10** | **0.687** | **0.812** |
| 0.50 | 0.683 | 0.828 |

### 2. Gated Adapter Architecture

An adapter block is added **in parallel** to the FFN of each Transformer layer, accompanied by an external **block gate**:

$$
\text{Output} = \text{FFN}(x) + g(x) \cdot \text{Adapter}(x)
$$

The internal structure of the adapter still utilizes the GLU formulation:

$$
\text{Adapter}(x) = \mathbf{W}_o \left( \sigma(\mathbf{W}_g x) \odot \mathbf{W}_i x \right)
$$

The **ReLU** activation function (rather than Sigmoid) is chosen for the gating mechanism, allowing the gate output to naturally shrink to zero on the original distribution.

### 3. Local Sparse Loss

The total training loss is defined as:

$$
\ell_{\text{train}} = \ell_{\text{LM}} + \alpha \, \ell_{\text{local}}
$$

- $\ell_{\text{local}}$: Evaluates the mean of the $\ell_1$ norms of all adapter outputs (normalized by the model dimension) on the original distribution.
- Default $\alpha = 0.01$.
- This loss is applied **only to the original distribution data**, forcing the adapter to output near-zero values when processing English $\rightarrow$ "neutral residues".

Comparison with Sigmoid + Cross-Entropy gating: ReLU + $\ell_1$ avoids explicit classification, allowing the gates to adaptively adjust response intensity and preventing the singular values of the gating weight matrix from becoming overly skewed.

### 4. Low-Variance Initialization

- The output matrix is initialized to **all zeros** (standard practice).
- The initialization variance of the input and gating matrices is scaled down from the He initialization standard of $2/d$ to $1/(d \cdot L)$ (where $L$ is the number of Transformer layers). This helps the model remain closer to the original network for a longer duration during the early stages of training.

### 5. FFN vs. MHA

Unlike LoRA, which typically performs better on attention, adding extra parameters to the **FFN** is more effective in model extension scenarios. This is consistent with the design choices of MoE (Mixture of Experts).

## Key Experimental Results

### Main Results: Extending Gemma-2B to Four Languages ($p=0.1$, 20% Extra Parameters)

| Target Language | Method | Forgetting (English avg) ↑ | Learning (Target avg) ↑ |
|:-:|:-:|:-:|:-:|
| French | Backbone | 53.2 | 44.6 |
| | Fine-tuning | 49.8 | **49.9** |
| | LoRA | 51.2 | 44.9 |
| | Adapter | 52.8 | 46.0 |
| | **Neutral Residues** | **53.3** | 48.2 |
| Danish | Fine-tuning | 47.2 | 42.8 |
| | **Neutral Residues** | **52.0** | **42.9** |
| Hungarian | Fine-tuning | 45.9 | 38.5 |
| | **Neutral Residues** | **52.6** | **38.8** |
| Slovak | Fine-tuning | 46.4 | 39.2 |
| | **Neutral Residues** | **51.5** | **38.6** |

**Key Findings**: Neutral Residues achieves the best English retention across all languages (second only to the backbone) while its performance in learning target languages is close to or surpasses that of fine-tuning, providing the best overall trade-off.

### Ablation Study: Gating and Local Loss

| Gating | Loss | English PPL | French PPL | English Tasks | French Tasks |
|:-:|:-:|:-:|:-:|:-:|:-:|
| None | $\ell_1$ | 0.687 | 0.801 | 45.3 | 42.4 |
| Sigmoid | CE | 0.677 | 0.800 | 45.2 | 42.6 |
| **ReLU** | **$\ell_1$** | **0.674** | **0.791** | **47.1** | **43.6** |
| Adapter baseline | — | 0.686 | 0.812 | 45.1 | 41.3 |

### Perplexity Comparison (EN-LM-1B, $p=0.1$)

| Method | English PPL ↓ | French PPL ↓ |
|:-:|:-:|:-:|
| Pre-trained Model | 0.663 | 1.175 |
| Fine-tuning | 0.811 | 0.758 |
| LoRA | 0.730 | 0.818 |
| Vanilla Adapter | 0.687 | 0.812 |
| **Neutral Residues** | **0.668** | **0.793** |

## Highlights & Insights

1. **Elegant "Neutral Residue" Concept**: The adapter outputs near-zero on the original distribution, causing the network behavior to fall back to the backbone, which naturally prevents forgetting.
2. **Three-Level Coordination (Data/Architecture/Training)**: Combining mixed training, gating, sparse loss, and low-variance initialization, with each component verified through clear ablation studies.
3. **ReLU > Sigmoid Gating**: Prevents singular value skewness in the gating matrix, proving more effective than explicit domain classifiers.
4. **Highly Practical**: Does not require the original training data and can operate using only an approximate distribution (e.g., Wikipedia).
5. **Conceptually Aligned with MoE but Simpler**: Only one extra FFN and gate are added per layer, avoiding the need for routing or load balancing.

## Limitations & Future Work

1. **Validated Only on Multilingual Scenarios**: Although the method is generalizable, it lacks validation on other domains such as code, mathematics, or multimodality.
2. **Limited Scale**: Evaluated at a maximum scale of Gemma-2B, without validation on 7B+ models.
3. **Inference Overhead**: Adding 20% parameters introduces permanent computational overhead during inference, unlike LoRA which can be merged.
4. **Interpretability of the Gate**: There is a lack of deep analysis regarding the activation distribution of the ReLU gate, and the boundary between "neutral" and "activated" states is not sufficiently clear.
5. **No Open-Source Code**: This restricts reproducibility.

## Rating

- Novelty: ⭐⭐⭐⭐ — The concept of neutral residues is elegant, and the combination of ReLU gate + $\ell_1$ sparse loss is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablation studies across multiple languages and benchmarks are provided, though validation on larger models and multiple domains is missing.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, thorough analysis, and well-designed figures/tables.
- Value: ⭐⭐⭐⭐ — Provides a practical and elegant approach for model extension, yielding valuable insights for the continual learning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Random Initialization of Gated Sparse Adapters (RIGSA)](random_initialization_of_gated_sparse_adapters.md)
- [\[ICCV 2025\] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](../../ICCV2025/model_compression/integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [\[ACL 2025\] Limited-Resource Adapters Are Regularizers, Not Linguists](../../ACL2025/model_compression/limited-resource_adapters_are_regularizers_not_linguists.md)
- [\[NeurIPS 2025\] Revisiting Semi-Supervised Learning in the Era of Foundation Models](../../NeurIPS2025/model_compression/revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)
- [\[ICLR 2026\] FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](../../ICLR2026/model_compression/freqkv_key-value_compression_in_frequency_domain_for_context_window_extension.md)

</div>

<!-- RELATED:END -->
