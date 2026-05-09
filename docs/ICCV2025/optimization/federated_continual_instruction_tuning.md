---
title: >-
  [Paper Note] Federated Continual Instruction Tuning
description: >-
  [Optimization] This paper introduces the first Federated Continual Instruction Tuning (FCIT) benchmark, covering 2 scenarios, 4 settings, and 12 datasets, and proposes the DISCO framework, which addresses data heterogeneity and catastrophic forgetting via Dynamic Knowledge Organization (DKO) and Subspace Selective Activation (SSA).
tags:
  - Optimization
date: 2026-05-08
content_hash: 8d2a576ad6bcf0ec
---

# Federated Continual Instruction Tuning

- **Conference**: ICCV 2025
- **arXiv**: [2503.12897](https://arxiv.org/abs/2503.12897)
- **Area**: Optimization
- **Keywords**: Federated Learning, Continual Learning, Instruction Tuning, LoRA, LMM, Knowledge Organization, Catastrophic Forgetting

## TL;DR

This paper introduces the first Federated Continual Instruction Tuning (FCIT) benchmark, covering 2 scenarios, 4 settings, and 12 datasets, and proposes the DISCO framework, which addresses data heterogeneity and catastrophic forgetting via Dynamic Knowledge Organization (DKO) and Subspace Selective Activation (SSA).

## Background & Motivation

The exceptional performance of Large Multimodal Models (LMMs) relies heavily on large-scale instruction tuning data, yet centralized training faces two practical obstacles:

**Data Privacy and Distribution**: Data from different institutions (e.g., hospitals) cannot be centralized, necessitating federated learning for collaborative training.

**Dynamic Knowledge Updates**: In real-world scenarios, new tasks continuously emerge (e.g., new virus strains), requiring models to learn new knowledge while retaining prior knowledge.

Limitations of existing approaches:
- **Federated learning methods** mostly assume a fixed task set and cannot handle dynamically growing tasks.
- **Continual learning methods** are unable to share knowledge across clients.
- **Federated continual learning methods** target conventional image classification tasks and cannot accommodate the complex instruction tuning scenarios of LMMs.

Taking medical emergencies as an example: large hospitals collaboratively train on case data via federated learning to build a knowledge base, while small clinics update local cases with global knowledge; over time, continual learning is needed to update the knowledge base, and the emergence of new virus strains requires simultaneous knowledge integration. This represents a genuine need for the organic combination of federated learning and continual learning.

## Method

### Overall Architecture

The DISCO (**D**ynamic knowledge organ**I**zation and **S**ubspace sele**C**tive activati**O**n) framework comprises two core components:

1. **DKO (training phase)**: Maintains a dynamic cache on the global server and organizes knowledge from different tasks into corresponding subspaces via an identity token matching mechanism.
2. **SSA (inference phase)**: Selectively activates relevant subspaces based on test input features while suppressing irrelevant outputs.

### Key Designs

**1. Dynamic Knowledge Organization (DKO)**

FCIT faces two types of conflicts:
- **Intra-stage conflict**: Parameter space conflicts arising from different clients learning different tasks within the same stage.
- **Cross-stage conflict**: Catastrophic forgetting caused by new tasks modifying the parameter space of prior tasks.

The proposed solution decomposes LoRA updates into task-specific subspaces:

$$\Delta\mathbf{W} = \mathbf{B}\mathbf{A} \Leftrightarrow \{\mathbf{B}_1\mathbf{A}_1, \cdots, \mathbf{B}_T\mathbf{A}_T\}$$

**Identity Token Mechanism**:
- Each client uses a frozen CLIP text encoder to extract the mean feature of training instruction data as its local identity token $\mu_k^t$.
- Upon uploading to the server, cosine similarity (threshold $\tau=0.9$) is used to match against global identity tokens.
- On a match: the corresponding global token and subspace parameters are updated via sample-weighted aggregation.
- On a mismatch: a new global token and subspace are initialized.

Key advantage: Text features (rather than image features) are used to distinguish tasks, since different visual instruction tuning datasets tend to exhibit higher similarity at the image level than at the text level (e.g., CLEVR-Math and super-CLEVR share similar images but differ in instructions).

**2. Subspace Selective Activation (SSA)**

Directly concatenating all subspaces at inference introduces irrelevant information (e.g., a subspace trained for long descriptions may interfere with tasks requiring concise answers). SSA dynamically controls each subspace's output via activation factors:

$$\Delta\mathbf{W} = \bar{\mathbf{B}} \begin{bmatrix} \alpha_1 \cdot \mathbf{I} & & \\ & \ddots & \\ & & \alpha_T \cdot \mathbf{I} \end{bmatrix} \bar{\mathbf{A}}$$

Activation factor computation:
1. Compute the cosine similarity $s_i$ between the test input instruction feature and each global identity token.
2. Normalize via softmax with temperature $\varepsilon=0.05$ to obtain $\alpha_i$:

$$\alpha_i = \frac{\exp(s_i/\varepsilon)}{\sum_{j=1}^T \exp(s_j/\varepsilon)}$$

This ensures that subspaces relevant to the current input are amplified while irrelevant subspaces are suppressed.

### Loss & Training

Standard autoregressive cross-entropy loss for LMM instruction tuning (Eq. 2), with LoRA for parameter-efficient fine-tuning to reduce communication overhead.

### FCIT Benchmark Design

**Two scenarios**:
- **Homogeneous FCIT (Hom-FCIT)**: All clients learn the same task within each stage.
- **Heterogeneous FCIT (Het-FCIT)**: Different clients may learn different tasks within the same stage.

**Two settings**:
- **Capability-related (4 stages)**: 12 datasets grouped into four capability categories: General / Math / Chart / Other.
- **Task-related (8 stages)**: 8 datasets each treated as an independent stage.

**Data heterogeneity**: A Dirichlet distribution with $\beta \in \{0.5, 1.0, 5.0\}$ controls the non-IID degree of client data.

## Key Experimental Results

### Main Results (Hom-FCIT, Task-related, β=1.0)

| Method | Last | Avg |
|--------|------|-----|
| Zero-shot | 29.08 | - |
| Centralized MTL (upper bound) | 66.60 | - |
| Finetune | 47.20 | 68.79 |
| EWC | 47.92 | 69.22 |
| O-LoRA | 49.87 | 70.26 |
| M-LoRA | 48.53 | 71.58 |
| MoELoRA | 49.02 | 70.65 |
| **DISCO** | **56.22** | **73.03** |

DISCO surpasses the best baseline O-LoRA by approximately 6.4 percentage points on the Last metric.

### Main Results (Het-FCIT, Task-related, β=1.0)

| Method | Last | Avg |
|--------|------|-----|
| Finetune | 57.96 | 54.22 |
| O-LoRA | 59.74 | 55.20 |
| MoELoRA | 59.14 | 54.69 |
| **DISCO** | **63.25** | **61.99** |

In Het-FCIT, DISCO outperforms competitors by approximately 6.8 percentage points on Avg, demonstrating an even more pronounced advantage.

### Ablation Study

**Identity token extraction strategy** (Task-related, β=1.0):

| Strategy | Hom-FCIT Last | Het-FCIT Last |
|----------|---------------|---------------|
| Text features | 56.22 | 63.25 |
| Image features | 55.63 (−0.59) | 63.00 (−0.25) |
| Text + Image | 55.96 (−0.26) | 63.02 (−0.23) |

Pure text features yield the best performance, since visual instruction tuning tasks exhibit greater similarity at the image level than at the text level.

**SSA activation factor computation**:

| Strategy | Hom-FCIT Last | Het-FCIT Last |
|----------|---------------|---------------|
| Softmax (Ours) | 56.22 | 63.25 |
| Direct concatenation | 51.74 (−4.48) | 60.36 (−2.89) |
| Cosine similarity | 52.83 (−3.39) | 60.92 (−2.33) |
| Argmax | 55.74 (−0.48) | 62.88 (−0.37) |

Softmax normalization achieves the best results; direct concatenation introduces excessive irrelevant information, causing a sharp performance drop.

**Compatibility verification**: DISCO is compatible with federated optimization algorithms including FedAvgM, FedAdam, FedAdagrad, and FedYogi, with FedAvg achieving the best performance as the default.

### Key Findings

- **Zero-shot transfer**: DISCO's zero-shot performance on unseen tasks (33.08) significantly exceeds that of O-LoRA (28.27), indicating that knowledge organization avoids negative transfer.
- **General benchmark retention**: MME 1436.6 vs. original 1476.9, POPE 83.9 vs. 86.4, with minimal performance degradation.
- **Activation factor visualization**: During inference, only the activator corresponding to the current task responds; all others are effectively suppressed.

## Highlights & Insights

1. **First LMM federated continual learning benchmark**: 2 scenarios × 2 settings × 3 heterogeneity levels = a comprehensive evaluation matrix that fills the gap in this cross-disciplinary area.
2. **Elegant identity token design**: Task fingerprints are derived from the mean of text features using a frozen encoder, incurring zero additional training overhead.
3. **Training-free inference enhancement via SSA**: The intrinsic mixing matrix of LoRA is redefined as a dynamic activation matrix without introducing additional parameters.
4. **Automatic subspace discovery via threshold control**: With $\tau=0.9$, 8 subspaces are automatically formed in the 8-task setting without requiring the number of tasks to be predefined.
5. **Realistic scenario modeling**: Het-FCIT faithfully reflects the real-world demand in domains such as healthcare, where different institutions handle different tasks simultaneously.

## Limitations & Future Work

1. Validation is limited to LLaVA-1.5-7B; larger-scale models (e.g., 13B, 72B) have not been tested.
2. The number of communication rounds is fixed at 10; the effect of varying communication budgets is not investigated.
3. Identity tokens rely on the CLIP text encoder and may struggle to distinguish tasks with highly similar instruction styles.
4. LoRA is embedded only in FFN layers, without fully exploiting attention layers.
5. The selection of 12 datasets may not be sufficiently comprehensive, particularly lacking generative tasks.

## Related Work & Insights

- **Federated learning**: FedAvg, FLoRA, etc., all assume a fixed task set.
- **Continual learning**: EWC (elastic weight consolidation), L2P (learning to prompt), O-LoRA (orthogonal subspace constraint) — all lack cross-client knowledge sharing.
- **Federated continual learning**: MFCL, PILoRA, etc. focus on image classification; AFCL supports asynchronous multi-task learning but remains limited to classification.
- **LMM continual learning**: Continual LLaVA, COIN, etc. consider only centralized training.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First LMM-FL-CL cross-domain benchmark + novel and complete dual-component DKO/SSA design.
- **Practicality**: ⭐⭐⭐⭐ — Targets realistic distributed training scenarios; code and data are publicly available.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Full matrix experiments across 4 settings × 3 heterogeneity levels, with comprehensive ablations and clear visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous problem formulation, realistic scenario modeling, and systematic method description.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)
- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)
- [\[NeurIPS 2025\] PROFIT: A Specialized Optimizer for Deep Fine Tuning](../../NeurIPS2025/optimization/profit_a_specialized_optimizer_for_deep_fine_tuning.md)
- [\[ICCV 2025\] Class-Wise Federated Averaging for Efficient Personalization](class-wise_federated_averaging_for_efficient_personalization.md)
- [\[ICCV 2025\] Cooperative Pseudo Labeling for Unsupervised Federated Classification](cooperative_pseudo_labeling_for_unsupervised_federated_classification.md)

</div>

<!-- RELATED:END -->
