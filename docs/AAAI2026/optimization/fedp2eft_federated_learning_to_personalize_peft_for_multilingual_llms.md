---
title: >-
  [Paper Note] FedP²EFT: Federated Learning to Personalize PEFT for Multilingual LLMs
description: >-
  [AAAI2026][Optimization][Federated Learning] This paper proposes FedP²EFT, which collaboratively trains a Personalization Strategy Generator (PSG) via federated learning to automatically generate personalized LoRA rank s…
tags:
  - "AAAI2026"
  - "Optimization"
  - "Federated Learning"
  - "Personalized PEFT"
  - "LoRA Rank Selection"
  - "Multilingual LLM"
  - "Bayesian Sparse Selection"
date: 2026-05-08
content_hash: 377ec5fcbb65c76f
---

# FedP²EFT: Federated Learning to Personalize PEFT for Multilingual LLMs

**Conference**: AAAI2026  
**arXiv**: [2502.04387](https://arxiv.org/abs/2502.04387)  
**Authors**: Royson Lee, Minyoung Kim, Fady Rezk, Rui Li, Stylianos I. Venieris, Timothy Hospedales (Samsung AI / Univ. of Edinburgh)  
**Code**: [GitHub](https://github.com/SamsungLabs/fedp2eft)  
**Area**: Optimization  
**Keywords**: Federated Learning, Personalized PEFT, LoRA Rank Selection, Multilingual LLM, Bayesian Sparse Selection  

## TL;DR

This paper proposes FedP²EFT, which collaboratively trains a Personalization Strategy Generator (PSG) via federated learning to automatically generate personalized LoRA rank structures for each client, substantially outperforming manually designed PEFT configurations and existing FL personalization methods in multilingual LLM fine-tuning.

## Background & Motivation

### The Federated Learning Dilemma for Multilingual LLMs
Federated learning enables multilingual LLMs to leverage low-resource language data distributed across different regions while complying with privacy regulations such as GDPR. However, existing methods face three major challenges:

1. **Curse of Multilinguality**: As the number of languages increases, the performance of a single global model degrades.
2. **Negative Interference**: Different languages compete for limited model capacity.
3. **Lack of Personalization Strategy**: Existing methods employ manually designed, uniform PEFT configurations that overlook the heterogeneous personalization needs of different clients.

### Why Is Personalizing LoRA Rank More Important Than Learning Rate?
Existing FL hyperparameter optimization methods (e.g., FedL2P) primarily learn personalized learning rates; however, LLMs typically use adaptive optimizers such as Adam and are thus robust to learning rate variations. In contrast, the structure of PEFT adapters—which layers to apply LoRA to and what rank to use—has a far more critical impact on cross-lingual transfer learning.

## Core Problem

How to automatically learn the optimal personalized LoRA rank configuration for each client in a federated learning setting, while avoiding overfitting caused by limited per-client data?

## Method

### BayesTune-LoRA (BT-LoRA)

Inspired by BayesTune, rank-wise latent variables $\lambda \in \mathbb{R}^r, \lambda_i > 0$ are introduced for each LoRA matrix, modifying LoRA to $B\lambda A$. The optimization objective is:

$$\theta^* = \arg\min_\theta \mathcal{L}_{\text{CE}}(\theta; D) + \frac{\alpha_s}{N}\mathcal{L}_s(\boldsymbol{\lambda}, \boldsymbol{B}) + \frac{\alpha_p}{N}\mathcal{L}_p(\boldsymbol{\lambda})$$

where $\mathcal{L}_s$ is the log of a Laplace prior (encouraging larger $\lambda$ for important ranks):

$$\mathcal{L}_s(\boldsymbol{\lambda}, \boldsymbol{B}) = \sum_l^L \sum_i^r \frac{\|B_{l,i}\|_1}{\lambda_{l,i}}$$

and $\mathcal{L}_p$ is the log of a Gamma hyperprior (encouraging overall small $\lambda$ for sparsity):

$$\mathcal{L}_p(\boldsymbol{\lambda}) = \sum_l^L \sum_i^r (\log \lambda_{l,i} + 100 \cdot \lambda_{l,i})$$

Intuitively, $\mathcal{L}_p$ drives $\lambda$ toward zero (sparsification), while $\mathcal{L}_s$ maintains large $\lambda$ for ranks with substantial updates. Their interplay retains important ranks and prunes unimportant ones.

### PSG: Personalization Strategy Generator

A single-hidden-layer MLP serves as the PSG. It takes client metadata (mean and standard deviation of each base model layer's features) as input and outputs personalized $\hat{\boldsymbol{\lambda}}$:

$$\hat{\boldsymbol{\lambda}} = \text{MLP}(\phi;\; E(h_0), SD(h_0), E(h_1), SD(h_1), \ldots, E(h_{L-1}), SD(h_{L-1}))$$

### Federated Training Procedure

In each federated round, each sampled client $i$ performs:

1. Receives PSG parameters $\phi$ from the server and computes $\hat{\boldsymbol{\lambda}}^i$ via forward pass.
2. **Stage 1**: Inserts $\hat{\boldsymbol{\lambda}}^i$ into BT-LoRA and fine-tunes for $s$ steps according to the above objective, yielding optimized $\hat{\boldsymbol{\lambda}}^{i,s}$.
3. **Stage 2**: Uses $\hat{\boldsymbol{\lambda}}^{i,s}$ as regression targets and trains the MLP with an L1 loss.
4. Sends updated $\phi$ back to the server for FedAvg aggregation.

### Inference

At deployment, new clients (including those unseen during training) use the PSG to generate $\boldsymbol{\lambda}$, select the top-$(r \cdot L)$ largest ranks according to the resource budget $r \cdot L$, freeze $\boldsymbol{\lambda}$, and perform standard fine-tuning. A single PSG training run supports all rank budgets $\leq r_{\text{max target}}$.

## Key Experimental Results

### MasakhaNEWS Text Classification (16 African languages, Seen clients, $r=2$)

| Language | LoRA | AdaLoRA | BT-LoRA | FedL2P | **FedP²EFT** |
|----------|------|---------|---------|--------|-------------|
| eng | 90.4 | 89.9 | 89.9 | 90.7 | **92.0** |
| amh | 45.7 | 45.2 | 45.2 | 45.7 | **52.0** |
| tir | 44.9 | 44.9 | 44.9 | 45.3 | **63.5** |
| orm | 64.2 | 64.0 | 64.0 | 64.4 | **72.2** |
| fra | 88.6 | 88.6 | 88.6 | 89.1 | **93.5** |

FedP²EFT's advantage is especially pronounced on low-resource languages (tir, amh, orm), e.g., Tigrinya improves from 44.9% to 63.5% (+18.6 pp).

### Unseen Client Generalization

| Language | LoRA | FedL2P | **FedP²EFT** |
|----------|------|--------|-------------|
| xho | 64.2 | 64.4 | **78.5** |
| tir | 41.9 | 41.9 | **58.3** |
| orm | 62.0 | 62.2 | **73.0** |
| run | 82.0 | 82.6 | **88.4** |

FedP²EFT also achieves substantial improvements on clients entirely absent from training, validating the generalization capability of the PSG.

### XNLI + FedDPA-T Compatibility with Personalized FL

| Language | LoRA | FedL2P | **FedP²EFT** |
|----------|------|--------|-------------|
| ur | 41.9 | 44.8 | **63.7** |
| bg | 45.8 | 47.5 | **64.4** |
| hi | 42.8 | 44.5 | **57.8** |

FedP²EFT integrates seamlessly with existing personalized FL methods (FedDPA-T, DEPT, etc.), further enhancing personalization performance.

## Highlights & Insights

- **First federated LoRA rank personalization**: Frames PEFT structure selection as a federated learning problem, avoiding overfitting from independent per-client training.
- **Single training run covers all rank budgets**: The sparse selection property of BT-LoRA makes PSG training a one-time cost.
- **Broad compatibility**: Can be plugged into standard FL, FedDPA-T, DEPT, and other FL frameworks.
- **Large gains on low-resource languages**: Improvements of up to 18.6 pp on extremely low-resource languages such as Tigrinya and Amharic.
- **Clear theoretical grounding**: The sparse prior over LoRA ranks is derived from Bayesian sparse model selection.

## Limitations & Future Work

- **PSG relies solely on summary statistics**: Means and standard deviations may discard distributional details; richer metadata extraction could yield further gains.
- **Only LoRA is evaluated**: Applicability to other PEFT methods (Adapter, Prefix Tuning, IA³) remains unexplored.
- **FedAvg aggregation only**: No comparison with more advanced aggregation strategies such as FedProx or SCAFFOLD.
- **Sensitivity to Stage 1 step count $s$**: Too large risks overfitting to client data; too small yields low-quality $\hat{\boldsymbol{\lambda}}^{i,s}$; no adaptive scheduling is investigated.
- **Limited experimental scale**: Instruction tuning is validated only on MobileLLaMA-1.4B and Llama-3.2-3B.

## Related Work & Insights

- **FedL2P**: Learns personalized learning rates via second-order optimization, which is of limited utility under LLM + Adam settings; FedP²EFT targets rank structure directly and avoids second-order computation.
- **AdaLoRA**: SVD-based rank allocation is prone to overfitting under the data-scarce FL regime; FedP²EFT mitigates data insufficiency through federated collaboration.
- **BT-LoRA (standalone)**: Independent per-client optimization of $\boldsymbol{\lambda}$ leads to overfitting; FedP²EFT trains the PSG federally before generating $\boldsymbol{\lambda}$, yielding better generalization.
- **DEPT / FedDPA-T**: Manually designate personalized layers (embeddings/LoRA) without automatic adaptation to client needs; FedP²EFT can serve as a complementary module on top of these methods.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of Bayesian sparse rank selection and federated meta-learning is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers text classification and instruction tuning, seen/unseen clients, and multiple FL backbones, though model scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated, method derivation is complete, and figures are intuitive.
- Value: ⭐⭐⭐⭐ — Addresses a practical pain point in federated LLM personalization, with significant value for low-resource language scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DeepAFL: Deep Analytic Federated Learning](../../ICLR2026/optimization/deepafl_deep_analytic_federated_learning.md)
- [\[AAAI 2026\] FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters](fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)
- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[AAAI 2026\] pFed1BS: Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching](personalized_federated_learning_with_bidirectional_communication_compression_via.md)
- [\[AAAI 2026\] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)

</div>

<!-- RELATED:END -->
