---
title: >-
  [Paper Note] Is the Information Bottleneck Robust Enough? Towards Label-Noise Resistant Information Bottleneck Learning
description: >-
  [AAAI 2026][Knowledge Editing][Information Bottleneck] This paper identifies the inherent vulnerability of the Information Bottleneck (IB) principle under label noise and proposes LaT-IB…
tags:
  - "AAAI 2026"
  - "Knowledge Editing"
  - "Information Bottleneck"
  - "Label Noise"
  - "Representation Learning"
  - "Robustness"
  - "Mutual Information"
date: 2026-05-08
content_hash: 0f84829c5bef7ad6
---

# Is the Information Bottleneck Robust Enough? Towards Label-Noise Resistant Information Bottleneck Learning

**Conference**: AAAI 2026
**arXiv**: [2512.10573](https://arxiv.org/abs/2512.10573)
**Code**: [https://github.com/RingBDStack/LaT-IB](https://github.com/RingBDStack/LaT-IB)
**Area**: Knowledge Editing
**Keywords**: Information Bottleneck, Label Noise, Representation Learning, Robustness, Mutual Information

## TL;DR

This paper identifies the inherent vulnerability of the Information Bottleneck (IB) principle under label noise and proposes LaT-IB, which decomposes representations into a clean-label subspace and a noisy-label subspace. Combined with a Minimal-Sufficient-Clean (MSC) criterion and a three-stage training framework, LaT-IB significantly outperforms existing IB methods across diverse noise conditions.

## Background & Motivation

**Limitations of the Information Bottleneck**: The IB principle retains task-relevant information by maximizing $I(Y;Z)$ while compressing redundancy by minimizing $I(X;Z)$. However, its strong reliance on accurate labels makes it highly vulnerable to label noise — when $Y$ is corrupted, maximizing $I(Y;Z)$ causes the representation $Z$ to capture the noise $Y_n$.

**Empirical Evidence of Vulnerability**: On CIFAR-10, VIB's accuracy collapses to 10.0% under 50% symmetric noise; on the Cora graph, GIB degrades continuously from 69.5% to 55.1% under 40% noise, demonstrating that existing IB methods cannot effectively handle label noise.

**Suboptimality of Two-Stage Approaches**: Cascading a denoising step with IB introduces theoretically guaranteed cumulative degradation — the authors prove that the error lower bound of a cascaded model is higher than that of an end-to-end model (Theorem 1.1), as elongated information pathways incur unavoidable information loss.

**Ubiquity of Label Noise**: Label noise is pervasive in real-world scenarios, affecting both image annotation and node labeling in graph data, and significantly degrades model performance.

**Absence of Representation-Level Constraints**: Most existing label-noise learning methods (sample selection, robust losses, data augmentation, etc.) neglect representation-level constraints, making it difficult to learn task-relevant, noise-invariant features under severe noise or distribution shift.

**Need for End-to-End Solutions**: A unified method is required that performs denoising and feature extraction simultaneously within the IB framework, rather than relying on external denoising modules — necessitating a redesign of both the IB objective and the optimization strategy.

## Method

### Core Idea: Minimal-Sufficient-Clean (MSC) Criterion

LaT-IB decomposes the latent representation into two components: $S$ (aligned with the clean label space) and $T$ (aligned with the noisy label space). The objective is:

$$\min \underbrace{-I(Y;S,T)}_{\text{Sufficiency}} + \beta \underbrace{I(\mathcal{D};S,T)}_{\text{Minimality}} + \gamma \underbrace{I(S;T|Y)}_{\text{Cleanness}}$$

subject to: $\max(I(Y_n;S), I(Y_c;T)) \leq K$

### Theoretical Foundations

- **Lemma 4.1 (Redundancy Invariance)**: Optimizing the prediction and compression terms reduces the model's reliance on features $\mathcal{D}_n$ that are irrelevant to $Y$.
- **Lemma 4.2 (Feature Convergence)**: When $I(Y_n;S)$ and $I(Y_c;T)$ are sufficiently small, optimizing the disentanglement term strengthens the mappings $S \to Y_c$ and $T \to Y_n$.
- Via upper- and lower-bound analysis (Propositions 4.1–4.4), intractable multi-variable mutual information terms are converted into implementable loss functions.

### Dual-Encoder Architecture

A dual-encoder, single-decoder architecture is adopted: two encoders map the input to high-dimensional Gaussian distributions, from which $S$ and $T$ are sampled via the reparameterization trick, and a shared decoder performs prediction.

### Three-Stage Training Framework

**Stage 1: Warmup** — $\text{encoder}_S$ is pretrained on the full dataset using standard cross-entropy loss $\mathcal{L}_{CE}(\hat{y}_S, y)$ to establish basic discriminative capability.

**Stage 2: Knowledge Injection** — An InfoJS selector (based on mutual information $I(S;Y)$ and JS divergence $D_{JS}(S \| T)$) partitions samples into clean, noisy, and uncertain sets. For clean and noisy samples, inter-encoder divergence is maximized to promote differentiation; for uncertain samples, divergence is minimized to guide learning. A minimal representation regularizer is also incorporated.

**Stage 3: Robust Training** — The full objective is optimized. A ConCE loss (a smooth approximation of $\sum \min(\mathcal{L}_{CE}(\hat{y}_S, y), \mathcal{L}_{CE}(\hat{y}_T, y))$) promotes encoder consistency, and a discriminator is alternately trained to minimize the conditional mutual information $I(S;T|Y)$.

## Key Experimental Results

### Experiment 1: Image Classification under Real-World Label Noise (CIFAR-10N/100N)

| Method | CIFAR-10N aggre | CIFAR-10N worst | CIFAR-100N noisy100 | Animal-10N |
|------|:-:|:-:|:-:|:-:|
| VIB | 86.11 | 73.80 | 53.29 | 76.28 |
| (ELR+)+VIB | 92.65 | 86.68 | 61.06 | 85.87 |
| Promix+VIB | 92.35 | **91.24** | **63.91** | 85.47 |
| **LaT-IB** | **94.17** | 87.95 | 63.59 | **88.49** |

LaT-IB achieves 94.17% on CIFAR-10N aggre, substantially outperforming all baselines, and improves Animal-10N accuracy by 2.6 percentage points.

### Experiment 2: Robustness under Adversarial Attack (CIFAR-10N + FGSM)

| Method | aggre (no attack) | aggre (ε=0.1) | worst (no attack) | worst (ε=0.1) |
|------|:-:|:-:|:-:|:-:|
| VIB | 86.11 | 43.18 | 73.80 | 36.56 |
| Promix+VIB | 92.35 | 36.43 | 91.24 | 36.05 |
| **LaT-IB** | **94.17** | **60.66** | **87.95** | **54.18** |

Two-stage methods suffer catastrophic degradation under adversarial attack (Promix+VIB: 92.35→36.43), whereas LaT-IB demonstrates strong robustness (94.17→60.66), maintaining a lead of approximately 14–18 percentage points at ε=0.1.

### Graph Node Classification (Pubmed, 40% Uniform Noise)

LaT-IB achieves 73.40%, substantially outperforming GIB (64.30%) and various improved baselines, demonstrating strong cross-domain generalization.

## Highlights & Insights

- **First systematic theoretical and empirical exposure of IB's vulnerability to label noise**, along with a formal proof of the suboptimality of two-stage approaches.
- **The MSC criterion unifies denoising and representation learning**, achieving noise separation within the IB framework without external denoising modules.
- **The three-stage progressive training strategy is elegantly designed**, guiding representation disentanglement in a structured warmup → injection → robust training progression.
- **Exceptional performance under the compound scenario of adversarial attack and label noise**, demonstrating the natural robustness conferred by the minimal-sufficient property.
- **Effective across tasks and domains**, applicable to both image classification and graph node classification.

## Limitations & Future Work

- **Complex training pipeline**: The three-stage training requires tuning multiple stage-specific hyperparameters ($E_{Warmup}$, $E_{Injection}$, $\beta$, $\gamma$, $\delta$), incurring substantial tuning cost.
- **InfoJS selector is threshold-sensitive**: Sample partition quality is highly sensitive to $\delta$ — too small a value yields insufficient training data, while too large a value causes the two encoders to converge.
- **Limited noise assumptions**: Validation is primarily conducted on symmetric, asymmetric, and instance-dependent noise; more complex noise patterns such as open-set noise are not addressed.
- **Small-scale graph benchmarks**: Graph experiments are conducted only on small graphs (Cora, Citeseer, Pubmed); scalability to large-scale graphs remains to be examined.
- **Practical satisfaction of theoretical assumptions**: Lemma 4.2 requires $\max(I(Y_n;S), I(Y_c;T))$ to be sufficiently small, yet the degree to which this condition is satisfied in practice is difficult to measure.

## Related Work & Insights

| Dimension | LaT-IB | VIB / GIB |
|------|--------|---------|
| Noise Handling | Built-in noise-aware disentanglement mechanism | No noise handling; directly maximizes $I(Y;Z)$ |
| Representation Structure | Dual encoders separating clean / noisy components | Single unified representation space |
| Adversarial Robustness | Minimal sufficiency + noise separation | Minimal sufficiency only, no noise separation |

| Dimension | LaT-IB | Denoise + IB (e.g., Promix+VIB) |
|------|--------|------|
| Architecture | End-to-end unified framework | Cascaded two-stage pipeline |
| Theoretical Guarantee | Cumulative degradation theorem proves superiority over cascading | Information transfer loss present |
| Adversarial Scenario | Single attack surface | Both stages are attackable, doubling vulnerability |

## Rating

- ⭐⭐⭐⭐ **Novelty**: First work to theoretically and empirically expose IB's vulnerability to label noise; the MSC criterion is elegantly designed.
- ⭐⭐⭐⭐ **Technical Depth**: Rigorous theoretical derivations (multiple propositions and lemmas); upper/lower-bound analysis integrates well with the progressive training framework.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Covers image and graph domains, real-world and synthetic noise, and adversarial attack settings.
- ⭐⭐⭐ **Practicality**: The three-stage training pipeline is relatively complex with numerous hyperparameters, requiring considerable tuning effort for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rote Learning Considered Useful: Generalizing over Memorized Training Examples](../../ICLR2026/knowledge_editing/rote_learning_considered_useful_generalizing_over_memorized_training_examples.md)
- [\[ICLR 2026\] EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](../../ICLR2026/knowledge_editing/eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)
- [\[ACL 2026\] EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing](../../ACL2026/knowledge_editing/evoedit_evolving_null-space_alignment_for_robust_and_efficient_knowledge_editing.md)
- [\[AAAI 2026\] Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)](multiplicative_orthogonal_sequential_editing_for_language_models.md)
- [\[AAAI 2026\] Catastrophic Forgetting in Kolmogorov-Arnold Networks](catastrophic_forgetting_in_kolmogorov-arnold_networks.md)

</div>

<!-- RELATED:END -->
