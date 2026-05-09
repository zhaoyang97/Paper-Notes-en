---
title: >-
  [Paper Note] Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning
description: >-
  [AAAI 2026][Continual Learning] This paper proposes **EDD (Expandable and Differentiable Dual Memory)**, an exemplar-free continual learning method that decomposes data into reusable sub-features via **differentiable shared and task-specific memories**, combined with **memory expansion-pruning** and **orthogonal regularization** mechanisms. EDD surpasses 14 state-of-the-art methods on CIFAR-10/100 and Tiny-ImageNet, achieving final accuracies of 55.13%, 37.24%, and 30.11%, respectively.
tags:
  - AAAI 2026
  - Continual Learning
  - Catastrophic Forgetting
  - Dual Memory System
  - Orthogonal Regularization
  - Exemplar-free Replay
date: 2026-05-08
content_hash: b4bf59268a782237
---

# Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning

**Conference**: AAAI 2026
**arXiv**: [2511.09871](https://arxiv.org/abs/2511.09871)
**Code**: [https://github.com/axtabio/EDD](https://github.com/axtabio/EDD)
**Area**: Other
**Keywords**: Continual Learning, Catastrophic Forgetting, Dual Memory System, Orthogonal Regularization, Exemplar-free Replay

## TL;DR

This paper proposes **EDD (Expandable and Differentiable Dual Memory)**, an exemplar-free continual learning method that decomposes data into reusable sub-features via **differentiable shared and task-specific memories**, combined with **memory expansion-pruning** and **orthogonal regularization** mechanisms. EDD surpasses 14 state-of-the-art methods on CIFAR-10/100 and Tiny-ImageNet, achieving final accuracies of 55.13%, 37.24%, and 30.11%, respectively.

## Background & Motivation

### Problem Setting

**Continual Learning (CL)** faces the core challenge of **Catastrophic Forgetting**: knowledge of previous tasks is lost when learning new ones. This is especially severe under the **exemplar-free** setting, where no old samples can be stored for replay.

### Fundamental Limitations of Prior Work

While existing methods partially mitigate forgetting, they share a commonly overlooked fundamental flaw — **they treat future tasks as entirely independent from the past, neglecting potentially useful inter-task relationships**:

**Regularization-based methods** (e.g., EWC, LwF): constrain parameter updates to protect old knowledge
- Drawback: limit model plasticity and fail to exploit shared knowledge across tasks

**Architecture expansion / parameter isolation methods** (e.g., PNN, DEN): allocate dedicated neurons per task and freeze them
- Drawback: uncontrolled model growth; frozen parameters cannot be reused by new tasks

**Shared limitation**: all methods protect old knowledge by either "penalizing new information" or "isolating old and new parameters," rather than actively **reusing old knowledge to facilitate learning of new data**.

### Core Motivation: From "Defensive Protection" to "Active Reuse"

The central idea of EDD is to **decompose past data features into small sub-features stored in memory, enabling new data to retrieve and reuse these existing sub-features** (Figure 1c). This represents a paradigm shift:
- Prior methods: past vs. future → isolation / penalization
- EDD: past + future → sharing / reuse

The inspiration comes from **Complementary Learning Systems (CLS) theory**: the human brain achieves continual learning through the collaboration of the hippocampus (rapid acquisition of new knowledge) and the neocortex (slow accumulation of shared knowledge).

## Method

### Overall Architecture

EDD inserts two complementary memories into intermediate layers of ResNet-18:
- **Shared Memory $M^s$**: encodes general representations reusable across tasks (analogous to the neocortex in CLS)
- **Task-specific Memory $M^t$**: builds upon shared knowledge to capture fine-grained discriminative features unique to each task (analogous to the hippocampus)

Both memories are **fully differentiable key-value memories**, jointly optimized end-to-end with the encoder and classifier.

### Key Designs

#### 1. **Differentiable Key-Value Memory**

- **Structure**: each memory $M^\ell$ contains $L_\ell$ learnable slots, each with a key $\mathbf{k}_j^\ell$ and a value $\mathbf{v}_j^\ell$
- **Read mechanism**: given intermediate feature $\mathbf{h}$, memory is read via cosine-similarity attention:
$$w_j = \frac{\exp(\langle \mathbf{k}_j^\ell, \mathbf{h} \rangle)}{\sum_{i=1}^{L_\ell} \exp(\langle \mathbf{k}_i^\ell, \mathbf{h} \rangle)}$$
$$\hat{\mathbf{h}} = \sum_{j=1}^{L_\ell} w_j \mathbf{v}_j^\ell$$
- **Design Motivation**:
    - Full differentiability allows the network to autonomously learn optimal latent representations for each sample
    - Key-value separation enables flexible knowledge organization and retrieval
    - Memory output $\hat{\mathbf{h}}$ directly replaces the original feature for forward propagation

#### 2. **Memory Expansion and Knowledge Pruning**

After training on each task, the memories undergo self-organization:

**Pruning (Freezing)**: identifies the most important slots for the current task and freezes them:
$$\Delta_j^\ell = |K_j^\ell - (K_j^\ell)^{(t-1)}|_2 + |V_j^\ell - (V_j^\ell)^{(t-1)}|_2$$
Slots exhibiting the largest changes are considered most critical and are frozen to preserve acquired knowledge.

**Number of frozen slots** is proportional to the class ratio:
$$|\mathcal{F}_t^\ell| \approx \frac{|\mathcal{C}_t|}{|\mathcal{C}_{1:t}|} \cdot \text{unfrozen slots}$$

**Expansion**: an equal number of new slots (randomly initialized) are added after freezing, ensuring sufficient learning capacity for subsequent tasks.

- **Design Motivation**:
    - Pruning ensures stability (old knowledge is not overwritten)
    - Expansion ensures plasticity (new knowledge has room to grow)
    - Growth rate is matched to class diversity, avoiding uncontrolled expansion

#### 3. **Orthogonal Regularization**

**Applied exclusively to task-specific memory $M^t$**, enforcing geometric orthogonality between frozen slots and new slots:

$$\mathcal{L}_{\text{orth}} = \|K_F^t (K_U^t)^\top\|_F^2 + \|V_F^t (V_U^t)^\top\|_F^2$$

This penalizes any alignment between frozen and active slots, encouraging new task-specific features to occupy subspaces **orthogonal to those of old features**.

- **Design Motivation**: although freezing prevents forgetting caused by direct parameter updates, **representational overlap** between old and new knowledge may still cause interference. Orthogonal constraints enforce subspace separation.
- **Applied only to task-specific memory**: the shared memory does not require orthogonal constraints, as cross-task sharing is its intended purpose.

#### 4. **Memory-Guided Representation Alignment**

The memory activation patterns of the previous model are used to guide the current model:

$$\mathcal{L}_{\text{align}} = \sum_{\ell \in \{s,t\}} \mathbb{E}_{x \sim \mathcal{T}_t} [1 - \cos(A_{\text{new}}^\ell(x), A_{\text{old}}^\ell(x))]$$

where $A$ denotes the attention weight vector of the memory. This ensures the current model retrieves similar memory slot combinations as the previous model for the same input.

- **Design Motivation**: rather than storing old samples, knowledge is transferred by enforcing consistency in memory activation patterns — a form of "exemplar-free distillation."

### Loss & Training

**Total loss**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}^{(t)} + \lambda_{\text{mem}} \mathcal{L}_{\text{align}} + \lambda_{\text{orth}} \mathcal{L}_{\text{orth}}$$

**Training procedure**:
1. First task: trained from scratch using $\mathcal{L}_{\text{CE}}$ only
2. Subsequent tasks:
    - Freeze the previous model; calibrate its BN layers to the new data distribution (batch adaptation)
    - Copy the previous model as initialization for the current model
    - Train on new task data using the total loss
    - Apply expansion-pruning after training
    - Current model becomes the previous model for the next task

**Hyperparameters**: $\mathcal{L}_\ell = 1000$ slots, $\lambda_{\text{mem}} = 20$, $\lambda_{\text{orth}} = 10$, pruning ratio = 0.15, Adam optimizer, lr = 0.001

## Key Experimental Results

### Dataset Settings
- **S-CIFAR-10**: 5 tasks × 2 classes
- **S-CIFAR-100**: 10 tasks × 10 classes and 20 tasks × 5 classes
- **S-Tiny-ImageNet**: 10 tasks × 20 classes and 20 tasks × 10 classes

### Main Results (Table 1, comparison with 14 methods)

| Method | S-CIFAR-10 | S-CIFAR-100 (10T) | S-CIFAR-100 (20T) | S-TinyImgNet (10T) | S-TinyImgNet (20T) |
|--------|-----------|-------------------|-------------------|--------------------|--------------------|
| JT (upper bound) | 83.38 | 70.44 | 70.44 | 59.99 | 59.99 |
| FT (lower bound) | 18.35 | 4.43 | 2.91 | 5.84 | 2.51 |
| PEC (Prev. SOTA) | 52.19 | 21.82 | 18.29 | 15.97 | 13.51 |
| DualNet (buffer) | 41.69 | 28.96 | 15.91 | 24.48 | 12.56 |
| LUCIR (buffer) | 42.78 | 30.57 | 19.99 | 25.84 | 14.51 |
| **EDD (Ours)** | **55.13** | **37.24** | **21.68** | **30.11** | **18.34** |

**Key comparisons**:
- vs. PEC (Prev. SOTA): CIFAR-10 +5.6%, TinyImageNet-20T **+35.8%**
- vs. DualNet (buffer=500): EDD outperforms comprehensively under **exemplar-free** conditions
- **As task complexity increases, EDD's advantage grows from 5.6% to 26.4%**

### Ablation Study (Table 2)

| Configuration | CIFAR-100 (10T) | TinyImageNet (10T) |
|---------------|-----------------|--------------------| 
| Naive $\mathcal{L}_{\text{CE}}$ | 32.47 ± 0.62 | 25.38 ± 0.54 |
| + $\mathcal{L}_{\text{align}}$ | 34.82 ± 0.58 (+2.35) | 27.12 ± 0.47 (+1.74) |
| + $\mathcal{L}_{\text{orth}}$ | 33.95 ± 0.71 (+1.48) | 26.55 ± 0.52 (+1.17) |
| + $\mathcal{L}_{\text{align}}$ + $\mathcal{L}_{\text{orth}}$ | 35.67 ± 0.49 (+3.20) | 28.03 ± 0.61 (+2.65) |
| + BA + $\mathcal{L}_{\text{align}}$ + $\mathcal{L}_{\text{orth}}$ | **37.24 ± 0.29** (+4.77) | **30.11 ± 0.22** (+4.73) |

**Key Findings**:
- Memory alignment and orthogonal regularization are complementary (combined > sum of individual contributions)
- Batch Adaptation contributes an additional +1.57% / +2.08% by alleviating distribution shift

### Feature Alignment Analysis with Joint Training

Across four metrics (cosine similarity ↑, KL divergence ↓, Wasserstein distance ↓, feature distance ↓), **EDD most closely approximates joint training on all metrics**.

**Per-class cosine similarity on CIFAR-10 (Figure 7)**:
- EDD maintains consistently high similarity across all classes and task steps
- Other methods (e.g., RPC, GSS) see similarity drop to near zero by the final task
- Notably, the fourth class learned in the second task achieves the **highest similarity score**, even after all subsequent tasks have been trained

### Surprising Finding: Future Tasks Enhance Past Knowledge

Re-evaluating T5 accuracy after training on T18 shows an **improvement** (Figure A.2). This counter-intuitive phenomenon — whereby subsequent tasks enhance the performance of earlier ones — stems from EDD's memory mechanism: high-importance memory slots frozen during T18 training are re-utilized by T5 test data. This strongly supports EDD's core claim of promoting cross-task feature sharing and reuse.

### Computational Complexity
- Memory: at most 2× the base model (a full copy of the previous model)
- Per-step computational cost is constant (does not grow with the number of tasks)
- Runtime on CIFAR-100 is near minimal; on TinyImageNet, less than 1 minute per epoch

## Highlights & Insights

1. **Paradigm shift**: from "protecting old knowledge from being overwritten" to "reusing old knowledge to facilitate new learning." This is not only a technical innovation but a fundamental redefinition of the continual learning problem.
2. **Elegant instantiation of CLS theory**: shared memory = neocortex (slow learning of general knowledge); task-specific memory = hippocampus (fast learning of unique features); both are end-to-end differentiable.
3. **Exemplar-free surpasses exemplar-based**: under a strict exemplar-free setting, EDD outperforms methods using buffer=500 (DualNet, LUCIR), challenging the intuition that replay is necessary.
4. **Discovery of "forward knowledge enhancement"**: learning future tasks can retrospectively improve performance on past tasks — a phenomenon rarely observed in conventional CL research.

## Limitations & Future Work

1. **Scalability to long task sequences**: beyond 50 tasks, the cumulative overhead of memory management may become a bottleneck.
2. **Extreme task shift**: when consecutive tasks share almost no structural similarity, the shared memory may struggle to capture useful common features.
3. **High-dimensional input / large output space**: scaling to very high-resolution images or large-scale classification may face memory footprint issues.
4. **ResNet-18 only**: the method has not been validated on other backbones such as Vision Transformers.
5. **Initial memory slot count**: $\mathcal{L}_\ell = 1000$ is fixed; more principled initialization strategies warrant further exploration.

## Related Work & Insights

- **DualNet (Pham et al. 2021)**: CLS-inspired fast-slow network, but relies on a buffer and self-supervised learning; EDD eliminates these dependencies via differentiable memories.
- **CLS-ER (Arani et al. 2022)**: short-term / long-term semantic memory + episodic buffer, but integrates knowledge indirectly through learning rate scheduling.
- **PNN (Rusu et al. 2016)**: one column of parameters per task with lateral connections, but growth is uncontrolled.
- **Insights**:
    - "Decompose–store–reuse" is a powerful paradigm; decomposing features into atomic sub-features stored in memory enables knowledge sharing at arbitrary granularity.
    - The selective application of orthogonal constraints to task-specific memory only (rather than globally) is a nuanced design choice.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The combination of differentiable dual memory, expansion-pruning, and orthogonal regularization is entirely novel and grounded in cognitive science theory
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 14 baseline comparisons, 3 datasets across 5 settings, thorough ablations, feature alignment analysis, and computational complexity analysis
- Writing Quality: ⭐⭐⭐⭐ — Method is described clearly and systematically, though some notation definitions are verbose
- Value: ⭐⭐⭐⭐⭐ — Substantially advances the state of the art under the strict exemplar-free setting, with significant implications for the CL community

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FEAT: Federated Geometry-Aware Correction for Exemplar Replay under Continual Dynamic Heterogeneity](../../CVPR2026/others/feat_federated_geometry_aware_correction_for_exemplar_replay_under_continual_dynamic_heterogeneity.md)
- [\[AAAI 2026\] DS-ATGO: Dual-Stage Synergistic Learning via Forward Adaptive Threshold and Backward Gradient Optimization for Spiking Neural Networks](ds-atgo_dual-stage_synergistic_learning_via_forward_adaptive_threshold_and_backw.md)
- [\[AAAI 2026\] Cost-Free Neutrality for the River Method](cost-free_neutrality_for_the_river_method.md)
- [\[NeurIPS 2025\] Exact Learning of Arithmetic with Differentiable Agents](../../NeurIPS2025/others/exact_learning_of_arithmetic_with_differentiable_agents.md)
- [\[AAAI 2026\] Forget Less by Learning from Parents Through Hierarchical Relationships](forget_less_by_learning_from_parents_through_hierarchical_relationships.md)

<!-- RELATED:END -->
