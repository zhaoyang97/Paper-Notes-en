---
title: >-
  [Paper Note] CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds
description: >-
  [AAAI 2026 (Neuro for AI & AI for Neuro Workshop, PMLR)][LLM Safety][Spiking Neural Networks] This paper proposes CATFormer, a data-replay-free continual learning framework built upon a spiking Vision Transformer, which achieves task-specific neuronal excitability modulation via context-adaptive dynamic firing thresholds. Over sequences of up to 100 tasks, the model not only avoids forgetting but actually improves in accuracy — a phenomenon the authors term "reverse forgetting."
tags:
  - "AAAI 2026 (Neuro for AI & AI for Neuro Workshop, PMLR)"
  - LLM Safety
  - Spiking Neural Networks
  - Continual Learning
  - Dynamic Thresholds
  - Class-Incremental Learning
  - Vision Transformer
date: 2026-05-08
content_hash: 19d11d34d0cdbf04
---

# CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds

**Conference**: AAAI 2026 (Neuro for AI & AI for Neuro Workshop, PMLR)
**arXiv**: [2603.15184](https://arxiv.org/abs/2603.15184)
**Code**: N/A
**Area**: LLM Safety
**Keywords**: Spiking Neural Networks, Continual Learning, Dynamic Thresholds, Class-Incremental Learning, Vision Transformer

## TL;DR

This paper proposes CATFormer, a data-replay-free continual learning framework built upon a spiking Vision Transformer, which achieves task-specific neuronal excitability modulation via context-adaptive dynamic firing thresholds. Over sequences of up to 100 tasks, the model not only avoids forgetting but actually improves in accuracy — a phenomenon the authors term "reverse forgetting."

## Background & Motivation

### Problem Definition

Deep neural networks suffer from **catastrophic forgetting** in real-world continual deployment: models lose knowledge of prior tasks when learning new ones. This is especially severe on resource-constrained edge devices (e.g., robots, autonomous systems), where storing past data for rehearsal is infeasible due to energy, privacy, and storage limitations.

### Limitations of Prior Work

**Regularization methods** (EWC, SI): Prevent forgetting by constraining updates to important parameters, but perform poorly on SNNs (SI achieves only 1.84% on CIFAR-100 with 50 tasks).

**Replay methods** (iCaRL, DER++): Require storing past data, violating privacy constraints and infeasible on hardware-limited platforms.

**Architecture methods** (DSD-SNN): Performance continuously degrades as task count grows, dropping from 60.47% at 10 tasks to 50.55% at 50 tasks.

**SNN continual learning** research has focused primarily on CNN architectures; spiking Vision Transformers for continual learning remain unexplored.

### Biological Inspiration

The brain's resistance to forgetting is closely tied to **neuromodulation**. Neuromodulators such as acetylcholine enable rapid encoding of new memories while reducing interference with prior information by regulating membrane excitability and synaptic plasticity. This **dynamic regulation of neuronal firing thresholds** inspires CATFormer's core design.

## Method

### Overall Architecture

CATFormer is built on the SpikFormer (spiking Vision Transformer) backbone and employs a **two-stage training protocol** combined with **gated dynamic head selection** for task-agnostic inference. The core innovation is: after freezing backbone weights, the model adapts to new tasks solely by learning **task-specific dynamic firing thresholds**.

### Key Designs

#### 1. **Dynamic Threshold LIF Neuron Model (DTLIF)**

Context-adaptive learnable firing thresholds are introduced on top of the standard LIF neuron:

- **Membrane potential update**: $\tilde{V}_j^{(t)} = (1 - \frac{1}{\tau}) V_j^{(t-1)} + \frac{1}{\tau} I_j^{(t)}$
- **Spike generation**: $S_j^{(t)} = \Theta(\tilde{V}_j^{(t)} - \phi_j^{(k)})$, where $\phi_j^{(k)}$ is the task-$k$-specific threshold
- **Soft reset**: $V_j^{(t)} = \tilde{V}_j^{(t)} - S_j^{(t)} \phi_j^{(k)}$

Thresholds are updated via gradient descent: $\phi_j^{(k)} \leftarrow \phi_j^{(k)} - \eta \frac{\partial \mathcal{L}}{\partial \phi_j^{(k)}}$

This allows each channel to adjust its firing threshold per task, realizing task-adaptive spiking dynamics.

#### 2. **Two-Stage Training Protocol**

- **Task 0 (base task)**: Jointly trains the entire backbone $\theta$, initial thresholds $\phi^{(0)}$, and classification head $W_0$ using cross-entropy loss.
- **Task k > 0 (incremental tasks)**: **All prior parameters are frozen**; only the new classification head $W_k$ and new threshold parameters $\phi^{(k)}$ are optimized.

$$\min_{\{\phi^{(k)}, W_k\}} \mathbb{E}_{(x,y) \sim \mathcal{D}^k} [\mathcal{L}_{CE}(W_k \cdot f(x; \theta, \phi^{(k)}), y)]$$

Each task requires storing only approximately 16,032 threshold parameters (~64.2 KB in FP32), making the approach extremely lightweight.

#### 3. **Gated Dynamic Head Selection (G-DHS)**

At inference, a two-layer MLP gating network performs task prediction and classification head routing:

- Features $\mathbf{f}_{base}(x)$ are extracted using the **base threshold** $\phi_{init}$
- The gating network predicts the task ID: $k^* = \arg\max(\mathcal{G}(\mathbf{f}_{base}(x)))$
- Task-specific threshold $\phi^{(k^*)}$ is then applied to re-extract features for final classification

Gating network structure: $\mathcal{G}(\mathbf{f}) = \text{Linear}(\text{ReLU}(\text{Linear}(\mathbf{f})))$, $\mathbb{R}^D \to \mathbb{R}^{D/4} \to \mathbb{R}^k$

### Loss & Training

- The backbone and classification heads are trained with standard cross-entropy loss $\mathcal{L}_{CE}$.
- The gating MLP is trained separately after each task using features extracted under the base threshold (also with cross-entropy); features are only used locally within the current task and are not stored across tasks.
- Thresholds are initialized at $\phi_{init} = 0.5$.

## Key Experimental Results

### Main Results

| Dataset | Tasks | Ours (CATFormer) | DSD-SNN | EWC | iCaRL | DER++ |
|--------|--------|-----------------|---------|-----|-------|-------|
| CIFAR-100 | 10 | **68.33** ± 4.51 | 60.47 ± 0.72 | 18.81 | 33.46 | 34.99 |
| CIFAR-100 | 25 | **71.34** ± 1.75 | 53.79 ± 2.67 | 15.73 | 22.37 | 24.90 |
| CIFAR-100 | 50 | **75.66** ± 2.72 | 50.55 ± 1.76 | 9.73 | 10.89 | 13.12 |
| CIFAR-10 | 5 | **89.29** ± 2.53 | — | 80.39 (SA-SNN+EWC) | — | — |
| Tiny-ImageNet | 100 | **48.56** ± 0.81 | — | — | — | 40.11* |
| CIFAR10-DVS | 5 | **87.14** ± 2.78 | 76.57 | — | — | — |
| SHD (T=16) | 10 | **87.85** ± 1.20 | 80.47 | — | — | — |

### Ablation Study

| Configuration | Accuracy (%) | Task 0 Accuracy (%) | Notes |
|------|-----------|-----------------|------|
| **CATFormer (Full)** | **89.29** ± 2.53 | 93.87 ± 0.45 | Full model |
| Fixed Threshold | 42.87 ± 1.26 | 72.59 ± 1.86 | Fixed threshold → severe forgetting |
| SpikIdentityFormer | 59.38 ± 0.98 | 70.62 ± 1.75 | Attention removed |
| Random Identity Former | 53.17 ± 2.13 | 62.43 ± 0.99 | Attention randomly replaced |
| FFN Frozen | 63.24 ± 1.78 | 72.17 ± 1.59 | Feed-forward network frozen |

### Key Findings

1. **Reverse forgetting phenomenon**: CATFormer's accuracy increases with the number of tasks (68.33% → 75.66%), which is in complete contrast to the declining trend exhibited by all existing methods.
2. **High parameter efficiency**: Only ~1.4M parameters are updated per task (vs. 9.32M+ for other methods), requiring only 64.2 KB of storage per task.
3. **Fixed Threshold ablation** demonstrates that dynamic thresholds, not synaptic plasticity alone, are the key to preventing forgetting.
4. **Neuromorphic dataset compatibility**: The model also achieves strong performance on event-driven datasets such as CIFAR10-DVS and SHD.

## Highlights & Insights

1. **Reverse forgetting** is the most surprising finding: when each task contains fewer classes, the model learns better. This more closely reflects real-world scenarios (robots do not encounter 50 new categories at once).
2. The design philosophy of **"thresholds as memory"** is novel: rather than modifying weights, new knowledge is encoded solely by regulating neuronal excitability, analogous to neuromodulatory mechanisms in the brain.
3. Continual learning research is advanced from CNN backbones to spiking Transformers, filling an important gap in the literature.

## Limitations & Future Work

1. The task prediction accuracy of the gating network is not reported separately, which may represent a system bottleneck.
2. Validation is limited to classification tasks; more complex tasks such as detection and segmentation are not addressed.
3. Deployment on real neuromorphic hardware (e.g., Loihi 2) has not been verified.
4. Initial task training requires substantial computation (10.5M parameters), which partially contradicts the claimed lightweight nature of the approach.
5. Although threshold parameters are few, they grow linearly with the number of tasks; no upper-bound analysis is provided.

## Related Work & Insights

- **DSD-SNN** (Han et al., 2023) was the previous strongest replay-free SNN continual learning method, but is CNN-based and exhibits continuously declining performance.
- **SpikFormer** (Zhou et al., 2023) introduced the Transformer architecture into SNNs but did not explore continual learning.
- The threshold modulation idea is generalizable to other SNN architectures and non-visual tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ (Dynamic thresholds for continual learning is novel, though the overall architecture is relatively straightforward)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers both static and neuromorphic datasets with sufficient ablation)
- Writing Quality: ⭐⭐⭐⭐ (Well-organized; biological motivation is clearly articulated)
- Value: ⭐⭐⭐⭐ (The reverse forgetting phenomenon carries important implications)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](attention_retention_for_continual_learning_with_vision_transformers.md)
- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](../../CVPR2026/llm_safety/elastic_weight_consolidation_done_right_for_continual_learning.md)
- [\[NeurIPS 2025\] Finding Structure in Continual Learning](../../NeurIPS2025/llm_safety/finding_structure_in_continual_learning.md)
- [\[AAAI 2026\] PANDA: Patch and Distribution-Aware Augmentation for Long-Tailed Exemplar-Free Continual Learning](panda_--_patch_and_distribution-aware_augmentation_for_long-tailed_exemplar-free.md)
- [\[AAAI 2026\] Uncovering Bias Paths with LLM-guided Causal Discovery: An Active Learning and Dynamic Scoring Approach](uncovering_bias_paths_with_llm-guided_causal_discovery_an_active_learning_and_dy.md)

</div>

<!-- RELATED:END -->
