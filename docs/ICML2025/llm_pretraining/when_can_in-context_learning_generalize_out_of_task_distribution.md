---
title: >-
  [Paper Note] When Can In-Context Learning Generalize Out of Task Distribution?
description: >-
  [ICML2025][LLM Pretraining][in-context learning] By systematically varying the coverage of the training task distribution (semi-angle $\phi$ of a hyperspherical cap) on linear regression ICL tasks, a sharp phase transition from "specialized solutions" to "general-purpose solutions" in transformers is identified: when task diversity exceeds a critical threshold ($\phi \gtrsim 120°$), the model can generalize to the entire task space, even surpassing the OOD performance of the…
tags:
  - "ICML2025"
  - "LLM Pretraining"
  - "in-context learning"
  - "out-of-distribution generalization"
  - "task diversity"
  - "phase transition"
  - "linear regression"
  - "transformer"
  - "specialization"
date: 2026-05-08
content_hash: 769ba32d5ca14e6f
---

# When Can In-Context Learning Generalize Out of Task Distribution?

**Conference**: ICML2025  
**arXiv**: [2506.05574](https://arxiv.org/abs/2506.05574)  
**Code**: [GitHub](https://github.com/cwgoddard/OOD_ICL)  
**Area**: LLM Pre-training  
**Keywords**: in-context learning, out-of-distribution generalization, task diversity, phase transition, linear regression, transformer, specialization

## TL;DR

By systematically varying the coverage of the training task distribution (semi-angle $\phi$ of a hyperspherical cap) on linear regression ICL tasks, a sharp phase transition from "specialized solutions" to "general-purpose solutions" in transformers is identified: when task diversity exceeds a critical threshold ($\phi \gtrsim 120°$), the model can generalize to the entire task space, even surpassing the OOD performance of the Bayes optimal estimator.

## Background & Motivation

### Generalization Ability of In-Context Learning

In-context learning (ICL) is a remarkable capability of pre-trained transformers, enabling models to infer new tasks from a few in-context examples without retraining. Existing studies primarily focus on the impact of the **number of pre-training tasks** on the emergence of ICL, while ignoring another key dimension: **task similarity**.

### Core Problem

If a model is pre-trained only on a subset of the task space, can it generalize to the remainder of the task space?

Specifically, when the task pool is limited and the distribution is restricted, does the model learn a **specialized solution** that only functions within the training distribution, or can it develop a **general-purpose solution** that covers the entire task space?

### Experimental Paradigm

Linear regression is selected as the experimental platform for studying ICL:

- Input sequence $\{x_i, y_i\}_{i=1}^n$, where $y_i = w^T x_i + \epsilon_i$
- Tasks are defined by the weight vector $w \in \mathbb{R}^d$
- The goal of the transformer is to predict $y_k$ given the context $C_k = \{x_1, y_1, \dots, x_k\}$

## Method

### Geometric Control of Task Distributions

The task distribution is parameterized via a hyperspherical cap:

$$S^{d-1}(\phi) = \{w \in S^{d-1} \mid \text{angle}(w, v) \leq \phi\}$$

- $v$ is a fixed "polar" direction.
- $\phi \in [0°, 180°]$ controls the coverage: $\phi = 180°$ represents the entire hypersphere.
- Training distribution: $p_\phi(w) = \text{Unif}(S^{d-1}(\phi))$
- **Larger $\phi$ indicates higher diversity of training tasks.**

### Test Distributions

Hyperspherical bands are used as test distributions:

$$B^{d-1}(\delta, \Delta\delta) = \{w \in S^{d-1} \mid \delta \leq \text{angle}(w, v) \leq \delta + \Delta\delta\}$$

- $\delta = 0°$: the test distribution is in-distribution (ID).
- $\delta = 175°$: the test distribution is on the "opposite side" of the training distribution (extreme OOD).

### Training Setup

- GPT-2 style transformer with 128 hidden dimensions, 10 layers, and 8 attention heads.
- $d = 10$-dimensional regression, with $n = 50$ samples per context.
- 12 models trained with $\phi \in [15°, 180°]$ in steps of $15°$.
- Optimization objective: $L_{\text{train}}(\theta) = \mathbb{E}_{w \sim p_\phi}\left[\frac{1}{n}\sum_{k=1}^n (T_\theta(C_k) - y_k)^2\right]$

### Bayes Optimal Baseline

Given the prior $p_\phi(w)$, the Bayes optimal estimator is the posterior mean:

$$\hat{w} = \frac{\int dw \, p(w) w \prod_{k=1}^{n-1} p(y_k | x_k, w)}{\int dw \, p(w) \prod_{k=1}^{n-1} p(y_k | x_k, w)}$$

Key property: **$p_\phi(w)$ has no support outside of $S^{d-1}(\phi)$**, which forces $\hat{w}$ to lie within $S^{d-1}(\phi)$. Consequently, the Bayes optimal estimator **cannot produce meaningful OOD generalizations**.

## Key Experimental Results

### Core Findings: Specialized-to-General Phase Transition

| $\phi$ | $\delta = 0°$ (In-distribution) | $\delta = 175°$ (Extreme OOD) | Solution Type |
|:---:|:---:|:---:|:---:|
| $\leq 90°$ | Low error | High error | Specialized Solution |
| $\approx 120°$ | Low error | Starting to drop | Critical Point |
| $\geq 120°$ | Low error | Low error | General-purpose Solution |

**Sharp transition at $\phi \approx 120°$**:

- $\phi < 120°$: Model performance drops sharply outside the training distribution.
- $\phi \geq 120°$: Model performs consistently well across all test angles $\delta$.
- With noise ($\sigma^2 = 0.25$), the critical point shifts to $\phi \approx 135°$.

### Properties of Specialized vs. General-Purpose Solutions

| Property | Specialized Solution ($\phi < 120°$) | General-Purpose Solution ($\phi \geq 120°$) |
|------|------------------------|--------------------------|
| Short-context Performance | **Outperforms OLS** | Comparable to OLS |
| OOD Performance | Poor | **Outperforms Bayes Optimal** |
| Essence | Fits strong priors | Learns general algorithms (like OLS) |

**Specialized solutions outperform OLS in-distribution**: By fitting the prior of the training distribution, specialized solutions perform better on ID tasks given a small number of samples ($k < d$), but at the expense of OOD generalization.

**General-purpose solutions outperform Bayes optimal**: While the Bayes optimal estimator is restricted to $S^{d-1}(\phi)$ by the training prior, the model achieves OOD generalization precisely because it **fails to fit the Bayesian solution perfectly**.

### Phase Diagram: Interaction of Two Types of Task Diversity

A grid of 480 models ($\phi \times N$, where $N$ is the number of distinct training tasks) is trained, revealing three phases:

| Phase | In-distribution Generalization | Out-of-distribution Generalization | Conditions |
|:---:|:---:|:---:|------|
| In-weights Learning (IWL) | ✕ | ✕ | Small $N$, arbitrary $\phi$ |
| In-distribution ICL | ✓ | ✕ | Large $N$, $\phi < 120°$ |
| OOD ICL | ✓ | ✓ | Large $N$, $\phi \geq 120°$ |

The phase boundaries exhibit a diagonal structure, indicating a trade-off between the two dimensions of diversity.

### Influence of Dimension and Depth

| Factor | Effect on Critical Angle $\phi_c$ |
|------|------------------------|
| Regression Dimension $d$ ($3, 5, 10, 20$) | **No effect**, $\phi_c \approx 120°$ remains constant. |
| Model Depth ($2, 3, 10$ layers) | **No effect**, $\phi_c \approx 120°$ remains constant. |

The critical angle does not vary with dimension or depth, indicating that this is not a simple consequence of high-dimensional geometry.

### Extension to Non-Linear Regression

| Task Type | Critical Angle $\phi_c$ |
|----------|:--------------:|
| Linear Regression | $\approx 120°$ |
| Logistic Regression (Classification) | $\approx 135°$ |
| Non-linear Regression (Single-hidden-layer ReLU network, sampled independently) | $\approx 135°$ |
| Non-linear Regression (Jointly sampled) | $\approx 60°$ |

The phase transition phenomenon persists in non-linear and classification tasks, suggesting it as a **universal feature** of ICL.

### Generalization Beyond the Hypersphere

Can a model trained on the unit hypersphere ($R = 1$) generalize to other radii?

- At $\phi \geq 45°$, the model generalizes perfectly to the inner sphere ($R < 1$).
- Task diversity drives not only OOD generalization on the sphere, but also **generalization beyond the sphere**.

## Highlights & Insights

1. **Discovery of phase transition**: For the first time, a sharp phase transition from specialized to general-purpose solutions in ICL generalization is demonstrated, with the critical point remaining robust to changes in dimension and depth.
2. **A new dimension of task diversity**: This work proposes measuring diversity using the coverage of the task space ($\phi$) rather than just the number of tasks ($N$), revealing that the two are fundamentally different.
3. **Outperforming Bayes optimal in OOD generalization**: The OOD performance of general-purpose solutions surpasses the Bayes optimal estimator because the model **fails** to perfectly fit the training prior.
4. **Three-phase phase diagram**: The three-phase structure of IWL $\rightarrow$ in-distribution ICL $\rightarrow$ OOD ICL provides a clear picture of the transition from memorization to generalization.
5. **Implications for LLMs**: Explains why LLMs can solve tasks outside the pre-training distribution in ICL—highly diverse training data encourages models to learn general-purpose algorithms instead of task-specific priors.

## Limitations & Future Work

1. **Verified only in simplified setups**: Linear or simple non-linear regressions are far from representing the actual complexity of ICL tasks faced by LLMs.
2. **Limited metric for task similarity**: While linear tasks naturally use the inner product $w_1^T w_2$ to measure similarity, more general tasks lack a corresponding metric.
3. **Lack of analytical theory**: All conclusions are empirical observations; a rigorous theory explaining the phase transition mechanism is still missing (unlike the solvable models in e.g., Lu et al., 2024).
4. **Small model scale**: The experiments use a small transformer with a hidden dimension of 128, which is far from the scale of real-world LLMs.
5. **No consideration of concept shift**: The study focuses solely on task distribution shift, whereas Yadlowsky et al. (2024) have demonstrated that transformers typically fail to generalize under concept shift.

## Rating

⭐⭐⭐⭐ — Elegant experimental design that uncovers a clean and robust phase transition phenomenon, yielding critical theoretical insights into the generalization mechanism of ICL. However, the simplified settings limit its direct applicability to real-world LLM behaviors.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](../../ACL2025/llm_pretraining/data_whisperer_data_selection.md)
- [\[NeurIPS 2025\] The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation](../../NeurIPS2025/llm_pretraining/the_atlas_of_in-context_learning_how_attention_heads_shape_in-context_retrieval_.md)
- [\[ICML 2025\] In-Context Adaptation to Concept Drift for Learned Database Operations](in-context_adaptation_to_concept_drift_for_learned_database_operations.md)
- [\[ICLR 2026\] Task-Aware Data Selection via Proxy-Label Enhanced Distribution Matching for LLM Fine-Tuning](../../ICLR2026/llm_pretraining/task-aware_data_selection_via_proxy-label_enhanced_distribution_matching_for_llm.md)
- [\[NeurIPS 2025\] Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](../../NeurIPS2025/llm_pretraining/ricl_temporal_credit.md)

</div>

<!-- RELATED:END -->
