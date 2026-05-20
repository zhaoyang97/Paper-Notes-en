---
title: >-
  [Paper Note] Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods
description: >-
  [NeurIPS 2025][LLM Pretraining][Training Data Attribution] This paper explicitly formulates the "Final-Model-Only" (FiMO) setting for training data attribution (TDA)…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "Training Data Attribution"
  - "Influence Functions"
  - "Gradient Methods"
  - "Further Training"
  - "Final-Model-Only"
date: 2026-05-08
content_hash: 107a4d4c38542ae1
---

# Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods

**Conference**: NeurIPS 2025
**arXiv**: [2412.03906](https://arxiv.org/abs/2412.03906)  
**Code**: [IBM/fimoda](https://github.com/IBM/fimoda)  
**Area**: LLM Pre-training
**Keywords**: Training Data Attribution, Influence Functions, Gradient Methods, Further Training, Final-Model-Only

## TL;DR
This paper explicitly formulates the "Final-Model-Only" (FiMO) setting for training data attribution (TDA), reframes the problem from measuring *contribution* to measuring *sensitivity*, proposes further training as the gold standard, and provides a unified derivation showing that various gradient-based methods (Grad-Dot, influence functions, TRAK, DataInf, etc.) are all approximations of further training at different orders.

## Background & Motivation

**Background**: Training data attribution (TDA) aims to explain model behavior in terms of training data. Existing methods fall into three broad categories: (1) retraining-based methods (Data Shapley, Datamodels); (2) methods that trace the training trajectory (TracIn); and (3) gradient-based methods applied solely to the final model (influence functions, TRAK). However, the literature has never explicitly distinguished the differences in **problem settings** underlying these methods.

**Limitations of Prior Work**: Existing TDA literature implicitly assumes access to the training algorithm or intermediate checkpoints. In practice, the most common scenario is "final model only"—such as open-source models downloaded from HuggingFace. Under this setting, neither retraining nor intermediate checkpoints are available, and neither the ideal objective nor the gold standard is well-defined.

**Key Insight**: The authors formally define three levels of access—TAA (training algorithm available), CPA (checkpoints available), and FiMO (final model only)—focus on the FiMO setting, and reframe the TDA problem from a *contribution measure* to a *sensitivity measure*.

## Core Problem

- **What should TDA target under the FiMO setting?** Without the ability to "go back in time" and trace the training process, how should the influence of training samples on the final model be measured?
- **Absence of a gold standard**: Existing proxy tasks (e.g., mislabeled sample detection) are insufficient for evaluating and advancing FiMO methods; a directly measurable ideal standard is needed.
- **Unclear relationships among gradient methods**: Methods such as Grad-Dot, influence functions, TRAK, and DataInf appear to be independent, lacking a unified perspective.

## Method

### 1. Reframing the FiMO Problem: From Contribution to Sensitivity

In the TAA setting, the natural question concerns **contribution**—how much does training sample $z_i$ contribute through the training process? In the FiMO setting, however, the training process cannot be traced back. The authors reframe the question as **sensitivity**—given the final model, how sensitive is it to training sample $z_i$?

### 2. Further Training as the Gold Standard

Starting from the final parameters $\theta^f$, further training is performed on both the full training set $\mathcal{D}$ and the set $\mathcal{D}_{-i}$ with sample $i$ removed:

$$a_i^* = \mathbb{E}_\xi\left[g(z, \theta^f + \Delta\theta(\mathcal{D}_{-i}, \xi)) - g(z, \theta^f + \Delta\theta(\mathcal{D}, \xi))\right]$$

Two key refinements are introduced:

- **Non-convergence correction**: Since $\theta^f$ is typically not a stationary point of the empirical risk, further training on $\mathcal{D}$ itself produces a non-zero update $\Delta\theta(\mathcal{D})$; this "training-only effect" must be subtracted.
- **Averaging over randomness**: The expectation is taken over the stochasticity of the training algorithm (e.g., mini-batch order $\xi$) to eliminate random noise.

### 3. Unified Derivation: Gradient Methods ≈ Approximate Further Training

A Taylor expansion with regularization is applied to the further training objective:

$$\widehat{\Delta\theta}(\mathcal{D}') = \arg\min_{\Delta\theta} \nabla R(\mathcal{D}'; \theta^f)^T \Delta\theta + \frac{1}{2}\Delta\theta^T (\nabla^2 R + \lambda I) \Delta\theta$$

After applying a first-order approximation of the evaluation function $g$ in $\Delta\theta$, the attribution score simplifies to:

$$\hat{a}_i = \nabla_\theta g(z, \theta^f)^T (\widehat{\Delta\theta}(\mathcal{D}_{-i}) - \widehat{\Delta\theta}(\mathcal{D}))$$

#### First-Order Method → Grad-Dot

Omitting the Hessian term directly yields:

$$\hat{a}_i \propto \nabla_\theta g(z, \theta^f)^T \nabla_\theta L(z_i, \theta^f)$$

This is the gradient inner product, corresponding to Grad-Dot (also a special case of TracIn using only the final checkpoint).

#### Second-Order Methods → Influence Function Family

Retaining the Hessian term and applying the implicit function theorem yields the generalized influence function (Proposition 1):

$$\widehat{\Delta\theta}(\mathcal{D}_{-i,\epsilon}) - \widehat{\Delta\theta}(\mathcal{D}) \approx \epsilon (H(\theta^f) + \lambda I)^{-1} (\nabla_\theta L(z_i; \theta^f) + \nabla^2_\theta L(z_i; \theta^f) \widehat{\Delta\theta}(\mathcal{D}))$$

This reduces to the classical form when the model is at a stationary point, i.e., $\widehat{\Delta\theta}(\mathcal{D})=0$. Introducing the Gauss-Newton approximation further yields Corollary 2.

#### Unified Placement of Each Method

| Method | Position in the Unified Framework |
|------|-----|
| **Grad-Dot** | First-order expansion, gradient inner product |
| **Grad-Cos** | First-order + normalization (theoretically unjustified) |
| **CG / LiSSA** | Second-order, iterative inverse Hessian-vector product |
| **TRAK$_{M=1}$** | Gauss-Newton + random projection, $\lambda=0, V=I$ |
| **EK-FAC** | Gauss-Newton + layer-wise block + Kronecker factorization |
| **DataInf** | Gauss-Newton + identity loss + swapped averaging and inversion |

### 4. Generalized Influence Functions

The key distinction from classical derivations is that **convexity and stationarity are not assumed**. Proposition 1 retains the additional term $\nabla^2_\theta L(z_i; \theta^f) \widehat{\Delta\theta}(\mathcal{D})$, which accounts for the non-convergence of the model. Near a stationary point, a reverse Taylor expansion simplifies this to:

$$\widehat{\Delta\theta}(\mathcal{D}_{-i,\epsilon}) - \widehat{\Delta\theta}(\mathcal{D}) \approx \epsilon (H(\theta^f) + \lambda I)^{-1} \nabla_\theta L(z_i; \theta^f + \widehat{\Delta\theta}(\mathcal{D}))$$

That is, the gradient is evaluated at the parameters obtained after further training on $\mathcal{D}$.

## Key Experimental Results

### Experimental Setup

- **Datasets**: Tabular data (Concrete, Energy, FICO, Folktables), images (CIFAR-10 + ResNet-9), text (SST-2 + BERT)
- **Gold Standard**: LOO further training averaged over 100 random seeds
- **Evaluation Metric**: Cosine similarity between attribution score vectors
- **Baselines**: Grad-Dot, Grad-Cos, CG, LiSSA, LiSSA-H, TRAK$_{M=1}$, EK-FAC, DataInf (8 methods in total)

### Key Findings

1. **First-order vs. influence functions**: First-order methods (Grad-Dot) achieve the highest initial cosine similarity (up to ~0.9) but decay rapidly as the amount of further training increases; influence function methods (CG, LiSSA) are more stable but consistently yield lower peak similarity.
2. **DataInf ≈ Grad-Dot**: Despite DataInf's attempt to incorporate second-order information, its behavior closely resembles a first-order method (cosine similarity between the two exceeds 0.95).
3. **TRAK$_{M=1}$ underperforms**: In the FiMO setting, only $M=1$ is feasible (no retraining of multiple models), substantially degrading performance.
4. **Averaging improves quality**: Increasing the number of further training random seeds (from 1 to 100) consistently raises the similarity between the gold standard and gradient-based methods, confirming the effectiveness of averaging.
5. **Non-tabular data is harder**: Cosine similarity for all methods on CIFAR-10 and SST-2 is substantially lower than on tabular data.

### Computational Cost

Further training BERT on SST-2 requires approximately 1,000 GPU-hours (V100); total experiments require approximately 3,000 GPU-hours.

## Highlights & Insights

1. **Clarification of problem settings**: This is the first work to explicitly define the FiMO setting and systematically articulate its distinctions from TAA/CPA, providing important conceptual clarity for the field.
2. **Unified perspective**: Eight seemingly disparate gradient-based methods are unified as approximations of further training at different orders, yielding a theoretically clean and powerful framework.
3. **Generalized influence functions**: A generalized formulation is proposed that does not rely on convexity or stationarity assumptions, incorporating a non-convergence correction term.
4. **Experimental rigor**: Averaging over 100 random seeds—far exceeding the scale of prior work—reveals the critical importance of averaging for gold standard quality.
5. **Counterintuitive finding**: Influence functions (second-order) do not consistently outperform simple Grad-Dot (first-order), at least under the FiMO setting.

## Limitations & Future Work

1. **High computational cost**: The further training gold standard is itself expensive to compute (~1,000 GPU-hours per dataset), limiting experimental scale.
2. **Model scale**: The largest model evaluated is BERT-base; true LLM-scale models (e.g., GPT-3/LLaMA-level) are not considered.
3. **LOO limitation**: Only single-sample removal is considered; group influence is not explored.
4. **Poor performance on non-tabular data**: Approximation quality is unsatisfactory for all methods on CIFAR-10 and SST-2, indicating a gap from practical applicability.
5. **Choosing the amount of further training**: The paper provides no clear criterion for how much further training is "sufficient" to measure sensitivity.
6. **LoRA acceleration**: The authors mention parameter-efficient fine-tuning as a potential substitute for full further training but do not experiment with it.

## Related Work & Insights

| Work | Distinction from This Paper |
|------|------------|
| **Koh & Liang (2017)** | Pioneered influence functions in ML, but derivation assumes convexity/stationarity and evaluates only 2 methods. |
| **Bae et al. (2022)** | Proposes PBRF as an alternative gold standard, but PBRF uses a non-standard Bregman divergence (designed specifically to align with influence functions) and is less general than further training. |
| **Schioppa et al. (2023)** | Also observes decay in approximation quality over training, but does not explicitly formulate the FiMO setting or average over stochasticity. |
| **Basu et al. (2021)** | Finds that influence functions degrade with model depth/width, but also does not distinguish FiMO or apply non-convergence correction. |
| **Park et al. (2023) TRAK** | Performs well in the TAA setting (multiple checkpoints, $M \gg 1$) but degrades substantially under FiMO ($M=1$). |

### Connections and Implications

1. **First- vs. second-order trade-off**: Experiments reveal an intriguing pattern—first-order approximations are better in the short range, while second-order approximations are more stable in the long range. Interpolation between the two via the damping parameter $\lambda$ may be worth exploring.
2. **Practical gold standard**: The authors note that 10–20 seeds capture most of the benefit compared to 100; combined with LoRA, further training could become a viable evaluation tool.
3. **Potential of generalized influence functions**: The correction term involving $\widehat{\Delta\theta}(\mathcal{D})$ in Proposition 1 is neglected by existing methods and may offer a pathway to improved performance on non-tabular data.
4. **Connection to model auditing**: The FiMO setting is naturally suited for third-party model auditing and data compliance inspection (e.g., data impact assessment under GDPR).

## Rating

- Novelty: ⭐⭐⭐⭐ (Significant contribution in clarifying problem settings and providing a unified perspective, though no entirely new method is proposed)
- Experimental Thoroughness: ⭐⭐⭐⭐ (8 methods × 6 datasets × 100 seeds; substantial scale, though model size is limited)
- Writing Quality: ⭐⭐⭐⭐⭐ (Well-organized, mathematically rigorous, with in-depth discussion)
- Value: ⭐⭐⭐⭐ (Valuable for conceptual clarity in TDA; experimental findings are practically informative)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](enhancing_training_data_attribution_with_representational_optimization.md)
- [\[NeurIPS 2025\] Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training](through_the_river_understanding_the_benefit_of_schedule-free_methods_for_languag.md)
- [\[NeurIPS 2025\] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](language_model_behavioral_phases_are_consistent_across_archi.md)
- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)

</div>

<!-- RELATED:END -->
