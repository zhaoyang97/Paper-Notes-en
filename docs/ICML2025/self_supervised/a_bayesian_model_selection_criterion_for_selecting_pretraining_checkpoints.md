---
title: >-
  [Paper Note] A Bayesian Model Selection Criterion for Selecting Pretraining Checkpoints
description: >-
  [ICML 2025][Self-Supervised Learning][Bayesian Model Selection] Introduces "downstream free energy" as a Bayesian model selection criterion for pretraining checkpoint adaptability, proves that "pretraining free energy" serves as its upper bound proxy (without requiring downstream data), and experimentally validates that a large learning rate, small batch size, and high momentum improve downstream transfer performance by reducing pretraining free energy.
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "Bayesian Model Selection"
  - "Free Energy"
  - "Pretraining Checkpoints"
  - "Transfer Learning"
  - "Loss Landscape"
  - "Implicit Regularization"
date: 2026-05-08
content_hash: 5c60c61fad333064
---

# A Bayesian Model Selection Criterion for Selecting Pretraining Checkpoints

**Conference**: ICML 2025  
**arXiv**: [2410.05612](https://arxiv.org/abs/2410.05612)  
**Code**: None  
**Area**: Transfer Learning Theory  
**Keywords**: Bayesian Model Selection, Free Energy, Pretraining Checkpoints, Transfer Learning, Loss Landscape, Implicit Regularization

## TL;DR

Introduces "downstream free energy" as a Bayesian model selection criterion for pretraining checkpoint adaptability, proves that "pretraining free energy" serves as its upper bound proxy (without requiring downstream data), and experimentally validates that a large learning rate, small batch size, and high momentum improve downstream transfer performance by reducing pretraining free energy.

## Background & Motivation

**Background**: The "pretrain-finetune" paradigm of foundation models dominates NLP and CV. Numerous checkpoints are generated during pretraining, but selecting which checkpoint to fine-tune has lacked theoretical guidance—in practice, it mainly relies on empirical heuristics: using the final checkpoint or assuming a large learning rate is better.

**Limitations of Prior Work**: (1) Lack of principled checkpoint selection criteria; (2) Known empirical rules (large learning rate/small batch size benefit transfer) lack a unified theoretical explanation; (3) Prior theoretical works have respective limitations—the Hessian trace in Liu et al. 2023a lacks formal bounds, the neural collapse in Galanti et al. 2022 lacks practical regularization methods and mainly targets linear probing rather than full-parameter fine-tuning.

**Key Challenge**: Downstream tasks are unknown during pretraining—how to determine the checkpoint's adaptation potential without access to downstream data?

**Goal**: Provide a principled theoretical framework for checkpoint selection, establishing a provable connection from "pretraining characteristics" to "downstream transfer performance".

**Key Insight**: Leverage the concept of marginal likelihood (free energy) in Bayesian statistics—restricted to the neighborhood of the checkpoint—as a measure of adaptability.

**Core Idea**: Low free energy implies a high concentration of good parameters near the checkpoint, making fine-tuning more likely to succeed. The theoretical chain is: downstream test error $\lesssim$ downstream free energy $\lesssim$ pretraining free energy.

## Method

### Overall Architecture

Establishes a three-layer theoretical chain:

$$\text{downstream Bayesian test error} \lesssim \text{downstream free energy} \lesssim \text{pretraining free energy}$$

Core steps: (1) Define downstream free energy as a measure of checkpoint adaptability; (2) Define pretraining free energy as a proxy computable using only pretraining data; (3) Prove that pretraining free energy controls downstream free energy (Proposition 5.1); (4) Validate using known SGD implicit biases.

### Key Designs

1. **Downstream Free Energy**: For a checkpoint $w^* = (v^*, \theta^*) \in U_0$, define:

    $$\bar{F}^1(B_\gamma(w^*)) = -\log \int_{B_\gamma(w^*)} \exp\{-m K^1(w)\} \varphi(w) \, dw$$

    where $B_\gamma(w^*) = \{w = (v^*, \theta) : \|\theta - \theta^*\|^2 \leq 1/\gamma\}$ is the neighborhood of the backbone parameters (freezing the linear head), and $K^1(w)$ is the downstream test loss. The asymptotic expansion is $\bar{F}^1 = mK^1(w^{*1}) + \lambda^1(w^*) \log m + O(\log\log m)$, where $\lambda^1$ (the local learning coefficient) measures local model complexity. Checkpoints with high loss but low complexity may outperform those with low loss but high complexity—reflecting Bayesian Occam's razor.

2. **Pretraining Free Energy as a Proxy**: $F^0(B_\gamma(w^*); \beta) = -\log \int_{B_\gamma(w^*)} \exp\{-n\beta \hat{K}^0(w)\} \varphi(w) \, dw$, based on the pretraining training loss $\hat{K}^0$ and the inverse temperature $\beta$. Proposition 5.1 proves that under covariate shift conditions, $\bar{F}^1 \leq F^0 + \text{shift term}$—the pretraining free energy upper-bounds the downstream free energy.

3. **Connection with SGD Implicit Regularization**: Lau et al. (2025) proved that a large learning rate, small batch size, and high momentum implicitly reduce the $\lambda^0$ (local learning coefficient) in the free energy. This work validates the loop: pretraining hyperparameters that reduce pretraining free energy $\rightarrow$ better downstream transfer performance. This provides practically actionable guidance: adjusting pretraining hyperparameters can indirectly optimize checkpoint transferability.

### Loss & Training

- The backbone $\phi_\theta$ is shared between pretraining and fine-tuning, each having independent linear heads $v$ and $u$.
- Fine-tuning adopts limited fine-tuning (backbone uses a smaller learning rate).
- Both pretraining and downstream losses are in the form of KL divergence: $K^i(w) = \mathbb{E}_{r^i(x)} D_\text{KL}(r^i(y|x) \| p(y|x,w))$.

## Key Experimental Results

### Main Results: Pretraining Free Energy vs. Transfer Accuracy ($R^2$)

| Hyperparameters | $R^2$ | Trend |
|-------|-------|------|
| Learning rate | 0.91 | Large lr → low free energy → high transfer accuracy |
| Batch size | 0.87 | Small batch → low free energy → high transfer accuracy |
| Momentum | 0.85 | High momentum → low free energy → high transfer accuracy |

### Comparison with Other Pretraining Metrics

| Pretraining Metric | Correlation with Downstream Performance | Characteristics |
|----------|---------------|------|
| Pretraining loss | Weak | Low loss does not guarantee good transfer |
| Hessian trace (sharpness/flatness) | Moderate | Correlated but not causal |
| Neural Collapse | Moderate | Lack of explicit regularization methods |
| **Pretraining Free energy** | **Strongest** | Simultaneously captures fit and complexity |

### Ablation Study

| Analysis Dimension | Conclusion |
|---------|------|
| $mK^1(w^{*1})$ (fit) vs. $\lambda^1 \log m$ (complexity) | Both jointly determine checkpoint quality, not a single factor |
| Linear probing vs. Full fine-tuning | Ours applies to full fine-tuning (prior work mostly considered linear probing) |
| Multi-dataset validation | Consistent across CIFAR-10/100 and multiple architectures (ResNet, ViT-Small) |

### Key Findings

- Pretraining free energy correlates more strongly with downstream performance than all other candidate metrics.
- Provides a unified explanation for known empirical rules: large lr / small batch / high momentum → reduced free energy.
- Free energy decomposition reveals the essence of checkpoint selection—not striving for the lowest loss, but balancing fit and complexity.
- Serves as an online monitoring metric during the pretraining process.

## Highlights & Insights

- Introduces the classical Bayesian framework to checkpoint selection in transfer learning for the first time—an application of old theory to a new setting.
- Complete theoretical chain: pretraining hyperparameters → pretraining free energy → downstream free energy → downstream performance.
- Explains the underlying reasons why SGD implicit regularization (flat minima) benefits transfer learning under a unified view.
- Practical guidance: monitoring free energy during pretraining allows selecting better checkpoints without knowing the downstream task.

## Limitations & Future Work

- Exact computation of free energy is intractable in high dimensions—requiring MCMC or variational inference approximations, and the computational overhead remains to be evaluated.
- The covariate shift assumption in Proposition 5.1 may not strictly hold in reality.
- Experiments are only validated on small-to-medium-scale models (ResNet, ViT-Small) and have not been extended to billion-parameter foundation models.
- Free energy is not invariant to parameter scaling—parameter scaling in pure ReLU networks does not change the outputs but changes the free energy, requiring batch norm/weight decay to break this invariance.

## Related Work & Insights

- Liu et al. (2023a): Empirical relation between Hessian trace and transfer, but lacks formal bounds.
- Galanti et al. (2022): Neural collapse chain, but lacks explicit regularization methods.
- Munn et al. (2024): geometric complexity→neural collapse, which this work further advances.
- Lau et al. (2025): Theoretical foundations for the local learning coefficient and SGD free energy regularization.

## Rating

⭐⭐⭐⭐ — Theoretically elegant and has practical guiding value, connecting classical Bayesian model selection with transfer learning, and explaining several known empirical laws in a unified manner. The main limitations lie in the relatively small experimental scale and the practical feasibility of approximating free energy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o.md)
- [\[ICML 2025\] Griffin: Towards a Graph-Centric Relational Database Foundation Model](griffin_towards_a_graph-centric_relational_database_foundation_model.md)
- [\[ICML 2025\] What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)
- [\[ICLR 2026\] A Bayesian Nonparametric Framework for Learning Disentangled Representations](../../ICLR2026/self_supervised/a_bayesian_nonparametric_framework_for_learning_disentangled_representations.md)
- [\[CVPR 2025\] MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining](../../CVPR2025/self_supervised/map_unleashing_hybrid_mamba-transformer_vision_backbones_potential_with_masked_a.md)

</div>

<!-- RELATED:END -->
