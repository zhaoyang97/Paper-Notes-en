---
title: >-
  [Paper Note] Optimizer Choice Matters for the Emergence of Neural Collapse
description: >-
  [ICLR 2026][Optimization][Neural Collapse] Through 3,900+ training experiments and theoretical analysis, this work reveals that the choice of optimizer (specifically the coupling of weight decay) plays a decisive role in the emergence of Neural Collapse—AdamW (decoupled weight decay) fails to produce Neural Collapse, while SGD and Adam (coupled weight decay) succeed.
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Neural Collapse"
  - "Optimizer Selection"
  - "Weight Decay Coupling"
  - "AdamW vs Adam"
  - "Implicit Bias"
date: 2026-05-08
content_hash: d9d8648e3fa951db
---

# Optimizer Choice Matters for the Emergence of Neural Collapse

**Conference**: ICLR 2026  
**arXiv**: [2602.16642](https://arxiv.org/abs/2602.16642)  
**Code**: None  
**Area**: Optimization Theory / Deep Learning Theory  
**Keywords**: Neural Collapse, Optimizer Selection, Weight Decay Coupling, AdamW vs Adam, Implicit Bias

## TL;DR

Through 3,900+ training experiments and theoretical analysis, this work reveals that the choice of optimizer (specifically the coupling of weight decay) plays a decisive role in the emergence of Neural Collapse—AdamW (decoupled weight decay) fails to produce Neural Collapse, while SGD and Adam (coupled weight decay) succeed.

## Background & Motivation

Neural Collapse (NC) is a phenomenon discovered by Papyan et al. (2020) at the end of deep network training: last-layer feature vectors and classifier weights self-organize into highly symmetric geometric structures. NC consists of four properties:
- **NC1**: Collapse of within-class variability (features collapse to class means).
- **NC2**: Convergence of class centers to a Simplex ETF (Equiangular Tight Frame).
- **NC3**: Alignment between classifier weights and class means (Self-Duality).
- **NC4**: Simplification of classification to a nearest-class-center classifier.

Existing theoretical analyses mostly ignore the role of the optimizer, suggesting NC is universal to all optimization methods. This paper challenges this assumption, proving that the choice of optimizer—specifically the coupling of weight decay—is crucial for the emergence of NC. A key finding is that **Adam produces NC, but the algorithmically near-identical AdamW does not**.

## Method

### Overall Architecture

This work centers on one question: why Adam and AdamW, despite being algorithmically near-identical, exhibit vastly different behaviors regarding Neural Collapse. The authors first propose a tractable diagnostic metric, NC0, proving its convergence to zero is a necessary condition for NC. They then conduct theoretical analysis on SGD and SignGD (idealized cases of Adam/AdamW) and finally validate theoretical predictions with 3,900+ training runs.

### Key Designs

**1. NC0 Diagnostic Metric: Compressing hard-to-track NC convergence into scalar dynamics**

Original NC1–NC3 metrics involve complex structures like feature means, Simplex ETF alignment, and self-duality, making them difficult for direct theoretical analysis or for identifying optimizer impact paths. The authors define NC0 as the squared norm of the row sum of the last-layer weight matrix:

$$\alpha_t = \frac{1}{K}\|W_t^\top \mathbf{1}\|_2^2$$

They prove (Proposition 2.1) that the validity of NC2 and NC3 necessarily implies NC0 converges to zero, making NC0 convergence a **necessary (not sufficient) condition** for NC. This reduces a high-dimensional geometric problem to the convergence of a single scalar: if NC0 diverges during training, it can be stated with certainty that NC cannot occur, without tracking full geometric structures.

**2. Coupling vs. Decoupling of Weight Decay: The true source of the Adam vs. AdamW difference**

The only substantive difference between the two types of optimizers lies in the position of the weight decay term. Coupled weight decay (SGD/Adam) incorporates the decay term inside the gradient: $V_{t+1} = \beta V_t + \nabla L_{CE} + \lambda W_t$. Decoupled weight decay (SGDW/AdamW) applies the decay directly to parameters: $W_{t+1} = (1-\eta\lambda)W_t - \eta V_{t+1}$. While these are equivalent for original SGD, they are **no longer equivalent** once the optimizer uses adaptive or signed coordinate-wise scaling (like Adam/SignGD), as the coupled term $\lambda W_t$ is scaled before updating, whereas the decoupled term bypasses scaling.

**3. NC0 Fates of Four Optimizer-Decay Combinations: Theoretically distinguishing who can produce NC**

The authors solve NC0 dynamics under the Unconstrained Feature Model (UFM). A key observation is that the row sum of the cross-entropy loss gradient is always zero: $\nabla L_{CE}(W_t)^\top \mathbf{1}_K = 0$. Thus, the evolution of NC0 is driven solely by weight decay and momentum, independent of the loss landscape. Theoretically, SGD with either coupled or decoupled decay (Theorem 3.1/3.2) leads to NC0 converging to zero exponentially. SignGD with coupled decay (Theorem 3.4, Adam case) shows a non-monotonic trajectory but eventually converges with learning rate decay. However, SignGD with decoupled decay (Theorem 3.3, AdamW case) causes NC0 to **monotonically increase** from zero to a positive constant $\frac{(K-2)^2}{\lambda^2}$, never reaching zero. This explains why AdamW fails to produce NC while Adam succeeds.

### Loss & Training

Experiments utilize Cross-Entropy loss with L2 regularization, training ResNet9 and VGG9 on MNIST, FashionMNIST, and CIFAR10. To systematically isolate optimizer effects, the authors scanned 108 combinations (3 learning rates × 6 momentum values × 6 weight decay values) for six optimizers: Adam, AdamW, SGD, SGDW, Signum, and SignumW. Training lasted 200 epochs with batch size 128, decaying the learning rate by 10x at 1/3 and 2/3 of training.

## Key Experimental Results

### Main Results

Final NC metrics for ResNet9 on FashionMNIST (lower is better):

| Optimizer | NC0↓ | NC1↓ | NC2↓ | NC3↓ |
|-----------|------|------|------|------|
| SGD | 2.14e-04 (<-99.5%) | 0.05 (-99.3%) | 0.29 (-63.0%) | 0.35 (-75.1%) |
| Adam | 0.34 (-80.6%) | 0.04 (-99.5%) | 0.29 (-63.9%) | 0.29 (-79.5%) |
| AdamW | **5.33 (>100%)** | 0.20 (-97.2%) | 0.54 (-32.4%) | 0.78 (-45.2%) |
| SGDW | 0.55 (-68.9%) | 0.26 (-96.3%) | 0.46 (-42.4%) | 0.80 (-43.5%) |

### Ablation Study

| Configuration | Key Indicator | Description |
|---------------|---------------|-------------|
| Adam vs AdamW Interpolation | NC0/NC2/NC3 improve smoothly as coupled WD increases | Validation accuracy remains largely unchanged |
| Momentum acceleration of NC | NC metrics are significantly lower with mom=0.9 than 0.7 at same loss | Momentum acceleration of NC exceeds its acceleration of training |
| Optimal NC3 Hyperparameters | SGD NC3=0.13, AdamW NC3=0.49 | SGD achieves the strongest NC across all optimizers |

### Key Findings

1. **Coupled weight decay is necessary for adaptive optimizers to produce NC**: NC metrics for AdamW/SignumW remain significantly higher than Adam/Signum even with much higher weight decay.
2. **Momentum accelerates NC beyond just convergence**: Two SGD runs with the same training loss but different momentum reach solutions with distinct geometric structures.
3. **SGD behavior is relatively insensitive to coupling/decoupling**: Differences between SGD and SGDW are minor, aligning with theory.
4. **Partial Neural Collapse**: AdamW can achieve optimal values for NC1 and NC2 while NC0 diverges and NC3 fails, indicating NC properties do not always emerge simultaneously.
5. **NC4 is redundant**: As long as training accuracy reaches ~100%, NC4 is satisfied regardless of other NC metrics.

## Highlights & Insights

- **Proposed the NC0 diagnostic metric**: Convergence to zero is a necessary condition for NC, making it easier to track and analyze than original metrics.
- **Challenged the universality of NC**: Proved that optimizer choice decisively determines whether NC emerges.
- **Revealed subtle neglected differences**: The seemingly minor difference in weight decay coupling between Adam and AdamW leads to vastly different representation geometries.
- **NC does not necessarily imply better generalization**: All optimizers achieved similar validation accuracies despite significant differences in NC strength.
- **Extensive experimentation**: Systematically controlled variables across 3,900+ training runs.

## Limitations & Future Work

1. **Simplified theoretical settings**: Theorem 3.3/3.4 are based on SignGD in UFM, which does not fully capture the complexity of deep networks and adaptive optimizers.
2. **Focus on NC0**: Full understanding of NC1-NC3 behavior under realistic optimization dynamics remains an open problem.
3. **Limited to the last layer**: NC properties in intermediate layers were not analyzed.
4. **Newer optimizers not covered**: The NC behavior of Lion, MARS, Shampoo, SOAP, Muon, and others remains to be explored.
5. **Need for larger models**: Experiments with larger architectures like ViT and DenseNet were limited (preliminary ViT results are in the appendix).

## Related Work & Insights

- Papyan et al. (2020) first discovered the NC phenomenon.
- Pan & Cao (2024) and Jacot et al. (2024) studied weight decay's impact on NC but did not distinguish between coupling and decoupling.
- Loshchilov & Hutter (2019) introduced AdamW, but its impact in the context of NC was previously overlooked.
- **Insight**: Optimizers do not just affect convergence speed; they decisively influence the geometric structure of learned representations as an implicit inductive bias.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First to reveal optimizer-dependent NC emergence and propose NC0)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (3,900+ runs, systematic control, multiple datasets/architectures)
- Writing Quality: ⭐⭐⭐⭐ (Tight integration of theory and experiments, clear structure)
- Value: ⭐⭐⭐⭐ (Important insights for understanding optimization and representation geometry)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[ICLR 2026\] Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear programming](constraint_matters_multi-modal_representation_for_reducing_mixed-integer_linear_.md)
- [\[ICLR 2026\] How does the optimizer implicitly bias the model merging loss landscape?](how_does_the_optimizer_implicitly_bias_the_model_merging_loss_landscape.md)
- [\[ICLR 2026\] Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension](towards_efficient_optimizer_design_for_llm_via_structured_fisher_approximation_w.md)
- [\[ICLR 2026\] LoRA meets Riemannion: Muon Optimizer for Parametrization-independent Low-Rank Adapters](lora_meets_riemannion_muon_optimizer_for_parametrization-independent_low-rank_ad.md)

</div>

<!-- RELATED:END -->
