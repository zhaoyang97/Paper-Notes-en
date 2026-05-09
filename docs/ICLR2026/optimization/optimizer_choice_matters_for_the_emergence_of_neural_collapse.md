---
title: >-
  [Paper Note] Optimizer Choice Matters for the Emergence of Neural Collapse
description: >-
  [ICLR 2026][Optimization][Neural Collapse] Through 3,900+ training experiments and theoretical analysis, this paper reveals that optimizer choice—particularly the coupling mechanism of weight decay—plays a decisive role in the emergence of Neural Collapse: AdamW (decoupled weight decay) fails to produce Neural Collapse, whereas SGD and Adam (coupled weight decay) succeed.
tags:
  - ICLR 2026
  - Optimization
  - Neural Collapse
  - Optimizer Choice
  - Weight Decay Coupling
  - AdamW vs Adam
  - Implicit Bias
date: 2026-05-08
content_hash: d0cefd8372ccfdc9
---

# Optimizer Choice Matters for the Emergence of Neural Collapse

**Conference**: ICLR 2026
**arXiv**: [2602.16642](https://arxiv.org/abs/2602.16642)
**Code**: N/A
**Area**: Optimization Theory / Deep Learning Theory
**Keywords**: Neural Collapse, Optimizer Choice, Weight Decay Coupling, AdamW vs Adam, Implicit Bias

## TL;DR

Through 3,900+ training experiments and theoretical analysis, this paper reveals that optimizer choice—particularly the coupling mechanism of weight decay—plays a decisive role in the emergence of Neural Collapse: AdamW (decoupled weight decay) fails to produce Neural Collapse, whereas SGD and Adam (coupled weight decay) succeed.

## Background & Motivation

Neural Collapse (NC) is a phenomenon observed by Papyan et al. (2020) in the terminal phase of deep network training, wherein last-layer feature vectors and classifier weights self-organize into highly symmetric geometric structures. NC comprises four properties:
- **NC1**: Within-class variability vanishes (features collapse to class means)
- **NC2**: Class means converge to a Simplex ETF (equiangular tight frame)
- **NC3**: Classifier weights align with class means (Self-Duality)
- **NC4**: Classification reduces to a nearest class-center classifier

Existing theoretical analyses largely neglect the role of the optimizer, implicitly assuming NC is universal across all optimization methods. This paper challenges that assumption and demonstrates that optimizer choice—especially the coupling scheme of weight decay—is critical to the emergence of NC. A key finding is that **Adam produces NC, whereas the algorithmically similar AdamW does not**.

## Method

### Overall Architecture

This paper investigates the effect of optimizers on NC at both theoretical and empirical levels:
1. Introducing a new diagnostic metric NC0
2. Providing theoretical analysis for SGD and SignGD (special cases of Adam/AdamW)
3. Validating the theory through large-scale experiments (3,900+ training runs)

### Key Designs

1. **NC0 Diagnostic Metric**:

    - Definition: $\alpha_t = \frac{1}{K}\|W_t^\top \mathbf{1}\|_2^2$, the squared norm of the row-sum of the last-layer weight matrix
    - Core property: Convergence of NC0 to zero is a **necessary condition** for NC2 and NC3 to hold (Proposition 2.1)
    - Advantage: More tractable for tracking and theoretical analysis than the original NC metrics; if NC0 diverges, the occurrence of NC can be conclusively ruled out

2. **Coupled vs. Decoupled Weight Decay**:

    - **Coupled weight decay** (e.g., SGD/Adam): $V_{t+1} = \beta V_t + \nabla L_{CE} + \lambda W_t$, decay term inside the gradient
    - **Decoupled weight decay** (e.g., SGDW/AdamW): $W_{t+1} = (1-\eta\lambda)W_t - \eta V_{t+1}$, decay term applied directly to parameters
    - For vanilla SGD the two are equivalent, but for adaptive optimizers (e.g., Adam) they are **not equivalent**

3. **Theoretical Theorems**:

    - **Theorem 3.1** (SGD + decoupled WD): NC0 converges to zero at an exponential rate proportional to $\lambda$
    - **Theorem 3.2** (SGD + coupled WD): NC0 converges to zero at an exponential rate proportional to $\lambda$ and $\beta$
    - **Theorem 3.3** (SignGD + decoupled WD, i.e., AdamW special case): NC0 **monotonically increases** to a positive constant $\frac{(K-2)^2}{\lambda^2}$ and **does not converge to zero**
    - **Theorem 3.4** (SignGD + coupled WD, i.e., Adam special case): Under a learning rate decay schedule, NC0 can converge to zero

   Key theoretical insight: The row-sum of the cross-entropy loss gradient satisfies $\nabla L_{CE}(W_t)^\top \mathbf{1}_K = 0$, so the dynamics of NC0 depend solely on weight decay and momentum.

### Loss & Training

- All experiments use cross-entropy loss with L2 regularization
- ResNet9 and VGG9 architectures
- MNIST, FashionMNIST, and CIFAR10 datasets
- 6 optimizers: Adam, AdamW, SGD, SGDW, Signum, SignumW
- 3 learning rates × 6 momentum values × 6 weight decay values = 108 hyperparameter combinations per optimizer
- 200 epochs, batch size 128, learning rate decayed by 10× at 1/3 and 2/3 of training

## Key Experimental Results

### Main Results

Final NC metrics of ResNet9 on FashionMNIST (lower is better):

| Optimizer | NC0↓ | NC1↓ | NC2↓ | NC3↓ |
|-----------|------|------|------|------|
| SGD | 2.14e-04 (<-99.5%) | 0.05 (-99.3%) | 0.29 (-63.0%) | 0.35 (-75.1%) |
| Adam | 0.34 (-80.6%) | 0.04 (-99.5%) | 0.29 (-63.9%) | 0.29 (-79.5%) |
| AdamW | **5.33 (>100%)** | 0.20 (-97.2%) | 0.54 (-32.4%) | 0.78 (-45.2%) |
| SGDW | 0.55 (-68.9%) | 0.26 (-96.3%) | 0.46 (-42.4%) | 0.80 (-43.5%) |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Adam vs AdamW interpolation | NC0/NC2/NC3 improve smoothly as coupled WD increases | Validation accuracy remains largely unchanged |
| Momentum effect on NC | At equal training loss, mom=0.9 yields substantially lower NC metrics than 0.7 | Momentum accelerates NC beyond its effect on training loss |
| Best NC3 hyperparameters | SGD NC3=0.13, AdamW NC3=0.49 | SGD achieves the strongest NC among all optimizers |

### Key Findings

1. **Coupled weight decay is a necessary condition for adaptive optimizers to produce NC**: NC metrics of AdamW/SignumW consistently remain far above those of Adam/Signum, and increasing weight decay by orders of magnitude does not remedy this
2. **Momentum accelerates NC but not merely by accelerating convergence**: Two SGD runs reaching the same training loss with different momentum values arrive at solutions with markedly different geometric structures
3. **SGD's NC behavior is insensitive to coupling/decoupling**: The NC metric gap between SGD and SGDW is small, consistent with theory
4. **Partial Neural Collapse**: AdamW can achieve optimal NC1 and NC2 values while NC0 diverges and NC3 is not satisfied—NC properties need not emerge simultaneously
5. **NC4 is redundant**: Whenever training accuracy approaches 100%, NC4 is always satisfied and is uncorrelated with other NC metrics

## Highlights & Insights

- **Proposes NC0 as a new diagnostic metric**: Convergence to zero is a necessary condition for NC and is more tractable than the original metrics
- **Challenges the universality assumption of NC**: Demonstrates that optimizer choice decisively determines whether NC emerges
- **Reveals an overlooked subtle distinction**: The seemingly minor difference in weight decay coupling between Adam and AdamW leads to drastically different representational geometry
- **NC does not necessarily imply better generalization**: All optimizers achieve comparable validation accuracy, yet NC strength varies substantially—limiting the utility of NC as a lens for understanding generalization
- **Large-scale experimental rigor**: 3,900+ training runs with systematic control of variables

## Limitations & Future Work

1. **Theoretical analysis is restricted to simplified settings**: Theorems 3.3/3.4 are based on SignGD in the Unconstrained Feature Model (UFM), and do not fully capture the complexity of deep networks and adaptive optimizers
2. **Only NC0 is analyzed**: Fully characterizing the behavior of NC1–NC3 under realistic optimization dynamics remains an open problem
3. **Restricted to the last layer**: NC properties of intermediate layers are not analyzed (prior work suggests NC may also emerge in intermediate layers)
4. **Novel optimizers not covered**: The NC behavior of Lion, MARS, Shampoo, SOAP, Muon, and similar optimizers remains to be explored
5. **Needs extension to larger models**: Experiments on larger architectures such as ViT and DenseNet are limited (preliminary ViT results are provided in the appendix)

## Related Work & Insights

- Papyan et al. (2020) first identified the NC phenomenon
- Pan & Cao (2024) and Jacot et al. (2024) studied the effect of weight decay on NC without distinguishing between coupled and decoupled variants
- Loshchilov & Hutter (2019) proposed AdamW, but its implications in the NC context had previously been overlooked
- **Implication**: Optimizers not only affect convergence speed but also decisively shape the geometry of learned representations—optimizer choice constitutes a form of implicit inductive bias

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First to reveal optimizer-dependent NC emergence; proposes the NC0 metric)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (3,900+ runs, systematic variable control, multiple datasets and architectures)
- Writing Quality: ⭐⭐⭐⭐ (Theory and experiments tightly integrated; clear structure)
- Value: ⭐⭐⭐⭐ (Significant implications for understanding deep learning optimization and representational geometry)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rolling Ball Optimizer: Learning by Ironing Out Loss Landscape Wrinkles](rolling_ball_optimizer_learning_by_ironing_out_loss_landscape_wrinkles.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[ICLR 2026\] Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear Programming](constraint_matters_multi-modal_representation_for_reducing_mixed-integer_linear_.md)
- [\[ICLR 2026\] RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)
- [\[NeurIPS 2025\] PROFIT: A Specialized Optimizer for Deep Fine Tuning](../../NeurIPS2025/optimization/profit_a_specialized_optimizer_for_deep_fine_tuning.md)

</div>

<!-- RELATED:END -->
