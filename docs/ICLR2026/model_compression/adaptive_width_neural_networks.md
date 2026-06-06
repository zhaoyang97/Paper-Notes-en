---
title: >-
  [Paper Note] Adaptive Width Neural Networks
description: >-
  [ICLR 2026][Model Compression][adaptive width] This paper proposes the AWN framework, which automatically learns the unbounded width (number of neurons) of each layer during training via variational inference. A monotoni…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "adaptive width"
  - "variational inference"
  - "neuron importance ordering"
  - "network compression"
  - "hyperparameter learning"
date: 2026-05-08
content_hash: 5fa7bfe2776aa810
---

# Adaptive Width Neural Networks

**Conference**: ICLR 2026
**arXiv**: [2501.15889](https://arxiv.org/abs/2501.15889)  
**Code**: [https://github.com/nec-research/Adaptive-Width-Neural-Networks](https://github.com/nec-research/Adaptive-Width-Neural-Networks)  
**Area**: Model Compression / Neural Architecture Learning
**Keywords**: adaptive width, variational inference, neuron importance ordering, network compression, hyperparameter learning

## TL;DR
This paper proposes the AWN framework, which automatically learns the unbounded width (number of neurons) of each layer during training via variational inference. A monotonically decreasing importance function imposes a soft ordering on neurons, enabling width to adapt to task difficulty and supporting zero-cost post-training truncation for compression.

## Background & Motivation

**Background**: For nearly 70 years, layer widths in neural networks have relied on manual selection or hyperparameter search (grid search/NAS), remaining a fundamental open problem in deep learning.

**Limitations of Prior Work**: The search space for width as a hyperparameter grows exponentially with depth; in practice, the simplified strategy of using the same width for all layers is commonly adopted. For foundation models with billions of parameters, the computational cost of hyperparameter tuning is entirely prohibitive.

**Key Challenge**: Networks need to be "wide enough" to learn good representations, yet "too wide" wastes resources. Existing methods either search over a fixed width space (NAS) or require separate training-pruning pipelines (pruning/distillation).

**Goal**: Can each layer's width automatically grow or shrink via gradient descent in a single training run, without requiring a predefined upper bound?

**Key Insight**: A latent variable $\lambda_\ell$ is introduced to control the truncation width of each layer, and a monotonically decreasing importance distribution imposes an ordering on neurons—lower-index neurons are more important, higher-index neurons less so, and newly added neurons naturally occupy low-importance positions.

**Core Idea**: Width learning is formalized as a variational inference problem, jointly optimizing width parameters and network weights through an ELBO objective.

## Method

### Overall Architecture
AWN (Adaptive Width Networks) introduces two sets of latent variables per layer: $\lambda_\ell$ (controlling width) and $\theta_\ell$ (network weights). Variational inference is used to maximize the ELBO, dynamically adjusting the number of neurons $D_\ell$ per layer during training. Width changes are implemented via standard backpropagation without specialized optimizers.

### Key Designs

1. **Probabilistic Graphical Model and Variational Objective**

    - Function: Models width learning as a probabilistic inference problem.
    - Mechanism: An infinite sequence of i.i.d. latent variables $\theta_{\ell n}$ (weights of the $n$-th neuron in layer $\ell$) is assumed, and a latent variable $\lambda_\ell$ determines the effective width $D_\ell$ via the quantile function of distribution $f_\ell$. The variational distribution $q(\lambda, \theta)$ causes neurons beyond $D_\ell$ to revert to the prior. The resulting ELBO contains three terms: width regularization $\log \frac{p(\nu_\ell)}{q(\nu_\ell)}$, weight regularization $\sum_{n=1}^{D_\ell} \log \frac{p(\rho_{\ell n})}{q(\rho_{\ell n})}$, and predictive performance $\sum_i \log p(y_i | \nu, \rho, x_i)$.
    - Design Motivation: The probabilistic framework naturally provides regularization (via priors) and uncertainty quantification, and the variational parameters $\nu_\ell, \rho_{\ell n}$ can be directly optimized as network parameters via gradient descent.

2. **Soft Neuron Importance Ordering**

    - Function: Rescales each neuron's activation using a monotonically decreasing function $f_\ell(j; \nu_\ell)$.
    - Mechanism: The standard MLP activation is modified to $h_j^\ell = \sigma(\sum_k w_{jk}^\ell h_k^{\ell-1}) \cdot f_\ell(j; \nu_\ell)$, where $f_\ell$ adopts a discretized exponential distribution. Lower-index neurons receive larger $f_\ell$ values (more important); higher-index neurons receive smaller values (less important); newly added neurons automatically acquire low importance.
    - Design Motivation: (a) Breaks the permutation symmetry of weight matrices, eliminating the "jostling" effect at the onset of training; (b) newly added neurons do not abruptly perturb network outputs; (c) naturally supports post-training truncation—removing trailing neurons has minimal impact. Bounded activations (ReLU6/tanh) are required to prevent subsequent layers from compensating for the rescaling factors.

3. **Kaiming+ Initialization for Deep AWN (Theorem 3.1)**

    - Function: Derives a weight initialization scheme that accounts for importance rescaling.
    - Mechanism: Requires $\text{Var}[w_{jk}^\ell] = \frac{2}{\sum_{j=1}^{D_{\ell-1}} f_\ell^2(j)}$, keeping the variance of deep-layer activations constant at initialization. The difference from standard Kaiming initialization is that the denominator changes from $D_{\ell-1}$ to $\sum_j f_\ell^2(j) < D_{\ell-1}$.
    - Design Motivation: Without this adjustment, the rescaling factors cause deep-layer activations to rapidly decay to zero, leading to vanishing gradients and training failure.

### Loss & Training
- At each training step, each layer's width $D_\ell$ is first updated via the quantile function, followed by standard forward and backward passes.
- When width increases, new neuron weights are initialized from a standard normal distribution; when width decreases, excess weights are discarded.
- In mini-batch training, the prediction loss is scaled by $N/M$ (from a Bayesian perspective, regularization should weaken as data increases).
- Bounded activation functions (ReLU6) are recommended to ensure width convergence.

## Key Experimental Results

### Main Results
Comprehensive evaluation on tabular, image, text, sequential, and graph data:

| Model/Dataset | Fixed Acc/Loss | AWN Acc/Loss | Fixed Width | AWN Learned Width |
|---|---|---|---|---|
| MLP/DoubleMoon | 100.0 | 100.0 | 8 | 8.1 |
| MLP/Spiral | 99.5 | 99.8 | 16 | 65.9 |
| MLP/SpiralHard | 98.0 | **100.0** | 32 | 227.4 |
| ResNet-20/CIFAR10 | 91.4 | 91.4 | Linear | 80.1 |
| RNN/PMNIST | 91.1 | **95.7** | 24 | 806.3 |
| GIN/REDDIT-B | 87.0 | **90.2** | 96–320 | 793.6 |
| Transformer/Multi30k | 1.43 | 1.51 | 24576 | **123.2** (200× fewer) |

### Ablation Study

| Importance Function Family | Avg. Accuracy | Max Accuracy | Learned Width |
|---|---|---|---|
| Exponential | 80.27 | 100.00 | 954 |
| Power Law | 81.82 | 100.00 | 2952 (heavy tail) |
| Sigmoidal | 76.85 | 100.00 | 427 (sharp transition) |

### Key Findings
- **Width adapts to task difficulty**: DoubleMoon→Spiral→SpiralHard yields learned widths of 8→66→227, perfectly aligned with intuition.
- **Post-training truncation**: On the Spiral dataset, a learned 83-unit MLP can be truncated by 30% (~58 units) with no accuracy loss, followed by graceful degradation—a zero-cost form of "distillation."
- **Online compression**: On SpiralHard, introducing an informative prior after 1,000 iterations reduces width from 800 to 300 (−62%) with no accuracy loss.
- **200× Transformer compression**: On the Multi30k translation task, the FFN learns a width of only 123 (vs. fixed 24,576), with only a marginal loss increase of 0.08.
- Starting width does not affect the final converged width under bounded activations, demonstrating that the hyperparameter space is genuinely reduced.

## Highlights & Insights
- **Elegance of the probabilistic formulation**: Width learning is fully integrated into the standard variational inference framework without heuristic rules; neurons can be added or removed purely via backpropagation—a paradigm shift from "architecture search" to "parameter learning."
- The **soft ordering → free truncation** side effect is highly practical: trained networks can be compressed simply by removing trailing rows/columns, far simpler than pruning.
- While technically straightforward, **Kaiming+ initialization** is critical for the trainability of deep AWN, highlighting that dynamic architecture methods require special attention to initialization.
- The 200× compression result on Transformers suggests substantial redundancy may exist in LLM feed-forward networks.

## Limitations & Future Work
- The framework currently learns **widths of MLP layers only**; extending to the number of filters in CNNs requires a different formulation (explicitly noted as beyond scope by the authors).
- AWN occasionally fails to converge on CIFAR-100 (avg. 63.1 vs. fixed 66.5), and training stability needs improvement.
- The choice of exponential distribution as the default $f_\ell$ lacks theoretical optimality support; different distribution families yield substantially different widths (exponential 954 vs. power law 2952).
- Validation on truly large-scale models (e.g., LLMs) is absent; the Transformer experiments are limited to the small-scale Multi30k translation task.
- The first-order approximation in variational inference ($\mathbb{E}[f(\lambda)] \approx f(\nu)$) discards uncertainty information, weakening the advantages of the Bayesian framework.

## Related Work & Insights
- **vs. NAS**: NAS searches over discrete architecture spaces and requires multiple training runs; AWN operates in a continuous space with a single training run, but covers only the width dimension.
- **vs. Pruning**: Pruning requires pre-training a large model before compression; AWN learns appropriate widths from the start and supports both growth and shrinkage simultaneously.
- **vs. Unbounded Depth Network (Nazaret & Blei 2022)**: The same conceptual approach extended to the width dimension, but requires a monotonically decreasing importance distribution (not needed for depth).
- **vs. Firefly (Wu et al. 2020)**: Firefly alternates between training and growth and relies on heuristic rules; AWN is purely gradient-driven.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formalizing width learning as variational inference with support for unbounded growth is conceptually elegant and theoretically rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers five data domains with thorough ablation analysis, but lacks large-scale validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous, analytically thorough, and logically well-structured.
- Value: ⭐⭐⭐⭐ Conceptually compelling, but practical feasibility at scale remains to be demonstrated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](a_recovery_guarantee_for_sparse_neural_networks.md)
- [\[ICLR 2026\] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](fine-tuning_quantized_neural_networks_with_zeroth-order_optimization.md)
- [\[ICML 2026\] SURGE: Surrogate Gradient Adaptation in Binary Neural Networks](../../ICML2026/model_compression/surge_surrogate_gradient_adaptation_in_binary_neural_networks.md)
- [\[NeurIPS 2025\] The Graphon Limit Hypothesis: Understanding Neural Network Pruning via Infinite Width Analysis](../../NeurIPS2025/model_compression/the_graphon_limit_hypothesis_understanding_neural_network_pruning_via_infinite_w.md)
- [\[ICML 2026\] Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation](../../ICML2026/model_compression/partial_fusion_of_neural_networks_efficient_tradeoffs_between_ensembles_and_weig.md)

</div>

<!-- RELATED:END -->
