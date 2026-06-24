---
title: >-
  [Paper Note] The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions
description: >-
  [Optimization] Through the "spawn-and-perturb" experimental paradigm, the sensitivity of neural network training trajectories to initial conditions is systematically investigated, showing that infinitesimal perturbations (even to a single weight) at the very beginning of training can lead to entirely different convergence outcomes—the "butterfly effect". This instability is independent of training noise and rapidly decays as training progresses.
tags:
  - "Optimization"
date: 2026-05-08
content_hash: ee4809303a75817b
---

# The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions

- **Conference**: ICML 2025
- **arXiv**: [2506.13234](https://arxiv.org/abs/2506.13234)
- **Code**: [gsaltintas/lmc](https://github.com/gsaltintas/lmc)
- **Area**: Training Dynamics / Optimization Theory / Loss Landscapes
- **Keywords**: Butterfly Effect, Training Stability, Loss Barrier, Linear Mode Connectivity, Model Merging, Perturbation Experiments

## TL;DR

Through the "spawn-and-perturb" experimental paradigm, the sensitivity of neural network training trajectories to initial conditions is systematically investigated, showing that infinitesimal perturbations (even to a single weight) at the very beginning of training can lead to entirely different convergence outcomes—the "butterfly effect". This instability is independent of training noise and rapidly decays as training progresses.

## Background & Motivation

Neural network training is known to be sensitive to initialization and SGD stochasticity. However, a key open question is: **to what extent does this sensitivity lead to substantially different networks**—both in terms of weights and the learned functions?

Existing research divides training into "chaotic" (early) and "stable" (late) phases, but suffers from the following limitations:
1. Prior work only measured instability caused by training noise (batch sampling, data augmentation, GPU non-determinism).
2. It is impossible to distinguish whether instability originates from noise or the training process itself.
3. It fails to precisely quantify how large of a perturbation is required at different locations in the loss landscape to cause divergence.

From the perspective of **deterministic dynamical systems**, this paper eliminates all training noise through controlled perturbations to accurately characterize when and to what extent training is sensitive to initial conditions. This has direct guiding significance for practical scenarios such as model merging, fine-tuning, and ensemble learning.

## Method

### Overall Architecture: Spawn-And-Perturb Experiment

1. Select initial parameters $\theta_0$ and train to the spawn time $t$ to obtain $\theta_t = \mathcal{T}^t(\theta_0; \xi_{1:t})$.
2. Clone two copies of the network and add a perturbation of size $\sigma$ to one of them: $\theta'_t = \theta_t + \sigma \varepsilon$.
3. Train both networks to convergence **using exactly the same training noise** $\xi_{t:T}$.
4. Measure the divergence $d(\theta_T, \theta'_T)$.

The key difference from the method of Frankle et al. (2020) is that the latter uses independent training noise starting from time $t$, whereas this paper eliminates all differences in training noise to isolate only the effect of the perturbation itself.

### Perturbation Types

**Batch Perturbations** (along the training direction):

$$\hat{\varepsilon}_{\text{Batch}} = \frac{1}{n} \sum_{i=1}^{b} \nabla \ell(x_i, y_i; \theta_t)$$

**Gaussian Perturbations** (isotropic):

$$\hat{\varepsilon}_{\text{Gaussian}} = [\varepsilon_i^{(l)}], \quad \varepsilon_i^{(l)} \sim \mathcal{N}\left(0, \frac{2}{n_{l-1}}\right)$$

Both types of perturbations are normalized by the initialization scale to ensure they do not disproportionately affect certain layers.

### Functional Similarity Evaluation Metrics

1. **$L^2$ distance**: The Euclidean distance between weight vectors in the parameter space $\|\theta_T - \theta'_T\|_2$.
2. **Loss Barrier**: The maximum loss increment along the linear interpolation path:

$$\sup_{\alpha \in (0,1)} \ell(\alpha \theta_T + (1-\alpha)\theta'_T) - \alpha \ell(\theta_T) - (1-\alpha)\ell(\theta'_T)$$

3. **Barrier after permutation alignment**: Calculate the barrier between $\theta_T$ and $P[\theta'_T]$ after finding the permutation $P$ using weight matching algorithms.
4. **Representation similarity**: Angular CKA is employed to measure the similarity of the penultimate layer representations.

## Key Experimental Results

### Key Finding 1: The Butterfly Effect

| Perturbation Setup | ResNet-20 on CIFAR-10 | |
|---|---|---|
| Perturbing 1 weight at initialization | Generates a significant loss barrier | |
| Perturbing 0.01% of weights at initialization | Generates a large barrier | |
| Same perturbation after 0.5% of training progress | Barrier decreases significantly | |
| Perturbation after 50% of training progress | Only very large perturbations (10% of initialization scale) generate a non-zero barrier | |

**Key Conclusion**: The initial phase of training (approximately the first 0.5% of steps) is extremely sensitive; altering a single weight is sufficient to cause convergence to different loss basins.

### Key Finding 2: Permutation is Not the Root Cause

Comparing the barriers before and after permutation alignment reveals that **permutation matching cannot reduce the barrier**. This indicates that training instability produces true functional differences, rather than mere permutation transformations of equivalent weights.

### Key Finding 3: Influence of Hyperparameters

| Setup | Impact on Stability |
|---|---|
| Wider/shallower networks (ResNet-8) | Most stable |
| 10x learning rate warmup | Significantly improves stability |
| 4x batch size | Slightly improves stability |
| Adam optimizer | Reduces stability |
| Weight decay | Reduces stability |
| Wide architecture + long warmup | Further improves but does not eliminate the initialization barrier |

### Key Finding 4: Stability in Pre-training and Fine-tuning

| Model | Scenario | Key Finding |
|---|---|---|
| ResNet-50 | CIFAR-100→10 | More stable than CIFAR-10→100; longer pre-training improves stability |
| Multi-BERT | GLUE tasks | Longer pre-training does not necessarily increase stability; 2000k checkpoint is instead the most unstable on QNLI/RTE |
| OLMo | GSM8K | Longer pre-training can also reduce fine-tuning stability |

### Key Finding 5: Relationship between $L^2$ and the Barrier

- In vision models (ResNet), **the barrier exhibits a strong log-linear relationship with the $L^2$ distance**.
- In language models (BERT fine-tuning), **there is almost no correlation between the two**.
- The growth of $L^2$ and barriers **does not conform to the exponential growth predicted by linearized dynamical systems**.

### Ensemble Performance Experiments

Angular CKA dissimilarity is positively correlated with ensemble performance—intentional perturbations can increase model diversity and improve ensemble performance. However, ViT on CIFAR-100 does not follow this trend.

## Highlights & Insights

1. **Deep insights from minimalist experimental designs**: By eliminating training noise to isolate the effect of perturbations, the experimental design is highly elegant.
2. **"One weight is enough"**: The finding that initialization perturbations of a single weight can lead to divergence is highly shocking.
3. **Clear practical values**: 
    - Model merging: There is a need to ensure networks emerge from the same training basin (avoiding early divergence).
    - Ensemble learning: Diversity can be increased through intentional perturbations.
    - Fine-tuning: Longer pre-training does not necessarily guarantee stability; attention must be paid to fragility caused by overfitting.
4. **Fundamental differences between language and vision models**: The $L^2$-barrier relationship behaves entirely differently in the two domains, suggesting structural differences in their loss landscapes.

## Limitations & Future Work

1. **Lack of theoretical explanation**: While the butterfly effect is discovered, a theoretical analysis of why it exists is not provided.
2. **$L^2$/barrier growth rates do not match dynamical system predictions**: This indicates that linearized dynamical system models are not applicable to neural network training, but no alternative theory is proposed.
3. **Limited experimental scale**: OLMo is the largest model but only underwent preliminary experiments; systematic validation on larger LLMs is lacking.
4. **Inconsistent trends**: For example, the impact of longer pre-training on stability varies by task, and a unified explanation is lacking.
5. **Limitations of permutation alignment methods**: It is impossible to rule out the possibility that better permutation algorithms could reduce the barrier.

## Related Work & Insights

- **Training Stability and Optimization**: Edge of Stability (Cohen et al., 2021), SGD noise analysis (Wu et al., 2018)
- **Linear Mode Connectivity**: LMC research derived from the Lottery Ticket Hypothesis (Frankle et al., 2020a)
- **Model Merging/Ensembles**: Git Re-Basin (Ainsworth et al., 2023), Weight averaging (Wortsman et al., 2021)
- **Spawning Experiments**: Training noise stability experiments from Frankle et al. (2020) and Fort et al. (2020)
- **Fine-tuning Stability**: Juneja et al. (2023) discovered instability in language model fine-tuning

## Rating

⭐⭐⭐⭐⭐ (5/5)

This is an outstanding empirical study. The experimental design is meticulous (eliminating noise to isolate disturbances), the findings are startling (single-weight butterfly effect), the coverage is comprehensive (vision/language, pre-training/fine-tuning, multiple architectures), and the practical value is clear (guidelines for model merging and ensembles). Although it lacks theoretical explanation, it significantly advances the empirical understanding of training dynamics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg](widening_the_network_mitigates_the_impact_of_data_heterogeneity_on_fedavg.md)
- [\[ICML 2025\] Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation](sparse_causal_discovery_with_generative_intervention_for_unsupervised_graph_doma.md)
- [\[ICCV 2025\] Federated Continual Instruction Tuning](../../ICCV2025/optimization/federated_continual_instruction_tuning.md)
- [\[CVPR 2025\] Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression](../../CVPR2025/optimization/automatic_joint_structured_pruning_and_quantization_for_efficient_neural_network.md)
- [\[ICML 2025\] Interior-Point Vanishing Problem in Semidefinite Relaxations for Neural Network Verification](interior-point_vanishing_problem_in_semidefinite_relaxations_for_neural_network_.md)

</div>

<!-- RELATED:END -->
