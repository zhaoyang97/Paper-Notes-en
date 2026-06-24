---
title: >-
  [Paper Note] Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg
description: >-
  [Optimization] Based on Neural Tangent Kernel (NTK) theory, this work proves that the upper bound of model divergence caused by data heterogeneity in FedAvg is $\mathcal{O}(n^{-1/2})$ (where $n$ represents the network width). In the infinite-width limit, both global and local models linearized. Consequently, FedAvg with the same number of iterations is equivalent to centralized gradient descent, achieving identical generalization performance.
tags:
  - "Optimization"
date: 2026-05-08
content_hash: b10e8729e85b845f
---

# Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg

- **Conference**: ICML 2025
- **arXiv**: [2508.12576](https://arxiv.org/abs/2508.12576)
- **Code**: [kkhuge/ICML2025](https://github.com/kkhuge/ICML2025)
- **Area**: Optimization
- **Keywords**: FedAvg, Data Heterogeneity, Network Width, Neural Tangent Kernel, Overparameterization, Model Divergence, Federated Learning Convergence

## TL;DR

Based on Neural Tangent Kernel (NTK) theory, this work proves that the upper bound of model divergence caused by data heterogeneity in FedAvg is $\mathcal{O}(n^{-1/2})$ (where $n$ represents the network width). In the infinite-width limit, both global and local models linearized. Consequently, FedAvg with the same number of iterations is equivalent to centralized gradient descent, achieving identical generalization performance.

## Background & Motivation

Federated Learning (FL) enables decentralized clients to collaboratively train models without sharing their local data. A core challenge is the **non-independent and identically distributed (non-IID)** nature of client data. Distribution shifts—caused by user behaviors, geographical differences, and device-specific patterns—lead to divergences in local optimization directions, thereby degrading the convergence and generalization capabilities of the global model.

**Limitations of Prior Work**:
1. Many convergence analyses rely on **strict assumptions**, such as convex loss functions, bounded gradient similarity, and bounded gradients, which are difficult to satisfy in practice.
2. Methods like FedProx and SCAFFOLD require complex hyperparameter tuning or constraint relaxation.
3. Research on overparameterized FL (e.g., FL-NTK) is mostly restricted to two-layer networks, limiting model capacity.
4. Song et al. (2023) only focused on the convergence of training loss, without analyzing generalization performance.

**Core Problem**: Can increasing the network width fundamentally mitigate the impact of data heterogeneity in FL?

## Method

### Overall Architecture

Consider a standard FL setup with $M$ clients, an $L$-layer fully connected network, MSE loss, and FedAvg aggregation. Each client performs $\tau$ local GD iterations before uploading parameters for aggregation.

**Model Divergence Metric**: The metric used to quantify the impact of data heterogeneity is

$$\sum_{i=1}^{M} p_i \|\Delta \theta_i^{(t+1)\tau}\|_2 = \sum_{i=1}^{M} p_i \|\theta_i^{t\tau+\tau} - \theta^{(t+1)\tau}\|_2$$

Under IID data, this value approaches zero, whereas under non-IID data, it increases as heterogeneity grows.

### Key Theoretical Result 1: Model Divergence Bound (Theorem 1)

**Assumptions** (all standard assumptions for overparameterization analysis):
- The minimum width $n$ is sufficiently large.
- The analytical NTK matrix $\Theta$ is of full rank (with minimum eigenvalue $\lambda_m > 0$).
- The norm of input data is bounded, $\|x\|_2 \leq 1$.
- The activation function is Lipschitz continuous.

**Core Inequality**: Upper bound of model divergence

$$\sum_{i=1}^{M} p_i \|\Delta \theta_i^{(t+1)\tau}\|_2 \leq \zeta \triangleq \frac{2\eta_0 \tau C R_0}{\sqrt{n}(1-q)}$$

where $q = 1 - \frac{\eta_0 \tau \lambda_m}{3|\mathcal{D}|} + \frac{\eta_0^2 \tau^2 C^4}{2} e^{\eta_0 \tau C^2}$.

**Key Implications**:
- $\zeta = \mathcal{O}(n^{-1/2})$: Increasing the width directly reduces the impact of heterogeneity.
- As $n \to \infty$, $\zeta \to 0$: The impact of heterogeneity completely vanishes, and the convergence recovers a linear rate.

**Training Error Convergence**:

$$\|g(\theta^{t\tau})\|_2 \leq q^t R_0 + \frac{2\eta_0 \tau C C_1 R_0 \zeta (1-q^t)}{(1-q)^2}$$

$\zeta > 0$ makes the convergence no longer strictly linear, but as $n \to \infty$, it recovers linear convergence and the training error approaches zero.

**NTK Stability**: The variation of both global and local NTK is of order $\mathcal{O}(n^{-1/2})$. As the width approaches infinity, the NTK remains constant, extending the lazy training phenomenon in centralized learning to FL.

### Key Theoretical Result 2: Linearization and Equivalence (Theorem 2 & 3)

**Theorem 2**: As $n \to \infty$, both global and local models can be approximated by a linear model via first-order Taylor expansion:

$$\sup_{t \geq 0} \|f^{\text{lin}}(\theta^{t\tau}) - f(\theta^{t\tau})\|_2 = \mathcal{O}(n^{-1/2})$$

**Theorem 3**: Infinite-width FedAvg has closed-form solutions for both global parameters and outputs:

$$\theta^{t\tau} = -\frac{1}{n} J(\theta^0)(\Theta^0)^{-1}\left(I - e^{-\frac{\eta_0 t\tau}{|\mathcal{D}|}\Theta^0}\right)g(\theta^0) + \theta^0$$

**Equivalence Conclusion** (Eq. 42): When the total iterations of centralized GD is $t' = t\tau$:

$$\theta_{\text{cen}}^{t'} = \theta^{t\tau}, \quad f(x, \theta_{\text{cen}}^{t'}) = f(x, \theta^{t\tau})$$

That is, **infinite-width FedAvg and centralized GD yield exactly identical model parameters and outputs**, leading to equivalent generalization performance.

### Proof Framework

Theorem 1 is proved using **mathematical induction**:
1. Prove the Lipschitz continuity of global and local Jacobians.
2. Approximate GD as a gradient flow (since the learning rate $\eta = \eta_0/n$ is extremely small).
3. Utilize the mean value theorem for integrals to establish recursive relations for local model parameter changes.
4. Establish recursive inequality for global errors via Taylor expansions.

## Key Experimental Results

### Main Results: Network Width vs. Non-IID Impact

| Model Family | Width Factor $k$ | IID $\to$ Non-IID Accuracy Drop |
|--------|-------------|---------------------|
| FNN1 | 1 | -17.4% |
| FNN2 | 2 | -9.5% |
| FNN4 | 4 | -6.3% |
| FNN16 | 16 | **-2.0%** |
| CNN1 | 1 | -44.9% |
| CNN2 | 2 | -26.7% |
| CNN8 | 8 | -5.1% |
| CNN32 | 32 | **-2.4%** |
| ResNet1 | 1 | -44.6% |
| ResNet4 | 4 | -18.7% |
| ResNet16 | 16 | **-14.8%** |

**Key Conclusion**: The wider the network, the smaller the impact of non-IID data. The convergence curves of FNN32 and CNN32 on IID and non-IID datasets almost overlap.

### NTK and Parameter Stability Verification

Experiments on mini-MNIST using FNNs trained with GD+MSE verify that:
- As the network gets wider, the change in global and local NTKs becomes smaller.
- The update magnitude of model parameters becomes smaller (showing lazy training behavior).
- The outputs of global/local models for FNN512 are almost identical to those of the corresponding linear models.

### FedAvg vs. Centralized Learning

On mini-CIFAR-10, with $\tau=2$ and $\tau=5$, the training/test loss of FedAvg and centralized learning almost completely overlap, verifying the equivalence established in Theorem 3.

## Highlights & Insights

1. **First to establish a quantitative relation between network width and heterogeneity impact**: The divergence bound of $\mathcal{O}(n^{-1/2})$ provides clear and actionable theoretical guidance.
2. **Extending NTK theory from centralized learning to FL**: Proving that both global and local NTKs remain constant in wide networks.
3. **The elegance of "assumption-free" formulation**: Eliminates reliance on restrictive assumptions commonly used, such as convexity and bounded gradients.
4. **Clear practical implications**:
    - When facing severely heterogeneous data, **widening the network** is a simple and effective mitigation strategy.
    - In the infinite-width limit, FL clients can transmit only model outputs instead of parameters, drastically reducing communication overhead.
5. **Cross-architecture consistency**: FNN, CNN, and ResNet all validate the theoretical predictions.

## Limitations & Future Work

1. **Theory relies on the infinite-width assumption**: Practical networks have finite width; the theoretical results are asymptotic, and gaps may be significant at finite widths.
2. **Limited to MSE loss and GD**: The theoretical proofs require MSE loss and GD, which do not cover the commonly used cross-entropy loss and SGD in practice (although SGD+CE was validated in experiments to show consistent trends).
3. **Learning rate constraint**: $\eta = \eta_0/n$ is extremely small in wide networks, on which the accuracy of the gradient flow approximation relies.
4. **Limited experimental scale**: Evaluated only on MNIST and CIFAR-10, lacking large-scale, real-world FL scenarios.
5. **Overlooked communication efficiency**: Wider networks entail more parameters to transmit, which may offset the convergence advantages gained from network width.
6. **Full participation assumption**: It assumes all clients participate in every round, omitting partial participation scenarios.

## Related Work & Insights

- **Data Heterogeneity in FL**: FedProx (Li et al., 2020), SCAFFOLD (Karimireddy et al., 2020), FedNova (Wang et al., 2020b)
- **Overparameterized FL**: FL-NTK (Huang et al., 2021, limited to two-layer), Song et al. (2023, linear convergence to zero training loss but without generalization analysis)
- **NTK Theory**: Jacot et al. (2018), Lee et al. (2019) linearization theory in wide networks
- **Communication-Efficient FL**: FedMA (Wang et al., 2020a) layer-wise matching aggregation

## Rating

⭐⭐⭐⭐ (4/5)

The theoretical contribution is outstanding—this work is the first to quantitatively establish the relationship between network width and heterogeneity, and the proof of infinite-width equivalence is both elegant and meaningful. However, a significant gap exists between theory and practice (due to the infinite-width assumption, MSE+GD limitations, and vanishing learning rate), and the scale of experiments is small. It holds great value for the theoretical community, but its practical guidance ("making the network wider") remains relatively limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions](the_butterfly_effect_neural_network_training_trajectories_are_highly_sensitive_t.md)
- [\[ICML 2025\] Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation](sparse_causal_discovery_with_generative_intervention_for_unsupervised_graph_doma.md)
- [\[ICCV 2025\] Federated Continual Instruction Tuning](../../ICCV2025/optimization/federated_continual_instruction_tuning.md)
- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](../../AAAI2026/optimization/data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](../../NeurIPS2025/optimization/efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)

</div>

<!-- RELATED:END -->
