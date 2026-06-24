---
title: >-
  [Paper Note] Understanding the Statistical Accuracy-Communication Trade-off in Personalized Federated Learning with Minimax Guarantees
description: >-
  [ICML2025][Optimization][Personalized Federated Learning] This paper provides the first quantitative characterization of how the personalization parameter $\lambda$ simultaneously affects statistical accuracy and communication efficiency in personalized federated learning, establishes the minimax optimal statistical rate, and proposes the FedCLUP algorithm to achieve the optimal statistical-communication trade-off.
tags:
  - "ICML2025"
  - "Optimization"
  - "Personalized Federated Learning"
  - "Statistical Accuracy-Communication Trade-off"
  - "Minimax Optimality"
  - "Bi-level Optimization"
  - "Regularization"
date: 2026-05-08
content_hash: 285db394d292540a
---

# Understanding the Statistical Accuracy-Communication Trade-off in Personalized Federated Learning with Minimax Guarantees

**Conference**: ICML2025  
**arXiv**: [2410.08934](https://arxiv.org/abs/2410.08934)  
**Code**: [ZLHe0/fedclup](https://github.com/ZLHe0/fedclup)  
**Area**: Optimization / Federated Learning  
**Keywords**: Personalized Federated Learning, Statistical Accuracy-Communication Trade-off, Minimax Optimality, Bi-level Optimization, Regularization

## TL;DR

This paper provides the first quantitative characterization of how the personalization parameter $\lambda$ simultaneously affects statistical accuracy and communication efficiency in personalized federated learning, establishes the minimax optimal statistical rate, and proposes the FedCLUP algorithm to achieve the optimal statistical-communication trade-off.

## Background & Motivation

- **Federated Learning (FL)** allows distributed clients to collaboratively train models, but faces challenges with data heterogeneity: a single global model struggles to adapt to all clients.
- **Personalized Federated Learning (PFL)** alleviates heterogeneity by learning a personalized model for each client, where the core problem lies in how to choose the degree of personalization.
- Most existing works study PFL from a purely optimization perspective, lacking analysis on **statistical accuracy**, which leads to a lack of theoretical guidance for the accuracy-communication trade-off.
- **Key Challenge**: How the degree of personalization quantitatively affects the sample efficiency and algorithmic efficiency of each client, as well as the intrinsic trade-off between them, has not been systematically explored.

## Method

### Problem Modeling

This work studies the widely adopted PFL formulation that simultaneously learns a global model $\mathbf{w}^{(g)}$ and local models $\mathbf{w}^{(i)}$:

$$\min_{\mathbf{w}^{(g)}, \{\mathbf{w}^{(i)}\}} \sum_{i \in [m]} p_i \left( L_i(\mathbf{w}^{(i)}, S_i) + \frac{\lambda}{2} \|\mathbf{w}^{(g)} - \mathbf{w}^{(i)}\|^2 \right)$$

- $\lambda$ controls the degree of personalization: $\lambda \to 0$ degenerates to purely local training (LocalTrain), while $\lambda \to \infty$ degenerates to global training (GlobalTrain).
- Statistical heterogeneity is measured by the parameter space $\mathcal{P}(R)$: $\|\mathbf{w}_\star^{(i)} - \sum_i p_i \mathbf{w}_\star^{(i)}\|^2 \le R^2$, where a larger $R$ indicates stronger heterogeneity.

### Statistical Accuracy Analysis (Theorem 1)

Under $L$-smoothness, $\mu$-strong convexity, and bounded gradient variance assumptions, the statistical error of the local models satisfies:

$$\mathbb{E}\|\tilde{\mathbf{w}}^{(i)} - \mathbf{w}_\star^{(i)}\|^2 \le \min\left\{ \mathcal{O}\left(\frac{1}{N} + R^2 + \frac{1}{q_1(\lambda)}\right),\; \mathcal{O}\left(\frac{1}{n} + q_2(\lambda)\right) \right\}$$

- The **first term** decreases as $\lambda$ increases: as $\lambda \to \infty$, it approaches $\mathcal{O}(1/N + R^2)$, which utilizes all $N=mn$ samples but is influenced by the heterogeneity bias $R^2$.
- The **second term** increases as $\lambda$ increases: as $\lambda \to 0$, it approaches $\mathcal{O}(1/n)$, which is the purely local training rate, unaffected by $R$ but suffers from low sample efficiency.

### Minimax Optimality (Corollary 1)

By appropriately choosing $\lambda$ (considering two cases depending on the relationship between $R$ and $1/\sqrt{n}$), the minimax lower bound can be achieved:

$$\mathbb{E}\|\tilde{\mathbf{w}}^{(i)} - \mathbf{w}_\star^{(i)}\|^2 \le C_3 \cdot \frac{1}{N} + C_4 \cdot \left(R^2 \wedge \frac{1}{n}\right)$$

This matches the known minimax lower bound $\Omega(1/N + R^2 \wedge 1/n)$, **proving for the first time that the solution to this PFL formulation is minimax optimal**.

### FedCLUP Algorithm

The original problem is reformulated into a bi-level optimization form, where the outer loop optimizes the global model $\mathbf{w}^{(g)}$ and the inner loop solves local subproblems for each client:

- Outer update: $\mathbf{w}_{t+1}^{(g)} = \mathbf{w}_t^{(g)} - \gamma \lambda \cdot \frac{1}{m}\sum_i (\mathbf{w}_t^{(g)} - \mathbf{w}_\star^{(i)}(\mathbf{w}_t^{(g)}))$
- The inner loop approximately solves the local subproblems using $K$-step gradient descent (warm start), avoiding the high computational cost of exact solutions.
- Communication cost: $\mathcal{O}\left(\kappa \cdot \frac{\lambda + \mu}{\lambda + L} \cdot \log \frac{1}{\varepsilon}\right)$, where a smaller $\lambda$ leads to less communication.
- Computational cost: $\tilde{\mathcal{O}}(\kappa \cdot \log(1/\varepsilon))$, which is independent of $\lambda$.

### Accuracy-Communication Trade-off (Corollary 3)

The total error decomposes into optimization error + statistical error:

- Increasing $\lambda$: The statistical error decreases (tends towards $1/N + R^2$), but communication rounds increase.
- Decreasing $\lambda$: The communication efficiency improves, but statistical accuracy degrades (tends towards $1/n$).
- **Practical Strategy**: When $R^2 \le 1/\sqrt{n}$ (collaboration is beneficial), one should first increase $\lambda$ to reduce the statistical error to the target magnitude, and then increase the communication rounds to match the optimization error.

## Experimental & Theoretical Validation

### Synthetic Data

| Setting | Evaluation Content | Conclusion |
|------|---------|------|
| Overdetermined Linear Regression | Statistical Rate vs. $\lambda$ | Validates the tightness of the upper bound in Theorem 1 |
| Logistic Regression | Communication Efficiency vs. $\lambda$ | Increasing $\lambda$ $\to$ more communication rounds, unchanged computational cost |
| Different $R$ | Accuracy-Communication Trade-off | Low heterogeneity (small $R$): large collaborative gain; High heterogeneity: local training is superior |

### Real Data (MNIST/CIFAR-10 + CNN)

- In the non-convex setting (CNN), the trade-off trend predicted by the theory still holds.
- FedCLUP outperforms GlobalTrain (with a significant boost in communication efficiency) and achieves a smaller total error than LocalTrain under the optimal $\lambda$.
- Compared with pFedMe, FedCLUP possesses both better theoretical guarantees and superior statistical accuracy.

### Key Findings

1. Personalization can **reduce communication costs without extra computational overhead**.
2. There exists an optimal $\lambda^*$ that minimizes the total error (statistical + optimization) under a given communication budget.
3. Although established under convex assumptions, the theory **generalizes well** in non-convex CNN experiments.

## Highlights & Insights

- **Significant Theoretical Contribution**: Establishes the minimax optimality of regularized PFL problems for the first time, and quantitatively characterizes the bilateral impact of $\lambda$ on statistical accuracy and communication for the first time.
- **Novel Analytical Techniques**: Derives the statistical rate directly from the properties of the objective function without relying on specific algorithms; utilizes the GlobalTrain solution as a bridge to prove the case of small $R$.
- **Practical Guidance Value**: Provides a theoretically grounded strategy for selecting the personalization degree $\lambda$.
- **Bi-level Optimization Perspective**: Naturally reformulates the PFL problem into a Moreau envelope form, leading to an elegant algorithm design.

## Limitations & Future Work

- Statistical analysis relies on **strongly convex + smooth** assumptions, while the non-convex theory is only validated experimentally.
- Heterogeneity measurement only considers the model parameter distance $R$, without covering more fine-grained distribution shift structures.
- Only considers **full participation** (all clients participate in each round), without analyzing partial participation scenarios.
- The optimal choice of $\lambda$ requires knowledge of parameters such as $R$, $\rho$, $\mu$, and $L$, which are difficult to obtain in practice.
- The trade-off under privacy constraints (such as differential privacy) is not considered.

## Related Work & Insights

- **L2GD** (Hanzely & Richtárik, 2020): Also studies the optimization complexity of Problem (2), but without statistical analysis.
- **pFedMe** (T Dinh et al., 2020): Employs Moreau envelopes for personalization, but the communication complexity depends on $1/\varepsilon$ instead of $\log(1/\varepsilon)$.
- **Chen et al. (2023c)**: Attempts to establish statistical rates but yields incorrect conclusions when $\lambda \to \infty$, and the rate is loose.
- Insights: The approach of combining statistical estimation theory with distributed optimization can be extended to other FL variants (such as federated transfer learning and federated meta-learning).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Establishes the minimax optimality and accuracy-communication trade-off of PFL for the first time)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Synthetic + real data + non-convex validation, although the real-world data scenarios are relatively simple)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, progressive theoretical derivation)
- Value: ⭐⭐⭐⭐⭐ (Provides the first comprehensive theoretical framework for selecting the personalization level in PFL)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Statistical and Computational Guarantees of Kernel Max-Sliced Wasserstein Distances](statistical_and_computational_guarantees_of_kernel_max-sliced_wasserstein_distan.md)
- [\[ICML 2025\] The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning](the_panaceas_for_improving_low-rank_decomposition_in_communication-efficient_fed.md)
- [\[CVPR 2026\] Few-for-Many Personalized Federated Learning](../../CVPR2026/optimization/few-for-many_personalized_federated_learning.md)
- [\[AAAI 2026\] pFed1BS: Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching](../../AAAI2026/optimization/personalized_federated_learning_with_bidirectional_communication_compression_via.md)
- [\[ICML 2025\] On Understanding Attention-Based In-Context Learning for Categorical Data](on_understanding_attention-based_in-context_learning_for_categorical_data.md)

</div>

<!-- RELATED:END -->
