---
title: >-
  [Paper Note] Training Robust Graph Neural Networks by Modeling Noise Dependencies
description: >-
  [NeurIPS 2025][Optimization][GNN robustness] This paper proposes Dependency-Aware Graph Noise (DANG) and the DA-GNN framework, which model a causal dependency chain from node feature noise → graph structure noise → label…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "GNN robustness"
  - "noise dependency"
  - "causal modeling"
  - "variational inference"
  - "data generating process"
date: 2026-05-08
content_hash: d84816be99728fac
---

# Training Robust Graph Neural Networks by Modeling Noise Dependencies

**Conference**: NeurIPS 2025
**arXiv**: [2502.19670](https://arxiv.org/abs/2502.19670)  
**Code**: [GitHub](https://github.com/yeonjun-in/torch-DA-GNN)  
**Area**: Optimization
**Keywords**: GNN robustness, noise dependency, causal modeling, variational inference, data generating process

## TL;DR

This paper proposes Dependency-Aware Graph Noise (DANG) and the DA-GNN framework, which model a causal dependency chain from node feature noise → graph structure noise → label noise, and employ variational inference to derive an ELBO objective for training GNNs robust to multi-source correlated noise.

## Background & Motivation

One of the core challenges facing GNNs in real-world applications is data noise. Existing robust GNN methods address feature noise (AirGNN), structural noise (RSGNN, STABLE), or label noise (NRGNN, RTGNN) in isolation, all under the assumption that noise sources are mutually independent — the so-called Independent Feature Noise (IFN) assumption.

**Core limitation**: The IFN assumption is severely at odds with reality. In social networks, Bob creates a fake profile (feature noise) → Alice and Tom connect with Bob based on the fake profile (structural noise) → Alice and Tom's community labels are consequently corrupted (label noise). Such cascading noise dependencies are pervasive in social, e-commerce, and biological graph applications.

**Fundamental limitations of prior work**: Every existing robust GNN method assumes that at least one data source (features/structure/labels) is clean. For instance, AirGNN assumes a noise-free structure; RTGNN relies on structural information to mitigate label noise (even though the structure itself may be noisy); SG-GSR assumes clean labels. When all data sources are noisy, these assumptions are violated and performance degrades substantially.

**Core Idea**: Formally define DANG and its Data Generating Process (DGP), introducing three latent variables (noise variable $\epsilon$, latent clean graph structure $Z_A$, latent clean labels $Z_Y$) and establishing their causal relationships with observed variables ($X$, $A$, $Y$). A deep generative model, DA-GNN, is then designed to directly capture these causal relationships.

## Method

### Overall Architecture

DA-GNN is a variational inference-based encoder–decoder framework:
- **Inference encoders** ($\phi_1, \phi_2, \phi_3$): infer latent variables $Z_A, \epsilon, Z_Y$ from noisy observations
- **Generative decoders** ($\theta_1, \theta_2, \theta_3$): model the process of generating observed data from latent variables
- Trained by minimizing the negative ELBO

### Key Designs

1. **DGP for DANG**: Six causal relationships are defined —

    - $X \leftarrow (\epsilon, Z_Y)$: the noise variable and true labels jointly produce the (potentially noisy) features
    - $A \leftarrow (Z_A, X)$: the latent clean structure and features jointly produce the (potentially noisy) edges
    - $A \leftarrow \epsilon$: the noise variable can also directly induce structural noise
    - $Y \leftarrow (Z_Y, X, A)$: true labels, features, and structure jointly produce the (potentially noisy) observed labels

   A key property of this DGP is that no data source in the graph is assumed to be entirely clean.

2. **Inference encoder design**:

    - **$q_{\phi_1}(Z_A|X,A)$ (inferring clean graph structure)**: A GCN encoder computes node representations $\mathbf{Z} = \text{GCN}_{\phi_1}(\mathbf{X}, \mathbf{A})$, and a latent graph is obtained via cosine similarity: $\hat{p}_{ij} = \rho(s(\mathbf{Z}_i, \mathbf{Z}_j))$. A $\gamma$-hop subgraph similarity prior is used for regularization (corresponding to the KL divergence term). To avoid $O(N^2)$ full-graph computation, a pre-defined proxy graph restricts the computational scope.

    - **$q_{\phi_3}(Z_Y|X,A)$ (inferring clean labels)**: A GCN classifier operating on the inferred clean graph $\hat{\mathbf{A}}$. Homophily regularization encourages connected nodes to have similar predictions:

    $\mathcal{L}_{\text{hom}} = \sum_{i \in \mathcal{V}} \frac{\sum_{j \in \mathcal{N}_i} \hat{p}_{ij} \cdot kl(\hat{\mathbf{Y}}_j || \hat{\mathbf{Y}}_i)}{\sum_{j \in \mathcal{N}_i} \hat{p}_{ij}}$

    - **$q_{\phi_2}(\epsilon|X,A,Z_Y)$ (inferring the noise variable)**: Decomposed into structural noise $\epsilon_A$ (edge cleanliness estimated via small-loss criterion during an early-learning phase) and feature noise $\epsilon_X$ (inferred from $X$ and $Z_Y$ via an MLP, regularized toward a standard normal distribution). An EMA trick mitigates uncertainty in point estimates: $\hat{p}_{ij}^{el} \leftarrow \xi \hat{p}_{ij}^{el} + (1-\xi)\hat{p}_{ij}^c$, with $\xi=0.9$.

3. **Generative decoder design**:

    - **$p_{\theta_1}(A|X,\epsilon,Z_A)$ (reconstructing noisy edges)**: An edge reconstruction loss that regularizes both predictions and labels to handle noisy supervision. Prediction regularization: $\hat{p}_{ij}^{reg} = \theta_1 \hat{p}_{ij} + (1-\theta_1)s(\mathbf{X}_i, \mathbf{X}_j)$, which penalizes edge predictions when feature similarity is high (as spurious connections may arise from feature noise). Labels are smoothed to the interval $[0.9, 1]$.
    - **$p_{\theta_2}(X|\epsilon,Z_Y)$ (reconstructing noisy features)**: An MLP decoder taking $\epsilon_X$ and $Z_Y$ as input, with samples drawn via the reparameterization trick.
    - **$p_{\theta_3}(Y|X,A,Z_Y)$ (reconstructing noisy labels)**: A GCN classifier modeling the transition from clean to noisy labels.

### Loss & Training

The total loss is a weighted sum of ELBO terms:

$$\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{cls-enc}} + \lambda_1 \mathcal{L}_{\text{rec-edge}} + \lambda_2 \mathcal{L}_{\text{hom}} + \lambda_3(\mathcal{L}_{\text{rec-feat}} + \mathcal{L}_{\text{cls-dec}} + \mathcal{L}_{\text{p}})$$

where $\lambda_3=0.001$ is fixed (these three terms have limited impact on performance), $\lambda_1$ and $\lambda_2$ require tuning, and $k$ (the proxy graph parameter) is the most critical hyperparameter.

## Key Experimental Results

### Main Results (Node classification accuracy % under DANG)

| Dataset | Noise Level | WSGNN | GraphGLOW | AirGNN | STABLE | RTGNN | SG-GSR | DA-GNN |
|--------|---------|-------|-----------|--------|--------|-------|--------|--------|
| Cora | Clean | 86.2 | 85.2 | 85.0 | 86.1 | 86.1 | 85.7 | **86.2** |
| Cora | DANG-10% | 80.7 | 79.7 | 79.7 | 82.2 | 81.8 | 82.7 | **82.9** |
| Cora | DANG-30% | 70.0 | 71.6 | 71.5 | 74.3 | 72.6 | 76.1 | **78.2** |
| Cora | DANG-50% | 55.9 | 59.6 | 56.2 | 62.8 | 60.9 | 64.3 | **69.7** |
| Photo | DANG-50% | 31.9 | 85.4 | 57.8 | 80.2 | 79.2 | 84.1 | **87.6** |
| Comp | DANG-50% | 39.6 | 80.1 | 44.1 | 68.8 | 69.4 | 78.6 | **82.2** |

### Ablation Study (Incrementally removing causal relationships from the DGP)

| Configuration | Cora Clean | Cora DANG-30% | Cora DANG-50% | Description |
|------|-----------|--------------|--------------|------|
| Case 1 (IFN, remove all dependencies) | 84.6 | 68.3 | 55.2 | Degenerates to independent noise assumption |
| Case 2 (remove $A \leftarrow X$) | 84.8 | 68.5 | 56.1 | Ignores feature→structure noise propagation |
| Case 3 (remove only $Y \leftarrow (X,A)$) | 86.2 | 77.3 | 68.7 | Ignores label transition relationship |
| **Proposed (full DANG)** | **86.2** | **78.2** | **69.7** | Full causal modeling |

### Real-World DANG Datasets

| Task | Dataset | Metric | SG-GSR (runner-up) | DA-GNN |
|------|--------|------|-------------|--------|
| Node classification | Auto + DANG | Accuracy | 62.0±1.1 | **61.4±0.4** |
| Node classification | Garden + DANG | Accuracy | 80.2±0.4 | **80.2±0.8** |
| Link prediction | Auto + DANG | ROC-AUC | 65.6±7.4 | **73.6±0.6** |
| Link prediction | Garden + DANG | ROC-AUC | 86.0±7.2 | **92.4±0.4** |

### Key Findings

- **The advantage of DA-GNN grows with noise level**: On Cora DANG-50%, DA-GNN (69.7%) outperforms the runner-up SG-GSR (64.3%) by 5.4 percentage points.
- **Every causal relationship in the DGP contributes**: Incrementally removing causal edges leads to monotonically decreasing performance (69.7→68.7→56.1→55.2).
- **Broad applicability**: DA-GNN achieves state-of-the-art or competitive performance across five settings: DANG, feature-only noise, structure-only noise, label-only noise, and extreme noise (all three simultaneously).
- **On the large-scale Arxiv graph**, most baselines fail due to OOM; DA-GNN achieves 44.0% accuracy under DANG-50%.

## Highlights & Insights

- **DANG fills an important gap**: The IFN assumption in existing robust GNN research is a widely accepted but unrealistic simplification; DANG provides a noise model that more faithfully reflects reality.
- **Elegance of causal modeling**: The DGP, expressed as a directed graphical model, cleanly defines six causal relationships, yielding a principled and theoretically grounded ELBO derivation.
- **Practical new datasets**: The Auto and Garden datasets, constructed from Amazon review data to simulate DANG in e-commerce settings, provide useful benchmarks for future research.
- **Well-designed ablation**: By incrementally removing causal edges from the DGP, the contribution of each dependency is clearly demonstrated.

## Limitations & Future Work

- DANG does not consider the reverse dependency $X \leftarrow A$ (structural noise feeding back into node features), which is also natural in certain settings.
- Training complexity is relatively high (three latent variables must be inferred jointly), which may limit scalability to large graphs.
- Proxy graph precomputation incurs $O(N^2)$ initialization overhead; although optimized, it remains a bottleneck.
- The hyperparameter $k$ (proxy graph parameter) requires tuning; while the search space is small, this adds a practical barrier.
- The performance gains on real-world DANG datasets are smaller than on synthetic ones, suggesting that synthetic settings may overestimate the method's advantage.

## Related Work & Insights

- The approach is theoretically related to instance-dependent label noise (IDN) generative methods, but extending to the graph domain introduces additional latent variables and more complex causal relationships.
- WSGNN and GraphGLOW also employ variational inference and graph structure inference, but assume a noise-free graph, limiting their applicability.
- Insight: In other graph learning tasks such as recommendation systems and knowledge graphs, inter-noise dependencies are equally prevalent; the DANG modeling paradigm can be generalized accordingly.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The definition of DANG and the DGP-based causal modeling methodology make important conceptual contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers synthetic/real DANG, individual noise types, and extreme noise settings, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ The causal modeling section is clear, though the model instantiation details are dense and require careful reading.
- Value: ⭐⭐⭐⭐⭐ Advances the robust GNN research paradigm — from "assume some data source is clean" to "all data sources may be noisy."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rapid Training of Hamiltonian Graph Networks using Random Features](../../ICLR2026/optimization/rapid_training_of_hamiltonian_graph_networks_using_random_features.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes](quantitative_convergence_of_trained_single_layer_neural_networks_to_gaussian_pro.md)
- [\[NeurIPS 2025\] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)

</div>

<!-- RELATED:END -->
