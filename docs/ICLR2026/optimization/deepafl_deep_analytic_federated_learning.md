---
title: >-
  [Paper Note] DeepAFL: Deep Analytic Federated Learning
description: >-
  [ICLR 2026][Optimization][Federated Learning] This paper proposes DeepAFL, which designs gradient-free analytic residual blocks and introduces a layer-wise federated training protocol…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Federated Learning"
  - "Analytic Learning"
  - "Gradient-Free Training"
  - "Residual Blocks"
  - "Data Heterogeneity"
date: 2026-05-08
content_hash: 158548346f801c25
---

# DeepAFL: Deep Analytic Federated Learning

**Conference**: ICLR 2026
**arXiv**: [2603.00579](https://arxiv.org/abs/2603.00579)  
**Code**: None  
**Area**: Optimization / Federated Learning
**Keywords**: Federated Learning, Analytic Learning, Gradient-Free Training, Residual Blocks, Data Heterogeneity

## TL;DR

This paper proposes DeepAFL, which designs gradient-free analytic residual blocks and introduces a layer-wise federated training protocol, achieving for the first time a deep analytic federated learning model with representation learning capability. The method maintains ideal invariance to data heterogeneity while overcoming the fundamental limitation of existing analytic approaches to single-layer linear models, surpassing the state of the art by 5.68%–8.42% across three benchmark datasets.

## Background & Motivation

**Federated Learning (FL)** is the dominant distributed learning paradigm for breaking down data silos. However, conventional gradient-based FL methods (e.g., FedAvg, FedProx, SCAFFOLD) face four core challenges: (1) **Data heterogeneity**—distributional discrepancies across clients degrade model performance after aggregation, especially in non-IID settings; (2) **Convergence**—heterogeneous data causes client model divergence, potentially leading to suboptimal global solutions after aggregation; (3) **Scalability**—communication and computational overhead grows substantially as more clients participate; (4) **Communication cost**—multi-round gradient exchanges require significant bandwidth.

In recent years, **Analytic Learning** has offered a promising alternative to these problems. Its core idea is to replace iterative gradient updates with closed-form solutions, thereby fundamentally eliminating the instability of gradient-based training. Several works have introduced analytic learning into federated settings (e.g., FedAnalytic), demonstrating superior invariance to data heterogeneity—since closed-form solutions do not depend on learning rates or iterative rounds, they are unaffected by non-IID data distributions.

However, existing analytic FL methods face a **fundamental bottleneck**: they are restricted to training **single-layer linear models** (e.g., ridge regression / least-squares classifiers) on top of frozen pre-trained backbones. Without representation learning capability, these models rely entirely on the quality of pre-trained features and perform suboptimally on tasks requiring feature adaptation.

The core tension addressed in this paper is: **how to endow analytic models with deep representation learning capability while preserving invariance to data heterogeneity?** The core idea draws on the success of ResNet by designing gradient-free analytic residual blocks—each layer admits a closed-form solution, and stacking them progressively enables deep representation learning.

## Method

### Overall Architecture

The overall pipeline of DeepAFL proceeds as follows: (1) all clients share a pre-trained backbone for initial feature extraction; (2) analytic residual blocks are stacked layer-by-layer on top of the backbone; (3) training at each layer is completed in a single round of local client computation followed by server aggregation (no multi-round communication required); (4) after layer-wise training is complete, a full deep model is obtained. The inputs are local data from each client, and the output is a globally applicable deep classification/regression model.

### Key Designs

1. **Gradient-Free Analytic Residual Block**: Inspired by ResNet, each residual block takes the form $\mathbf{h}^{(l+1)} = \mathbf{h}^{(l)} + f^{(l)}(\mathbf{h}^{(l)})$, where $f^{(l)}$ is a linear transformation with nonlinear activation. The key innovation is that the parameters of $f^{(l)}$ are obtained via **least-squares** (rather than gradient descent), yielding a closed-form solution.

   Specifically, given the input feature matrix $\mathbf{H}^{(l)}$ and target $\mathbf{Y}$ at layer $l$, the parameters $\mathbf{W}^{(l)}$ of the residual mapping $f^{(l)}$ are obtained by solving:

   $\mathbf{W}^{(l)} = \arg\min_{\mathbf{W}} \|\phi(\mathbf{H}^{(l)}) \mathbf{W} - (\mathbf{Y} - \mathbf{H}^{(l)})\|_F^2 + \lambda \|\mathbf{W}\|_F^2$

   where $\phi(\cdot)$ denotes a nonlinear feature mapping (e.g., random features or kernel approximations) and $\lambda$ is the regularization coefficient. This problem admits the closed-form solution: $\mathbf{W}^{(l)} = (\phi(\mathbf{H}^{(l)})^\top \phi(\mathbf{H}^{(l)}) + \lambda \mathbf{I})^{-1} \phi(\mathbf{H}^{(l)})^\top (\mathbf{Y} - \mathbf{H}^{(l)})$.

   The skip connection ensures stable information flow—even if a given layer's mapping is suboptimal, the shortcut preserves input information. Stacking multiple such blocks enables progressive feature refinement.

2. **Layer-Wise FL Protocol**: Conventional FL trains the entire model, requiring many rounds of communication. DeepAFL adopts a layer-wise protocol: for layer $l$, each client $k$ computes locally the covariance matrix $\mathbf{A}_k^{(l)} = \phi(\mathbf{H}_k^{(l)})^\top \phi(\mathbf{H}_k^{(l)})$ and the cross-covariance matrix $\mathbf{B}_k^{(l)} = \phi(\mathbf{H}_k^{(l)})^\top (\mathbf{Y}_k - \mathbf{H}_k^{(l)})$, and uploads these matrices to the server.

   The server simply performs **summation aggregation**: $\mathbf{A}^{(l)} = \sum_k \mathbf{A}_k^{(l)}$, $\mathbf{B}^{(l)} = \sum_k \mathbf{B}_k^{(l)}$, and computes the global solution: $\mathbf{W}^{(l)} = (\mathbf{A}^{(l)} + \lambda \mathbf{I})^{-1} \mathbf{B}^{(l)}$.

   This protocol offers three key advantages:
   - **Invariance to data heterogeneity**: Due to the associativity of matrix summation, the aggregated result is mathematically equivalent to centralized training regardless of how data is distributed across clients.
   - **Single-round communication per layer**: Each layer requires only one communication round (upload matrices → server computes → broadcast parameters), with no iteration needed.
   - **Privacy-friendly**: Only aggregated statistics, rather than raw data or gradients, are transmitted.

3. **Feature Mapping Strategy**: To introduce nonlinearity while preserving closed-form solutions, DeepAFL employs Random Features to approximate kernel mappings. This is a classical technique: random projections followed by nonlinear activations implicitly compute high-dimensional kernel features at controlled computational cost. Different random feature mappings can be used at each layer to increase representational diversity.

### Loss & Training

The training objective at each layer is a regularized least-squares regression:

$$\mathcal{L}^{(l)} = \|\phi(\mathbf{H}^{(l)}) \mathbf{W}^{(l)} - (\mathbf{Y} - \mathbf{H}^{(l)})\|_F^2 + \lambda \|\mathbf{W}^{(l)}\|_F^2$$

Since this is a convex problem, it has a unique global optimum. The training procedure is purely feed-forward—layers are solved sequentially from bottom to top, each in a single pass, with no backpropagation required. The total number of training rounds equals the number of layers (rather than the hundreds or thousands of communication rounds typical of gradient-based FL).

## Key Experimental Results

### Main Results

Comparison on three benchmark datasets under non-IID federated settings:

| Method | Dataset 1 | Dataset 2 | Dataset 3 | Training Paradigm |
|--------|-----------|-----------|-----------|-------------------|
| FedAvg | Baseline | Baseline | Baseline | Multi-round gradient |
| FedProx | ~FedAvg | ~FedAvg | ~FedAvg | Multi-round gradient + regularization |
| SCAFFOLD | > FedAvg | > FedAvg | > FedAvg | Variance reduction |
| FedAnalytic (single-layer) | Limited by linear model | Limited by linear model | Limited by linear model | Single-layer analytic |
| **DeepAFL** | **SOTA (+5.68%~8.42%)** | **SOTA** | **SOTA** | Deep analytic |

DeepAFL outperforms the previous state-of-the-art by 5.68%–8.42% across all three benchmark datasets.

### Ablation Study

| Configuration | Key Metric | Remark |
|---------------|-----------|--------|
| 1 layer vs. multi-layer | Multi-layer significantly better | Validates the necessity of deep representation learning |
| With vs. without skip connections | With skip connections more stable | Skip connections ensure information flow |
| Varying number of layers | Diminishing returns | Improvement plateaus after 3–5 layers |
| IID vs. non-IID | Negligible performance gap | Confirms invariance to data heterogeneity |
| Varying number of clients | Stable performance | Good scalability |

### Key Findings

- **Depth + Analytic = Win-Win**: DeepAFL is the first to demonstrate that analytic learning can be made "deep," and that depth yields substantial performance gains (surpassing both single-layer analytic methods and multi-round gradient-based methods).
- **Heterogeneity invariance validated both theoretically and empirically**: Regardless of how data is partitioned in a non-IID manner, DeepAFL produces results consistent with centralized training—a property that gradient-based FL cannot achieve.
- **Highly communication-efficient**: Each layer requires only one communication round; the total number of rounds equals the number of layers (typically 3–5), far fewer than the hundreds of rounds required by gradient-based methods.
- **No hyperparameter tuning burden**: There are no learning rates, momentum terms, or similar hyperparameters to tune; the regularization coefficient $\lambda$ is the only hyperparameter that needs to be set.

## Highlights & Insights

- **Breaking the perception that "analytic learning = shallow models"**: Through the design of analytic residual blocks, this work demonstrates that gradient-free methods can construct deep networks—a methodological breakthrough.
- **Elegant transfer of the ResNet paradigm**: The most successful architectural design in deep learning (residual connections) is seamlessly transferred to the analytic learning framework, exemplifying cross-paradigm methodological synthesis.
- **A paradigm shift for federated learning**: Rather than mitigating the central challenge of data heterogeneity in FL through various heuristics, DeepAFL fundamentally eliminates its influence—a qualitative rather than incremental improvement.
- **Minimalist algorithmic design**: The entire method involves only matrix multiplication, inversion, and summation, making it straightforward to implement and theoretically transparent.
- **Complete theoretical guarantees**: Invariance to data heterogeneity is established with rigorous mathematical proofs, not merely empirical observations.

## Limitations & Future Work

- **Dependence on pre-trained backbone quality**: Although DeepAFL introduces representation learning capability, it still operates on top of frozen pre-trained features. Poor backbone quality is difficult to compensate for with deeper analytic blocks.
- **Computational bottleneck of matrix inversion**: Each layer requires inverting a $d \times d$ matrix (where $d$ is the feature dimension), which becomes non-trivial when feature dimensions are large (e.g., 1024-dimensional features from ViT-Large).
- **Limitations of random features**: While random feature approximations of kernel mappings are efficient, the resulting representations may still lag behind the hierarchical features learned by end-to-end deep networks.
- **Restricted task types**: Validation is currently limited to classification tasks. Applicability to generative tasks (e.g., federated LLM training) remains unexplored.
- **Privacy risks from transmitted matrices**: Although only aggregated statistics rather than raw data are transmitted, covariance matrices may still leak statistical properties of client data, necessitating further differential privacy analysis.
- **Potential future directions**: Integration with differential privacy; end-to-end analytic feature learning without freezing the backbone; more efficient matrix computation methods (e.g., the Woodbury identity).

## Related Work & Insights

- **FedAvg** (McMahan et al., 2017): The foundational FL algorithm, aggregating client models through multi-round averaging. DeepAFL replaces multi-round approximate averaging with single-round exact summation.
- **Analytic federated learning** (e.g., FedCR, ACIL-FL): Direct predecessors of DeepAFL, but constrained to single-layer linear models. DeepAFL's residual block design overcomes this fundamental limitation.
- **Extreme Learning Machines (ELM)**: A classical approach combining random features with least-squares solutions, which can be viewed as a special case of DeepAFL's single-layer setting.
- **Deep Unfolding**: The idea of unrolling iterative optimization steps layer-by-layer shares conceptual similarity with DeepAFL's layer-wise solving approach.
- **Insights**: Analytic learning, as an alternative paradigm to gradient-based learning, exhibits unique advantages in federated settings that place high demands on convergence stability. Future work could explore analytic learning in other distributed and decentralized scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [\[ICLR 2026\] Weak-SIGReg: Covariance Regularization for Stable Deep Learning](weak-sigreg_covariance_regularization_for_stable_deep_learning.md)
- [\[ICLR 2026\] FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments](feddag_clustered_federated_learning_via_global_data_and_gradient_integration_for.md)
- [\[AAAI 2026\] FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters](../../AAAI2026/optimization/fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)
- [\[AAAI 2026\] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](../../AAAI2026/optimization/tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)

</div>

<!-- RELATED:END -->
