---
title: >-
  [Paper Note] Exact and Linear Convergence for Federated Learning under Arbitrary Client Participation is Attainable
description: >-
  [NeurIPS 2025][Optimization][Federated Learning] This paper introduces stochastic matrices and time-varying graphs as modeling tools to unify client participation and local update processes in federated learning into a m…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Federated Learning"
  - "Arbitrary Client Participation"
  - "Exact Convergence"
  - "Push-Pull Strategy"
  - "Stochastic Matrices"
date: 2026-05-08
content_hash: cc801a7019594cb4
---

# Exact and Linear Convergence for Federated Learning under Arbitrary Client Participation is Attainable

**Conference**: NeurIPS 2025
**arXiv**: [2503.20117](https://arxiv.org/abs/2503.20117)
**Code**: [github.com/BichengYing/FedASL](https://github.com/BichengYing/FedASL)
**Area**: Optimization / Federated Learning
**Keywords**: Federated Learning, Arbitrary Client Participation, Exact Convergence, Push-Pull Strategy, Stochastic Matrices

## TL;DR
This paper introduces stochastic matrices and time-varying graphs as modeling tools to unify client participation and local update processes in federated learning into a matrix multiplication formulation. It proposes the FOCUS algorithm (based on a push-pull strategy), which achieves, for the first time, **exact convergence and linear convergence rates** under **arbitrary client participation** and data heterogeneity.

## Background & Motivation

**Background**: Federated learning (FL) enables multiple clients to collaboratively train models without sharing raw data. FedAvg is the most widely used algorithm but suffers from fundamental convergence bias.

**Limitations of Prior Work**: (a) Multiple local update steps combined with non-i.i.d. data cause **client drift**—the fixed point of FedAvg deviates from the true global optimum, with bias $\|x^o - x^\star\|^2 = \Omega((\tau-1)\eta)\|x^\star\|^2$; (b) Arbitrary client participation introduces additional objective bias—the global model converges to a stationary point of a distorted objective weighted by participation probabilities, rather than the true optimum of problem (1); (c) Existing methods rely on **decaying learning rates** to mitigate bias, resulting in slower convergence.

**Key Challenge**: Under arbitrary (unknown) client participation probabilities, can one achieve exact convergence to the global optimum using a **constant learning rate**?

**Goal**: Design an FL algorithm that achieves exact convergence (at a linear rate) under arbitrary client participation, without requiring bounded heterogeneity assumptions or knowledge of participation probabilities.

**Key Insight**: Unify the three core operations of FL (client participation, local updates, model aggregation) as stochastic matrix multiplications, and redesign the FL algorithm from the perspective of decentralized optimization.

**Core Idea**: Model the pull operation with row-stochastic matrices and the push operation with column-stochastic matrices; employ a push-pull dual strategy to simultaneously eliminate client drift and participation bias.

## Method

### Overall Architecture
FOCUS reformulates the FL problem as a constrained decentralized optimization problem: $\min_{\{x_0,...,x_N\}} \frac{1}{N}\sum_{i=0}^N f_i(x_i)$, subject to $R(S_r)\boldsymbol{x} = \boldsymbol{x}$ (consensus constraint). It is solved via a push-pull primal-dual method with the core iteration:

$$\boldsymbol{x}_{k+1} = R_k(\boldsymbol{x}_k - \eta D_k \boldsymbol{y}_k), \quad \boldsymbol{y}_{k+1} = C_k(\boldsymbol{y}_k + \nabla\boldsymbol{f}(\boldsymbol{x}_{k+1}) - \nabla\boldsymbol{f}(\boldsymbol{x}_k))$$

### Key Designs

1. **Stochastic Matrix Modeling Framework**:

    - Function: Unify all FL operations as matrix-vector multiplications.
    - Mechanism: Four matrices are defined—
        - **Model distribution matrix** $R(S_r)$: row-stochastic matrix; for participating clients $i \in S_r$, set $R[i,0]=1$ (pull model from server); for non-participating clients, $R[i,i]=1$ (retain local model).
        - **Model averaging matrix** $A(S_r)$: row-stochastic matrix; server row takes the average over participants $A[0,j] = 1/|S_r|$.
        - **Column-stochastic matrix** $C(S_r) = R(S_r)^\top$: used for the push operation.
        - **Doubly stochastic matrix** $W(S_r)$: applicable in decentralized settings (not directly used in this paper).
    - Design Motivation: Classical FL uses per-client notation, which precludes the use of powerful matrix-theoretic tools. Stacking variables as $\boldsymbol{x}_k = \text{vstack}[x_{k,0}; x_{k,1}; ...; x_{k,N}]$ naturally maps the problem onto graph and matrix theory.

2. **Modeling Arbitrary Client Participation (Assumption 1)**:

    - Function: Cover full participation, active participation, and passive participation within a unified framework.
    - Mechanism: Each client $i$ participates with unknown probability $p_i \in (0,1]$; the average weight is $q_i = \mathbb{E}[\mathbb{I}_i / \sum_j \mathbb{I}_j]$. The structures of the expected matrices $\bar{R}$ and $\bar{A}$ are explicitly derived.
    - Key Special Cases: Uniform sampling ($p_i \equiv m/N, q_i \equiv 1/m$); active participation (independent Bernoulli); passive participation (sampling without replacement from a categorical distribution).

3. **Push-Pull Mechanism of FOCUS**:

    - Function: Track the global gradient via dual variables to eliminate biases introduced by local updates and non-uniform participation.
    - Mechanism:
        - **Pull (row-stochastic matrix $R$)**: Participating clients pull the model from the server: $x_{0,i}^{(r)} \Leftarrow x_r$.
        - **Local updates**: Each client performs $\tau$ steps of gradient tracking: $y_{t+1,i}^{(r)} = y_{t,i}^{(r)} + \nabla f_i(x_{t,i}^{(r)}) - \nabla f_i(x_{t-1,i}^{(r)})$, $x_{t+1,i}^{(r)} = x_{t,i}^{(r)} - \eta y_{t+1,i}^{(r)}$.
        - **Push (column-stochastic matrix $C$)**: Clients push $y_{\tau,i}^{(r)}$ to the server (**summation, not averaging**): $y_{r+1} \Leftarrow y_r + \sum_{i \in S_r} y_{\tau,i}^{(r)}$.
    - **Two Key Properties**:
        - **Consensus**: $R(S_r)\boldsymbol{x}^\star = \boldsymbol{x}^\star$ → all models ultimately converge to the same value.
        - **Tracking**: $\mathbf{1}^\top \boldsymbol{y}_k = \mathbf{1}^\top \nabla\boldsymbol{f}(\boldsymbol{x}_k)$ → the sum of dual variables always equals the sum of global gradients.
    - Design Motivation: The mixing matrix $W_k$ in FedAvg is **not doubly stochastic** under partial participation, preventing exact convergence. Push-pull employs two distinct stochastic matrices to separately handle model consensus (row-stochastic) and gradient tracking (column-stochastic), circumventing this limitation.

4. **Key Differences from FedAvg**:

    - FedAvg pulls model $x$ and averages models → FOCUS pulls model $x$ but pushes gradient $y$.
    - FedAvg aggregates via row-stochastic matrix $A$ → FOCUS pushes via column-stochastic matrix $C$.
    - FedAvg does not maintain dual variables → FOCUS maintains a global $y_r$ on the server.

### Stochastic Gradient Variant: SG-FOCUS
Deterministic gradients are replaced with stochastic gradients $\hat{\nabla}f_i$; both theory and experiments demonstrate faster convergence and higher accuracy.

## Key Experimental Results

### Convergence Comparison

| Algorithm | Exact Convergence | Strongly Convex Complexity | Non-convex Complexity | Participation Assumption | Heterogeneous Gradient Assumption |
|------|:---:|---------|---------|---------|-----------|
| FedAvg | ✗ | $O(1/\epsilon)$ | $O(1/\epsilon^2)$ | Uniform | Bounded |
| SCAFFOLD | ✓* | $O(\log(1/\epsilon))$ | $O(1/\epsilon)$ | Uniform | None |
| FedAU | ✗ | — | $O(1/\epsilon^2)$ | Arbitrary | Bounded |
| FedAWE | ✗ | — | $O(1/\epsilon^2)$ | Arbitrary | Bounded |
| MIFA | ✗ | — | $O(1/\epsilon^2)$ | Arbitrary | Bounded |
| **FOCUS** | **✓** | $O(\log(1/\epsilon))$ | $O(\log(1/\epsilon))$† | **Arbitrary** | **None** |

*SCAFFOLD has no convergence guarantee under arbitrary participation; †requires PL condition.

### FedAvg Fixed-Point Bias Analysis

| Scenario | FedAvg Fixed-Point Bias | FOCUS |
|------|----------------|-------|
| Uniform participation + i.i.d. | $\Omega((\tau-1)\eta)$ | 0 (exact) |
| Non-uniform participation | Additional bias $\delta_q^2$ | 0 (exact) |
| Data heterogeneity | Related to $\sigma_g^2$ | 0 (exact) |

### Key Findings
- FOCUS is the **first** FL algorithm to achieve exact convergence and linear convergence rates under arbitrary client participation.
- It requires neither knowledge of participation probabilities (unlike FedAU and FedAWE) nor a bounded heterogeneous gradient assumption.
- Under the PL condition, non-convex problems also achieve logarithmic complexity (vs. polynomial complexity of all existing methods).
- The stochastic variant SG-FOCUS outperforms the deterministic version in both theory and experiments.
- Communication overhead is similar to SCAFFOLD ($2d$ vectors per round) while supporting arbitrary participation.

## Highlights & Insights
- **The Power of Perspective Shift**: Reformulating FL from per-client notation to matrix multiplication directly enables mature tools from decentralized optimization (push-pull, time-varying graphs). This notational shift opens an entirely new algorithmic design space.
- **Elegance of Push-Pull**: Row-stochastic matrices guarantee consensus; column-stochastic matrices guarantee tracking. The dual interplay of these two matrix types precisely eliminates the two sources of bias in FedAvg, requiring only one additional variable $y_r$ maintained at the server.
- **Critical Role of the Tracking Property**: The invariant $\mathbf{1}^\top \boldsymbol{y}_k = \mathbf{1}^\top \nabla\boldsymbol{f}(\boldsymbol{x}_k)$ ensures that global gradient information is accurately tracked at all times, even as the set of participating clients changes.

## Limitations & Future Work
- Communication cost is $2d$ per round (model $x$ + dual variable $y$), twice that of vanilla FedAvg ($d$).
- Theoretical analysis assumes deterministic gradients; theory for SG-FOCUS remains incomplete.
- Linear convergence for non-convex problems requires the PL condition—the precise complexity of FOCUS for general non-convex problems is not provided.
- Validation on large-scale practical experiments (thousands of clients, real heterogeneous data) is not presented in the main text.
- The server must store each participating client's contribution $y_{\tau,i}^{(r)}$ for summation, which imposes memory pressure at very large participation scales.

## Related Work & Insights
- **vs. FedAvg [McMahan et al., 2017]**: Simple and intuitive, but exhibits irreducible bias under non-uniform participation. FOCUS eliminates bias from first principles.
- **vs. SCAFFOLD [Karimireddy et al., 2020]**: Eliminates client drift via control variates, but requires uniform sampling. FOCUS makes no assumption on the sampling distribution.
- **vs. FedAU/FedAWE [Wang & Ji, 2024; Xiang et al., 2024]**: Correct bias by learning participation probabilities, but the learning process itself slows convergence and precludes exact convergence.
- **vs. MIFA [Gu et al., 2021]**: Addresses participation bias via variance reduction, but requires the server to store each client's model and assumes a bounded staleness condition.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The perspective shift (FL → matrix multiplication → push-pull) is highly ingenious; this is the first work to achieve exact linear convergence under arbitrary participation.
- Experimental Thoroughness: ⭐⭐⭐ Primarily a theoretical contribution; experimental validation is in the appendix and limited in scale.
- Writing Quality: ⭐⭐⭐⭐ The stepwise introduction of the matrix modeling framework is clear, and the illustrations are intuitive.
- Value: ⭐⭐⭐⭐⭐ Addresses a fundamental theoretical problem in FL with strong practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](../../ICCV2025/optimization/federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)
- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[NeurIPS 2025\] Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections](personalized_subgraph_federated_learning_with_differentiable_auxiliary_projectio.md)

</div>

<!-- RELATED:END -->
