---
title: >-
  [Paper Note] Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning
description: >-
  [NeurIPS 2025][Optimization][federated learning] This paper proposes FedLUAR, which uses a gradient-to-weight ratio metric to identify low-priority layers and recycles their updates from the previous round (rather than d…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "federated learning"
  - "communication efficiency"
  - "gradient recycling"
  - "layer-wise aggregation"
  - "non-IID"
date: 2026-05-08
content_hash: e8f6890fffb7acda
---

# Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning

**Conference**: NeurIPS 2025
**arXiv**: [2503.11146](https://arxiv.org/abs/2503.11146)  
**Code**: [swblaster/FedLUAR](https://github.com/swblaster/FedLUAR)  
**Area**: Optimization
**Keywords**: federated learning, communication efficiency, gradient recycling, layer-wise aggregation, non-IID

## TL;DR
This paper proposes FedLUAR, which uses a gradient-to-weight ratio metric to identify low-priority layers and recycles their updates from the previous round (rather than discarding them), achieving accuracy nearly identical to FedAvg with only 17% communication overhead.

## Background & Motivation
**Background**: Communication overhead from model aggregation is a central bottleneck in federated learning (FL), a problem that worsens as model size increases.

**Limitations of Prior Work**: (a) Quantization methods (FedPAQ): uniformly reducing precision degrades the representation quality of all parameters; (b) Pruning methods (PruneFL): directly reducing parameter count impairs learning capacity; (c) Low-rank decomposition (FedPara): introduces additional computational overhead by increasing the number of network layers; (d) A common weakness — all these methods ultimately *discard* information.

**Core Idea**: Layers with small gradient magnitudes are not necessarily less important to the model — the ratio of gradients to weights is the more relevant quantity. When gradients are large but weights are also large, the impact on layer outputs is limited.

**Key Insight**: Rather than discarding updates from low-priority layers, recycling the updates from the previous round reduces communication without completely losing update information.

## Method

### Layer-wise Priority Metric

The priority score for layer $l$ at round $t$ is defined as:

$$s_{t,l} = \frac{\|\Delta_{t,l}\|}{\|x_{t,l}\|}$$

where $\Delta_{t,l}$ is the cumulative update averaged across all clients and $x_{t,l}$ is the layer's initial parameters. A small $s_{t,l}$ indicates that the parameter change is insignificant relative to its magnitude.

**Zero computation overhead**: Both $x_{t,l}$ and $\Delta_{t,l}$ are already available on the server, requiring no additional communication.

### Stochastic Layer Selection Mechanism

A probability distribution over layers is constructed from $s_{t,l}$ to sample $\delta$ recycled layers:

$$p_{t,l} = \frac{1/s_{t,l}}{\sum_{l=0}^{L-1} 1/s_{t,l}}$$

Layers with lower priority (smaller $s_{t,l}$) are sampled with higher probability. Weighted random sampling prevents the same layer from being recycled consecutively — when a layer is not selected, it undergoes normal aggregation and its $s_{t,l}$ is updated accordingly.

### Update Recycling Scheme

- For selected layers $l \in \mathcal{R}_t$: the previous round's update $r_t = [\hat{\Delta}_{t-1,l}]$ is reused.
- For remaining layers: normal client update aggregation $u_t = [\Delta_{t,l}]$ is applied.
- The global update is composed as: $\hat{\Delta}_t = [r_t, u_t]$

Clients only need to upload updates for $L - \delta$ layers, reducing communication volume proportionally to the parameter count of those layers.

### Convergence Analysis

**Noise definition**: Recycling introduces noise $n_t = \hat{\Delta}_t - \Delta_t = \frac{1}{m}\sum_i\sum_j (\hat{g}_{t-k,j}^i - \hat{g}_{t,j}^i)$

**Lemma 1 (Noise Bound)**: Under assumptions of Lipschitz continuity, unbiased gradients, and bounded variance, if $\eta \le 1/(\mathcal{L}\tau)$, the accumulated noise is bounded, and remains controllable when $\kappa = \|\nabla\hat{F}(x_t)\|^2 / \|\nabla F(x_t)\|^2$ is sufficiently small (independent of the number of recycling steps $k$).

**Theorem 2 (Convergence Rate)**: If $\eta \le \frac{1-16\kappa}{6\sqrt{30}\mathcal{L}\tau}$ and $\kappa < 1/16$, then

$$\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}[\|\nabla F(x_t)\|^2] \le \frac{4}{(1-16\kappa)\eta\tau T}(F(x_0) - F(x_T)) + O\left(\frac{\sigma_L^2}{1-16\kappa}\right) + O\left(\frac{\mathcal{L}^2\eta^2\tau^2\sigma_G^2}{1-16\kappa}\right)$$

The method converges to a neighborhood of a stationary point. The condition on $\kappa$ is naturally governed by $\delta$ — the fewer layers recycled, the smaller $\kappa$.

## Key Experimental Results

### Comparison with State-of-the-Art Communication-Efficient FL Methods

| Method | CIFAR-10 (ResNet20) | Comm. Ratio | CIFAR-100 (WRN-28) | Comm. Ratio | FEMNIST (CNN) | Comm. Ratio | AG News (DistillBERT) | Comm. Ratio |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FedAvg | 61.27% | 1.00 | 59.88% | 1.00 | 71.01% | 1.00 | 82.66% | 1.00 |
| LBGM | 54.87% | 0.65 | 57.13% | 0.87 | 69.83% | 0.71 | 77.96% | 0.23 |
| FedPAQ | 57.42% | 0.50 | 36.15% | 0.50 | 71.54% | 0.25 | 82.72% | 0.25 |
| FedPara | 55.16% | 0.51 | 46.14% | 0.61 | 67.69% | — | — | — |
| **FedLUAR** | **61.27%** | — | **59.88%** | — | **71.01%** | **0.17** | **82.66%** | **0.17** |

FedLUAR achieves accuracy on par with FedAvg on FEMNIST and AG News using only 17% of the communication budget, substantially outperforming all baselines.

### Memory Usage Comparison

| Dataset (Model) | FedAvg Memory | FedLUAR Memory | $\delta$ |
|---------------|:---:|:---:|:---:|
| CIFAR-10 (ResNet20) | 33.49 MB | 15.23 MB | 10 |
| CIFAR-100 (WRN28-10) | 4,462.80 MB | 2,604.88 MB | 14 |
| FEMNIST (CNN) | 806.11 MB | 204.73 MB | 2 |
| AG News (DistillBERT) | 8,294.18 MB | **1,825.42 MB** | 30 |

Memory is reduced by 78% on AG News and 75% on FEMNIST.

### Recycling vs. Discarding

Experiments demonstrate that for the same set of layers, recycling the previous round's updates leads to faster convergence and higher final accuracy than simply discarding (zeroing out) updates — the key benefit being the retention of approximate gradient direction information rather than its complete elimination.

### Non-IID Robustness

Under highly non-IID conditions (Dirichlet $\alpha=0.1$), FedLUAR maintains accuracy close to FedAvg. Theoretical analysis shows that a lower learning rate is required to sustain convergence as the degree of non-IID heterogeneity increases.

## Highlights & Insights
1. The **"recycle rather than discard"** principle is conceptually simple yet had been overlooked — a low-hanging fruit with meaningful impact.
2. The gradient-to-weight ratio metric better captures a layer's influence on the model than raw gradient magnitude alone.
3. Stochastic sampling prevents persistent recycling of the same layer without requiring a rigid cap.
4. Layer selection is performed entirely on the server side, incurring no additional communication cost.
5. The method is optimizer-agnostic and can be combined with FedProx, SCAFFOLD, and other FL algorithms.

## Limitations & Future Work
1. The convergence guarantee only covers a neighborhood of a stationary point (not an exact optimum); the $(4+9\mathcal{L}^2)\sigma_L^2$ term does not vanish as $\eta \to 0$.
2. Experiments are conducted at limited scale (128 clients, 2 GPUs); performance in thousand-client scenarios has not been validated.
3. $\delta$ is a manually tuned hyperparameter; while ablation analysis is provided, no adaptive selection strategy is proposed.
4. Downlink communication (server to client) is not compressed; only uplink communication is optimized.
5. Joint use with quantization or pruning remains unexplored.

## Related Work & Insights
- **vs. LBGM**: LBGM uses low-rank approximation for compression, reducing communication at the cost of significant accuracy loss; FedLUAR maintains full-precision updates.
- **vs. FedPAQ/FedBAT** (quantization/binarization): Uniform precision reduction causes catastrophic accuracy degradation on CIFAR-100 (36%); FedLUAR does not suffer from this issue.
- **vs. PruneFL**: Pruning permanently removes parameters; FedLUAR only defers updates, preserving full model capacity.
- **vs. YOGA** (decentralized layer-wise aggregation): YOGA assumes a peer-to-peer setting without a central server, making it unsuitable for centralized FL.

The recycling paradigm is generalizable to other communication-constrained settings (e.g., distributed training, edge computing). The gradient-to-weight ratio metric may also inform layer-wise learning rate scheduling. Combining gradient compression with recycling is a promising direction for future exploration.

## Rating
- ⭐ Novelty: 3/5 — The core idea is simple and intuitive yet effective; the layer-wise recycling mechanism is relatively novel.
- ⭐ Experimental Thoroughness: 4/5 — Four datasets, multiple baselines, ablation studies, and memory analysis provide comprehensive coverage.
- ⭐ Writing Quality: 4/5 — Well-structured with appropriate integration of theoretical and empirical contributions.
- ⭐ Value: 4/5 — Highly practical; directly integrable into existing FL systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multiplayer Federated Learning: Reaching Equilibrium with Less Communication](multiplayer_federated_learning_reaching_equilibrium_with_less_communication.md)
- [\[NeurIPS 2025\] MAR-FL: A Communication Efficient Peer-to-Peer Federated Learning System](mar-fl_a_communication_efficient_peer-to-peer_federated_learning_system.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](efficient_adaptive_federated_optimization.md)
- [\[NeurIPS 2025\] FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning](fedqs_optimizing_gradient_and_model_aggregation_for_semi-asynchronous_federated_.md)
- [\[ICML 2026\] Delayed Momentum Aggregation: Communication-efficient Byzantine-robust Federated Learning with Partial Participation](../../ICML2026/optimization/delayed_momentum_aggregation_communication-efficient_byzantine-robust_federated_.md)

</div>

<!-- RELATED:END -->
