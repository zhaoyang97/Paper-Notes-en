---
title: >-
  [Paper Note] FedRTS: Federated Robust Pruning via Combinatorial Thompson Sampling
description: >-
  [NeurIPS 2025][Optimization][Federated Learning] This paper reformulates federated dynamic pruning as a Combinatorial Multi-Armed Bandit (CMAB) problem and proposes TSAdj, a Thompson Sampling-based topology adjustment mechanism. By replacing deterministic decisions with probabilistic ones, the method obtains more robust sparse model topologies while significantly reducing communication overhead.
tags:
  - NeurIPS 2025
  - Optimization
  - Federated Learning
  - Dynamic Pruning
  - Thompson Sampling
  - Combinatorial Multi-Armed Bandit
  - Sparse Training
date: 2026-05-08
content_hash: 11a07fd1f96fce83
---

# FedRTS: Federated Robust Pruning via Combinatorial Thompson Sampling

**Conference**: NeurIPS 2025
**arXiv**: [2501.19122](https://arxiv.org/abs/2501.19122)
**Code**: [GitHub](https://github.com/Little0o0/FedRTS)
**Area**: Optimization
**Keywords**: Federated Learning, Dynamic Pruning, Thompson Sampling, Combinatorial Multi-Armed Bandit, Sparse Training

## TL;DR
This paper reformulates federated dynamic pruning as a Combinatorial Multi-Armed Bandit (CMAB) problem and proposes TSAdj, a Thompson Sampling-based topology adjustment mechanism. By replacing deterministic decisions with probabilistic ones, the method obtains more robust sparse model topologies while significantly reducing communication overhead.

## Background & Motivation

**Background**: Federated Learning (FL) enables distributed devices to collaboratively train models without sharing data, but the high computational and communication demands of deep models limit participation from resource-constrained devices. Existing federated pruning frameworks adopt dynamic sparse training, where an inner loop updates weights with a fixed topology and an outer loop adjusts the topology (pruning + reactivation) based on aggregated information.

**Limitations of Prior Work**: Existing methods (PruneFL, FedTiny, FedDST, FedMef, etc.) suffer from three critical issues:
   - **Greedy adjustment**: Only the myopic information from a small subset of participating clients in the current round is utilized, discarding all historical observations.
   - **Topology instability**: Deterministic decisions are made based on unreliable aggregated information, leading to high variance under heterogeneous data distributions.
   - **Communication inefficiency**: Uploading auxiliary data such as full-size gradients incurs communication costs approaching those of dense models.

**Key Challenge**: The fundamental issue lies in the "myopic observation + deterministic decision" paradigm — deterministic selections are made based solely on a small fraction of client data in the current round, lacking a global perspective.

**Key Insight**: Topology adjustment is cast as a CMAB problem, where each parameter position is treated as an arm and each outer loop selects $K$ arms to activate.

**Core Idea**: Replace deterministic magnitude-based ranking with Beta posterior distributions for topology selection, and substitute myopic aggregated information with farsighted comprehensive information.

## Method

### Overall Architecture
FedRTS adopts a dual-loop training procedure. The inner loop fixes the topology $m$ and updates weights $W$ via standard FL over $\Delta T$ rounds; the outer loop adjusts the topology via TSAdj — the server samples from Beta distributions, selects Top-$K$ links as the new topology, and updates distribution parameters based on comprehensive observations. The initial topology is obtained via random pruning and Beta distribution sampling.

### Key Designs

1. **CMAB Problem Formulation**:

    - Function: Maps topology adjustment to the combinatorial multi-armed bandit framework.
    - Mechanism: Each parameter position $m_i$ is treated as an arm; at outer loop $t$, the action $S_t$ (containing $K = d\langle m\rangle$ arms) is selected. The objective is to maximize $\max \sum_{t=1}^T r(S_t, \mu)$. Observation $X_t$ is obtained after the action is executed and guides the next round of decisions.
    - Design Motivation: The CMAB framework enables rigorous analysis of the three deficiencies in existing methods and naturally motivates a Thompson Sampling solution.

2. **Thompson Sampling-based Adjustment (TSAdj)**:

    - Function: Replaces deterministic decisions with probabilistic ones for topology adjustment.
    - Mechanism: A Beta posterior $P_i \sim \text{Beta}(\alpha_i, \beta_i)$ is maintained for each link $i$. At adjustment time, $\xi_i \sim P_i$ is sampled and Top-$K$ links are selected: $S_t = \{i \in \text{Top}(\xi, K)\}$. Parameters are updated as $(\alpha_i, \beta_i) \leftarrow (\alpha_i + \lambda X_{t,i}, \beta_i + \lambda(1 - X_{t,i}))$, where $\lambda$ is a scaling factor.
    - Design Motivation: Probabilistic decisions naturally balance exploration and exploitation, exhibit lower variance than deterministic methods, and the posterior distribution inherently accumulates historical information.

3. **Farsighted Comprehensive Observation Mechanism**:

    - Function: Integrates inner/outer loop information and individual/aggregated signals to construct observations.
    - Mechanism: The final observation is a weighted fusion $X_t = \gamma X_t^{agg} + (1-\gamma)\sum_{n \in C_t} p_n X_t^n$. Active arms are evaluated by weight magnitude (distinguishing $\kappa$ core links); inactive arms are evaluated by gradient magnitude (selecting $K-\kappa$ candidates for reactivation).
    - Design Motivation: Fusing individual and aggregated information eliminates single-source bias; the Beta distribution accumulates historical information to compensate for partial client participation.

4. **Communication-Efficient Design**:

    - Function: Clients upload only Top gradient indices.
    - Mechanism: Each client uploads $\mathcal{I}_{t+1}^n = \text{Top}(|G_{t+1}^n|, K-\kappa)$, eliminating the need to transmit full gradients.
    - Design Motivation: The server does not use aggregated gradients to compute $X_{t,i}^{agg}$ for inactive arms (set to 0.5 to avoid high variance), relying only on individual Top indices.

### Loss & Training
- Standard FL task loss (cross-entropy for image classification; language modeling loss for NLP).
- SGD optimizer, learning rate $\eta=0.01$, 5 local epochs, batch size 64 (CV) / 16 (NLP).
- $\lambda=10$, $\gamma=0.5$, $\alpha_{adj}=0.4$; per-layer sparsity follows the ERK distribution.

### Theoretical Guarantees
Under assumptions of independence, $L$-continuity, and mean-field approximation, the regret bound of TSAdj is $Reg(T) = O\Big(\sum_{s \neq s_1^*} \frac{\Delta_{max} L^2 \log T}{\Delta_{s,k}^2} + \Delta_{max}\Big)$, representing the first regret bound for federated pruning derived from a CMAB perspective.

## Key Experimental Results

### Main Results — NLP Task (TinyStories + GPT-2-32M)

| Density | Method | PPL ↓ | Avg. Acc ↑ | Comm. Cost ↓ |
|---------|--------|-------|-----------|--------------|
| 100% | FedAVG | 20.56 | 0.4387 | 260.41MB |
| 50% | FedDST | 20.10 | 0.4261 | 138.30MB |
| 50% | FedMef | 20.61 | 0.4352 | 165.96MB |
| 50% | **FedRTS** | **18.54** | **0.4422** | **138.84MB** |
| 20% | FedDST | 26.49 | 0.4219 | 60.22MB |
| 20% | FedMef | 21.53 | 0.4293 | 72.26MB |
| 20% | **FedRTS** | **19.93** | **0.4333** | **60.34MB** |

### Ablation Study (CIFAR-10 + ResNet18)

| Configuration | Performance | Note |
|---------------|-------------|------|
| FedRTS (full) | Best | Probabilistic decisions + farsighted info + fusion |
| FedRTS ($\gamma=0$) | Close to full | Individual info only, no aggregated info |
| FedRTS ($\gamma=1$) | Notable degradation | Aggregated info only, stability lost |
| FedRTS w/o TS ($\gamma=1$) | Significant degradation | Momentum-based deterministic adjustment |
| FedTiny | Worst | None of the three advantages |

### Key Findings
- FedRTS at 20% density achieves better PPL than full-size FedAVG (19.93 vs. 20.56).
- Probabilistic decisions contribute the most: FedRTS w/o TS falls far behind FedRTS ($\gamma=1$).
- Individual information is more reliable than aggregated information: $\gamma=0$ is close to $\gamma=0.5$, while $\gamma=1$ degrades significantly.
- Advantages are more pronounced under high heterogeneity (Dirichlet $a=0.1$) and low participation rate (10%).

## Highlights & Insights
- **CMAB Perspective**: This is the first work to connect federated pruning with combinatorial bandit theory, providing a novel analytical framework and algorithm design paradigm.
- **Dual Benefits of Probabilistic Decisions**: Thompson Sampling simultaneously addresses the exploration–exploitation trade-off and reduces decision variance.
- **Communication Efficiency as a Natural Byproduct**: TSAdj requires only Top gradient indices, which emerges naturally from the method design rather than as an explicit engineering choice.

## Limitations & Future Work
- The theoretical analysis simplifies the inner loop and partial observations, creating a gap with the actual implementation.
- The independence assumption does not hold strictly in practice.
- Validation is limited to lightweight models (ResNet18 / ShuffleNetV2 / GPT-2-32M); experiments on large-scale models are absent.
- Whether the Beta distribution is the optimal posterior choice remains an open question.

## Related Work & Insights
- **vs. FedDST/FedMef**: Deterministic magnitude pruning vs. probabilistic sampling — FedRTS shows clear advantages in heterogeneous settings.
- **vs. PruneFL**: Client-side adjustment is less stable; FedRTS's server-side global probabilistic adjustment is superior.
- **vs. FedTiny**: Uploading full gradients vs. only Top indices — a substantial communication efficiency gap.

## Rating
- Novelty: ⭐⭐⭐⭐ First use of CMAB + TS for federated pruning; theoretically novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers CV + NLP with complete ablations, but lacks large-model experiments.
- Writing Quality: ⭐⭐⭐⭐ The derivation from CMAB formulation to TS solution is natural and well-structured.
- Value: ⭐⭐⭐⭐ Provides a practical and efficient sparse training solution for resource-constrained federated learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Stable Coresets via Posterior Sampling: Aligning Induced and Full Loss Landscapes](stable_coresets_via_posterior_sampling_aligning_induced_and_full_loss_landscapes.md)
- [\[NeurIPS 2025\] Oracle-Efficient Combinatorial Semi-Bandits](oracle-efficient_combinatorial_semi-bandits.md)
- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](probing_neural_combinatorial_optimization_models.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](doubly_robust_alignment_for_large_language_models.md)
- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](robust_estimation_under_heterogeneous_corruption_rates.md)

</div>

<!-- RELATED:END -->
