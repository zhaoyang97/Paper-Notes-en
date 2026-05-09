---
title: >-
  [Paper Note] Hephaestus: Mixture Generative Modeling with Energy Guidance for Large-scale QoS Degradation
description: >-
  [NeurIPS 2025][Image Generation][QoS Degradation] This paper proposes Hephaestus, a three-stage generative framework (Forge-Morph-Refine) that combines a Predicted Path Pressurization (PPS) algorithm, an energy-guided mixture CVAE, and latent-space reinforcement learning optimization to address large-scale network QoS degradation problems.
tags:
  - NeurIPS 2025
  - Image Generation
  - QoS Degradation
  - Mixture CVAE
  - Energy-Based Model
  - Reinforcement Learning
  - Network Optimization
date: 2026-05-08
content_hash: 078f0e2a210f24d0
---

# Hephaestus: Mixture Generative Modeling with Energy Guidance for Large-scale QoS Degradation

**Conference**: NeurIPS 2025
**arXiv**: [2510.17036](https://arxiv.org/abs/2510.17036)
**Code**: None
**Area**: Image Generation
**Keywords**: QoS Degradation, Mixture CVAE, Energy-Based Model, Reinforcement Learning, Network Optimization

## TL;DR

This paper proposes Hephaestus, a three-stage generative framework (Forge-Morph-Refine) that combines a Predicted Path Pressurization (PPS) algorithm, an energy-guided mixture CVAE, and latent-space reinforcement learning optimization to address large-scale network QoS degradation problems.

## Background & Motivation

**State of the Field**: The QoS Degradation (QoSD) problem studies how to perturb edge weights at minimum cost such that the shortest path of critical source-destination pairs in a network exceeds a threshold $T$. This serves as a core vulnerability modeling problem in network security, transportation systems, blockchain, and GNN research.

**Limitations of Prior Work**: (a) QoSD is NP-complete, with a non-submodular objective and exponentially growing combinatorial search space; (b) classical approximation algorithms (AT/IG/SA) handle source-destination pairs independently, resulting in inefficient budget allocation; (c) ILP-based machine learning methods (DiffILO/Predict-and-Search) only handle linear edge weight functions and cannot scale to nonlinear settings.

**Root Cause**: The problem simultaneously requires handling (a) nonlinear edge weight functions (e.g., quadratic convex, logarithmic concave); (b) large-scale graph structures (e.g., RoadCA: ~2M nodes, Skitter); and (c) globally coupled path constraints.

**Paper Goals**: To design an end-to-end, scalable generative framework that efficiently solves large-scale QoSD problems under both linear and nonlinear edge weight functions.

**Starting Point**: The problem is decomposed into three stages: (1) generating a set of feasible solutions; (2) learning the solution distribution via a generative model; (3) optimizing solution quality in latent space using RL with iterative self-enhancement.

**Core Idea**: The combinatorial optimization problem is reformulated as a conditional generative modeling problem. An EBM-guided mixture CVAE captures the solution distribution, and RL efficiently searches for superior solutions in the continuous latent space.

## Method

### Overall Architecture

Three-stage Forge-Morph-Refine pipeline:
1. **Forge**: Generate a feasible solution dataset $\mathfrak{D}^{\text{sol}}$ using SPAGAN and the PPS algorithm.
2. **Morph**: Train an EBM + Mix-CVAE to learn the conditional solution distribution $p(\mathbf{x} \mid \mathbf{c})$.
3. **Refine**: An RL agent optimizes in the latent space, feeding generated solutions back into $\mathfrak{D}^{\text{sol}}$ to form a self-enhancing closed loop.

### Key Designs

**1. Forge: Predicted Path Pressurization (PPS)**

- **Function**: Generates diverse feasible perturbation vectors $\mathbf{x}$ for each graph instance.
- **Mechanism**: A SPAGAN $\mathfrak{F}_\theta$ is trained to predict shortest path costs and guides greedy search — at each step, the edge $e^*$ and increment $\Delta^*$ that maximize the predicted benefit-to-cost ratio are selected:
$$
(e^*, \Delta^*) = \arg\max_{e, \Delta} \frac{\mathcal{C}(P, \mathbf{x} + \Delta \cdot \mathbf{1}_e) - \mathcal{C}(P, \mathbf{x})}{\Delta}
$$
- **Design Motivation**: Traditional methods require repeated exact computation of shortest paths; SPAGAN provides efficient estimation, substantially reducing the cost of feasibility verification.
- **Theoretical Guarantee**: Theorem 1 establishes an approximation ratio $\mathbb{E}[\|\mathbf{x}\|_1] \leq (1 + h\ln(n) + \ln T + \ln(1/\bar{\epsilon})) \cdot \text{OPT}/\mathsf{a}$.

**2. Morph: Energy-Guided Mixture CVAE**

- **Function**: Learns the conditional solution distribution $p(\mathbf{x} \mid [G, \mathcal{K}, T])$.
- **Mechanism**:
    - An EBM $q(\mathbf{x}) = \frac{1}{Z}\exp(-E(\mathbf{x})/\tau)$ is trained as a proxy for the true distribution.
    - A Mix-CVAE $\Omega = [\Omega_0, ..., \Omega_N]$ is trained to approximate the EBM distribution.
    - A minimax optimization is employed to avoid computing the normalization constant $Z$:

$$\min_{q \in \mathcal{Q}} \max_{\Omega \in \mathfrak{E}} \{\text{KL}(p \| q) - \text{KL}(\Omega \| q)\}$$

  - A new CVAE expert is dynamically added when the density discrepancy $\chi(\mathbf{x}) = \log q(\mathbf{x})/\Omega(\mathbf{x}) > \delta$.

- **Theorem 2**: Each additional expert reduces $\text{KL}(q \| \Omega')$ by at least $\gamma(\delta, \epsilon) = a_0(\delta + \log c)\epsilon_0 > 0$.
- **Theorem 3**: The minimax objective is independent of the normalization constant $Z$, eliminating the need for MCMC estimation.

**3. Refine: Latent-Space RL Optimization**

- **Function**: Searches for superior solutions in the Mix-CVAE latent space via RL.
- **Mechanism**: An MDP is defined with state $(z_i, \mathbf{c})$, action as modifications to the latent vector $(\mu_i, \sigma_i)$, and a differentiable reward balancing feasibility and cost:

$$\mathcal{R}(\mathbf{x}_{i+1}) = \digamma(G, \mathcal{K}, \hat{\mathbf{x}}_{i+1}) - \varkappa \cdot \log(1 + \|\bar{\mathbf{x}}_{i+1}\|_1)$$

where the feasibility score is softened via sigmoid: $\digamma = \sum_{(s,t)} \frac{1}{1+\exp(-\zeta(\mathfrak{F}_\theta(\cdot) - T))}$

- **Theorem 4**: Small perturbations along the reward gradient in latent space strictly improve the reward value.
- New solutions are added back to $\mathfrak{D}^{\text{sol}}$ to enable self-enhancement.

### Loss & Training

- SPAGAN: Huber loss for supervised regression of shortest path distances.
- Mix-CVAE experts: Guided ELBO + energy penalty $\mathcal{L}_{\Omega_i}^{guide} = \mathcal{L}_{\Omega_i}^{ELBO} + \lambda \cdot \mathbb{E}[E_\theta(\tilde{p}_\phi(\mathbf{x} | \mathbf{z}, c))]$.
- EBM: Minimization of the expected energy difference under data and model distributions.
- RL: Standard policy gradient + gradient ascent warm-up.

## Key Experimental Results

### Main Results

Linear edge weight functions on real-world networks, threshold $T \in \{140\%, 180\%, 220\%, 260\%\}$:

| Method | Email-260% | Gnutella-260% | RoadCA-260% | Skitter-260% |
|--------|-----------|--------------|------------|-------------|
| AT | 9675 | 10656 | 32935 | 3018467 |
| DiffILO | 9695 | 10987 | 32976 | 3012839 |
| P&S | 9701 | 11364 | 33954 | 3197237 |
| Exact | 9318 | 10073 | — | — |
| **Ours** | **9601** | **10495** | **27699** | **2199372** |

The proposed method shows substantial advantages on large-scale networks: a **28.1%** reduction in total cost over DiffILO on Skitter and **16.8%** on RoadCA.

### Ablation Study

| Mix-CVAE Expert Count | EBM Alignment Quality |
|----------------------|----------------------|
| 3 experts | Noticeable mode gaps |
| 6 experts | Reasonable coverage |
| 9 experts | Close to EBM distribution |

UMAP visualizations of the latent space show clear clustering under different thresholds $T$.

### Key Findings

1. Best performance is achieved on all four real-world datasets at the highest threshold.
2. The framework remains effective under nonlinear edge weights (quadratic convex, logarithmic concave), where ILP-based methods fail entirely.
3. Exact solvers cannot run on large-scale graphs (RoadCA/Skitter) due to memory and time constraints.
4. PPS-I inference refinement is faster than Gurobi and guarantees 100% feasibility.

## Highlights & Insights

1. **Elegant problem decomposition**: The NP-hard combinatorial optimization is decomposed into three stages — data generation, distribution learning, and latent optimization — each with explicit theoretical guarantees.
2. **Avoiding the normalization constant**: Theorem 3 eliminates the computation of $Z$ entirely through the minimax framework, yielding high practical utility.
3. **Self-enhancing closed loop**: Solutions produced by RL optimization are fed back into the dataset, automatically improving the generative model in a positive feedback cycle.
4. **Generalization ability**: Training is conducted on synthetic graphs and transfers to unseen real-world networks.

## Limitations & Future Work

1. The framework depends on the quality of initial solutions — if the PPS approximation algorithm degrades, the entire pipeline is affected.
2. The generalization capability of SPAGAN to unseen graphs with substantially different structures remains uncertain.
3. The framework is architecturally complex (SPAGAN + EBM + Mix-CVAE + RL), resulting in a heavy training pipeline.
4. Validation is limited to the QoSD problem; applicability to other graph optimization problems (e.g., network design, traffic engineering) remains unexplored.
5. The computational overhead associated with increasing the number of experts is not discussed.

## Related Work & Insights

- **Classical network interdiction**: QoSD is a representative soft interdiction problem; this work is the first to solve it end-to-end with a generative model.
- **ML for Combinatorial Optimization**: Compared to Predict-and-Search and DiffILO, the core advantage of Hephaestus is its independence from ILP solvers (e.g., Gurobi).
- **Insights**: The EBM-guided dynamic expert expansion paradigm is transferable to other multimodal combinatorial optimization problems, such as scheduling and routing optimization.

## Rating

⭐⭐⭐⭐ (4/5)

- Novelty ⭐⭐⭐⭐: The three-stage Forge-Morph-Refine framework is original; the EBM-guided Mix-CVAE design is innovative.
- Theory ⭐⭐⭐⭐⭐: Four theorems cover approximation ratio, KL convergence, normalization-free training, and reward consistency.
- Experimental Thoroughness ⭐⭐⭐⭐: Strong results on large-scale real-world networks.
- Value ⭐⭐⭐: The system is complex but demonstrates clear advantages in large-scale scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Large-Scale Training Data Attribution for Music Generative Models via Unlearning](large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)
- [\[NeurIPS 2025\] UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset](ultrahr-100k_enhancing_uhr_image_synthesis_with_a_large-scale_high-quality_datas.md)
- [\[NeurIPS 2025\] ICEdit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer](in-context_edit_enabling_instructional_image_editing_with_in-context_generation_.md)
- [\[NeurIPS 2025\] Diffusion Generative Modeling on Lie Group Representations](diffusion_generative_modeling_on_lie_group_representations.md)
- [\[NeurIPS 2025\] Coupling Generative Modeling and an Autoencoder with the Causal Bridge](coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)

<!-- RELATED:END -->
