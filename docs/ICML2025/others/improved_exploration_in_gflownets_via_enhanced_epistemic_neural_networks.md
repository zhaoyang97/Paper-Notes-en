---
title: >-
  [Paper Note] Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks
description: >-
  [ICML 2025][GFlowNets] This paper integrates Epistemic Neural Networks (ENN/epinet) into GFlowNets to achieve uncertainty-driven exploration, proposing the ENN-GFN-Enhanced algorithm. It significantly improves mode discovery efficiency and distribution learning quality on HyperGrid and sequence generation tasks.
tags:
  - "ICML 2025"
  - "GFlowNets"
  - "epistemic uncertainty"
  - "Epistemic Neural Networks"
  - "Thompson sampling"
  - "exploration"
date: 2026-05-08
content_hash: db31cc5e085d3944
---

# Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks

**Conference**: ICML 2025  
**arXiv**: [2506.16313](https://arxiv.org/abs/2506.16313)  
**Code**: None  
**Area**: Others  
**Keywords**: GFlowNets, epistemic uncertainty, Epistemic Neural Networks, Thompson sampling, exploration

## TL;DR
This paper integrates Epistemic Neural Networks (ENN/epinet) into GFlowNets to achieve uncertainty-driven exploration, proposing the ENN-GFN-Enhanced algorithm. It significantly improves mode discovery efficiency and distribution learning quality on HyperGrid and sequence generation tasks.

## Background & Motivation
**Background**: GFlowNets are a class of generative models that sample objects proportional to a reward through sequential construction, finding crucial applications in scientific discoveries such as molecular design.

**Limitations of Prior Work**: (a) GFlowNets are prone to mode collapse during training, getting attracted to prematurely discovered modes; (b) conventional exploration strategies (on-policy, $\epsilon$-noisy) are inefficient; (c) existing Thompson Sampling (TS-GFN) approximates the posterior using ensembles, which entails high computational overhead and limited joint prediction quality.

**Key Challenge**: The performance of GFlowNets depends heavily on the quality of sampled trajectories, yet effective exploration requires awareness of epistemic uncertainty—namely, knowing "what is not known".

**Key Insight**: Use ENN (epinet) instead of ensembles to obtain more efficient joint prediction and uncertainty quantification.

**Core Idea**: Attach a lightweight epinet module to the GFlowNet's policy network to implement implicit Thompson Sampling via epistemic index sampling.

## Method

### Overall Architecture
Input: GFlowNet + Reward function $\rightarrow$ Attach epinet after the policy network $\rightarrow$ Sample epistemic index $z \sim P_Z$ $\rightarrow$ Generate uncertainty-aware policy $\rightarrow$ Sample trajectories $\rightarrow$ Update with TB loss $\rightarrow$ Iteratively improve distribution learning.

### Key Designs

1. **ENN and Epinet**:

    - The ENN output additionally depends on an epistemic index $z$: $f_\theta(x,z) = \mu_\zeta(x) + \sigma_\eta(\text{sg}[\phi_\zeta(x)], z)$
    - $\mu_\zeta(x)$: base network output
    - $\sigma_\eta$: learnable epinet (lightweight MLP) + fixed prior function $\sigma_P$
    - sg denotes stop-gradient to prevent epinet from influencing base network feature learning
    - Joint prediction: $\hat{P}^{\text{ENN}}(y_{1:\tau}) = \int_z P_Z(dz) \prod_t \text{softmax}(f_\theta(x_t, z))_{y_t}$
    - Design Motivation: ENN achieves calibrated epistemic uncertainty estimation with minimal computational overhead.

2. **ENN-GFN (Basic Version)**:

    - Attach epinet to the forward policy network of GFlowNet.
    - Sample a set of $z$ for each trajectory, combining prior network outputs via weighted sum.
    - Design Motivation: Direct application of the ENN framework to GFlowNets.

3. **ENN-GFN-Enhanced (Enhanced Version)**:

    - Key difference: Instead of weighted sum, randomly select one prior ensemble member.
    - Maintains an approximate posterior policy similar to TS-GFN.
    - However, uncertainty estimation comes from the epinet rather than an ensemble.
    - Design Motivation: Combining the explorative advantage of Thompson Sampling with the efficient uncertainty estimation of ENN.

### Loss & Training
- Trajectory Balance (TB) Loss: $$\mathcal{L}_{\text{TB}}(\tau;\theta) = (\log \frac{Z_\theta \prod P_F(s_{t+1}|s_t)}{R(s_n) \prod P_B(s_t|s_{t+1})})^2$$
- Simultaneously update parameters of both the base network and epinet.

## Key Experimental Results

### Main Results (2D HyperGrid, 8×8, $R_0=10^{-4}$)

| Algorithm | $L_1$ Distance (↓) | 4 Modes Discovered | Description |
|------|---------------|------------|------|
| Default-GFN | High | Partial | Only finds modes near the starting corner |
| TS-GFN | High | Partial | Ensemble brings improvement but is insufficient |
| ENN-GFN | **Low** | ✓ All | Epinet provides better uncertainty |
| ENN-GFN-Enhanced | **Lowest** | ✓ All | Combines TS strategy for best results |

### Ablation Study

| Configuration | 4D Grid ($R_0=10^{-4}, H=8$) | Description |
|------|---------------------------|------|
| ENN-GFN-Enhanced | Lowest $L_1$ | Rapid discovery of all modes |
| ENN-GFN | Close to optimal | Slightly worse than Enhanced |
| TS-GFN | Moderate | Ensemble is effective but inefficient |
| Default-GFN | Highest | Insufficient exploration |

| Environment (2D, $R_0=10^{-5}$) | DB-GFN | ENN-GFN | ENN-GFN-Enhanced |
|--------------------------|--------|---------|------------------|
| 64×64 ($L_1 \times 10^{-5}$) | Baseline | Improved | **Optimal** |
| 128×128 ($L_1 \times 10^{-5}$) | Baseline | Improved | **Optimal** |

### Valid Bit Sequences (Transformer)

| Sequence Length | With Epinet | Without Epinet |
|----------|-------------|----------------|
| 2N=12 |  **260**  | 207 |
| 2N=14 |  **216**  | 189 |
| 2N=16 |  **135**  | 120 |

### Key Findings
- ENN-GFN-Enhanced consistently performs best across all environments.
- The advantage is more pronounced in larger and sparser environments.
- Epinet also brings significant diversity improvements to Transformer architectures.
- ENN-GFN performs well in small environments but may degrade on the 16×16 grid.

## Highlights & Insights
- **Lightweight uncertainty**: epinet only adds small MLPs in the last few layers, with computational overhead far smaller than an ensemble.
- **Importance of joint prediction**: Uncertainty-driven exploration requires joint predictions rather than marginal predictions.
- **Elegance of the Enhanced version**: Randomly selecting ensemble members instead of adopting a weighted sum better simulates Thompson Sampling.

## Limitations & Future Work
- Evaluated only on toy environments (HyperGrid, Bit Sequences), lacking real-world applications such as molecular design.
- The reason for the performance degradation of ENN-GFN in larger environments warrants deeper analysis.
- Comparisons with other exploration-enhancement methods like GAFN, RND, etc., are not comprehensive enough.

## Related Work & Insights
- Thompson Sampling for GFlowNets (Rector-Brooks et al. 2023) is a direct predecessor.
- ENN (Osband et al. 2023) provides the core technical components.
- Random Network Distillation is alternative exploration enhancement technique.
- Insight: Reliable uncertainty estimation is the foundation of efficient exploration.

## Rating
- Novelty: ⭐⭐⭐⭐ The integration of ENN and GFlowNet, along with the Enhanced variant, represents a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐ Environments are somewhat simplistic, lacking practical application scenarios.
- Writing Quality: ⭐⭐⭐⭐ Background introduction is comprehensive and the methodology is clear.
- Value: ⭐⭐⭐⭐ Provides a lightweight and effective solution for the GFlowNet exploration problem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Symmetry-Aware GFlowNets](symmetry-aware_gflownets.md)
- [\[ICML 2026\] On the Epistemic Uncertainty of Overparametrized Neural Networks](../../ICML2026/others/on_the_epistemic_uncertainty_of_overparametrized_neural_networks.md)
- [\[ICML 2025\] Rethinking Aleatoric and Epistemic Uncertainty](rethinking_aleatoric_and_epistemic_uncertainty.md)
- [\[ICML 2025\] Permutation Equivariant Neural Networks for Symmetric Tensors](permutation_equivariant_neural_networks_for_symmetric_tensors.md)
- [\[ICML 2025\] Curvature Enhanced Data Augmentation for Regression](curvature_enhanced_data_augmentation_for_regression.md)

</div>

<!-- RELATED:END -->
