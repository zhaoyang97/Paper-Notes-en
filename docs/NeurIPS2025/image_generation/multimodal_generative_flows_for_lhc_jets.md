---
title: >-
  [Paper Note] Multimodal Generative Flows for LHC Jets
description: >-
  [NeurIPS 2025][Image Generation][flow matching] This paper proposes a Transformer-based multimodal flow matching framework (MMF) that jointly models continuous flow matching and continuous-time Markov jump bridges, enabling unified generation of particle kinematics (continuous) and flavor quantum numbers (discrete) in LHC jets.
tags:
  - NeurIPS 2025
  - Image Generation
  - flow matching
  - multimodal generation
  - particle physics
  - continuous-time Markov jump
  - transformer
date: 2026-05-08
content_hash: 42c5ee3523aeab61
---

# Multimodal Generative Flows for LHC Jets

**Conference**: NeurIPS 2025
**arXiv**: [2509.01736](https://arxiv.org/abs/2509.01736)
**Code**: [Available](https://github.com/dfaroughy/Multimodal-flows)
**Area**: Generative Models / Scientific AI / High-Energy Physics
**Keywords**: flow matching, multimodal generation, particle physics, continuous-time Markov jump, transformer

## TL;DR

This paper proposes a Transformer-based multimodal flow matching framework (MMF) that jointly models continuous flow matching and continuous-time Markov jump bridges, enabling unified generation of particle kinematics (continuous) and flavor quantum numbers (discrete) in LHC jets.

## Background & Motivation

The Large Hadron Collider (LHC) produces billions of proton–proton collisions per second, and **jets**—collimated sprays of high-energy particles—are central objects in QCD research and new physics searches. Generative models can be applied to data-driven simulation, anomaly detection, and related tasks.

**Core Problem**: Jet data are inherently **multimodal**—each particle carries both continuous kinematic features ($p_T$, $\Delta\eta$, $\Delta\phi$) and discrete quantum numbers (charge and flavor, covering 8 categories such as photons, charged hadrons, and leptons). Existing approaches (e.g., diffusion models, flow matching) operate only in continuous spaces, and handling discrete modes via dequantization or separate modeling destroys physically meaningful cross-modal correlations.

**Motivation**: A unified probabilistic framework is needed that can **jointly** handle continuous and discrete modalities while preserving their physical correlations.

## Method

### Overall Architecture

A probabilistic flow equation is constructed over the hybrid space $\mathbb{R}^3 \otimes \mathcal{F}$, where the continuous component is driven by flow matching and the discrete component by a continuous-time Markov jump bridge. The core probability path satisfies:

$$\partial_t P_t(\bm{z}_t) = -\nabla_{\bm{x}} \cdot [\bm{u}_t P_t] + \sum_{\bm{j} \neq \bm{k}_t} [\bm{W}_t(\bm{z}_t, \bm{j}) P_t(\bm{j}) - \bm{W}_t(\bm{j}, \bm{z}_t) P_t(\bm{z}_t)]$$

The first term is the continuity equation for the continuous modality (velocity field $\bm{u}_t$), and the second term is the Master equation for the discrete modality (jump rate matrix $\bm{W}_t$).

### Key Designs

1. **Conditional Dynamics**:

   - **Continuous modality**: Standard uniform flow with a straight-line path from source to target, $u_t^d = x_1^d - x_0^d$.
   - **Discrete modality**: A generalized **multi-state telegraph process** is proposed, where the jump rate matrix depends on the target flavor token and a stochasticity hyperparameter $\beta$; jump frequency is controlled via $\omega_t = \exp(-S\beta(1-t))$.

2. **Analytical Posterior Tractability**: The expected discrete jump rates can be computed **analytically** (without approximation), reducing posterior learning to a multi-class classification task. A time-dependent classifier $h_t^\theta$ outputs posterior probabilities via softmax and is trained with cross-entropy loss.

3. **Multimodal ParticleFormer Architecture** (Fig. 2):

   - **Two modality-specific encoders**: processing continuous kinematics and discrete flavor separately.
   - **One fusion encoder**: based on a non-causal particle Transformer (stacked multi-head self-attention).
   - **Two task heads**: a regression head predicting the velocity field (MSE loss) and a classification head outputting logits (CE loss).
   - The overall architecture maintains **permutation equivariance**.

### Loss & Training

A multi-task weighted loss is adopted, inspired by uncertainty weighting from Kendall et al.:

$$\mathcal{L}_{\text{MMF}} = \mathbb{E}\left[\frac{\|u_t^\theta - u_t\|^2}{2(\sigma_t^1)^2} - \frac{\log h_t^\theta(\bm{z}_t, \bm{k}_1)}{2(\sigma_t^1)^2} + \log(\sigma_t^1 \sigma_t^2)\right]$$

A key innovation is elevating the uncertainty weights from fixed scalars to **time-dependent functions** $\sigma_t^i = \exp(-w_t^i)$, output by an auxiliary network, allowing dynamic rebalancing of the two modalities across different generation stages. This auxiliary network is discarded at inference. During sampling, the continuous component is integrated via the Euler method for the ODE, while the discrete component is simulated via $\tau$-leaping for the Markov jump process; temperature scaling $T$ is introduced to improve sample quality.

## Key Experimental Results

### Main Results — Wasserstein Distance Comparison

Dataset: AspenOpenJets (AOJ) from CMS Open Data; 1.25M jets for training, 270K generated.

| Metric | EPiC-FM | MMF (Ours) |
|--------|---------|-----------|
| $W_1^{p_T}$ | **0.92** | 4.64 |
| $W_1^m$ (mass) | 1.63 | **1.26** |
| $W_1^\eta$ | $1.2 \times 10^{-3}$ | $\mathbf{6.3 \times 10^{-4}}$ |
| $W_1^\phi$ | $2.8 \times 10^{-3}$ | $\mathbf{2.3 \times 10^{-4}}$ |
| $W_1^{\tau_{21}}$ (substructure) | $3.1 \times 10^{-2}$ | $\mathbf{2.3 \times 10^{-3}}$ |
| $W_1^{\tau_{32}}$ (substructure) | $1.8 \times 10^{-2}$ | $\mathbf{2.8 \times 10^{-3}}$ |
| $W_1^{\mathcal{Q}}$ (jet charge) | $9.5 \times 10^{-3}$ | $\mathbf{1.4 \times 10^{-3}}$ |

### Flavor Multiplicity Wasserstein Distance

| Flavor | EPiC-FM | MMF (Ours) |
|--------|---------|-----------|
| $N^\gamma$ (photon) | **0.23** | 0.34 |
| $N^{h^0}$ (neutral hadron) | 0.10 | **0.01** |
| $N^{h^-}$ (negative hadron) | 0.28 | **0.09** |
| $N^{h^+}$ (positive hadron) | 0.23 | **0.10** |
| $N^{e^-}$ | $\mathbf{5.6 \times 10^{-4}}$ | $5.7 \times 10^{-2}$ |
| $N^{\mu^-}$ | $\mathbf{2.6 \times 10^{-3}}$ | $4.3 \times 10^{-2}$ |

### Key Findings

- MMF substantially outperforms the baseline on jet substructure ($\tau_{21}$, $\tau_{32}$) and jet charge $\mathcal{Q}$ (a cross-modal correlation metric), indicating that the Transformer architecture better captures inter-particle correlations.
- EPiC-FM performs better on rare lepton multiplicities ($e^\pm$, $\mu^\pm$), which constitute only a few per mille of the training data.
- Temperature scaling at $T=0.85$ is critical for sample quality; deviating from this value systematically distorts the neutral hadron distribution.

## Highlights & Insights

- **Theoretical Elegance**: Continuous flow matching and discrete Markov jump bridges are unified within a single probabilistic path framework—conditionally independent yet marginally coupled.
- **Analytical Tractability of the Discrete Modality**: The expected jump rates are analytically computable, reducing discrete generation to a classification problem.
- **Time-Adaptive Loss Weighting**: Outperforms fixed weighting by dynamically balancing the training of both modalities.
- First work to jointly generate particle kinematics and flavor on real CMS data.

## Limitations & Future Work

- Generation quality for rare particle classes (leptons) is suboptimal; post-hoc calibration or oversampling strategies may be needed.
- The irregular shape of the $p_T$ peak is difficult for both methods to reproduce faithfully.
- The temperature hyperparameter $T$ is sensitive and requires careful tuning.
- The work serves as a proof of concept; no comprehensive architecture or hyperparameter optimization has been performed.

## Related Work & Insights

- Pure discrete flow methods (Gat et al.) and multimodal approaches in protein design (Campbell et al.) adopt different conditional path constructions.
- Generator matching (Holderrieth et al.) provides a more general theoretical framework, of which the proposed method can be viewed as a special case.
- The framework offers broader inspiration for scientific problems requiring joint modeling of continuous–discrete data, such as molecular generation and materials design.

## Rating

- Novelty: ⭐⭐⭐⭐ — The theoretical construction of multimodal flow matching is elegant, and the application of Markov jump bridges in generative modeling is novel.
- Experimental Thoroughness: ⭐⭐⭐ — Validation on real CMS data is credible, but only one baseline is compared and ablation studies are absent.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are clear and complete, with detailed supplementary material.
- Value: ⭐⭐⭐⭐ — Provides a general framework for mixed-modality generation in scientific AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[NeurIPS 2025\] FALCON: Few-step Accurate Likelihoods for Continuous Flows](falcon_few-step_accurate_likelihoods_for_continuous_flows.md)
- [\[NeurIPS 2025\] Amortized Sampling with Transferable Normalizing Flows](amortized_sampling_with_transferable_normalizing_flows.md)
- [\[NeurIPS 2025\] Show-o2: Improved Native Unified Multimodal Models](show-o2_improved_native_unified_multimodal_models.md)
- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](mmada_multimodal_large_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
