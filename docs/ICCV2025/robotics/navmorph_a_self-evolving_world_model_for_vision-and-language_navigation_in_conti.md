---
title: >-
  [Paper Note] NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments
description: >-
  [ICCV 2025][Robotics][Vision-and-Language Navigation] This paper proposes NavMorph, an RSSM-based **self-evolving world model** that models continuous environment dynamics in latent space via a World-aware Navigator and…
tags:
  - "ICCV 2025"
  - "Robotics"
  - "Vision-and-Language Navigation"
  - "World Model"
  - "Continuous Environments"
  - "Self-Evolving"
  - "RSSM"
date: 2026-05-08
content_hash: 2a4b0227fef4357f
---

# NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments

**Conference**: ICCV 2025
**arXiv**: [2506.23468](https://arxiv.org/abs/2506.23468)  
**Code**: [https://github.com/Feliciaxyao/NavMorph](https://github.com/Feliciaxyao/NavMorph)  
**Area**: Robotic Navigation / Embodied Intelligence
**Keywords**: Vision-and-Language Navigation, World Model, Continuous Environments, Self-Evolving, RSSM

## TL;DR

This paper proposes NavMorph, an RSSM-based **self-evolving world model** that models continuous environment dynamics in latent space via a World-aware Navigator and a Foresight Action Planner, and introduces a Contextual Evolution Memory (CEM) for rapid online test-time adaptation.

## Background & Motivation

VLN-CE (Vision-and-Language Navigation in Continuous Environments) requires agents to execute low-level actions (move forward 0.25 m, rotate 15°, etc.) in real 3D environments. Unlike discrete VLN, continuous environments feature vast and continuous state spaces, posing unique challenges:

**Limitations of existing world models**: Prior works such as PathDreamer and DreamWalker either support only discrete state transitions or rely on pixel-level prediction (computationally expensive), failing to effectively model continuous spatiotemporal dynamics.

**Insufficient online adaptation**: These world models are fixed after pretraining and suffer performance degradation under distributional shift in unseen environments.

**Lack of foresight planning**: Conventional VLN methods make decisions based on current observations, without the ability to anticipate future environmental changes.

The core idea of NavMorph is inspired by human "mental representations": constructing a **continuously evolving model of environment dynamics** in latent space, enabling agents to plan ahead and adapt online.

## Method

### Overall Architecture

NavMorph consists of three core components:
- **World-aware Navigator (inference network)**: Infers the latent environment state from observations.
- **Foresight Action Planner (prediction network)**: Predicts future scenes and makes decisions based on latent states.
- **Contextual Evolution Memory (CEM)**: Accumulates navigation experience across episodes to support online self-evolution.

### Key Designs

#### 1. Latent Space Modeling via RSSM

A Recurrent State-Space Model is adopted to decompose the latent state into two components:
- **Deterministic history $\mathbf{h}_t$**: Encodes temporal dynamics through a recurrent module, $\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{s}_{t-1})$
- **Stochastic state $\mathbf{s}_t$**: Models environmental uncertainty.

The inference network samples from the posterior distribution:
$$q(\mathbf{s}_t | \mathbf{o}_{1:t}, \mathbf{a}_{1:t-1}) \sim \mathcal{N}(\mu_\phi(\mathbf{h}_t, \mathbf{a}_{t-1}, \mathbf{x}_t), \sigma_\phi(\mathbf{h}_t, \mathbf{a}_{t-1}, \mathbf{x}_t)\mathbf{I})$$

The prediction network samples from the prior distribution:
$$p(\hat{\mathbf{s}}_t | \mathbf{h}_t, \hat{\mathbf{s}}_{t-1}) \sim \mathcal{N}(\mu_\theta(\mathbf{h}_t, \hat{\mathbf{a}}_{t-1}), \sigma_\theta(\mathbf{h}_t, \hat{\mathbf{a}}_{t-1})\mathbf{I})$$

A key distinction is that the inference network has access to real observations, whereas the prediction network relies solely on past latent states and predicted actions. Actions are defined as $\Delta position_t$ (displacement between consecutive steps) rather than low-level control commands.

#### 2. Contextual Evolution Memory (CEM)

CEM maintains $N_m$ scene context features $\{\mathbf{v}_m\}_{m=1}^{N_m}$ as the core augmentation of the recurrent module, replacing conventional RNN/LSTM:

**Retrieval augmentation**: Top-K relevant features are retrieved from CEM and fused into the current state:
$$\tilde{\mathbf{h}}_t = (1-\alpha)\mathbf{h}_t + \alpha \sum_{k=1}^{K} \mu_k \mathbf{v}_k$$

**Forward update** (without gradient backpropagation):
$$\mathbf{v}_k \leftarrow (1-\beta)\mathbf{v}_k + \beta \mathbf{h}_t$$

This design offers two key advantages:
- **During training**: Navigation knowledge across multiple environments is accumulated via gradient optimization.
- **During testing**: Rapid adaptation to new environments is achieved through forward updates, without backpropagation, yielding high efficiency.

#### 3. Feature-Level Prediction Instead of Pixel-Level

The visual decoder $d_\theta$ predicts **visual embeddings** $\hat{\mathbf{x}}_t$ (feature vectors) rather than pixel images. This avoids the high computational cost of generative models while retaining sufficient semantic information for action planning.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_W + \mathcal{L}_{IL}$, where the world model loss is:

$$\mathcal{L}_W = \underbrace{\ell_{re}}_{\text{reconstruction}} + \underbrace{\ell_{ac}}_{\text{action prediction}} + \gamma \cdot \underbrace{\ell_{kl}}_{\text{KL divergence}}$$

- $\ell_{re}$: Visual embedding reconstruction (with NDTW regularization to ensure temporal consistency).
- $\ell_{ac}$: Action prediction (L2 loss + NDTW regularization).
- $\ell_{kl}$: Alignment between posterior and prior distributions (closed-form computation).
- $\mathcal{L}_{IL}$: Imitation learning loss (DAgger-style teacher action supervision).

During pretraining, the model observes $T$ steps and predicts the subsequent $T_p$ steps, forming a two-phase "observe–predict" training scheme.

## Key Experimental Results

### Main Results (R2R-CE Val Unseen)

| Method | Camera | NE↓ | SR(%) | SPL(%) |
|--------|--------|:---:|:---:|:---:|
| VLN-3DFF | Monocular | 6.05 | 43.8 | 29.4 |
| **NavMorph** | **Monocular** | **5.75** | **47.9** | **33.2** |
| ETPNav | Panoramic | 4.69 | 57 | 49 |
| **NavMorph** | **Panoramic** | **4.62** | **59** | **50** |
| HNR | Panoramic | 4.57 | 61 | 51 |
| **NavMorph** | **Panoramic** | **4.37** | **64** | **53** |

### Ablation Study (R2R-CE Val Unseen, Monocular VLN-3DFF Baseline)

| Method | TL↓ | NE↓ | SR(%) | SPL(%) |
|--------|:---:|:---:|:---:|:---:|
| Base model | 26.16 | 6.05 | 43.77 | 29.39 |
| NavMorph (full) | **22.54** | **5.75** | **47.91** | **33.22** |
| w/o $\ell_{re}$ | 20.25 | 5.85 | 45.51 | 32.38 |
| w/o $\ell_{ac}$ | 25.14 | 5.96 | 44.81 | 31.22 |
| w/o $\ell_{kl}$ | 25.69 | 6.30 | 44.10 | 30.44 |
| NavMorph w/o SE | 23.34 | 5.92 | 45.08 | 31.19 |
| CEM vs LSTM | CEM: 21.22s | - | 47.91 | 33.22 |
| | LSTM: 44.56s | - | 43.67 | 29.81 |

### Key Findings

1. NavMorph achieves **+4.1% SR** and **+3.8% SPL** in the monocular setting while **reducing trajectory length by 14%**, indicating more efficient planning.
2. CEM self-evolves **2.1× faster** than LSTM and outperforms it across all metrics, validating forward updates as a superior adaptation strategy over gradient backpropagation.
3. The optimal CEM size is 1,000: too small limits storage capacity, while too large introduces noisy redundancy.
4. Even without self-evolution (w/o SE), the world model architecture alone contributes +1.3% SR / +1.8% SPL gains.

## Highlights & Insights

- **Design philosophy of CEM**: The momentum-style forward update eliminates the computational overhead of test-time backpropagation, achieving truly zero-extra-training-cost online adaptation.
- Feature-level prediction avoids the complexity of pixel generation, while NDTW regularization ensures temporal coherence—a pragmatic engineering trade-off.
- The decoupled design of the world model and navigation policy allows NavMorph to serve as a plug-and-play module to enhance various existing VLN methods.

## Limitations & Future Work

- CEM still relies on cosine similarity for retrieval, which limits the depth of semantic understanding.
- The reward function lacks direct signals from ground-truth goals, precluding explicit planning (acknowledged by the authors).
- The sensitivity of forward-update hyperparameters $\alpha$ and $\beta$ has not been thoroughly investigated.
- Evaluation is conducted solely in the Matterport3D environment; transferability to real robotic systems remains unverified.

## Related Work & Insights

- NavMorph follows the world model design lineage of the Dreamer series (Hafner et al.), representing the first systematic adaptation to the VLN-CE task.
- The CEM mechanism shares conceptual similarities with memory-augmented RL approaches such as Neural Episodic Control and MERLIN.
- This work inspires future research directions, including integrating foundation models (e.g., VLMs) as components of world models, or designing modular and composable evolving architectures.

## Rating

- Novelty: ⭐⭐⭐⭐ (Self-evolving world model + novel CEM design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (R2R-CE / RxR-CE + multiple baselines + comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear architecture diagrams, complete derivations)
- Value: ⭐⭐⭐⭐ (Clear advancement for the VLN-CE research community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World](../../NeurIPS2025/robotics/coopera_continual_open-ended_human-robot_assistance.md)
- [\[ICCV 2025\] COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation](cosmo_combination_of_selective_memorization_for_low-cost_vision-and-language_nav.md)
- [\[ICCV 2025\] DexVLG: Dexterous Vision-Language-Grasp Model at Scale](dexvlg_dexterous_vision-language-grasp_model_at_scale.md)
- [\[ICML 2026\] Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model](../../ICML2026/robotics/dual-stream_diffusion_for_world-model_augmented_vision-language-action_model.md)
- [\[ICCV 2025\] CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games](combatvla_an_efficient_vision-language-action_model_for_combat_tasks_in_3d_actio.md)

</div>

<!-- RELATED:END -->
