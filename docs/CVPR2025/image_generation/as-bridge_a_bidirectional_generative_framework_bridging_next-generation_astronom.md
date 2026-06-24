---
title: >-
  [Paper Note] AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys
description: >-
  [CVPR 2025][Image Generation][Brownian Bridge] This work proposes AS-Bridge, a bidirectional Brownian Bridge diffusion model designed to model the stochastic mapping relationship between two major astronomical surveys: the ground-based LSST and the space-based Euclid. It achieves probabilistic cross-survey translation and rare event detection (strong gravitational lensing), and demonstrates that the epsilon-prediction training objective benefits both reconstruction quality an…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Brownian Bridge"
  - "Astronomical Surveys"
  - "Cross-Modal Translation"
  - "Anomaly Detection"
  - "Probabilistic Reconstruction"
date: 2026-05-08
content_hash: 3cb408261346cfc3
---

# AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys

**Conference**: CVPR 2025  
**arXiv**: [2603.11928](https://arxiv.org/abs/2603.11928)  
**Code**: [https://github.com/ZHANG7DC/AS-Bridge](https://github.com/ZHANG7DC/AS-Bridge)  
**Area**: Image Generation / Astronomical Image Translation  
**Keywords**: Brownian Bridge, Astronomical Surveys, Cross-Modal Translation, Anomaly Detection, Probabilistic Reconstruction

## TL;DR

This work proposes AS-Bridge, a bidirectional Brownian Bridge diffusion model designed to model the stochastic mapping relationship between two major astronomical surveys: the ground-based LSST and the space-based Euclid. It achieves probabilistic cross-survey translation and rare event detection (strong gravitational lensing), and demonstrates that the epsilon-prediction training objective benefits both reconstruction quality and likelihood estimation.

## Background & Motivation

**Background**: Over the next decade, astronomical observations will be dominated by two flagship surveys: LSST (ground-based, multi-band optical, ~0.7 arcsec seeing) and Euclid (space-based, high-resolution near-infrared, 0.1 arcsec pixel size), which overlap over an area of approximately 7,000–9,000 $\text{deg}^2$.

**Limitations of Prior Work**: The two surveys exhibit vastly different PSFs, noise statistics, bandpasses, and scanning strategies, making direct joint analysis highly challenging.

**Key Challenge**: Cross-survey mapping is ill-posed in both directions: LSST-to-Euclid requires recovering details blurred by atmospheric seeing, while Euclid-to-LSST requires inferring multi-band spectra from a single band. Deterministic mapping struggles to capture such inherent ambiguities.

**Goal**: To build a unified bidirectional generative framework that models the probabilistic conditional distribution between the two surveys, and leverage reconstruction inconsistency for detecting rare astronomical events.

**Key Insight**: To model cross-survey translation as a Brownian Bridge diffusion process, anchoring both ends of the bridge to the respective data distributions of the two surveys.

**Core Idea**: Astronomical survey translation is inherently a bidirectional stochastic process. Thus, the Brownian Bridge is naturally suited for modeling the probabilistic relationship between two imperfect observations.

## Method

### Overall Architecture

The two survey observations are treated as noisy projections of the same unobservable astrophysical process. AS-Bridge establishes a stochastic path between them using a Brownian Bridge: the endpoints correspond to the observations of the two surveys, and the intermediate states are defined under a Gaussian distribution. Training is performed on paired data from the overlapping sky region.

### Key Designs

1. **Bidirectional Brownian Bridge Diffusion**

    - **Function**: Trains a bidirectional generative model on the overlapping sky region.
    - **Mechanism**: The forward process performs stochastic interpolation from source to target (instead of mapping data to pure noise), and the reverse process utilizes Bayesian posterior transitions.
    - **Design Motivation**: Standard diffusion requires passing through high-noise states, which is inefficient. The Brownian Bridge directly interpolates between the two distributions.

2. **Epsilon-prediction Maximum Likelihood Training**

    - **Function**: Replaces the traditional bridge loss with a noise prediction loss.
    - **Mechanism**: Proves that the $\epsilon$-loss is equivalent to the standard loss scaled by a weight of $\sqrt{\Delta_t}$, balancing both high-noise likelihood and endpoint stability.
    - **Design Motivation**: Directly using $\Delta_t$ weighting results in weights tending to zero near the endpoints.

3. **Cross-Survey Rare Event Detection**

    - **Function**: Translates the detection of rare objects (e.g., strong gravitational lensing) into reconstruction inconsistency.
    - **Mechanism**: Uses midpoint fusion combined with reverse reconstruction, multiple sampling to find the minimum error, and flux normalization.
    - **Design Motivation**: The model learns representations of "normal" astronomical objects well, whereas reconstruction failure of rare events generates a strong signal.

### Loss & Training

- Epsilon-prediction loss, equivalent to the standard bridge loss weighted by $\sqrt{\Delta_t}$.
- Flux normalization of anomaly scores to prevent bias towards bright sources.
- Dataset: 115K regular galaxies + 5K strong gravitational lensing events.

## Key Experimental Results

### Main Results

| Method | CRPS (L to E) | CRPS (E to L) |
|------|------------|------------|
| SPADE | 3.39 | 16.52 |
| OASIS | 4.65 | 13.33 |
| Pix2Pix | 4.35 | 73.03 |
| Palette | 2.43 | 7.98 |
| Joint Diffusion | 3.14 | 15.15 |
| AS-Bridge (epsilon) | **2.38** | **7.90** |

### Ablation Study

| Training Objective | CRPS (L to E) | CRPS (E to L) |
|---------|-------------|-------------|
| Standard Bridge Loss | 2.55 | 7.90 |
| $\sqrt{\delta} \times \epsilon$ | 3.59 | 11.24 |
| epsilon (Ours) | **2.38** | **7.90** |

| Anomaly Detection | FPR@1%TPR | FPR@5%TPR | AUPR |
|---------|----------|----------|------|
| AS-Bridge | **0.00%** | **0.18%** | **0.80** |
| Deco-Diff | 1.1% | 5.0% | 0.61 |
| CFM | 0.24% | 1.2% | 0.75 |

### Key Findings

- Diffusion and Bridge methods consistently outperform non-diffusion baselines.
- Epsilon-prediction achieves the best CRPS in both directions.
- Multi-modal anomaly detection significantly outperforms uni-modal approaches.
- LSST-to-Euclid translation successfully recovers the multi-object structure of blended sources.

## Highlights & Insights

- Translates cross-astronomical survey translation into a probabilistic inference framework for the first time.
- Provides a rigorous proof of equivalence between epsilon-prediction and the weighted standard bridge loss.
- Anomaly detection does not require annotations of anomalous samples.

## Limitations & Future Work

- Validated only on simulated data; a sim-to-real gap remains.
- Anomaly detection is evaluated only on strong gravitational lensing.
- The dataset scale is limited.

## Related Work & Insights

- Brownian Bridge Diffusion offers a new paradigm to replace conditional diffusion.
- The framework can be generalized to other multi-modal astronomical data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to apply the Brownian Bridge to astronomical survey translation.
- **Experimental Thoroughness**: ⭐⭐⭐ Thorough simulations but lacks real-world data.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and rigorous theory.
- **Value**: ⭐⭐⭐⭐ Inspiring for both astronomy and cross-modal translation fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[CVPR 2025\] DreamRelation: Bridging Customization and Relation Generation](dreamrelation_bridging_customization_and_relation_generation.md)
- [\[CVPR 2025\] DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](diffsensei_bridging_multi-modal_llms_and_diffusion_models_for_customized_manga_g.md)
- [\[CVPR 2025\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)
- [\[NeurIPS 2025\] Coupling Generative Modeling and an Autoencoder with the Causal Bridge](../../NeurIPS2025/image_generation/coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)

</div>

<!-- RELATED:END -->
