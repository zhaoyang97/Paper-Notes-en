---
title: >-
  [Paper Note] AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys
description: >-
  [CVPR 2026][Image Generation][Astronomical Surveys] AS-Bridge is proposed to model the conditional probability distribution between ground-based LSST and space-based Euclid survey observations using a bidirectional Brownian Bridge diffusion process, enabling cross-survey probabilistic image translation and unsupervised strong gravitational lens detection by leveraging reconstruction inconsistency.
tags:
  - CVPR 2026
  - Image Generation
  - Astronomical Surveys
  - Brownian Bridge
  - Bidirectional Image Translation
  - Rare Event Detection
  - Probabilistic Reconstruction
date: 2026-05-08
content_hash: d73d057f0a9b9843
---

# AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys

**Conference**: CVPR 2026  
**arXiv**: [2603.11928](https://arxiv.org/abs/2603.11928)  
**Code**: [github.com/ZHANG7DC/AS-Bridge](https://github.com/ZHANG7DC/AS-Bridge)  
**Area**: Astronomical Imaging / Generative Models / Cross-domain Translation  
**Keywords**: Astronomical Surveys, Brownian Bridge, Bidirectional Image Translation, Rare Event Detection, Probabilistic Reconstruction

## TL;DR

AS-Bridge is proposed to model the conditional probability distribution between ground-based LSST and space-based Euclid survey observations using a bidirectional Brownian Bridge diffusion process, enabling cross-survey probabilistic image translation and unsupervised strong gravitational lens detection by leveraging reconstruction inconsistency.

## Background & Motivation

**Background**: Next-generation astronomical observations are dominated by LSST (ground-based, 6 optical bands, ~0.7" resolution, atmospheric interference) and Euclid (space-based, high-resolution near-infrared, VIS pixel 0.1"), with an overlapping sky area of approximately 7000–9000 square degrees.

**Limitations of Prior Work**: Systematic distribution shifts in PSF, bands, and noise statistics make joint cross-survey analysis difficult. A single deterministic mapping cannot capture inherent ambiguities—LSST→Euclid requires recovering fine morphology from atmospheric blur (ill-posed), and Euclid→LSST requires inferring multi-band color from fewer bands (unidentifiable).

**Key Challenge**: Observations of the same celestial object by two surveys are two stochastic realizations $x = \mathcal{O}(\Phi) + \epsilon(\mathcal{O}(\Phi))$ sharing a latent astrophysical process $\Phi$. Since $\Phi$ is not directly observable, it is necessary to learn the conditional distribution $p(x_{Euclid}|x_{LSST})$ and its inverse through marginalization.

**Key Insight**: Brownian Bridge defines a stochastic interpolation process between two endpoints, naturally suited for modeling probabilistic relationships between two observation domains.

**Core Idea**: Use a bidirectional Brownian Bridge diffusion process to model the conditional distribution between surveys, and utilize reconstruction failure on OOD rare objects to achieve unsupervised anomaly detection.

## Method

### Overall Architecture

Extract paired images from LSST-Euclid overlapping regions → Model cross-survey translation as a bidirectional Brownian Bridge process → Share a single diffusion model, achieving bidirectional inference by selecting the bridge's start/end points → Faithfully reconstruct routine objects while failing on rare objects (strong gravitational lenses) → Use the inconsistency of multiple sampled reconstructions as the anomaly score.

### Key Designs

1. **Brownian Bridge Diffusion Process**

    - **Function**: Establish a stochastic path between two survey observation domains, replacing the "data → pure noise" path of standard diffusion.
    - **Mechanism**: Given endpoints $(x_0, x_T)$, the intermediate state $x_t | (x_0, x_T) \sim \mathcal{N}((1-m_t)x_0 + m_t x_T, \delta_t I)$, where $m_t = t/T$ and $\delta_t = m_t(1-m_t)$. Reverse transitions are derived via Bayes' theorem, preserving the source-target conditional dependency.
    - **Design Motivation**: Standard conditional diffusion (e.g., Palette) starts from pure noise, with the source image acting only as an external condition; BB interpolates directly between the two domains, avoiding high-noise states, improving sampling efficiency, and better maintaining conditional distributions.

2. **$\epsilon$-prediction Maximum Likelihood Training**

    - **Function**: Prove that the $\epsilon$-prediction loss is equivalent to the standard BB loss multiplied by a mild weight $\sqrt{\delta_t}$.
    - **Mechanism**: Under variance-exploding diffusion, the likelihood objective requires time-step weights $\delta_t$, but in BB, $\delta_t$ tends to zero at endpoints, causing vanishing gradients. The $\epsilon$-prediction loss $\|\epsilon_\theta - \epsilon\|_2^2$ is equivalent to score matching with weight $\sqrt{\delta_t}$, which emphasizes likelihood-oriented high-noise time steps while maintaining stable gradients at endpoints.
    - **Design Motivation**: Direct use of $\delta_t$ weighting causes vanishing gradients at endpoints during BB training; $\epsilon$-prediction provides a moderate intermediate solution.

3. **Multi-sampling Anomaly Detection**

    - **Function**: Utilize the model's reconstruction failure on out-of-distribution objects to discover rare events without anomaly labels.
    - **Mechanism**: For a paired observation $(x_{Euclid}, x_{LSST})$, perform multiple stochastic reconstructions after fusion via the forward process. The pixel-level anomaly map is the minimum per-pixel error across reconstructions $\mathcal{A}(p) = \min_i \|\hat{x}_0^{(i)}(p) - x_0(p)\|_2^2$; image-level scores are normalized by flux to eliminate brightness bias.
    - **Design Motivation**: The low SNR of astronomical images causes single reconstruction errors to be dominated by noise; taking the minimum of multiple samples suppresses noise fluctuations and preserves systematic reconstruction failure signals.

### Loss & Training

The training loss is $\epsilon$-prediction MSE. Data is based on SLSim-simulated 115K routine galaxies + 5K strong gravitational lens paired images (64×64), including Euclid VIS single band + LSST gri three bands.

## Key Experimental Results

### Main Results

| Direction/Task | Metric | AS-Bridge | Palette | SPADE | pix2pix | Joint Diffusion |
|---|---|---|---|---|---|---|
| LSST→Euclid | CRPS↓ | **2.38** | 2.43 | 3.39 | 4.35 | 3.14 |
| Euclid→LSST | CRPS↓ | **7.90** | 7.98 | 16.52 | 73.03 | 15.15 |

| Anomaly Detection Method | FPR@1%TPR↓ | FPR@5%TPR↓ | AUPR↑ |
|---|---|---|---|
| AS-Bridge | **0.00%** | **0.18%** | **0.80** |
| CFM (Cross-modal) | 0.24% | 1.20% | 0.75 |
| Deco-Diff (Uni-modal) | 1.10% | 5.00% | 0.61 |

### Ablation Study

| Training Objective | LSST→Euclid CRPS↓ | Euclid→LSST CRPS↓ |
|---|---|---|
| $\epsilon$-prediction (Ours) | **2.38** | **7.90** |
| Standard BB loss | 2.55 | 8.12 |
| $\delta_t$-weighted loss | 2.51 | 8.30 |

### Key Findings

- Diffusion/Bridge methods consistently outperform GANs in probabilistic reconstruction—GANs tend toward mode collapse, which is unsuitable for astronomical probabilistic inference.
- Euclid→LSST CRPS is approximately 3.3 times that of the reverse direction—inferring multi-band colors from a single wide band is inherently more difficult.
- Uni-modal anomaly detection (Deco-Diff) fails completely; cross-modal information is crucial for rare event detection.
- Taking the minimum error from multiple random reconstructions effectively suppresses the noise-dominated issues in low-SNR astronomical images.

## Highlights & Insights

- Formalizes joint astronomical survey analysis as a probabilistic image translation problem, elegantly handling cross-domain mapping uncertainty—the methodology is generalizable to any multi-sensor remote sensing scenario.
- Rare event detection is completely unsupervised—leveraging the model's systematic reconstruction failure on out-of-distribution samples without anomaly labels. Proposes "discovery-oriented" evaluation metrics (FPR@lowTPR, AUPR) instead of high-recall metrics used in industrial anomaly detection.
- The $\epsilon$-prediction equivalence proof bridges BB training with maximum likelihood principles via a concise 3-line proof.

## Limitations & Future Work

- Entirely based on simulated data (SLSim), making the simulation-to-reality gap inevitable—re-validation on real LSST/Euclid data is required after its release.
- 64×64 resolution is too low for practical scientific analysis and needs to be scaled to higher resolutions.
- Only LSST(gri) and Euclid(VIS) sub-bands were considered; joint modeling of all bands remains unexplored.
- Anomaly detection was only validated for strong gravitational lenses; generalizability to other rare astronomical events is unknown.

## Related Work & Insights

- **vs Palette**: Conditional diffusion starts from pure noise, whereas BB establishes a direct stochastic path between source and target, improving sampling efficiency.
- **vs BBDM**: Shares the BB framework, but this work adds $\epsilon$-prediction theoretical improvements and astronomical scientific application validation.
- **vs CFM**: Cross-modal feature mapping relies on explicit feature fusion modules, whereas AS-Bridge performs implicit fusion through generative reconstruction.

## Rating

- Novelty: ⭐⭐⭐⭐ Creative combination of survey probabilistic translation + reconstruction-based anomaly detection.
- Experimental Thoroughness: ⭐⭐⭐ Simulation validation is sufficient, but lacks real data.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formalization and solid astrophysical background.
- Value: ⭐⭐⭐⭐ High potential for direct application once LSST/Euclid data becomes available.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[CVPR 2026\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)
- [\[CVPR 2026\] Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models](probing_and_bridging_geometry-interaction_cues_for_affordance_reasoning_in_visio.md)
- [\[CVPR 2026\] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)
- [\[CVPR 2026\] SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models](segquant_diffusion_model_quantization.md)

<!-- RELATED:END -->
