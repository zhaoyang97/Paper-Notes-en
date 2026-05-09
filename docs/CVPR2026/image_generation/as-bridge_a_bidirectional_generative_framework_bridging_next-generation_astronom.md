---
title: >-
  [Paper Note] AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys
description: >-
  [CVPR 2026][Image Generation][Astronomical Surveys] The authors propose AS-Bridge, a bidirectional generative framework based on the Brownian Bridge diffusion process. It models the probabilistic conditional distribution between ground-based LSST and space-based Euclid astronomical surveys, enabling cross-survey image translation and rare event detection (gravitational lensing), while improving likelihood estimation of the standard Brownian Bridge via an $\epsilon$-prediction training objective.
tags:
  - CVPR 2026
  - Image Generation
  - Astronomical Surveys
  - Brownian Bridge
  - Cross-modal Translation
  - Anomaly Detection
  - Probabilistic Generation
date: 2026-05-08
content_hash: 4618ff8e42669bc0
---

# AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys

**Conference**: CVPR 2026  
**arXiv**: [2603.11928](https://arxiv.org/abs/2603.11928)  
**Code**: [Available](https://github.com/ZHANG7DC/AS-Bridge)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Astronomical Surveys, Brownian Bridge, Cross-modal Translation, Anomaly Detection, Probabilistic Generation

## TL;DR
The authors propose AS-Bridge, a bidirectional generative framework based on the Brownian Bridge diffusion process. It models the probabilistic conditional distribution between ground-based LSST and space-based Euclid astronomical surveys, enabling cross-survey image translation and rare event detection (gravitational lensing), while improving likelihood estimation of the standard Brownian Bridge via an $\epsilon$-prediction training objective.

## Background & Motivation
**Background**: Observational cosmology in the next decade will be driven by large-scale surveys: the ground-based LSST (Vera C. Rubin Observatory) provides deep multi-band optical images but suffers from limited resolution and source blending due to atmospheric turbulence; the space-based Euclid provides high-resolution near-infrared imaging but with fewer bands and incomplete spectral information. The two surveys share approximately 7,000-9,000 deg² of overlapping sky area, observing the same celestial objects but producing fundamentally different data.

**Limitations of Prior Work**: Cross-survey inference is an ill-posed problem in both directions: recovering Euclid-level morphology from LSST requires resolving ambiguities from atmospheric blurring and background noise; mapping from Euclid back to LSST requires inferring spectral information from fewer bands. Therefore, cross-survey translation should be viewed as a probabilistic process capable of sampling multiple valid realizations consistent with existing observations.

**Key Challenge**: Existing cross-modal methods (GAN-based, conditional diffusion) are typically developed and evaluated under a single-direction deterministic paradigm, failing to faithfully represent the full conditional distribution between observational modalities. Scientific applications require probabilistic generation with uncertainty quantification.

## Method

### Overall Architecture
AS-Bridge models cross-survey translation as a bidirectional Brownian Bridge process. Utilizing paired observations from overlapping sky areas as anchors for training, it learns stochastic paths between LSST and Euclid data distributions. Once trained, it can generate complementary observations in non-overlapping regions and be used for rare event detection.

### Key Designs
1.  **Survey Translation Formulation**:
    - **Function**: Treats observations from two surveys as two different realizations of a shared latent astrophysical process $\Phi$.
    - **Mechanism**: $x_{\text{Euclid}} = \mathcal{O}_{\text{Euclid}}(\Phi) + \epsilon_{\text{Euclid}}$, $x_{\text{LSST}} = \mathcal{O}_{\text{LSST}}(\Phi) + \epsilon_{\text{LSST}}$. Since the latent process is unobservable, the model directly learns the conditional distributions $p(x_{\text{Euclid}} | x_{\text{LSST}})$ and $p(x_{\text{LSST}} | x_{\text{Euclid}})$ by marginalizing $\Phi$.
    - **Design Motivation**: Observations are merely noisy projections of the underlying scene; the mapping is inherently stochastic rather than deterministic.

2.  **Brownian Bridge with $\epsilon$-prediction**:
    - **Function**: Derives an improved training objective within the standard Brownian Bridge framework.
    - **Mechanism**: The forward process of a standard Brownian Bridge is: $x_t | (x_0, x_T) \sim \mathcal{N}((1-m_t)x_0 + m_t x_T, \delta_t I)$, where $\delta_t = m_t(1-m_t)$. Standard training losses directly predict drift + denoising terms. The authors prove that $\epsilon$-prediction is equivalent to the standard loss multiplied by a $\sqrt{\delta_t}$ weight:
      $$\mathcal{L} = \|\epsilon_\theta - \epsilon\|_2^2$$
      This preserves the likelihood-inspired emphasis on high-noise timesteps while maintaining stable gradients near the bridge endpoints (avoiding gradient vanishing caused by direct $\delta_t$ weighting). The reconstruction objective is:
      $$\hat{x}_0 = \frac{x_t - m_t x_T - \sqrt{\delta_t} \epsilon_\theta(x_t, x_T, t)}{1-m_t}$$
    - **Design Motivation**: Scientific problems require models to faithfully match conditional probability distributions; direct $\delta_t$ weighting leads to gradient vanishing at endpoints, while $\sqrt{\delta_t}$ provides a gentler weighting.

3.  **Rare Event Detection**:
    - **Function**: Utilizes cross-survey reconstruction inconsistency for unsupervised anomaly detection.
    - **Mechanism**: Paired observations are fused through the forward process to generate an intermediate variable $x_t$, which is then reconstructed back to the Euclid domain. $N$ random reconstructions $\{\hat{x}_0^{(i)}\}_{i=1}^N$ are sampled, and the pixel-level anomaly score is defined as the minimum reconstruction error:
      $$\mathcal{A}(p) = \min_{i \in \{1,...,N\}} \|\hat{x}_0^{(i)}(p) - x_0(p)\|_2^2$$
      The image-level score is aggregated via flux normalization: $\mathcal{A}(x_0) = \frac{\sum_p \mathcal{A}(p)}{\sum_p x_0(p)}$
    - **Design Motivation**: Rare events (e.g., strong gravitational lensing) are underrepresented in the training distribution, causing the model to fail in faithful reconstruction; reconstruction inconsistency serves as an anomaly signal, and taking the minimum error suppresses false positives from noise fluctuations.

### Loss & Training
- **Training Data**: 115,000 normal galaxies + 5,000 strong lensing systems simulated using SLSim.
- **LSST Images**: g/r/i bands, 64×64 pixels, ~0.7" seeing.
- **Euclid Images**: VIS band, 0.1" pixel scale, 64×64 pixels.
- 110,000 normal galaxies used for training, the rest for evaluation.

## Key Experimental Results

### Main Results (Probabilistic Reconstruction Quality CRPS↓)

| Method | LSST→Euclid | Euclid→LSST |
| :--- | :--- | :--- |
| SPADE | 3.39 | 16.52 |
| OASIS | 4.65 | 13.33 |
| Pix2Pix | 4.35 | 73.03 |
| Palette | 2.43 | 7.98 |
| Joint Diffusion | 3.14 | 15.15 |
| BB Standard Loss | 2.55 | 7.90 |
| **AS-Bridge ($\epsilon$-pred)** | **2.38** | **7.90** |

### Ablation Study

| Training Objective | CRPS (LSST→Euclid) | CRPS (Euclid→LSST) | Note |
| :--- | :--- | :--- | :--- |
| Standard Loss | 2.55 | 7.90 | Original BB objective |
| $\sqrt{\delta_t}$ Weight | 3.59 | 11.24 | Direct weighting performs worse |
| **$\epsilon$-pred** | **2.38** | **7.90** | Gentle weighting is optimal |

### Anomaly Detection (Strong Gravitational Lensing)

| Method | FPR@1%TPR↓ | FPR@5%TPR↓ | AUPR↑ |
| :--- | :--- | :--- | :--- |
| **AS-Bridge** | **0.00%** | **0.18%** | **0.80** |
| Deco-Diff | 1.1% | 5.0% | 0.61 |
| CFM | 0.24% | 1.2% | 0.75 |

### Key Findings
- Diffusion/Bridge methods significantly outperform non-diffusion methods (GAN-based), validating the advantages of score-based generative modeling in recovering true conditional distributions.
- Euclid→LSST (inferring multi-band color from a single band) is an extremely ill-posed problem, yet the model generates diverse reconstructions with consistent morphology and plausible colors.
- LSST→Euclid translation correctly restores the number and positions of galaxies in multi-source systems blended by atmospheric seeing.
- The single-modality method Deco-Diff fails to detect structural anomalies, highlighting that cross-modal information is crucial for rare event detection.

## Highlights & Insights
- Formulates cross-survey translation as a probabilistic inference problem for the first time, rather than a simple I2I translation.
- The formal proof of $\epsilon$-prediction equivalence is elegant and practical, providing theoretical guidance for Brownian Bridge training.
- Using reconstruction inconsistency for unsupervised anomaly detection is a clever scientific application—leveraging the "epistemic boundaries" of generative models to discover new phenomena.
- The design of evaluation metrics (CRPS for probabilistic quality, FPR at low TPR for scientific discovery) reflects a deep understanding of domain requirements.

## Limitations & Future Work
- Currently trained and evaluated only on simulated data; the sim-to-real domain gap remains a known limitation.
- CRPS for the Euclid→LSST direction remains relatively high (7.90), indicating significant uncertainty in multi-band color inference.
- Validated only with strong gravitational lensing as a proxy for anomalies; more types of rare celestial objects are needed.
- Fixed image size of 64×64 may be insufficient for modeling large-scale structures.

## Related Work & Insights
- Core difference from Palette (conditional diffusion I2I): Palette reverses from pure noise with the source image as a conditional signal; BB models a stochastic path directly between two distributions.
- The cross-modal anomaly detection idea can be generalized to other multi-sensor astronomical data (e.g., SKA radio + optical).
- The $\sqrt{\delta_t}$ equivalent weight analysis for $\epsilon$-prediction provides a reference for all works utilizing the Brownian Bridge.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First formulation of probabilistic translation between astronomical surveys; significant cross-domain innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Complete bidirectional translation + anomaly detection + ablation, though evaluated only on simulated data.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear problem definitions, rigorous mathematical derivations, and thoughtful metric design.
- **Value**: ⭐⭐⭐⭐ Provides a proof-of-concept and benchmark for the upcoming LSST-Euclid joint analysis.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[CVPR 2026\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](codrawagents_a_multiagent_dialogue_framework_for_c.md)
- [\[CVPR 2026\] Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models](probing_and_bridging_geometry-interaction_cues_for_affordance_reasoning_in_visio.md)
- [\[CVPR 2026\] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)
- [\[CVPR 2026\] SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models](segquant_diffusion_model_quantization.md)

<!-- RELATED:END -->
