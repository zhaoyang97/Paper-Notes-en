---
title: >-
  [Paper Note] AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys
description: >-
  [CVPR 2026][Image Generation][Brownian Bridge] The authors propose AS-Bridge, a bidirectional generative framework based on the Brownian Bridge diffusion process. It models the probabilistic conditional distribution between ground-based LSST and space-based Euclid astronomical surveys, enabling cross-survey image translation and rare event detection (strong gravita
tags:
  - CVPR 2026
  - Image Generation
  - Brownian Bridge
date: 2026-05-08
content_hash: ab54f578aeeb91c3
---
# AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys

**Conference**: CVPR 2026  
**arXiv**: [2603.11928](https://arxiv.org/abs/2603.11928)  
**Code**: [Available](https://github.com/ZHANG7DC/AS-Bridge)  
**Area**: Diffusion Models/Image Generation  
**Keywords**: Astronomical Surveys, Brownian Bridge, Cross-modal Translation, Anomaly Detection, Probabilistic Generation

## TL;DR
The authors propose AS-Bridge, a bidirectional generative framework based on the Brownian Bridge diffusion process. It models the probabilistic conditional distribution between ground-based LSST and space-based Euclid astronomical surveys, enabling cross-survey image translation and rare event detection (strong gravitational lensing). Furthermore, it improves likelihood estimation of the standard Brownian Bridge through an $\epsilon$-prediction training objective.

## Background & Motivation
Observational cosmology in the next decade will be driven by large-scale surveys: ground-based LSST (Vera C. Rubin Observatory) provides deep multi-band optical images but suffers from limited resolution and source blending due to atmospheric turbulence; space-based Euclid provides high-resolution near-infrared imaging but with fewer bands and incomplete spectral information. These two surveys have an overlapping sky area of approximately 7,000-9,000 deg², observing the same celestial objects but producing fundamentally different data.

Cross-survey inference is an ill-posed problem in both directions: recovering Euclid-level morphology from LSST requires resolving ambiguities caused by atmospheric blur and background noise; mapping Euclid back to LSST requires inferring spectral information from fewer bands. Therefore, cross-survey translation should be treated as a probabilistic process capable of sampling multiple valid realizations consistent with existing observations.

Existing cross-modal methods (GAN-based, conditional diffusion) are typically developed and evaluated under a single-direction deterministic paradigm, failing to faithfully represent the full conditional distribution between observation modalities. Scientific applications require probabilistic generation with uncertainty quantification.

## Method

### Overall Architecture
AS-Bridge models cross-survey translation as a bidirectional Brownian Bridge process. It utilizes paired observations from overlapping sky regions as anchors for training, learning stochastic paths between LSST and Euclid data distributions. Once trained, it can generate complementary observations in non-overlapping regions and be used for rare event detection.

### Key Designs

**1. Formulating cross-survey translation as a conditional distribution rather than a deterministic mapping**

LSST and Euclid observe the same sky, but their imaging physics differ entirely, meaning they do not have a one-to-one correspondence. AS-Bridge views both as two noisy projections of the same latent astrophysical process $\Phi$: $x_{\text{Euclid}} = \mathcal{O}_{\text{Euclid}}(\Phi) + \epsilon_{\text{Euclid}}$ and $x_{\text{LSST}} = \mathcal{O}_{\text{LSST}}(\Phi) + \epsilon_{\text{LSST}}$. Since $\Phi$ itself is unobservable, the authors marginalize it and directly learn the conditional distributions $p(x_{\text{Euclid}} \mid x_{\text{LSST}})$ and $p(x_{\text{LSST}} \mid x_{\text{Euclid}})$. This step shifts the problem from "GAN/deterministic I2I providing a single most-likely answer" to "providing a family of plausible realizations self-consistent with existing observations." Because inferring Euclid morphology from blurred LSST images or LSST multi-band colors from Euclid single-band data is inherently an ill-posed multi-solution problem, only probabilistic modeling can capture the underlying uncertainty.

**2. Rewriting the Brownian Bridge training objective with $\epsilon$-prediction to stabilize gradients at bridge endpoints**

The forward process of a standard Brownian Bridge uses real observations at both ends as anchors with intermediate noisy interpolation:

$$x_t \mid (x_0, x_T) \sim \mathcal{N}\big((1-m_t)x_0 + m_t x_T,\ \delta_t I\big),\qquad \delta_t = m_t(1-m_t)$$

The original loss directly regresses the drift plus denoising terms, but the variance $\delta_t$ approaches 0 at the bridge endpoints $t\to0$ and $t\to T$. Weighting by $1/\delta_t$ causes gradient vanishing near the endpoints. The authors prove that changing to an $\epsilon$-prediction target $\mathcal{L} = \|\epsilon_\theta - \epsilon\|_2^2$ is equivalent to multiplying the standard loss by a factor of $\sqrt{\delta_t}$. This "gentle weighting" maintains the likelihood-inspired "focus on high-noise timesteps" without suppressing gradients at the endpoints. The corresponding reconstruction formula is:

$$\hat{x}_0 = \frac{x_t - m_t x_T - \sqrt{\delta_t}\,\epsilon_\theta(x_t, x_T, t)}{1-m_t}$$

Ablations show that using $\delta_t$ as a weight actually yields poorer CRPS (see table below), confirming that the $\sqrt{\delta_t}$ gentle weighting is the key to the effectiveness of this training objective.

**3. Treating reconstruction inconsistency as an anomaly signal for unsupervised rare event detection**

Since the model is trained only on normal galaxies and has not seen rare structures like strong gravitational lenses, its reconstruction of such samples will "fail." This can be leveraged as a detection signal. Specifically, a pair of matched observations is fused into an intermediate variable $x_t$ via the forward process, then reconstructed back into the Euclid domain by sampling $N$ random realizations $\{\hat{x}_0^{(i)}\}_{i=1}^N$. The pixel-level anomaly score is defined as the minimum reconstruction error among these $N$ samples:

$$\mathcal{A}(p) = \min_{i \in \{1,\dots,N\}} \|\hat{x}_0^{(i)}(p) - x_0(p)\|_2^2$$

Taking the minimum rather than the average suppresses "false high errors" caused occasionally by sampling noise, retaining only true anomalies that the model cannot reconstruct regardless of the sample. The image-level score is then aggregated using flux normalization to ensure comparability between bright and faint sources:

$$\mathcal{A}(x_0) = \frac{\sum_p \mathcal{A}(p)}{\sum_p x_0(p)}$$

This effectively transforms the "epistemic boundary" of the generative model into a probe for discovering new phenomena—areas where the model performs poorly often represent rare astronomical objects underrepresented in the training distribution.

### Loss & Training
- Training Data: 115,000 normal galaxies + 5,000 strong lensing systems generated via SLSim simulation.
- LSST Images: g/r/i bands, 64×64 pixels, ~0.7" seeing.
- Euclid Images: VIS band, 0.1" pixel scale, 64×64 pixels.
- 110,000 normal galaxies used for training, others for evaluation.

## Key Experimental Results

### Main Results (Probabilistic reconstruction quality CRPS↓)

| Method | LSST→Euclid | Euclid→LSST |
|------|-------------|-------------|
| SPADE | 3.39 | 16.52 |
| OASIS | 4.65 | 13.33 |
| Pix2Pix | 4.35 | 73.03 |
| Palette | 2.43 | 7.98 |
| Joint Diffusion | 3.14 | 15.15 |
| BB Standard Loss | 2.55 | 7.90 |
| **AS-Bridge ($\epsilon$-pred)** | **2.38** | **7.90** |

### Ablation Study

| Training Objective | CRPS (LSST→Euclid) | CRPS (Euclid→LSST) | Description |
|---------|---------------------|---------------------|------|
| Standard Loss | 2.55 | 7.90 | Original BB objective |
| $\sqrt{\delta_t}$ Weight | 3.59 | 11.24 | Direct weighting performs worse |
| **$\epsilon$-pred** | **2.38** | **7.90** | Gentle weighting is optimal |

### Anomaly Detection (Strong Gravitational Lens Detection)

| Method | FPR@1%TPR↓ | FPR@5%TPR↓ | AUPR↑ |
|------|------------|------------|-------|
| **AS-Bridge** | **0.00%** | **0.18%** | **0.80** |
| Deco-Diff | 1.1% | 5.0% | 0.61 |
| CFM | 0.24% | 1.2% | 0.75 |

### Key Findings
- Diffusion/Bridge methods significantly outperform non-diffusion methods (GAN-based), validating the advantage of score-based generative modeling in recovering true conditional distributions.
- Euclid→LSST (inferring multi-band colors from a single band) is an extremely ill-posed problem, yet the model generates diverse reconstructions with consistent morphology and reasonable colors.
- LSST→Euclid translation correctly recovers the number and positions of galaxies in multi-source systems blended by atmospheric seeing.
- The single-modality method Deco-Diff completely fails to detect structural anomalies, highlighting that cross-modal information is crucial for rare event detection.

## Highlights & Insights
- This work is the first to formalize cross-survey translation as a probabilistic inference problem rather than simple I2I translation.
- The formal proof of $\epsilon$-prediction equivalence is elegant and practical, providing theoretical guidance for Brownian Bridge training.
- Utilizing reconstruction inconsistency for unsupervised anomaly detection is a clever scientific application—leveraging the "epistemic limits" of generative models to discover new phenomena.
- The design of evaluation metrics (CRPS for probabilistic quality, FPR@low TPR for scientific discovery contexts) reflects a deep understanding of domain requirements.

## Limitations & Future Work
- Currently trained and evaluated only on simulated data; the domain gap from simulation to real data is a known limitation.
- The CRPS for the Euclid→LSST direction remains relatively high (7.90), indicating significant uncertainty in multi-band color inference.
- Only strong gravitational lenses were used as representative anomalous events; verification with a wider variety of rare celestial objects is needed.
- Fixed image size of 64×64 may be insufficient for modeling large-scale structures.

## Related Work & Insights
- Core difference from Palette (conditional diffusion I2I): Palette reverses from pure noise while the source image acts only as a conditioning signal; BB models a stochastic path directly between two distributions.
- The cross-modal anomaly detection concept can be extended to other multi-sensor astronomical data (e.g., SKA radio + optical).
- The $\sqrt{\delta_t}$ equivalent weight analysis for $\epsilon$-prediction is a valuable reference for any work utilizing the Brownian Bridge.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Significantly formalizes probabilistic translation between surveys; strong cross-domain innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Bidirectional translation + anomaly detection + complete ablation, though evaluated only on simulation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definitions, rigorous mathematical derivation, and thoughtful metric design.
- Value: ⭐⭐⭐⭐ Provides a proof-of-concept and benchmark for the upcoming joint LSST-Euclid analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BiFM: Bidirectional Flow Matching for Few-Step Image Editing and Generation](bifm_bidirectional_flow_matching_for_few-step_image_editing_and_generation.md)
- [\[CVPR 2026\] Steering Where to Diffuse: Generative Modeling of Phenotypic Response Simulation with Steered Diffusion Bridge](steering_where_to_diffuse_generative_modeling_of_phenotypic_response_simulation_.md)
- [\[CVPR 2026\] Temporal Equilibrium MeanFlow: Bridging the Scale Gap for One-Step Generation](temporal_equilibrium_meanflow_bridging_the_scale_gap_for_one-step_generation.md)
- [\[CVPR 2026\] FVAR: Next-Focus Prediction for Visual Autoregressive Modeling](fvar_next-focus_prediction_for_visual_autoregressive_modeling.md)
- [\[CVPR 2026\] Scone: Bridging Composition and Distinction in Subject-Driven Image Generation via Unified Understanding-Generation Modeling](scone_bridging_composition_and_distinction_in_subject-driven_image_generation_vi.md)

</div>

<!-- RELATED:END -->
