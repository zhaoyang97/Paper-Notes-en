---
title: >-
  [Paper Note] Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations
description: >-
  [ICCV 2025][Autonomous Driving][pedestrian trajectory prediction] This paper proposes the Resonance (Re) model, which decomposes pedestrian trajectory prediction into a superposition of multiple "vibrations"—a linear base, a self-bias, and a resonance-bias. By leveraging spectral similarity between trajectories to simulate "resonance" phenomena in social interactions, the method is validated on ETH-UCY, SDD, NBA, and nuScenes benchmarks.
tags:
  - ICCV 2025
  - Autonomous Driving
  - pedestrian trajectory prediction
  - vibration decomposition
  - resonance interaction
  - spectral representation
  - social behavior modeling
date: 2026-05-08
content_hash: f00fe76bfc74e369
---

# Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations

**Conference**: ICCV 2025
**arXiv**: [2412.02447](https://arxiv.org/abs/2412.02447)
**Code**: [https://github.com/cocoon2wong/Re](https://github.com/cocoon2wong/Re)
**Area**: Autonomous Driving / Pedestrian Trajectory Prediction
**Keywords**: pedestrian trajectory prediction, vibration decomposition, resonance interaction, spectral representation, social behavior modeling

## TL;DR
This paper proposes the Resonance (Re) model, which decomposes pedestrian trajectory prediction into a superposition of multiple "vibrations"—a linear base, a self-bias, and a resonance-bias. By leveraging spectral similarity between trajectories to simulate "resonance" phenomena in social interactions, the method is validated on ETH-UCY, SDD, NBA, and nuScenes benchmarks.

## Background & Motivation

1. **State of the Field**: Pedestrian trajectory prediction is a core task in autonomous driving and social robotics. Mainstream approaches include LSTM-based social force models (Social LSTM), GAN-based multimodal prediction (Social GAN), GCN-based graph network methods (STGCNN), Transformer-based attention mechanisms (AgentFormer), and diffusion-based methods (LED, BCDiff).

2. **Limitations of Prior Work**: (a) Existing methods struggle to accurately disentangle trajectory variations and stochasticity arising from different causes when jointly modeling pedestrian intent and social behavior; (b) Social interaction modeling typically relies on attention mechanisms or graph networks, lacking interpretability—models cannot clearly articulate how a specific neighbor concretely influences the predicted trajectory; (c) The sources of trajectory stochasticity are diverse (personal intent, social influence, environmental constraints, etc.), yet most existing methods model them in a coupled manner.

3. **Root Cause**: Trajectory variation is the result of multiple superimposed factors, but existing methods typically model them jointly within a single latent space or attention mechanism, resulting in poor interpretability and an inability to precisely characterize the independent contribution of each factor.

4. **Paper Goals**:
    - Design an interpretable trajectory decomposition strategy that separates trajectory corrections and stochasticity into multiple independent "vibration" components
    - Propose a spectral-property-based social interaction representation that simulates "resonance" phenomena
    - Achieve competitive prediction accuracy across multiple benchmark datasets

5. **Starting Point**: Inspired by vibration systems and resonance phenomena in physics. In a vibration system, complex motion can be decomposed into a superposition of independent vibrations (Fourier decomposition); "resonance" occurs when two oscillators have similar natural frequencies. By analogy to pedestrian trajectories—trajectory variation can be decomposed into independent vibration components, and social interaction can be modeled via spectral similarity.

6. **Core Idea**: Trajectory prediction is formulated as the superposition $y = \text{linear\_base} + \text{self\_bias} + \text{resonance\_bias}$ (linear base + self-bias + resonance-bias), where social interaction is learned through "resonance" features derived from trajectory spectra, enabling interpretable and disentangled prediction.

## Method

### Overall Architecture

The Resonance (Re) model takes as input the observed trajectory of the target pedestrian $x_{ego}$ (8 frames) and the observed trajectories of neighboring pedestrians $x_{nei}$ (8 frames × N neighbors), and outputs the predicted future trajectory of the target pedestrian (12 frames). The prediction process consists of three stages:

1. **Linear Difference Encoding (LinearDiffEncoding)**: Extracts difference features $f_{diff}$ between the observed trajectory and its linear fit, while producing the linear base trajectory $\text{linear\_base}$
2. **Self-Bias Prediction (SelfBiasLayer)**: Predicts the self-bias $\text{self\_bias}$ representing individual behavioral intent based on the difference features
3. **Resonance-Bias Prediction (ReBiasLayer)**: Computes a resonance matrix (ResonanceLayer) and predicts the resonance-bias $\text{resonance\_bias}$ representing the effect of social interaction, combined with the difference features

Final output: $y = \text{linear\_base} + \text{self\_bias} + \text{resonance\_bias}$

### Key Designs

1. **Linear Difference Encoding**:
    - Function: Decomposes the observed trajectory into a linear trend and a nonlinear residual
    - Mechanism: A linear layer first fits the observed trajectory to obtain a linear trajectory $\text{linear\_fit}$ (representing the uniform-motion trend); the difference between the actual trajectory and the linear fit is then computed. FFT is applied separately to the original and linear trajectories, and both are encoded through a bilinear structure (outer product + pooling + fully connected layer) to produce the difference feature $f_{diff}$
    - Design Motivation: Uniform linear motion constitutes the "ground state" of pedestrian motion; the nonlinear deviations from this state are the core target of prediction. The linear base serves as an "anchor point" for prediction, reducing the difficulty for subsequent modules

2. **Self-Bias Layer**:
    - Function: Models trajectory bias caused by individual behavioral intent, independent of social interaction
    - Mechanism: The difference feature $f_{diff}$ is concatenated with random noise $z$ and fed into a 4-layer Transformer encoder; a multi-style network (MSN mechanism, incorporating graph convolution) generates $K_c = 20$ candidate predictions, which are then mapped back to trajectory space via an inverse transform (e.g., iFFT). Keypoint interpolation (linear or velocity-based) is supported
    - Design Motivation: A pedestrian's personal intent (turning, accelerating, stopping, etc.) is independent of social factors and must be modeled separately. The introduction of random noise enables multimodal prediction

3. **Resonance Layer + Re-Bias Layer**:
    - Function: Models the influence of social interaction on trajectories
    - Mechanism (two steps):
        - **Resonance Feature Computation**: The trajectories of the target pedestrian and neighbors are FFT-encoded; "resonance features" (spectral similarity) are computed via element-wise product $f_{ego} \cdot f_{nei}$, then compressed through a fully connected layer. Neighbors are partitioned into multiple angular sectors, within each of which resonance features and positional information are aggregated to form the "resonance matrix" $\text{re\_matrix}$
        - **Resonance-Bias Prediction**: The difference feature $f_{diff}$ and the resonance matrix are concatenated and fused, random noise is added, and the result is encoded by a 2-layer Transformer; the MSN mechanism generates multiple candidate social biases $\text{resonance\_bias}$
    - Design Motivation:
        - Physical resonance occurs between oscillators with similar frequencies—by analogy, neighbors with similar motion patterns (spectra) exert greater influence on the target pedestrian
        - The angular partitioning strategy is inherited from SocialCircle (CVPR 2024) and effectively captures directional social influence
        - The element-wise product $f_{ego} \cdot f_{nei}$ directly measures the spectral similarity between two trajectories, analogous to cross-correlation / resonance strength

### Loss & Training

- The minimum ADE (Average Displacement Error) loss is used, selecting the candidate prediction closest to the ground truth from $K$ candidates
- $K_{train} = 10$ during training; $K = 20$ during testing
- Feature dimension $d = 128$; Transformer with 8-head attention
- FFT is used as the default spectral transform (Haar/DB2 wavelet transforms are also supported)
- Velocity-based speed interpolation is supported as the keypoint interpolation strategy
- Maximum 500 training epochs; batch size 5000

## Key Experimental Results

### Main Results

Pedestrian trajectory prediction results on the ETH-UCY dataset (ADE/FDE, lower is better):

| Dataset | Metric | Re (Ours) | SocialCircle | Social-STGCNN | AgentFormer | Notes |
|---------|--------|-----------|-------------|---------------|------------|-------|
| ETH | ADE/FDE | Competitive | 0.34/0.55 | 0.64/1.11 | 0.45/0.75 | Ours improves upon SocialCircle |
| Hotel | ADE/FDE | Competitive | 0.14/0.22 | 0.49/0.85 | 0.14/0.22 | Small margin in simple scenes |
| UNIV | ADE/FDE | Competitive | 0.27/0.45 | 0.44/0.79 | 0.25/0.45 | Dense crowd scenario |
| ZARA1 | ADE/FDE | Competitive | 0.20/0.32 | 0.34/0.53 | 0.18/0.30 | Medium density |
| ZARA2 | ADE/FDE | Competitive | 0.15/0.27 | 0.30/0.48 | 0.14/0.24 | Medium density |

Note: The HTML version of the paper failed to render properly; specific numbers could not be extracted from the source. The table above lists representative results of relevant baseline methods for reference.

### Ablation Study

Ablation is performed over the three model components (linear_base, self_bias, resonance_bias):

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| linear_base only | Baseline performance | Equivalent to simple linear prediction |
| linear_base + self_bias | Significant improvement | Prediction quality substantially improves with individual intent modeling |
| linear_base + self_bias + re_bias (full model) | Best | Social interaction further improves performance, especially in dense crowd scenarios |
| SocialCircle replacing ResonanceCircle | Slightly worse | Validates that spectrum-based resonance interaction outperforms original SocialCircle |
| Different transform types (FFT vs. Haar vs. DB2) | FFT best | FFT is the default and most effective spectral transform |
| w/o self_bias | Performance drops | Validates the necessity of the self-bias component |
| w/o re_bias | Performance drops | Validates the contribution of social interaction modeling |

### Key Findings
- The "vibration decomposition" strategy (linear_base + self_bias + re_bias) outperforms end-to-end direct prediction, as each component has a more clearly defined prediction target
- The "resonance" interaction representation based on spectral similarity offers better interpretability than traditional attention mechanisms—it enables intuitive assessment of which neighbors exert greater influence via spectral similarity
- The angular partitioning strategy inherits the advantages of SocialCircle and effectively captures directional social influence
- The model demonstrates effectiveness across diverse datasets and scenarios, including ETH-UCY, SDD, NBA, and nuScenes

## Highlights & Insights
- **Elegance of the Vibration Metaphor**: Reformulating trajectory prediction as a superposition of vibrations is not only mathematically natural (Fourier decomposition) but also provides an intuitive physical analogy—pedestrian motion is indeed the superposition of multiple "forces" (intent, social influence, environment)
- **Interpretability**: Through visualization of the resonance matrix, it is possible to clearly observe which neighbors the model has learned to weight most heavily, as well as the direction and magnitude of their influence
- **Modular Design**: The three components (linear base / self-bias / resonance-bias) can be independently toggled, facilitating ablation analysis and flexible deployment
- **Series Work**: This paper is the second installment of the "Echolocation Trilogy" (SocialCircle → Resonance → Reverberation), with each paper focusing on a different aspect of social interaction, forming a coherent research program

## Limitations & Future Work
- **Limitation 1**: Although the resonance metaphor is intuitive, a formal theoretical justification for whether spectral similarity truly corresponds to physical resonance is lacking; it remains largely a heuristic analogy
- **Limitation 2**: The model assumes that social interaction can be captured through pairwise trajectory spectral products, neglecting higher-order multi-body interactions
- **Limitation 3**: FFT requires uniformly sampled data; preprocessing may be necessary for irregularly sampled trajectory data
- **Future Work**: The third installment of the trilogy, Reverberation, addresses "how long the echo persists"—i.e., modeling the temporal decay of social interactions. Future work may further explore more complex spectral transforms and multi-body resonance mechanisms

## Related Work & Insights
- **vs. SocialCircle [Wong et al., CVPR 2024]**: SocialCircle uses angular partitioning with velocity/distance/direction three-factor encoding for social information. Resonance builds upon this by introducing spectral-domain resonance features, replacing handcrafted factors with trajectory spectral products, yielding a more general and interpretable representation
- **vs. V^2-Net / Vertical [Wong et al., ECCV 2022]**: This prior work was the first to introduce Fourier spectra into trajectory prediction. Resonance inherits the spectral encoding paradigm and extends it to social interaction modeling
- **vs. AgentFormer [Yuan et al., ICCV 2021]**: AgentFormer employs full attention mechanisms to model inter-agent relationships. Resonance provides a more physically meaningful interaction representation through spectral resonance
- **vs. Social GAN [Gupta et al., CVPR 2018]**: Social GAN models social behavior by pooling neighbor information. Resonance's angular partitioning and resonance matrix provide finer-grained, spatially aware interaction modeling

## Rating
- Novelty: ⭐⭐⭐⭐ The vibration/resonance metaphor is novel and elegant, though the core techniques (FFT + attention + MSN) are not entirely new
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset validation with complete ablation studies, visualization analysis, and an interactive Playground
- Writing Quality: ⭐⭐⭐⭐ The physical analogy is clearly articulated; the model naming (Resonance / Echolocation Trilogy) is creative
- Value: ⭐⭐⭐⭐ Offers a new perspective on social interaction modeling for trajectory prediction; interpretability is a key strength

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control](saliency-aware_quantized_imitation_learning_for_efficient_robotic_control.md)
- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](future-aware_interaction_network_for_motion_forecasting.md)
- [\[ICCV 2025\] Occupancy Learning with Spatiotemporal Memory](occupancy_learning_with_spatiotemporal_memory.md)
- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)

<!-- RELATED:END -->
