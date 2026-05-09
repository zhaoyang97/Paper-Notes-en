---
title: >-
  [Paper Note] Diverse Text-to-Image Generation via Contrastive Noise Optimization
description: >-
  [ICLR 2026][Image Generation][Diffusion models] This paper proposes Contrastive Noise Optimization (CNO), which applies an InfoNCE contrastive loss over the Tweedie denoised prediction space to optimize initial noise vectors as a preprocessing step, improving the generation diversity of diffusion models while maintaining fidelity, without modifying the sampling process or model parameters.
tags:
  - ICLR 2026
  - Image Generation
  - Diffusion models
  - text-to-image generation
  - diversity
  - contrastive learning
  - noise optimization
  - InfoNCE
date: 2026-05-08
content_hash: 05608806266563c6
---

# Diverse Text-to-Image Generation via Contrastive Noise Optimization

**Conference**: ICLR 2026
**arXiv**: [2510.03813](https://arxiv.org/abs/2510.03813)
**Code**: Available (official open-source)
**Area**: Diffusion Models / Image Generation
**Keywords**: Diffusion models, text-to-image generation, diversity, contrastive learning, noise optimization, InfoNCE

## TL;DR

This paper proposes Contrastive Noise Optimization (CNO), which applies an InfoNCE contrastive loss over the Tweedie denoised prediction space to optimize initial noise vectors as a preprocessing step, improving the generation diversity of diffusion models while maintaining fidelity, without modifying the sampling process or model parameters.

## Background & Motivation

**Diversity bottleneck in diffusion models**: Current text-to-image diffusion models (e.g., SD1.5, SDXL, SD3) tend to produce highly similar outputs given the same prompt (mode collapse), especially under deterministic samplers (e.g., DDIM, FM-ODE), leading to severely limited output diversity.

**Root cause lies in the noise space distribution**: Randomly sampled Gaussian initial noises do not guarantee uniform dispersion in the semantic space after denoising, causing multiple noise vectors to map to similar generated results.

**Limitations of prior work**: Increasing stochasticity (e.g., stochastic samplers) sacrifices generation quality; post-processing methods (e.g., rejection sampling) incur high computational cost; modifying model architectures or training pipelines introduces significant invasiveness.

**Inspiration from contrastive learning**: The InfoNCE loss naturally embodies an attract-repel structure that simultaneously encourages fidelity (attraction term) and diversity (repulsion term) within a noise batch.

**Appeal of the preprocessing paradigm**: Optimizing only the initial noise before sampling enables seamless composition with arbitrary diffusion models and samplers, offering strong generality and plug-and-play capability.

**Need for theoretical controllability**: A theoretically grounded and analytically computable parameter is needed to balance diversity and fidelity, rather than relying on manual hyperparameter tuning.

## Method

### Overall Architecture

CNO is a **one-shot preprocessing** method. Given a batch of randomly sampled initial noises $\{z_i\}_{i=1}^B$, gradient-based optimization is applied so that each noise vector remains close to its original position (fidelity) while being pushed apart from others in the Tweedie denoised prediction space (diversity). The optimized noises are then fed into any standard diffusion sampling pipeline without modifying the sampler or model parameters.

### Key Designs

**1. Tweedie Denoised Prediction Space**

- For each noise $z_i$, a one-step denoising prediction (Tweedie denoised prediction) from the diffusion model yields a semantic-level representation $\hat{x}_0(z_i)$.
- Distances are computed in this space rather than in the raw noise space, as noise-space distances do not directly correspond to semantic differences in the final generated outputs.

**2. InfoNCE Contrastive Loss**

- **Attraction term**: Anchors each optimized noise near its original position to prevent excessive deviation that would degrade quality.
- **Repulsion term**: Pushes different noise vectors apart in the denoised prediction space to enhance output diversity.
- Loss formulation:

$$\mathcal{L}_{\text{CNO}} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\text{sim}(z_i, z_i^{\text{orig}})/\tau)}{\exp(\text{sim}(z_i, z_i^{\text{orig}})/\tau) + \sum_{j \neq i}\exp(\text{sim}(z_i, z_j)/\tau)}$$

**3. γ Parameter and Closed-Form Formula**

- $\gamma$ controls the balance between attraction and repulsion.
- Closed-form formula: $\gamma = (\tau \cdot \ln(B-1) + 1)^{-1}$
- As batch size $B$ increases, more repulsive samples are introduced, and $\gamma$ automatically reduces the repulsion strength to prevent excessive dispersion.

**4. Adaptive Latent Pooling**

- Spatial downsampling of latents using a window $w$ reduces the dimensionality for contrastive loss computation.
- $w=16$ is the optimal trade-off: computationally efficient with negligible performance loss.

**5. Stop-Gradient Strategy**

- Gradients through paired noises in the repulsion term are detached (stop-gradient), reducing computational overhead and avoiding instability caused by mutual coupling between noise vectors.

### Loss & Training

- Objective: minimize the CNO contrastive loss.
- Optimizer: standard gradient descent (e.g., Adam) applied to a batch of noise vectors for a small number of iterations.
- Only the noise vectors $z_i$ are optimized; model parameters remain unchanged.
- Temperature $\tau$ is a hyperparameter; $\gamma$ is jointly determined via the closed-form formula, eliminating the need for hyperparameter search.

## Key Experimental Results

### Main Results

Comparison of CNO against baseline methods across multiple diffusion models on diversity and quality metrics:

| Model | Method | MSS ↓ | Vendi Score ↑ | Coverage ↑ | PickScore ↑ |
|-------|--------|-------|--------------|------------|-------------|
| SD1.5 | DDIM | 0.1657 | 4.6949 | - | - |
| SD1.5 | **CNO** | **0.1317** | **4.7855** | - | - |
| SDXL | DDIM | 0.2169 | - | - | - |
| SDXL | **CNO** | **0.1623** | - | **0.7568** | - |
| SD3 | FM-ODE | - | 4.2205 | - | - |
| SD3 | **CNO** | - | **4.2644** | - | - |

### Ablation Study

| Component | MSS ↓ | Vendi ↑ | Note |
|-----------|-------|---------|------|
| Full CNO (w=16) | **0.1317** | **4.7855** | Optimal configuration |
| w/o attraction term | 0.1285 | 4.8012 | Slightly higher diversity but degraded quality |
| w/o repulsion term | 0.1648 | 4.7011 | Marginal diversity improvement |
| w=4 | 0.1325 | 4.7801 | High computational cost with limited gain |
| w=32 | 0.1398 | 4.7512 | Excessive information loss |
| w/o stop-gradient | 0.1321 | 4.7830 | Comparable performance but doubled computation |

### Key Findings

1. **Pareto optimality**: CNO occupies the dominant Pareto frontier on the PickScore (quality) vs. Vendi Score (diversity) scatter plot.
2. **Compatibility with few-step samplers**: CNO remains effective on few-step samplers such as FLUX and SDXL-Lightning, demonstrating the generality of the preprocessing paradigm.
3. **Validation of the closed-form γ formula**: The theoretically derived $\gamma$ value closely matches the grid-search optimum, eliminating hyperparameter tuning.
4. **Robustness to window size**: Performance is stable for $w \in [8, 32]$, with $w=16$ being optimal.

## Highlights & Insights

- **Plug-and-play**: As a preprocessing method, CNO is orthogonal to arbitrary diffusion models and samplers, requiring minimal engineering effort for deployment.
- **Theoretical elegance**: The closed-form formula for $\gamma$ unifies the effects of batch size and temperature into a single interpretable parameter.
- **Novel perspective on contrastive learning**: Transferring InfoNCE from representation learning to noise-space optimization represents a clever conceptual migration.
- **Insight into the Tweedie space**: Measuring distances in the semantically meaningful denoised prediction space—rather than in the raw noise space—is the key to the method's effectiveness.

## Limitations & Future Work

1. **Additional computational overhead**: Each generation requires extra optimization iterations (albeit one-shot), introducing latency for real-time applications.
2. **Batch dependency**: The method requires simultaneous optimization of a batch of noises and is therefore inapplicable in single-image generation scenarios.
3. **Insufficient validation of semantic diversity**: MSS and Vendi Score primarily measure pixel-level diversity; in-depth analysis of semantic-level diversity is lacking.
4. **Tweedie prediction accuracy**: The method relies on the quality of one-step denoising predictions, which may be insufficiently accurate for certain models or timesteps.
5. **Scalability**: For very large batch sizes, the computational and memory costs of the contrastive loss may become a bottleneck.

## Related Work & Insights

- **DDIM / DPM-Solver**: Deterministic samplers are efficient but lack diversity; CNO directly addresses this shortcoming.
- **DPP (Determinant Point Process)**: A classical diversity sampling method; the CNO contrastive loss can be viewed as a continuous analogue of DPP.
- **Contrastive learning (SimCLR / MoCo)**: InfoNCE has been extensively validated in representation learning; CNO transfers it to the noise space of generative models.
- **Noise scheduling research**: Prior work has examined the effect of noise scheduling on generation quality; CNO is the first to focus on the effect of noise initialization on diversity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying contrastive learning to noise optimization is a concise and novel idea; the closed-form γ formula adds theoretical depth.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers multiple models (SD1.5/SDXL/SD3/FLUX), includes comprehensive ablation studies, and presents convincing Pareto analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, method derivation is fluent, and figures are of high quality.
- **Value**: ⭐⭐⭐⭐ — Strong practical utility as a plug-and-play solution; provides a clean resolution to the diversity problem in diffusion model generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models](pcpo_proportionate_credit_policy_optimization_for_aligning_image_generation_mode.md)
- [\[CVPR 2026\] Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](../../CVPR2026/image_generation/neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)
- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](../../ICCV2025/image_generation/straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[ICLR 2026\] Improved Object-Centric Diffusion Learning with Registers and Contrastive Alignment (CODA)](improved_object-centric_diffusion_learning_with_registers_and_contrastive_alignm.md)
- [\[ICLR 2026\] RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation](rmflow_refined_mean_flow_by_a_noise-injection_step_for_multimodal_generation.md)

</div>

<!-- RELATED:END -->
