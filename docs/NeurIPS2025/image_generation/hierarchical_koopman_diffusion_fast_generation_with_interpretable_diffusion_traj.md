---
title: >-
  [Paper Note] Hierarchical Koopman Diffusion: Fast Generation with Interpretable Diffusion Trajectory
description: >-
  [NeurIPS 2025][Image Generation][Diffusion model acceleration] Grounded in Koopman operator theory, this work lifts the nonlinear denoising dynamics of diffusion models into a linear Koopman space…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Diffusion model acceleration"
  - "one-step generation"
  - "Koopman operator"
  - "interpretable generation"
  - "hierarchical dynamics"
  - "spectral analysis"
date: 2026-05-08
content_hash: c0d7928f0123efe8
---

# Hierarchical Koopman Diffusion: Fast Generation with Interpretable Diffusion Trajectory

**Conference**: NeurIPS 2025
**arXiv**: [2510.12220](https://arxiv.org/abs/2510.12220)
**Authors**: Hanru Bai (Fudan University), Weiyang Ding (Fudan University), Difan Zou (The University of Hong Kong)
**Area**: Image Generation
**Keywords**: Diffusion model acceleration, one-step generation, Koopman operator, interpretable generation, hierarchical dynamics, spectral analysis

## TL;DR

Grounded in Koopman operator theory, this work lifts the nonlinear denoising dynamics of diffusion models into a linear Koopman space, enabling one-step sampling through hierarchical decomposition while preserving the interpretability and controllability of intermediate generation states.

## Background & Motivation

Diffusion models have achieved remarkable success in high-fidelity image generation, yet their sampling procedures are inherently iterative, requiring tens to thousands of denoising steps, which severely limits practical efficiency.

Existing acceleration approaches fall into two broad categories:

**Distillation methods** (Knowledge Distillation, Progressive Distillation, Rectified Flow, etc.): distill pretrained diffusion models into one-step generators.

**Consistency models** (Consistency Models, iCT, ECM, etc.): learn time-consistent mappings from noise directly to clean data.

Although these methods achieve one-step generation, they fundamentally learn a "black-box" noise→image mapping, **entirely abandoning the temporal denoising trajectory inherent to diffusion models**. This leads to:

- Inaccessibility of intermediate generation states
- Loss of interpretability over the generation process
- Inability to perform fine-grained intervention at specific stages during inference (e.g., controllable editing)

The core motivation of this paper: **Can one-step sampling efficiency be retained while preserving the interpretability and controllability of the diffusion trajectory?** Starting from Koopman operator theory, the authors propose an "explicit" one-step generation paradigm to resolve this tension.

## Method

### Overall Architecture

The Hierarchical Koopman Diffusion (HKD) framework consists of three core components:

1. **Encoder $\mathcal{E}_\theta$**: Based on a U-Net downsampling structure, projects the noisy image $\boldsymbol{x}_t$ into a multi-scale Koopman space.
2. **Hierarchical Koopman dynamics module**: Drives state evolution through linear operators at each resolution level.
3. **Decoder $\mathcal{D}_\phi$**: Maps the evolved Koopman representations back to image space.

Workflow: noise $\boldsymbol{x}_T$ → encoder extracts multi-scale features $\{\boldsymbol{z}_T^{(l)}\}_{l=1}^L$ → Koopman linear evolution → decoder outputs $\hat{\boldsymbol{x}}_\epsilon$.

### Key Designs

#### 1. Theoretical Foundation of the Koopman Operator

The Koopman operator lifts a nonlinear dynamical system into an infinite-dimensional space of observable functions, rendering the system evolution linear in that space. Concretely, for state evolution $\boldsymbol{x}_{t+\Delta t} = \Phi(\boldsymbol{x}_t)$ along the diffusion ODE, there exists a linear operator $\mathcal{K}$ acting on an observable function $g$:

$$(\mathcal{K} \circ g)(\boldsymbol{x}) = g(\Phi(\boldsymbol{x}))$$

In continuous time, the dynamics in Koopman space reduce to a linear ODE: $d\boldsymbol{z}_t / dt = \boldsymbol{A} \boldsymbol{z}_t$, which admits a closed-form solution.

#### 2. Hierarchical Koopman Subspaces

Image generation is intrinsically multi-scale: global structure forms first, and local textures emerge later. To this end, HKD designs independent Koopman subspaces at different spatial resolution levels:

- The encoder extracts features $\boldsymbol{z}_t^{(l)} \in \mathbb{R}^{d_l \times h_l \times w_l}$ at $L$ levels.
- The feature vector $\boldsymbol{z}_t^{(l)}(i,j)$ at each spatial location $(i,j)$ is driven by a local linear operator $\boldsymbol{A}^{(l)}(i,j)$.
- $\boldsymbol{A}^{(l)}(i,j)$ adopts a block-diagonal structure, where each $2 \times 2$ block corresponds to a pair of conjugate complex eigenvalues $\alpha_k^{(l)} \pm i\beta_k^{(l)}$.

This spatially adaptive design allows different regions to exhibit distinct time-frequency behaviors, enabling fine-grained modeling of the generation dynamics.

#### 3. Closed-Form One-Step Mapping

Since the dynamics in Koopman space are linear, the state mapping from time $s$ to $t$ has an explicit solution:

$$\boldsymbol{z}_t^{(l)}(i,j) = e^{\boldsymbol{A}^{(l)}(i,j)(t-s)} \boldsymbol{z}_s^{(l)}(i,j)$$

At inference, the one-step mapping is completed by directly computing the matrix exponential, without iterative integration.

### Loss & Training

The total training loss is:

$$\mathcal{L} = \mathcal{L}_{t\text{-consist}} + \mathcal{L}_{\text{recon}}$$

- **Trajectory consistency loss $\mathcal{L}_{t\text{-consist}}$**: A state encoded and evolved from any intermediate time $t$ to $\epsilon$ should, after decoding, match the ground-truth $\boldsymbol{x}_\epsilon$. This is a core advantage over implicit methods — it explicitly supervises intermediate states.
- **Reconstruction loss $\mathcal{L}_{\text{recon}}$**: Directly supervises the one-step mapping from noisy $\boldsymbol{x}_T$ to clean $\boldsymbol{x}_\epsilon$.
- Distance metric: $d(\cdot, \cdot) = \lambda_1 \mathcal{L}_{\text{MSE}} + \lambda_2 \mathcal{L}_{\text{LPIPS}}$, where $\lambda_1$ is annealed to transition from coarse matching to perceptual refinement.

Notably, the authors theoretically prove that, under structural assumptions, minimizing the trajectory consistency loss in image space is equivalent to minimizing it in the Koopman latent space. Image space is preferred because perceptual metrics such as LPIPS are available there, yielding more effective gradient signals.

## Training & Inference

### Training Procedure

1. The encoder and decoder are initialized from EDM-pretrained U-Net weights.
2. Koopman matrices $\boldsymbol{A}^{(l)}$ are initialized to zero (so that Koopman evolution is initially the identity map).
3. Each iteration uniformly samples $s-1$ intermediate time points plus the terminal $T$, and computes the trajectory consistency loss over all time points.
4. The Adam optimizer is used with learning rate $1 \times 10^{-3}$ and weight decay 0.95.
5. The encoder $\mathcal{E}_\theta$, decoder $\mathcal{D}_\phi$, and all Koopman matrices $\{\boldsymbol{A}^{(l)}\}$ are trained end-to-end.
6. Training data are ODE trajectories $\{\boldsymbol{x}_t\}_{t \in [0,T]}$ generated by a pretrained diffusion model.

### Inference (One-Step Sampling)

Inference is remarkably concise: given noise $\boldsymbol{x}_T \sim \mathcal{N}(0, I)$, the output is computed in a single step:

$$\hat{\boldsymbol{x}}_\epsilon = \mathcal{D}_\phi(\{e^{(\epsilon - T)\boldsymbol{A}^{(l)}} \mathcal{E}_\theta^{(l)}(\boldsymbol{x}_T)\}_{l=1}^L)$$

The entire process — encode → matrix-exponential multiplication → decode — requires no iteration.

### Training Efficiency

- Training completes in 2–3 days on 8×V100 GPUs.
- Substantially faster than consistency model training (over one week on 8 GPUs).
- Training is more stable: the exponential form in Koopman space ensures sufficient gradients when spectral magnitudes are near 1 (avoiding the high-variance gradient problem of consistency models), and multi-time-point supervision stabilizes spectral estimation.

## Key Experimental Results

### Main Results: CIFAR-10 One-Step Generation

| Method | Category | NFE | FID ↓ |
|:---|:---|:---:|:---:|
| DDPM | Multi-step diffusion | 1000 | 3.17 |
| EDM | Multi-step diffusion | 35 | 1.97 |
| DDIM | Multi-step diffusion | 10 | 13.36 |
| KD | Distillation | 1 | 9.36 |
| PD | Distillation | 1 | 8.34 |
| DMD | Distillation | 1 | 3.77 |
| 2-Rectified Flow++ | Distillation | 1 | 3.38 |
| CT (LPIPS) | Consistency | 1 | 8.70 |
| CD (LPIPS) | Consistency distillation | 1 | 3.55 |
| iCT-deep | Consistency | 1 | 2.51 |
| ECM | Consistency | 1 | 3.60 |
| **HKD (Ours)** | **Koopman** | **1** | **3.30** |

### Main Results: FFHQ 64×64

| Method | NFE | FID ↓ |
|:---|:---:|:---:|
| EDM | 79 | 2.47 |
| EDM | 15 | 9.85 |
| ECM | 1 | 5.99 |
| **HKD (Ours)** | **1** | **5.70** |

### Ablation Study

| Koopman Evolution | Trajectory Consistency Loss | Hierarchical Design | FID ↓ |
|:---:|:---:|:---:|:---:|
| ✗ | ✗ | ✓ | 5.72 |
| ✓ | ✗ | ✓ | 5.57 |
| ✓ | ✓ | ✗ | 4.78 |
| ✓ | ✓ | ✓ | **3.30** |

### Key Findings

1. **FID 3.30** on CIFAR-10 one-step generation outperforms most distillation methods and ECM, approaching iCT-deep (2.51), which however relies on extensive hyperparameter tuning and exhibits training instability.
2. **FID 5.70 on FFHQ** outperforms ECM (5.99), validating the framework's effectiveness on more complex datasets.
3. Ablation studies show significant contributions from all three components: Koopman evolution (5.72→5.57), trajectory consistency loss (5.57→4.78), and hierarchical design (4.78→3.30).
4. Training requires only 2–3 days on 8×V100 GPUs, far faster than consistency models (one week on 8 GPUs), and is more stable.

### Koopman Spectral Analysis

By analyzing the spectral structure of the learned Koopman matrices $\boldsymbol{A}$, the authors reveal the semantic hierarchy of the generation process:

- **Spectral masking experiment**: At each resolution level, eigenvalues are sorted by their real parts; selected spectral bands (smallest/middle/largest) are retained while others are zeroed out before decoding.
    - Low-frequency spectral modes → global structure (rough contours and background)
    - Mid-frequency spectral modes → overall shape and pose
    - High-frequency spectral modes → local textures and fine details
- **Cumulative Effect (CE) tracking**: The contribution of each spectral component to reconstruction is monitored over time, quantifying the evolution of different modes.

### One-Step Image Editing Experiment

Leveraging the interpretability of the framework, the authors perform frequency-aware interventions at intermediate states along the diffusion trajectory:

- At a midpoint of the Koopman trajectory, high-frequency features from a reference image are injected into the lower half of a generated image at varying mixing ratios (10%–90%).
- As the mixing ratio increases, facial details from the reference image progressively appear, demonstrating that the frequency decomposition establishes meaningful correspondences.
- Comparison: frequency-agnostic editing (mixing all frequency bands) disrupts global structure, lacking disentangled control.
- Inpainting and colorization experiments on CIFAR-10 are also conducted, iteratively blending reference and generated images along the trajectory.

## Highlights & Insights

1. **Theoretical innovation**: This is the first work to introduce Koopman operator theory into image generation, offering a fundamentally new mathematical perspective on diffusion model dynamics. The paper theoretically proves that HKD's representational capacity is no weaker than that of black-box one-step mappings: $err_{\text{HKD}} \leq err_{\text{one-step}} + O(\kappa)$.
2. **Interpretability**: Koopman spectral analysis reveals the semantic hierarchy of the generation process — low-frequency spectral modes correspond to global structure, mid-frequency modes to overall shape and pose, and high-frequency modes to local texture details.
3. **Controllable editing**: Frequency-aware interventions at intermediate states enable image editing within one-step generation (e.g., injecting high-frequency features from a reference image into a specified region), which is inaccessible to implicit one-step methods.
4. **Training stability**: The exponential form in Koopman space ensures sufficient gradients when spectral magnitudes are near 1, avoiding the high-variance gradient issues common in consistency models; multi-time-point trajectory supervision further stabilizes spectral estimation.
5. **Elegant architectural design**: The framework reuses EDM's U-Net as the encoder–decoder backbone, with Koopman dynamics as the intermediate bridge — a clean and readily implementable design.

## Limitations & Future Work

1. **Generation quality gap**: FID 3.30 still trails iCT-deep (2.51); adversarial training and advanced training techniques have not been employed.
2. **Limited dataset scale**: Validation is conducted only on CIFAR-10 (32×32) and FFHQ (64×64); high-resolution datasets (e.g., ImageNet 256×256) have not been tested.
3. **Koopman space assumption**: Finite-dimensional Koopman space approximation may underperform on more complex data distributions.
4. **Preliminary editing capability**: Frequency-aware editing serves as a proof of concept and has not been extended to text-guided editing or attribute-level control.
5. **Lack of comprehensive conditional generation evaluation**: Although conditional generation results are provided in the appendix, the main experiments focus on unconditional generation.

**Future directions (as discussed by the authors)**:

- **Adversarial training integration**: Incorporating adversarial learning may further close the gap with iCT-deep.
- **High-resolution generation**: The hierarchical design is naturally suited for high-resolution scaling, though experimental validation is pending.
- **Semantic editing extension**: Explicit spectral decomposition supports interpretable intervention and can be extended to text-guided editing, attribute-specific control, and richer semantic editing tasks.
- **Generality of Koopman dynamics**: The framework may generalize to video generation (spatiotemporal multi-scale) and 3D generation.

## My Notes

### Assessment of Core Contributions

- **The first introduction of Koopman operator theory into image generation** is not merely a superficial application of existing theory; it provides a genuinely useful framework: linearization enables not only one-step sampling but also spectral analysis and intermediate-state intervention.
- **The theoretical guarantee** (Thm 3.1) bounds the HKD error to no more than that of a black-box one-step method plus a small term $O(\kappa)$, where $\kappa$ vanishes as the dataset size $N$ and Koopman dimension $m$ grow, providing theoretical backing for the method's viability.
- The distinction between **"explicit" vs. "implicit" one-step generation** is insightful: distillation and consistency models are all implicit black-box noise→image mappings, whereas HKD is the first "white-box" one-step method.

### Comparative Reflections on Related Work

- **vs. distillation methods**: Distillation methods (KD, PD, DMD, Rectified Flow) learn black-box mappings; HKD offers a "white-box" alternative that achieves competitive performance while endowing interpretability.
- **vs. consistency models**: Consistency models are training-unstable and hyperparameter-sensitive; HKD is naturally more stable due to the linear structure of Koopman space.
- **Koopman's success in other domains**: Time series analysis (Koopa, KoNODE), dynamical systems control, etc. — this paper is the first application to image generation.

### Potential Limitations and Bottlenecks

- Spatially adaptive $\boldsymbol{A}^{(l)}(i,j)$ implies independent parameterization at each spatial location, so parameter count grows quadratically with resolution; parameter-sharing strategies may be necessary for high-resolution settings.
- The behavior of finite-dimensional Koopman approximations under highly complex data distributions warrants further investigation.
- The editing capability demonstrated operates at the frequency level; it remains distant from practically useful semantic editing (e.g., "change hair color").

### Inspirations

- Koopman theory provides a principled tool for "nonlinear-to-linear" lifting; this paradigm may generalize to video generation (spatiotemporal multi-scale), 3D generation, and other complex generative tasks.
- The hierarchical linear space design may inspire multi-scale modeling in other domains (e.g., multi-scale NeRF, hierarchical flow matching).
- The idea behind the trajectory consistency loss — supervising intermediate states — can be transferred to other one-step generation frameworks to enhance stability.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First application of Koopman operators to image generation; the framework design is original and theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐ — Dataset scale and resolution are limited (only CIFAR-10 and FFHQ 64×64), though ablation and analysis experiments are relatively thorough.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretically rigorous, framework clearly described, spectral analysis visualizations intuitive.
- **Value**: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [\[NeurIPS 2025\] State-Covering Trajectory Stitching for Diffusion Planners](state-covering_trajectory_stitching_for_diffusion_planners.md)
- [\[NeurIPS 2025\] KLASS: KL-Guided Fast Inference in Masked Diffusion Models](klass_kl-guided_fast_inference_in_masked_diffusion_models.md)
- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)
- [\[NeurIPS 2025\] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms](fast_solvers_for_discrete_diffusion_models_theory_and_applications_of_high-order.md)

</div>

<!-- RELATED:END -->
