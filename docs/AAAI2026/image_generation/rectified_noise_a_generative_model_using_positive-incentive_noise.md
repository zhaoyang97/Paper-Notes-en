---
title: >-
  [Paper Note] Rectified Noise: A Generative Model Using Positive-incentive Noise
description: >-
  [AAAI 2026][Image Generation][Rectified Flow] This paper proposes Rectified Noise (ΔRN), which leverages the positive-incentive noise (π-noise) framework to learn a set of beneficial noise signals and inject them into th…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Rectified Flow"
  - "Positive-incentive Noise"
  - "Flow Matching"
  - "SiT"
  - "Generative Models"
date: 2026-05-08
content_hash: 2629c9ba48d5a05d
---

# Rectified Noise: A Generative Model Using Positive-incentive Noise

**Conference**: AAAI 2026
**arXiv**: [2511.07911](https://arxiv.org/abs/2511.07911)  
**Code**: [https://github.com/simulateuser538/Rectified-Noise](https://github.com/simulateuser538/Rectified-Noise)  
**Area**: Image Generation
**Keywords**: Rectified Flow, Positive-incentive Noise, Flow Matching, SiT, Generative Models

## TL;DR

This paper proposes Rectified Noise (ΔRN), which leverages the positive-incentive noise (π-noise) framework to learn a set of beneficial noise signals and inject them into the velocity field of a pretrained Rectified Flow model, achieving a reduction in FID from 10.16 to 9.05 on ImageNet-1k with only 0.39% additional parameters.

## Background & Motivation

### State of the Field

Rectified Flow (RF) is an efficient generative modeling approach that learns a velocity field by connecting the source and target distributions via straight-line paths. RF directly parameterizes a continuous-time transport map without introducing additional stochasticity, with a simple training objective:

$$\mathcal{L}_{velocity}(\theta) = \mathbb{E}_{x_*, \epsilon, t}\left[\|\mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{x}_* + \epsilon\|^2\right]$$

SiT (Scalable Interpolant Transformers), a family of RF models built upon DiT, achieves strong generative performance through systematic exploration of the design space.

### Limitations of Prior Work

**An Intriguing Observation**: Although RF is based on a probability flow ODE, recent work (SiT) finds that injecting stochastic noise during sampling via a reverse-time SDE actually **improves** generation quality (lower FID). This implies:
- Deterministic sampling in RF is not necessarily optimal.
- Certain forms of noise can be beneficial for RF.

### Core Problem

This observation raises two key questions:

**What kind of stochastic noise** can bring performance gains to RF?

**How** should beneficial noise be introduced into RF?

### Starting Point

The π-noise (positive-incentive noise) framework provides a theoretical foundation by learning beneficial noise through maximizing mutual information between the task and the noise:

$$\max_{\mathcal{E}} MI(\mathcal{T}, \mathcal{E}) = H(\mathcal{T}) - H(\mathcal{T}|\mathcal{E})$$

This paper establishes a connection between the π-noise framework and RF, and designs a π-noise generator to automatically learn the optimal noise.

## Method

### Overall Architecture

The Rectified Noise pipeline consists of two stages:
1. **Pretraining the RF model** to obtain optimal parameters $\psi^*$.
2. **Training the π-noise generator**: freezing the RF parameters, attaching trainable SiT blocks to predict π-noise, and injecting it into the velocity field.

At inference: standard RF inference with π-noise added to the predicted velocity field.

### Key Designs

#### 1. **Defining Task Entropy via the RF Loss**

**Mechanism**: The complexity of what the RF model must learn from a given dataset needs to be quantified. An auxiliary random variable $\alpha$ is introduced to connect the RF loss to information entropy:

$$\alpha | \mathbf{x}, t \sim \mathcal{N}(0, \exp(\mathcal{L}(\mathbf{x}, t; \psi^*)))$$

where $\mathcal{L}(\mathbf{x}, t; \psi^*)$ is the loss of the optimal RF model on sample $\mathbf{x}$ at time $t$. A higher loss yields a larger variance in the auxiliary distribution, a higher entropy, and thus a harder task.

Task entropy is defined as:

$$H(\mathcal{T}) = \frac{1}{2}\mathbb{E}_{\mathbf{x},t}\mathcal{L}(\mathbf{x}, t; \psi^*) + \frac{1}{2}\ln(2\pi e)$$

**Design Motivation**: The auxiliary Gaussian distribution elegantly bridges the regression loss of RF and information entropy, laying the theoretical groundwork for applying the π-noise framework to generative models.

#### 2. **Injecting π-noise into the RF Model**

**Core Derivation**: Maximizing mutual information is equivalent to minimizing the conditional entropy $H(\mathcal{T}|\mathcal{E})$. The noise-conditioned auxiliary loss is defined as:

$$\mathcal{L}(\mathbf{x}, \epsilon, t, \psi^*) = \|\mathbf{v}_{\psi^*} + \epsilon(\mathbf{x}_t, t) - \mathbf{x}_* + \mathbf{x}_0\|^2$$

**Key Insight**: When $p(\epsilon|\mathbf{x}, t) \rightarrow \delta(\epsilon)$ (a Dirac delta, i.e., noise is always zero), the optimization objective degenerates to the standard RF loss. This shows that **standard RF is a special case of ΔRN** where π-noise is identically zero.

The final optimization objective simplifies to:

$$\max_\theta \mathbb{E}_{\mathbf{x}, t, \epsilon \sim \epsilon_\theta} \mathcal{L}(\mathbf{x}, \epsilon, t; \psi^*)$$

The π-noise is parameterized by a neural network $\epsilon_\theta$, trained to maximize the noise-perturbed RF loss.

#### 3. **Two Optimization Strategies**

**Strategy 1: Joint optimization of $\theta$ and $\psi$**

Both parameter sets are unified via the reparameterization trick. For a Gaussian distribution:

$$\hat{\mathbf{v}} = \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) + \boldsymbol{\sigma}_\theta(\mathbf{x}_t, t) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

where $\hat{\mu}_\theta = \mathbf{v}_{\psi^*} + \mu_\theta$ can be predicted by a single network.

**Strategy 2: Freeze $\psi^*$, optimize $\theta$ only (recommended)**

Intermediate feature representations from the pretrained RF model are used as inputs; additional SiT blocks are appended as the π-noise generator, with the final linear layer initialized to zero to ensure the initial output matches the original RF predictions.

**Key Finding**: Strategy 1 exhibits training instability (convergence difficulties caused by injected stochasticity), whereas Strategy 2 (fine-tuning) is superior.

### π-noise Distribution Assumptions

Three reparameterizable distributions are explored:
- **Gaussian**: $\mathbf{z} = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$
- **Gumbel**: $\mathbf{z} = \mu - \beta \odot \log(-\log(\epsilon)), \quad \epsilon_i \sim U(0,1)$
- **Uniform**: $\mathbf{z} = \mathbf{a} + (\mathbf{b}-\mathbf{a}) \odot \epsilon, \quad \epsilon_i \sim U(0,1)$

### Loss & Training

- The pretrained RF model parameters are frozen.
- 0–4 additional SiT blocks are appended as the π-noise generator.
- Only the π-noise generator parameters are trained (0.39%–14.56% additional parameters).
- On ImageNet, the RF model is pretrained for 6M steps, followed by 100K steps of ΔRN fine-tuning; on AFHQ/CelebA-HQ, pretraining runs for 100K/200K steps followed by 10K fine-tuning steps.

## Key Experimental Results

### Main Results

ImageNet-1k 256×256 (without CFG):

| Model | Noise Setting | Extra SiT Blocks | Extra Params | FID↓ | IS↑ | sFID↓ | Prec.↑ | Rec.↑ |
|-------|--------------|-----------------|-------------|------|-----|-------|--------|-------|
| SiT-XL/2 | - | - | - | 10.16 | 123.86 | 12.02 | 0.50 | 0.62 |
| +ΔRN | $\mathcal{N}(\mu,\Sigma)$ | 0 | 0.39% | 9.06 | 130.21 | 11.18 | 0.52 | 0.61 |
| +ΔRN | $\mathcal{N}(\mu,\Sigma)$ | 1 | 3.93% | **9.05** | **132.10** | 11.23 | 0.52 | 0.62 |

Cross-dataset results:

| Dataset | Baseline FID | ΔRN FID | FID Gain |
|---------|-------------|---------|---------|
| ImageNet-1k | 10.16 | 9.05 | **−1.11** |
| AFHQ | 12.33 | 10.44 | **−1.89** |
| CelebA-HQ | 11.25 | 7.73 | **−3.52** |

### Ablation Study

Effect of different noise distribution assumptions (ImageNet-1k):

| Noise Distribution | FID↓ | IS↑ | sFID↓ | Prec.↑ | Rec.↑ |
|-------------------|------|-----|-------|--------|-------|
| None (baseline) | 10.16 | 123.86 | 12.02 | 0.50 | 0.62 |
| **Gaussian** | **9.05** | **132.10** | **11.23** | **0.52** | 0.62 |
| Gumbel | 9.42 | 129.73 | 11.42 | 0.52 | 0.61 |
| Uniform | 10.02 | 124.40 | 11.63 | 0.51 | 0.62 |

Effect of the number of additional SiT blocks ($\mathcal{N}(\mu,\Sigma)$):

| Extra Blocks | Param Ratio | FID↓ | Note |
|-------------|------------|------|------|
| 0 | 0.39% | 9.06 | Linear layer alone is effective |
| 1 | 3.93% | **9.05** | Optimal |
| 2 | 7.48% | 9.08 | Diminishing returns |
| 4 | 14.56% | 9.15 | Slight degradation with excess parameters |

### Key Findings

- **Effective with minimal parameters**: Only 0.39% additional parameters (linear layer only, no extra SiT blocks) suffice to reduce FID from 10.16 to 9.06.
- **Gaussian distribution is optimal**: Among the three noise distributions, Gaussian performs best, likely due to its natural alignment with the forward process of RF (Gaussian noise).
- **Largest improvement on CelebA-HQ**: FID decreases by 3.52, possibly because the more concentrated distribution of facial data makes it easier for π-noise to learn.
- **Fine-tuning strategy outperforms joint training**: Jointly optimizing θ and ψ leads to slower and more unstable FID convergence.
- **Diminishing returns with more SiT blocks**: Zero to one extra block is sufficient; additional blocks may introduce overfitting.

## Highlights & Insights

1. **Theoretical elegance**: The paper establishes a connection between the RF loss and information entropy via an auxiliary Gaussian variable. The derivation is rigorous and concise, ultimately revealing that standard RF is a special case of ΔRN where π-noise is zero.
2. **Exceptional parameter efficiency**: Significant improvement with only 0.39% additional parameters is particularly valuable given the prevailing trend of ever-larger models.
3. **Plug-and-play**: The architecture and weights of the pretrained RF model remain unchanged; only a lightweight π-noise generator is appended.
4. **π-noise visualization**: The paper visualizes how π-noise evolves across timesteps, revealing the spatiotemporal structure of beneficial noise.
5. **Generality**: Consistent improvements across three distinct datasets demonstrate that the method does not rely on a specific data distribution.

## Limitations & Future Work

- Validation is limited to SiT (a specific RF implementation); other Flow Matching architectures (e.g., Flux, SD3) are not tested.
- Experiments are conducted only at 256×256 resolution; performance at higher resolutions is unknown.
- Interaction with Classifier-Free Guidance (CFG) is not explored.
- The failure of the joint training strategy is not analyzed in sufficient depth.
- The interpretability of π-noise is limited — what information does the beneficial noise actually encode?
- Convergence behavior and hyperparameter sensitivity of the 10K fine-tuning steps are not thoroughly discussed.

## Related Work & Insights

- **Connection to SDE sampling**: SiT shows that SDE sampling outperforms ODE sampling; ΔRN can be seen as a further answer to the question "what noise is optimal?" — not random noise, but learned π-noise.
- **Success of π-noise in other tasks**: VPN enhances classical neural networks; PiNI enhances vision-language models; this paper extends the framework to generative models.
- **Insight**: The paradigm of pretrained model + lightweight enhancement module is highly efficient and could generalize to other generative tasks (text generation, video generation, etc.).
- The relationship between ΔRN and LoRA-style methods is worth investigating — both enhance pretrained models with minimal additional parameters.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The theoretical connection between π-noise and RF is a genuinely novel finding with elegant derivation.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three datasets with comprehensive ablations, but high-resolution and CFG experiments are absent.)
- Writing Quality: ⭐⭐⭐⭐ (Theoretical derivations are clear and experimental presentation is well-organized.)
- Value: ⭐⭐⭐⭐ (Reveals the existence of beneficial noise in RF and provides a learning framework with substantial room for follow-up research.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](../../ICCV2025/image_generation/straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[CVPR 2026\] TAUE: Training-free Noise Transplant and Cultivation Diffusion Model](../../CVPR2026/image_generation/taue_training-free_noise_transplant_and_cultivation_diffusion_model.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](../../ICLR2026/image_generation/flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [\[AAAI 2026\] Multi-Metric Preference Alignment for Generative Speech Restoration](multi-metric_preference_alignment_for_generative_speech_restoration.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](../../ICLR2026/image_generation/diverse_text-to-image_generation_via_contrastive_noise_optimization.md)

</div>

<!-- RELATED:END -->
