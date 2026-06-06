---
title: >-
  [Paper Note] Stable Velocity: A Variance Perspective on Flow Matching
description: >-
  [ICML 2026][Image Generation][flow matching] This paper revisits flow matching from the neglected perspective of "conditional velocity variance." It discovers that training trajectories naturally split into a high-varian…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "flow matching"
  - "variance reduction"
  - "representation alignment"
  - "sampling acceleration"
  - "stochastic interpolants"
date: 2026-05-08
content_hash: b0d8689d1ec2feba
---

# Stable Velocity: A Variance Perspective on Flow Matching

**Conference**: ICML 2026  
**arXiv**: [2602.05435](https://arxiv.org/abs/2602.05435)  
**Code**: https://github.com/linYDTHU/StableVelocity  
**Area**: Image Generation / Flow Matching / Diffusion Models  
**Keywords**: flow matching, variance reduction, representation alignment, sampling acceleration, stochastic interpolants

## TL;DR
This paper revisits flow matching from the neglected perspective of "conditional velocity variance." It discovers that training trajectories naturally split into a high-variance region near the prior and a low-variance region near the data. Based on this, it proposes a unified framework, Stable Velocity, featuring an unbiased multi-sample variance reduction loss (StableVM), a variance-aware representation alignment (VA-REPA) active only in low-variance regions, and a fine-tuning-free sampling accelerator (StableVS) utilizing closed-form solutions in low-variance regions. The method achieves improved training efficiency and $>2\times$ sampling acceleration on ImageNet 256 and large-scale models like SD3.5/Flux/Qwen-Image/Wan2.2.

## Background & Motivation

**Background**: The flow matching/stochastic interpolant paradigm, represented by Conditional Flow Matching (CFM), has unified diffusion and flow models. By training neural networks to fit the conditional velocity field $v_t(x_t \mid x_0)$, these models learn the probability flow from a prior $\mathcal{N}(0, I)$ to the data distribution. This approach is the standard training pipeline for large generative models such as SD3, Flux, and Wan2.2.

**Limitations of Prior Work**: The training objective of CFM, $v_t(x_t \mid x_0)$, is essentially a single-sample Monte Carlo estimate of the true marginal velocity field $v_t(x_t)$, resulting in extreme variance. Particularly as $t$ approaches 1 (near the prior and far from the data), a noisy sample $x_t$ can be explained by almost any data point, causing the regression target to fluctuate wildly and leading to slow, unstable optimization. Furthermore, auxiliary representation alignment losses like REPA are applied indiscriminately across all $t$, without analysis of their temporal effectiveness.

**Key Challenge**: CFM treats the entire interval $t \in [0, 1]$ as a "homeomorphic" segment for training and sampling. However, the variance of the conditional velocity $\mathcal{V}_{\text{CFM}}(t)$ is highly non-uniform along $t$: it is nearly zero in the early stages and explodes toward the end. Existing methods neither perform variance reduction for high-variance segments nor exploit the favorable structure of low-variance segments.

**Goal**: (1) Provide an analytical characterization of the training variance in flow matching, (2) construct an unbiased variance reduction objective for high-variance regions, (3) enable auxiliary representation alignment losses to adaptively act only on meaningful intervals, and (4) leverage simplified dynamics in low-variance regions to accelerate sampling.

**Key Insight**: By calculating the trace of the conditional velocity covariance, $\mathcal{V}_{\text{CFM}}(t)$, and plotting the curves on GMM, CIFAR-10, and ImageNet latent space, a distinct split point $\xi$ is observed: variance is near 0 for $t < \xi$ and rises sharply for $t \ge \xi$. The higher the data dimensionality, the closer $\xi$ is to 1, expanding the low-variance region. This curve serves as the physical basis for the designs in this paper.

**Core Idea**: Replace the uniformly applied CFM/REPA/sampler with a unified "time-partitioned governance" framework. High-variance regions employ multi-sample variance reduction, while low-variance regions enhance representation supervision and enable large-step closed-form sampling.

## Method

### Overall Architecture
Given data $q(x_0)$ and a linear interpolation path $x_t = (1-t)x_0 + t\varepsilon$, the variance of the CFM objective $\mathcal{V}_{\text{CFM}}(t)$ is first characterized at the probabilistic level to identify the split point $\xi$. On the training side, the CFM loss is replaced with StableVM (most beneficial in high-variance regions) and augmented with VA-REPA (weighted representation alignment restricted to the $t < \xi$ segment). During inference, the standard ODE/SDE solver is replaced by StableVS in the $t < \xi$ segment for large-step closed-form updates, while the original model is used for $t \ge \xi$. These three components share the same $\xi$, are mutually orthogonal, and can be integrated individually or together into existing REPA/REG/iREPA pipelines.

### Key Designs

1. **StableVM: Self-Normalized Multi-Sample Conditional Velocity Objective**:
    - **Function**: Replaces the single-sample CFM objective with a self-normalized weighted average of $n$ reference samples $\{x_0^i\}$, serving as a multi-sample unbiased estimate of the marginal velocity field $v_t(x_t)$.
    - **Mechanism**: $x_t$ is sampled from a mixture path $p_t^{\text{GMM}}(x_t \mid \{x_0^i\}) = \tfrac{1}{n}\sum_i p_t(x_t \mid x_0^i)$ instead of a single conditional path. The regression target becomes $\widehat{v}_{\text{StableVM}}(x_t; \{x_0^i\}) = \sum_k p_t(x_t \mid x_0^k) v_t(x_t \mid x_0^k) / \sum_j p_t(x_t \mid x_0^j)$. The paper proves this objective remains unbiased (Thm 3.1), shares the same global optimum $v_t(x_t)$ as CFM, and has strictly lower variance (Thm 3.2), decaying at a rate of $O(1/n)$ as $n$ increases (Thm 3.3). For conditional generation, a FIFO class-based memory bank (capacity $K=256$) is maintained to address intra-batch sparsity of identical classes.
    - **Design Motivation**: By directly targeting the variance of the objective, this approach addresses the root cause: it preserves the global optimum of CFM (remaining unbiased) while naturally extending biased schemes like STF to general stochastic interpolants through mixture sampling.

2. **VA-REPA: Variance-Aware Representation Alignment**:
    - **Function**: Ensures that REPA-like auxiliary semantic alignment losses are active only in low-variance regions and automatically disabled in high-variance regions.
    - **Mechanism**: It is observed that the pre-trained REPA model loss $\ell_{\text{RA}}$ remains low and learnable in low-variance regions but saturates at high values in high-variance regions, where deterministic semantic recovery from pure noise is ill-posed. A weight $w(t) \in [0, 1]$ is introduced, yielding the total loss $\mathcal{L} = \mathcal{L}_{\text{StableVM}} + \lambda_{\text{RA}} \, \mathbb{E}_{t,x_t}[w(t) \ell_{\text{RA}}(x_t)] / \mathbb{E}_t[w(t)]$. Three implementations for $w(t)$ are provided: a hard threshold $\mathbb{I}[t<\xi]$, sigmoid relaxation $\sigma(k(\xi - t))$, and an SNR-based form $\text{SNR}(t)/(\text{SNR}(t) + \text{SNR}(\xi))$, with sigmoid as the default. Normalization by $\mathbb{E}_t[w(t)]$ is critical to prevent auxiliary gradients from being diluted when most samples fall in high-variance regions.
    - **Design Motivation**: REPA works by using semantic encoders (e.g., DINO) to supplement semantic structures that the model itself struggles to learn in low-noise stages. Forcing the model to align with fixed semantics when $x_t$ is nearly pure noise introduces mismatch. Entrusting the "when to align" switch to the variance curve provides a targeted solution.

3. **StableVS: Closed-Form Sampling Accelerator for Low-Variance Regions**:
    - **Function**: Substitutes standard ODE/SDE solvers with large-step closed-form updates for $t < \xi$, achieving $>2\times$ sampling acceleration without fine-tuning.
    - **Mechanism**: In low-variance regions, $v_t(x_t) \approx v_t(x_t \mid x_0)$. Thus, the reverse SDE can be formulated as a DDIM-style posterior $p_\tau(x_\tau \mid x_t, v_t) = \mathcal{N}(\mu_{\tau \mid t}, \beta_t^2 I)$, where $\beta_t = f_\beta \sigma_\tau$ and mean $\mu_{\tau \mid t} = (\rho_t - \lambda_t \sigma'_t/\sigma_t) x_t + \lambda_t v_t(x_t)$. The corresponding PF-ODE also has a closed-form solution $x_\tau = \sigma_\tau[(1/\sigma_t - \sigma'_t/\sigma_t \cdot \Psi_{t,\tau}) x_t + \Psi_{t,\tau} v_t(x_t)]$. In the case of linear interpolation and $\beta_t = 0$, both degrade to $x_\tau = x_t + (\tau - t) v_t(x_t)$, implying that the trajectory in low-variance segments is a constant-velocity line, allowing for exact integration with arbitrarily large steps. On SD3.5/Flux/Qwen-Image/Wan2.2, setting $\xi = 0.85$ allows maintaining quality with only 9 steps in the low-variance region.
    - **Design Motivation**: Traditional solvers use small steps to handle unknown curvature throughout the path. Variance analysis informs the solver that the early segment is essentially linear, allowing the step quota to be reallocated to high-variance segments that truly require small steps.

### Loss & Training
The final training objective is the StableVM loss $\mathcal{L}_{\text{StableVM}}$ plus the variance-aware normalized $\lambda_{\text{RA}}$ REPA term. Default configuration: $\xi = 0.7$ (training) / $0.85$ (sampling), bank capacity $K = 256$, and $w_{\text{sigmoid}}$ weighting. The backbone is SiT-XL/2 on the ImageNet 256 latent space. The number of reference samples $n$ is implemented via intra-batch combinations without extra network overhead.

## Key Experimental Results

### Main Results
ImageNet 256×256, SiT-XL/2 + CFG ($w=1.8$, interval-based CFG):

| Method | Epoch | FID↓ | sFID↓ | IS↑ | Prec.↑ | Rec.↑ |
|:---|:---|:---|:---|:---|:---|:---|
| SiT-XL/2 | 1400 | 2.06 | 4.50 | 270.3 | 0.82 | 0.59 |
| REPA | 80 | 1.98 | 4.60 | 263.0 | 0.80 | 0.61 |
| REPA | 800 | 1.42 | 4.70 | 305.7 | 0.80 | 0.65 |
| iREPA | 80 | 1.93 | 4.59 | 268.8 | 0.80 | 0.60 |
| REG | 480 | 1.40 | 4.24 | 296.9 | 0.77 | 0.66 |
| **Ours (StableVM+VA-REPA)** | 80 | **1.80** | 4.52 | 272.4 | 0.81 | 0.60 |
| **Ours** | 480 | **1.44** | 4.49 | 302.9 | 0.80 | 0.64 |
| REPA-E† (VAE fine-tuned) | 800 | 1.12* | 4.09* | 302.9* | 0.79* | 0.66* |
| **Ours (class-balanced)** | 480 | **1.33*** | 4.46* | 307.8* | 0.80* | 0.64* |

Ours at 80 epochs outperforms the 80-epoch baselines of REPA/iREPA/REG. At 480 epochs, results are close to those of REPA-E which uses 800 epochs and VAE fine-tuning.

Across model scales (no CFG, 100k iter): SiT-B/2 FID 52.06→49.69, SiT-L/2 22.75→21.03, SiT-XL/2 18.59→17.12. For SiT-XL/2 at 400k iter, FID improves from 8.13→7.58.

### Ablation Study

Orthogonal addition to different REPA variants (100k iter):

| Method | FID↓ | sFID↓ | IS↑ | Prec.↑ | Rec.↑ |
|:---|:---|:---|:---|:---|:---|
| REPA | 18.59 | 5.39 | 70.6 | 0.64 | 0.62 |
| + Ours | **17.12** | 5.39 | 74.8 | 0.65 | 0.63 |
| REG | 8.90 | 5.50 | 125.3 | 0.72 | 0.59 |
| + Ours | **8.11** | 5.34 | 128.8 | 0.74 | 0.60 |
| iREPA | 16.62 | 5.31 | 76.7 | 0.65 | 0.63 |
| + Ours | **16.02** | 5.30 | 78.6 | 0.66 | 0.63 |

Ablation of split point $\xi$: $\xi=0.6$ is slightly better at 100k (17.38), but $\xi=0.7$ is superior overall at 400k; $\xi=0.8$ degrades performance (as it includes too much of the noise segment in alignment).

### Key Findings
- **Orthogonal/Drop-in components**: The three modules consistently improve performance when added to vanilla REPA, REG, or iREPA, proving "variance partitioning" is a universal design principle rather than a trick tied to a specific loss.
- **Stage-dependent $\xi$ selection**: Smaller $\xi$ values help initial convergence in early stages, while larger values provide a broader alignment interval for fine semantic structures in later stages. Final selection $\xi=0.7$ is a balanced compromise.
- **StableVS for T2I/T2V**: On SD3.5/Flux/Qwen-Image/Wan2.2, StableVS achieves $>2\times$ acceleration without fine-tuning while maintaining near-identical PSNR/SSIM/LPIPS, indicating that large models indeed learn nearly linear velocity fields in low-variance regions.

## Highlights & Insights
- Visualizes the "conditional velocity variance curve $\mathcal{V}_{\text{CFM}}(t)$" as an explicit physical quantity and derives a unified design for training, auxiliary losses, and samplers. This "diagnostic first, prescribe later" approach is rare in flow matching literature.
- The self-normalized importance sampling of StableVM provides an $O(1/n)$ variance decay for "free." Unlike the biased STF which is limited to VP diffusion, this theory enables variance reduction for all stochastic interpolants.
- VA-REPA exposes a mismatch in the underlying assumption of REPA-like works—treating semantic alignment as a task for the full interval despite semantic information being destroyed in high-noise segments. This insight is applicable to any task supervising diffusion models with pre-trained representations.
- The simplicity of StableVS ($x_\tau = x_t + (\tau-t) v_t$ under linear interpolation and $\beta_t=0$) suggests the existence of significant "low-variance linear segments" in mainstream T2I/T2V models. Future distillation/consistency methods could focus targets on high-variance segments to further compress step counts.

## Limitations & Future Work
- **Limitations acknowledged by authors**: The precise location of $\xi$ depends on unknown data distributions; it currently relies on empirical values ($0.7$ for training, $0.85$ for sampling) without an automatic selection mechanism. The class memory bank may remain sparse for extreme long-tail classes.
- **Independent observations**: (1) All training experiments are on SiT models with ImageNet 256; training gains on models like Flux/SD3.5 from scratch are not shown, requiring more evidence for scalability. (2) The "equivalent linear segment" assumption in StableVS depends on $v_t$ being accurately learned; it might introduce errors in under-trained models. (3) The trade-off between reference sample count $n$, batch size, and VRAM is not fully explored.
- **Improvement ideas**: Online estimation of $\mathcal{V}_{\text{CFM}}(t)$ to adaptively adjust $\xi$; linking the $w(t)$ of VA-REPA to the loss curve itself for automatic gating; and combining StableVS with consistency distillation to create hybrid 1-step/few-step sampling.

## Related Work & Insights
- **vs CFM (Tong et al., 2023)**: While CFM uses single-sample conditional velocity, this work provides a variance analysis and proposes StableVM as an unbiased multi-sample alternative that preserves the global optimum while strictly reducing variance.
- **vs STF (Xu et al., 2023)**: STF also uses multi-sample reweighting but is limited to VP diffusion and suffers from bias due to inputs originating from a single conditional path. StableVM achieves true unbiasedness for general interpolants via $p_t^{\text{GMM}}$ mixture sampling.
- **vs REPA / REG / iREPA (Yu et al., 2024; Wu et al., 2025b; Singh et al., 2025)**: These works apply representation alignment uniformly. This paper proves alignment is ill-posed in high-variance regions via the variance curve; VA-REPA adds a gate that can be stacked on these methods for consistent gains.
- **vs DDIM / Rectified Flow / Consistency Models**: StableVS is most similar to the first two—it requires no extra training but acknowledges that the model has already learned the low-variance segment as a near-straight line, using structure rather than training to achieve speed.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Using the variance curve as a unified design principle for training, auxiliary, and sampling stages is a highly systematic perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive ImageNet experiments and large-model sampling validation. Half a star deducted for the lack of training-side validation on large models.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative is clean, centered on the variance curve, with formal presentation of theorems and formulas.
- **Value**: ⭐⭐⭐⭐⭐ Drop-in training gains and fine-tuning-free 2× inference acceleration are directly applicable to industrial T2I/T2V deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)
- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](../../CVPR2026/image_generation/vecor_--_velocity_contrastive_regularization_for_flow_matching.md)
- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)
- [\[ICML 2026\] AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching](ag-repa_causal_layer_selection_for_representation_alignment_in_audio_flow_matchi.md)
- [\[ICML 2026\] Exploring and Exploiting Stability in Latent Flow Matching](exploring_and_exploiting_stability_in_latent_flow_matching.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](../../CVPR2026/image_generation/vecor_--_velocity_contrastive_regularization_for_flow_matching.md)
- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)
- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)
- [\[ICML 2026\] AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching](ag-repa_causal_layer_selection_for_representation_alignment_in_audio_flow_matchi.md)
- [\[ICML 2026\] Exploring and Exploiting Stability in Latent Flow Matching](exploring_and_exploiting_stability_in_latent_flow_matching.md)

</div>

<!-- RELATED:END -->
