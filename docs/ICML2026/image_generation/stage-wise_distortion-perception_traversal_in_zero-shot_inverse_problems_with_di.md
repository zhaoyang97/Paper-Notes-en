---
title: >-
  [Paper Note] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models
description: >-
  [ICML 2026][Image Generation][distortion-perception tradeoff] The paper proposes the MAP-RPS two-stage framework: it first utilizes the score function of a diffusion model for MAP estimation to approximate the MMSE solut…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "distortion-perception tradeoff"
  - "zero-shot inverse problem"
  - "diffusion posterior sampling"
  - "MAP estimation"
  - "latent diffusion"
date: 2026-05-08
content_hash: 802ebc79b8f53474
---

# Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models

**Conference**: ICML 2026  
**arXiv**: [2605.28711](https://arxiv.org/abs/2605.28711)  
**Code**: https://github.com/weigerzan/MAP_RPS (Available)  
**Area**: Diffusion Models / Image Restoration / Inverse Problems  
**Keywords**: distortion-perception tradeoff, zero-shot inverse problem, diffusion posterior sampling, MAP estimation, latent diffusion

## TL;DR
The paper proposes the MAP-RPS two-stage framework: it first utilizes the score function of a diffusion model for MAP estimation to approximate the MMSE solution (a low-distortion starting point), and then re-noises the MAP result to time $t_0$ followed by posterior sampling (traversing the D-P curve towards high perceptual quality). A single pre-trained diffusion model can flexibly navigate the distortion-perception trade-off at inference time, achieving multi-task SOTA on MS-COCO when extended to latent diffusion.

## Background & Motivation

**Background**: Diffusion models have become a mainstream framework for zero-shot Bayesian inverse problems (super-resolution, deblurring, inpainting, compressed sensing, HDR). Representative methods like DPS, $\Pi$GDM, ReSample, and PSLD sample from the posterior $p_{X\mid Y}$ by approximating $\nabla_{\mathbf{x}_t}\log p_t(\mathbf{y}\mid\mathbf{x}_t)$.

**Limitations of Prior Work**: As proven by Blau & Michaeli (2018), the distortion-perception (D-P) tradeoff dictates that distortion metrics (PSNR/SSIM) and perception metrics (LPIPS/FID) are inherently contradictory. Pure posterior sampling resides at the "high perception/high distortion" end of the D-P curve, while pure MMSE estimation resides at the other. Real-world applications (e.g., medical imaging favoring fidelity, consumer photography favoring perception) require sliding between these ends. Existing methods either rely on tuning sampling steps, averaging multiple samples, manual hyperparameter tuning, or training new models, **lacking a principled, inference-time controllable, and computationally efficient D-P traversal mechanism**.

**Key Challenge**: The two optimal estimators at the ends of the D-P curve—$X_{\text{MMSE}}=\mathbb{E}[X\mid Y]$ and the perception-optimal estimator (e.g., posterior sampling)—arise from completely different optimization objectives. Theoretically, Freirich et al. demonstrated that "linear interpolation between them traverses the D-P curve" (Equation 15). However, calculating the MMSE end in zero-shot diffusion is difficult, as it requires repeated posterior sampling and averaging, leading to explosive computational costs.

**Goal**: Decomposition into two sub-problems: (1) efficiently obtaining a low-distortion starting point (approximate MMSE) in a zero-shot diffusion framework without sample averaging; (2) **continuously and controllably** "pushing" this low-distortion point toward the high-perception end.

**Key Insight**: It is observed that in image restoration, the "ground truth" is often nearly unique, implying that the posterior distribution $p_{X\mid Y}$ is approximately **strongly log-concave** (unimodal and concentrated) in many scenarios. Under this mild assumption, the distance between MAP and MMSE is provably bounded ($\mathcal{O}(\sqrt{n_x/\mu})$), yet MAP is significantly cheaper to compute (gradient optimization vs. repeated sampling). In the second stage, the model leverages the natural bidirectional flow of "noising-denoising" in diffusion models: re-noising the MAP result to an intermediate time $t_0$ and running the posterior sampling SDE back to 0. A larger $t_0$ approaches pure posterior sampling (perceptually optimal), while $t_0=0$ degrades to MAP (distortion optimal). **A single scalar $t_0$ traverses the D-P curve during inference.**

**Core Idea**: Using **MAP as a low-distortion anchor + re-noise time $t_0$ as a D-P slider** transforms the binary choice between MMSE and posterior sampling into a continuous adjustment, without retraining or multi-sample averaging.

## Method

### Overall Architecture
MAP-RPS is a two-stage algorithm. The inputs are the observation $\mathbf{y}=\mathcal{A}(\mathbf{x})+\sigma_{\mathbf{y Vast}}\mathbf{n}$ and a **pre-trained (fixed)** diffusion score network $\mathbf{s}_\theta(\mathbf{x}_t,t)$. The output is the reconstructed image $\hat{\mathbf{x}}_0$:

- **Stage 1**: Starting from random initialization $\mathbf{x}^{(0)}$, $N$ steps of stochastic gradient ascent are performed using the prior gradient provided by the diffusion score to solve $\mathbf{x}_{\text{MAP}}=\arg\max_{\mathbf{x}}\log p_{Y\mid X}(\mathbf{y}\mid\mathbf{x})+\log p_X(\mathbf{x})$, yielding a low-distortion "anchor."
- **Stage 2**: $\mathbf{x}_{\text{MAP}}$ is noised along the forward SDE to time $t_0$, and an off-the-shelf posterior sampler (DPS / $\Pi$GDM, etc.) is run from $t_0$ back to $0$. $t_0\in[0,T]$ serves as the D-P slider set by the user at inference time.

Applying this process to the latent space (performing MAP and RPS in the VAE latent space, with likelihood approximated via the decoder $\log p_{Y\mid Z}(\mathbf{y}\mid\mathbf{z})\approx\log p_{Y\mid X}(\mathbf{y}\mid\mathcal{D}(\mathbf{z}))$) results in LMAP-RPS, which can directly utilize large models like Stable Diffusion.

### Key Designs

1.  **Stage 1: MAP as a Low-Distortion Anchor (with Provable Error Bounds)**:
    -   **Function**: Obtains the low-distortion endpoint of the D-P curve via a single optimization path without multi-sample averaging in a zero-shot setting.
    -   **Mechanism**: (a) Theoretical side: Theorem 3.2 assumes $\mu$-strongly log-concave posterior and proves $\mathbb{E}\|X_{\text{MAP}}-X_{\text{MMSE}}\|\le\sqrt{n_x/\mu}$ and $\mathbb{E}\|X-X_{\text{MAP}}\|^2\le D^*+n_x/\mu$. As the posterior concentrates ($\mu$ increases), the bound tightens. (b) Algorithmic side: Theorem 3.3 provides a formula for estimating prior gradients using the diffusion score: $\nabla_{\mathbf{x}}\log p_X(\mathbf{x})=\frac{1-\bar\alpha_{t_1}}{r_{t_1}^2\sqrt{\bar\alpha_{t_1}}}\mathbb{E}_{p_{X_{t_1}\mid X_0}}\nabla_{\mathbf{x}_{t_1}}\log p_{t_1}(\mathbf{x}_{t_1})$. Combined with an $\ell_2$ likelihood term, MAP can be solved via SGD.
    -   **Design Motivation**: Calculating MMSE directly is too expensive. The authors use the "approximate unimodality" of image restoration posteriors to proxy MMSE with MAP and use theory to ensure error control.

2.  **Stage 2: Re-noised Posterior Sampling with $t_0$ as a Parameter**:
    -   **Function**: Interpolates any point on the D-P curve using a scalar $t_0\in[0,T]$, where $t_0=0$ is MAP (lowest distortion) and $t_0=T$ is standard posterior sampling (best perception).
    -   **Mechanism**: $\mathbf{x}_{\text{MAP}}$ is pushed forward to $\mathbf{x}_{t_0}\sim\mathcal{T}(0,t_0)_\#\delta_{\mathbf{x}_{\text{MAP}}}$, then run back through the posterior sampling SDE $\tilde{\mathcal{T}}(t_0,0;\mathbf{y})_\#$. Theorem 3.5 proves the Wasserstein-2 distance bound to the true distribution: $W_2(p_X,\,p_{0\to t_0\to 0})\le(\bar\alpha_{t_0})^{1-L_s}\sqrt{2n_x/\mu}+\epsilon_{\text{score}}$. Corollary 3.6 states that if the Lipschitz constant $L_s < 1$, this bound is **monotonically decreasing** w.r.t $t_0$.
    -   **Design Motivation**: Diffusion models naturally provide asymmetric forward/backward structures. By anchoring the distortion end with MAP and re-injecting randomness via the score network at the perception end, $t_0$ becomes a bridge for plug-and-play D-P control.

3.  **LMAP-RPS: Leveraging Large Models in Latent Space**:
    -   **Function**: Extends MAP-RPS to pre-trained latent diffusion models (e.g., Stable Diffusion) for real-world MS-COCO scenarios.
    -   **Mechanism**: Optimization and sampling occur in latent $\mathbf{z}\in\mathbb{R}^d$. The likelihood is backpropagated through the decoder $\mathcal{D}$: $\log p_{Y\mid Z}(\mathbf{y}\mid\mathbf{z})\approx\log p_{Y\mid X}(\mathbf{y}\mid\mathcal{D}(\mathbf{z}))$.
    -   **Design Motivation**: Latent space allows leveraging high-resolution large models and reduces computational complexity compared to pixel-space baselines.

### Loss & Training
**Training-free**—The pipeline is a zero-shot inference algorithm; parameters are not updated. Stage 1 optimizes $-\log p_{Y\mid X}(\mathbf{y}\mid\mathbf{x})-\log p_X(\mathbf{x})$ via SGD. Stage 2 uses Euler–Maruyama to solve the posterior SDE, where the posterior score can be implemented via DPS/$\Pi$GDM. Key hyperparameters: MAP steps $N$, step size $\gamma$, and re-noise time $t_0$.

## Key Experimental Results

### Main Results
Evaluated on 6 latent-space tasks in MS-COCO (inpainting, SR 4×, deblur, CS 2×, HDR, nonlinear deblur) against baselines including Latent-DPS, ReSample, PSLD, and others:

| Task | Metric | LMAP-RPS (0) | LMAP-RPS (600) | Best Baseline |
|------|------|---------------|------------------|----------|
| Inpainting | PSNR↑ / LPIPS↓ / FID↓ | **28.14** / **0.2769** / **61.35** | 27.22 / 0.3084 / 92.23 | Latent-SITCOM 28.06 |
| SR 4× | PSNR↑ / LPIPS↓ / FID↓ | **25.03** / 0.3888 / 107.57 | 24.49 / **0.3505** / **87.20** | LDIR 25.01 |
| Deblur Aniso | PSNR↑ / LPIPS↓ / FID↓ | **26.42** / 0.3504 / 90.80 | 25.59 / **0.3503** / 85.01 | Latent-SITCOM 26.28 |
| CS 2× | PSNR↑ / LPIPS↓ / FID↓ | 22.87 / 0.3498 / **113.06** | **22.90** / **0.3497** / 113.58 | ReSample 21.82 |
| HDR | PSNR↑ / LPIPS↓ / FID↓ | **25.86** / **0.3595** / **108.79** | 23.06 / 0.3944 / 116.08 | Latent-DCDP 23.31 |

### Ablation Study
| Configuration | Key Observation | Explanation |
|------|---------|------|
| $t_0=0$ (Pure MAP) | Highest PSNR, poor LPIPS/FID | Low-distortion end of the D-P curve. |
| $t_0=600$ (High re-noise) | Better LPIPS/FID, lower PSNR | Increased $t_0 \rightarrow$ decreased $\bar\alpha_{t_0} \rightarrow$ lower perceptual bound. |
| Stage 1 Removed | Degradation to std. posterior sampling | Loss of distortion anchor; PSNR drops significantly. |
| Stage 2 Removed | Poor LPIPS/FID | Perceptual quality limited to MAP unimodal level. |

### Key Findings
- **Two $t_0$ settings achieve most SOTAs**: LMAP-RPS(0) excels in distortion-sensitive tasks (HDR, nonlinear deblur), while LMAP-RPS(600) leads in perception-sensitive tasks (SR 4× LPIPS/FID).
- **Lower computational cost**: Total NFE (neural function evaluations) is lower than精修 methods like ReSample/PSLD because the MAP phase is cheap and Stage 2 starts from $t_0$ instead of $T$.

## Highlights & Insights
- **Re-interpreting diffusion "noising/denoising"**: The paper operationalizes the theoretical D-P interpolation as a simple choice of $t_0$ on the diffusion time axis.
- **Theoretical Grounding**: Using MAP as a proxy for MMSE under strong log-concavity provides a computable alternative with closed-form error bounds.

## Limitations & Future Work
- **Reliance on Log-concavity**: The guarantee may fail in highly multimodal inverse problems (e.g., text-to-image with heavy occlusion).
- **Manual $t_0$ Selection**: Although simplified to a single parameter, an automated strategy for selecting $t_0$ based on application or user preference is missing.
- **VAB Approximation**: Approximate likelihoods in latent space $\log p_{Y\mid X}(\mathbf{y}\mid\mathcal{D}(\mathbf{z}))$ introduce non-linear errors that are not theoretically quantified.

## Related Work & Insights
- **vs. DPS / $\Pi$GDM**: These are limited to the perception end; MAP-RPS uses them as sub-modules but adds an anchor and slider.
- **vs. Wang et al. 2025**: Wang uses variance scaling which requires Gaussian assumptions; MAP-RPS uses a more interpretable time-domain $t_0$ with clearer bounds.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Saving Foundation Flow-Matching Priors for Inverse Problems](saving_foundation_flow-matching_priors_for_inverse_problems.md)
- [\[ICML 2026\] Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences](localizing_memorized_regions_in_diffusion_models_via_coordinate-wise_curvature_d.md)
- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](../../NeurIPS2025/image_generation/a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)
- [\[ICML 2026\] Zeroth-Order Non-Log-Concave Sampling with Variance Reduction and Applications to Inverse Problems](zeroth-order_non-log-concave_sampling_with_variance_reduction_and_applications_t.md)

</div>

<!-- RELATED:END -->
