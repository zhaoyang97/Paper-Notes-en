---
title: >-
  [Paper Note] Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation
description: >-
  [ICLR 2026][Medical Imaging][Cross-modality image translation] This paper proposes CDTSDE, a framework that embeds a learnable spatial-adaptive domain mixing field $\Lambda_t$ into the reverse SDE of diffusion models…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Cross-modality image translation"
  - "diffusion SDE"
  - "domain shift scheduling"
  - "spatial adaptive mixing"
  - "reverse SDE"
date: 2026-05-08
content_hash: aaf7c26446ce252e
---

# Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation

**Conference**: ICLR 2026
**arXiv**: [2601.18623](https://arxiv.org/abs/2601.18623)  
**Code**: [https://github.com/LaplaceCenter/CDTSDE](https://github.com/LaplaceCenter/CDTSDE)  
**Area**: Medical Imaging / Diffusion Models
**Keywords**: Cross-modality image translation, diffusion SDE, domain shift scheduling, spatial adaptive mixing, reverse SDE

## TL;DR
This paper proposes CDTSDE, a framework that embeds a learnable spatial-adaptive domain mixing field $\Lambda_t$ into the reverse SDE of diffusion models, enabling cross-modality translation paths to traverse low-energy manifolds. The approach achieves higher fidelity with fewer denoising steps on MRI modality conversion, SAR→Optical, and industrial defect semantic mapping tasks.

## Background & Motivation

**Background**: Cross-modality image translation (e.g., MRI T1→T2, SAR→Optical) has transitioned from the GAN era into the diffusion model era, with diffusion-based methods surpassing GANs in both stability and generation quality.

**Limitations of Prior Work**: Existing diffusion-based translation methods universally rely on a **fixed linear interpolation** $d_t = \eta_t \hat{x}_0^{\text{src}} + (1-\eta_t) x_0$ between source and target domains. This linear path traverses high-energy regions between the two modality manifolds, forcing the sampler to perform substantial off-manifold corrections.

**Key Challenge**: The linear interpolation assumption treats the source-to-target transformation as globally uniform, whereas real cross-modality discrepancies are spatially highly heterogeneous — certain regions (e.g., edges with large textural differences) require far more correction than homogeneous regions.

**Goal**: Can the domain shift schedule itself learn an "adaptively curved" path that bypasses high-energy regions, thereby reducing the denoising burden and improving semantic consistency?

**Key Insight**: The authors approach the problem from the geometric perspective of path energy functionals, proving that under mild heterogeneity conditions, pixel-wise adaptive paths have strictly lower energy than any globally scheduled path (Theorem 1).

**Core Idea**: Upgrade domain shift from "global linear interpolation" to a "pixel-wise, channel-wise learnable nonlinear mixing field," and embed it into the drift term of the diffusion SDE.

## Method

### Overall Architecture
CDTSDE (Cross-Domain Translation SDE) introduces an adaptive domain mixing field $\Lambda_t \in (0,1)^{C \times H \times W}$ into the VP diffusion process. The marginal distribution of the forward process has mean $\sqrt{\bar\alpha_t} \cdot d_t$ (where $d_t = \Lambda_t \odot \hat{x}_0^{\text{src}} + (1-\Lambda_t) \odot x_0$), and the reverse SDE drift term includes an explicit domain-shift restoring force. The input is a source-modality image, and the output is the translated target-modality result.

### Key Designs

1. **Adaptive Dynamic Domain Shift (Spatial-Adaptive Domain Mixing Field)**

    - **Function**: Predicts a full-resolution mixing field $\Lambda_t \in (0,1)^{C \times H \times W}$ at each reverse timestep $t$.
    - **Mechanism**: A lightweight convolutional network $\mathcal{S}_\theta$ receives the base linear schedule $\lambda_t^{\text{lin}}$ and positional encoding $\pi(p)$, and outputs a spatial modulation signal $h_{t,c}(p)$. A zero-centered transform $g = 2h-1$ and an endpoint-preserving interpolation formula $f_{t,c} = \lambda_t^{\text{lin}}[1 + g_{t,c}(1-\lambda_t^{\text{lin}})]$ are applied, followed by a calibrated logistic map to compress values into $(0,1)$, yielding $\Lambda_{t,c}(p)$.
    - **Design Motivation**: **Theorem 1** proves that under local geometric heterogeneity (different pixels have different optimal mixing ratios) and non-degenerate contrast conditions, $\inf_{\Lambda \in \mathcal{C}_{\text{pix}}} \mathcal{E}[d] < \inf_{\Lambda \in \mathcal{C}_{\text{glob}}} \mathcal{E}[d]$, i.e., pixel-wise scheduling strictly outperforms global scheduling. This provides theoretical justification for spatial adaptivity.

2. **Domain-Aware Forward/Reverse SDE (Cross-Modal Diffusion Process)**

    - **Function**: Embeds the adaptive mixing field into the forward marginals and reverse drift term of VP diffusion.
    - **Mechanism**: The forward marginal is $q(x_t | x_0, \hat{x}_0^{\text{src}}) = \mathcal{N}(\sqrt{\bar\alpha_t} d_t, \sigma_t^2 I)$. The additional drift $\sqrt{\bar\alpha_t} \dot\Lambda(t) \odot (\hat{x}_0^{\text{src}} - x_0)$ causes the forward mean to track the domain mixing path. The reverse SDE (Eq. 9) comprises three force terms: the standard drift $f(t)x_t$, a domain-shift restoring force, and the score function.
    - **Design Motivation**: Encoding domain shift dynamics directly into the generative process ensures that even large integration steps remain on-manifold, since each update inherently carries a domain-aware correction direction.

3. **Exact Solution & First-Order Sampler**

    - **Function**: Derives the exact solution of the reverse SDE under a change of coordinates (Proposition 1) and designs a first-order numerical sampler.
    - **Mechanism**: A coordinate transform $\Upsilon_t = \sqrt{\bar\alpha_t}(1-\Lambda_t)$, $y_t = x_t \oslash \Upsilon_t$, $\lambda_t = \sigma_t \oslash \Upsilon_t$ reduces the reverse SDE to a form solvable exactly via the variation-of-constants formula. Proposition 1 gives an exact solution with four terms: (a) scaled propagation, (b) data prediction integral, (c) source image recovery, and (d) stochastic term.
    - **Design Motivation**: The exact solution guarantees marginal consistency; the first-order sampler achieves ~15 dB PSNR in only 5 steps (1.8 s/image), far more efficient than methods such as BBDM that require 1000 steps.

4. **Middle-Point Truncation**

    - **Function**: Initializes sampling at $x_{t_1} \sim \mathcal{N}(\sqrt{\bar\alpha_{t_1}} \hat{x}_0^{\text{src}}, \sigma_{t_1}^2 I)$ for a starting time $t_1 < T$, skipping the first $T - t_1$ steps.
    - **Design Motivation**: For $t \geq t_1$, $\Lambda_t = 1$, so the forward mean becomes a noise process centered on the pure source image, eliminating the need to start from pure noise.

### Loss & Training
- The noise prediction model $\varepsilon_\theta$ and the domain scheduling network $\mathcal{S}_\theta$ are trained jointly.
- UNet backbone with PyTorch Lightning mixed-precision training.
- Moderate training steps per task: Sentinel 20K, IXI 10K, PSCDE 5K.

## Key Experimental Results

### Main Results
Comparison against Pix2Pix, BBDM, ABridge, DBIM, and DOSSR on three cross-modality translation tasks:

| Task | Metric | CDTSDE | DOSSR (2nd best) | Pix2Pix |
|------|--------|--------|-----------------|---------|
| Sentinel (SAR→Optical) | SSIM↑ | **0.382** | 0.360 | 0.230 |
| Sentinel | PSNR↑ (dB) | **17.46** | 17.14 | 15.12 |
| IXI (T2→T1) | SSIM↑ | **0.825** | 0.800 | 0.710 |
| IXI (T2→T1) | PSNR↑ (dB) | **24.33** | 24.13 | 22.24 |
| PSCDE (defect semantics) | Dice↑ | **0.488** | 0.460 | 0.178 |
| PSCDE | Hausdorff↓ | **39.87** | 59.53 | 156.28 |

CDTSDE ranks first on nearly all metrics. In terms of efficiency, it achieves 15 dB PSNR in only 5 sampling steps (1.8 s/image), approximately 2× faster than DOSSR (10 steps, 3.6 s).

### Ablation Study

| Schedule Type | Dice (PSCDE) | Hausdorff↓ | Description |
|--------------|-------------|-----------|-------------|
| Linear (global) | 0.46 | 59.5 | Fixed $\eta_t \cdot \mathbf{1}$ |
| Channel Non-linear | 0.46 | 43.0 | Per-channel nonlinear, spatially uniform |
| Dynamic (full) | **0.49** | **39.8** | Spatial + channel adaptive |

### Key Findings
- From Linear to Dynamic, Dice improves by 6.1% and Hausdorff decreases by 33%, demonstrating the core value of spatially adaptive domain scheduling.
- Channel Non-linear already substantially improves boundary quality (Hausdorff: 59.5→43.0), but does not improve region overlap; the spatial dimension of adaptivity provides the additional overlap gain.
- Bridge-based methods (BBDM, ABridge, DBIM) almost completely fail on the highly heterogeneous PSCDE task (Dice < 0.17), whereas CDTSDE and DOSSR perform far better due to their explicit domain shift designs.

## Highlights & Insights
- **Theory-driven design**: Theorem 1 rigorously proves from a path energy functional perspective that pixel-wise scheduling is superior to global scheduling. This theoretical result not only underpins the method but carries broader implications — spatially adaptive scheduling may benefit any generative task that requires learning transition paths between two distributions.
- **Exact solution → efficient sampling**: The coordinate transform yields an exact solution to the reverse SDE, enabling high-quality translation in 5 steps — a exemplary case of theory guiding practice.
- Embedding the **domain-shift force into the drift term** reduces the denoising model's role from "global alignment" to "local residual correction," substantially lowering the learning difficulty.

## Limitations & Future Work
- Improvements are limited in low-domain-gap scenarios (e.g., IXI: SSIM 0.80→0.82), suggesting that adaptive scheduling offers diminishing returns when modality differences are small.
- Training and evaluation are conducted solely on paired data; unpaired cross-modality translation is not explored.
- GAN-based methods may achieve better perceptual quality (sharpness); incorporating a lightweight perceptual or adversarial loss into CDTSDE could be beneficial.
- The impact of the capacity and architectural choices of the domain scheduling network $\mathcal{S}_\theta$ on performance is not thoroughly investigated.
- Experiments are limited to 256×256 resolution; computational cost and memory requirements at higher resolutions remain to be assessed.

## Related Work & Insights
- **vs. DOSSR**: Both are explicit domain-shift diffusion methods, but DOSSR uses a fixed linear schedule while CDTSDE uses a learnable spatial-adaptive schedule; the latter achieves 3 points higher Dice on PSCDE.
- **vs. BBDM/Bridge methods**: Bridge methods construct Brownian bridges between paired data but lack modeling of domain heterogeneity, leading to severe degradation on complex translation tasks.
- **vs. SDEdit**: SDEdit controls translation via a fixed noise level without an explicit domain shift mechanism, resulting in serious semantic drift in complex cross-modality scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Embedding domain shift physics into the SDE drift term and theoretically proving the superiority of spatial scheduling represent innovations in both theory and methodology.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three tasks of varying difficulty, ablation studies, and efficiency analysis are comprehensive, though dataset scales are relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are rigorous; the manifold path visualization in Fig. 1 is intuitive; overall logic is clear.
- **Value**: ⭐⭐⭐⭐ — Offers practical applicability in medical imaging and remote sensing; the adaptive scheduling idea is transferable to other conditional generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)
- [\[ICLR 2026\] Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity](improving_2d_diffusion_models_for_3d_medical_imaging_with_inter-slice_consistent.md)
- [\[ICCV 2025\] SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications](../../ICCV2025/medical_imaging/scivid_cross-domain_evaluation_of_video_models_in_scientific_applications.md)
- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](../../CVPR2026/medical_imaging/must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](../../CVPR2026/medical_imaging/interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)

</div>

<!-- RELATED:END -->
