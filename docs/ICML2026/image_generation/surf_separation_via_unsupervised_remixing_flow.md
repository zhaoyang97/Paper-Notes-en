---
title: >-
  [Paper Note] SURF: Separation via Unsupervised Remixing Flow
description: >-
  [ICML 2026][Image Generation][Single-channel source separation] SURF combines the supervised flow matching framework FLOSS with unsupervised teacher-student remixing training (ReMixIT / Self-Remixing). This allows a gene…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Single-channel source separation"
  - "Flow Matching"
  - "Unsupervised learning"
  - "Teacher-Student distillation"
  - "Wake-Sleep"
date: 2026-05-08
content_hash: 6776f142140d02e5
---

# SURF: Separation via Unsupervised Remixing Flow

**Conference**: ICML 2026  
**arXiv**: [2606.04921](https://arxiv.org/abs/2606.04921)  
**Code**: TBD  
**Area**: Audio & Speech / Source Separation / Generative Models  
**Keywords**: Single-channel source separation, Flow Matching, Unsupervised learning, Teacher-Student distillation, Wake-Sleep  

## TL;DR
SURF combines the supervised flow matching framework FLOSS with unsupervised teacher-student remixing training (ReMixIT / Self-Remixing). This allows a generative flow matching separator to be trained **entirely from mixture observations** (without any clean source samples). It nearly matches supervised flow performance on MNIST/CIFAR10 image separation and LibriSpeech/FUSS audio separation, establishing a new unsupervised SOTA.

## Background & Motivation

**Background**: Single-channel source separation (recovering $K$ underlying sources from a single mixture) is a highly ill-posed inverse problem. In the deep learning era, two main paradigms exist: discriminative regression (e.g., Conv-TasNet, TF-Locoformer), which directly maps mixtures to sources; and generative models (e.g., diffusion, flow matching), which treat separation as conditional generation constrained by a strong prior learned from clean sources. Recently, FLOSS (Scheibler et al., 2025) achieved SOTA results using flow matching.

**Limitations of Prior Work**: All generative methods assume **access to clean source samples** to train the prior. however, in scenarios like bioacoustics, hyperspectral imaging, or gravitational wave detection, capturing an isolated source is often impossible. Even if samples are available, the train-test domain shift remains a persistent issue. Unsupervised methods like MixIT, ReMixIT, and Self-Remixing have successfully trained discriminative separators using self-supervision (teacher estimates $\rightarrow$ remix into new mixture $\rightarrow$ student learns back), but these methods only apply to regression-based separators (which directly output $\hat{x}$). **Mapping "velocity fields" in flow matching to these unsupervised frameworks has remained unexplored.**

**Key Challenge**: The training objective of flow matching is to regress the velocity $v_\theta(x_t, t, m)$, whereas ReMixIT-style self-supervision objectives target direct source regression $\hat{x}$. These different output semantics make simple integration difficult due to conflicts in PIT (Permutation Invariant Training) alignment and mixture consistency injection.

**Goal**: To train a generative flow separator like FLOSS **entirely from mixture observations** without significant performance degradation.

**Key Insight**: The authors identified a simple identity: given a flow matching path $x_t = (1-t)x_0 + t x_1$ and $\boldsymbol{u}(x,t)=\mathbb{E}[x_1-x_0|x_t,m]$, it follows that $\mathbb{E}[x_1|x_t, m] \approx x_t + (1-t)v_\theta(x_t, t, m)$. This conceptually links the velocity field to the "clean source estimate" required by ReMixIT, enabling the grafting of ReMixIT / Self-Remixing losses onto flow matching.

**Core Idea**: Use an EMA (Exponential Moving Average) teacher flow model to generate pseudo-sources $\rightarrow$ remix them across batches to form new mixtures $\rightarrow$ train the student flow using FLOSS-style "PIT to select optimal permutation + regress velocity on the chosen path" $\rightarrow$ update the teacher via EMA. ReMixIT targets teacher pseudo-sources, while Self-Remixing targets the original mixture itself.

## Method

### Overall Architecture
SURF utilizes two flow matching networks with identical structures: $v_{\theta_\mathcal{T}}$ (teacher) and $v_\theta$ (student), both trained via FLOSS principles. Both are initialized from a MixIT-pretrained regression separator. In each iteration:

1. **Teacher Inference**: Starting from a batch of real mixtures $\boldsymbol{M}=[\boldsymbol{m}_1,\dots,\boldsymbol{m}_B]$, the teacher samples pseudo-sources $\mathcal{X} \in \mathbb{R}^{BK\times d}$ via the flow ODE.
2. **Remixing**: Sample a permutation $\boldsymbol{\Pi}$ from $S_{BK}$ to shuffle all $BK$ rows, obtaining $\tilde{\boldsymbol{X}}_1 = \boldsymbol{\Pi}\mathcal{X}$, then sum them to form new mixtures $\tilde{\boldsymbol{M}}=(\boldsymbol{I}_B\otimes \mathbf{1}^\top)\tilde{\boldsymbol{X}}_1$.
3. **FM Path Construction**: Define the noise end as $\tilde{\boldsymbol{X}}_0 = \tfrac{1}{K}(\boldsymbol{I}_B\otimes \mathbf{1})\tilde{\boldsymbol{M}} + (\boldsymbol{I}_B\otimes \boldsymbol{P}^\perp)\boldsymbol{Z}$ (to ensure mixture consistency). Use the student's $t=0$ velocity to perform PIT and find the optimal permutation $\boldsymbol{\Upsilon}$, resulting in the interpolation $\tilde{\boldsymbol{X}}_t^{\boldsymbol{\Upsilon}}=(1-t)\tilde{\boldsymbol{X}}_0 + t\boldsymbol{\Upsilon}\tilde{\boldsymbol{X}}_1$.
4. **Loss Calculation**: Define the residual $\boldsymbol{R}_t = v_\theta(\tilde{\boldsymbol{X}}_t^{\boldsymbol{\Upsilon}}, t, \tilde{\boldsymbol{M}}) - (\boldsymbol{\Upsilon}\tilde{\boldsymbol{X}}_1 - \tilde{\boldsymbol{X}}_0)$ and supervise via two possible paths.
5. **EMA Teacher Update**: $\theta_\mathcal{T} \leftarrow \alpha \theta_\mathcal{T} + (1-\alpha)\theta$.

The entire process **requires zero clean sources**—all targets are derived either from the teacher (ReMixIT variant) or the original observed mixtures (Self-Remixing variant).

### Key Designs

1. **Velocity-to-Denoiser Bridge (Enabling FM for ReMixIT)**:
    - **Function**: Mathematically connects the velocity $v_\theta$ learned by flow matching with the "clean source estimate" required by ReMixIT/Self-Remixing.
    - **Mechanism**: Based on $\boldsymbol{u}(x,t)=\mathbb{E}[x_1-x_0|x_t,m]$ and the path relation $x_1-x_0=(x_1-x_t)/(1-t)$, we obtain $\mathbb{E}[x_1|x_t,m]=x_t+(1-t)\boldsymbol{u}(x_t,t,m)\approx x_t+(1-t)v_\theta(x_t,t,m)$. This allows the use of this quantity at any step $t$ as a "time-dependent denoised estimate," which is directly compatible with regression losses like Self-Remixing.
    - **Design Motivation**: FM and regression self-supervision have different semantics. This identity allows the model to provide both "velocity field" and "denoised source" perspectives simultaneously, allowing ReMixIT logic to be applied without structural changes.

2. **ReMixIT-FM vs. Self-Remixing-FM Double Losses (Supervision vs. Reflection)**:
    - **Function**: Signals training for the student flow on resynthesized data.
    - **Mechanism**: ReMixIT-FM uses a FLOSS-style PIT-FM loss $\mathcal{L}_{\text{RM-FM}}=\mathbb{E}\|\boldsymbol{R}_t\|^2$, treating teacher pseudo-sources as ground truth. Conversely, Self-Remixing-FM requires only that the student's estimates sum back to the original mixture: $\mathcal{L}_{\text{SR-FM}}=\mathbb{E}\|(\boldsymbol{I}_B\otimes\mathbf{1}^\top)\boldsymbol{\Pi}^{-1}\boldsymbol{\Upsilon}^{-1}\boldsymbol{R}_t\|^2$. Both losses share the same $\boldsymbol{R}_t$ but apply different projections.
    - **Design Motivation**: ReMixIT provides denser signals but may "inherit" teacher errors. Self-Remixing avoids penalizing pseudo-source errors directly by ensuring mixture reconstruction consistency. The authors prove in the Appendix that the gradient relates to the system error correlation; when errors across sources are weakly correlated, the gradient reverts to pseudo-supervised FM.

3. **Wake-Sleep Interpretation + EMA Teacher (Closing the Self-Distillation Theory)**:
    - **Function**: Explains the convergence of the "teacher generate $\rightarrow$ remix $\rightarrow$ student learn $\rightarrow$ teacher EMA" cycle and guides the teacher update rule.
    - **Mechanism**: The marginal defined by the teacher, $\bar{p}_{\theta_\mathcal{T}}(\bar{\boldsymbol{x}})=\prod_k \bar{p}_{\theta_\mathcal{T}}(\bar{\boldsymbol{x}}^{(k)})$, is treated as an implicit prior, while the student $p_\theta(\bar{\boldsymbol{x}}|m)$ acts as an inference network. The **Sleep phase** (student update) is equivalent to maximum likelihood on synthesized pairs $(\bar{\boldsymbol{x}}, m)\sim \bar{p}_{\theta_\mathcal{T}}(\bar{\boldsymbol{x}})p(m|\bar{\boldsymbol{x}})$, which is exactly ReMixIT. The **Wake phase** (teacher update) ideally requires an unavailable aggregate posterior, thus it is approximated by moving $\theta_\mathcal{T}$ toward $\theta$ using EMA $\theta_\mathcal{T} \leftarrow \alpha\theta_\mathcal{T} + (1-\alpha)\theta$.
    - **Design Motivation**: Previous successes of ReMixIT were primarily empirical. Mapping it to the Wake-Sleep framework provides a probabilistic justification and shows that EMA is not just a stability trick, but a proxy for the Wake step.

### Loss & Training
The joint training objective is either $\mathcal{L}_{\text{RM-FM}}$ or $\mathcal{L}_{\text{SR-FM}}$. The teacher EMA decay $\alpha$ is a critical hyperparameter. The student network follows the FLOSS architecture (permutation-equivariant layers + mixture-consistency projection). Initialization uses a MixIT-pretrained regression separator as an unsupervised starting point.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (SURF-RM) | Prev. SOTA (Unsup.) | Supervised Flow Upper Bound |
|--------|------|------|----------|------|
| MNIST 2-source | PSNR ↑ | **37.26** | 23.13 (Self-Remixing) | 37.44 |
| MNIST 2-source | FID ↓ | **19.57** | 28.14 | 19.47 |
| CIFAR10 2-source | PSNR ↑ | **19.73** | 17.51 | 20.38 |
| CIFAR10 2-source | FID ↓ | **14.83** | 28.44 | 9.60 |
| LibriSpeech+FUSS (2 src) | SI-SDR ↑ | 14.98 / **15.23** (SR) | 14.81 (Self-Remixing) | 18.21 |
| FUSS 1-src | SI-SDR ↑ | **32.67** | 19.83 (ReMixIT) | 38.79 |

> Key takeaway: On image separation, SURF raises PSNR from 23 to 37, **matching supervised flow**. On CIFAR10, the FID drops from 28 to 14, even lower than BASIS (diffusion prior requiring clean data).

### Ablation Study

| Configuration | MNIST PSNR | Description |
|------|---------|------|
| MixIT (Initialization) | 21.90 | Regression baseline |
| Regression-ReMixIT | 22.81 | Standard unsupervised regression |
| SURF (ReMixIT-FM) | **37.26** | Flow + ReMixIT |
| SURF (Self-Remixing-FM) | 37.03 | Flow + Self-Remixing |
| Supervised Flow | 37.44 | Upper bound (requires clean data) |

### Key Findings
- "Flow + Self-supervision" significantly outperforms "Regression + Self-supervision" (37 vs 23 PSNR), demonstrating that generative priors are essential for removing regression artifacts.
- ReMixIT-FM and Self-Remixing-FM perform similarly, though Self-Remixing slightly leads on LibriSpeech audio (SI-SDR 15.23 vs 14.98), aligning with theories regarding decoupled teacher errors.
- SURF's CIFAR10 FID is actually **lower** than supervised regression (14.83 vs 25.44), confirming the advantage of generative priors in distribution matching.
- On FUSS with more sources (3 or 4), a 2-3 dB gap remains compared to supervised flow, suggesting that PIT alignment and EMA stability are future areas for improvement as $K$ increases.

## Highlights & Insights
- **The velocity-denoiser identity is the key plug**: The one-line derivation $\mathbb{E}[x_1|x_t, m]\approx x_t+(1-t)v_\theta$ allows any self-supervised loss relying on clean source estimates to be applied to any flow or diffusion separator. This is a general bridge transferable to a wide range of inverse problems.
- **Wake-Sleep Perspective**: Elevating ReMixIT from an engineering trick to a standard latent generative model paradigm provides methodological significance, framing EMA as a proxy for the intractable Wake step.
- **Image Separation as a Controlled Experiment**: While source separation is traditional in audio, using MNIST/CIFAR10 provides quantifiable PSNR/FID/LPIPS metrics common in computer vision, offering better interpretability and reproducibility than SI-SDR alone.

## Limitations & Future Work
- Theoretical analysis relies on the population limit ($B\to \infty$) and simplified assumptions; bias under finite batches is not fully characterized. The EMA decay $\alpha$ still requires empirical tuning.
- Training depends on a MixIT-pretrained seed separator. It is unclear if convergence holds if the initial seed is extremely poor (cold start).
- In 3-4 source scenarios (FUSS), a 2-3 dB gap exists with supervised flow, indicating that PIT complexity and teacher error accumulation increase with $K$.
- Future work could extend this "bridge + Wake-Sleep EMA" to other inverse problems (denoising, deblurring, super-resolution), particularly in scientific imaging where clean ground truth is scarce.

## Related Work & Insights
- **vs. FLOSS (Scheibler et al., 2025)**: FLOSS is the supervised SOTA for flow separation but depends on clean sources. SURF is effectively an unsupervised version of FLOSS.
- **vs. ReMixIT / Self-Remixing (Tzinis et al., 2022; Saijo & Ogawa, 2023)**: Original ReMixIT is restricted to discriminative separators (e.g., Conv-TasNet). SURF brings this "remix $\rightarrow$ student learn" loop to generative flow matching.
- **vs. BASIS / Diffusion Prior Separation (Jayaram & Thickstun, 2020; Mariani et al., 2024)**: These methods use clean sources to train a prior first. SURF does not require clean source training, making it applicable to fields where isolated sources cannot be recorded.
- **vs. Rozet et al., 2024 (Unsupervised Diffusion Prior with EM)**: These works attempt to learn unconditional priors from mixtures but require computationally expensive conditional diffusion approximations. SURF avoids explicit prior modeling by bootstrapping the conditional separator directly.

## Rating
- Novelty: ⭐⭐⭐⭐ The velocity-denoiser bridge unlocks the FM + self-supervision pipeline, and the Wake-Sleep interpretation closes the theoretical gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across four benchmarks (MNIST, CIFAR10, FUSS, LibriSpeech) with comparisons to supervised bounds and multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ The algorithms are clearly explained, and the theoretical decomposition is insightful.
- Value: ⭐⭐⭐⭐⭐ Provides a viable path for training generative separators without clean sources, with high potential in bioacoustics and scientific imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Adversarial Flow Models](adversarial_flow_models.md)
- [\[CVPR 2026\] Cinematic Audio Source Separation Using Visual Cues](../../CVPR2026/image_generation/cinematic_audio_source_separation_using_visual_cues.md)
- [\[ICLR 2026\] From Parameters to Behaviors: Unsupervised Compression of the Policy Space](../../ICLR2026/image_generation/from_parameters_to_behaviors_unsupervised_compression_of_the_policy_space.md)
- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)
- [\[ICCV 2025\] Pretrained Reversible Generation as Unsupervised Visual Representation Learning](../../ICCV2025/image_generation/pretrained_reversible_generation_as_unsupervised_visual_representation_learning.md)

</div>

<!-- RELATED:END -->
