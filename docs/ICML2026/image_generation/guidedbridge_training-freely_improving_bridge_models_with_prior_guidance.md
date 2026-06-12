---
title: >-
  [Paper Note] GuidedBridge: Training-freely Improving Bridge Models with Prior Guidance
description: >-
  [ICML 2026][Image Generation][bridge model] Addressing diffusion bridge models (data-to-data generation), this paper proposes a training-free Prior Guidance (PG). By perturbing a clean prior to construct a "weak prior" a…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "bridge model"
  - "prior guidance"
  - "frequency modulation"
  - "image translation"
  - "training-free"
date: 2026-05-08
content_hash: db93d21264428e2f
---

# GuidedBridge: Training-freely Improving Bridge Models with Prior Guidance

**Conference**: ICML 2026  
**arXiv**: [2606.03119](https://arxiv.org/abs/2606.03119)  
**Code**: TBD  
**Area**: Image Generation / Diffusion Bridge / Guidance  
**Keywords**: bridge model, prior guidance, frequency modulation, image translation, training-free

## TL;DR
Addressing diffusion bridge models (data-to-data generation), this paper proposes a training-free Prior Guidance (PG). By perturbing a clean prior to construct a "weak prior" and extrapolating between the denoising results of strong and weak priors, the model's utilization of the prior is amplified. Further utilizing U-shaped Frequency Modulation (FMPG) and a cascaded CFG-FMPG framework, the method stably improves the FID of pre-trained bridge models like DDBM/DBIM across tasks such as Edges→Handbags, DIODE, and ImageNet inpainting without additional training or increased NFE.

## Background & Motivation

**Background**: Guidance for Diffusion has matured into two major paradigms: CFG, which uses "conditional vs. unconditional" denoising results for extrapolation to enhance conditional alignment, and AG, which uses "well-trained vs. under-trained" denoising results for extrapolation to improve score accuracy. Both essentially "create a quality gap between two denoising steps and extrapolate towards the higher quality direction." Meanwhile, bridge models (DDBM, DBIM, I2SB, etc.) transform the generation process into data-to-data by conditioning on a clean prior $\bm{x}_T$, proving more efficient than diffusion from pure Gaussian noise for tasks with strong priors like image-to-image translation and restoration.

**Limitations of Prior Work**: While CFG/AG can be directly ported to bridge models, they fail to utilize the fundamental difference between bridges and diffusion—**prior exploitation** (the utilization of the clean prior provided by $\bm{x}_T$). In other words, ported guidance only "strengthens conditions" or "corrects score errors" without specifically reinforcing whether the model truly fully utilizes the prior. Furthermore, AG requires training an additional under-trained network, which is costly.

**Key Challenge**: The greatest advantage of bridge models over diffusion lacks a corresponding guidance design. Moreover, the SNR of bridge models follows a U-shaped trajectory (clean at both ends, noisiest in the middle), which differs significantly from the monotonically increasing SNR of diffusion. Consequently, applying a constant guidance scale across all timesteps and frequency bands ignores the bridge's inherent geometric structure.

**Goal**: (1) Design the first training-free guidance specifically for bridge models, mapping the guidance signal directly to "prior exploitation"; (2) Match the guidance intensity to the U-shaped SNR and frequency behavior of the bridge; (3) Ensure the method remains applicable even when the prior itself is weak (e.g., masked regions in inpainting).

**Key Insight**: Since both CFG and AG rely on "creating a lower quality denoising result and extrapolating," the most natural way to "degrade quality" for a bridge model is to **corrupt the prior available to the model**. Since the bridge has not seen perturbed priors, it will produce worse results. This "inferiority" corresponds exactly to "under-utilization of the prior," making the natural extrapolation direction "better utilization of the prior."

**Core Idea**: Training-freely apply a degradation operator $\mathcal{H}$ (such as Gaussian noise, blur, or JPEG compression) to $\bm{x}_T$ and $\bm{x}_t$ at each step. Use the difference between $D_{\bm{\theta}}(\bm{x}_t,t,\bm{x}_T)$ and $D_{\bm{\theta}}(\mathcal{H}(\bm{x}_t),t,\bm{x}_T)$ for guidance extrapolation, and schedule the guidance scale on high/low frequency bands according to the U-shaped SNR.

## Method

### Overall Architecture
The input consists of a **pre-trained bridge model** (DDBM / DBIM) and a clean prior $\bm{x}_T$ (e.g., an image to be translated or restored). During generation, instead of calling a single $D_{\bm{\theta}}(\bm{x}_t,t,\bm{x}_T)$ to predict $\bm{x}_0$, the following occurs at each sampling step:

1.  A degradation operator $\mathcal{H}$ corrupts the current noisy latent $\bm{x}_t$ into a "weak prior" $\mathcal{H}(\bm{x}_t)$ ($\bm{x}_T$ is always kept as a clean conditional signal and remains undecayed);
2.  Both $D_{\bm{\theta}}(\bm{x}_t,t,\bm{x}_T)$ and $D_{\bm{\theta}}(\mathcal{H}(\bm{x}_t),t,\bm{x}_T)$ are called to obtain strong and weak denoising results;
3.  The original denoising output is replaced by the extrapolation $D_{\text{PG}} = D_{\text{weak}} + w_{\text{PG}}(D_{\text{strong}} - D_{\text{weak}})$;
4.  FMPG further splits this extrapolation into low/high frequency bands with corresponding weights $w^{\text{LF}}_{\text{PG}}$ and $w^{\text{HF}}_{\text{PG}}$;
5.  For weak-prior tasks like inpainting, CFG is used for $t\in[T,t_s)$ to generate coarse structures, then switched to FMPG for $t\in[t_s,0)$ for detail refinement (CFG-FMPG cascade).

The entire process **requires no training and adds no parameters**, merely replacing denoising calls during sampling. The NFE is strictly aligned with the baseline (compensating for the extra forward pass per step by using fewer sampling steps).

### Key Designs

1.  **Prior Guidance (PG)—Creating Quality Gaps with Degraded Priors**:
    *   Function: Transfers the "gap-creation $\rightarrow$ extrapolation" paradigm of CFG/AG to bridge models, using "degraded prior results" as the source of the gap.
    *   Mechanism: A training-free degradation operator $\mathcal{H}$ (defaulting to additive Gaussian noise $\bm{\epsilon}\sim\mathcal{N}(\bm{0},\bm{I})$) is applied to the clean prior and each intermediate latent $\bm{x}_t$ to obtain $\mathcal{H}(\bm{x}_t)$. Since the bridge **never encountered** such degraded priors during pre-training, the result $D_{\bm{\theta}}(\mathcal{H}(\bm{x}_t),t,\bm{x}_T)$ is inevitably worse than $D_{\bm{\theta}}(\bm{x}_t,t,\bm{x}_T)$. The extrapolation $D_{\text{PG}} = D_{\bm{\theta}}(\mathcal{H}(\bm{x}_t),t,\bm{x}_T) + w_{\text{PG}}\,(D_{\bm{\theta}}(\bm{x}_t,t,\bm{x}_T) - D_{\bm{\theta}}(\mathcal{H}(\bm{x}_t),t,\bm{x}_T))$ naturally points towards "fuller prior utilization." The conditional signal $\bm{x}_T$ remains clean in the weak term to ensure the difference **stems only from prior degradation**.
    *   Design Motivation: Compared to AG, which requires an under-trained network, PG is entirely training-free. Compared to CFG’s "conditional alignment," PG directly addresses the bridge model's core advantage—prior exploitation. Robustness is observed across noise, blur, and JPEG compression.

2.  **Frequency-modulated PG (FMPG)—Band-split Scheduling per U-shaped SNR**:
    *   Function: Allows the guidance scale to be "targeted" along the sampling trajectory and frequency dimensions, rather than using a constant $w_{\text{PG}}$.
    *   Mechanism: By tracing the frequency-domain energy distribution of $\Delta\bm{x}_t \to \Delta\bm{x}_0$, the authors found that because bridge SNR is U-shaped, **HF (high frequency) priors are heavily submerged in noise** at intermediate timesteps—forcing guidance here amplifies noise. Conversely, **LF (low frequency) priors remain readable** throughout the trajectory—making intermediate steps ideal for LF enhancement. Thus, PG extrapolation is split: LF is extracted via a low-pass filter $I^{\text{LF}}$ with an **inverted U-shaped** $w^{\text{LF}}_{\text{PG}}$ (amplified in the middle); HF is extracted via $I^{\text{HF}}$ with a **U-shaped** $w^{\text{HF}}_{\text{PG}}$ (suppressed in the middle), formulated as $D^{\text{LF}}_{\text{FMPG}} = I^{\text{LF}}[D_{\text{weak}}] + w^{\text{LF}}_{\text{PG}}\,(I^{\text{LF}}[D_{\text{strong}}] - I^{\text{LF}}[D_{\text{weak}}])$.
    *   Design Motivation: The U-shaped SNR is a geometric feature of the data-to-data paradigm, unique to bridges. FMPG encodes this geometry into the guidance schedule, aligning guidance with bridge generation dynamics. The U/inverted-U shapes are **empirical coarse schedules**, making FMPG a lightweight training-free plug-in.

3.  **CFG-FMPG Cascade—Rescuing Weak Prior Tasks**:
    *   Function: When the prior itself is difficult to use (e.g., a 128x128 center mask in inpainting where no pixel information exists for PG degradation), CFG is utilized to draw the coarse structure before FMPG takes over for detail refinement.
    *   Mechanism: The sampling trajectory is divided: early $t\in[T,t_s)$ uses CFG on class labels $\bm{l}$ for conditional alignment to fill the mask with semantic structure; then $t\in[t_s,0)$ switches to FMPG, exploiting the CFG result as a "sufficiently strong prior." Both stages share the same bridge network and trajectory, **keeping NFE unchanged**.
    *   Design Motivation: PG alone fails on inpainting (no difference to create between strong/weak priors), while CFG alone only strengthens semantics without high-frequency detail. CFG and FMPG are complementary in "semantic alignment" and "prior exploitation."

### Loss & Training
Entirely training-free. All bridge checkpoints are used as-is (DDBM with the recommended hybrid SDE-ODE sampler, DBIM in pure ODE mode with $\eta=0.0$). Hyperparameter tuning involves two steps: selecting a degradation operator $\mathcal{H}$ from {noise, blur, JPEG, pooling}, then adjusting $w_{\text{PG}}$ (including FMPG band components).

## Key Experimental Results

### Main Results

NFE is strictly aligned with the baseline (fewer steps to compensate for the extra forward pass).

| Dataset | Metric | DBIM (Baseline) | DBIM+FMPG (Ours) | NFE |
| :--- | :--- | :--- | :--- | :--- |
| Edges→Handbags 64×64 | FID ↓ | 1.74 | **1.07** | 20 |
| Edges→Handbags 64×64 | FID ↓ | 0.91 | **0.78** | 100 |
| DIODE-Outdoor 256×256 | FID ↓ | 4.99 | **3.20** | 20 |
| DIODE-Outdoor 256×256 | FID ↓ | 2.58 | **2.06** | 100 |
| DIODE | LPIPS ↓ | 0.201 | **0.199** | 20 |
| Edges→Handbags | MSE ↓ | 0.005 | 0.005 | 20 |

Also effective on DDBM: DDBM+PG reduces FID on Edges→Handbags at NFE=150 / 300 from 1.30 / 0.65 to 1.23 / 0.59.

### Ablation Study

Comparison of PG variants and FMPG on DIODE (baseline is DBIM):

| Configuration | NFE=10 FID | NFE=20 FID | NFE=40 FID | Notes |
| :--- | :--- | :--- | :--- | :--- |
| DBIM (No guidance) | 7.99 | 4.99 | 3.35 | Original baseline |
| ECSI (Parallel work) | 6.83 | 4.12 | - | Fast sampler only |
| DBIM+PG (Blur) | 7.33 | 3.89 | 2.64 | Degradation via blur |
| DBIM+PG (Noise) | 6.25 | 3.77 | 2.96 | Degradation via noise |
| **DBIM+FMPG (Noise)** | **5.28** | **3.20** | **2.62** | Best after freq modulation |

### Key Findings
*   **FMPG > PG > Baseline**: Splitting constant $w_{\text{PG}}$ into U-shaped + inverted U-shaped band schedules further reduces FID across all NFEs, proving that the bridge's U-shaped SNR and frequency behavior benefit from a specialized schedule.
*   **Greater Gains at Lower NFE**: On DIODE at NFE=10, FID drops from 7.99 $\rightarrow$ 5.28 (-34%); at NFE=100 from 2.58 $\rightarrow$ 2.06 (-20%). FMPG is particularly valuable for fast sampling by "squeezing" out prior information unused in few-step sampling.
*   **CFG-FMPG is the key to Inpainting**: In center-mask inpainting, PG alone creates no difference; however, the CFG-FMPG cascade outperforms both pure CFG and simple hybrids. Qualitative results show simultaneous recovery of semantic layout and high-frequency textures.

## Highlights & Insights
*   **The Third Entry in the "Gap-Creation" Paradigm**: CFG creates a "conditional vs. unconditional" gap; AG creates a "well-trained vs. under-trained" gap; PG creates a "clean vs. degraded prior" gap. These are orthogonal, implying multiple guidance types can be stacked; CFG-FMPG is a miniature demonstration.
*   **Training-free yet Physically Interpretable**: The U-shaped HF schedule is not just a trick; it stems from the visualization of $\Delta\bm{x}_t \to \Delta\bm{x}_0$ energy transfer—high frequencies cannot pass through intermediate steps, so guidance should retreat. This "diagnose then schedule" design pattern can be applied to audio/video bridges.
*   **Honest NFE-aligned Comparison**: Matching total NFE by using fewer steps with extra forward passes avoids "trading 2× compute for performance," ensuring PG/FMPG truly qualifies as a "free lunch."

## Limitations & Future Work
*   **Degradation Operator Requires per-task Search**: The optimal choice between noise, blur, or JPEG depends on the dataset; a theoretical understanding of why certain degradations favor specific tasks is missing.
*   **Manual Frequency Schedule**: While the U-shape is derived from SNR observations, the exact peak positions and magnitudes still require tuning. An adaptive schedule might be superior.
*   **Domain Scope**: Only image translation and restoration were verified. Whether FMPG transfers to audio super-resolution or video bridges remains unexamined.

## Related Work & Insights
*   **vs. CFG (Ho & Salimans, 2021)**: CFG enhances conditional alignment based on condition presence; PG enhances prior exploitation based on prior integrity. They are orthogonal on bridges.
*   **vs. AG (Karras et al., 2024)**: AG requires an under-trained network; PG uses training-free degradation during inference, resulting in lower deployment costs and more direct alignment with bridge-specific prior exploitation.
*   **vs. DDBM / DBIM (Zhou et al., 2024; Zheng et al., 2025)**: These provide bridge training and fast samplers. Ours serves as a **universal enhancement plug-in** that leaves their parameters untouched.
*   **vs. ECSI (Zhang et al., 2025b)**: ECSI focuses on smarter sampling; PG/FMPG focuses on guidance. Theoretically, they can be combined to further reduce FID at low NFEs.

## Rating
*   Novelty: ⭐⭐⭐⭐ Porting the "gap-creation" paradigm to bridges and encoding U-shaped SNR/frequency behavior into guidance is original.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers two backbones and multiple tasks across various NFEs, though migration to audio/video is missing.
*   Writing Quality: ⭐⭐⭐⭐ Clear unified perspective on guidance types; effective frequency domain analysis and SNR visualizations.
*   Value: ⭐⭐⭐⭐ A truly training-free, NFE-neutral plug-in that provides direct value to image translation/restoration tasks using bridge models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models](../../ICLR2026/image_generation/stochastic_self-guidance_for_training-free_enhancement_of_diffusion_models.md)
- [\[ICML 2026\] Spectral Guidance for Flexible and Efficient Control of Diffusion Models](spectral_guidance_for_flexible_and_efficient_control_of_diffusion_models.md)
- [\[ICML 2026\] Bayesian Tensor Decomposition with Diffusion Model Prior](bayesian_tensor_decomposition_with_diffusion_model_prior.md)
- [\[ICML 2026\] Factored Classifier-Free Guidance](factored_classifier-free_guidance.md)
- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)

</div>

<!-- RELATED:END -->
