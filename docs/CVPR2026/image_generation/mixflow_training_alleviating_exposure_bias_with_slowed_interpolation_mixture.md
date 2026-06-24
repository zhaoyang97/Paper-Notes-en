---
title: >-
  [Paper Note] MixFlow Training: Alleviating Exposure Bias with Slowed Interpolation Mixture
description: >-
  [CVPR 2026][Image Generation][Diffusion Models] Addressing the exposure bias in diffusion and flow matching models—where training uses ground-truth interpolation but testing relies on self-generated noisy data—this work identifies a "Slow Flow" phenomenon. Specifically, the ground-truth interpolation closest to the noisy data generated at sampling timestep $t$ actually corresponds to a higher-noise (slower) timestep $m_t \le t$. Consequently…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Flow Matching"
  - "Exposure Bias"
  - "Training-Test Mismatch"
  - "Post-Training"
date: 2026-05-08
content_hash: a3cdfc17725a9b0b
---

# MixFlow Training: Alleviating Exposure Bias with Slowed Interpolation Mixture

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Li_MixFlow_Training_Alleviating_Exposure_Bias_with_Slowed_Interpolation_Mixture_CVPR_2026_paper.html)  
**Code**: Project Page https://mixflowgen.github.io/  
**Area**: Image Generation / Diffusion Models  
**Keywords**: Diffusion Models, Flow Matching, Exposure Bias, Training-Test Mismatch, Post-Training

## TL;DR
Addressing the exposure bias in diffusion and flow matching models—where training uses ground-truth interpolation but testing relies on self-generated noisy data—this work identifies a "Slow Flow" phenomenon. Specifically, the ground-truth interpolation closest to the noisy data generated at sampling timestep $t$ actually corresponds to a higher-noise (slower) timestep $m_t \le t$. Consequently, MixFlow is proposed: it replaces the training input interpolation with a mixture of interpolations within a slowed timestep interval. Performing post-training by modifying only 5 lines of code improves RAE to 1.43 FID (without guidance) and 1.10 FID (with guidance) on ImageNet.

## Background & Motivation
**Background**: Diffusion and flow matching models gradually transform noise into data by learning a predictive network (predicting noise/data/score/velocity). During training at timestep $t$, the input fed to the network is the ground-truth noisy data $x_t$, i.e., an interpolation of noise and data: $x_t = \alpha_t x_1 + \beta_t x_0$. During testing, however, the input is the noisy data **self-generated** in the previous step.

**Limitations of Prior Work**: These two inputs are inconsistent—which is the "exposure bias / training-test mismatch." Since the network never encounters its own (erroneous) self-generated intermediate states during training, it must make predictions on these states during testing. Consequently, errors accumulate along the sampling trajectory (error accumulation, sampling drift), ultimately degrading generation quality.

**Key Challenge**: Neither of the two existing approaches is ideal. Training-side solutions (e.g., Input Perturbation, which adds Gaussian perturbations to ground truth to simulate test errors) are highly sensitive to the perturbation intensity, performing worse if it is too large or too small. Self-Forcing directly uses generated data as training inputs, but its computational cost is prohibitive for standard multi-step diffusion, making it suitable only for few-step sampling. Sampling-side solutions (e.g., Epsilon Scaling for scaling predicted noise, Time-Shift Sampler for heuristically shifting sampling timesteps) are heuristic corrections applied during inference and lack calibration accuracy.

**Goal**: From the training perspective, find a ground-truth interpolation that truly resembles "what inputs the network actually encounters during testing" to perform training, keeping it simple with almost zero additional overhead.

**Key Insight**: The authors quantify "which ground-truth interpolated timestep is most similar to the noisy data generated during testing." Using SiT-B with 50-step sampling on ImageNet and collecting statistics from 20,000 images, they discover a stable phenomenon: the ground-truth interpolation nearest to the generated noisy data at sampling timestep $t$ falls at a **higher-noise** timestep $m_t \le t$ (termed the slowed timestep); moreover, the larger $t$ is, the wider the deviation range of $m_t$ from $t$ becomes. The intuition is that "generated data progresses slower than ground-truth data."

**Core Idea**: Since the generated data at test timestep $t$ actually resembles the ground-truth interpolation at a higher-noise timestep $m_t$, the training should not solely feed the interpolation at timestep $t$. Instead, it should feed the "mixture of interpolations within the slowed timestep interval $[(1-\gamma)t,\,t]$," exposing the network to the actual (slower) inputs it will encounter during testing at each training timestep.

## Method

### Overall Architecture
MixFlow does not modify the network architecture or the sampling process; it only alters **the input interpolation and the corresponding loss fed to the predictive network during training**, acting as a plug-and-play post-training scheme.

Standard training: At training timestep $t$, the ground-truth interpolation $x_t=\alpha_t x_1+\beta_t x_0$ is used as input to let the network $u_\theta(x_t,t)$ regress the ground-truth velocity $u^*(x_t,t)$ of that timestep.

MixFlow training: At training timestep $t$, the input is replaced with the **interpolation at the slowed timestep $m_t$** $x_{m_t}$ ($m_t \le t$, higher noise), but the **timestep label fed to the network is still $t$** (not $m_t$). The regression target becomes the ground-truth velocity of the slowed timestep $u^*(x_{m_t},m_t)$. The entire sampling process remains completely identical to standard training, and there is **no need to compute $m_t$** during inference.

In a single iteration (Algorithm 1), this merely inserts two sampling steps, modifies one interpolation, and adjusts the loss in the standard training loop: sample data $x_1$, sample noise $x_0$, sample training timestep $t\sim\mathrm{Beta}(2,1)$, sample slowed timestep $m_t\sim\mathcal{U}[(1-\gamma)t,\,t]$, compute $x_{m_t}=\beta_{m_t}x_0+\alpha_{m_t}x_1$, calculate the loss $\lVert u_\theta(x_{m_t},t)-u^*\rVert_2^2$, and backpropagate. In the official implementation, integrating with RAE requires modifying only 5 lines of code.

### Key Designs

**1. Slow Flow Phenomenon: Quantifying Exposure Bias as "Time Decelerating"**

Exposure bias was previously a qualitative concept (mismatched training/testing inputs). The paper quantifies it as an observable and exploitable pattern. By measuring "which timestep's ground-truth interpolation is nearest to the noisy data generated at sampling timestep $t$," the authors show that it systematically falls on the $m_t\le t$ (higher noise) side. This is shown in Figure 1, where the shaded region always lies below the diagonal $x=y$. Furthermore, the larger $t$ is, the wider the possible range of $m_t$. The physical meaning is: generated data "moves too slowly" along the trajectory, and when reaching $t$, it mathematically resembles the ground truth at an earlier (noisier) timestep. This phenomenon is the foundation of the method—it suggests that training inputs should be biased towards "higher noise," rather than adding undirected noise as in Input Perturbation.

**2. Slowed Interpolation Mixture: Replacing Single-point Interpolation with a High-noise Interval**

Standard training uses only a single interpolation $x_t$ as input at each timestep $t$, failing to cover the "slowed states" encountered during testing. MixFlow replaces the single-point input with an **interpolation mixture set**:

$$\mathcal{X}_t=\{x_{m_t}\mid x_{m_t}=\beta_{m_t}x_0+\alpha_{m_t}x_1,\ m_t\in\mathcal{M}_t\},$$

where the mixture interval is defined according to the Slow Flow phenomenon as:

$$\mathcal{M}_t=[(1-\gamma)t,\ t].$$

The interval length $\gamma t$ scales linearly with the training timestep—corresponding to the observation that "the larger $t$, the larger the slowed deviation." Thus, a wider mixture range is allocated closer to the data end. $\gamma$ controls the extent of extension towards higher noise, which can be selected empirically or simply set to 1. The key to this design is **extending only to the higher-noise side** ($m_t\le t$). Ablation studies show that including lower-noise timesteps $\mathcal{U}[0,1]$ degrades performance below the baseline, as lower-noise interpolations do not represent states encountered during testing and pollute the training.

**3. Slowed Interpolation Loss: Inputting Slowed Interpolation while Retaining the Original Timestep Label**

MixFlow's loss introduces an additional variable, the slowed timestep $m_t$, compared to the standard loss:

$$\mathbb{E}_{t,x_0,x_1,m_t}\big[\lVert u_\theta(x_{m_t},t)-u^*(x_{m_t},m_t)\rVert_2^2\big].$$

In contrast to the standard loss $\mathbb{E}_{t,x_0,x_1}[\lVert u_\theta(x_t,t)-u^*(x_t,t)\rVert_2^2]$, there are two differences: the input interpolation shifts from $x_t$ to the higher-noise $x_{m_t}$; the regression target shifts from $u^*(x_t,t)$ to $u^*(x_{m_t},m_t)$ (the ground-truth velocity of the slowed timestep). Crucially, the **timestep condition given to the network remains $t$**. Thus, the network learns "how to predict when told the current timestep is $t$ but the input actually resembles a slower state"—exactly aligning with the actual situation during testing.

**4. Beta(2,1) Sampling of Training Timestep: Uniformly Covering the (m_t, t) Pairs over the Joint Space**

The slowed timestep is sampled from the conditional distribution $m_t\sim\mathcal{U}[(1-\gamma)t,\,t]$, where the conditional density $p(m_t|t)=\frac{1}{\gamma t}$ varies with $t$ (narrower intervals yield higher density). If the training timestep $t$ is still sampled from the conventional $\mathcal{U}[0,1]$, the probability of sampling each $(m_t,t)$ pair is non-uniform, oversampling the small $t$ region. The authors require the input pairs $(x_{m_t},t)$ to be **uniformly sampled**, meaning $p(m_t,t)=p(m_{t'},t')$. From $p(m_t,t)=p(m_t|t)p(t)$, we get $p(t)\propto t$. Incorporating the normalization $\int_0^1 p(t)=1$ yields:

$$t\sim\mathrm{Beta}(2,1)\quad(p(t)=2t).$$

Ablation studies confirm: under the premise $m_t\sim\mathcal{U}[0,t]$, using Beta(2,1) for $t$ is significantly better than using $\mathcal{U}[0,1]$ (gFID 15.64 vs 16.57). This proves that uniformly covering the $(m_t,t)$ pairs indeed yields gains rather than being an optional engineering detail.

### Loss & Training
- The training objective is the slowed interpolation loss defined above, with velocity as the prediction target. Under the GVP diffusion setting, the authors also introduce an alternative ground-truth velocity target $u^*(x_t,t)$, which is found to accelerate convergence (achieving the effect of 2.5 million steps of the old target in just 500k steps).
- Key hyperparameter $\gamma$: Swept across $\{0.3,0.5,0.7,0.8,0.9,1.0\}$, values in $[0.7, 1.0]$ perform well and similarly, with $\gamma=0.8$ being slightly superior (default). $\gamma=0$ (reducing to standard training) performs slightly worse than the baseline. Over-slowing (e.g., $m_t\in[0,0.2t]$) is not beneficial, as standard models at $t$ are unlikely to decelerate that early.
- Post-training throughout: Models are warm-started from official pre-trained models (SiT post-trained for 500k steps, RAE post-trained for 200 epochs, SD3.5 post-trained for 10k steps), keeping training/evaluation hyperparameters identical to their respective original models.

## Key Experimental Results

### Main Results
ImageNet class-conditional generation (second-order Heun, 250 steps, guidance scale 1.5; RAE uses its own setting of 50-step auto-guidance):

| Model / Setting | gFID w/o Guidance | gFID w/ Guidance |
|------|------|------|
| SiT-B/2 (256) | 17.97 | 4.46 |
| SiT-B/2 + MixFlow | **15.64** | **3.91** |
| SiT-XL/2 (256) | 9.35 | 2.15 |
| SiT-XL/2 + MixFlow | **7.87** | **1.99** |
| SiT-XL/2 (512) | 9.72 | 2.71 |
| SiT-XL/2 + MixFlow (512) | **7.99** | **2.53** |
| REPA-XL/2 | 6.90 | 1.65 |
| REPA-XL/2 + MixFlow | **6.28** | **1.59** |
| RAE-XL | 1.51 | 1.13 |
| RAE-XL + MixFlow | **1.43** | **1.10** |

Comparison with methods dedicated to exposure bias (SiT-B base): MixFlow achieves the largest improvement. Input Perturbation requires precise tuning of perturbation scale, and Time-Shift / Epsilon Scaling are heuristic corrections during inference, which are inferior to direct training modification.

Gains are larger with fewer sampling steps (gFID w/o guidance, SiT-B/2):

| Steps | 250 | 50 | 20 |
|------|------|------|------|
| Gain | 2.33 | 2.34 | 2.58 |

The reason is that fewer steps lead to coarser sampler approximations, larger training-test gaps, and a more prominent Slow Flow issue.

### Ablation Study

| Configuration | gFID w/o Guidance / w/ Guidance | Description |
|------|------|------|
| $t\sim\mathcal{U}[0,1],\ m_t\sim\mathcal{U}[0,1]$ | 18.27 / 5.07 | Including lower-noise timesteps degrades performance beyond baseline |
| $t\sim\mathrm{Beta}(2,1),\ m_t\sim\mathcal{U}[0,1]$ | 18.25 / 5.06 | Same as above, lower-noise interpolation is harmful |
| $t\sim\mathcal{U}[0,1],\ m_t\sim\mathcal{U}[0,t]$ | 16.57 / 4.25 | Selecting only high-noise already outperforms baseline |
| $t\sim\mathrm{Beta}(2,1),\ m_t\sim\mathcal{U}[0,t]$ (Full) | **15.64 / 3.93** | Full scheme |
| Standard Post-training (same steps) | 17.96 / 4.46 | Virtually unchanged; gains do not stem from extra training |
| Baseline w/o Post-training | 17.97 / 4.46 | — |

### Key Findings
- **Direction is more important than intensity**: Mixing only towards the higher-noise side ($m_t\sim\mathcal{U}[0,t]$) is effective; including lower-noise timesteps degrades performance. This explains why Input Perturbation's "directionless noise addition" is extremely tedious to tune.
- **Gains stem from the training scheme rather than extra training steps**: Standard post-training with the same steps yields 17.96 ≈ 17.97 (without post-training), whereas MixFlow pushes it down to 15.64.
- **Fewer steps and larger $t$ bring larger gains**, which is consistent with the Slow Flow phenomenon (larger slowed deviation at larger $t$, and coarser approximations with fewer steps).
- Strong generalizability: Effective across linear interpolation flow matching (SiT), variance preserving diffusion (GVP), representation alignment training (REPA), representation autoencoder (RAE), and text-to-image models (SD3.5).

## Highlights & Insights
- **Quantifying the vague "exposure bias" as an exploitable directional phenomenon (Slow Flow)**: Instead of vaguely stating a training-test mismatch, the authors measure that "generated data corresponds to a higher-noise timestep $m_t\le t$ and the deviation increases with $t$." The method directly designs the mixture interval $[(1-\gamma)t,t]$ matching this pattern, resulting in a seamless fit between motivation and implementation.
- **Extremely low adaptation cost**: Modifies neither the architecture nor the sampling, introducing zero additional inference overhead. Integrating with RAE needs only 5 lines of code, yet achieves SOTA-level FID practically for free.
- **Rigorous derivation of the sampling distribution**: The $t\sim\mathrm{Beta}(2,1)$ sampling is analytically derived from the goal of "uniformly covering the $(m_t,t)$ pairs", rather than heuristic tuning. Ablation studies prove its actual contribution to gains. This idea of "back-solving the sampling distribution to ensure uniform statistics" is transferable to other joint timestep/noise training schemes.

## Limitations & Future Work
- **Reliance on a converged pre-trained model for post-training**. The Slow Flow phenomenon itself is observed and calculated from already-trained models. Whether the phenomenon holds and is equally effective during scratch training is not directly verified in the paper.
- The optimal value of $\gamma$ varies across models (0.8 for SiT, 0.4 for RAE), requiring minor tuning. The authors do not provide a method to automatically determine $\gamma$ based on model/data.
- Gains diminish under large models and many-step settings (e.g., SiT-XL/2 with guidance only yields 2.15→1.99, RAE only yields 1.51→1.43). The method is more suitable for few-step scenarios with larger training-test discrepancies.
- The "nearest neighbor corresponding to a higher-noise timestep" in slowed interpolation is a statistical observation. While the paper explains it via intuition and a toy example, a more rigorous theoretical analysis is lacking. The acceleration mechanism of replacing the velocity target under GVP is also speculative.

## Related Work & Insights
- **vs Input Perturbation**: Both modify training, but Input Perturbation adds undirected Gaussian noise to ground truth and is highly sensitive to intensity (worse performance if too large or too small). MixFlow leverages Slow Flow to provide a clear direction (mixing only towards higher noise). Ablations show including lower-noise is harmful, which explains why the former is hard to tune.
- **vs Self-Forcing**: Self-Forcing directly uses generated data as inputs, resulting in prohibitive computational costs for standard multi-step diffusion (suitable only for few-step/autoregressive video). MixFlow approximates generated states using ground-truth slowed interpolations, running with near-zero overhead and integrating seamlessly with standard multi-step training.
- **vs Epsilon Scaling / Time-Shift Sampler**: Both of these are heuristic corrections during inference (scaling noise / shifting sampling timesteps), which are inaccurate. MixFlow modifies training to align input distributions at the source, decouples from the sampling process, and outperforms both in practice.

## Rating
- Novelty: ⭐⭐⭐⭐ The quantitative observation of the Slow Flow phenomenon and the "slowed interpolation mixture" angle are clear and novel, though the essence is still a training-side improvement for the exposure bias family.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers SiT/GVP/REPA/RAE/SD3.5, multiple resolutions and step counts, cross-comparisons against three classes of exposure-bias mitigation methods, accompanied by exhaustive ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ The progression from phenomenon to method, derivation, and experiments is logically sound, though some formula formatting (in cache) was cluttered and required cross-referencing with the original.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, with almost zero cost to push existing strong models further (e.g., pushing RAE to 1.10 FID), offering high utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Linear Image Generation by Synthesizing Exposure Brackets](linear_image_generation_by_synthesizing_exposure_brackets.md)
- [\[CVPR 2026\] Elucidating the SNR-t Bias of Diffusion Probabilistic Models](dcw_snr_t_bias_diffusion.md)
- [\[CVPR 2026\] Mixture of Style Experts for Diverse Image Stylization](mixture_of_style_experts_for_diverse_image_stylization.md)
- [\[ICLR 2026\] MOLM: Mixture of LoRA Markers](../../ICLR2026/image_generation/molm_mixture_of_lora_markers.md)
- [\[CVPR 2026\] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](leapalign_post_training_flow_matching_models_at_any_generation_step.md)

</div>

<!-- RELATED:END -->
