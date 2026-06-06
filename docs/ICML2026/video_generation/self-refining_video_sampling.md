---
title: >-
  [Paper Note] Self-Refining Video Sampling
description: >-
  [ICML 2026][Video Generation][Video Diffusion] The pretrained flow matching video generator is reinterpreted as a "denoising autoencoder." During inference…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Video Diffusion"
  - "Flow Matching"
  - "Self-Refining Sampling"
  - "Denoising Autoencoder"
  - "Physical Consistency"
date: 2026-05-08
content_hash: f33e82846765f3d8
---

# Self-Refining Video Sampling

**Conference**: ICML 2026  
**arXiv**: [2601.18577](https://arxiv.org/abs/2601.18577)  
**Code**: https://agwmon.github.io/self-refine-video/ (Project Page)  
**Area**: Video Generation / Diffusion Models / Inference-time Sampling  
**Keywords**: Video Diffusion, Flow Matching, Self-Refining Sampling, Denoising Autoencoder, Physical Consistency

## TL;DR
The pretrained flow matching video generator is reinterpreted as a "denoising autoencoder." During inference, a "Predict-and-Perturb" inner loop is used within the same noise level to repeatedly correct latents. An uncertainty mask derived from model self-consistency is applied to refine only dynamic regions. This approach significantly improves motion coherence and physical plausibility without any external verifiers or additional training, achieving over 70% human preference.

## Background & Motivation

**Background**: Modern video generators such as Wan2.1/2.2 and Cosmos-Predict are typically based on flow matching, where a learned vector field pushes Gaussian noise toward the data distribution along an ODE in VAE latent space. Although regarded as early "world models," physical dynamics (e.g., multi-argument interactions, complex human movements, rigid-body free fall) remain a known weakness.

**Limitations of Prior Work**: There are two primary paths to improving physical realism, both of which are computationally heavy. The first is **external verifier + rejection sampling** (e.g., Cosmos-Reason1, Bansal et al.), which generates multiple candidates and selects the one with the highest score. This suffers from low acceptance rates, high costs, and domain-specific verifiers with limited temporal/physical assessment capabilities. The second is **additional training/post-training** (e.g., WISA, VideoJAM, CGI synthetic data fine-tuning), which requires high-quality physical annotations and massive training costs, while reward models struggle to capture fine-grained motion.

**Key Challenge**: Large-scale video generators already encode priors of "realistic motion + structure" in their weights. However, standard ODE solvers follow a one-way trajectory—once coarse motion is determined in the early steps, no mechanism exists to revisit and correct it. While LLMs can critique-and-revise their output tokens, video generators lack such explicit feedback signals, particularly for high-dimensional, temporally coupled latents.

**Goal**: To find a (i) training-free, (ii) self-contained, (iii) computationally manageable, and (iv) plug-and-play inference-time self-refinement mechanism for existing ODE solvers.

**Key Insight**: The authors observe that the flow matching objective can be rewritten as $\mathcal{L}_{\text{FM}}(\theta)=\mathbb{E}\bigl[\tfrac{1}{(1-t)^2}\lVert\hat z_1^\theta-z_1\rVert_2^2\bigr]$, where $\hat z_1^\theta = z_t+(1-t)u_\theta(z_t,t)$ is the model's prediction of the clean sample. This is exactly a weighted version of the **generalized denoising autoencoder** (generalized DAE, Bengio 2013) objective. This means a flow matching model is essentially a time-conditioned DAE across all noise levels, possessing the implicit ability to "corrupt-and-reconstruct" back to the data manifold.

**Core Idea**: Reinterpret the flow matching video generator as a DAE and initiate a pseudo-Gibbs inner loop at each sampling step $t$: use the model to predict the clean endpoint $\hat z_1$ (Predict), then perturb it back to the same noise level (Perturb). Repeat this 2–3 times before performing a standard ODE step—termed **Predict-and-Perturb (P&P)**. Additionally, apply an uncertainty mask based on self-consistency to refine only motion areas, preventing over-saturation in static regions caused by cumulative CFG.

## Method

### Overall Architecture
The input remains Gaussian noise $z_{t_0}\sim\mathcal{N}(0,\mathbf{I})$, which follows an ODE along discretized time steps $t_0<\cdots<t_T=1$. In the early "motion determination window" ($t\le\alpha T$, where $\alpha\approx 0.2$), a standard ODE step yields $z_{t_{i+1}}^{(0)}$. This is followed by a P&P inner loop of $K_f\le 3$ iterations. In each iteration, the "refined ODE result" and an "uncertainty mask" are calculated simultaneously. The mask is used to fuse the refined result with the previous round's result at each spatial-temporal position. Mid-to-late steps skip refinement and proceed with standard ODE. Total inference time increases by only $\sim 1.5\times$ without extra training or external models.

### Key Designs

1.  **Reinterpreting Flow Matching as DAE**:
    - **Function**: Redefines the "sampling process" as a pair of repeatable operators: the Predict operator $D_\theta(z_t,t):=z_t+(1-t)u_\theta(z_t,t)$ maps $z_t$ at any noise level to a clean prediction $\hat z_1$; the Perturb operator $R_\epsilon(z,t):=tz+(1-t)\epsilon$ adds noise to a clean sample back to level $t$.
    - **Mechanism**: The authors prove $\mathcal{L}_{\text{FM}}$ is equivalent to a weighted DAE reconstruction objective $\mathbb{E}\bigl[\lVert\hat z_1^\theta-z_1\rVert_2^2\bigr]/(1-t)^2$. Thus, a trained flow matching model acts as a valid DAE at any fixed $t$, usable an infinite number of times.
    - **Design Motivation**: This shifts the perspective from "one-way ODE trajectory" to a "re-entrant DAE at each $t$," providing the theoretical foundation—without this, repeated perturb-and-predict would lack justification.

2.  **Predict-and-Perturb (P&P) Inner Loop**:
    - **Function**: Performs pseudo-Gibbs sampling within a fixed noise level $t$ to pull $z_t$ toward high-density regions (i.e., the physically and temporally plausible video manifold) before the ODE solver continues.
    - **Mechanism**: A single iteration is defined as $z_t^{(k+1)}=\operatorname{P\&P}_{\epsilon_k}(z_t^{(k)},t):=R_{\epsilon_k}(D_\theta(z_t^{(k)},t),t)$. Starting from $z_t^{(0)}=z_t$, Predict→Perturb is repeated with newly sampled Gaussian noise for local resampling. The refined $z_t^*:=z_t^{(K_f)}$ is then used in a standard ODE step: $z_{t_{i+1}}=z_t^*+\Delta t\cdot u_\theta(z_t^*,t)$.
    - **Design Motivation**: Consistent with findings that "motion and physics are locked in its first few steps," P&P is only triggered for $t<0.2$. Local resampling at the same noise level maintains a larger exploration radius to alleviate early lock-in without the error accumulation seen in cross-level jumps like Restart.

3.  **Uncertainty-aware Selective Refinement (Uncertainty-aware P&P)**:
    - **Function**: Identifies spatial-temporal regions that the model has not yet "seen clearly" and applies P&P only to those areas. This preserves static backgrounds and avoids over-saturation or artifacts caused by cumulative CFG.
    - **Mechanism**: The difference between two consecutive P&P predictions serves as a "self-consistency" signal: $\mathbf{U}(z_{t}^{(k-1)},z_{t}^{(k)}):=\tfrac{1}{C}\lVert D_\theta(z_{t}^{(k-1)},t)-D_\theta(z_{t}^{(k)},t)\rVert_1$ (averaged over channels). This map is binarized with threshold $\tau=0.25$ to produce mask $M_{t_i}^{(k)}$. Fusion is performed as: $z_{t_{i+1}}^{(k)}\leftarrow M_{t_i}^{(k)}\odot z_{t_{i+1}}^{(k)}+(1-M_{t_i}^{(k)})\odot z_{t_{i+1}}^{(k-1)}$. Mask calculation is integrated into the Predict step to avoid extra NFE.
    - **Design Motivation**: Observation showed that $K_f>3$ without masking leads to CFG accumulation on static backgrounds, causing color shifts and exaggerated reflections. Moving regions, which differ between predictions, do not suffer this accumulation. The consistency-based mask automatically separates motion that requires revision from static backgrounds.

### Loss & Training
The method is **completely training-free**. It utilizes the weights of pretrained Wan2.1 / Wan2.2 / Cosmos-Predict-2.5 without any gradient updates or fine-tuning. The algorithm introduces 3 tunable hyperparameters: P&P iterations $K_f$ (default 2–3), uncertainty threshold $\tau$ (default 0.25), and P&P trigger window $\alpha$ (default first ~20% of steps). All operations are executed in latent space.

## Key Experimental Results

### Main Results

Dynamic-bench (Wan2.2-A14B T2V, 120 challenging motion prompts, VBench auto-eval + 20-person subjective eval):

| Method | Human Motion (%) | Human Text (%) | VBench Motion ↑ | VBench Const. ↑ | NFE | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Wan2.2 T2V (Default UniPC) | 73.57 | 57.64 | 98.01 | 90.68 | 40 | 1.0× |
| + NFE×2 | 74.05 | 57.55 | 98.03 | 90.66 | 80 | 2.0× |
| + CFG-Zero | 81.53 | 65.71 | 98.27 | 91.16 | 40 | 1.0× |
| + FlowMo (training-free) | 70.57 | 61.71 | 97.68 | 90.95 | 40* | 3.9× |
| + **Ours** | — | — | **98.41** | **91.33** | 60 | **1.5×** |

PAI-Bench Robotics I2V (Gemini 3 Flash evaluating grasp, Qwen2.5-VL-72B evaluating Robot-QA):

| Base Model + Method | Grasp ↑ | Robot-QA ↑ | Quality ↑ | NFE | Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Cosmos-Predict-2.5 | 79.2 | 71.7 | 75.1 | 35 | 1.0× |
| + Verifier best-of-4 | 84.4 | 72.3 | 75.3 | 140 | 4.0× |
| + **Ours** | **89.6** | **76.3** | 75.1 | 57 | 1.6× |
| Wan2.2-I2V-A14B | 77.3 | 77.4 | 75.3 | 40 | 1.0× |
| + Verifier best-of-4 | 80.5 | 78.1 | 75.3 | 144 | 4.0× |
| + **Ours** | **85.7** | **80.3** | **75.5** | 60 | 1.5× |

**Physical Alignment**: On PhyWorldBench, the PC score improved from 29.3 to **40.0** (+10.7). In VideoPhy2 human evaluations, **84%** of users preferred this method over the default sampler. **Spatial Consistency**: SSIM improved from 0.401 to **0.485**, and PSNR from 14.96 to **17.21 dB**.

### Ablation Study

| Configuration | Key Observation | Explanation |
| :--- | :--- | :--- |
| Full (P&P + Uncertainty mask) | Optimal motion/physical consistency | Complete method |
| P&P, $K_f=5$, No mask | Over-saturation, tone drift | CFG accumulation in static regions |
| P&P, $K_f=2{-}3$, No mask | Improved motion, occasional background simplification | Acceptable with limited iterations |
| P&P at $t<0.2$ only vs. whole process | Early trigger is sufficient | Late triggers yield marginal gains but add NRT |
| $\tau$ 0.15 / 0.25 / 0.40 | 0.25 is most stable | Too low: background altered; too high: motion missed |

Visualized masks show that thresholded 1-regions nearly perfectly track moving objects (hands, limbs), while 0-regions correspond to static backgrounds—proving that "model self-consistency difference" is a valid signal for locating areas needing refinement.

### Key Findings
-   **Early steps determine everything**: Limiting P&P to the first 20% of steps captures almost all gains, confirming that motion/physical structure is locked in early steps.
-   **Uncertainty mask is the cure for over-saturation**: P&P alone can introduce artifacts; the mask ensures stability for larger $K_f$ and prevents CFG accumulation in backgrounds.
-   **Video is more robust than images**: P&P on images can trigger drastic semantic shifts, while on video it only causes gentle motion structural changes. Cross-frame consistency makes P&P a "local search" rather than "global resampling."
-   **Mode-seeking improves temporal consistency**: Iterative P&P is mode-seeking. In videos, this manifests as reduced jitter and flicker, as temporally inconsistent videos lie in low-density regions.
-   **Not all visual reasoning can be fixed**: Graph traversal success jumped from 0.1 to 0.8; however, maze solving showed zero improvement—indicating P&P fixes "motion/temporal" errors but not discrete/semantic correctness, which still requires external verifiers.

## Highlights & Insights
-   **Theoretical Highlight: Flow matching as DAE**. This allows each fixed $t$ to be treated as an independent DAE that can be called repeatedly without breaking the distribution.
-   **Engineering Highlight: Zero-NFE mask calculation**. Binding the next-step ODE calculation with the mask calculation into the same Predict call allows the mask to be a "free lunch."
-   **Methodological Highlight: Uncertainty from prediction variance**. Using the L1 difference between two P&P outputs of the same model locates areas needing refinement without external headers or ensembles.
-   **Transferable Trick**: Adding 2–3 P&P iterations for the first 20% of steps requires only a few lines of code and provides high ROI performance gains.

## Limitations & Future Work
-   **Dependency on model priors**: P&P cannot generate knowledge the base model lacks (e.g., maze solving). It is essentially a more stable search for high-density modes.
-   **Mode-seeking in images**: $K_f=8$ can collapse image diversity. The success in video depends on the unique property of cross-frame consistency.
-   **Empirical hyperparameters**: $\alpha, \tau, K_f$ are robust but currently empirical; there is no automatic estimation mechanism yet.
-   **Future Work**: (i) Adaptive $K_f$ based on mask energy; (ii) Combining P&P with reward-guided fine-tuning; (iii) Exploring masks in 3D generation or video editing.

## Related Work & Insights
-   **vs. Restart Sampling**: Restart uses forward-backward macro cycles across noise levels to fix late-stage errors; this method performs micro-refinement **within the same noise level** to fix early-stage motion errors.
-   **vs. FreeInit**: FreeInit refinies initial noise by rerunning the entire denoising chain; this method refines **intermediate latents** with far higher efficiency.
-   **vs. Verifier-based methods**: Rejection sampling takes 4.0× time for lower gains; P&P's 1.6× time is more cost-effective for physical consistency.
-   **vs. VideoJAM / FlowMo**: Unlike these, this method requires zero gradients and no additional networks, representing a significant leap in engineering friendliness.

## Rating
-   Novelty: ⭐⭐⭐⭐ 
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
-   Writing Quality: ⭐⭐⭐⭐⭐ 
-   Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Self-Correcting Text-to-Video Generation with Misalignment Detection and Localized Refinement](../../ACL2026/video_generation/self-correcting_text-to-video_generation_with_misalignment_detection_and_localiz.md)
- [\[CVPR 2026\] From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning](../../CVPR2026/video_generation/from_static_to_dynamic_exploring_self-supervised_image-to-video_representation_t.md)
- [\[CVPR 2026\] Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout](../../CVPR2026/video_generation/infinity-rope_action-controllable_infinite_video_generation_emerges_from_autoreg.md)
- [\[NeurIPS 2025\] Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](../../NeurIPS2025/video_generation/self_forcing_bridging_the_train-test_gap_in_autoregressive_video_diffusion.md)
- [\[ICML 2026\] Exploring Data-Free LoRA Transferability for Video Diffusion Models](exploring_data-free_lora_transferability_for_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
