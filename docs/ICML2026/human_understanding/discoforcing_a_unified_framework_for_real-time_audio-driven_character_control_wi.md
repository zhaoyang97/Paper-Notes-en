---
title: >-
  [Paper Note] DiscoForcing: A Unified Framework for Real-Time Audio-Driven Character Control with Diffusion Forcing
description: >-
  [ICML 2026][Human Understanding][Audio-driven Character Control] DiscoForcing reformulates the "music $\to$ full-body dance" offline generation problem into a strictly causal…
tags:
  - "ICML 2026"
  - "Human Understanding"
  - "Audio-driven Character Control"
  - "Diffusion Forcing"
  - "Streaming Generation"
  - "VQ-PAE"
  - "Temporal Guidance"
date: 2026-05-08
content_hash: 56e98bcd26058652
---

# DiscoForcing: A Unified Framework for Real-Time Audio-Driven Character Control with Diffusion Forcing

**Conference**: ICML 2026  
**arXiv**: [2605.28491](https://arxiv.org/abs/2605.28491)  
**Code**: https://discoforcing.github.io/ (Available)  
**Area**: Human Understanding / Music-driven Motion Generation / Streaming Diffusion  
**Keywords**: Audio-driven Character Control, Diffusion Forcing, Streaming Generation, VQ-PAE, Temporal Guidance

## TL;DR
DiscoForcing reformulates the "music $\to$ full-body dance" offline generation problem into a strictly causal, bounded-latency streaming task. Using a causal VQ-PAE music encoder, latent-space Diffusion Forcing, a hybrid-mode noise scheduler, and temporal guidance sampling, it translates music streams in real-time into 30 FPS full-body motions capable of driving Unity avatars and Unitree G1 humanoid robots.

## Background & Motivation
**Background**: Existing music-to-motion frameworks (FACT, Bailando, EDGE, Lodge, MEGADance) are mostly designed for offline settings: they either require full future context, use long time windows, or allow non-causal revisitations of history. While these achieve high metrics offline, they cannot be directly deployed in real-time loops like VR avatars, interactive animations, or humanoid robots where music arrives "as it plays."

**Limitations of Prior Work**: Deploying offline models in streaming scenarios without modification leads to three major issues: significant beat synchronization lag, long-horizon cumulative drift and jitter, and slow response to sudden changes (e.g., drum entries, style shifts, or user edits). These stem not from poor engineering but from the structural redefinition of the task in streaming: it must rely solely on causal observations, make final predictions without backtracking, and strictly adhere to per-frame compute budgets.

**Key Challenge**: Streaming music-driven control inherently faces a stability vs. responsiveness tradeoff. Over-reliance on motion history ensures smoothness but slows down reactivity; aggressive history overriding increases responsiveness but causes jitters and discontinuities. Coupled with exposure bias in autoregressive rollouts, small prediction errors pollute the future conditioning history, amplifying errors under non-stationary music.

**Goal**: To formulate music-driven character control as bounded-latency streaming motion generation—predicting the next short motion segment given causal music features $\mathbf{c}_t$ and a finite motion history $\mathbf{m}_{t-h:t-1}$, while building a "model + real-time system" ecosystem where metrics align with real-world interactive experiences.

**Key Insight**: Diffusion Forcing (Chen et al., 2024a) utilizes token-wise heterogeneous noise during sequential diffusion training, making it naturally more robust to imperfect autoregressive histories. Adapting it into a "latent-space + streaming + music-conditioned" version with a specialized sampling scheduler can solve the structural challenges of streaming.

**Core Idea**: The music encoder is designed to be strictly causal and decoupled into beat tokens + periodic phases; the diffusion backbone uses Latent Diffusion Forcing with a triple-mode noise scheduler; and Temporal Guidance is introduced in sampling to "dilute" distant history with noise, replacing standard CFG. This trio collectively addresses causality, long-term stability, and responsiveness to sudden transitions.

## Method

### Overall Architecture
DiscoForcing executes a three-stage pipeline at each streaming timestep $t$:
**Input**: 30 Hz real-time audio stream + motion history buffer of length $h$.
**Stage 1 — Causal Music Processing**: Extracts causal features $\mathbf{c}_t=[\mathbf{f}_t^{vq};\mathbf{f}_t^{pae}]$ (VQ-PAE, beat tokens + FFT phases) from a fixed-length sliding audio window.
**Stage 2 — Streaming Audio-driven Motion Diffusion**: Compresses 272-dimensional SMPL frames into a latent sequence $\mathbf{z}_\mathcal{T}$ via a motion VAE. A Diffusion Forcing transformer then performs joint denoising within a length-$l$ tail FIFO window, outputting one clean latent token per step and appending a new pure noise token.
**Stage 3 — Real-Time Interactive System**: Decoded SMPL motions are distributed via ROS2 to a Unity avatar platform (for visualization/interaction) and a Unitree G1 humanoid robot (via GMR retargeting + 50 Hz WBC tracking).
**Output**: 30 FPS causal streaming full-body motion + physically executable joint commands.

### Key Designs

1.  **VQ-PAE Decoupled Causal Music Encoding**:
    - **Function**: Encodes each sliding audio window $\mathbf{w}_t$ into a compact, strictly causal condition vector $\mathbf{c}_t$ that preserves both high-level beat semantics and low-level phase continuity.
    - **Mechanism**: Dilated 1D causal convolutions produce a shared latent representation $\mathbf{f}^{causal}=\mathrm{Conv1D}(\mathbf{w}_t)$. It then branches: the beat branch uses Residual Vector Quantization $\mathbf{f}^{vq}=\mathrm{RVQ}(\mathrm{FFN}_{vq}(\mathbf{f}^{causal}))$ for discrete style/trigger codes; the periodic branch estimates amplitude $\mathbf{A}$, offset $\mathbf{B}$, frequency $\mathbf{F}$, and phase $\boldsymbol{\phi}=\tan^{-1}(\mathrm{FC}_{phase}(\mathbf{f}^{causal}))$ in the frequency domain, reconstructing $\mathbf{f}^{pae}(t)=\mathbf{A}\sin(2\pi(\mathbf{F}\cdot t-\boldsymbol{\phi}))+\mathbf{B}$ for smooth phase-aligned signals.
    - **Design Motivation**: Real-time deployment cannot access the future, necessitating strict causality. Music contains both "discrete events" (drum kicks) and "continuous phases" (beat cycles); a single representation either loses granularity or smoothes out rhythms. Ablations show VQ-PAE reduces FIDk from 25.49 to 23.23 compared to Librosa, proving learned semantic conditions are more critical than hand-crafted features for realism.

2.  **Latent Diffusion Forcing + Triple-mode Noise Scheduler**:
    - **Function**: Trains a transformer $\mathbf{v}_\theta$ to accept independent noise levels for each time token, making it robust to "past-clean/future-noisy" combinations in both training and streaming inference.
    - **Mechanism**: Motions enter the latent space via VAE. Probability paths are defined for each token: $p(\mathbf{x}_t^{k_t}|\mathbf{x}_t^0)=\mathcal{N}(\alpha_{k_t}\mathbf{x}_t^0,\sigma_{k_t}^2\mathbf{I})$, where $k_t\in[0,1]$ is the token-wise diffusion time. Flow matching learns the velocity field $\mathbf{v}_t=\dot\alpha_{k_t}\mathbf{x}_t^0+\dot\sigma_{k_t}\boldsymbol{\epsilon}_t$, with loss aggregated only on tokens where $k_t>0$. During training, three schedulers are sampled: Random ($k_t\sim\mathcal{U}(0,1)$), Monotonic (linearly increasing from 0 to 1 within window $[\tau-l,\tau]$), and Trapezoid (adding noise to distant past on top of Monotonic, $k_t^{trap}=\max(k_t^{hist},k_t^{mono})$).
    - **Design Motivation**: Vanilla Diffusion Forcing uses only a Random schedule, creating a discrepancy between training patterns and inference patterns (where the past is clean, the current window is denoising, and the distant past is diluted). Triple-mode scheduling explicitly incorporates inference patterns into training to bridge the train-test gap. The latent space reduces computation and implicitly filters sensor jitter.

3.  **Streaming Sampler and Temporal Guidance (TG)**:
    - **Function**: Performs joint denoising under strictly bounded latency and uses trapezoid noise on distant history to allow audio conditions to actively override outdated history, parameterizing the stability-responsiveness tradeoff.
    - **Mechanism**: A FIFO denoising window of length $l$ is maintained. At each step $\tau$, a solver runs once on all tokens (step size $\delta=1/l$ ensures one update per output frame). The left-most token $(k_t=0)$ is output, and a new noise token is appended. TG replaces CFG's "cond vs. uncond" with "audio cond + trapezoid history vs. no cond + monotonic history": $\mathbf{v}^{guided}=\mathbf{v}_\theta(\hat{\mathbf{x}}^{k^{mono}},k^{mono},\varnothing)+\omega[\mathbf{v}_\theta(\hat{\mathbf{x}}^{k^{trap}},k^{trap},\mathbf{c})-\mathbf{v}_\theta(\hat{\mathbf{x}}^{k^{mono}},k^{mono},\varnothing)]$. The guidance scale $\omega$ controls response intensity.
    - **Design Motivation**: CFG assumes static global conditions, which fails for non-stationary music streams. Streaming requires "layering dynamic conditions over a dynamic history." By adding trapezoid noise to distant history, the guidance term implicitly filters out high-frequency motion priors that conflict with current music, enabling immediate synchronization with new drum entries without losing medium-term context. TG improves FIDk from 23.23 to 18.87 and reduces FSR significantly compared to CFG.

### Loss & Training
Two-stage training: first, the motion VAE is trained with $\mathcal{L}_{VAE}=\|\mathbf{m}_\mathcal{T}-\hat{\mathbf{m}}_\mathcal{T}\|_2^2+\lambda D_{KL}(q(\mathbf{z}|\mathbf{m})\|\mathcal{N}(0,I))$; then, the VAE is frozen and the Diffusion Forcing transformer is trained with $\mathcal{L}_{DF}=\mathbb{E}_{k_\mathcal{T},\mathbf{z}_\mathcal{T},\boldsymbol{\epsilon}_\mathcal{T}}[\|\mathbf{v}_\theta(\mathbf{x}^{k_\mathcal{T}}_\mathcal{T},k_\mathcal{T},\mathbf{c})-\mathbf{v}_\mathcal{T}\|_\mathcal{K}^2]$, where the masked norm only counts tokens with $k_t>0$. Inference utilizes a 10-step denoising process to maintain 30 FPS.

## Key Experimental Results

### Main Results

| Dataset | Metric | DiscoForcing | Prev. SOTA (Method) | Gain |
|--------|------|------|----------|------|
| FineDance | FIDk ↓ | 23.84 | 50.00 (Lodge/MEGA) | −52% |
| FineDance | FIDg ↓ | 8.62 | 13.02 (MEGA) | −34% |
| FineDance | FSR ↓ | 0.142 | 0.028 (Lodge) | Slightly Worse |
| FineDance | BAS ↑ | 0.225 | 0.226 (Lodge/MEGA) | Comparable |
| AIST++ | FIDk ↓ | 18.87 | 25.89 (MEGA) | −27% |
| AIST++ | FIDg ↓ | 11.57 | 9.62 (Bailando) | Slightly Worse |
| AIST++ | BAS ↑ | 0.244 | 0.242 (Lodge) | +0.002 |

DiscoForcing significantly leads in motion quality (FIDk) across datasets, maintains beat alignment (BAS) comparable to the best methods, and is the only system in the comparison featuring all five capabilities: online, long-horizon, music-transition, physical platform, and user-interactive.

### Ablation Study (AIST++)

| Configuration | FIDk ↓ | FSR ↓ | BAS ↑ | Latency (ms/frame) | Description |
|------|--------|-------|-------|--------------|------|
| 263d (HumanML3D) | 26.47 | 0.060 | 0.245 | 26.60 | Lacks joint rotation; requires IK |
| 272d (Ours) | 25.49 | 0.115 | 0.247 | 26.68 | Direct FK; real-time retargetable |
| Librosa Encoding | 25.49 | 0.115 | 0.247 | 26.68 | Hand-crafted beat features |
| VQ-PAE Encoding | 23.23 | 0.097 | 0.238 | 26.73 | Learned semantic conditions |
| CFG Guidance | 23.23 | 0.097 | 0.238 | 26.73 | Standard unconditional dropout |
| Temporal Guidance | **18.87** | **0.059** | **0.244** | 26.26 | Trapezoid history noise |
| 5-step Denoising | 28.63 | 0.062 | 0.242 | 14.02 | Too fast; loses realism |
| 10-step (Ours) | 18.87 | 0.059 | 0.244 | 26.26 | 30 FPS real-time sweet spot |
| 100-step | 17.58 | 0.080 | 0.248 | 261.91 | Diminishing returns; breaks latency |

### Key Findings
- **Relative contribution of core modules**: Temporal Guidance > VQ-PAE > 272d representation. TG alone reduces FIDk from 23.23 to 18.87 (−19%) while improving both FSR and BAS, proving that "how to handle history" is more critical than "how to encode music" in streaming.
- **10-step denoising is the Latency-Quality Pareto sweet spot**: There is a massive jump from 5 to 10 steps; however, increasing from 10 to 100 steps only yields a 7% FIDk improvement while increasing latency by 10x. This sets a clear ROI boundary for future work.
- **BAS (Beat Alignment)**: DiscoForcing only "matches" mainstream methods (0.225 vs. 0.226). However, the authors emphasize that baselines use future context ("cheating" in streaming terms), making the comparable score a structural victory.

## Highlights & Insights
- **Paradigm Shift from CFG to Temporal Guidance**: While CFG treats conditions as static global variables, TG treats "distant history" as a controllable condition strength. This formalizes the stability-responsiveness trade-off into a single $\omega$ dial, a concept extensible to any "dynamic condition + autoregressive history" task (e.g., streaming video, TTS, robot policies).
- **Matching Training Schedules to Inference Patterns**: Vanilla Diffusion Forcing's instability in streaming arises because it never encounters the "clean past, noisy future, trapezoid-diluted history" pattern during training. Training distributions must proactively cover deployment schedules.
- **Deployment-Faithful Evaluation**: The authors insist on matched causality and latency, building a dual frontend of Unity + Unitree G1. This "benchmark + real-world interaction" dual-track approach should become the new standard for streaming generative models.

## Limitations & Future Work
- **Limitations**: Robustness in out-of-distribution scenarios (extreme beat changes, noisy music), richer user control signals, stronger physical feasibility constraints, and broader real-world testing need improvement.
- **Personal Insights**: FSR (foot sliding) on FineDance (0.142) is much higher than Lodge (0.028), indicating that foot contact consistency remains unresolved under streaming constraints. Geometric dance vocabulary (FIDg) on AIST++ still lags behind offline Bailando (9.62). The 26ms per-frame budget for 30 FPS strictly depends on 10-step denoising; scaling to larger models or higher resolutions may challenge this stability.
- **Humanoid Path Details**: Quantized results for GMR retargeting and WBC tracking stability are sparse; this remains the potentially weakest link in end-to-end experience.
- **Future Directions**: Incorporating foot contact and physical feasibility into the Diffusion Forcing objective (via contact loss or simulator-in-the-loop), making $\omega$ adaptive to music change rates, and extending TG to "soft causal" versions with minimal lookahead buffers (e.g., 100ms).

## Related Work & Insights
- **vs. Bailando / FACT (Autoregressive Transformer)**: These suffer from cumulative drift and weakened beats over long rollouts; DiscoForcing uses token-wise heterogeneous noise + latent VAE to mitigate exposure bias and high-frequency jitter.
- **vs. EDGE / Lodge (Offline Diffusion)**: These rely on future audio context and non-causal revisitation for coherence; DiscoForcing uses Hybrid Scheduling + TG to compensate for missing future info under strict causality.
- **vs. MEGADance (MoE Mamba-Transformer)**: MEGADance scales backbone capacity; DiscoForcing focuses on "correct streaming condition management," surpassing MEGADance in FIDk with a smaller model. These routes are orthogonal and can be combined.
- **vs. Self Forcing / Rolling Forcing**: While targeting streaming diffusion, these assume static or slowly changing conditions; DiscoForcing treats "non-stationary real-time music" as a first-class citizen, providing a template for streaming + dynamic condition scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Adapting Diffusion Forcing for streaming music control and replacing CFG with TG is a clear, transferable innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers two datasets, extensive ablations, and latency data; loses one star for lacking quantitative evaluation on the physical humanoid path.
- **Writing Quality**: ⭐⭐⭐⭐⭐范 The problem statement, motivation, and three-stage method are exceptionally clear; the inclusion of pseudo-code makes it a model for streaming generation papers.
- **Value**: ⭐⭐⭐⭐⭐ Beyond SOTA numbers, it provides a complete paradigm (model + real-time system + dual deployment) that serves as an immediate engineering reference for animation, VR, and robotics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking](../../CVPR2026/human_understanding/unils_end-to-end_audio-driven_avatars_for_unified_listening_and_speaking.md)
- [\[ACL 2026\] Hybrid Autoregressive-Diffusion Model for Real-Time Sign Language Production](../../ACL2026/human_understanding/hybrid_autoregressive-diffusion_model_for_real-time_sign_language_production.md)
- [\[CVPR 2026\] ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data](../../CVPR2026/human_understanding/remogen_real-time_human_interaction-to-reaction_generation_via_modular_learning_.md)
- [\[AAAI 2026\] New Synthetic Goldmine: Hand Joint Angle-Driven EMG Data Generation Framework for Micro-Gesture Recognition](../../AAAI2026/human_understanding/new_synthetic_goldmine_hand_joint_angle-driven_emg_data_generation_framework_for.md)
- [\[NeurIPS 2025\] MOSPA: Human Motion Generation Driven by Spatial Audio](../../NeurIPS2025/human_understanding/mospa_human_motion_generation_driven_by_spatial_audio.md)

</div>

<!-- RELATED:END -->
