---
title: >-
  [Paper Note] Epona: Autoregressive Diffusion World Model for Autonomous Driving
description: >-
  [ICCV 2025][Autonomous Driving][world model] This paper proposes Epona, an autoregressive diffusion world model that achieves a unified framework for high-resolution long-horizon driving video generation and real-time tr…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "world model"
  - "autoregressive diffusion"
  - "trajectory planning"
  - "video generation"
date: 2026-05-08
content_hash: 306789190f609c8f
---

# Epona: Autoregressive Diffusion World Model for Autonomous Driving

**Conference**: ICCV 2025
**arXiv**: [2506.24113](https://arxiv.org/abs/2506.24113)
**Code**: [https://github.com/Kevin-thu/Epona/](https://github.com/Kevin-thu/Epona/)
**Area**: Autonomous Driving
**Keywords**: world model, autoregressive diffusion, trajectory planning, video generation, autonomous driving

## TL;DR

This paper proposes Epona, an autoregressive diffusion world model that achieves a unified framework for high-resolution long-horizon driving video generation and real-time trajectory planning through decoupled spatiotemporal modeling and asynchronous multimodal generation.

## Background & Motivation

Existing driving world models fall into two main categories: 1) **diffusion-based** methods (e.g., Vista) that model fixed-length video frames via joint distribution, achieving high visual quality but lacking the ability to generate variable-length sequences or integrate trajectory planning; 2) **GPT-style** autoregressive methods (e.g., GAIA-1) that support variable-length generation via next-token prediction, but whose quantization and tokenization processes severely degrade visual quality and planning precision. The two paradigms exhibit complementary weaknesses — diffusion models lack temporal decomposition capability, while autoregressive Transformers sacrifice continuous visual fidelity. A unified framework is therefore needed to reconcile the strengths of both approaches.

## Method

### Overall Architecture

Epona reformulates world modeling as a step-by-step future prediction process along the temporal dimension. Given historical driving observations and trajectories, the model simultaneously predicts: 1) the policy distribution $\pi$ for future trajectory planning; and 2) the conditional distribution $p$ for the next-frame camera observation. The overall framework consists of three core components: the Multimodal Spatiotemporal Transformer (MST), the Trajectory Planning DiT (TrajDiT), and the Next-Frame Prediction DiT (VisDiT). The total model size is 2.5B parameters.

### Key Designs

1. **Multimodal Spatiotemporal Transformer (MST, 1.3B parameters)**: Encodes historical context $\{O_t, a_t\}$ into a compact latent representation. It employs interleaved multimodal spatial attention layers and causal temporal attention layers. Visual latent patches $Z \in \mathbb{R}^{B \times T \times L \times C}$ and action sequences $a \in \mathbb{R}^{B \times T \times 3}$ are first projected into an embedding space, then concatenated and processed alternately through causal temporal layers (with causal masking) and multimodal spatial layers. The embedding of the last frame $F \in \mathbb{R}^{B \times (L+3) \times D}$ is extracted as a compact historical representation. This design substantially reduces memory consumption from full-sequence attention and naturally supports variable-length historical context.

2. **Trajectory Planning Diffusion Transformer (TrajDiT, 50M parameters)**: A lightweight diffusion Transformer with a Dual-Single-Stream architecture that predicts future 3-second trajectories. In the dual-stream stage, the historical latent representation $F$ and trajectory data are processed independently and coupled only via attention operations; in the single-stream stage, they are concatenated and fused through subsequent Transformer blocks. During training, noise is added to the target trajectory $\bar{a} \in \mathbb{R}^{B \times N \times 3}$ and optimized with a Rectified Flow loss: $\mathcal{L}_\text{traj} = \mathbb{E}[\|v_\text{traj}(\bar{a}_{(t)}, t) - (\bar{a} - \varepsilon)\|^2]$.

3. **Next-Frame Prediction Diffusion Transformer (VisDiT, 1.2B parameters)**: Shares a similar architecture with TrajDiT, with an additional modulation branch for action control $a_{T \to T+1}$. It also uses a Flow Matching objective: $\mathcal{L}_\text{vis} = \mathbb{E}[\|v_\text{vis}(Z_{T+1(t)}, t) - (Z_{T+1} - \varepsilon)\|^2]$. During inference, the next-frame latent is obtained by denoising conditioned on $F$ and actions (either predicted by TrajDiT or user-provided), which is then decoded into an image via the DCAE decoder.

### Loss & Training

- **Total Loss**: $\mathcal{L} = \mathcal{L}_\text{traj} + \mathcal{L}_\text{vis}$, trained end-to-end jointly.
- **Chain-of-Forward Training Strategy**: To mitigate autoregressive drift (the domain gap between ground-truth conditioning at training time and self-predicted conditioning at inference time), a multi-step forward pass is executed every 10 steps — the model estimates the denoised latent in a single step via the predicted velocity $v_\Theta$: $\hat{x}_{(0)} = x_{(t)} + t \cdot v_\Theta(x_{(t)}, t)$, which is then used as the condition for the next step. Three forward passes are performed each time to simulate inference-time noise and improve robustness.
- **Temporally-Aware DCAE Decoder**: Spatiotemporal self-attention layers are introduced before the DCAE decoder to enhance inter-frame consistency and resolve flickering artifacts from per-frame decoding. The encoder is frozen and only the decoder is fine-tuned.
- **Training Setup**: Trained on 48 A100 GPUs for approximately two weeks, 600K iterations, batch size 96, AdamW optimizer with lr=$1\text{e}{-4}$ and weight decay=$5\text{e}{-2}$. Image resolution $512 \times 1024$.

## Key Experimental Results

### Main Results

| Method | FID ↓ | FVD ↓ | Max Duration / Frames |
|--------|-------|-------|----------------------|
| DriveGAN | 73.4 | 502.3 | N/A |
| DriveDreamer | 52.6 | 452.0 | 4s / 48 |
| Drive-WM | 15.8 | 122.7 | 8s / 16 |
| Vista | 6.9 | 89.4 | 15s / 150 |
| DrivingWorld | 7.4 | 90.9 | 40s / 400 |
| **Epona** | **7.5** | **82.8** | **120s / 600** |

NAVSIM planning performance:

| Method | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|--------|------|-------|-------|---------|------|--------|
| UniAD | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| DRAMA | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| **Epona** | **97.9** | **95.1** | **93.8** | **99.9** | **80.4** | **86.2** |

### Ablation Study

| Configuration | NC ↑ | DAC ↑ | PDMS ↑ |
|---------------|------|-------|--------|
| w/o joint training (trajectory only) | 94.5 | 89.7 | 78.1 |
| Full Epona | 97.9 | 95.1 | 86.2 |

Chain-of-Forward training effect: without this strategy, visual quality degrades rapidly after 10–20 seconds; with it, high-quality generation is maintained at the minute scale.

Temporally-aware DCAE decoder:

| Method | FVD10 ↓ | FVD25 ↓ | FVD40 ↓ |
|--------|---------|---------|---------|
| w/o temporal module | 52.95 | 76.46 | 100.11 |
| Full model | 50.77 | 61.46 | 74.88 |

### Key Findings

- Joint training of video and trajectory via shared latents significantly improves planning performance (PDMS: 78.1 → 86.2).
- The benefit of the Chain-of-Forward strategy grows increasingly pronounced as sequence length increases in long-horizon generation.
- Increasing conditioning frames from 2 to 10 reduces FVD40 from 103.70 to 74.88.
- The model implicitly learns traffic rules (e.g., stopping at red lights) through self-supervised future prediction alone.

## Highlights & Insights

- **Paradigm Innovation**: Epona is the first to decouple and unify autoregressive and diffusion models along the spatiotemporal dimension, preserving the visual quality of diffusion models while gaining the temporal flexibility of autoregressive models.
- **Real-Time Planning**: Through modular design, real-time trajectory planning at 20 Hz is achievable using only MST + TrajDiT.
- **Extremely Long Generation**: The 120-second / 600-frame generation length substantially surpasses contemporary methods (Vista achieves only 15s).
- Chain-of-Forward is a general autoregressive drift mitigation strategy transferable to other domains.

## Limitations & Future Work

- The FID score (7.5) remains slightly higher than Vista (6.9), leaving room for improvement in single-frame quality.
- Only a front-facing monocular camera is used; the framework has not been extended to multi-view panoramic generation.
- Training cost is high (48 A100 GPUs for two weeks), posing a significant deployment barrier.
- Robustness under extreme weather conditions and rare scenarios has not been evaluated.

## Related Work & Insights

- Compared to GPT-style methods such as DrivingWorld, Epona performs autoregressive generation in continuous space rather than discrete token space, preserving visual detail.
- Diffusion Forcing and FIFO-Diffusion also explore the combination of autoregressive and diffusion models, but Epona redefines the architecture as a two-stage end-to-end framework.
- The modular design (MST / TrajDiT / VisDiT can be used independently) enables flexible deployment scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The unified autoregressive diffusion framework is both novel and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation and ablation are provided across both video generation and trajectory planning dimensions.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with complete mathematical derivations and rich figures and tables.
- **Value**: ⭐⭐⭐⭐ Makes a significant contribution to the advancement of world models for autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving](drivex_omni_scene_modeling_for_learning_generalizable_world_knowledge_in_autonom.md)
- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [\[ICLR 2026\] Astra: General Interactive World Model with Autoregressive Denoising](../../ICLR2026/autonomous_driving/astra_general_interactive_world_model_with_autoregressive_denoising.md)
- [\[ICCV 2025\] LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)
- [\[ICCV 2025\] Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)

</div>

<!-- RELATED:END -->
