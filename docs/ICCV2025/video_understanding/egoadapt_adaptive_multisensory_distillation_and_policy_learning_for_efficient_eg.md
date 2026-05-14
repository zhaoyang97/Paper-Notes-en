---
title: >-
  [Paper Note] EgoAdapt: Adaptive Multisensory Distillation and Policy Learning for Efficient Egocentric Perception
description: >-
  [ICCV 2025][Video Understanding][egocentric perception] EgoAdapt is a framework that jointly trains cross-modal distillation and policy learning to adaptively select the optimal modality combination…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "egocentric perception"
  - "multimodal distillation"
  - "policy learning"
  - "efficient inference"
  - "multisensory fusion"
date: 2026-05-08
content_hash: 8f68dd340943694b
---

# EgoAdapt: Adaptive Multisensory Distillation and Policy Learning for Efficient Egocentric Perception

**Conference**: ICCV 2025
**arXiv**: [2506.21080](https://arxiv.org/abs/2506.21080)
**Code**: None
**Area**: Video Understanding
**Keywords**: egocentric perception, multimodal distillation, policy learning, efficient inference, multisensory fusion

## TL;DR

EgoAdapt is a framework that jointly trains cross-modal distillation and policy learning to adaptively select the optimal modality combination, achieving up to 89% GMACs reduction while maintaining performance on par with or superior to SOTA on egocentric perception tasks.

## Background & Motivation

Modern AR/VR systems rely on multi-sensor data streams (RGB video, multi-channel audio, behavioral data) for egocentric perception, yet SOTA models incur prohibitive computational costs that preclude real-time deployment on resource-constrained devices. A key observation is that **not all modalities need to be processed simultaneously**. For instance, in active speaker localization, visual cues suffice when multiple people are visible, whereas multi-channel audio becomes more effective when the speaker is outside the field of view.

Existing approaches either perform model distillation alone (static, unable to adapt to varying task demands) or adaptive modality selection alone (still relying on expensive models), lacking a **unified framework** that addresses both. EgoAdapt's core insight is that jointly training distillation and policy optimization yields a **synergistic effect**—the efficiency of the distilled model is complemented by the adaptability of policy optimization, resulting in a system that is simultaneously lightweight and flexible.

## Method

### Overall Architecture

EgoAdapt comprises two core modules: a **Cross-modal Feature Distillation (CFD)** module Φ and a **Task-aware Multisensory Policy Learning (TeMPLe)** module Π, optimized end-to-end via a three-stage joint training strategy. The framework is applicable to three egocentric tasks: Action Recognition (AR), Active Speaker Localization (ASL), and Behavior Anticipation (BA).

### Key Designs

1. **Cross-modal Feature Distillation (CFD)**: A lightweight student model Φ is trained to approximate the performance of a heavy teacher model Ω. Features $z_I, z_A, z_B$ are extracted from visual frames I, audio A, and behavioral data B respectively, and fused via a fusion network ξ to obtain $z_\phi$, such that $\Phi(I, A, B) \approx \Omega(V)$. The training loss comprises three terms:

    - L1 feature matching loss: $\mathcal{L}_1 = \sum \| z_{\Omega_i} - z_{\phi_i} \|_1$
    - KL divergence knowledge distillation loss: $\mathcal{L}_{KD} = \sum D_{KL}(\sigma(\Omega(V)/\tau), \sigma(\Phi(I,A,B)/\tau))$
    - Cross-entropy prediction loss: $\mathcal{L}_{GT} = \sum \mathcal{L}_{CE}(c_i, \sigma(\Phi(I,A,B)))$
    - Combined loss: $\mathcal{L}_\Phi = \alpha \mathcal{L}_{KD} + (1-\alpha)\mathcal{L}_{GT} + \beta \mathcal{L}_1$

2. **Task-aware Multisensory Policy Learning (TeMPLe)**: A policy network consisting of lightweight modality feature extractors and an LSTM module is designed, with **Gumbel-Softmax sampling** enabling differentiable training of discrete policies. At each timestep t, the LSTM receives joint features and historical hidden states, outputting binary policy decisions $u_{t,k}$ that determine whether modality k is activated.

    - For ASL and BA tasks: the model learns modality switching policies (which modalities and audio channels to select)
    - For AR tasks: an **audio preview** strategy is adopted—audio is first analyzed to detect regions of interest, from which the single most informative frame is selected for recognition

3. **Audio-guided Frame Selection (AR-specific)**: A "handshake" mechanism between multi-head attention and a recurrent CNN extracts temporally aware audio features; an LSTM then detects potential event regions, with only one frame selected per region for action recognition, achieving extreme frame efficiency. The handshake operation is formulated as $z_{l+1}^{RCNN} = z_l^{RCNN} + \rho z_l^{MH}$, where ρ is a learnable parameter.

### Loss & Training

Three-stage training:
- **Stage 1**: The policy module is disabled; only the distillation module Φ is trained.
- **Stage 2**: Φ is frozen; the policy module Π is trained using an efficiency-penalized loss.
- **Stage 3**: Φ and Π are jointly fine-tuned with the combined objective $\mathcal{L}_\Theta = \eta_1 \mathcal{L}_\Pi + \eta_2 \mathcal{L}_\Phi$.

Efficiency penalty during policy training: $\mathcal{C}_k = (|U_k|_0 / C)^2$, with λ_k and γ controlling the accuracy–efficiency trade-off.

## Key Experimental Results

### Main Results

| Method | Verb↑ | Noun↑ | Action↑ | GMACs↓ |
|--------|-------|-------|---------|--------|
| TIM AV (Teacher) | 77.19 | 67.22 | 57.57 | 26.62 |
| Ego-only | 73.33 | 59.48 | 52.59 | 507.39 |
| AdaMML | 64.95 | 55.27 | 41.73 | 277.76 |
| EgoAdapt w/o TeMPLe | 68.34 | 59.02 | 50.88 | 5.79 |
| **EgoAdapt** | **76.65** | **66.83** | **56.74** | **7.14** |

On the ASL task (EasyCom): EgoAdapt achieves 89.74% mAP with only 0.070 GMACs and 0.39M parameters, representing an approximately 99% reduction compared to MAVSLC+E at 6.852 GMACs.

### Ablation Study

| λ_K Setting | γ | AR Acc↑ | AR GMACs↓ | ASL mAP↑ | ASL GMACs↓ |
|-------------|---|---------|-----------|----------|------------|
| [0,0,0] | 0 | 56.99 | 13.68 | 89.77 | 0.391 |
| [1,1,1] | 1 | 56.27 | 9.23 | 89.48 | 0.102 |
| [1,0.05,0.03] | 1 | 56.83 | 7.67 | 89.76 | 0.092 |
| [1,0.05,0.03] | 10 | 56.74 | 7.14 | 89.75 | 0.070 |

Training stage ablation: CFD with a random policy yields only 67.41 mAP; Stages 1+2 reach 83.64; joint training through Stage 3 achieves 89.74, demonstrating the importance of joint training.

### Key Findings

- EgoAdapt approaches the teacher model TIM (26.62 GMACs, 57.57% Action accuracy) on AR with only 7.14 GMACs.
- On ASL, it achieves an 82.02% parameter reduction while matching the performance of the teacher MUST.
- On BA, MAE is reduced by 18.08%, with energy consumption of only 0.003J compared to 0.972J for GLC.
- After 4-bit quantization, ASL still achieves 78.92 mAP with power consumption of only 9.94mW.

## Highlights & Insights

- This work is the first to **unify distillation and policy learning within a single differentiable framework**, resolving discrete policy optimization via Gumbel-Softmax.
- The audio preview single-frame strategy is particularly elegant—using the cheapest modality to guide the use of the most expensive one.
- The policy module's action space is flexibly adjustable per task, offering strong generalizability.
- The system achieves >180 FPS at 28% GPU utilization on a GTX 2080Ti.

## Limitations & Future Work

- Validation is currently limited to three egocentric tasks; more complex hand-object interaction and long-horizon activity understanding remain unexplored.
- The policy space could be extended to dimensions such as spatial resolution selection and network quantization.
- The audio preview strategy may fail in silent scenarios.

## Related Work & Insights

- Compared to AdaMML, which only performs adaptive multimodal learning, EgoAdapt additionally introduces distillation to reduce the cost of the base model.
- Compared to EgoDistill, which only performs distillation, EgoAdapt adds dynamic modality selection capability.
- The joint training paradigm is generalizable to other multimodal efficiency optimization scenarios, such as multi-sensor fusion in autonomous driving.

## Rating

- Novelty: ⭐⭐⭐⭐ (The unified framework jointly combining distillation and policy learning is innovative)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three tasks, three datasets, comprehensive ablation, quantization, and qualitative analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, complete formulations)
- Value: ⭐⭐⭐⭐ (Practically significant for edge deployment in AR/VR)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning](../../ICLR2026/video_understanding/stabilizing_policy_gradients_for_sample-efficient_reinforcement_learning_in_llm_.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[NeurIPS 2025\] Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning](../../NeurIPS2025/video_understanding/adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)
- [\[ICCV 2025\] MobileViCLIP: An Efficient Video-Text Model for Mobile Devices](mobileviclip_an_efficient_video-text_model_for_mobile_devices.md)
- [\[ICCV 2025\] General Compression Framework for Efficient Transformer Object Tracking](general_compression_framework_for_efficient_transformer_object_tracking.md)

</div>

<!-- RELATED:END -->
