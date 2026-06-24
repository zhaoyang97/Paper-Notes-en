---
title: >-
  [Paper Note] Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations
description: >-
  [ICML 2025 (Spotlight)][Image Generation][Video Diffusion Model] This paper utilizes the "predictive visual representation" within Video Diffusion Models (VDMs), which simultaneously encodes current and future frame information, to implicitly learn an inverse dynamics model. This allows for action generation in a high-frequency, closed-loop manner, substantially outperforming existing methods on both simulation and real-world manipulation tasks.
tags:
  - "ICML 2025 (Spotlight)"
  - "Image Generation"
  - "Video Diffusion Model"
  - "Predictive Visual Representation"
  - "Inverse Dynamics"
  - "Robot Manipulation"
  - "Generalist Policy"
date: 2026-05-08
content_hash: 0e48c3bfb2b97efc
---

# Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations

**Conference**: ICML 2025 (Spotlight)  
**arXiv**: [2412.14803](https://arxiv.org/abs/2412.14803)  
**Code**: [Project Page](https://video-prediction-policy.github.io)  
**Area**: Image Generation  
**Keywords**: Video Diffusion Model, Predictive Visual Representation, Inverse Dynamics, Robot Manipulation, Generalist Policy

## TL;DR

This paper utilizes the "predictive visual representation" within Video Diffusion Models (VDMs), which simultaneously encodes current and future frame information, to implicitly learn an inverse dynamics model. This allows for action generation in a high-frequency, closed-loop manner, substantially outperforming existing methods on both simulation and real-world manipulation tasks.

## Background & Motivation

Building generalist robot policies capable of executing diverse tasks is a major focus in current research, with the **visual encoder** being a critical component. Existing visual pre-training methods mainly include:

- **Single-image reconstruction** (e.g., MAE, VC-1): only captures static spatial information.
- **Two-frame contrastive learning** (e.g., R3M, VIP): only compares two frames, resulting in limited dynamic information.
- **Image-text contrastive learning** (e.g., CLIP): focuses on semantic alignment, lacking physical dynamics modeling.

The common limitation of these methods is their **inability to explicitly encode future states**, whereas robot manipulation fundamentally requires predicting future physical evolution.

Meanwhile, Video Diffusion Models (VDMs) have demonstrated strong capabilities in understanding the physical world during video generation. This paper proposes a key hypothesis: the intermediate representations of VDMs naturally possess a $(T, H, W)$ structure, where the first frame represents the current state and the subsequent $T-1$ frames denote the predicted future—making this "predictive visual representation" extremely valuable for robotic tasks.

Prior works leveraging video prediction for control (e.g., SuSIE, UniPi) require complete denoising to generate images, leading to slow inference and low control frequencies. GR-1 is autoregressive but outputs only one frame at a time, and it does not utilize pre-trained video foundation models. VPP directly extracts the internal representations of the VDM with a single forward pass, bypassing multi-step denoising to achieve high-frequency, closed-loop control.

## Method

### Overall Architecture

VPP adopts a **two-stage training** process:

1. **Stage 1 — Fine-tuning the TVP Model**: Fine-tunes the general video foundation model SVD (1.5B parameters) into a text-guided video prediction model tailored for the manipulation domain, using internet human manipulation data, robot data, and self-collected data.
2. **Stage 2 — Learning Actions from Predictive Representations**: Freezes the TVP model as a visual encoder, aggregates spatio-temporal features via a Video Former, and then generates action sequences using a Diffusion Policy head.

Key Insight: The downstream policy can **implicitly track robot motion trajectories** within the predictive representations to learn the inverse dynamics model. As long as the video model can accurately predict future scenes, the policy can generate correct actions through implicit tracking, thereby transferring the generalization capability of the video prediction model to the policy.

### Key Design 1: Text-Guided Video Prediction Model (TVP)

Modified based on Stable Video Diffusion: a cross-attention layer is integrated for CLIP language features $l_{emb}$ to achieve text guidance, the output resolution is adjusted to $16 \times 256 \times 256$, and pre-trained SVD weights are maintained.

The training objective is the diffusion reconstruction loss:

$$\mathcal{L}_D = \mathbb{E}_{x_0 \sim D, \epsilon, t} \| V_\theta(x_t, l_{emb}, s_0) - x_0 \|^2$$

where $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_t$ is the noisy video and $s_0$ is the initial frame. To balance different datasets, sampling coefficients are introduced:

$$\mathcal{L}_{video} = \lambda_H \mathcal{L}_{D_H} + \lambda_R \mathcal{L}_{D_R} + \lambda_C \mathcal{L}_{D_C}$$

Training data includes: 191K Something-Something-v2 human manipulation trajectories (sampling ratio 0.30), approximately 155K Open X-Embodiment robot trajectories, and downstream task data. Fine-tuning takes around 2-3 days using 8×A100 GPUs.

### Key Design 2: Single-Step Forward Extraction of Predictive Representations + Video Former

**Single-step forward encoding**: The current image $s_0$ is concatenated with pure noise and fed into the TVP model. Performing only **one forward pass**, it extracts the intermediate features of the upsampling layers:

$$L_m = V_\theta(x_{t'}, l_{emb}, s_0)^{(m)}, \quad L_m \in \mathbb{R}^{T \times C_m \times W_m \times H_m}$$

Features from each layer are interpolated to a uniform size and concatenated along the channel dimension to obtain $F_p \in \mathbb{R}^{T \times (\sum_m C_m) \times W_p \times H_p}$. Visualization demonstrates that although single-step forward predictions are imprecise in texture, they successfully capture object motion and robotic arm trajectories.

**Video Former**: Initializes learnable tokens $Q_{[0:T, 0:L]}$ and aggregates multi-view and multi-frame features through spatio-temporal attention:

$$Q' = \{ \text{Spat-Attn}(Q[i], (F_p^{static}[i], F_p^{wrist}[i])) \}_{i=0}^{T}, \quad Q'' = \text{FFN}(\text{Temp-Attn}(Q'))$$

### Key Design 3: Diffusion Policy Action Head

$Q''$ is injected into the Diffusion Transformer blocks via cross-attention, and action sequences are generated using the diffusion policy:

$$\mathcal{L}_{diff}(\psi; A) = \mathbb{E}_{a_0, \epsilon, k} \| D_\psi(a_k, l_{emb}, Q'') - a_0 \|^2$$

Utilizing action chunking (10 steps), the overall inference requires only a single forward pass of the TVP (<160ms), achieving a **7-10 Hz** control frequency on an RTX 4090.

## Key Experimental Results

### Calvin ABC→D Zero-shot Long-horizon Evaluation

| Method | Type | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg. Len |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| RT-1 | Direct Action | 0.533 | 0.222 | 0.094 | 0.038 | 0.013 | 0.90 |
| Diffusion Policy | Direct Action | 0.402 | 0.123 | 0.026 | 0.008 | 0.00 | 0.56 |
| GR-1 | Future Prediction | 0.854 | 0.712 | 0.596 | 0.497 | 0.401 | 3.06 |
| Vidman | Future Prediction | 0.915 | 0.764 | 0.682 | 0.592 | 0.467 | 3.42 |
| RoboUniview | 3D Method | 0.942 | 0.842 | 0.734 | 0.622 | 0.507 | 3.65 |
| **VPP (Ours)** | **Predictive Rep.** | **0.965** | **0.909** | **0.866** | **0.820** | **0.769** | **4.33** |
| GR-1 (10% data) | — | 0.672 | 0.371 | 0.198 | 0.108 | 0.069 | 1.41 |
| VPP (10% data) | — | 0.878 | 0.746 | 0.632 | 0.540 | 0.453 | 3.25 |

VPP improves the SOTA Avg. Len from 3.65 to **4.33** (+18.6%). Even with only 10% data, VPP (3.25) still outperforms all baselines trained on full data.

### Real-world Experiment Success Rate

| Platform / Task Type | Diffusion Policy | Susie | GR-1 | **VPP** |
|:---|:---:|:---:|:---:|:---:|
| **Franka Panda — Seen Tasks** | 0.42 | 0.56 | 0.52 | **0.85** |
| **Franka Panda — Unseen Tasks** | 0.25 | 0.46 | 0.38 | **0.73** |
| **Dexterous Hand — Seen Tasks** | 0.28 | 0.45 | 0.32 | **0.75** |
| **Dexterous Hand — Unseen Tasks** | 0.11 | 0.28 | 0.15 | **0.60** |
| **Dexterous Hand — Tool Use** | 0.05 | 0.23 | 0.15 | **0.68** |

On the dexterous hand, VPP improves unseen tasks by **31.6%** compared to the strongest baseline Susie, showing particularly significant improvements in tool-use tasks (0.68 vs 0.23).

## Key Findings

1. **Predictive representations significantly outperform static representations**: Replacing the encoder with Stable-VAE / VC-1 / Voltron drops the Avg. Len to 2.58 / 1.23 / 1.54, respectively (compared to 4.33 for VPP).
2. **A single forward pass contains sufficient information**: Visualization highlights that single-step predictions, despite having imprecise textures, are sufficient for capturing object movements and robotic arm trajectories.
3. **Internet data and pre-training are crucial**: Removing internet data drops Avg. Len from 4.33 to 3.97; further removing SVD pre-training plummets the performance to 1.63.
4. **Video Former is indispensable**: Removing it degrades performance to 3.86 and increases inference latency from 140ms to 450ms.
5. **Multi-layer feature aggregation outperforms single-layer**: Using only the final layer reduces the Avg. Len from 4.33 to 3.60.
6. **Outstanding video prediction quality**: On the Bridge dataset, FVD = 41.4, which is significantly better than Seer (246.3).
7. **Extremely high data efficiency**: Using only 10% of Calvin data is sufficient to outperform all baselines trained on full data.

## Highlights & Insights

- **Elegant Core Innovation**: Redefines the video diffusion model from being a mere "video generator" to a "predictive visual encoder." A single forward pass is sufficient to extract representations containing future developments, perfectly balancing representation quality and inference efficiency.
- **Clear Generalization Mechanism**: (1) The video model learns reasonable predictions for unseen scenes through internet-scale pre-training; (2) the low-level policy only needs to implicitly track robot motion without having to focus on specific objects or backgrounds.
- **Paradigm Shift from "Generation" to "Representation"**: Unlike SuSIE/UniPi, which require complete denoising to obtain images, VPP directly utilizes the intermediate representations, avoiding informational loss and high latency.
- **Reasonable Multi-source Data Fusion Strategy**: Balances heterogeneous data through sampling ratios (30% human manipulation, 15% RT-1, 15% Bridge, etc.).
- **Impressive Dexterous Hand Tool-Use Experiments**: On a 12-DoF dexterous hand, VPP achieves an average success rate of 68% for tasks involving a spoon, hammer, drill, and pipette, whereas baselines fail almost completely.

## Limitations & Future Work

1. **Large Computational Overhead**: The TVP model has 1.5B parameters, and a single forward pass takes ~140ms, limiting its deployment on low-compute platforms.
2. **Strong Dependence on Video Pre-training**: Performance plummets to 1.63 when SVD pre-training is removed.
3. **More Advanced Video Models Unexplored**: Only SVD 1.5B is used, and stronger models like CogVideoX have not been attempted.
4. **Lack of Theoretical Analysis for Single-step Forward Pass**: Why is one step sufficient? Only visualization verification is provided.
5. **Action Space Alignment Still Requires Demonstration Data**: Each new platform requires demonstration data.

## Related Work & Insights

- **Compared with GR-1**: GR-1 generates frame-by-frame autoregressively. VPP produces multi-frame predictive representations at once, offering higher quality without requiring training of the decoder part.
- **Compared with SuSIE/UniPi**: They require complete denoising to generate target images, whereas VPP directly utilizes intermediate representations, resulting in an order of magnitude faster inference.
- **Compared with Vidman**: Vidman also uses video diffusion representations but does not fine-tune the video model. The domain fine-tuning in VPP brings significant improvements.
- **Insights**: Intermediate representations of generative models can be more valuable than their final outputs. This concept can be extended to using intermediate representations of LLMs for decision-making, intermediate features of image diffusion for detection, etc.

## Rating

⭐⭐⭐⭐⭐

Well deserved ICML 2025 Spotlight. It presents a simple yet powerful paradigm: utilizing the video diffusion model as a predictive visual encoder to acquire representations containing future dynamics in a single forward pass. The experiments cover 2 simulation platforms and 2 real-world platforms, achieving substantial leads across all configurations. The core idea possesses strong generalizability and is highly insightful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Failure Prediction at Runtime for Generative Robot Policies](../../NeurIPS2025/image_generation/failure_prediction_at_runtime_for_generative_robot_policies.md)
- [\[ICML 2025\] Theoretical Guarantees on the Best-of-n Alignment Policy](theoretical_guarantees_on_the_best-of-n_alignment_policy.md)
- [\[NeurIPS 2025\] Flattening Hierarchies with Policy Bootstrapping](../../NeurIPS2025/image_generation/flattening_hierarchies_with_policy_bootstrapping.md)
- [\[ICML 2025\] Discriminative Policy Optimization for Token-Level Reward Models](discriminative_policy_optimization_for_token-level_reward_models.md)
- [\[CVPR 2026\] Seeing What Matters: Visual Preference Policy Optimization for Visual Generation](../../CVPR2026/image_generation/seeing_what_matters_visual_preference_policy_optimization_for_visual_generation.md)

</div>

<!-- RELATED:END -->
