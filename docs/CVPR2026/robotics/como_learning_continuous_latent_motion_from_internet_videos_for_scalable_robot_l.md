---
title: >-
  [Paper Note] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning
description: >-
  [CVPR 2026][Robotics][continuous latent motion] CoMo is proposed to jointly address the shortcut learning problem in continuous latent motion learning via two mechanisms — Early Temporal Difference (Td) and Temporal Cont…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "continuous latent motion"
  - "pseudo-action labels"
  - "inverse dynamics model"
  - "temporal contrastive learning"
  - "video-robot joint training"
date: 2026-05-08
content_hash: 7af77a65a96c5828
---

# CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning

**Conference**: CVPR 2026
**arXiv**: [2505.17006](https://arxiv.org/abs/2505.17006)  
**Code**: [github.com/MCG-NJU/CoMo](https://github.com/MCG-NJU/CoMo)  
**Area**: Robot Learning
**Keywords**: continuous latent motion, pseudo-action labels, inverse dynamics model, temporal contrastive learning, video-robot joint training

## TL;DR

CoMo is proposed to jointly address the shortcut learning problem in continuous latent motion learning via two mechanisms — Early Temporal Difference (Td) and Temporal Contrastive Learning (Tcl) — enabling the extraction of fine-grained continuous pseudo-action labels from internet videos and joint training of video data and robot actions under a unified continuous distribution, substantially improving policy performance.

## Background & Motivation

Learning latent motion from internet videos is a key direction for scaling robot learning, yet existing methods face fundamental bottlenecks:

**Information loss from discretization**: Pioneer works such as LAPA, UniVLA, and Moto-GPT discretize latent motion via VQ-VAE (codebook size of only 8–16). While this effectively suppresses shortcut learning, discretization causes significant loss of motion information, making it difficult to capture complex and fine-grained dynamics.

**Shortcut learning problem**: When continuous latent motion is learned directly (without VQ), the inverse dynamics encoder (IDM) tends to extract abundant static background information from future frames rather than foreground motion, since the decoder can more easily reconstruct pixel-level details. The model degenerates into an ineffective future-frame predictor.

**Distribution mismatch between discrete and continuous representations**: A fundamental distributional gap exists between discrete latent motion and continuous robot actions, hindering unified policy learning and necessitating complex multi-stage training pipelines.

Core problem: **Can continuous latent motion be learned from unannotated videos without VQ?** Continuous representations are better suited for modeling fine-grained inter-frame changes and are naturally aligned with the distribution of continuous robot actions.

## Method

### Overall Architecture

CoMo adopts the classical **Inverse Dynamics Model–Forward Dynamics Model (IDM-FDM)** architecture:
- **IDM**: Given the current frame $O_t$ and a future frame $O_{t+n}$, extracts continuous latent motion $Z_{t,t+n}$.
- **FDM**: Conditioned on $Z_{t,t+n}$ and $O_t$, reconstructs the future frame $\hat{O}_{t+n}$.

CoMo introduces Td and Tcl as replacements for VQ to enable continuous latent motion learning.

### Key Designs

1. **Early Temporal Difference (Td)**: Inspired by temporal difference networks in video understanding, two core modifications are made:

    - **Feature differencing to enhance motion cues**: A shared MAE-pretrained ViT extracts token-level features $F_t, F_{t+n}$ from the current and future frames; the element-wise difference $D_t = F_{t+n} - F_t$ is computed, where sparse feature differencing explicitly amplifies motion signals.
    - **Removal of future-frame features**: $[F_t, D_t]$ is concatenated (rather than $[F_t, F_{t+n}]$) and fed into the Motion Q-former, fundamentally increasing the difficulty of shortcut learning — the encoder can no longer directly access the complete representation of the future frame.

2. **Temporal Contrastive Learning (Tcl)**: Sparse differencing from Td alone may retain irrelevant information, and contrastive learning alone may capture only foreground object identity ("what" and "where") while ignoring motion patterns ("how"). CoMo designs specialized positive and negative sample pairs:

    - **Positive pairs**: $Z_{t,t+n+\delta}$ and $Z_{t,t+n}$ — motion pairs with slight temporal offsets ($\delta \in [-n/5, n/5]$).
    - **Negative pairs**: $Z_{t+n,t}$ and $Z_{t,t+n}$ — motion pairs with reversed temporal direction.
    - The loss function is InfoNCE:
    $\mathcal{L}_{\text{tcl}} = -\log\frac{e^{S_1}}{e^{S_1} + e^{S_2} + e^{S_3}}$

   Td ensures enhanced motion cues; Tcl ensures foreground focus. The two mechanisms operate synergistically.

3. **Joint unified policy learning**: A key advantage — continuous latent motion and continuous robot actions share the same continuous distribution, enabling direct joint training within a unified policy model (with only distinct lightweight heads assigned), without requiring multi-stage pretraining or an explicit two-stage motion-before-action pipeline.

4. **Evaluation metric design**: Two latent motion analysis metrics are proposed that do not require policy evaluation:

    - **MSE**: An MLP is trained to regress true actions from latent motion, evaluating the encoding capacity for action-relevant information.
    - **S-PCFC**: The cosine similarity between past-to-current and future-to-current motion, diagnosing action-irrelevant background noise.

### Loss & Training

- CoMo jointly minimizes a weighted combination of InfoNCE loss, pixel-level reconstruction loss, and perceptual loss.
- The IDM-FDM is trained on 120K internet videos (SAM-V 40K + EgoVid 40K + Droid 40K).
- Policy training supports both diffusion-based (LIBERO) and autoregressive (CALVIN) architectures.

## Key Experimental Results

### Main Results

LIBERO benchmark (only 10 robot trajectories per task + video pseudo-labels):

| Method | Spatial | Object | Goal | Long | Avg. SR |
|--------|---------|--------|------|------|---------|
| DP (no video) | 72.3 | 82.3 | 70.3 | 56.7 | 70.4 |
| GR2-like (future-frame features) | 76.0 | 92.0 | 73.3 | 53.7 | 73.8 |
| GR00T (discrete latent motion) | 80.7 | 83.3 | 80.0 | 59.7 | 75.9 |
| Dynamo (covariance regularization) | 75.3 | 92.7 | 80.7 | 46.0 | 73.7 |
| **CoMo** | **80.3** | **97.0** | **81.0** | **62.0** | **80.1** |

CALVIN ABC→D benchmark:

| Method | 1-step | 2-step | 3-step | 4-step | 5-step | Avg. |
|--------|--------|--------|--------|--------|--------|------|
| No motion | 0.772 | 0.494 | 0.307 | 0.191 | 0.114 | 1.878 |
| Discrete (Moto) | 0.801 | 0.575 | 0.409 | 0.283 | 0.187 | 2.255 |
| **CoMo** | **0.882** | **0.732** | **0.589** | **0.490** | **0.377** | **3.070** |
| CoMo ×4 dims | 0.891 | 0.758 | 0.646 | 0.529 | 0.423 | 3.247 |

### Ablation Study

| Configuration | Avg. SR | MSE↓ | S-PCFC↓ | Notes |
|---------------|---------|------|---------|-------|
| Future-frame features | 73.8 | 2.14 | 1.000 | Heavy background noise |
| Discrete (Dis.) | 75.9 | 5.67 | 0.481 | Severe information loss |
| Continuous baseline (Con.) | 75.2 | 1.63 | 0.927 | Shortcut learning issue |
| Con.+Td | 76.9 | 1.52 | 0.889 | Td reduces S-PCFC |
| Con.+Tcl | 77.9 | 1.31 | 0.623 | Tcl substantially reduces S-PCFC |
| **Con.+Td+Tcl (CoMo)** | **80.1** | **1.26** | **0.550** | Synergistic optimum |

### Key Findings

- Td and Tcl are individually effective yet complementary: Td primarily reduces MSE (enhancing motion cues), while Tcl primarily reduces S-PCFC (suppressing background noise).
- Continuous representations outperform discrete ones: CoMo achieves 80.1% average SR vs. 75.9% for discrete, with particularly notable gains on fine-grained manipulation tasks (Object: 97.0% vs. 83.3%).
- Dimensionality scalability: CoMo continuously improves as dimensions scale from 128 to 512 (CALVIN: 3.070→3.247), whereas Td alone introduces more noise as dimensions increase.
- Real-world experiments also validate the advantage of continuous latent motion, especially for low-tolerance fine-grained manipulation tasks (opening drawers, inserting bread).

## Highlights & Insights

- The case for **continuous over discrete** is thorough and compelling: discretization suppresses shortcut learning but at too high a cost.
- The **synergistic design of Td + Tcl** is elegant: Td addresses "where to learn from" (removing direct future-frame information), while Tcl addresses "what to learn" (positive/negative pairs guiding foreground focus and motion directionality).
- The **MSE + S-PCFC evaluation framework** provides a rapid diagnostic tool that avoids costly policy evaluation.
- The simplicity of joint training offers significant engineering value: single-stage unified training versus complex multi-stage pipelines.

## Limitations & Future Work

- Validation is limited to LIBERO and CALVIN; generalization to more complex real-world scenarios remains unexplored.
- There is no theoretical guidance on the upper bound or optimal selection of latent motion dimensionality.
- The positive/negative sample construction in Tcl relies on a temporal-offset assumption (range of $\delta$), which may be suboptimal in fast-motion scenarios.
- Real-world experiments are limited in scale (25 trajectories + 25 human-hand videos per task).

## Related Work & Insights

- The core comparison with discrete latent motion methods such as LAPA and GR00T demonstrates that continuous representation is the more promising direction.
- Temporal differencing originates from the video understanding community (TDN); its cross-domain transfer to robot learning proves highly effective.
- The positive/negative sample design for contrastive learning is creative: using temporal reversal as negative samples is naturally suited to distinguishing motion directionality.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The concept of replacing VQ with Td+Tcl is clear and novel, though individual components are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Ablations are highly comprehensive, MSE/S-PCFC analysis is in-depth, and cross-architecture validation is thorough.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear and experimental analysis is rigorous, though some passages are slightly redundant.
- **Value**: ⭐⭐⭐⭐⭐ Provides a practical and principled solution for scaling robot learning from internet videos.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] IGen: Scalable Data Generation for Robot Learning from Open-World Images](igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)
- [\[ICCV 2025\] Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](../../ICCV2025/robotics/moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [\[CVPR 2026\] DAWN: Pixel Motion Diffusion is What We Need for Robot Control](dawn_pixel_motion_diffusion_robot_control.md)
- [\[CVPR 2026\] Chain of World: World Model Thinking in Latent Motion (CoWVLA)](chain_of_world_world_model_thinking_in_latent_motion.md)
- [\[CVPR 2026\] AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots](atomicvla_unlocking_the_potential_of_atomic_skill_learning_in_robots.md)

</div>

<!-- RELATED:END -->
