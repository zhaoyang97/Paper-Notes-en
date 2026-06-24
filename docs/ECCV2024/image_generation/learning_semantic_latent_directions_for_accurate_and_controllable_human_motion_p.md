---
title: >-
  [Paper Note] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction
description: >-
  [ECCV 2024][Image Generation] Ours proposes the Semantic Latent Directions (SLD) method. By constructing a set of orthogonal latent base directions and representing future motion hypotheses as a linear combination of these directions, more accurate, diverse, and semantically controllable motion prediction is achieved in stochastic human motion prediction.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 1bcffbfff8ca4d29
---

# Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction

**Conference**: ECCV 2024  
**arXiv**: [2407.11494](https://arxiv.org/abs/2407.11494)  
**Area**: Image Generation

## TL;DR

Ours proposes the Semantic Latent Directions (SLD) method. By constructing a set of orthogonal latent base directions and representing future motion hypotheses as a linear combination of these directions, more accurate, diverse, and semantically controllable motion prediction is achieved in stochastic human motion prediction.

## Background & Motivation

The core challenge of Stochastic Human Motion Prediction (SHMP) is modeling the conditional distribution $p(Y|X)$. Existing methods (such as DLow, STARS, HumanMAC, etc.) typically assume that the latent variable $z$ follows a Gaussian distribution, but this weak constraint leads to:

**Mode Collapse**: Models tend to focus on the major modes of the latent distribution, ignoring minority modes.

**Inaccurate Predictions**: Generating unnatural future actions that are incoherent with historical motion (such as joint distortions).

**Lack of Controllability**: The Gaussian latent space struggles to achieve motion control at the semantic level.

**Key Insight**: The Gaussian prior constraint is insufficient to endow the latent space with meaningful motion semantics. Ours proposes to replace the Gaussian prior with orthogonal base directions, discretizing the latent space into finite semantic prototypes to reduce learning difficulty.

## Method

### Overall Architecture

The SLD module is embedded into a simple encoder-decoder framework:
1. **Encoder**: Extracts frequency-domain features of past motions (via DCT transform).
2. **SLD Module**: Projects motion queries to the coefficients of latent directions and calculates the semantic embedding.
3. **Decoder**: Generates future motion by combining past motion features with semantic embeddings (via IDCT inverse transform).

### Key Designs

**Semantic Latent Directions (SLD)**: We define $M$ learnable latent directions $D = [d_1, ..., d_M] \in \mathbb{R}^{M \times C}$, constrained to be mutually orthogonal (achieved via SVD decomposition). The latent factor of future motion is represented as:

$$z = \sum_{m=1}^{M} w_m \cdot d_m$$

where $w = [w_1, ..., w_M]$ represents the coefficients predicted from the past motion.

**Design Advantages**:
- The orthogonal constraint allows different directions to capture distinct motion semantics (e.g., direction 1 = stand $\leftrightarrow$ sit, direction 2 = arm swing range), achieving implicit disentanglement.
- Discretizing the latent space into finite prototypes forces all predictions to align with the prototypes, avoiding abnormal predictions.
- During inference, semantically controllable predictions are achieved by editing the coefficients.

**Diverse Motion Queries**: We introduce $K$ learnable motion queries $Q = [q_1, ..., q_K]$. Each query, along with the past motion, is mapped to a different coefficient combination $w_m^k$ via a QLP (Query to Latent Projection) network, thereby producing $K$ diverse predictions. A key improvement is **projecting queries into the SLD space** (instead of direct concatenation), ensuring accuracy during diverse sampling.

**QLP Network**: Implemented with 3 layers of STGCN + 3 layers of MLP, which maps motion queries and past motion encodings to $M$ coefficients.

### Loss & Training

$$\mathcal{L} = \lambda_r \mathcal{L}_r + \lambda_d \mathcal{L}_d + \lambda_c \mathcal{L}_c$$

- $\mathcal{L}_r$: Reconstruction loss, representing the distance between the ground truth and the closest of the $K$ predictions.
- $\mathcal{L}_d$: Diversity-promoting loss, representing the pairwise distance among the $K$ predictions.
- $\mathcal{L}_c$: Motion constraint loss, ensuring the physical plausibility of the predicted motions.

The model is trained end-to-end in a single stage, where the SLD is automatically learned along with the encoder and decoder.

## Key Experimental Results

### Main Results

Comparison on Human3.6M and HumanEva-I datasets ($K=50$):

| Method | APD ↑ (H3.6M) | ADE ↓ (H3.6M) | FDE ↓ (H3.6M) | MMADE ↓ (H3.6M) | MMFDE ↓ (H3.6M) |
|---|---|---|---|---|---|
| DLow | 11.741 | 0.425 | 0.518 | 0.495 | 0.531 |
| GSPS | 14.757 | 0.389 | 0.496 | 0.476 | 0.525 |
| STARS | 15.884 | 0.358 | 0.445 | 0.442 | 0.471 |
| HumanMAC | 6.301 | 0.369 | 0.480 | 0.509 | 0.545 |
| Belfusion | 7.602 | 0.372 | 0.474 | 0.473 | 0.507 |
| MotionDiff | 15.353 | 0.411 | 0.509 | 0.508 | 0.536 |
| **SLD (Ours)** | 8.741 | **0.348** | **0.436** | **0.435** | **0.463** |

HumanEva-I dataset (observe 15 frames, predict 60 frames):

| Method | APD ↑ | ADE ↓ | FDE ↓ | MMADE ↓ | MMFDE ↓ |
|---|---|---|---|---|---|
| DLow | 4.855 | 0.251 | 0.268 | 0.362 | 0.339 |
| DivSamp | 6.109 | 0.220 | 0.234 | 0.342 | 0.316 |
| STARS | 6.031 | 0.217 | 0.241 | 0.328 | 0.321 |
| HumanMAC | 6.554 | 0.209 | 0.223 | 0.342 | 0.320 |
| **SLD (Ours)** | 4.066 | **0.193** | **0.209** | **0.305** | **0.293** |

### Ablation Study

Contributions of each component:

| Configuration | APD ↑ (Eva) | ADE ↓ (Eva) | FDE ↓ (Eva) | APD ↑ (H3.6M) | ADE ↓ (H3.6M) | FDE ↓ (H3.6M) |
|---|---|---|---|---|---|---|
| MQ (Motion queries only, w/o SLD) | 1.562 | 0.219 | 0.248 | 7.286 | 0.361 | 0.449 |
| MQ+SLD (w/o projection) | 3.365 | 0.202 | 0.218 | 7.936 | 0.352 | 0.442 |
| **MQ-P+SLD (Full model)** | **4.066** | **0.193** | **0.209** | **8.741** | **0.348** | **0.436** |

### Key Findings

- SLD outperforms the state-of-the-art across **all accuracy metrics**, including ADE, FDE, MMADE, and MMFDE.
- Although the APD (diversity) is lower than that of methods like STARS, the high diversity of the latter is accompanied by a large number of unnatural predictions (with noticeable joint abnormalities in the visualizations).
- The ablation study shows a step-by-step improvement of the three components: Motion queries without SLD $\rightarrow$ adding SLD $\rightarrow$ SLD projection, where each step brings improvements in both accuracy and diversity.
- The learned latent directions automatically encode motion semantics (such as stand $\leftrightarrow$ sit, arm directions), achieving the first-ever controllable motion prediction at the **latent semantic level**.
- Lightweight and efficient: Trained on a single RTX 3090 GPU, taking 25 hours on Human3.6M and 7 hours on HumanEva-I.

## Highlights & Insights

1. **Elegant Design of Orthogonal Bases**: Discretizing the continuous Gaussian latent space into finite orthogonal prototypes simultaneously reduces the learning difficulty and the probability of abnormal predictions.
2. **Automatic Semantic Emergence**: Without explicit semantic annotations, meaningful motion semantics are automatically learned by the orthogonal directions during the training process.
3. **General Plug-and-Play**: SLD can be seamlessly integrated into existing frameworks as an information bottleneck.
4. **Natural Acquisition of Controllability**: The controllability brought by the linear combination representation—editing the coefficients corresponds to editing the motion semantics.
5. **Interesting Comparison with Belfusion**: Belfusion utilizes explicit behavioral representations but performs worse than the implicit semantic directions of SLD, which suggests that letting the model learn autonomously is better than imposing hard constraints.

## Limitations & Future Work

- Currently, the method is only validated on controlled laboratory datasets (Human3.6M, HumanEva-I) and does not involve dynamic scenarios of human-environment interaction.
- The diversity metric (APD) is not optimal, indicating that the number of orthogonal bases $M$ may limit the range of explorable motion patterns.
- The "Image Generation" area label in the header may not be entirely accurate—the core of this work is 3D human motion prediction/generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The concept of orthogonal semantic directions is novel, and the approach of using linear combinations to achieve controllable prediction is intuitive and elegant.
- **Practicality**: ⭐⭐⭐⭐ — Lightweight, integrable, and achieves the first semantic-level controllable motion prediction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 12 baselines, 2 datasets, comprehensive ablations + visualization analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — The method is simple yet effective, the experiments are solid, and the paper is well-structured.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [\[ECCV 2024\] MotionChain: Conversational Motion Controllers via Multimodal Prompts](motionchain_conversational_motion_controllers_via_multimodal_prompts.md)

</div>

<!-- RELATED:END -->
