---
title: >-
  [Paper Note] FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait
description: >-
  [ICCV 2025][Image Generation][Flow Matching] This paper proposes FLOAT, an audio-driven talking portrait generation method based on Flow Matching, which employs a Transformer architecture to predict vector fields in an orthogonal motion latent space. The approach enables efficient (~10-step sampling), temporally consistent, high-quality talking video generation, with additional support for speech-driven emotion enhancement and test-time head pose editing.
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Flow Matching"
  - "Talking Portrait Generation"
  - "Motion Latent Space"
  - "Orthogonal Basis"
  - "Speech Emotion Enhancement"
date: 2026-05-08
content_hash: c5a3b0119efee673
---

# FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait

**Conference**: ICCV 2025
**arXiv**: [2412.01064](https://arxiv.org/abs/2412.01064)  
**Code**: [GitHub](https://deepbrainai-research.github.io/float/)  
**Area**: Image Generation
**Keywords**: Flow Matching, Talking Portrait Generation, Motion Latent Space, Orthogonal Basis, Speech Emotion Enhancement

## TL;DR

This paper proposes FLOAT, an audio-driven talking portrait generation method based on Flow Matching, which employs a Transformer architecture to predict vector fields in an orthogonal motion latent space. The approach enables efficient (~10-step sampling), temporally consistent, high-quality talking video generation, with additional support for speech-driven emotion enhancement and test-time head pose editing.

## Background & Motivation

Audio-driven talking portrait generation aims to synthesize natural talking videos from a single portrait image and driving audio, requiring accurate lip synchronization, rhythmic head motion, and fine-grained expression variation. Existing methods face the following challenges:

**Efficiency bottleneck of diffusion models**: Methods such as EMO perform pixel-level video generation based on Stable Diffusion. While achieving high quality, they require tens of minutes to generate a few seconds of video, and their iterative sampling nature leads to inter-frame flickering and temporal inconsistency.

**Excessive reliance on auxiliary facial priors**: Many methods require 2D keypoints, 3D meshes, or bounding boxes as intermediate representations. These strong spatial priors constrain the diversity and fidelity of head motion.

**Limitations of pixel space**: Directly generating video frames in pixel/VAE latent space lacks explicit motion modeling, making temporal consistency difficult to guarantee.

**Absence of emotion control**: Most existing methods do not support emotion-aware motion generation, despite the fact that speakers naturally convey emotion through prosody.

The core insights of this paper are:
- **Motion latent space outperforms pixel latent space**: Sampling in a learned orthogonal motion latent space naturally ensures temporal consistency.
- **Flow matching outperforms diffusion**: Transport along straight paths at constant velocity substantially reduces the number of sampling steps (10 vs. 50).
- **Orthogonal basis structure**: Renders motion editable, enabling direct manipulation of head pose at test time.

## Method

### Overall Architecture

FLOAT consists of two stages:
1. **Motion latent space autoencoder (pre-trained)**: Learns to encode images into an orthogonal decomposition of identity latent variables and motion latent variables.
2. **Flow matching motion sampler**: Uses a Transformer to predict the vector field from Gaussian noise to target motion latent variables, conditioned on audio, emotion, and reference motion.

At inference: sample motion latent variables → add identity latent variables → decode into video frames.

### Key Designs

1. **Orthogonal motion latent space**: Following the LIA (Latent Image Animator) architecture, a source image $S$ is encoded as:
$$w_S = w_{S \to r} + w_{r \to S}$$
where $w_{S \to r} \in \mathbb{R}^d$ is the identity latent variable and $w_{r \to S} = \sum_{m=1}^M \lambda_m(S) \cdot \mathbf{v}_m$ is the motion latent variable. $V = \{\mathbf{v}_m\}_{m=1}^M$ denotes a learned orthogonal basis ($M=20$ directions, $d=512$ dimensions), and $\lambda_m(S)$ are source-dependent motion coefficients.

The advantage of orthogonality: at test time, the coefficient of any motion direction can be extracted via inner product $\lambda_k(\hat{D}) = \langle w_{r \to \hat{D}}, \mathbf{v}_k \rangle$, enabling independent editing and re-composition.

2. **Facial component-aware perceptual loss $\mathcal{L}_{comp-lp}$**: At high resolution, fine-grained details in small regions such as teeth and eyeballs are easily overwhelmed by large-scale dynamics. A perceptual loss targeting facial components is proposed, significantly improving the fidelity of subtle motions such as teeth and eye movements, without relying on pre-trained models such as SD.

3. **Transformer-based flow matching vector field predictor**: Built upon the DiT architecture with a key modification — **frame-wise conditional modulation** (Frame-wise AdaLN):

    - Each frame $l$'s input is modulated by its corresponding frame-$l$ condition via AdaLN:
    $\gamma_i^l \times \text{LN}(X_t^l) + \beta_i^l$
    - The modulated features are passed through masked self-attention layers to model temporal relationships across adjacent $2T$ frames.
    - This decouples "condition injection" from "temporal modeling," outperforming the approach of using cross-attention to handle both simultaneously.

4. **OT-based flow matching**: Optimal transport (OT) straight paths connect Gaussian noise $x_0$ and target motion latent variables $w_{r \to D^{1:L}}$:
$$\mathcal{L}_{OT}(\theta) = \|v_t((1-t)x_0 + t \cdot w_{r \to D^{1:L}}; \theta) - (w_{r \to D^{1:L}} - x_0)\|^2$$

Straight paths with constant-velocity transport allow high-quality motion generation with as few as ~10 ODE steps.

5. **Speech-driven emotion enhancement**: A pre-trained speech emotion predictor is used to extract softmax probabilities $w_e \in \mathbb{R}^7$ over 7 emotion categories (angry, disgust, fear, happy, neutral, sad, surprise), which are fed as additional conditions into the vector field predictor.

At inference, **incremental CFV (Classifier-Free Vector field)** is applied to independently control the guidance strength of audio and emotion:
$$\tilde{v}_t \approx v_t(\cdot|_{\{a, w_e\}}) + \gamma_a[v_t(\cdot|_{w_e}) - v_t(\cdot|_{\{a, w_e\}})] + \gamma_e[v_t(\cdot) - v_t(\cdot|_{w_e})]$$

Emotion retargeting is supported: the ambiguous speech emotion can be replaced with an explicit one-hot encoding.

### Loss & Training

Overall objective:
$$\mathcal{L}_{total}(\theta) = \lambda_{OT} \mathcal{L}_{OT}(\theta) + \lambda_{vel} \mathcal{L}_{vel}(\theta)$$

- $\mathcal{L}_{OT}$: Primary OT flow matching loss
- $\mathcal{L}_{vel} = \|\Delta v_t - \Delta u_t\|$: Velocity consistency loss promoting temporal smoothness
- $\lambda_{OT} = \lambda_{vel} = 1$
- Driving conditions: Wav2Vec2.0 audio features + speech emotion + source motion latent variables + flow timestep embedding
- Dropout applied to $w_r$, $w_e$, and $a^{1:L}$ each with probability 0.1 (for CFV)
- Window length $L=50$ frames (2.4 seconds), with $L'=10$ preceding frames for smooth transition
- Adam optimizer, batch size 8, lr $10^{-5}$, trained on a single A100 for approximately 2 days

## Key Experimental Results

### Main Results

Quantitative comparison on HDTF / RAVDESS datasets:

| Method | FID↓ | FVD↓ | CSIM↑ | E-FID↓ | P-FID↓ | LSE-D↓ | LSE-C↑ |
|--------|------|------|-------|--------|--------|--------|--------|
| SadTalker | 71.95/119.43 | 339.06/376.29 | 0.644/0.644 | 1.914/3.500 | 1.456/2.045 | 7.947/7.273 | 7.305/4.748 |
| Hallo | 25.36/57.65 | 197.20/375.56 | **0.869/0.860** | 1.039/2.492 | 0.037/0.050 | 7.792/7.613 | 7.582/4.795 |
| EchoMimic | 33.55/81.84 | 296.76/320.22 | 0.823/0.805 | 1.234/3.201 | **0.023/0.047** | 8.903/8.161 | 6.242/4.144 |
| **FLOAT** | **21.10/31.68** | **162.05/166.36** | 0.843/0.810 | **1.229/1.367** | 0.032/0.031 | **7.290/6.994** | **8.222/5.730** |

FLOAT achieves the best performance on most metrics, with particularly notable advantages in video quality (FID/FVD) and lip synchronization (LSE-D/LSE-C).

### Ablation Study

| Configuration | FID↓ | FVD↓ | E-FID↓ | LSE-D↓ | NFEs↓ |
|---------------|------|------|--------|--------|-------|
| Cross-Attention (replacing AdaLN) | 21.87 | 162.70 | 1.452 | 7.757 | 10 |
| Diffusion ($\epsilon$-pred) | 21.19 | 161.67 | 1.213 | 9.922 | 50 |
| Diffusion ($x_0$-pred) | 21.70 | 162.85 | 1.278 | 9.048 | 50 |
| **FLOAT (Flow Matching)** | **21.10** | **162.05** | **1.229** | **7.290** | **10** |

Effect of additional conditions:

| Configuration | FID↓ | FVD↓ | E-FID↓ | P-FID↓ |
|---------------|------|------|--------|--------|
| FLOAT (base) | 21.10/31.68 | 162.05/166.36 | 1.229/1.367 | 0.032/0.031 |
| + 3DPose | **19.72/29.72** | **126.66/112.89** | **0.926/1.152** | **0.012/0.016** |
| − S2E (w/o speech emotion) | 21.24/32.04 | 155.03/166.87 | 1.254/1.502 | 0.031/0.025 |

### Key Findings

- **Flow matching vs. diffusion**: Under the same architecture, flow matching with only 10 NFEs matches or surpasses diffusion models at 50 NFEs; LSE-D drops substantially from 9.0+ to 7.29.
- **Frame-wise AdaLN vs. Cross-Attention**: The condition–temporal decoupling design of AdaLN clearly outperforms Cross-Attention on expression generation (E-FID) and lip synchronization (LSE-D).
- **Contribution of speech emotion**: Removing S2E leads to mild degradation in E-FID and P-FID, confirming that speech emotion provides valuable motion priors.
- **3DPose conditioning**: Adding head pose parameters significantly improves all metrics, demonstrating the scalability of the framework.
- **Efficiency advantage**: FLOAT's forward FPS substantially exceeds that of diffusion-based methods (~5×).

## Highlights & Insights

- **Choice of motion latent space**: Abandoning the SD pixel VAE latent space in favor of a motion-semantic latent space (LIA) yields a natural advantage in temporal consistency.
- **Editability of the orthogonal basis**: $\lambda$-control enables independent adjustment of motion dimensions such as head orientation at test time, without additional training.
- **Efficiency of flow matching**: OT-based straight paths allow high-quality results with only 10 sampling steps.
- **Incremental CFV**: Independently controlling the guidance strength of audio and emotion enables fine-grained generation control.
- **Natural chain of speech → emotion → motion**: Emotion labels need not be provided by the user; they are automatically extracted from speech.

## Limitations & Future Work

- The orthogonal basis dimensionality $M=20$ may be insufficient to capture all fine-grained motion patterns.
- Emotion recognition relies on an external pre-trained model, whose accuracy directly affects motion quality.
- Training data covers only approximately 250 identities, limiting generalization to rare ethnicities or extreme expressions.
- The LIA decoder may produce artifacts under large head rotations.
- The model operates at $512\times512$ resolution; scaling to higher resolutions requires retraining.
- Training and evaluation are conducted exclusively on English speech data.

## Related Work & Insights

- **VASA-1** [Xu et al., 2024]: Also employs a motion latent space, but FLOAT's orthogonal structure renders motion editable.
- **EMO** [Tian et al., 2024]: SD-based pixel-level generation achieves high quality but at low efficiency.
- **LIA** [Wang et al., 2022]: Foundational motion autoencoder architecture; this work introduces orthogonality constraints and facial component losses on top of it.
- **DiT** [Peebles & Xie, 2023]: Source of inspiration for the AdaLN conditional injection mechanism.
- Insight: The choice of latent space for a generative model matters more than the model architecture; orthogonal structure naturally confers editability.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of flow matching, orthogonal motion latent space, and speech emotion is novel and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive comparisons, thorough ablations, and rich application demonstrations (pose editing / emotion retargeting / multi-condition control).
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear and architecture diagrams are highly informative.
- **Value**: ⭐⭐⭐⭐⭐ Achieves the best balance between efficiency and quality; orthogonal basis editability is a distinctive selling point.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](contrastive_flow_matching.md)
- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[ICCV 2025\] Deeply Supervised Flow-Based Generative Models](deeply_supervised_flow-based_generative_models.md)
- [\[ICML 2026\] From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection](../../ICML2026/image_generation/from_talking_to_singing_a_new_challenge_for_audio-visual_deepfake_detection.md)
- [\[ICCV 2025\] Domain Generalizable Portrait Style Transfer](domain_generalizable_portrait_style_transfer.md)

</div>

<!-- RELATED:END -->
