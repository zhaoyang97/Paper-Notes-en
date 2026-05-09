---
title: >-
  [Paper Note] Less is More: Improving Motion Diffusion Models with Sparse Keyframes
description: >-
  [Image Generation] This paper proposes sMDM, a motion diffusion framework centered on sparse keyframes. By introducing a masking-interpolation strategy and the Visvalingam-Whyatt keyframe selection algorithm, sMDM reduces redundant frame processing and consistently outperforms dense-frame baselines in text alignment and motion quality.
tags:
  - Image Generation
date: 2026-05-08
content_hash: 56dd70296a107f98
---

# Less is More: Improving Motion Diffusion Models with Sparse Keyframes

> **Conference**: ICCV 2025
> **arXiv**: [2503.13859](https://arxiv.org/abs/2503.13859)
> **Area**: Motion Generation · Diffusion Models · Keyframes
> **Keywords**: motion diffusion, sparse keyframes, masking-interpolation, Visvalingam-Whyatt, Lipschitz MLP

## TL;DR

This paper proposes sMDM, a motion diffusion framework centered on sparse keyframes. By introducing a masking-interpolation strategy and the Visvalingam-Whyatt keyframe selection algorithm, sMDM reduces redundant frame processing and consistently outperforms dense-frame baselines in text alignment and motion quality.

## Background & Motivation

Existing motion diffusion models (e.g., MDM) take dense frame sequences as input and output, and face two core issues:

**High complexity in training and inference**: The computational cost of self-attention layers grows quadratically with the number of frames, making training on large-scale motion datasets expensive. In practice, motion diffusion models suffer severe quality degradation under low diffusion step settings.

**Lack of controllability**: Generating all frames simultaneously makes it difficult to interpret or control model behavior. In contrast, professional animators define sparse keyframes first and then interpolate intermediate frames.

**Core Motivation**: Inspired by professional animation workflows—where animators focus on keyframes rather than every frame—sMDM builds a diffusion framework around sparse, geometrically meaningful keyframes.

## Method

### Overall Architecture (Fig. 2)

sMDM introduces three core modifications to the Transformer backbone of MDM:

1. **Masking**: A binary keyframe mask $\mathbf{M}$ suppresses non-keyframe tokens in self-attention, reducing attention complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(K^2)$ ($K \ll N$).
2. **Interpolation**: After self-attention, keyframe features are linearly interpolated to reconstruct non-keyframe features.
3. **Lipschitz MLP**: Input/output linear layers are replaced with Lipschitz MLPs to ensure smooth interpolation.

### Key Design 1: Keyframe Selection

**During training**: The Visvalingam-Whyatt geometric simplification algorithm automatically identifies geometrically significant keyframes. It iteratively removes the frame with the smallest "area" (i.e., lowest geometric importance) until the target keyframe count is reached. A compression rate of 80% is used (retaining 20% of frames as keyframes).

**During inference (two-stage strategy)**:
- **Early stage ($t > T' = \gamma \cdot T$)**: A uniform mask is used (keyframes evenly distributed), providing a stable denoising baseline.
- **Late stage ($t \leq T'$)**: **Dynamic mask update** — Visvalingam-Whyatt is re-applied to the intermediate frames $\mathbf{x}_t$ to reselect keyframes, focusing attention on the most critical frames. $\gamma = 0.1$ yields the best results.

### Key Design 2: Lipschitz MLP

To ensure that linear interpolation in feature space produces smooth motion outputs, a Lipschitz constraint is imposed on the input/output MLPs:

$$\|g_\theta(y_1) - g_\theta(y_2)\|_p \leq \alpha \|y_1 - y_2\|_p$$

Sine activations are used in place of ReLU to better capture high-frequency motion details (e.g., abrupt direction changes).

### Loss & Training

Standard diffusion reconstruction loss with Lipschitz regularization:

$$\mathcal{L} = \|\mathbf{x}_0 - \hat{\mathbf{x}}_0\|^2 + \lambda \mathcal{L}_{lip}$$

where $\hat{\mathbf{x}}_0 = f_\theta(x_t, t, c)$. The loss is computed over **interpolated dense frames** rather than keyframes alone.

## Key Experimental Results

### Main Results: Text-to-Motion Generation (Tab. 1, HumanML3D)

| Method | R@1 ↑ | R@3 ↑ | FID ↓ | MM-Dist ↓ |
|--------|-------|-------|-------|-----------|
| MDM | 0.320 | 0.611 | 0.544 | 5.566 |
| MLD | 0.481 | 0.772 | 0.473 | 3.196 |
| T2M-GPT | 0.606 | 0.838 | 12.475 | 16.812 |
| MoMask | 0.621 | 0.846 | 12.232 | 16.138 |
| ReMoDiffuse§ | 0.510 | 0.795 | 0.103 | 2.974 |
| **sMDM** | 0.494 | 0.776 | **0.130** | **3.051** |
| **sMDM-stella†** | **0.554** | **0.829** | **0.151** | **2.740** |

Using the same CLIP encoder, sMDM substantially outperforms MDM. With the Stella-1.5B encoder (sMDM-stella), it surpasses MotionGPT, MotionLCM, and MotionBase—all of which use larger encoders—across all metrics.

### Ablation Study (Tab. 1, lower half)

| Configuration | FID ↓ | R@3 ↑ | MM-Dist ↓ |
|---------------|-------|-------|-----------|
| Random keyframes | 0.249 | 0.777 | 3.086 |
| No interpolation (keyframe loss only) | 0.267 | 0.773 | 3.127 |
| No Lipschitz | 0.329 | 0.768 | 3.149 |
| **sMDM (full)** | **0.130** | **0.776** | **3.051** |

Key findings:
- Geometric keyframe selection substantially outperforms random selection (FID 0.130 vs. 0.249).
- Computing loss over interpolated dense frames is more effective than computing it over keyframes alone.
- The Lipschitz constraint contributes significantly to FID improvement.

### Stability Across Diffusion Steps (Tab. 2)

| Steps | Standard FID | + Dynamic Sampling FID |
|-------|-------------|------------------------|
| 1000 | 0.291 | **0.246** |
| 500 | 0.322 | **0.229** |
| 100 | 0.230 | **0.190** |
| 50 | 0.130 | **0.134** |
| 10 | 0.349 | 0.385 |

Key finding: Dynamic mask updates are especially effective under **large step** settings (FID reduced from 0.322 to 0.229), consistent with observations from spatial guidance methods such as OmniControl.

### Long-Sequence Generation (Tab. 3, DoubleTake)

sPriorMDM generates more expressive motions (EES 2.338 vs. 1.856) with greater fidelity to text instructions, though transition-region FID is slightly higher—likely because the generated motions are more dynamic and such transition patterns are underrepresented in the training set.

## Highlights & Insights

1. **Sparse training paradigm**: This work demonstrates that "less is more"—focusing on 20% of keyframes yields better results than processing all frames, consistent with animation production intuition.
2. **Visvalingam-Whyatt algorithm**: Its adoption endows keyframe selection with geometric meaning, surpassing random or uniform sampling strategies.
3. **Plug-and-play**: sMDM does not alter MDM's core architecture; improvements are achieved solely through masking, interpolation, and Lipschitz constraints, making it straightforward to apply to other diffusion-based motion models.
4. The method maintains strong quality at very low diffusion steps (10 steps), which is practically valuable for real-time applications.

## Limitations & Future Work

- The keyframe compression rate is a fixed hyperparameter; different motion types may require different optimal rates.
- Dynamic mask updates are less effective at extremely low step counts (10 steps).
- Linear interpolation may be insufficient for reconstructing complex, high-frequency motion details.

## Related Work & Insights

- Motion diffusion: MDM, MLD, MotionDiffuse, FlowMDM
- Keyframe-based methods: CondMDI, KEYIN
- Real-time control: DiP, CLoSD, CAMDM

## Rating

- **Novelty**: ★★★★☆ — The idea of training motion diffusion models with sparse keyframes is both novel and practical.
- **Technical Depth**: ★★★★☆ — Each component is concisely designed and effective, with thorough ablations.
- **Experimental Thoroughness**: ★★★★★ — Validated across three downstream tasks with detailed multi-step stability analysis.
- **Writing Quality**: ★★★★☆ — Motivation is clearly articulated; the analogy to animation workflows is intuitive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)

</div>

<!-- RELATED:END -->
