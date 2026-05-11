---
title: >-
  [Paper Note] GGTalker: Talking Head Synthesis with Generalizable Gaussian Priors and Identity-Specific Adaptation
description: >-
  [ICCV 2025][Human Understanding][Talking head synthesis] GGTalker proposes a prior-adaptation two-stage training strategy that learns generalizable audio-to-expression and expression-to-visual priors from large-scale dat…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "Talking head synthesis"
  - "3D Gaussian splatting"
  - "prior-adaptation"
  - "FLAME"
  - "large-scale pretraining"
date: 2026-05-08
content_hash: 4cfb921d9ff57bd3
---

# GGTalker: Talking Head Synthesis with Generalizable Gaussian Priors and Identity-Specific Adaptation

**Conference**: ICCV 2025
**arXiv**: [2506.21513](https://arxiv.org/abs/2506.21513)
**Code**: N/A
**Area**: Human Understanding
**Keywords**: Talking head synthesis, 3D Gaussian splatting, prior-adaptation, FLAME, large-scale pretraining

## TL;DR

GGTalker proposes a prior-adaptation two-stage training strategy that learns generalizable audio-to-expression and expression-to-visual priors from large-scale datasets, then rapidly adapts to a specific identity. The method achieves state-of-the-art performance across rendering quality, 3D consistency, lip synchronization, and training efficiency, requiring only 20 minutes of adaptation to generate photorealistic talking-head videos at 120 FPS.

## Background & Motivation

Audio-driven 3D talking head synthesis is in high demand for applications such as virtual reality and digital humans. With the advancement of 3D rendering techniques such as NeRF and 3DGS, 3D methods have attracted increasing attention due to their superior identity consistency and rendering speed. However, existing 3D methods suffer from three core issues:

**Insufficient generalization**: These methods only support audio inference similar to the training distribution, and perform poorly on out-of-distribution (OOD) audio, such as cross-speaker or cross-language inputs.

**Failure under large head rotations**: Synthesis of large-angle head movements (e.g., profile views, upward tilts) produces artifacts and holes, as monocular training videos lack sufficient 3D information.

**Low training efficiency**: Per-identity training from scratch requires hours (ER-NeRF: 5h, SyncTalk: 5h, AD-NeRF: 30h), and some methods even require costly multi-view synchronized video capture.

The authors attribute the root cause to: **the lack of sufficient 3D priors in existing methods**. The shape, texture, and audio-to-lip-motion correlations of all human heads follow universal patterns that can be learned from large-scale data and then fine-tuned for specific identities. This prior-adaptation paradigm not only alleviates overfitting but also substantially improves training efficiency.

## Method

### Overall Architecture

GGTalker consists of three components:
1. **Audio-Expression Model**: Generates expression parameter sequences from audio.
2. **Expression-Visual Model**: Predicts a coarse head texture from a single reference image.
3. **Customized Adaptation**: Adapts facial texture and speaking style to a specific identity.

The FLAME parametric model is used as an intermediate representation throughout, with 3D Gaussians bound to the FLAME mesh to enable explicit pose and expression control.

### Key Designs

1. **Audio-Expression Priors**

   A conditional diffusion Transformer is employed to predict expression sequences from audio:

   - **Audio Condition Encoder**: Wav2Vec 2.0 extracts audio features $\mathbf{a}_t \in \mathbb{R}^{1280}$, linearly projected to $d=512$ dimensions, and encoded for temporal dependencies via a shallow Transformer. An identity embedding $\mathbf{I} \in \mathbb{R}^{64}$ is introduced to retrieve speaking style. Outputs include frame-level conditions $\mathbf{C}' \in \mathbb{R}^{T \times d}$ and a global condition $\bar{\mathbf{c}}$.

   - **Diffusion Time Conditioner**: DDPM is used to iteratively refine expression sequences. The diffusion timestep $n$ is converted to a time embedding $\mathbf{t}_n$ via sinusoidal positional encoding and an MLP, and injected into the model via FiLM modulation and tokenization.

   - **Transformer Decoder**: $L=8$ layers, each containing self-attention (to capture temporal dependencies) and cross-attention (to align with audio features). Classifier-free guidance is applied (condition dropout probability $p=0.1$). Output predicted expressions: $\hat{\mathbf{e}}_t = f_\theta(\mathbf{z}_t, \mathbf{C}', \bar{\mathbf{c}}, \mathbf{t}_n)$

   Loss function: $\mathcal{L}_{\text{A2E}} = \lambda_{temp}\mathcal{L}_{temp} + \lambda_{exp}\mathcal{L}_{exp}$, where $\mathcal{L}_{temp}$ is the Huber loss between adjacent frames (temporal smoothness) and $\mathcal{L}_{exp}$ is L2 regularization.

2. **Expression-Visual Priors**

   **Gaussian Binding**: 3D Gaussians are bound to triangles of the FLAME mesh. The center $\mathbf{C}^i$ of each triangle serves as the origin of a local coordinate system, and the triangle's scale $\mathbf{l}^i$ and rotation $\mathbf{R}^i$ determine the global attribute transformation of the associated Gaussian.

   **Identity-Gaussian Generator**: Leveraging the UV layout of the FLAME mesh, this module predicts a UV Gaussian map $M \in \mathbb{R}^{H \times W \times 14}$ from a single reference image, where each pixel corresponds to a 14-dimensional Gaussian parameter. By uniformly sampling the UV map and placing Gaussians relative to the canonical mesh, the approach elegantly bridges 2D image input and 3D Gaussian head representation—a process that previously required per-frame fitting.

   **Source-Target Self-supervised Training**: Two frames of the same identity are randomly selected as source and target. The source image is passed through the Generator to predict a Gaussian head, which is then driven by the target's expression and pose and rendered from the target's camera viewpoint, supervised by the ground-truth target image. This enables learning of 3D priors without multi-view data.

   Loss function: $\mathcal{L}_{\text{E2V}} = \lambda_{\text{L1}}\mathcal{L}_{\text{L1}} + \lambda_{\text{SSIM}}\mathcal{L}_{\text{SSIM}} + \lambda_{\text{vgg}}\mathcal{L}_{\text{vgg}} + \lambda_\mu\mathcal{L}_\mu$

3. **Customized Adaptation**

   - **Expression-Visual Fine-tuning**: A coarse UV Gaussian map $\hat{M}_{id}$ is generated from the reference image, driven by FLAME parameters from the full training video, and rendered under supervision from ground-truth frames. FLAME parameters are first frozen while only $\hat{M}_{id}$ is optimized; both are then jointly optimized to correct monocular tracking errors.

   - **Audio-Expression Fine-tuning**: The audio encoder is frozen while the condition encoder and Transformer decoder are fine-tuned to adapt to the speaking style of the target identity. A low learning rate and early stopping are employed to prevent overfitting.

   - **Color MLP $\mathcal{M}_{\text{SH}}$**: Dynamically adjusts Gaussian color attributes based on expression and pose parameters: $\mathbf{SH}_l = \mathcal{M}_{\text{SH}}(\hat{\mathbf{SH}_l}, \mathcal{F}_{exp}, \mathcal{F}_{pose})$, producing sharp textures aligned with motion.

   - **Body Inpainter $\mathcal{I}$**: A lightweight U-Net that blends the rendered head result with the torso and background, avoiding artifacts caused by hard compositing: $I_{vid} = \mathcal{I}(I_{res}, (1-\text{Dilate}(\mathbf{M}))I_{ori})$

### Loss & Training

- Prior stage: Audio-Expression model trained on HDTF + CN-CVS + 100h of in-house data; Expression-Visual model trained on VFHQ + NeRSemble. Each takes approximately 2 days (8×A100).
- Adaptation stage: ~20 minutes (1×A100), lr=1e-5.
- Inference: 120 FPS.

## Key Experimental Results

### Main Results

**Self-reenactment quantitative results**:

| Method | PSNR↑ | LPIPS↓ | SSIM↑ | FID↓ | LMD↓ | AUE↓ | LSE-C↑ | Training Time↓ | FPS↑ |
|--------|-------|--------|-------|------|------|------|--------|---------------|------|
| ER-NeRF | 30.438 | 0.0408 | 0.9331 | 5.516 | 4.014 | 4.774 | 5.008 | 5h | 38.2 |
| SyncTalk | 32.545 | 0.0334 | 0.9630 | 6.820 | 2.963 | 3.618 | 7.693 | 5h | 31.9 |
| GaussianTalker | 32.941 | 0.0531 | 0.9531 | 6.392 | 3.061 | 2.980 | 6.109 | 1h | 72.8 |
| **GGTalker** | **35.203** | **0.0281** | **0.9816** | **4.624** | **2.328** | **2.171** | **8.210** | **0.3h** | **120** |

GGTalker substantially outperforms all baselines on nearly every metric: PSNR is 2.3 dB higher, LPIPS is 16% lower, training requires only 20 minutes, and inference runs at 120 FPS—3.75× faster than SyncTalk.

**OOD audio lip synchronization**:

| Method | Cross-ID LSE-D↓ | Cross-ID LSE-C↑ | Cross-Lang LSE-D↓ | Cross-Lang LSE-C↑ |
|--------|----------------|----------------|------------------|------------------|
| SyncTalk | 8.732 | 5.640 | 9.756 | 5.301 |
| TalkingGaussian | 9.501 | 4.344 | 9.831 | 3.118 |
| **GGTalker** | **8.051** | **6.268** | **8.923** | **5.769** |

GGTalker achieves state-of-the-art performance not only in self-reenactment but also in cross-identity and cross-language OOD scenarios, validating the generalization capability conferred by prior learning.

### Ablation Study

| Configuration | LPIPS↓ | LMD↓ | LSE-C↑ | Notes |
|--------------|--------|------|--------|-------|
| **Full GGTalker** | **0.0281** | **2.328** | **5.769** | Full model |
| w/o A-E Priors | 0.0306 | 2.741 | 3.268 | No audio-expression priors; trained from scratch; severe OOD degradation |
| w/o A-E Fine-tuning | 0.0385 | 3.287 | 4.780 | No speaking style adaptation; unnatural motion |
| w/o E-V Priors | 0.0438 | 3.826 | 4.682 | No visual priors; trained from scratch; poor 3D consistency |
| w/o E-V Fine-tuning | 0.0473 | 2.925 | 4.524 | No texture fine-tuning; identity overly smoothed |
| w/o Color Fine-tuning | 0.0336 | 2.470 | 5.431 | Fixed colors; subtle expressions lost |

### Key Findings

- Audio-Expression Priors are critical for OOD generalization—removing them causes LSE-C to drop sharply from 5.769 to 3.268.
- The two stages of Expression-Visual Priors and Fine-tuning are both indispensable: priors alone yield overly smooth textures, while fine-tuning alone produces artifacts and holes in novel views.
- The Color MLP is lightweight yet contributes substantially to texture sharpness (LPIPS: 0.0336→0.0281).
- Joint optimization of FLAME parameters has a positive effect on correcting monocular tracking errors.

## Highlights & Insights

- **Elegance of the prior-adaptation paradigm**: Learning universal patterns (head texture distributions, audio-to-lip correlations) from large-scale data and rapidly transferring them to specific individuals is a powerful design philosophy. Training time is reduced from several hours to 20 minutes—an efficiency gain of over 10×.
- **Clever exploitation of UV space**: By leveraging the UV layout of the FLAME mesh, 3D Gaussian prediction is elegantly reformulated as a 2D image generation problem, eliminating the need for tedious per-frame fitting.
- **120 FPS real-time rendering**: Owing to the efficient rendering of Gaussian splatting combined with a compact Gaussian representation, inference speed far surpasses all competing methods.

## Limitations & Future Work

- FLAME models only the head region; torso and hand regions require additional processing (the simple U-Net Body Inpainter leaves room for improvement).
- Robustness to extreme facial occlusions (e.g., hand-over-face, glasses reflections) is not discussed.
- Expression priors are trained on large-scale data but cover a limited emotional range, and may generalize poorly to extreme expressions.
- The accuracy of the FLAME tracking algorithm directly impacts final output quality; joint optimization only partially mitigates this issue.

## Related Work & Insights

- 2D methods (e.g., Wav2Lip, Hallo) offer strong generative capabilities but suffer from poor identity consistency; GGTalker achieves superior identity preservation through explicit 3D representation.
- The Gaussian head modeling approaches of GGHead and GaussianAvatars directly inspired this work.
- The prior-adaptation paradigm is well established in NLP (e.g., pretraining-finetuning); this paper successfully introduces it to the domain of 3D talking head synthesis.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The prior-adaptation framework is systematically realized in talking head synthesis for the first time; the UV-space Gaussian prediction design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers self-reenactment, cross-identity, and cross-language scenarios; ablations are comprehensive; qualitative comparisons are thorough.
- **Writing Quality**: ⭐⭐⭐⭐ Technical details are complete and ablation analysis is clear.
- **Value**: ⭐⭐⭐⭐⭐ Breakthroughs in training efficiency and rendering speed bring talking head technology closer to practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)
- [\[NeurIPS 2025\] VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](../../NeurIPS2025/human_understanding/vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)
- [\[ICCV 2025\] Controllable and Expressive One-Shot Video Head Swapping](controllable_and_expressive_one-shot_video_head_swapping.md)
- [\[ICCV 2025\] ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling](imhead_a_large-scale_implicit_morphable_model_for_localized_head_modeling.md)
- [\[NeurIPS 2025\] OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild](../../NeurIPS2025/human_understanding/omnigaze_reward-inspired_generalizable_gaze_estimation_in_the_wild.md)

</div>

<!-- RELATED:END -->
