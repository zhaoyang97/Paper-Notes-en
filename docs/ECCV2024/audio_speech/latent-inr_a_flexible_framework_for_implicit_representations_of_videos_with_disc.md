---
title: >-
  [Paper Note] Latent-INR: A Flexible Framework for Implicit Representations of Videos with Discriminative Semantics
description: >-
  [ECCV2024][Audio & Speech][Implicit Neural Representations] This paper proposes the Latent-INR framework. By learning an implicit latent code for each video frame and combining it with a hypernetwork for low-rank weight modulation, the framework decouples the spatial and temporal modeling of video INR. While maintaining competitive compression performance, it equips video representations with semantic discriminative capabilities, supporting various downstream tasks such as re…
tags:
  - "ECCV2024"
  - "Audio & Speech"
  - "Implicit Neural Representations"
  - "Video Compression"
  - "Video Retrieval"
  - "Hypernetwork"
  - "CLIP Alignment"
date: 2026-05-08
content_hash: 1ab883a737fcc8f7
---

# Latent-INR: A Flexible Framework for Implicit Representations of Videos with Discriminative Semantics

**Conference**: ECCV2024  
**arXiv**: [2408.02672](https://arxiv.org/abs/2408.02672)  
**Code**: To be confirmed  
**Area**: Audio and Speech  
**Keywords**: Implicit Neural Representations, Video Compression, Video Retrieval, Hypernetwork, CLIP Alignment

## TL;DR

This paper proposes the Latent-INR framework. By learning an implicit latent code for each video frame and combining it with a hypernetwork for low-rank weight modulation, the framework decouples the spatial and temporal modeling of video INR. While maintaining competitive compression performance, it equips video representations with semantic discriminative capabilities, supporting various downstream tasks such as retrieval, video frame interpolation, and arbitrary-resolution inference.

## Background & Motivation

- **Implicit Neural Representations (INR)** avoid generalization issues by training a small network for each individual video. However, existing video INR methods primarily focus on compression, and the learned representations lack semantic information, rendering them unsuitable for direct use in downstream tasks like retrieval.
- Although traditional codecs (HEVC, H.264, AV1) are highly mature, ML-based codecs generally lack clear advantages beyond compression in practical deployments, making them difficult to adopt widely.
- Existing video INR methods face two key challenges: (1) poor architectural scalability for long videos; (2) excessive encoding time, as each video requires training an individual network.
- The core idea of this work is not to directly shorten the encoding time, but to demonstrate that the computational cost of training is worthwhile—by enabling INR to support semantic downstream tasks in addition to compression.

## Core Problem

How can a video INR framework be designed such that, while maintaining competitive compression performance, the learned representations possess semantic discriminative power to support multiple tasks, including retrieval, question answering, video frame interpolation, and arbitrary-resolution inference?

## Method

### Overall Architecture: Auto-Decoder + Hypernetwork

The framework consists of two parts:

1. **Frame-level learnable latent dictionary**: A latent code $z_t \in \mathbb{R}^D$ is learned for each frame of the video.
2. **Shared hypernetwork ensemble**: Given the latent $z_t$, the hypernetwork predicts frame-specific weight modulation parameters used to modify the weights of the shared base network.

The fundamental formulation is:

$$f_\theta((x,y)|\theta_t) = Y_t, \quad \theta_t = h(z_t)$$

where the base network $f_\theta$ takes the spatial coordinate grid $(x,y)$ as input to output the corresponding frame. This design decouples the spatial and temporal modeling of the video—the hypernetwork learns the overall structure and style of the video, while the latent code is responsible for conditioning the output of specific frames.

### Low-Rank Weight Modulation

Predicting all weights of the base network directly using a hypernetwork is computationally prohibitive; hence, low-rank matrix modulation is adopted:

$$\theta_t^l = \sigma(P^l \times Q^l) \cdot \theta^l, \quad h_l(z_t) = [P^l, Q^l]$$

where $P^l \in \mathbb{R}^{N \times r}$, $Q^l \in \mathbb{R}^{M \times r}$, and $r \ll (N, M)$. The rank $r$ and the number of modulated layers are hyperparameters that control the compression-performance trade-off. This approach is reminiscent of a sub-network selection mechanism.

### Network Details

- **Base network**: A 6-layer MLP with a layer width of 512, using ReLU activation. Some variants incorporate a convolutional upsampling module (PixelShuffle) to process patch centroid inputs.
- **Hypernetwork**: One hypernetwork is mapped to each modulated layer, containing a 128-dimensional hidden layer and a tanh non-linearity.
- **Positional encoding**: A hash-grid is used to achieve high-quality reconstruction (can be replaced with Fourier features for faster training speed).
- **Latent initialization**: Standard normal distribution with a small variance, which helps accelerate convergence.

### Model Compression

After end-to-end training with MSE loss, standard quantization (with a specified bit width $b$) is applied to all network parameters, followed by Huffman coding for further compression.

### Video Frame Interpolation

Linear interpolation is performed on the frame latents to generate intermediate frames:

$$z_{\text{inter}} = \beta_i \cdot z_t + (1 - \beta_i) \cdot z_{t-1}$$

The interpolated latent is then fed into the hypernetwork to obtain the weight modulation for the intermediate frame. This supports $\alpha \in \{2, 4, 8\}$ times frame interpolation.

### Semantic Alignment

- **CLIP Alignment (Retrieval)**: A cosine similarity loss between the latent and CLIP image embeddings is added to the training loss: $L = L_{\text{MSE}} + \lambda \cdot L_{\text{clip}}$, where $\lambda = 0.01$.
- **VideoLlama Alignment (QA/Chat)**: The latent dictionary is projected as tokens into the VideoLlama space, replacing the original video input tokens, to support open-ended text dialogue.

## Key Experimental Results

### Video Compression (UVG Dataset)

- Achieves performance comparable to NVP (the then-SOTA) on the PSNR/BPP rate-distortion curve, while offering better decoding FPS.
- Supports **arbitrary-resolution inference**: The same model can decode at different resolutions without modification (whereas traditional encoders require separate encoding for each resolution).

### Video Frame Interpolation

| Dataset | $\alpha$ | NeRV | NIRVANA | NVP | **Ours** |
|---------|----------|------|---------|-----|----------|
| Bunny   | 2        | 15.92| 19.14   |20.10| **33.17**|
| Bunny   | 4        | 15.43| 18.90   |19.11| **28.08**|
| Bunny   | 8        | 13.68| 18.67   |18.08| **25.88**|
| TaiChi  | 2        | 16.91| 18.19   |19.33| **35.13**|
| TaiChi  | 8        | 15.72| 16.21   |17.70| **27.72**|

Demonstrates a huge advantage over other INR methods (surpassing NVP by approximately 13-16 dB at $\alpha=2$).

### Video Retrieval

- On MSR-VTT, T2V R@1=30.2, basically on par with directly using CLIP features (30.1).
- On COIN class-level retrieval, R@1=34.4, even surpassing CLIP's 31.6.
- On ActivityNet, long video retrieval also performs consistently with CLIP features.

### Ablation on CLIP $\lambda$

$\lambda=0.01$ is the optimal trade-off point: PSNR only drops from 30.03 to 29.46 (-0.57 dB), while T2V R@1 surges from 0.1 to 30.2.

## Highlights & Insights

- **Versatility**: This is the first work to simultaneously achieve compression, super-resolution, arbitrary-resolution inference, video frame interpolation, text retrieval, and video question answering within a single video INR framework.
- **Semantically Rich Latent Space**: UMAP visualizations show that even when trained solely with compression objectives, the frame latents can capture scene semantics (repetitive patterns cluster together, and dynamic scenes unfold along trajectories).
- **Outstanding Interpolation Performance**: The linear interpolation property of the latent space enables it to significantly outperform other INR methods in video frame interpolation.
- **Flexible Alignment**: The latents can be aligned with arbitrary large models (e.g., CLIP, VideoLlama), demonstrating strong scalability.

## Limitations & Future Work

- Encoding time remains high (requires training the entire system individually for each video), which does not fundamentally solve the training efficiency issue of INR.
- Although the compression performance is comparable, it does not significantly surpass methods like NVP, and the additional semantic capabilities come at an extra computational cost.
- The quality of the Chat functionality is primarily limited by the aligned LLM itself (limitations of VideoLlama), and experimental results are mostly presented qualitatively.
- Compression is evaluated only on UVG (7 videos), representing a limited dataset scale.
- The classification of the field as "audio_speech" might be inaccurate; the core focus of this paper is video representation and compression.

## Related Work & Insights

| Method | Compression | Frame Interpolation | Retrieval | Arbitrary Resolution | Chat |
|--------|-------------|---------------------|-----------|----------------------|------|
| NeRV   | ✓           | ✗                   | ✗         | ✗                    | ✗    |
| NIRVANA| ✓           | ✗                   | ✗         | ✗                    | ✗    |
| NVP    | ✓           | ✗                   | ✗         | ✗                    | ✗    |
| **Latent-INR** | ✓   | ✓                   | ✓         | ✓                    | ✓    |

- The core difference from NeRV/NVP/NIRVANA lies in the introduction of the auto-decoder framework and the latent dictionary, which decouples space and time.
- The low-rank modulation concept is similar to LoRA but applied to the INR scenario, where the latent is not a redundant intermediate representation but the core information carrier.
- Compared to traditional codecs, the greatest advantage is that the same model supports arbitrary-resolution and semantic tasks.

## Insights & Connections

- The concept of utilizing latent as a proxy for weights is noteworthy—indirectly manipulating network outputs by operating on the latent space represents an elegant design paradigm.
- Achieving good retrieval performance with a CLIP alignment loss of only $\lambda=0.01$ while barely compromising reconstruction quality suggests that semantic and reconstruction information can coexist within the representation space.
- Arbitrary-resolution inference represents a significant advantage over traditional codecs, offering potential value in practical streaming scenarios (such as adaptive bitrate streaming).
- It is worth tracking whether future works will extend this framework to larger-scale video datasets or introduce more efficient training strategies.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to integrate compression and semantic tasks within video INR, with a novel framework design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers four major tasks: compression, interpolation, retrieval, and Chat, complete with ablations and visualizations, though the compression evaluation dataset is relatively small.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-elaborated motivation.
- Value: ⭐⭐⭐⭐ — Provides compelling arguments for the practical adoption of ML-based codecs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning](../../ICML2026/audio_speech/position_text_embeddings_should_capture_implicit_semantics_not_just_surface_mean.md)
- [\[ECCV 2024\] Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](action2sound_ambientaware_generation_of_action_sounds_from_e.md)
- [\[ECCV 2024\] CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)
- [\[CVPR 2025\] ImViD: Immersive Volumetric Videos for Enhanced VR Engagement](../../CVPR2025/audio_speech/imvid_immersive_volumetric_videos_for_enhanced_vr_engagement.md)
- [\[ACL 2025\] In-the-wild Audio Spatialization with Flexible Text-guided Localization](../../ACL2025/audio_speech/tas_audio_spatialization.md)

</div>

<!-- RELATED:END -->
