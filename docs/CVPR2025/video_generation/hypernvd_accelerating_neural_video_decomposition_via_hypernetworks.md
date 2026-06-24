---
title: >-
  [Paper Note] HyperNVD: Accelerating Neural Video Decomposition via Hypernetworks
description: >-
  [CVPR 2025][Video Generation][Video Decomposition] HyperNVD proposes using a hypernetwork to dynamically generate the parameters of Implicit Neural Representations (INR) based on video embeddings encoded by VideoMAE. This establishes a universal video decomposition model across videos, which achieves the same PSNR over 30 minutes faster than training from scratch on new videos, while improving the final performance by 0.8dB on average.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Video Decomposition"
  - "Hypernetworks"
  - "Implicit Neural Representations"
  - "Meta-Learning"
  - "Video Editing"
  - "Layered Decomposition"
date: 2026-05-08
content_hash: 2f8de99e928ba215
---

# HyperNVD: Accelerating Neural Video Decomposition via Hypernetworks

**Conference**: CVPR 2025  
**arXiv**: [2503.17276](https://arxiv.org/abs/2503.17276)  
**Code**: [https://hypernvd.github.io/](https://hypernvd.github.io/)  
**Area**: Video Understanding/Video Editing  
**Keywords**: Video Decomposition, Hypernetworks, Implicit Neural Representations, Meta-Learning, Video Editing, Layered Decomposition

## TL;DR

HyperNVD proposes using a hypernetwork to dynamically generate the parameters of Implicit Neural Representations (INR) based on video embeddings encoded by VideoMAE. This establishes a universal video decomposition model across videos, which achieves the same PSNR over 30 minutes faster than training from scratch on new videos, while improving the final performance by 0.8dB on average.

## Background & Motivation

**Background**: Layer-based video decomposition methods represent a video as multiple texture layers (foreground/background), where each layer corresponds to specific content, facilitating independent editing that can be propagated throughout the video. Current mainstream methods (LNA, Hashing-nvd, CoDeF) are based on Implicit Neural Representations (INR), which map pixel coordinates and frame indices to a canonical 2D texture space and then decode them into RGB values.

**Limitations of Prior Work**: (1) INR methods need to be trained independently for each video and lack generalization capability—a new video requires tens of minutes of training from scratch (typically >40 minutes for 480p videos); (2) Each model can only handle a single video, failing to exploit shared knowledge across multiple videos; (3) Training from random initialization converges slowly and is prone to overfitting to specific features of a single video.

**Key Challenge**: The advantages of INR (compact representation, precise reconstruction) and its disadvantages (no generalization, slow training) are two sides of the same coin—compact parameters mean the model is highly specialized to a single video.

**Goal**: Design a general video decomposition meta-model that enables fast convergence on new videos without sacrificing reconstruction quality.

## Method

### Overall Architecture

HyperNVD consists of three components: (1) **VideoMAE Encoder**—a pre-trained self-supervised video model that compresses the input video into a compact embedding (768×1); (2) **Hypernetwork (Hypernet)**—a series of MLPs that dynamically generate all parameters of the target NVD model (including multi-resolution hash encodings and network weights) based on the video embedding; (3) **Neural Video Decomposition (NVD) Model**—consisting of a foreground layer module, a background layer module, and an alpha module, which maps coordinates (x,y,t) to layered RGB outputs.

### Key Designs

1. **Hypernetwork Parameter Generation**:
    - **Function**: Dynamically generate the full NVD model parameters based on the video embedding.
    - **Mechanism**: The hypernetwork consists of a series of MLPs (four-layer fully connected, hidden dimension 64), where each MLP is responsible for generating the parameters of a specific layer in the NVD model. The input is the video embedding $e$, and the outputs are the weights of all layers and the multi-resolution hash encoding parameters.
    - During training, only the weights of the hypernetwork are learnable (~290M parameters). The NVD model (~4.4M parameters) acts as a differentiable layer used for backpropagation but is not directly optimized.

2. **VideoMAE Video Embeddings**:
    - **Function**: Compress high-dimensional video data into a compact, information-rich low-dimensional representation.
    - **Mechanism**: Features are extracted using a frozen VideoMAE (a self-supervised pre-trained video Transformer) and then compressed into a 768×1 dimensional embedding via an additional autoencoder, which is trained using an L1 loss.
    - **Design Motivation**: Directly using learnable embeddings requires joint training with the hypernetwork and cannot generalize to new videos. VideoMAE embeddings naturally encode motion and scene information, enabling direct inference on new videos.

3. **Layered Decomposition Structure of the NVD Model**:
    - **Function**: Decompose the video into two independently editable layers: foreground and background.
    - **Mechanism**: Both the foreground and background consist of three sub-modules: a mapping module (coordinates to texture space), a texture module (texture coordinates to RGB), and a residual module (frame-level illumination/color correction). The final output is blended via alpha blending.
    - Multi-resolution hash encoding (MRHE) is used in the texture and residual modules to accelerate training.

### Loss & Training

The loss formulation follows prior work (LNA, Hashing-nvd):
- **Reconstruction Loss**: Ensures video reconstruction quality.
- **Consistency Loss**: Uses optical flow supervision to guarantee accurate motion representation.
- **Sparsity Loss**: Prevents duplicate content from appearing in different texture layers.
- **Residual Consistency Loss**: Maintains smooth lighting conditions.
An additional rigidity loss and alpha guidance loss are used in the initial stage.

Before training, a pre-training step is conducted: the mapping network is configured to generate an initial aligned rectangular texture shape to prevent incorrect texture orientations.

## Key Experimental Results

| Comparison | Metric | Result |
|------|------|------|
| Single-video training vs. Baseline (hike) | PSNR | 30.06 vs. Hashing-nvd 29.12, LNA 30.02 |
| Single-video training vs. Baseline (bear) | PSNR | 31.58 vs. Hashing-nvd 31.56, LNA 29.62 |
| 1 vs. 15 vs. 30 video joint training | PSNR Drop | Only ~3dB |
| Meta-model fine-tuning vs. Training from scratch (10 new videos) | Average PSNR Gain | +0.8dB |
| Meta-model fine-tuning vs. Training from scratch | Time to reach same PSNR | **30+ minutes faster** |
| Hypernetwork parameters | - | ~290M |
| NVD model parameters | - | ~4.4M |

## Highlights & Insights

1. **Meta-learning approach to solve the INR generalization problem**: The "per-video independent training" of INR has always been a practicality bottleneck. Hypernetworks provide an elegant solution by learning a cross-video "initialization expert," allowing fast convergence on new videos through fine-tuning from this starting point.
2. **Minimal PSNR drop (~3dB) when training on multiple videos jointly**: Moving from 1 video to 30 videos results in an extremely mild performance drop, demonstrating that the hypernetwork indeed learns general video decomposition knowledge rather than overfitting to specific videos.
3. **Choice of VideoMAE embeddings**: Using embeddings from a pre-trained video model works better than learnable embeddings because it naturally encodes motion and scene semantics, reducing the learning difficulty for the hypernetwork.
4. **Clear practical value**: The 30-minute speedup is substantial for video editing workflows—meaning edits can be prepped in minutes instead of nearly an hour.

## Limitations & Future Work

1. The hypernetwork parameter size (~290M) is much larger than the target NVD model (~4.4M), incurring high storage and training costs.
2. Currently, it has only been validated on short videos (16 frames, 768×432) from the DAVIS dataset; its applicability to long videos and high resolutions remains unknown.
3. Joint training on multiple videos still suffers from an approximate 3dB quality loss, which might be unacceptable for professional editing scenarios with extremely high precision requirements.
4. It only supports two-layer (foreground/background) decomposition; complex scenes (with multiple moving objects) require further expansion.

## Related Work & Insights

- **Layered Video Decomposition**: LNA (Neural Atlases, the first INR video decomposition), Omnimatte (modelling shadows/reflections with appearance priors), Hashing-nvd (accelerating optimization with hash encoding), CoDeF (hash encoding + content deformation fields).
- **Hypernetworks**: HyP-NeRF (generating NeRF parameters via hypernetworks for 3D reconstruction), MetaSDF (accelerating SDF training via meta-learning).
- **Video Editing**: Traditional frame-by-frame/layered editing workflows, keyframe interpolation in motion tracking tools.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of hypernetworks and INR video decomposition is novel, with a clear motivation)
- Utility: ⭐⭐⭐⭐ (The 30-minute speedup + performance gains provide direct value to video editing workflows)
- Technical Depth: ⭐⭐⭐ (The approach is straightforward, and the technical implementation is relatively standard)
- Clarity: ⭐⭐⭐⭐⭐ (The structure is clear, with sufficient illustrations and comprehensive experimental analysis)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Identity-Preserving Text-to-Video Generation by Frequency Decomposition](identity-preserving_text-to-video_generation_by_frequency_decomposition.md)
- [\[CVPR 2026\] Generative Neural Video Compression via Video Diffusion Prior](../../CVPR2026/video_generation/generative_neural_video_compression_via_video_diffusion_prior.md)
- [\[ACL 2025\] Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval](../../ACL2025/video_generation/q2e_query-to-event_decomposition_for_zero-shot_multilingual_text-to-video_retrie.md)
- [\[ICLR 2026\] NeRV-Diffusion: Diffuse Implicit Neural Representation for Video Synthesis](../../ICLR2026/video_generation/nerv-diffusion_diffuse_implicit_neural_representation_for_video_synthesis.md)
- [\[ICCV 2025\] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation](../../ICCV2025/video_generation/fvgen_accelerating_novel-view_synthesis_with_adversarial_video_diffusion_distill.md)

</div>

<!-- RELATED:END -->
