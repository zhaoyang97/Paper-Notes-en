---
title: >-
  [Paper Note] Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation
description: >-
  [CVPR 2025][Image Generation][Video Tokenizer] This paper proposes Divot, a continuous video tokenizer that utilizes a diffusion process for self-supervised video representation learning. By training representations through a diffusion model conditioned on tokenizer features for denoising, and using a Gaussian Mixture Model (GMM) to model the continuous video feature distribution output by the LLM, a unified framework for video comprehension and generation is achieved.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Video Tokenizer"
  - "Diffusion Models"
  - "Unified Video Comprehension and Generation"
  - "Gaussian Mixture Model"
  - "Large Language Models"
date: 2026-05-08
content_hash: 44d402dcd4fda76e
---

# Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.04432](https://arxiv.org/abs/2412.04432)  
**Code**: [GitHub](https://github.com/TencentARC/Divot)  
**Area**: Image Generation  
**Keywords**: Video Tokenizer, Diffusion Models, Unified Video Comprehension and Generation, Gaussian Mixture Model, Large Language Models

## TL;DR
This paper proposes Divot, a continuous video tokenizer that utilizes a diffusion process for self-supervised video representation learning. By training representations through a diffusion model conditioned on tokenizer features for denoising, and using a Gaussian Mixture Model (GMM) to model the continuous video feature distribution output by the LLM, a unified framework for video comprehension and generation is achieved.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) have made significant progress in unifying image comprehension and generation, but unification in the video domain lags behind. Recent pioneered works (e.g., LWM, VILA-U) employ discrete video tokenizers to map videos into token sequences for autoregressive generation by LLMs.

**Limitations of Prior Work**: Although discrete video tokens facilitate generation (next-token prediction), they significantly degrade multimodal comprehension performance—continuous representations are more suitable for understanding tasks. However, modeling continuous representations with LLMs for generation is difficult; simple MSE regression often leads the LLM to learn over-averaged features, producing generated videos with repetitive patterns.

**Key Challenge**: Discrete representations benefit generation but hurt comprehension, whereas continuous representations benefit comprehension but hinder generation. A tokenizer is required to satisfy both directions simultaneously.

**Goal**: Design a continuous video tokenizer that supports both video comprehension (as input) and video generation (as output conditioning for decoding) in LLMs.

**Key Insight**: If a diffusion model can successfully denoise conditional on tokenizer features, it indicates that the tokenizer has captured sufficient spatial-temporal information; concurrently, this diffusion model can naturally serve as a de-tokenizer to decode videos.

**Core Idea**: Use diffusion denoising as a proxy task to train the video tokenizer (self-supervised), and substitute deterministic regression with GMM probabilistic modeling to allow LLMs to generate continuous video features.

## Method

### Overall Architecture
Sparsely sampled frames (2fps) are input to the tokenizer to obtain spatial-temporal representations, while densely sampled frames (8fps) are encoded by a VAE and subsequently perturbed with noise to train a U-Net to denoise conditioned on the tokenizer features. Once trained, the U-Net serves as the de-tokenizer. On the LLM side, comprehension is enabled by feeding video tokens via next-word prediction, while generation is achieved by predicting GMM parameters, sampling video features from the GMM, and decoding them through the de-tokenizer.

### Key Designs

1. **Diffusion-Driven Video Tokenizer**:

    - **Function**: Acquire continuous video representations that capture spatial-temporal information via self-supervised learning.
    - **Mechanism**: The tokenizer consists of a pre-trained ViT + Spatial-Temporal Transformer + Perceiver Resampler. During training, 64 tokens output by the tokenizer are used as cross-attention conditions for a DynamiCrafter U-Net to denoise VAE latents. The denoising objective forces the tokenizer to encode rich spatial-temporal details.
    - **Design Motivation**: Diffusion denoising requires the conditional signal to contain fine-grained spatial and temporal information for successful reconstruction, making it a natural proxy task for representation learning. The Perceiver Resampler compresses patch-level features into a fixed number of high-level tokens, reducing the number of tokens the LLM needs to predict.

2. **GMM Probabilistic Modeling of Video Features**:

    - **Function**: Enable LLMs to model and generate continuous video feature distributions effectively.
    - **Mechanism**: The LLM output is trained to predict GMM parameters ($2kd+k$ parameters: mean, variance, and mixture weights), optimized using negative log-likelihood (NLL) loss. During inference, features are sampled from the predicted GMM distribution to condition the de-tokenizer. Three schemes are compared: MSE regression (over-averaged), Diffusion modeling (high-level features are sensitive to noise), and GMM modeling (performs best).
    - **Design Motivation**: Deterministic MSE regression forces the LLM to learn the average features of all possible videos, resulting in repetitive patterns. Probabilistic modeling allows for diverse sampling, and GMM is more stable than diffusion modeling because high-level semantic features are highly sensitive to noise.

3. **Sparse-to-Dense Frame Sampling Strategy**:

    - **Function**: Balance tokenizer efficiency and video reconstruction quality.
    - **Mechanism**: The tokenizer takes sparsely sampled frames (5 frames, 2fps) to reduce the token sequence length, while the denoising target uses densely sampled frames (16 frames, 8fps) to ensure the complete learning of temporal dynamics.
    - **Design Motivation**: Adjacent frames exhibit high semantic redundancy, making sparse sampling sufficient for comprehension, whereas generation demands temporal details from dense frames.

### Loss & Training
Tokenizer training: Standard diffusion denoising loss. LLM training: Next-token prediction cross-entropy for comprehension, and GMM NLL loss for generation. Three stages: Tokenizer pre-training (10M videos) → LLM pre-training (video-text pairs) → SFT (multi-task).

## Key Experimental Results

### Main Results

| Model | LLM Size | Video Gen | EgoSchema | MVBench | ActivityNet |
|------|---------|---------|-----------|---------|-------------|
| Video-LLaVA | 7B | × | 38.4 | 41.0 | 45.3 |
| VideoChat2 | 7B | × | 42.2 | 51.1 | 49.1 |
| Video-LaVIT | 7B | ✓ | - | - | - |
| **Divot-LLM** | 7B | ✓ | **43.6** | **52.8** | **50.2** |

### Ablation Study

| Feature Modeling | FVD↓ | Similarity↑ |
|-------------|------|--------|
| MSE Regression | Poor | Lower |
| Diffusion Modeling | Medium | Medium |
| **GMM Modeling** | **Best** | **Highest** |

### Key Findings
- Divot-LLM is highly competitive with specialized comprehension models in video understanding, while additionally gaining video generation capability.
- GMM modeling significantly outperforms MSE regression and diffusion modeling, validating the importance of probabilistic modeling for continuous feature generation.
- High-level tokens generated by the Perceiver Resampler, which are position-independent, are easier for the LLM to fit compared to patch tokens that preserve spatial structures.
- The model supports video storytelling—alternating between generating narrative text and corresponding video segments.

## Highlights & Insights
- The concept of "diffusion as representation learning" is novel; the denoising objective naturally requires the condition to contain rich information, and the trained denoising network can be directly used as a decoder.
- The comparison between GMM, MSE, and Diffusion modeling is highly valuable, revealing the fundamental differences in modeling strategies between high-level semantic features and low-level pixel/latent features.
- The continuous tokenizer paradigm for unifying comprehension and generation may be more promising than the discrete token path.

## Limitations & Future Work
- Video generation quality is limited by the performance of the proxy diffusion model (DynamiCrafter).
- Currently tested only with Mistral-7B; scalability on stronger LLMs remains unknown.
- Video length is limited (2-second clips); long video generation warrants further research.
- The number of mixture components $k$ in the GMM is a hyperparameter whose optimal choice may vary across tasks.

## Related Work & Insights
- **vs VILA-U/LWM (Discrete Tokens)**: Discretization impairs comprehension precision; Divot employs continuous tokens to maintain comprehension performance while enabling generation.
- **vs Emu3 (Discrete VQ Token Gen)**: Emu3 directly uses LLM next-token prediction to generate discrete video tokens; Divot samples continuous tokens via GMM probabilistic sampling + diffusion decoding.
- **vs MAR (Image-Domain Diffusion Modeling)**: MAR successfully applies diffusion modeling on VAE latents, but this work finds that diffusion modeling underperforms GMM on high-level semantic features.

## Rating
- Novelty: ⭐⭐⭐⭐ Diffusion as a representation learning proxy coupled with GMM modeling for continuous tokens is a novel combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on multiple comprehension and generation benchmarks with a comprehensive comparison of modeling methods.
- Writing Quality: ⭐⭐⭐⭐ Clear framework and well-defined motivations.
- Value: ⭐⭐⭐⭐ Provides a robust solution for the unified comprehension and generation of video LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GraphGPT-o: Synergistic Multimodal Comprehension and Generation on Graphs](graphgpt-o_synergistic_multimodal_comprehension_and_generation_on_graphs.md)
- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [\[CVPR 2025\] StyleMaster: Stylize Your Video with Artistic Generation and Translation](stylemaster_stylize_your_video_with_artistic_generation_and_translation.md)
- [\[CVPR 2025\] EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)
- [\[CVPR 2025\] SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer](softvq-vae_efficient_1-dimensional_continuous_tokenizer.md)

</div>

<!-- RELATED:END -->
