---
title: >-
  [Paper Note] End-to-End Multi-Modal Diffusion Mamba
description: >-
  [ICCV 2025][Image Generation][Multimodal Models] This paper proposes Multi-Modal Diffusion Mamba (MDM), an end-to-end multimodal model based on the Mamba architecture. By employing a unified VAE encoder-decoder and a multi-step selective diffusion model, MDM enables simultaneous generation of images and text with computational complexity $\mathcal{O}(MLN^2)$, surpassing existing end-to-end models on tasks including image generation, image captioning, and VQA.
tags:
  - ICCV 2025
  - Image Generation
  - Multimodal Models
  - Mamba
  - Diffusion Models
  - End-to-End Generation
  - State Space Models
date: 2026-05-08
content_hash: c16a191133148493
---

# End-to-End Multi-Modal Diffusion Mamba

**Conference**: ICCV 2025
**arXiv**: [2510.13253](https://arxiv.org/abs/2510.13253)
**Code**: None
**Area**: Diffusion Models / Image Generation
**Keywords**: Multimodal Models, Mamba, Diffusion Models, End-to-End Generation, State Space Models

## TL;DR

This paper proposes Multi-Modal Diffusion Mamba (MDM), an end-to-end multimodal model based on the Mamba architecture. By employing a unified VAE encoder-decoder and a multi-step selective diffusion model, MDM enables simultaneous generation of images and text with computational complexity $\mathcal{O}(MLN^2)$, surpassing existing end-to-end models on tasks including image generation, image captioning, and VQA.

## Background & Motivation

Current multimodal large models face three structural challenges:

**Traditional multi-encoder/decoder architectures** (e.g., LLaVA, Flamingo): employ separate encoders and decoders for each modality, hindering joint representation learning and resulting in long inference times.

**Bottlenecks of end-to-end Transformer models**:
   - **Quadratic complexity**: The $O(L^2)$ complexity of Transformers renders them inefficient for high-resolution image and long-sequence text generation.
   - **Multi-objective conflict**: Simultaneously optimizing objectives for image and text introduces optimization conflicts, impeding convergence and joint representation learning.

**Limitations of existing end-to-end approaches**:
   - Autoregressive models (e.g., Chameleon) are constrained by sequential dependencies.
   - Hybrid generative models (e.g., Seed-X) introduce additional complexity.
   - Hybrid autoregressive-diffusion models (e.g., MonoFormer) still struggle to unify multimodal processing.

Mamba/State Space Models (SSMs) have emerged as a promising alternative due to their linear complexity and capacity to model long-range dependencies. However, existing Mamba-based multimodal works still adopt multi-objective approaches and lack genuine end-to-end joint representation learning.

The core contribution of MDM is the integration of Mamba with the diffusion process: a **unified VAE encoder-decoder** handles all modalities, while a **multi-step selective diffusion model** serves as a unified generative framework. Score Entropy Loss replaces Markov chain methods to improve efficiency.

## Method

### Overall Architecture

The MDM architecture consists of three components: (1) a VAE noisy latent space encoder that maps images and text uniformly into a noisy latent space; (2) a Mamba-based multi-step selective diffusion decoder that progressively denoises the data; and (3) a VAE noisy latent space decoder that reconstructs denoised latent variables into images or text. The entire pipeline processes and generates multimodal data simultaneously.

### Key Designs

1. **Unified VAE Encoder-Decoder**:

    - **Function**: Encodes image patches and text embeddings uniformly into a shared noisy latent space.
    - **Mechanism**: Images are processed via a patchify operation; text is tokenized using SentencePiece+BPE and embedded. Both are then passed through VAE sampling to obtain $z_n = s_n + \epsilon_n$. The encoder generates Gaussian distribution parameters $(\mu, \sigma)$ and incorporates learnable padding tokens (time, class, pad).
    - **Design Motivation**: A unified encoder-decoder eliminates the representational gap between modalities, enabling the model to learn genuine joint multimodal representations within a shared latent space.

2. **Multi-step Selective Diffusion Model**:

    - **Function**: Combines the diffusion process with Mamba's selection mechanism to progressively generate multimodal information.
    - **Mechanism**: The forward diffusion follows the standard formulation:
    $z_{n,t}^g = \sqrt{\bar{\alpha}_t^g} z_{n,0}^g + \sqrt{1-\bar{\alpha}_t^g} \epsilon_{n,t}^g$
      Rather than using a traditional Markov chain for denoising, Score Entropy Loss (SE) is adopted as a unified objective:
    $se = \sum_{y} \omega_{z_{n,t}^g}^g \left(s_\theta(z_{n,t}^g) - \frac{p_{data}(y)}{p_{data}(z_{n,t}^g)} \log s_\theta(z_{n,t}^g) + K(\cdot)\right)$
      The selection process leverages Mamba's SSM selection mechanism, determining which information to attend to or discard based on whether the score ratio approaches the true ratio (Theorem 3: $s_\theta(z_{n,t}^g) \approx \frac{p_{data}(y)}{p_{data}(z_{n,t}^g)}$).
    - **Design Motivation**: SE is a generalized score matching objective that directly learns probability density ratios between discrete states. It is more efficient than Markov chain methods in high-dimensional spaces and more readily extensible to discrete data (text).

3. **Image/Text Scan Switching + Mamba Block**:

    - **Function**: Captures sequential relationships through multi-directional scanning, followed by information selection via Mamba-2's SSM.
    - **Mechanism**: Images employ 4 scanning directions (following DiM); text employs 2 scanning directions. The Mamba Block updates its internal state via SSM:
    $H_{n,t}^g = \bar{A} H_{n,t-1}^g + \bar{B} z_{n,t}^g$
    $z_{n-1,t}^g = C H_{n,t}^g + D z_{n,t}^g$
      Denoising steps utilize the DPM-Solver second-order method to improve sampling precision:
    $z_{n,t-\Delta t}^g = z_{n,t}^g - \frac{\Delta t}{2}[f_\theta(z_{n,t}^g, t) + f_\theta(z_{n,t-\Delta t}^g, t-\Delta t)]$
    - **Design Motivation**: Mamba's selection mechanism is naturally suited to diffusion denoising — at each step, the model must decide which information is sufficiently clear to retain and which remains noisy and requires correction. Multi-directional scanning ensures that diverse spatial relationships are captured.

### Loss & Training

The overall optimization objective combines four components:

$$L_{total} = L_{rec}^{img} + L_{rec}^{txt} + \beta L_{KL} + \lambda L_{se}$$

- $L_{rec}^{img}$: L2 reconstruction loss for images
- $L_{rec}^{txt}$: cross-entropy loss for text
- $L_{KL}$: KL divergence regularization for the VAE
- $L_{se}$: Score Entropy Loss

The model comprises 7B parameters, 49 Mamba Blocks, and a hidden dimension of 2048.

## Key Experimental Results

### Main Results

Image Generation (ImageNet & COCO 256×256):

| Model | Type | Params | FID↓ | IS↑ | Precision | Recall |
|------|------|------|------|-----|-----------|--------|
| DiT-XL/2 | Diff | 675M | 2.27 | 278.2 | 0.83 | 0.57 |
| LlamaGen | AR | 3.1B | 2.81 | 311.5 | 0.84 | 0.54 |
| MonoFormer | AR+Diff | 1.1B | 2.57 | 272.6 | 0.84 | 0.56 |
| **MDM** | **Diff** | **7B** | **2.49** | **281.4** | **0.86** | **0.59** |

Text-to-Image Generation (COCO):

| Model | FID↓ | GenEval↑ |
|------|------|---------|
| SDXL | 4.40 | 0.55 |
| Chameleon | 26.74 | 0.39 |
| Transfusion | 6.78 | 0.63 |
| **MDM** | **5.91** | **0.68** |

Multi-task Comprehensive Evaluation:

| Model | IC-COCO | VQAv2 | PIQA | MMLU | GSM8k |
|------|---------|-------|------|------|-------|
| Chameleon (34B) | 120.2 | 66.0 | 79.6 | 52.1 | 41.6 |
| NExT-GPT (7B) | 124.9 | 66.7 | — | — | — |
| **InstructMDM (7B)** | **122.1** | **66.7** | **83.7** | **54.4** | **46.0** |
| Mistral (7B, text-only) | — | — | 83.0 | 60.1 | 52.1 |

### Ablation Study

| Configuration | ImageNet FID↓ | COCO FID↓ | Note |
|------|-------------|----------|------|
| MDM w/o selection | 3.21 | 7.84 | No selection mechanism; full processing |
| MDM w/ 1 scan direction | 2.85 | 6.73 | Single-direction scanning |
| MDM w/ SE loss | **2.49** | **5.91** | Full model (SE + multi-direction) |
| MDM w/ Markov chain | 2.97 | 6.92 | SE replaced by conventional DDPM |

Computational Complexity Comparison:

| Model | Complexity | Note |
|------|--------|------|
| MonoFormer | $O(ML^2N/G)$ | Quadratic Transformer complexity |
| **MDM** | **$O(MLN^2)$** | Linear sequence complexity via Mamba |

### Key Findings

- MDM achieves competitive ImageNet FID (2.49) compared to DiT-XL/2 (2.27) and MonoFormer (2.57) while supporting multitask capabilities.
- On text-to-image generation, MDM (5.91 FID, 0.68 GenEval) substantially outperforms autoregressive end-to-end models such as Chameleon (26.74, 0.39).
- InstructMDM approaches or exceeds text-only models of comparable scale on language understanding benchmarks (e.g., PIQA 83.7 vs. Mistral 83.0).
- Mamba's linear complexity makes it significantly more efficient than Transformer-based end-to-end models when handling high-resolution images and long text sequences.

## Highlights & Insights

- **Genuine end-to-end multimodality**: A single VAE and a single Mamba decoder are employed, with no modality-specific encoders, decoders, or fusion modules.
- **Novel integration of diffusion and Mamba**: Mamba's selection mechanism guides the direction of diffusion denoising, supported theoretically by Theorem 3.
- **Score Entropy Loss**: Unifies the generative objectives for continuous (image) and discrete (text) modalities, offering greater efficiency than Markov chain methods.
- **Simultaneous multimodal generation**: Unlike models that generate text prior to images, MDM can jointly output images and their corresponding captions.

## Limitations & Future Work

- The 7B parameter count is substantial, imposing non-trivial deployment costs for an end-to-end model.
- Image captioning metrics (e.g., Flickr30K CIDEr 62.4) leave room for improvement relative to specialized models, though direct comparison is complicated by differing evaluation protocols.
- The theoretical analysis of Score Entropy Loss is grounded in discrete states; its rigor when applied to continuous latent variables warrants further investigation.
- Evaluations at higher resolutions (e.g., 512×512 or 1024×1024) are absent.
- The model's capacity for video generation remains unexplored.

## Related Work & Insights

- MDM directly competes with MonoFormer and Transfusion as hybrid autoregressive-diffusion end-to-end models, but replaces Transformers with Mamba to achieve linear complexity.
- The multi-directional scanning strategy from DiM is adopted for images and extended to the text modality.
- The SE loss from Score Entropy Discrete Diffusion (SEDD) is effectively repurposed as a unified multimodal learning objective.
- Key insight: Mamba's selection mechanism is inherently aligned with diffusion denoising's core operation — deciding what to retain and what to remove.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The end-to-end multimodal combination of Mamba and diffusion is a first-of-its-kind contribution with imaginative architectural design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluations span image generation, VQA, and language understanding, though high-resolution and video generation assessments are missing.
- **Writing Quality**: ⭐⭐⭐⭐ The architecture is described in detail with complete mathematical derivations, though the overall length is considerable.
- **Value**: ⭐⭐⭐⭐ The work opens a promising new direction (Mamba-based end-to-end multimodal modeling), though the advantage over Transformer baselines has yet to be decisively demonstrated.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](repae_unlocking_vae_for_endtoend_tuning_of_latent_diffusion.md)
- [\[CVPR 2026\] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation](../../CVPR2026/image_generation/deco_frequency-decoupled_pixel_diffusion_for_end-to-end_image_generation.md)
- [\[NeurIPS 2025\] LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss](../../NeurIPS2025/image_generation/lineas_end-to-end_learning_of_activation_steering_with_a_distributional_loss.md)
- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)
- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)

<!-- RELATED:END -->
