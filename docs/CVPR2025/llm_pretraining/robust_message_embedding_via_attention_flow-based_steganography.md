---
title: >-
  [Paper Note] Robust Message Embedding via Attention Flow-Based Steganography
description: >-
  [CVPR 2025][LLM Pretraining][Message embedding] This paper proposes the RMSteg (Robust Message Steganography) framework, which integrates the Transformer attention mechanism into normalizing flow networks (AttnFlow) for the first time. Combined with an invertible QR code transition and an invertible token fusion module, it achieves high-quality, high-capacity, and robust message-to-image steganography. The stego-images can be decoded accurately even after extreme distortions…
tags:
  - "CVPR 2025"
  - "LLM Pretraining"
  - "Message embedding"
  - "Normalizing flows"
  - "Attention mechanism"
  - "QR codes"
  - "Robust steganography"
date: 2026-05-08
content_hash: a32a5a10b147db17
---

# Robust Message Embedding via Attention Flow-Based Steganography

**Conference**: CVPR 2025  
**arXiv**: [2405.16414](https://arxiv.org/abs/2405.16414)  
**Code**: To be confirmed  
**Area**: Image Steganography  
**Keywords**: Message embedding, Normalizing flows, Attention mechanism, QR codes, Robust steganography

## TL;DR

This paper proposes the RMSteg (Robust Message Steganography) framework, which integrates the Transformer attention mechanism into normalizing flow networks (AttnFlow) for the first time. Combined with an invertible QR code transition and an invertible token fusion module, it achieves high-quality, high-capacity, and robust message-to-image steganography. The stego-images can be decoded accurately even after extreme distortions such as print-and-capture.

## Background & Motivation

**Background**: Image steganography hides secret information within cover images to generate visually indistinguishable stego-images. Deep learning-based methods (especially normalizing flows/invertible neural networks) have made significant progress in steganographic quality and capacity, being widely used in scenarios like copyright protection and information provenance.

**Limitations of Prior Work**: (1) Trade-off between quality and robustness: Existing robust steganography methods (e.g., StegaStamp, ChartStamp) can resist physical distortions like print-and-capture, but suffer from poor stego-image quality (visible artifacts) and low embedding capacity (only tens of bits). (2) Limitations of CNN backbones: Existing normalizing flow methods use CNNs (such as DenseNet) as backbones for affine coupling blocks, which lack intra-channel feature fusion capability and tend to generate obvious artifacts when striving for robustness. (3) Perfect correctness requirement in message embedding: Unlike image-to-image steganography (which allows some errors), message embedding requires 100% accurate decoding, significantly increasing the difficulty.

**Key Challenge**: High robustness requires embedding information "deeper" into the image (more significant modifications), whereas high quality demands modifications to be "as small as possible"—a fundamental conflict.

**Goal**: How to achieve robust message embedding against extreme physical distortions such as print-and-capture while maintaining high stego-image quality?

**Key Insight**: It is observed that the tokenized representation of images is naturally suited for steganographic tasks. Token-level feature interaction is more abstract and robust than pixel-level CNN representations, based on which Transformer attention is introduced into normalizing flows.

**Core Idea**: Introducing the attention mechanism into normalizing flows (AttnFlow) for high-quality steganography + invertible QR code transition to adapt to the cover image + token fusion to improve quality.

## Method

### Overall Architecture

The RMSteg pipeline: (1) Encodes the secret message into a QR code image $I_q$. (2) Invertible QR Code Transition (IQRT)—transforms the QR code based on the features of the cover image $I_h$ to reduce black-white contrast and mitigate steganographic artifacts. (3) Invertible Token Fusion (ITF)—performs an invertible matrix transformation on the tokenized QR code. (4) AttnFlow—a normalizing flow model based on attention affine coupling blocks that executes information hiding. (5) Incorporates a distortion simulation module during training. The decoding process is the inverse of this flow.

### Key Designs

1. **Attention Affine Coupling Block (AACB)**:
    - **Function**: Replaces traditional CNN backbones to achieve higher-quality invertible steganographic transformations.
    - **Mechanism**: In each coupling block, the cover image tokens $T_h$ and QR code tokens $T_q$ are treated as two branches. The $T_h$ branch gradually integrates QR code information through a self-attention block $\phi(T_q)$ and a cross-attention block $\mathcal{C}(T_q, T_h^{(0)})$. The $T_q$ branch is modified via an affine transformation $T_q^{(i)} = \eta(T_h^{(i)}) + T_q^{(i-1)} \odot \exp(\rho(T_h^{(i)}))$. Cross-attention introduces the cover image's initial token $T_h^{(0)}$ as the key/value, allowing QR code information to adaptively distribute into regions compatible with the cover image.
    - **Design Motivation**: CNN backbones only perform intra-channel convolutions, lacking global interactions between patches. The attention mechanism allows long-range dependencies among tokens, making information embedding smoother and more uniform.

2. **Invertible QR Code Transition (IQRT)**:
    - **Function**: Preprocesses the QR code to make it easier to embed without generating artifacts.
    - **Mechanism**: Uses a lightweight invertible network to transform the appearance (color, brightness) of the QR code based on the features of the cover image. The transformed QR code is no longer simply black and white, but has its brightness and hue adjusted according to the cover image while still maintaining sufficient module contrast to be recognized by QR scanners. The original state is reconstructed using the inverse transform during decoding. A Gaussian kernel constraint from ArtCoder is used to ensure the recognizeability of the transformed QR code.
    - **Design Motivation**: The distribution of black-and-white QR codes differs greatly from that of natural images; direct embedding would yield prominent artifacts. Pre-disguising the QR code to closely resemble natural images can significantly reduce artifacts.

3. **Invertible Token Fusion (ITF)**:
    - **Function**: Further optimizes the distribution of QR code tokens through a learnable matrix transformation.
    - **Mechanism**: Inspired by the invertible 1x1 convolution in GLOW, a learnable matrix $\mathcal{M} \in \mathbb{R}^{N \times N}$ (initialized as an orthogonal matrix via Cholesky decomposition) is introduced to perform matrix multiplication on the QR code tokens: $T_q' = \mathcal{M} \cdot T_q$. Reconstruction is performed using the inverse matrix during decoding. Unlike the channel mixing in GLOW, ITF implements feature interaction across patches.
    - **Design Motivation**: Experiments show that introducing this small matrix alone significantly improves steganographic quality because it learns the optimal patch-level information redistribution strategy.

## Key Experimental Results

### Key Findings

- RMSteg's stego-images significantly outperform StegaStamp and ChartStamp in terms of PSNR/SSIM (with a PSNR gain of approximately 5-8 dB).
- Under extreme print-and-capture distortions, the message decoding accuracy is close to 100%.
- The embedding capacity far exceeds bit-level methods: approximately 196 bytes of information (~1500 bits) can be embedded via the QR code.
- Ablation studies demonstrate that IQRT makes the largest contribution (~3 dB PSNR gain), with AACB and ITF each contributing about 1-2 dB.
- It also exhibits good robustness against digital distortions such as JPEG compression, Gaussian noise, and brightness changes.
- The token-level representation of AACB can distribute embedded information more uniformly compared to the pixel-level representation of CNNs, thereby reducing local artifacts.

## Highlights & Insights

- **First to introduce Transformers into normalizing flow steganography**: The design of AACB is a core innovation, as token-level representations are naturally suited for robust steganography.
- **Ingenious QR code transition approach**: Modifying the inputs rather than the steganography network itself reduces the task difficulty.
- **ITF is highly simple yet effective**: A single learnable matrix can significantly improve quality, demonstrating that patch-level information redistribution is crucial.
- **High practical value**: Resistance to print-and-capture implies applicability to physical copyright protection scenarios.
- The combination of tokenized representation and normalizing flows provides a new perspective for other tasks requiring invertible transformations.

## Limitations & Future Work

- The error correction capacity limit of QR codes bounds the amount of embeddable information volume.
- The distortion simulation during training may not cover all real-world scenarios (such as non-planar printing and strong perspective distortion).
- Inference speed is affected by the number of normalizing flow layers, potentially rendering it unsuitable for real-time applications.
- Future work could explore extending attention flow to video steganography and 3D printing anti-counterfeiting.
- The performance of the current method on low-resolution images is yet to be verified.
- Generalization to more types of QR codes (e.g., Micro QR codes) remains to be confirmed.
- Multi-scale feature fusion could potentially further improve steganography quality.
- Exploring the integration of image enhancement techniques to boost robustness under weak distortion scenarios is worthwhile.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)
- [\[ICML 2025\] Benign Overfitting in Token Selection of Attention Mechanism](../../ICML2025/llm_pretraining/benign_overfitting_in_token_selection_of_attention_mechanism.md)
- [\[ICML 2025\] Towards Robust Influence Functions with Flat Validation Minima](../../ICML2025/llm_pretraining/towards_robust_influence_functions_with_flat_validation_minima.md)
- [\[CVPR 2025\] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)
- [\[ICCV 2025\] ConstStyle: Robust Domain Generalization with Unified Style Transformation](../../ICCV2025/llm_pretraining/conststyle_robust_domain_generalization_with_unified_style_transformation.md)

</div>

<!-- RELATED:END -->
