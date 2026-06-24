---
title: >-
  [Paper Note] Language-Guided Image Tokenization for Generation
description: >-
  [CVPR 2025][Image Generation][Image Tokenization] TexTok proposes incorporating textual descriptions as conditions during the image tokenization stage, offloading high-level semantic information to text. This allows image tokens to focus on encoding fine-grained visual details, thereby achieving higher compression rates while maintaining or even improving reconstruction quality, leading to a state-of-the-art (SOTA) generation FID score of 1.46 on ImageNet.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Tokenization"
  - "Text Conditioning"
  - "Image Compression"
  - "Diffusion Transformers"
  - "Efficient Generation"
date: 2026-05-08
content_hash: 2057fdaa083ca833
---

# Language-Guided Image Tokenization for Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.05796](https://arxiv.org/abs/2412.05796)  
**Code**: [https://kaiwenzha.github.io/textok/](https://kaiwenzha.github.io/textok/) (Project Page)  
**Area**: Image Generation / Diffusion Models  
**Keywords**: Image Tokenization, Text Conditioning, Image Compression, Diffusion Transformers, Efficient Generation

## TL;DR
TexTok proposes incorporating textual descriptions as conditions during the image tokenization stage, offloading high-level semantic information to text. This allows image tokens to focus on encoding fine-grained visual details, thereby achieving higher compression rates while maintaining or even improving reconstruction quality, leading to a state-of-the-art (SOTA) generation FID score of 1.46 on ImageNet.

## Background & Motivation

1. **Background**: The core of image generation relies on image tokenizers, which compress raw pixels into compact latent space representations, allowing generative models (diffusion models, autoregressive models) to operate efficiently in the compressed space. Mainstream methods include discrete tokenization via VQ-VAE/VQGAN and continuous tokenization via VAE.
2. **Limitations of Prior Work**: Current tokenization methods face a fundamental trade-off between compression rate and reconstruction quality—high compression reduces computational costs but sacrifices reconstruction quality, while pursuing quality leads to high computational overhead. This issue is particularly severe in high-resolution image generation.
3. **Key Challenge**: Image tokens must simultaneously convey high-level semantic information and low-level visual details, which cannot be achieved concurrently when the number of tokens is limited.
4. **Goal**: How to significantly reduce the number of tokens (higher compression rate) while maintaining or improving image reconstruction and generation quality.
5. **Key Insight**: Humans describe images by summarizing semantics before adding details—if textual context already carries the high-level semantics, the image tokens can allocate their entire capacity to encoding fine-grained details.
6. **Core Idea**: Let textual descriptions shoulder the burden of semantic learning, thereby freeing up the learning capacity of image tokens to capture finer visual details.

## Method

### Overall Architecture
TexTok employs a ViT-based encoder (tokenizer) and decoder (detokenizer). The inputs consist of the image and its corresponding text description (generated offline via a VLM). The encoder receives a concatenation of three types of inputs: image patch tokens, learnable image tokens, and text tokens (extracted by a frozen T5 text encoder). Only the learned image tokens in the encoder's output are retained as the latent representation. Similarly, the decoder receives three types of inputs: learnable patch tokens, image tokens, and the same text tokens, and reconstructs the image. In the generation stage, only the DiT needs to generate image tokens, as text tokens are directly provided during decoding.

### Key Designs

1. **Text Token Injection**:

    - **Function**: Inject high-level semantic information into both the encoder and decoder through text tokens, alleviating the semantic learning burden on image tokens.
    - **Mechanism**: Utilize a frozen T5 text encoder (XL for 256 resolution, XXL for 512 resolution) to encode image descriptions into text embeddings $\mathbf{T} \in \mathbb{R}^{N_t \times D}$. After aligning dimensions via linear projection, these are concatenated with image patch tokens and learnable image tokens, and fed together into the ViT encoder. On the decoder side, the same text tokens are injected. The text encoder remains frozen throughout and does not participate in training.
    - **Design Motivation**: Direct injection instead of forced alignment—unlike prior approaches that force image tokens to align with text representations, TexTok merely treats text as an auxiliary condition, avoiding the reconstruction quality degradation caused by the domain gap between visual and language representations.

2. **1D Global Image Token Architecture**:

    - **Function**: Achieve flexible and controllable token quantities, supporting different compression rates ranging from 32 to 256.
    - **Mechanism**: Adopt a 1D tokenizer paradigm, aggregating information from the image via $N$ randomly initialized learnable tokens $\mathbf{L} \in \mathbb{R}^{N \times D}$. The encoder output is projected linearly to obtain $\mathbf{Z} \in \mathbb{R}^{N \times d}$ ($d=8$). Unlike 2D spatial tokens that require a fixed downsampling rate, the number of 1D global tokens can be configured freely.
    - **Design Motivation**: A flexible token budget allows researchers to balance precision and efficiency as needed, and makes the benefits of text conditioning more pronounced at lower token counts.

3. **Text Utilization Strategy in Generation Stage**:

    - **Function**: Seamlessly utilize textual information during the inference stage.
    - **Mechanism**: For text-to-image generation, the given text description is used directly. For class-conditional generation, the DiT generates latent tokens based on the class, and then an unseen description is sampled from a pre-generated list of descriptions for that class. This description is fed alongside the generated latent tokens into the decoder to produce the final image. During the generation stage, only image tokens need to be generated, while text tokens are provided for free.
    - **Design Motivation**: Text descriptions are naturally available in text-to-image tasks without additional annotation overhead. For class-conditional tasks, they are generated offline in batches via a VLM, which incurs a low, one-time cost and is highly reusable.

### Loss & Training
Training uses a combination of $\ell_2$ reconstruction loss, GAN adversarial loss, perceptual loss, and LeCAM regularization loss. The GAN discriminator employs a StyleGAN discriminator (~24M parameters). The encoder and decoder each consist of a 12-layer ViT-Base (~176M parameters), with a token channel dimension of $d=8$. The DiT generator uses a patch size of 1 and is trained for 350 epochs.

## Key Experimental Results

### Main Results

| Setting | Resolution | Token Count | rFID (Reconstruction) | gFID (Generation) | Gain over Baseline |
|------|--------|----------|-----------|-----------|------------------|
| TexTok-32 | 256×256 | 32 | 2.40 | 3.55 | rFID -37.2%, gFID -28.6% |
| TexTok-64 | 256×256 | 64 | 1.53 | 2.88 | rFID -25.0%, gFID -12.7% |
| TexTok-256 | 256×256 | 256 | 0.69 | 2.68 | rFID -24.2%, gFID -7.9% |
| TexTok-32 | 512×512 | 32 | 2.33 | 3.61 | rFID -69.7%, gFID -60.8% |
| TexTok-256 + DiT-XL | 256×256 | 256 | - | **1.46** | SOTA |
| TexTok-256 + DiT-XL | 512×512 | 256 | - | **1.62** | SOTA |

TexTok surpasses the performance of the original DiT using 4096 tokens at 512 resolution using only 32 tokens, achieving a **93.5× inference speedup**.

### Ablation Study

| Configuration | rFID (256) | Description |
|------|-----------|------|
| TexTok (Full) | 1.04 | Text in both encoder and decoder |
| Tokenizer only | 1.11 | Text in encoder only |
| Detokenizer only | 1.28 | Text in decoder only |
| Baseline (w/o text) | 1.49 | No text conditioning |
| TexTok + Text-to-Image | 2.82 FID, 29.23 CLIP | T2I task also benefits |

### Key Findings
- **Fewer tokens lead to larger gains from text**: At 32 tokens, rFID improves by 37.2% (256 resolution) and 69.7% (512 resolution), while at 256 tokens, the improvement is around 24%. This indicates that text shoulders more semantic responsibility under low-bandwidth constraints.
- **High-resolution gains are more significant**: The improvement brought by text conditioning at 512 resolution is nearly double that of 256 resolution, as high-resolution images exhibit more semantic redundancy.
- TexTok can achieve the same rFID as the Baseline with half the number of tokens (for 256 resolution) or even a quarter of the tokens (for 512 resolution).
- Injecting text into both the encoder and decoder yields the best performance, with a larger contribution coming from the encoder side.

## Highlights & Insights
- **Utilizing text at the tokenization stage rather than the generation stage**: This is an counter-intuitive yet highly effective design. Previously, text conditioning was always applied during the generation stage, whereas TexTok is the first to shift it forward to the tokenization stage, offloading the semantic burden to the text—a prime example of the "separation of concerns" principle.
- **A free lunch**: In text-to-image tasks, the text descriptions themselves are already part of the training data, requiring no additional annotation. In class-conditional tasks, descriptions only need to be generated once offline via a VLM, incurring almost no extra overhead for subsequent training and inference.
- **Practical value of 93.5× inference speedup**: Compressing DiT from 4096 tokens to 32 tokens at 512 resolution without performance degradation is of great significance for deploying high-resolution generative models. This approach can be directly transferred to higher-dimensional scenarios like video generation.

## Limitations & Future Work
- **Dependency on text description quality**: Reconstruction and generation quality are affected by the correctness of the text descriptions; incorrect VLM-generated descriptions may guide the tokenizer to encode incorrect semantics.
- **Text still required during inference**: Class-conditional generation requires providing text descriptions during inference, which increases system complexity.
- **Validated only in specific domains (faces/objects)**: While exhibiting significant performance on ImageNet, its effectiveness in more complex domains (such as medical or remote sensing imagery) remains unverified.
- Integrating text conditions with other modal conditions (such as depth maps or segmentation masks) can be explored to further improve tokenization efficiency.
- Adaptive and dynamic determination of the token count per image (based on image complexity) could further optimize efficiency.

## Related Work & Insights
- **vs TiTok**: TiTok also uses 1D global tokens but does not utilize text conditioning; TexTok outperforms it comprehensively at the same token counts.
- **vs SD-VAE**: SD-VAE uses 1024 2D spatial tokens ($d=4$). Even the baseline (without text) surpasses its reconstruction performance using only 32 1D tokens, and the advantage becomes even larger with text.
- **vs Text alignment methods (e.g., LQAE, Spae)**: These methods force image tokens to align with the text space, resulting in degraded reconstruction quality. TexTok merely uses text as an auxiliary external condition to supplement, rather than replace, visual representations.
- The concept of text-assisted tokenization can be transferred to video tokenization (text describing temporal variations) and 3D tokenization (text describing spatial structures).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce text conditioning at the tokenization stage; the concept is elegant, simple, and highly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation studies covering multiple resolutions, token counts, and tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation with highly logical experimental presentation.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm for efficient image generation; the 93.5× speedup is highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Yo'Chameleon: Personalized Vision and Language Generation](yochameleon_personalized_vision_and_language_generation.md)
- [\[CVPR 2025\] Visual Lexicon: Rich Image Features in Language Space](visual_lexicon_rich_image_features_in_language_space.md)
- [\[CVPR 2025\] Efficient Long Video Tokenization via Coordinate-based Patch Reconstruction](efficient_long_video_tokenization_via_coordinate-based_patch_reconstruction.md)
- [\[CVPR 2026\] MacTok: Robust Continuous Tokenization for Image Generation](../../CVPR2026/image_generation/mactok_robust_continuous_tokenization_for_image_generation.md)
- [\[ICML 2025\] Representative Language Generation](../../ICML2025/image_generation/representative_language_generation.md)

</div>

<!-- RELATED:END -->
