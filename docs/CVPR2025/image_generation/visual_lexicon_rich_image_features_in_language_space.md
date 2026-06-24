---
title: >-
  [Paper Note] Visual Lexicon: Rich Image Features in Language Space
description: >-
  [CVPR 2025][Image Generation][Visual Lexicon] ViLex proposes a visual encoder that encodes images into the text vocabulary space. Through self-supervised training using a frozen text-to-image (T2I) diffusion model, the generated image tokens capture both high-level semantics and fine-grained visual details, outperforming conventional methods in both image reconstruction and visual understanding tasks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Visual Lexicon"
  - "Image Representation"
  - "Diffusion Models"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: 59cde921fb2e4aae
---

# Visual Lexicon: Rich Image Features in Language Space

**Conference**: CVPR 2025  
**arXiv**: [2412.06774](https://arxiv.org/abs/2412.06774)  
**Code**: None  
**Area**: Image Generation / Visual Representation Learning  
**Keywords**: Visual Lexicon, Image Representation, Diffusion Models, Image Generation, Vision-Language Models

## TL;DR

ViLex proposes a visual encoder that encodes images into the text vocabulary space. Through self-supervised training using a frozen text-to-image (T2I) diffusion model, the generated image tokens capture both high-level semantics and fine-grained visual details, outperforming conventional methods in both image reconstruction and visual understanding tasks.

## Background & Motivation

**Background**: Image representation in computer vision has long been divided into two paradigms: understanding-oriented representations represented by CLIP/DINO, which capture high-level semantics but discard pixel-level details; and reconstruction-oriented representations represented by VAE/MAE, which retain visual details but exhibit weak semantic information.

**Limitations of Prior Work**: These two types of representations perform poorly outside their respective domains. CLIP features cannot be used for high-fidelity image reconstruction, while VAE features underperform in downstream understanding tasks (such as linear probing). Methods like DeDiffusion attempt to invert images into discrete text tokens, but the expressive power of text tokens is limited, and the reconstruction quality is constrained by the description limits of natural language.

**Key Challenge**: There is a fundamental trade-off between semantic representations and reconstruction representations—can a single representation achieve optimal performance in both image generation and visual understanding?

**Goal**: To design a unified image representation that can serve as a "text prompt" for T2I models to achieve high-fidelity image reconstruction, and simultaneously function as a visual encoder to enhance understanding capabilities in VLMs.

**Key Insight**: The authors observe that diffusion models naturally encode rich semantic and visual details during the denoising process. Instead of extracting features directly from diffusion models, it is better to leverage them as decoders within an autoencoder framework, training a lightweight encoder to "distill" the rich visual knowledge within the diffusion model.

**Core Idea**: Map images into the text vocabulary embedding space of T2I diffusion models, and utilize the frozen diffusion model as a decoder for image reconstruction training, thereby allowing the visual representation to inherit the semantic-visual richness of the diffusion model.

## Method

### Overall Architecture

ViLex adopts an autoencoder architecture: the encoder consists of a ViT visual encoder combined with an attention pooling layer, which converts the input image into a set of ViLex tokens (lying within the text vocabulary embedding space); the decoder is a frozen, pre-trained T2I diffusion model (Imagen). During training, only the encoder parameters are updated, with gradients backpropagated via the diffusion model's image reconstruction loss. Once trained, ViLex tokens can be directly fed into the frozen text encoder and diffusion model as "text prompts," reconstructing highly consistent images in both semantics and visual details without requiring actual text.

### Key Designs

1. **Image-to-Text Projection**:

    - **Function**: Convert patch-level visual features output by ViT into vocabulary embeddings compatible with the text encoder of the T2I model.
    - **Mechanism**: A multi-head cross-attention layer with $n$ learnable queries is utilized, taking the $k$ patch tokens from the ViT as keys and values, to pool the visual information into $n$ ViLex embedding vectors. These embeddings are trained to project implicitly into the latent space of the BPE vocabulary lookup matrix $\mathcal{V}$, ensuring compatibility with the T2I diffusion model.
    - **Design Motivation**: Since the text vocabulary space is compositional, ViLex tokens can be used independently or concatenated with natural language just like real text tokens, enabling multimodal image generation.

2. **TailDrop Dynamic Token Compression Strategy**:

    - **Function**: Randomly drop the last $k$ ViLex tokens during training to encourage earlier tokens to carry richer semantic information.
    - **Mechanism**: Similar to the variable bitrate strategy in SoundStream, since earlier tokens are more frequently used in isolation for image generation during training, the model is forced to encode as much semantic information as possible into the earlier tokens. During inference, the number of tokens can be dynamically adjusted to balance compression rate and details.
    - **Design Motivation**: Different images contain varying amounts of information. TailDrop provides a flexible token budgeting mechanism, allowing a seamless transition from coarse semantics with 1 token to fine-grained reconstruction with 75 tokens.

3. **Text-Free Guidance (TFG)**:

    - **Function**: Balance the influence of ViLex visual tokens versus text prompts during multimodal image generation.
    - **Mechanism**: Similar to Classifier-Free Guidance, TFG combines noise predictions conditioned on vision+text and vision-only conditions: $\epsilon_{\text{tfg}} = \epsilon_\theta(x_t, v) + w_{\text{tfg}} \cdot (\epsilon_\theta(x_t, [v,c]) - \epsilon_\theta(x_t, v))$, where the guidance scale $w_{\text{tfg}}$ controls the text's influence on the generated results.
    - **Design Motivation**: TFG allows ViLex to achieve zero-shot, unsupervised DreamBooth-style multimodal image generation without fine-tuning the T2I model or altering its architecture.

### Loss & Training

The training loss is the standard diffusion model denoising objective: $\mathcal{L}_{\text{denoise}} = \mathbb{E}_{x_0, \epsilon, t}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$. The training data comes from the WebLI dataset, which can be trained using only images or jointly with image-text pairs. It uses the Adafactor optimizer with a batch size of 2048, for 300K steps (approx. 2.5 days on 64 TPUv5). ViT is initialized from pre-trained SigLIP, while the attention pooling layer is randomly initialized, using different learning rates (ViT: $1\times10^{-5}$, pooling layer: $3\times10^{-4}$).

## Key Experimental Results

### Main Results

| Method | Token Count | FID ↓ | IS ↑ |
|------|---------|-------|------|
| Imagen (text→image) | - | 6.52 | 14.06 |
| DeDiffusion | 75 | 3.89 | 14.68 |
| ViLex | 1 | 3.65 | 15.33 |
| ViLex | 16 | 2.91 | 15.42 |
| ViLex | 75 | 2.07 | 15.88 |

In human evaluations, ViLex achieves win rates of 98%/95%/98% against DeDiffusion and 91%/76%/90% against DALL·E 3 on layout/semantic/style consistency, respectively.

### Ablation Study

| Backbone | FID ↓ | COCOcap | TextCaps | VQAv2-Val | SciQA | RC-val |
|----------|-------|---------|----------|-----------|-------|--------|
| Original SigLIP | 2.54 | 139.7 | 122.1 | 81.4 | 85.9 | 66.2 |
| ViLex SigLIP | 2.38 | 141.5 | 124.0 | 81.6 | 87.9 | 67.6 |
| ViLex (Full Model incl. Pooling Layer) | **2.07** | **142.8** | **137.7** | - | - | - |

### Key Findings

- Even with only **1 continuous token**, ViLex's FID (3.65) outperforms DeDiffusion using 75 discrete tokens (3.89), demonstrating that the expressive power of continuous embeddings far exceeds that of discrete text tokens.
- Replacing the original SigLIP with the ViLex encoder consistently improves performance across 15 VLM benchmarks, including image/video captioning, VQA, and referring segmentation, proving that reconstruction and understanding can synergistically enhance each other.
- As the number of tokens increases from 1 to 75, the reconstruction progresses from capturing high-level semantics (category, count, pose) to fine-grained visual details (color, texture, object shape), demonstrating an elegant content hierarchical scaling property.

## Highlights & Insights

- **Using the diffusion model as an autoencoder decoder** is a clever design: instead of extracting features directly from diffusion models (such as ODISE, l-DAE), it allows the diffusion model to "teach" the encoder to learn rich representations, keeping the encoder lightweight and transferable to understanding tasks.
- The finding that **continuous tokens outperform discrete tokens** is impressive: a single ViLex token outperforms 75 DeDiffusion text tokens, quantitatively proving the expressiveness bottleneck of natural language.
- The **TailDrop strategy** can be directly transferred to any scenario requiring variable token length, such as video token compression or visual token budget control in multimodal LLMs.

## Limitations & Future Work

- Currently, image reconstruction is only validated at 64×64 resolution; the performance and efficiency at higher resolutions remain to be checked.
- ViLex is dependent on a specific T2I model (Imagen); its generalizability to other diffusion models (such as SDXL, Flux) remains unknown.
- Although zero-shot text-plus-vision DreamBooth eliminates the need for fine-tuning, its identity preservation precision may not match that of specially optimized LoRA-based solutions.
- The upper limit of tokens in the attention pooling layer is constrained by the 77-token context length of the CLIP text encoder. Scaling to longer sequences would require replacing the text encoder.

## Related Work & Insights

- **vs DeDiffusion**: DeDiffusion converts images into discrete text tokens before generation, which is limited by the expressive power of natural language. ViLex bypasses discretization, encoding visual information directly in the continuous embedding space, resulting in significantly superior quality.
- **vs CLIP/SigLIP**: These methods only optimize for understanding objectives. ViLex introduces an additional diffusion reconstruction objective on top of SigLIP, simultaneously enhancing both understanding and generation capabilities.
- **vs Textual Inversion/DreamBooth**: These methods require test-time fine-tuning for each instance. ViLex is a general-purpose encoder that obtains identity embeddings in a single forward pass.

## Rating

- Novelty: ⭐⭐⭐⭐ Cleverly repositioning the diffusion model as an autoencoder decoder to learn a visual lexicon offers a refreshing perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative and human evaluations are provided for both generation and understanding, though high-resolution validation is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, beautiful tables and figures, and a complete storyline.
- Value: ⭐⭐⭐⭐ Unifying representation directions for both generation and understanding holds significant research value, and techniques like TailDrop can be widely reused.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Do Visual Imaginations Improve Vision-and-Language Navigation Agents?](do_visual_imaginations_improve_vision-and-language_navigation_agents.md)
- [\[CVPR 2025\] CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](ctrl-o_language-controllable_object-centric_visual_representation_learning.md)
- [\[CVPR 2025\] Probability Density Geodesics in Image Diffusion Latent Space](probability_density_geodesics_in_image_diffusion_latent_space.md)
- [\[CVPR 2025\] Language-Guided Image Tokenization for Generation](language-guided_image_tokenization_for_generation.md)
- [\[CVPR 2025\] CleanDIFT: Diffusion Features without Noise](cleandift_diffusion_features_without_noise.md)

</div>

<!-- RELATED:END -->
