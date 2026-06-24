---
title: >-
  [Paper Note] Dual Diffusion for Unified Image Generation and Understanding
description: >-
  [Image Generation] The Dual Diffusion Transformer (D-DiT) is proposed, which simultaneously uses continuous diffusion to model image distribution and discrete masked diffusion to model text distribution within a single MM-DiT architecture. It represents the first end-to-end fully diffusion-based multimodal model, supporting a comprehensive suite of tasks including image generation, image captioning, and visual question answering.
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 3921945bcb7fe51d
---

# Dual Diffusion for Unified Image Generation and Understanding

## TL;DR

The Dual Diffusion Transformer (D-DiT) is proposed, which simultaneously uses continuous diffusion to model image distribution and discrete masked diffusion to model text distribution within a single MM-DiT architecture. It represents the first end-to-end fully diffusion-based multimodal model, supporting a comprehensive suite of tasks including image generation, image captioning, and visual question answering.

## Background & Motivation

Significant methodological fragmentation exists in the current multimodal generation field: diffusion models excel in text-to-image (T2I) generation (e.g., Stable Diffusion, FLUX), whereas autoregressive models dominate vision-language understanding (I2T) (e.g., LLaVA, BLIP-2). A natural question arises: can these two capabilities be unified within a single model?

For autoregressive models, extensive work has demonstrated their ability to generate images in reverse (LLM + visual tokenizer). However, for diffusion models, establishing reverse text generation has remained challenging due to the lack of empirically effective discrete diffusion processes. Existing multimodal diffusion models (e.g., UniDiffuser, Versatile Diffusion) either rely on autoregressive models to decode text latents, or append diffusion losses onto pretrained LLMs (e.g., Show-O, Transfusion), fundamentally still relying on next-token prediction for text generation.

The core motivation of this paper is to leverage recent advances in discrete masked diffusion to construct a **purely diffusion-based** multimodal model that can simultaneously perform both image generation and text generation without any autoregressive components.

## Method

### Overall Architecture

D-DiT is based on the MM-DiT dual-branch Transformer architecture of SD3. The image branch outputs velocity field predictions for flow matching, while the text branch outputs denoised token predictions for discrete masked diffusion. The two branches interact through cross-attention in every attention layer. During training, a joint loss function $L_{\text{dual}} = L_{\text{image}} + \lambda_{\text{text}} L_{\text{text}}$ is utilized to simultaneously train the conditional generation of both modalities.

### Key Designs

#### 1. Cross-Modal Joint Diffusion Training

- **Function**: To simultaneously learn $p(\mathbf{x}^{(\text{img})}|\mathbf{x}^{(\text{txt})})$ and $p(\mathbf{x}^{(\text{txt})}|\mathbf{x}^{(\text{img})})$ within the same model.
- **Mechanism**: The image branch employs continuous flow matching (velocity field regression), and the text branch employs discrete masked diffusion (masked token prediction). During training, noise is not applied to both modalities simultaneously; when training text generation, the image remains noise-free, and vice versa.
- **Design Motivation**: Different modalities are naturally suited for different diffusion methods: continuous vectors are ideal for flow matching, and discrete tokens are ideal for absorbing state diffusion. Joint training allows both modalities to share Transformer parameters, forming a unified representation.

#### 2. Masked Diffusion-Based Text In-filling Mechanism

- **Function**: To support tasks requiring conditional text generation, such as visual question answering (VQA).
- **Mechanism**: During sampling, the question tokens are kept unchanged (unmasked), while only the answer part is masked and iteratively denoised. The intrinsic in-filling capability of discrete diffusion is leveraged to complete conditional text completion.
- **Design Motivation**: Previous multimodal diffusion models (e.g., UniDiffuser, Versatile Diffusion) perform text diffusion in the CLIP latent space, making token-level conditional completion impossible. Masked diffusion operates directly in the token space, naturally supporting in-filling.

#### 3. Initialization from SD3 Pretrained Weights and Three-Stage Training

- **Function**: To rapidly adapt text generation capabilities while maintaining image generation quality.
- **Mechanism**: The model is initialized with the pretrained DiT weights of SD3, and a linear head is added on top of the text branch. The three-stage training consists of: (1) joint pretraining on 30M image-text pairs for 60K steps; (2) fine-tuning on high-quality understanding datasets for 200K steps while unfreezing mask token embeddings; (3) visual instruction fine-tuning.
- **Design Motivation**: SD3's MM-DiT already possesses strong image-text alignment capabilities, requiring only a small amount of text data to adapt to text generation. Utilizing the existing `<extra_id0>` from the T5 encoder as the mask token reduces the domain gap.

### Loss & Training

$$L_{\text{image}} = \mathbb{E}_{t, q^{(\text{img})}} \|\mathbf{v}_\theta(\mathbf{x}_t^{(\text{img})}, t, \mathbf{x}^{(\text{txt})}) - (\epsilon - \mathbf{x}^{(\text{img})})\|_2^2$$

$$L_{\text{text}} = \mathbb{E}_{q^{(\text{txt})}} \left[-\frac{1}{K}\sum_{i=1}^{K} \log[\mathbf{x}_\theta(\mathbf{x}_{t_i}^{(\text{txt})}, \mathbf{x}^{(\text{img})}) \cdot \mathbf{x}] / t_i \right]$$

$$L_{\text{dual}} = L_{\text{image}} + \lambda_{\text{text}} L_{\text{text}}$$

## Key Experimental Results

### Main Results

**T2I Generation (GenEval)**:

| Model | Overall | Single Obj | Two Obj | Counting | Colors | Position | Color Attr |
|------|---------|-----------|---------|----------|--------|----------|------------|
| SD3 | 0.62 | 0.98 | 0.74 | 0.63 | 0.67 | 0.34 | 0.36 |
| D-DiT (ours) | **0.65** | 0.97 | **0.80** | 0.54 | **0.76** | 0.32 | **0.50** |
| Show-O | 0.68 | 0.98 | 0.80 | 0.66 | 0.84 | 0.31 | 0.50 |

**I2T Understanding (VQA/Captioning)**:

| Model | VQAv2 | VizWiz | GQA | POPE | MME |
|------|-------|--------|-----|------|-----|
| D-DiT 256 | 60.7 | 33.9 | 52.2 | 79.7 | 1089 |
| D-DiT 512 | 64.1 | 35.1 | 55.1 | 83.2 | 1213 |
| Show-O | 61.0 | 28.6 | 48.7 | 82.0 | 1097 |

### Ablation Study

- Joint training **does not lead to catastrophic forgetting of image generation**: D-DiT maintains the image generation quality of SD3, with some metrics (such as color accuracy) even improving.
- For text in-filling, 16 sampling steps are sufficient for VQA (short text), while 256 steps are used for dialogue/long text.

### Key Findings

- After initializing from the SD3 pretrained checkpoint, only 25B text tokens of training are required to generate meaningful text outputs.
- For the first time, a pure diffusion model has achieved competitive performance with hybrid models like Show-O on VQA tasks.
- The parallel sampling characteristic of masked diffusion allows text generation to be performed non-autoregressively.

## Highlights & Insights

1. **Paradigm Breakthrough**: The first end-to-end purely diffusion-based multimodal model, proving that diffusion models can simultaneously model continuous images and discrete text without relying on autoregressive decoding.
2. **Elegant Training Objective**: The joint loss function is extremely concise, being merely a weighted sum of the diffusion losses of the two modalities, eliminating the need for complex training strategies.
3. **Strong Transferability**: By leveraging SD3 pretrained weights and the T5 mask token, the model rapidly acquires text generation capabilities with a minimal amount of text data.
4. **Architectural Symmetry**: Image and text branches are processed symmetrically within the same Transformer, naturally interacting via cross-attention.

## Limitations & Future Work

1. **Discrepancy in Text Understanding Performance**: Compared to specialized VLMs (e.g., LLaVA-1.5, InternVL), there is still a noticeable gap in VQA performance.
2. **Lack of Pure Text Generation Training**: The model is never trained on pure text data, which limits its language modeling capabilities.
3. **Small Scale of Training Data**: The model is trained on only ~40M image-text pairs, far fewer than mainstream VLMs.
4. **Inference Efficiency**: Iterative sampling of discrete diffusion is slower than autoregressive generation, particularly for long text which requires 256 steps.
5. **Model Scale**: With 2B parameters, it is much smaller than mainstream VLMs, and its scaling behavior warrants exploration.

## Related Work & Insights

- **Show-O / Transfusion**: Hybrid diffusion + autoregressive schemes, which still rely on AR for text generation.
- **MDLM / SEDD**: The theoretical foundation for discrete diffusion language modeling, rendering pure diffusion-based text generation possible.
- **SD3 / FLUX**: The foundation of the MM-DiT architecture, providing a natural dual-branch structure.
- **Insight**: The key to cross-modal unification lies not in unifying the diffusion process itself, but in unifying the architecture and sharing parameters, allowing different modalities to utilize their most suitable diffusion methods.

## Rating

⭐⭐⭐⭐

Highly innovative (the first purely diffusion-based multimodal model), with a simple and elegant method. However, a performance gap remains compared to autoregressive VLMs, and its practicality needs further improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [\[CVPR 2025\] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)
- [\[NeurIPS 2025\] InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](../../NeurIPS2025/image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)
- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)
- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](multi-party_collaborative_attention_control_for_image_customization.md)

</div>

<!-- RELATED:END -->
