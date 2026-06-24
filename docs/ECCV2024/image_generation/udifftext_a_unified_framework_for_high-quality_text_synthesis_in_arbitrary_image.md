---
title: >-
  [Paper Note] UDiffText: A Unified Framework for High-quality Text Synthesis in Arbitrary Images via Character-aware Diffusion Models
description: >-
  [ECCV 2024][Image Generation][Text Synthesis] This paper proposes UDiffText, which achieves high-precision and visually harmonious text synthesis in arbitrary images by replacing the CLIP encoder with a lightweight character-level text encoder, fine-tuning cross-attention layers with a local attention loss (based on character segmentation maps) and a Scene Text Recognition (STR) loss, and applying a noised latent refinement step during inference. It outperforms state-of-the-a…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Text Synthesis"
  - "diffusion model"
  - "Character-level Encoder"
  - "Cross-Attention"
  - "Scene Text Editing"
date: 2026-05-08
content_hash: 9d2cda37d6d31d43
---

# UDiffText: A Unified Framework for High-quality Text Synthesis in Arbitrary Images via Character-aware Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2312.04884](https://arxiv.org/abs/2312.04884)  
**Code**: [https://github.com/ZYM-PKU/UDiffText](https://github.com/ZYM-PKU/UDiffText)  
**Area**: Text Image Generation / Scene Text Editing  
**Keywords**: Text Synthesis, diffusion model, Character-level Encoder, Cross-Attention, Scene Text Editing

## TL;DR

This paper proposes UDiffText, which achieves high-precision and visually harmonious text synthesis in arbitrary images by replacing the CLIP encoder with a lightweight character-level text encoder, fine-tuning cross-attention layers with a local attention loss (based on character segmentation maps) and a Scene Text Recognition (STR) loss, and applying a noised latent refinement step during inference. It outperforms state-of-the-art (SOTA) methods in sequence accuracy (SeqAcc) across various scenarios.

## Background & Motivation

**Background**: Text-to-Image (T2I) generation methods based on diffusion models (e.g., Stable Diffusion, DALL-E 3, Midjourney) have achieved phenomenal performance in general image generation, but they often produce severe spelling errors—such as missing, incorrect, or redundant characters—when generating images containing text.

**Limitations of Prior Work**:
1. CLIP/T5 text encoders tokenize input text at the word or subword level, failing to perceive the internal character structure within words.
2. Even when DALL-E 3 employs a massive T5 encoder, rendering text enclosed in quotes remains unstable.
3. Existing methods either rely on extra glyph images as visual guidance (e.g., GlyphControl, GlyphDraw) or require concatenated segmentation masks as input (e.g., TextDiffuser), resulting in complex pipelines and limited flexibility.

**Key Challenge**: The catastrophic neglect and incorrect attribute binding of T2I models lead to inaccurate text rendering, while current encoding paradigms cannot provide precise character-level guidance.

**Key Insight**: Designing a lightweight character-level encoder coupled with local attention constraints and test-time latent refinement allows substantial improvements in text rendering accuracy by only fine-tuning the cross-attention parameters.

**Core Idea**: Using a character-level encoder to provide highly discriminative character embeddings, enabling the cross-attention layers to learn the precise spatial regions corresponding to each character.

## Method

### Overall Architecture

Built on the Inpainting version of Stable Diffusion v2.0. The inputs are the noisy text image $\mathbf{x}_0 + \mathbf{n}$, the binary mask of the text region $\mathcal{M}$, the masked image $\mathbf{x}_{\mathcal{M}}$, and the target text $\mathcal{T}$. The output is the image with the target text rendered within the masked region. The overall pipeline consists of three steps: (1) training a character-level encoder, (2) fine-tuning the U-Net cross-attention layers, and (3) performing noised latent refinement during inference.

### Key Designs

1. **Character-level (CL) Text Encoder**:

    - **Function**: Encodes the target words into character-level embedding sequences, replacing the original CLIP encoder.
    - **Mechanism**: A codebook maps character indices to learnable embeddings, which are then passed through a Transformer with position embeddings to produce the output $(B, L, d_{emb})$.
    - **Design Motivation**: CLIP/T5 tokenize at the word/subword level and lack character structure awareness, while character-level models like ByT5 contain too many parameters (~20B scale). The proposed encoder has only 302M parameters.
    - **Training Strategy**: It is trained using a contrastive learning loss $\mathcal{L}_{clip} = -\text{CS}(W_t \mathbf{e}_{text}, W_i \mathbf{e}_{image})$ to align text and ViTSTR image features, along with a multi-label classification loss $\mathcal{L}_{ce} = \text{CE}(\mathcal{H}_{MLC}(\mathbf{e}_{text}), Ids)$ to ensure high discriminability of the embeddings.
    - Total Loss: $\mathcal{L} = \mathcal{L}_{clip} + \lambda_{ce} \mathcal{L}_{ce}$

2. **Local Attention Loss + STR Loss Fine-tuning**:

    - **Function**: Supervises the cross-attention maps using character segmentation maps, forcing the attention of each character to precisely focus on its corresponding region.
    - **Mechanism**: For a character sequence $\mathcal{T} = \{c^1, c^2, \dots, c^L\}$, its corresponding segmentation map is $\mathcal{S}_T = \{S^1, S^2, \dots, S^L\}$. The cross-attention map $\mathcal{A}_i$ is extracted from the U-Net, and the local attention loss is computed as:
    $$\mathcal{L}_{loc} = \frac{1}{C}\sum_{i=1}^{C}\left\{\frac{1}{L}\sum_{j=1}^{L}\max(\mathbb{G}(A_i^j) \odot (J - S^j)) - \frac{1}{L}\sum_{j=1}^{L}\max(\mathbb{G}(A_i^j) \odot S^j)\right\}$$
    - Auxiliary STR loss: A pre-trained OCR model is used to recognize the text region of the denoised result, computing the cross-entropy loss $\mathcal{L}_{str} = \text{CE}(S(D_\theta(\cdot) \odot \mathcal{M}), \mathcal{T})$.
    - **Design Motivation**: Relying solely on the L2 distance of the DSM loss (which measures pixel mean differences) cannot guarantee character rendering accuracy. The local attention loss coaxes attention maps to focus on correct character steps.
    - **Key Point**: During training, only the cross-attention parameters are updated (75.9M / 891M), freezing the remaining parameters to preserve general image generation capabilities.

3. **Noised Latent Refinement (Inference Stage)**:

    - **Function**: Optimizes the initial noise and latent variables at each step during inference to address catastrophic neglect.
    - **Mechanism**: (a) Sample $N$ initial noises, run 2 quick denoising steps, and select the noise with the minimum $\mathcal{L}_{aae}$ as the optimal initial noise; (b) refine the latent variable at each timestep using gradient updates: $\mathbf{z}_t' = \mathbf{z}_t - \alpha_t \cdot \nabla_{\mathbf{z}_t} \mathcal{L}_{aae}$.
    - $\mathcal{L}_{aae}$ Design: Maximizes the minimum of the maximum values of each character's attention map within the mask region, defined as: $\mathcal{L}_{aae} = -\frac{1}{C}\sum_{i=1}^{C}\min_{1 \le j \le N}(\max(\mathbb{G}(A_i^j) \odot \mathcal{M}))$.
    - **Design Motivation**: Even after fine-tuning with local attention loss, the model might still omit some characters. Refinement guarantees that every character is "activated" in the attention maps.

### Loss & Training

Complete training objective: $\mathcal{L} = \mathcal{L}_{DSM} + \lambda_{loc}\mathcal{L}_{loc} + \lambda_{str}\mathcal{L}_{str}$

Hyperparameter setup: $\lambda_{ce} = 0.1$, $\lambda_{loc} = 0.01$, $\lambda_{str} = 0.001$

Training strategy:
- The CL encoder is first pre-trained for 8k steps (batch size = 256, lr = 1e-5) using contrastive learning, then frozen.
- The U-Net is trained on SynthText for 100k steps, then on LAION-OCR for another 100k steps (batch size = 64, lr = 5e-5, image size 512×512).
- Inference uses 50 sampling steps with a CFG scale of 5.0.

## Key Experimental Results

### Main Results

Compared with MOSTEL, SD-Inpainting, DiffSTE, and TextDiffuser on three datasets: ICDAR13, TextSeg, and LAION-OCR:

| Method | SeqAcc-Recon (ICDAR13) | SeqAcc-Edit (ICDAR13) | FID↓ | LPIPS↓ |
|------|----------------------|---------------------|------|--------|
| MOSTEL | 68.0% | 28.0% | 25.09 | 0.0605 |
| SD-Inpainting | 29.0% | 7.0% | 26.78 | 0.0696 |
| DiffSTE | 37.0% | 29.0% | 51.67 | 0.1050 |
| TextDiffuser | 81.0% | 75.0% | 32.25 | 0.0834 |
| **UDiffText** | **91.0%** | **83.0%** | **15.79** | **0.0564** |

TextSeg SeqAcc-Recon: 93% (vs TextDiffuser 68%), LAION-OCR SeqAcc-Editing: 78% (vs TextDiffuser 64%).

### Ablation Study

| Configuration | SeqAcc-Recon (%) | Description |
|------|----------------|------|
| Base (SD v2.0 + CLIP) | 8.0 | Baseline |
| + CL encoder | 40.0 | Character encoder +32% |
| + $\mathcal{L}_{loc}$ | 54.0 | Local attention loss +14% |
| + $\mathcal{L}_{str}$ | 65.0 | STR loss +11% |
| + Refinement | 76.0 | Inference optimization +11% |

### Key Findings
- The CL encoder contributes the most (+32%), serving as the core component to resolve spelling issues.
- The local attention loss focuses attention maps on correct character regions, visualizable as precise individual attention maps.
- Refinement acts as the final touch, addressing remaining catastrophic neglect issues.
- On SimpleBench, the text rendering accuracy of SDXL is improved from 8.0% to 60.0%.

## Highlights & Insights

- **Knowledge Complement Fine-tuning**: Updating only cross-attention layers allows the model to learn character shapes and appearances while preserving general image generation capabilities, serving as an efficient fine-tuning paradigm.
- **Character Segmentation Maps as Attention Supervision**: Applying spatial segmentation information to supervise attention mechanisms is a highly generalizable technique that can be extended to other generation tasks requiring precise spatial control.
- **Inference-Time Latent Refinement**: Directing gradient optimization towards the noised latent variables to address catastrophic neglect provides an effective inference-time enhancement strategy that avoids retraining.
- **Lightweight Character Encoder**: Compared to massive models like ByT5 (20B), achieving robust character-level encoding with only 302M parameters is both simple and highly practical.

## Limitations & Future Work

- When scene backgrounds are too simple, the model's ability to render text purely based on visual context is somewhat constrained.
- Current support is limited to short texts of up to 12 characters, making it unable to generate long paragraphs or documents.
- Inference refinement introduces extra forward and backward propagation steps, which increases inference latency.
- Future work can explore word-by-word synthesis to handle longer text sequences.

## Related Work & Insights

- **vs TextDiffuser**: TextDiffuser concatenates segmentation masks as inputs and utilizes a character-aware loss, but requires an auxiliary segmentor module. UDiffText is cleaner, operating solely on text conditioning.
- **vs GlyphControl**: GlyphControl relies on ControlNet to inject glyph reference images, which requires pre-rendering. UDiffText obtains guidance directly from the character encoder.
- **vs Attend-and-Excite**: Borrowed the concepts from generative semantic nursing, but tailored a new target $\mathcal{L}_{aae}$ specifically for text-rendering scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Complete three-part design featuring character-level encoding, local attention loss, and inference refinement.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset, multi-task evaluations with solid ablation studies and convincing visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Highly logical, with detailed descriptions of methodology and clean mathematical formulation.
- **Value**: ⭐⭐⭐⭐ Addresses text rendering, a major bottleneck of T2I generation. Practical and open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] A High-Quality Robust Diffusion Framework for Corrupted Dataset](a_highquality_robust_diffusion_framework_for_corrupted_datas.md)
- [\[ECCV 2024\] Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models](enhancing_perceptual_quality_in_video_super-resolution_through_temporally-consis.md)
- [\[ECCV 2024\] Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)
- [\[ICLR 2026\] PosterCraft: Rethinking High-Quality Aesthetic Poster Generation in a Unified Framework](../../ICLR2026/image_generation/postercraft_rethinking_high-quality_aesthetic_poster_generation_in_a_unified_fra.md)
- [\[ECCV 2024\] EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation](emdm_efficient_motion_diffusion_model_for_fast_and_high-quality_motion_generatio.md)

</div>

<!-- RELATED:END -->
