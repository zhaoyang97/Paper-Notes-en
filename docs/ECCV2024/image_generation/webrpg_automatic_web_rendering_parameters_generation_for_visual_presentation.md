---
title: >-
  [Paper Note] WebRPG: Automatic Web Rendering Parameters Generation for Visual Presentation
description: >-
  [ECCV 2024][Image Generation] This work proposes a new task called Web Rendering Parameters Generation (WebRPG), which aims to automatically generate visual presentation parameters (layout, text style, and color) of web elements based on HTML code. By using a VAE to compress the rendering parameters and custom HTML embeddings to capture semantic and hierarchical information, two baseline models (autoregressive and diffusion) are established…
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 83192421425697c4
---

# WebRPG: Automatic Web Rendering Parameters Generation for Visual Presentation

**Conference**: ECCV 2024  
**arXiv**: [2407.15502](https://arxiv.org/abs/2407.15502)  
**Area**: Image Generation

## TL;DR

This work proposes a new task called Web Rendering Parameters Generation (WebRPG), which aims to automatically generate visual presentation parameters (layout, text style, and color) of web elements based on HTML code. By using a VAE to compress the rendering parameters and custom HTML embeddings to capture semantic and hierarchical information, two baseline models (autoregressive and diffusion) are established, where the autoregressive model significantly outperforms the diffusion model and GPT-4.

## Background & Motivation

- Generative models have revolutionized image, text, and audio creation, but the critical area of **web design automation** remains under-explored.
- Web design is complex and time-consuming; developers lacking design experience often produce web pages with poor visual quality.
- Existing works focus on specific sub-tasks of CSS (layout generation, font recommendation, color scheme), lacking a **comprehensive web visual design** solution from scratch.
- CSS coding practices are complex (numerous selector options), making the direct automatic generation of CSS code challenging.
- **Mechanism**: Standardize CSS into rendering parameters (RPs), transforming the problem into generating rendering parameters for each element given the HTML code.

**Gap with existing work**:
- Graphic design methods limit the number of elements to $\le 25$, using 5 tokens to describe one element; in contrast, web pages have hundreds of elements, each requiring 13 rendering parameters.
- 1D sequence representation ignores the hierarchical structure information of web pages.

## Method

### Overall Architecture

Latent space generation method:
1. **VAE** compresses all rendering parameters of each element into a latent vector.
2. **HTML Embedding** encodes semantic, hierarchical, and character count information.
3. **Generative Model** (autoregressive or diffusion) generates the latent vector based on the HTML embedding.
4. **VAE Decoder** decodes the latent vector back to rendering parameters.

### Key Designs

**Rendering Parameter Definition**: 13 common CSS attributes, divided into three categories:
- **Layout attributes**: left, top, width, height
- **Text attributes**: font-style, font-weight, font-size, line-height, text-align, text-decoration, text-transform
- **Color attributes**: color, background-color

**VAE Rendering Parameter Compression**:
- Compresses the $\mathcal{W}$ rendering parameters of each element into a $d=128$ dimensional latent vector.
- The encoder and decoder are both 5-layer MLPs.
- Pre-trained on synthetic data to ensure that the latent space covers as many element appearance combinations as possible.
- Makes the input length dependent only on the number of elements $S$, rather than $S \times \mathcal{W}$.

**HTML Embedding**:
- **Semantic embedding**: Extracts HTML token features using a frozen MarkupLM_large, followed by average pooling.
- **Hierarchical embedding**: Encodes the element's position in the DOM tree using an XPath embedding layer.
- **Character count embedding**: Maps the number of characters in the element's text to a dense vector (since element sizes are positively correlated with character count).

**Two Generative Models**:
- **WebRPG-AR** (autoregressive): Introduces masked ground-truth latent vectors $\mathcal{Z}_{mask}$ to stabilize training, with all elements masked during inference.
- **WebRPG-DM** (diffusion): Performs a standard diffusion process on the VAE latent space.

### Loss & Training

Autoregressive model: $L = \log p_\psi(\mathcal{P}|\mathcal{H}, \mathcal{Z}_{mask}) + L_{VAE}$

Diffusion model: $L = \mathbb{E}_{\mathcal{Z},\epsilon,t}[\|\epsilon - \epsilon_\psi(\mathcal{Z}_t, t, \mathcal{H})\|_2^2] + L_{VAE}$

VAE loss: reconstruction term + KL divergence regularization term

## Key Experimental Results

### Main Results

Quantitative comparison of WebRPG baselines and LLMs:

| Model | FID ↓ | FID_layout ↓ | Ele. IoU ↑ | FID_style ↓ | SC Score ↑ |
|------|-------|-------------|------------|-------------|------------|
| **WebRPG-AR** | **0.1281** | **0.1520** | **0.7082** | **0.2124** | **0.9474** |
| WebRPG-DM | 62.021 | 60.942 | 0.0357 | 106.95 | 0.3671 |
| GPT-4 | 4.2141 | 47.732 | 0.0347 | 8.8898 | 0.5515 |
| StarCoder2-7b | 11.899 | 51.432 | 0.0309 | 18.186 | 0.3639 |
| DeepSeek-Coder-6.7b | 5.8219 | 55.744 | 0.0330 | 7.4542 | 0.3949 |
| CodeLlama-13b | 9.2826 | 55.427 | 0.0278 | 11.625 | 0.3864 |
| Real Web Page | 0.0027 | 0.0015 | 1.0000 | 0.0074 | 1.0000 |
| Plain HTML | 8.5342 | 52.438 | 0.0354 | 8.4951 | 0.3668 |

The FID of WebRPG-AR is only 0.1281, far surpassing the diffusion model (62.021) and GPT-4 (4.2141). Notably, in terms of layout, its Ele. IoU reaches 0.7082, whereas other methods are all below 0.04.

### Ablation Study

Component ablation study based on WebRPG-AR:

| # | VAE | $\mathcal{Z}_{mask}$ | Semantic | Hierarchy | Char Count | FID ↓ | Ele. IoU ↑ | SC Score ↑ |
|---|-----|-----|------|------|--------|-------|------------|------------|
| 1 | ✗ | ✓ | ✓ | ✓ | ✓ | 0.9702 | 0.5954 | 0.8053 |
| 2 | ✓ | ✗ | ✓ | ✓ | ✓ | 0.1487 | 0.6462 | 0.9332 |
| 3 | ✓ | ✓ | ✗ | ✓ | ✓ | 0.1797 | 0.6620 | 0.9323 |
| 4 | ✓ | ✓ | ✓ | 1D Position | ✓ | 0.3003 | 0.6345 | 0.8982 |
| 5 | ✓ | ✓ | ✓ | ✓ | ✗ | 0.1575 | 0.6769 | 0.9434 |
| **6** | **✓** | **✓** | **✓** | **✓** | **✓** | **0.1281** | **0.7082** | **0.9474** |

### Key Findings

1. **VAE is crucial** (#1 vs #6): Without VAE compression, the FID deteriorates from 0.13 to 0.97, as the 1D unfolded sequence becomes excessively long and contains redundant information.
2. **Diffusion models are unsuitable for this task**: Web elements form a hierarchical structure in non-Euclidean space, and the task requires precise control, which conflicts with the Euclidean space assumption and the blurry generation characteristics of diffusion models.
3. **Hierarchical embedding is vital for layout** (#4): Replacing XPath hierarchical embedding with 1D position embedding leads to a chaotic and disorganized layout.
4. **Semantic embedding helps understand element relationships** (#3): Without semantic information, the model struggles to recognize semantic relationships such as key-value pairs.
5. **Character count embedding affects size prediction** (#5): Without it, element widths fail to match the content, resulting in text truncation.
6. GPT-4 shows basic design capabilities on simple HTML, but its layout performance is limited when facing complex HTML structures.

## Highlights & Insights

- **Pioneering Task Definition**: WebRPG transforms the complex CSS generation problem into structured rendering parameters generation, reducing the learning difficulty.
- **Elegant Design of VAE Compression**: Compressing the multi-dimensional rendering parameters of each element into a single latent vector makes the sequence length depend only on the number of elements.
- **Three-Dimensional Information Integration in HTML Embedding**: Semantics (MarkupLM), hierarchy (XPath), and character count are all indispensable.
- **Prospect on Fully Automated Workflow**: Combining LLM-generated HTML with WebRPG-generated visual presentations enables end-to-end web development.
- **Comparative Perspective with LLMs**: Reveals the capability boundary of GPT-4 on this task—reasonable style processing but insufficient layout ability.

## Limitations & Future Work

- The dataset is limited to Klarna e-commerce web pages, resulting in restricted domain diversity.
- The model does not process image content (retaining only `<img>` tags) and cannot consider the coordination between images and design.
- Performance decreases as the number of elements and the depth of the DOM tree increase (both layout and style metrics decline).
- The autoregressive model is more prone to errors for elements at the end of the sequence (elements located at the bottom of the HTML code).
- Dynamic components and interactive designs are not supported.
- Only 13 CSS attributes are considered, while the scale of CSS attributes in real-world web pages far exceeds this scope.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)
- [\[ECCV 2024\] Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality](learning_trimodal_relation_for_audio-visual_question_answering_with_missing_moda.md)
- [\[ECCV 2024\] Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion](mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)

</div>

<!-- RELATED:END -->
