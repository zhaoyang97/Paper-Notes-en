---
title: >-
  [Paper Note] Implicit Style-Content Separation using B-LoRA
description: >-
  [ECCV 2024][Image Generation][Style Transfer] This paper proposes B-LoRA. By analyzing the SDXL architecture, it is discovered that implicitly separating style and content of a single image can be achieved by jointly training the LoRA weights of only two specific transformer blocks (Block 4 controls content, and Block 5 controls style), supporting various tasks such as style transfer, text-based stylization, and consistent style generation.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Style Transfer"
  - "LoRA"
  - "Style-Content Separation"
  - "SDXL"
  - "Image Stylization"
date: 2026-05-08
content_hash: 180a35cc7975a7c3
---

# Implicit Style-Content Separation using B-LoRA

**Conference**: ECCV 2024  
**arXiv**: [2403.14572](https://arxiv.org/abs/2403.14572)  
**Code**: [Project Page](https://B-LoRA.github.io/B-LoRA/)  
**Area**: Model Compression  
**Keywords**: Style Transfer, LoRA, Style-Content Separation, SDXL, Image Stylization

## TL;DR

This paper proposes B-LoRA. By analyzing the SDXL architecture, it is discovered that implicitly separating style and content of a single image can be achieved by jointly training the LoRA weights of only two specific transformer blocks (Block 4 controls content, and Block 5 controls style), supporting various tasks such as style transfer, text-based stylization, and consistent style generation.

## Background & Motivation

Image stylization is a classic computer vision task that requires changing the style of an image while preserving its content. The core challenge lies in the **strong coupling of style and content**, leading to an inherent trade-off between style transformation and content preservation.

Limitations of prior work:

**Fine-tuning methods** (DreamBooth, LoRA, etc.): Fine-tuning the model on specific images is prone to overfitting, and style and content remain entangled.

**Combination of two independent LoRAs**: This requires training a style LoRA and a content LoRA separately, and the combination method is ambiguous (weight interpolation requires manual search for coefficients).

**ZipLoRA**: Proposes learning mixture coefficients to merge two LoRAs, but each new style-content combination requires additional optimization.

**Zero-shot methods** (StyleAligned, IP-Adapter, etc.): Unable to explicitly separate style and content, which may cause semantic leakage from the style image.

The core finding of B-LoRA: **Style-content decoupling naturally exists within the SDXL architecture**—specific transformer blocks independently control the content and style of the generated image. Exploiting this property, implicit separation can be achieved by training the LoRA weights of only these two blocks.

## Method

### Overall Architecture

**Core pipeline:**
1. Analyze the 11 transformer blocks of SDXL to determine which blocks control content and which control style.
2. Only train the LoRA weights of **Block 4** ($\Delta W^4$, content) and **Block 5** ($\Delta W^5$, style).
3. After training, the two B-LoRAs can be used as independent plug-and-play components.

### Key Designs

#### 1. SDXL Architecture Analysis

The UNet of SDXL contains 70 attention layers, divided into 11 transformer blocks. Specifically, the 6 intermediate blocks ($W_0^1$ to $W_0^6$) each contain 10 attention layers.

**Analysis Method**: Inject a different text prompt $\hat{p}$ into a specific block while keeping the prompt $p$ for the remaining blocks. The influence of this block on content/style is then evaluated using CLIP similarity:

$$\mathcal{C}(I_{\hat{p}\to i, p\to j}, \hat{p}) = sim(C_I(I_{\hat{p}\to i, p\to j}), C_T(\hat{p}))$$

Experiments conducted on 400 pairs of content and style prompts show that:
- **Block 2 and Block 4** dominate content (injecting prompts of different objects can change the generated objects).
- **Block 5** dominates color/style.

**Why choose Block 4 instead of Block 2?**

Comparative experiments on LoRA training reveal that $\{\Delta W^4, \Delta W^5\}$ is superior to $\{\Delta W^2, \Delta W^5\}$:
- $\Delta W^4$ better captures the fine details of the input object (deeper layers preserve finer characteristics, consistent with findings in Plug-and-Play).
- $\{\Delta W^4, \Delta W^5\}$ yields higher reconstruction quality.

#### 2. B-LoRA Training Scheme

Given a single input image $I$:
- **Freeze** all model weights of SDXL and the text encoder.
- **Only optimize** the LoRA weights of Block 4 and Block 5, $\Delta W^4, \Delta W^5$.
- The objective is to reconstruct the input image using a generic prompt "A [v]" (deliberately not specifying content or style descriptions).
- Training result: $\Delta W^4$ captures the **content**, and $\Delta W^5$ captures the **style**.

Key advantages:
- Training only two blocks is sufficient—**storage is reduced by 70%**.
- Joint training allows both blocks to perform their specialized tasks—**individual training cannot achieve separation**.
- Reaches 1000 steps without overfitting (standard LoRA is typically limited to 400 steps to prevent overfitting).

#### 3. Three Stylization Applications

**(a) Reference image-based style transfer:**
- Train B-LoRAs separately for the content image $I_c$ and the style image $I_s$.
- Combine $\Delta W_c^4$ (content) and $\Delta W_s^5$ (style) and insert them directly into SDXL.
- During inference, use a prompt in the format of "A [c] in [s] style".
- **No additional optimization or fine-tuning is required**.

**(b) Text-based stylization:**
- Use only $\Delta W_c^4$ (content) and discard $\Delta W_c^5$.
- Specify style through text prompts, such as "oil painting", "sketch", etc.
- Since style and content are separated, dramatic style transformations can be achieved.

**(c) Consistent style generation:**
- Use only $\Delta W_s^5$ (style) and discard $\Delta W_s^4$.
- The model is adapted to a specific style, enabling the generation of any content in that style using text.

### Loss & Training

Standard DreamBooth LoRA training configurations are adopted:

- **Base Model**: SDXL v1.0
- **Optimizer**: Adam, learning rate $5\times10^{-5}$
- **LoRA rank**: $r=64$
- **Training prompt**: "A [v]" (generic placeholder)
- **Training steps**: 1000 steps (approx. 10 mins / single image / A100)
- **Data augmentation**: Center crop only
- **Loss Function**: Standard diffusion denoising loss

$$\mathcal{L} = \mathbb{E}_{z_t, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2\right]$$

Key: The text encoder is fully frozen during training, avoiding style-content information coupling via the text pathway.

## Key Experimental Results

### Main Results

Quantitative comparison with other methods (DINO ViT-B/8 feature cosine similarity):

| Method | Style Score (Multi-image) ↑ | Style Score (Single-image) ↑ | Content Score (Multi-image) ↑ | Content Score (Single-image) ↑ |
|------|:---:|:---:|:---:|:---:|
| StyleDrop | 0.826±0.07 | 0.790±0.06 | 0.817±0.06 | 0.874±0.08 |
| StyleAligned | 0.855±0.05 | 0.829±0.05 | 0.779±0.05 | 0.792±0.06 |
| ZipLoRA | 0.796±0.07 | 0.782±0.05 | **0.841±0.05** | **0.933±0.05** |
| DB-LoRA | 0.863±0.06 | **0.881±0.05** | 0.769±0.05 | 0.790±0.05 |
| **B-LoRA** | **0.863±0.06** | **0.881±0.05** | 0.769±0.05 | 0.790±0.05 |

User study results (34 participants, 1020 votes):

| Baseline Method | Preference Rate for B-LoRA |
|----------|:---:|
| vs. StyleAligned | **94%** |
| vs. ZipLoRA | **91%** |
| vs. StyleDrop | **88%** |

### Ablation Study

Ablation of block selection (training different block combinations):

| Trained Blocks | Reconstruction Quality | Content Capture | Style Separation |
|------|------|------|------|
| $\{W^2, W^5\}$ | Suboptimal | Details lost | Decent |
| $\{W^4, W^5\}$ | **Good** | **Details preserved** | **Good** |

Impact of prompt selection: using the generic prompt "A [v]" yields the best performance. Explicitly specifying the content or style in prompts hinders natural style-content separation.

### Key Findings

1. **Overwhelming user preference**: 88-94% of users prefer B-LoRA, significantly outperforming all baseline methods.
2. **Content overfitting of ZipLoRA**: ZipLoRA achieves the highest content score but a low style score, indicating that it overfits the content and fails to transfer the style effectively.
3. **Single-image vs. Multi-image**: For all methods, content scores increase while style scores decrease under single-image conditions, suggesting overfitting. B-LoRA suffers from this the least due to its natural separation mechanism.
4. **No additional optimization required**: Unlike ZipLoRA, which requires extra training for every new combination, the components of B-LoRA are directly plug-and-play.
5. **Stylized images as content inputs**: B-LoRA can extract content structures from stylized images, which poses a significant challenge for other methods.

## Highlights & Insights

1. **Discovery of internal structural properties of SDXL**: Different transformer blocks naturally possess functional specialization for style and content, an observation of standalone academic value.
2. **Minimalist yet effective design**: No extra networks, training datasets, or complex injection mechanisms are needed. A simple combination of LoRA and block selection suffices.
3. **Modularity and composability**: Trained once, the style and content B-LoRAs can be recombined infinitely without retraining.
4. **Counter-intuitive overfitting behavior**: Training for more steps (1000 steps vs. 400 steps in standard LoRA) actually yields better results because training only two blocks acts as a natural regularizer.
5. **High practical value**: 10-minute training, 70% storage reduction, and seamless out-of-the-box compatibility with the SDXL ecosystem.

## Limitations & Future Work

1. **Color captured as style**: The color of objects is often encoded into the style component. In certain cases where color is critical to identity (e.g., a red fire truck), this can lead to poorer identity preservation.
2. **Background leakage**: The style component of a single reference image might incorporate background elements, rather than solely focusing on the style of the central object.
3. **Limitations in complex scenes**: For complex scenes containing numerous elements, the content B-LoRA may fail to preserve the complete structure of the scene.
4. **Applicable only to SDXL**: Relies heavily on the block distribution specific to the SDXL architecture, making it difficult to directly transfer to other diffusion models.
5. Future work can explore more fine-grained separation (sub-components such as structure, shape, color, and texture).

## Related Work & Insights

- **Plug-and-Play**: Discovered that deep features control structure while shallow features control appearance, which is complementary to the block analysis in this paper.
- **ZipLoRA**: Proposes learning mixture coefficients to merge two LoRAs but requires additional optimization; B-LoRA completely eliminates this step.
- **StyleAligned**: Achieves consistent style generation through shared attention, but lacks explicit content control.
- **Custom Diffusion**: Also investigated which layers are sensitive to specific attributes, maintaining a similar methodology.
- **Insights**: Functional partitioning within large models is a direction worthy of in-depth study, extending beyond style-content separation.

## Rating

- **Novelty**: ★★★★★ — Deep analysis of SDXL architecture reveals unexpected style-content separation properties.
- **Experimental Thoroughness**: ★★★★☆ — Strongly supported by the user study, though quantitative evaluations relying on DINO similarity are somewhat limited.
- **Writing Quality**: ★★★★★ — Well-argued, logically clear, with rich and intuitive visualizations.
- **Value**: ★★★★★ — Simple, highly efficient, modular, and highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs](../../CVPR2025/image_generation/k-lora_unlocking_training-free_fusion_of_any_subject_and_style_loras.md)
- [\[CVPR 2026\] CRAFT-LoRA: Content-Style Personalization via Rank-Constrained Adaptation and Training-Free Fusion](../../CVPR2026/image_generation/craft-lora_content-style_personalization_via_rank-constrained_adaptation_and_tra.md)
- [\[ECCV 2024\] ZigMa: A DiT-style Zigzag Mamba Diffusion Model](zigma_a_dit-style_zigzag_mamba_diffusion_model.md)
- [\[ECCV 2024\] ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)
- [\[ECCV 2024\] StyleTokenizer: Defining Image Style by a Single Instance for Controlling Diffusion Models](styletokenizer_defining_image_style_by_a_single_instance_for_controlling_diffusi.md)

</div>

<!-- RELATED:END -->
