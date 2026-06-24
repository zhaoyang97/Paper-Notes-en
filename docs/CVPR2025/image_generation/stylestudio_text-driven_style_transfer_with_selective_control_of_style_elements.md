---
title: >-
  [Paper Note] StyleStudio: Text-Driven Style Transfer with Selective Control of Style Elements
description: >-
  [CVPR 2025][Image Generation][Text-driven style transfer] StyleStudio proposes three complementary strategies—cross-modal AdaIN, style-based classifier-free guidance (SCFG), and a teacher model—to address style overfitting, text misalignment, and layout instability in text-driven style transfer, achieving selective control of style elements.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Text-driven style transfer"
  - "AdaIN"
  - "Classifier guidance"
  - "Layout stabilization"
  - "Style overfitting"
date: 2026-05-08
content_hash: 9d8da05da5fc67b2
---

# StyleStudio: Text-Driven Style Transfer with Selective Control of Style Elements

**Conference**: CVPR 2025  
**arXiv**: [2412.08503](https://arxiv.org/abs/2412.08503)  
**Code**: [Project Page](https://stylestudio-official.github.io/)  
**Area**: Image Generation / Style Transfer  
**Keywords**: Text-driven style transfer, AdaIN, Classifier guidance, Layout stabilization, Style overfitting

## TL;DR

StyleStudio proposes three complementary strategies—cross-modal AdaIN, style-based classifier-free guidance (SCFG), and a teacher model—to address style overfitting, text misalignment, and layout instability in text-driven style transfer, achieving selective control of style elements.

## Background & Motivation

Text-driven style transfer aims to merge the style of a reference image with the content described by a text prompt. Existing methods face three core problems:

1. **Style overfitting**: Models excessively replicate all elements (color, texture, lighting, etc.) of the reference style image, causing the generated output to mirror the reference features too closely and reducing aesthetic flexibility. For instance, when the text specifies a "red apple" but the reference style is dominantly blue, the model tends to follow the blue color of the style image rather than the red described in the text.

2. **Difficulty in text-alignment**: The dominant colors or patterns of the style image override the guidance of the text prompt. As shown in Figure 2, even when the color is explicitly specified, the model still prioritizes adopting the color scheme of the style image.

3. **Layout instability**: The complexity introduced by style transfer leads to artifacts such as checkerboard effects (e.g., the CSGO method shown in Figure 3), and a lack of aggregation in core generation areas within the cross-attention.

Existing methods (such as weighted sum fusion in IP-Adapter, selective injection in InstantStyle, etc.) either provide insufficient style accuracy or fail to resolve text-style conflicts. StyleStudio proposes three plug-and-play complementary strategies to systematically address these issues.

## Method

### Overall Architecture

StyleStudio is based on the adapter architecture of CSGO, introducing three complementary modules on top of it: (1) cross-modal AdaIN to replace weighted summation for style-text feature fusion; (2) style-based classifier-free guidance (SCFG) to achieve selective control of style elements; and (3) a teacher model to stabilize spatial layout during early denoising stages. These three modules can be integrated into existing style transfer frameworks without fine-tuning.

### Key Designs

#### Key Design 1: Cross-Modal AdaIN

- **Function**: Integrates style and text features in a way that preserves text semantic structure, eliminating conflicts between them
- **Mechanism**: Within the cross-attention of each UNet layer, style and text conditional queries are used to obtain two sets of grid feature maps, $f_{\text{style}}$ and $f_{\text{text}}$, respectively. AdaIN is then used to normalize the text features using the statistics of the style features: $\hat{f}_{\text{af}} = \gamma_{\text{style}} \cdot \frac{f_{\text{text}} - \mu_{\text{text}}}{\sigma_{\text{text}}} + \beta_{\text{style}}$. The normalized features are added to the UNet features in a residual manner.
- **Design Motivation**: In the traditional weighted sum $f_{\text{ip}} = A(Q,K_t,V_t) + \lambda A(Q,K_i,V_i)$, text and style play similar roles, leading to suboptimal results when information conflicts. AdaIN allows the text to maintain the semantic structure (content) and the style to provide only statistical attributes (mean/variance), naturally separating their roles.

#### Key Design 2: Style-Based Classifier-Free Guidance (SCFG)

- **Function**: Selectively transfers only the target style and filters out irrelevant styles when the reference style image contains multiple style elements
- **Mechanism**: ControlNet is used to generate a "negative style image" that preserves the structure of the reference image but lacks the target style, serving as a negative sample. CFG is extended to SCFG: $\hat{\epsilon}_\theta = (1+w) \cdot \epsilon_\theta(z_t, y_{\text{cond}}^{text}, y_{\text{cond}}^{style}) - w \cdot \epsilon_\theta(z_t, y_{\text{neg}}^{text}, y_{\text{neg}}^{style})$, guiding the model to focus on the target style through the difference between positive and negative style images.
- **Design Motivation**: Style images are often mixed with multiple style elements (e.g., cartoon style + night scene). Text negative prompts cannot effectively eliminate style elements at the image level (e.g., "snow scene", "golden leaves"), necessitating image-level negative samples.

#### Key Design 3: Teacher Model for Layout Stabilization

- **Function**: Provides a stable spatial layout during early denoising stages to eliminate artifacts such as checkerboard effects
- **Mechanism**: The original text-to-image model (without style modules) is used as a teacher model. Denoising is performed synchronously using the same text prompt, sharing the self-attention maps of the teacher model with the style model at each timestep. Attention maps are replaced only during initial denoising steps (not all steps) to prevent styles from being lost due to excessive intervention.
- **Design Motivation**: Self-attention captures high-level spatial relations and stabilizes the base layout. Analysis shows that checkerboard artifacts are related to dispersed attention of keywords like "apple" in cross-attention. Excessively long intervention timesteps by the teacher model cause loss of style details.

### Loss & Training

Uses the standard denoising loss of diffusion models $\mathcal{L}(\theta) = \mathbb{E}_{t,z,c,\epsilon\sim\mathcal{N}(0,1)}[\|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2]$ (inherited from CSGO, without extra training).

## Key Experimental Results

### Main Results

| Method | Text Alignment↑ | User-study Text% | User-study Style% |
|------|-----------------|-------------------|-------------------|
| IP-Adapter | 0.221 | 7.48% | 6.63% |
| InstantStyle | 0.229 | 6.46% | 8.67% |
| CSGO | 0.216 | 7.99% | 6.97% |
| StyleCrafter | 0.189 | 3.06% | 8.67% |
| DEADiff | 0.229 | 1.87% | 5.27% |
| **StyleStudio** | **0.235** | **62.92%** | **50.85%** |

### Ablation Study

| Configuration | Text Alignment↑ | Relative Gain |
|------|-----------------|---------|
| CSGO Baseline | 0.216 | - |
| +Cross-Modal AdaIN | 0.223 | +3.2% |
| +Teacher Model | 0.228 | +5.5% |
| **+AdaIN+Teacher** | **0.235** | **+8.7%** |

### Key Findings

- In user studies, StyleStudio leads significantly in both text alignment (62.92%) and style similarity (50.85%).
- Cross-Modal AdaIN and the Teacher Model exhibit complementary effects, achieving the highest gains when combined.
- SCFG effectively eliminates style elements (such as snow scenes and golden leaves) that text negative prompts cannot address.
- Excessively long intervention timesteps for the Teacher Model (up to 50 steps) lead to style loss, requiring a balanced threshold.

## Highlights & Insights

1. **Complementarity of the Three Strategies**: AdaIN addresses style-text conflicts at the feature level, SCFG handles style selection at the image level, and the teacher model stabilizes layout at the generation level, collectively covering different tiers of problems.
2. **Training-Free Plug-and-Play Design**: All three strategies can be directly integrated into existing adapter-based style transfer frameworks without training.
3. **In-Depth Analysis of Artifact Causes**: Visualization of cross-attention reveals the relationship between checkerboard effects and dispersed attention.

## Limitations & Future Work

- The teacher model introduces extra inference time (inference time increases from 6 seconds to 17 seconds).
- Generating negative style images requires certain empirical experience and manual operation.
- Future work can explore automated generation of negative style images and more efficient teacher model schemes.

## Related Work & Insights

- **CSGO**: The base architecture of StyleStudio, which is trained with adapters and style datasets but suffers from style overfitting.
- **InstantStyle**: Mitigates content leakage through selective injection, but lacks sufficient style accuracy.
- **StyleID (CVPR'24)**: A training-free method with good content preservation but limited style-transfer performance.
- Insight: AdaIN, a classic technique, remains highly effective in feature fusion for modern diffusion models and outperforms simple weighted summation.

## Rating

⭐⭐⭐⭐ — The three complementary strategies clearly target three different levels of problems, with ablation studies thoroughly validating the contribution of each component. The overwhelming advantage in the user study (62.92% vs. <10%) is highly convincing. The training-free plug-and-play design enhances practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HSI: A Holistic Style Injector for Arbitrary Style Transfer](hsi_a_holistic_style_injector_for_arbitrary_style_transfer.md)
- [\[CVPR 2025\] SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer](samam_style-aware_state_space_model_for_arbitrary_image_style_transfer.md)
- [\[CVPR 2025\] OmniStyle: Filtering High Quality Style Transfer Data at Scale](omnistyle_filtering_high_quality_style_transfer_data_at_scale.md)
- [\[CVPR 2025\] SCSA: A Plug-and-Play Semantic Continuous-Sparse Attention for Arbitrary Semantic Style Transfer](scsa_a_plug-and-play_semantic_continuous-sparse_attention_for_arbitrary_semantic.md)
- [\[ICCV 2025\] Domain Generalizable Portrait Style Transfer](../../ICCV2025/image_generation/domain_generalizable_portrait_style_transfer.md)

</div>

<!-- RELATED:END -->
