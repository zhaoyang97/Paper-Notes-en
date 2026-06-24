---
title: >-
  [Paper Note] HSI: A Holistic Style Injector for Arbitrary Style Transfer
description: >-
  [CVPR 2025][Image Generation][Style Transfer] HSI proposes a style transfer module based on global style statistical features and element-wise multiplication, replacing the quadratic complexity of self-attention with linear complexity. Simultaneously, it enhances stylization quality through a dual-relation learning mechanism, outperforming existing methods in both effectiveness and efficiency.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Style Transfer"
  - "Attention Mechanism"
  - "Linear Complexity"
  - "Global Style"
  - "Element-wise Multiplication"
date: 2026-05-08
content_hash: 06d7ff86c2040553
---

# HSI: A Holistic Style Injector for Arbitrary Style Transfer

**Conference**: CVPR 2025  
**arXiv**: [2502.04369](https://arxiv.org/abs/2502.04369)  
**Code**: None  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Style Transfer, Attention Mechanism, Linear Complexity, Global Style, Element-wise Multiplication

## TL;DR
HSI proposes a style transfer module based on global style statistical features and element-wise multiplication, replacing the quadratic complexity of self-attention with linear complexity. Simultaneously, it enhances stylization quality through a dual-relation learning mechanism, outperforming existing methods in both effectiveness and efficiency.

## Background & Motivation

**Background**: The field of arbitrary style transfer (AST) is divided into global transformation methods (such as AdaIN and WCT) and local matching methods (such as SANet and AdaAttN). Recently, attention-based methods have achieved significant progress by establishing point-wise semantic correspondences between content and style features.

**Limitations of Prior Work**: The attention mechanism suffers from three critical issues in AST: (1) Over-focusing on local patterns—point-wise matching causes prominent local regions in style images (e.g., eyes in a face) to be repeatedly copied into the stylized results, producing artifacts; (2) Bias patterns from softmax exponential operations—over-attending to a salient style region while ignoring the overall distribution; (3) Quadratic complexity of $O((H\times W)^2 \times C)$ caused by matrix multiplication—leading to out-of-memory (OOM) issues on GPUs when processing 2K resolution images.

**Key Challenge**: Although the attention mechanism can establish semantic correspondences, it naturally favors local matching, which contradicts the "global style coherence" requirement of style transfer.

**Goal**: To design a style transfer module that can capture global style features while maintaining linear computational complexity.

**Key Insight**: The authors observe that individual point features struggle to represent a complete style pattern. Therefore, one should directly extract global style statistical features (mean, variance, skewness, and kurtosis) and substitute matrix multiplication with element-wise multiplication to establish content-style relationships.

**Core Idea**: Employing global style statistical features and element-wise multiplication instead of point-wise attention and matrix multiplication to achieve high-quality style transfer with linear complexity.

## Method

### Overall Architecture
A simple encoder-decoder architecture is adopted: the encoder is a pretrained VGG-19 with fixed parameters (up to the relu_4_1 layer), which extracts content features $F_c$ and style features $F_s$. A series of cascaded HSI modules then accomplishes style transfer in the feature space, generating stylized features $F_{cs}$, which are subsequently mapped back to the image space via a mirrored decoder. In the decoder, upsampling is used instead of pooling layers to avoid checkerboard artifacts.

### Key Designs

1. **Global Styles Extraction**:

    - **Function**: To extract comprehensive global statistical representations from style features, replacing point-wise feature matching.
    - **Mechanism**: Compute four channel-wide statistics of the style key features $K$: mean $\mu$, variance $\sigma^2$, skewness $\gamma_1$, and kurtosis $\gamma_2$. The mean and variance describe the overall tone and variability of the style, while the skewness and kurtosis describe the symmetry and concentration of the style feature distribution. Then, a dynamic network (depthwise separable convolution + global average/max pooling) maps the mean and maximum values of $K$ to weights $W$ and biases $b$ to dynamically weight and combine these four statistical features.
    - **Design Motivation**: Point-wise features cannot represent complete style patterns. The four complementary statistical features can describe style distribution characteristics more comprehensively, preventing disharmonious artifacts caused by over-focusing on prominent local regions.

2. **Dynamic Dual Relations Construction**:

    - **Function**: To establish two complementary content-style relationships and dynamically adjust their weights based on semantic similarity.
    - **Mechanism**: Simultaneously construct "local content - global style" relationships ($Q \odot K_s$) and "global content - global style" relationships ($Q_c \odot K_s$), which are dynamically fused using a similarity coefficient $\lambda_g$: $F_{qk} = \lambda_g \times (Q_c \odot K_s) \oplus (1-\lambda_g) \times (Q \odot K_s)$. Here, $\lambda_g$ is computed based on the cosine similarity between $Q$ and $K$. When the content and style share similar semantics (e.g., both are faces), the weight of the global relationship increases, enhancing overall harmony.
    - **Design Motivation**: Local relationships preserve content details, while global relationships improve style consistency. When they share similar semantics, the model should rely more on global relationships to avoid disharmonies of local style patterns.

3. **Linear-Complexity Transfer Process**:

    - **Function**: To reduce the computational complexity from $O((H\times W)^2 \times C)$ to $O(H \times W \times C)$.
    - **Mechanism**: All feature interactions within the HSI module utilize element-wise multiplication $\odot$ instead of matrix multiplication $\otimes$. Element-wise multiplication only involves multiplying elements at corresponding positions, resulting in a complexity that scales linearly with the size of the feature maps. The entire HSI architecture is similar to self-attention (Q-K interaction $\rightarrow$ normalization $\rightarrow$ weighted V $\rightarrow$ residual connection) but replaces all matrix multiplications with element-wise multiplications.
    - **Design Motivation**: This is the first time element-wise multiplication has been applied to establish semantic relationships in style transfer. The linear complexity enables the method to process arbitrary-resolution images (successfully handling 2K resolution), whereas all attention-based methods experience OOM at high resolutions.

### Loss & Training
The total loss is defined as: $\mathcal{L} = 60\mathcal{L}_s + 5\mathcal{L}_c + 50\mathcal{L}_{adv}$. Style loss $\mathcal{L}_s$ aligns the mean and standard deviation between the stylized and style images in the VGG feature space; content loss $\mathcal{L}_c$ computes the VGG feature distance for both color and grayscale images; adversarial loss $\mathcal{L}_{adv}$ utilizes a multi-scale discriminator to enhance the overall style transfer. Training is performed using the Adam optimizer with a batch size of 4 and a learning rate of 0.0001, requiring approximately 4 hours on a 4090Ti GPU.

## Key Experimental Results

### Main Results

| Method | Content Loss↓ | Style Loss↓ | LPIPS↓ | FID↓ |
|------|-------------|-------------|--------|------|
| AdaIN | 0.97 | 1.44 | 0.65| 19.68 |
| SANet | 1.18 | 1.26 | 0.63| 18.74 |
| AdaAttN | 1.21 | 1.52 | 0.57| 19.34 |
| StyTr2 | 0.69 | 1.34 | 0.56| 18.97 |
| StyA2K | **0.59** | 1.21 | 0.49| 19.85 |
| **HSI (Ours)** | 0.62 | **0.95** | **0.46**| **18.46** |

### Efficiency Comparison (GPU Memory / Inference Speed)

| Resolution | SANet | StyTr2 | StyA2K | HSI (Ours) |
|--------|-------|--------|--------|-----------|
| 1024×1024 | 2.48GB | OOM | 6.01GB | **2.11GB** |
| 2048×2048 | OOM | OOM | OOM | **8.12GB** |
| 1K Inference Speed | - | OOM | - | **~50fps** |

Only HSI and AdaIN can run at 2K resolution, with HSI yielding far superior stylization quality compared to AdaIN.

### Key Findings
- Each of the four statistical features contributes distinct colors and brushstroke styles, and their combination injects richer style elements.
- Constructing local relationships alone preserves content details but yields less consistent styles; constructing global relationships alone produces more cohesive styles but loses details. Dynamically combining both achieves the optimal balance.
- When the HSI module is used as a plug-and-play component to replace the attention module in SANet (SANet+HSI), repeated style patterns and distorted regions are significantly reduced, validating the generalizability of the module.

## Highlights & Insights
- **Insight on substituting matrix multiplication with element-wise multiplication**: It is found for the first time that element-wise multiplication is sufficient for establishing effective content-style semantic relationships in style transfer, inherently possessing linear complexity. This insight may inspire other complexity-sensitive tasks that require global feature interactions.
- **Diversity design of global statistical features**: Describing the style distribution comprehensively using four statistics (mean, variance, skewness, and kurtosis) is richer than AdaIN (which only uses mean and variance) and more stable than point-wise matching in attention mechanisms.
- **Plug-and-play module design**: The HSI architecture is highly similar to self-attention, allowing it to directly replace attention modules in existing methods to improve performance, lowering the barrier to adoption.

## Limitations & Future Work
- The content loss is slightly higher than StyA2K (0.62 vs 0.59), indicating a potential loss of minor content details in extreme cases.
- Global statistical features may be limited when there is a massive discrepancy between content and style (e.g., natural landscapes + abstract paintings).
- Temporal consistency in video style transfer has not been explored, although the advantage of linear complexity is even more valuable in video scenarios.
- Future directions: introducing multi-scale HSI to handle style features at different frequencies, and combining with diffusion models for finer style control.

## Related Work & Insights
- **vs SANet/AdaAttN**: Pure attention-based methods over-focus on local patterns, leading to repetitive artifacts (e.g., facial eyes appearing in multiple places). HSI completely avoids this issue via global style extraction.
- **vs StyTr2**: Based on the Transformer architecture, which suffers from OOM at 1K resolution; HSI's linear complexity handles 2K with better results.
- **vs StyA2K**: StyA2K uses all-to-key attention to alleviate some issues, but its complexity remains quadratic and it is prone to under-stylization; HSI achieves linear complexity with better style fidelity.
- **vs AdaIN**: Both are global statistical methods, but AdaIN only employs mean and variance. HSI adds skewness and kurtosis along with a dynamic dual-relation mechanism, delivering significantly higher stylization quality.

## Rating
- Novelty: ⭐⭐⭐⭐ The insight of replacing matrix multiplication with element-wise multiplication is novel, though the overall framework (encoder-decoder + VGG) is conventional.
- Experimental Thoroughness: ⭐⭐⭐⭐ The qualitative and quantitative comparisons are comprehensive. Ablation studies validate each component, supplemented by plug-play experiments and high-resolution evaluations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, in-depth comparative analysis with attention mechanisms, and clear illustrations.
- Value: ⭐⭐⭐⭐ Linear complexity style transfer holds high practical value, and the plug-and-play design lowers the barrier to adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer](samam_style-aware_state_space_model_for_arbitrary_image_style_transfer.md)
- [\[CVPR 2025\] SCSA: A Plug-and-Play Semantic Continuous-Sparse Attention for Arbitrary Semantic Style Transfer](scsa_a_plug-and-play_semantic_continuous-sparse_attention_for_arbitrary_semantic.md)
- [\[CVPR 2025\] OmniStyle: Filtering High Quality Style Transfer Data at Scale](omnistyle_filtering_high_quality_style_transfer_data_at_scale.md)
- [\[CVPR 2025\] StyleStudio: Text-Driven Style Transfer with Selective Control of Style Elements](stylestudio_text-driven_style_transfer_with_selective_control_of_style_elements.md)
- [\[CVPR 2026\] StyleGallery: Training-free and Semantic-aware Personalized Style Transfer from Arbitrary Image References](../../CVPR2026/image_generation/stylegallery_training-free_and_semantic-aware_personalized_style_transfer_from_a.md)

</div>

<!-- RELATED:END -->
