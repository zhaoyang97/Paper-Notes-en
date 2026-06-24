---
title: >-
  [Paper Note] A3GS: Arbitrary Artistic Style into Arbitrary 3D Gaussian Splatting
description: >-
  [ICCV 2025][3D Vision][3DGS style transfer] A3GS is proposed as the first feed-forward zero-shot 3DGS style transfer framework. It encodes 3DGS scenes into a latent space via a GCN-based autoencoder and injects arbitrary style features using AdaIN, completing style transfer from any style to any 3D scene in approximately 10 seconds — two orders of magnitude faster than optimization-based methods.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3DGS style transfer"
  - "zero-shot"
  - "feed-forward network"
  - "graph convolutional network"
  - "AdaIN"
date: 2026-05-08
content_hash: 43efe91dec935276
---

# A3GS: Arbitrary Artistic Style into Arbitrary 3D Gaussian Splatting

**Conference**: ICCV 2025
**Code**: N/A  
**Area**: 3D Vision / 3D Style Transfer
**Keywords**: 3DGS style transfer, zero-shot, feed-forward network, graph convolutional network, AdaIN

## TL;DR
A3GS is proposed as the first feed-forward zero-shot 3DGS style transfer framework. It encodes 3DGS scenes into a latent space via a GCN-based autoencoder and injects arbitrary style features using AdaIN, completing style transfer from any style to any 3D scene in approximately 10 seconds — two orders of magnitude faster than optimization-based methods.

## Background & Motivation

**Background**: 3D scene style transfer is an important requirement in domains such as the metaverse and gaming. With the emergence of 3D Gaussian Splatting (3DGS), methods such as StylizedGS and G-Style leverage gradient-based optimization to stylize 3DGS scenes, achieving real-time rendering with satisfactory stylization quality.

**Limitations of Prior Work**: Optimization-based methods suffer from two critical issues: (1) each scene–style combination requires minutes to hours of optimization, making rapid style switching infeasible; (2) GPU memory consumption grows linearly with the number of Gaussian primitives, making large-scale scenes intractable — for instance, StyleGaussian runs out of memory beyond 300K Gaussian primitives.

**Key Challenge**: Style transfer quality depends on a thorough understanding of 3D spatial structural features, yet efficient feed-forward approaches struggle to extract meaningful 3D local features from unstructured 3DGS data. The information carried by individual Gaussian primitives is insufficient to capture 3D structural style characteristics, which require collaborative color distributions across multiple local Gaussian primitives.

**Goal**: Design a feed-forward network that enables zero-shot 3DGS style transfer — requiring no additional training or optimization for arbitrary new scenes and styles.

**Key Insight**: The authors observe that in 2D image style transfer, CNNs can efficiently inject styles in a zero-shot manner within the feature space. By analogy, if a suitable feature extraction network can be designed for the unstructured nature of 3DGS data, a similar style transfer mechanism can be realized in 3D feature space.

**Core Idea**: A 3D Graph Convolutional Network (3D-GCN) autoencoder aggregates and encodes local Gaussian primitives from a 3DGS scene into a latent space. AdaIN is then applied in the latent space to inject VGG-extracted style features from a reference image, followed by decoding back to 3DGS colors — enabling fast, feed-forward style transfer.

## Method

### Overall Architecture
The input consists of a 3DGS scene ($N$ Gaussian primitives, each with position, rotation, scale, opacity, and color attributes) and an arbitrary style reference image. The geometric attributes and opacity of Gaussian primitives are kept fixed; only colors are modified. The pipeline comprises three stages: (1) a GCN encoder maps Gaussian primitive colors into the latent space; (2) an AdaIN-based stylizer aligns content and style features in the latent space; (3) a GCN decoder maps the stylized features back to per-primitive colors. The entire process takes approximately 10 seconds.

### Key Designs

1. **3D Graph Convolutional Layer (3D-GCN Layer)**:

    - Function: Extracts local 3D spatial features from unstructured 3DGS data.
    - Mechanism: For each Gaussian primitive, $M$ spatially nearest neighbors are defined as the receptive field. Convolutional kernels $K^S = \{k_C, k_s\}_{s=1}^S$ with learnable shapes and weights are introduced, where the center point $k_C = (0,0,0)$ and support points $k_s \in \mathbb{R}^3$ are learnable. This adaptive kernel shape allows the network to adjust its convolution operation according to varying 3D geometric structures, analogous to deformable convolutions in 2D CNNs.
    - Design Motivation: Standard GCNs lack 3D spatial awareness (confirmed by ablation studies), and MLPs have no local feature aggregation capability whatsoever. The learnable kernel shapes of 3D-GCN better adapt to irregular 3D Gaussian distributions, addressing the sensitivity of conventional point cloud networks to geometric transformations.

2. **GCN-based Autoencoder**:

    - Function: Compresses large-scale 3DGS scenes into a compact latent space while preserving locally aggregated 3D structural information.
    - Mechanism: The encoder alternates between 3D graph convolutional layers and pooling layers; pooling layers aggregate features within the receptive field via channel-wise max pooling and then downsample at rate $r$: $\text{Encoder}(P, C) = (\tilde{P}, F_c)$. The decoder employs the same type of convolutional layers combined with inverse-distance-weighted interpolation for upsampling, mapping latent features back to the colors of all Gaussian primitives: $\text{Decoder}(\tilde{P}, F_{cs}) = (P, C')$.
    - Design Motivation: Performing style transfer directly on individual Gaussian primitives is both computationally expensive and ineffective, since a single primitive lacks sufficient information to capture the regional color distribution patterns required for stylization. The autoencoder compresses locally clustered structural and color information into latent vectors, providing a suitable feature space for subsequent style injection.

3. **AdaIN-based Stylizer**:

    - Function: Injects statistical style features from a reference image into 3DGS content features within the latent space.
    - Mechanism: Since the 3D content feature space and the 2D image style feature space are misaligned, an MLP $\phi$ first maps the 3D content features into the image feature space, after which AdaIN aligns means and variances: $F_{cs} = \psi(\text{AdaIN}(\phi(F_c), F_s))$. Another MLP $\psi$ then maps the result back to the 3D feature space. Style features $F_s$ are extracted from the reference image by a VGG network.
    - Design Motivation: Ablation studies show that applying AdaIN directly without spatial mapping causes stylization failure, and applying AdaIN at every layer leads to loss of fine-grained details. Performing a single global AdaIN at the intermediate latent layer represents the optimal trade-off.

### Loss & Training
A two-stage training strategy is adopted:

**Stage 1 — Autoencoder Training**: The style module is frozen. The encoder and decoder are trained with an RGB reconstruction loss $L_{rgb} = \frac{1}{N}\sum_{i=1}^{N}(c_i - \hat{c}_i)^2$ to learn effective 3DGS feature extraction and color reconstruction.

**Stage 2 — Stylization Training**: The autoencoder is frozen and the style module is introduced. 3DGS scenes and style images are randomly selected; multi-view images are rendered and used to compute the style loss $L_{style} = L_c + \lambda L_s$ (content loss plus weighted style loss). A background masking mechanism is introduced to filter out background features, preventing background colors from biasing the VGG feature statistics and causing foreground color shift.

**Dataset**: 40,000 diverse 3DGS objects are generated from a subset of Objaverse (using TriplaneGaussian from single images); 90,000 artistic images from WikiArt are used as style references.

## Key Experimental Results

### Main Results

| Metric | StyleRF | StyleSplat | StyleGaussian | A3GS (Ours) |
|--------|---------|------------|---------------|-------------|
| Short-range LPIPS↓ | 0.092 | 0.033 | 0.035 | **0.033** |
| Short-range RMSE↓ | 0.084 | 0.045 | **0.033** | 0.029 |
| Long-range LPIPS↓ | 0.154 | 0.052 | 0.057 | 0.061 |
| Long-range RMSE↓ | 0.186 | 0.062 | 0.060 | **0.057** |

| Metric | StyleRF | StyleSplat | StyleGaussian | A3GS (Ours) |
|--------|---------|------------|---------------|-------------|
| Style Consistency↑ | 2.1 | 3.9 | 3.5 | **4.1** |
| Content Preservation↑ | 3.7 | **4.5** | 3.3 | **4.5** |
| Visual Naturalness↑ | 3.5 | 4.3 | 3.6 | **4.5** |

### Ablation Study

| Configuration | Effect | Note |
|---------------|--------|------|
| Full model (3D-GCN + AdaIN + Mask) | High-quality stylization | Complete model |
| MLP replacing GCN | Artifacts produced | Lack of local feature perception |
| Standard GCN replacing 3D-GCN | Content structure lost | Unable to extract 3D structural features effectively |
| AdaIN removed | Color tone change only | No true style transfer; filter-like effect |
| Background mask removed | Severe color shift | Background biases VGG feature statistics |
| AdaIN applied at every layer | Detail loss | Excessive stylization destroys fine-grained structure |

### Key Findings
- The learnable kernel shapes of 3D-GCN are critical — standard GCNs and MLPs both fail to extract effective local 3D features from unstructured 3DGS data.
- A3GS requires only 10 seconds on typical scenes from the TNT dataset, achieving a 30× speedup over StyleSplat. When the number of Gaussian primitives exceeds 48 million, StyleSplat runs out of memory, whereas A3GS can handle arbitrarily large scenes through batch processing (up to 15 million primitives per batch).
- Background masking is critical for training quality, as objects generated from Objaverse do not occupy the full image.

## Highlights & Insights
- **GCN Paradigm for 3DGS**: Treating 3DGS primitives as enriched point clouds and aggregating local features via graph convolution is a paradigm extensible to other downstream tasks on 3DGS, including editing, compression, and semantic segmentation.
- **Two-Stage Training Strategy**: Decoupled training — autoencoder first, stylization second — prevents interference between the two objectives and serves as a general paradigm for problems involving "feature space learning + feature space manipulation."
- **Batch Processing to Break Scale Limitations**: By exploiting the locality of feed-forward network processing, large scenes can be handled in batches, fundamentally resolving the memory bottleneck of optimization-based methods.

## Limitations & Future Work
- Generalization may be limited for rare or extreme artistic styles outside the training distribution.
- Reliance on local feature aggregation may cause inconsistencies in scenarios that require global, scene-level context.
- Fine-grained style details may be lost during the latent space transformation.
- Future work could consider incorporating global attention mechanisms or multi-scale style injection to improve global consistency.

## Related Work & Insights
- **vs. StyleGaussian**: StyleGaussian is also a feed-forward method but requires hours of per-scene training and is constrained by scene scale. A3GS achieves truly zero-shot transfer via the GCN autoencoder.
- **vs. StyleSplat / G-Style**: These optimization-based methods require hundreds of seconds of optimization per style. A3GS is two orders of magnitude faster and has no theoretical limitation on scene scale.
- **vs. 2D Style Transfer**: The core contribution of A3GS is extending the 2D AdaIN paradigm to 3D — the key innovation being the replacement of CNNs with 3D-GCN to address the feature extraction challenge posed by unstructured data.

## Rating
- Novelty: ⭐⭐⭐⭐ First truly zero-shot feed-forward 3DGS style transfer; the GCN-based approach for handling 3DGS is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative comparisons, user studies, and ablation experiments are all included; the speed comparison is highly convincing.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive illustrations, and complete method description.
- Value: ⭐⭐⭐⭐ Addresses the efficiency bottleneck of 3DGS style transfer with direct applicability to real-time applications in the metaverse and gaming.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)
- [\[CVPR 2026\] 3D Gaussian Splatting at Arbitrary Resolutions with Compact Proxy Anchors](../../CVPR2026/3d_vision/3d_gaussian_splatting_at_arbitrary_resolutions_with_compact_proxy_anchors.md)
- [\[AAAI 2026\] Arbitrary-Scale 3D Gaussian Super-Resolution](../../AAAI2026/3d_vision/arbitrary-scale_3d_gaussian_super-resolution.md)
- [\[NeurIPS 2025\] Styl3R: Instant 3D Stylized Reconstruction for Arbitrary Scenes and Styles](../../NeurIPS2025/3d_vision/styl3r_instant_3d_stylized_reconstruction_for_arbitrary_scenes_and_styles.md)
- [\[NeurIPS 2025\] CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting](../../NeurIPS2025/3d_vision/clipgaussian_universal_and_multimodal_style_transfer_based_on_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
