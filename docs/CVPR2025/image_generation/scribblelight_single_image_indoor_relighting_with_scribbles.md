---
title: >-
  [Paper Note] ScribbleLight: Single Image Indoor Relighting with Scribbles
description: >-
  [CVPR 2025][Image Generation][Indoor Relighting] ScribbleLight proposes a scribble-guided generative model for single-image indoor relighting. It preserves the original texture and color using Albedo-conditioned Stable Image Diffusion, and introduces an encoder-decoder ControlNet architecture to achieve geometry-preserving, fine-grained illumination control. Users can easily perform actions such as turning lights on/off and casting shadows using simple scribbles.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Indoor Relighting"
  - "Scribble Control"
  - "Stable Diffusion"
  - "ControlNet"
  - "Intrinsic Image Decomposition"
date: 2026-05-08
content_hash: 57740f4545b79426
---

# ScribbleLight: Single Image Indoor Relighting with Scribbles

**Conference**: CVPR 2025  
**arXiv**: [2411.17696](https://arxiv.org/abs/2411.17696)  
**Code**: None (Project Page: [https://chedgekorea.github.io/ScribbleLight/](https://chedgekorea.github.io/ScribbleLight/) )  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Indoor Relighting, Scribble Control, Stable Diffusion, ControlNet, Intrinsic Image Decomposition

## TL;DR

ScribbleLight proposes a scribble-guided generative model for single-image indoor relighting. It preserves the original texture and color using Albedo-conditioned Stable Image Diffusion, and introduces an encoder-decoder ControlNet architecture to achieve geometry-preserving, fine-grained illumination control. Users can easily perform actions such as turning lights on/off and casting shadows using simple scribbles.

## Background & Motivation

**Background**: Image relighting has crucial applications in areas such as real estate, virtual staging, and interior design. Outdoor relighting is relatively simple because the primary light source (the sun) is single and predictable. Indoor scenes are the most challenging for relighting, as they involve multiple light sources (ceiling lights, table lamps, light transmitted through windows, etc.), which generate complex overlapping soft shadows.

**Limitations of Prior Work**: Existing 3D relighting methods require dense scene acquisition. Implicit methods (e.g., latent space editing) can only achieve coarse-grained global illumination changes, failing to control local details. Explicit lighting representations (e.g., Spherical Harmonics, Spherical Gaussians, or irradiance fields) provide users with indirect and complex control interfaces. While users want to directly annotate "where to make brighter and where to make darker," no existing method supports scribble-driven indoor relighting.

**Key Challenge**: Scribbles are extremely sparse control signals, providing only high-level guidance. A core technical challenge is how to generate physically plausible lighting effects from such sparse inputs while keeping the original image colors and textures (i.e., albedo) intact.

**Goal**: To design a generative model that allows users to achieve diverse indoor lighting effects, including turning lights on/off, adding highlights, and casting shadows, using simple binary scribbles (1 = brighter, 0 = darker).

**Key Insight**: Leverage the general image priors embedded in a large-scale pre-trained diffusion model (Stable Diffusion v2) to resolve the ambiguity of scribble guidance, while preserving the intrinsic properties of the original image through albedo conditioning.

**Core Idea**: A two-stage training scheme: first, fine-tune an Albedo-conditioned SD to learn image generation under various illumination conditions while preserving the albedo; second, train a ControlNet to accept scribbles and normal maps to guide the relighting effects.

## Method

### Overall Architecture

ScribbleLight employs a two-stage training process. In the first stage, the albedo image is encoded into a latent representation and concatenated with the noisy image as input to the U-Net, training the Albedo-conditioned SD. In the second stage, the ScribbleLight ControlNet is trained, which consists of an encoder-decoder structure: the encoder encodes scribbles and normal maps into illumination feature maps, while the decoder reconstructs normals and target shading to regularize the encoded representations. The encoder output is injected into the first-stage SD model to guide the generation.

### Key Designs

1. **Albedo-conditioned Stable Image Diffusion**:

    - **Function**: Preserving the color and texture of the original image during relighted image generation.
    - **Mechanism**: Map the image $I$ and the albedo $A$ to latent vectors $z^I$ and $z^A$ using a VAE encoder. The image latent vector is diffused with noise at timestep $t$, while a fixed amount of noise ($T=200$) is added to the albedo latent vector. Both are concatenated along the feature dimension and fed into the SD U-Net (with doubled input channels and zero-initialized new weights). The training objective is $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_{\theta^S}(z_t^I, z_T^A, t, p)\|_2^2]$.
    - **Design Motivation**: Feeding the precise albedo directly causes the model to rely excessively on it, resulting in insufficient lighting changes. Additionally, errors from the albedo predictor would propagate directly as artifacts. Adding fixed-step noise introduces uncertainty, which preserves the fundamental color structure while forcing the model to rely more on image priors.

2. **ScribbleLight ControlNet Encoder-Decoder**:

    - **Function**: Extracting control features containing 3D geometry and illumination information from scribbles and normal maps.
    - **Mechanism**: The encoder $\mathcal{E}^C$ encodes the concatenated scribble $M$ and normal $N$ into an illumination feature map $f$. The decoder $\mathcal{D}^C$ reconstructs the normals and monochromatic shading from the feature map: $$\mathcal{L}_D = \|\mathcal{D}^C(\mathcal{E}^C(M,N)) - (S_{mono}, N)\|_2^2$$. ControlNet takes the feature map $f$, the noisy latent vector $z_t^I$, and the text prompt $p$ as inputs. It is initialized with the original Stable Diffusion v2 weights (non-albedo conditioned version) and jointly trained.
    - **Design Motivation**: A pure encoder lacks constraints, which may cause latent features to lose geometric information. Reconstructing normals and shading via the decoder ensures that the features capture complete information required for relighting. Removing the normal maps leads to the generation of random objects, while removing the decoder leads to hallucinations.

3. **Automatic Scribble Generation Strategy**:

    - **Function**: Automatically creating training scribble annotations from real image datasets.
    - **Mechanism**: Based on thresholding the shading intensity distribution—pixels with $I(x)>\mu+\sigma$ are labeled as bright (1), $I(x)<\mu-\sigma$ as dark (0), and others as neutral (0.5). To simulate the roughness of real user scribbles, random-sized morphological dilation and erosion (using $3 \times 3$ to $19 \times 19$ kernels) are applied.
    - **Design Motivation**: There are no paired scribble-relighting datasets. Threshold boundaries that are strictly aligned with image content do not resemble real human scribbles; morphological operations are used to break this alignment.

### Loss & Training

- The first stage is trained on a 100K subset of LSUN Bedrooms, where albedos are predicted by an IID method and text prompts are generated by BLIP-2.
- The second stage freezes the Albedo SD and trains the ControlNet + Encoder-Decoder separately. Normals are predicted by DSINE, and shading is extracted by an IID method.
- The albedo noise is fixed at step $T=200$, which was empirically found to be optimal.

## Key Experimental Results

### Main Results

| Method | RMSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|--------|---------|
| LightIt* | 0.341(0.302) | 9.61(10.65) | 0.232(0.332) | 0.564(0.518) |
| RGB↔X | 0.269(0.251) | 12.47(12.99) | 0.416(0.437) | 0.439(0.425) |
| **ScribbleLight** | **0.206(0.190)** | **14.29(15.01)** | **0.436(0.504)** | **0.394(0.370)** |

Evaluated on 206 pairs of test images from the BigTime timelapse dataset. Reports mean(best) values over 5 random seeds.

### Ablation Study

| Albedo Conditioning Type | Noising | RMSE ↓ | PSNR ↑ | LPIPS ↓ |
|----------------|------|--------|--------|---------|
| ControlNet Input | - | 0.2305 | 13.19 | 0.4839 |
| SD Conditioning | No | 0.2082 | 14.07 | 0.4193 |
| SD Conditioning | Yes | **0.2059** | **14.29** | **0.3942** |

| Normal Map | Decoder | RMSE ↓ | PSNR ↑ | LPIPS ↓ |
|--------|--------|--------|--------|---------|
| × | ✓ | 0.2224 | 13.61 | 0.4251 |
| ✓ | × | 0.2098 | 14.06 | 0.4093 |
| ✓ | ✓ | **0.2059** | **14.29** | **0.3942** |

### Key Findings

- Albedo-conditioned SD performs significantly better than injecting albedo into ControlNet (LPIPS 0.3942 vs. 0.4839).
- Adding noise to the albedo latent space significantly enhances both robustness and lighting diversity.
- The normal map and the control decoder contribute independently; missing either leads to artifacts or geometric inconsistencies.
- Even if the scribbles are physically inconsistent, the model can still generate visually plausible results (e.g., "imagining" out-of-frame light sources).
- Results generated with different random seeds consistently follow the scribble guidance while providing diverse illumination variations.
- Supports progressive scribbling (coarse-to-fine), enabling iterative refinement by the user.

## Highlights & Insights

1. Scribbles serve as an intuitive and natural interaction modality for relighting control, significantly lowering the user barrier.
2. Adding fixed noise to the albedo condition is simple yet effective, tolerating prediction errors while promoting lighting diversity.
3. The encoder-decoder regularization ensures that the latent features encode informative geometric and illumination characteristics.
4. The model automatically generates plausible secondary lighting effects (such as soft halos around lamps), even if they are not explicitly specified in the scribbles.

## Limitations & Future Work

- Cannot correct highly physically inconsistent scribbles, which may result in implausible lighting effects.
- Color lighting adjustment is not supported; the generated results tend to bias towards common colors (e.g., yellow, blue).
- The training data is limited to LSUN Bedrooms; generalizing to other indoor scenes requires more diverse datasets.
- Future work could support color scribbles to control the color of light sources.

## Related Work & Insights

- Compared to LightIt, albedo conditioning significantly improves texture and color preservation.
- Compared to the intrinsic decomposition-recomposition approach of RGB↔X, scribbles do not require pixel-accurate shading.
- The concept of scribble control can be extended to other image editing tasks that require fine-grained, localized user control.

## Rating

- **Novelty**: 7/10 — The combination of scribbles and relighting is novel, though the technical components (albedo-conditioned SD, ControlNet) have precedents.
- **Experimental Thoroughness**: 7/10 — Quantitative comparisons and ablation studies are provided, but the test set is limited and a user study is missing.
- **Writing Quality**: 8/10 — Clear structure, rich illustrations, and well-defined problem formulation.
- **Value**: 7/10 — Provides a practical and intuitive tool for indoor lighting editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting](luminet_latent_intrinsics_meets_diffusion_models_for_indoor_scene_relighting.md)
- [\[CVPR 2025\] Comprehensive Relighting: Generalizable and Consistent Monocular Human Relighting and Harmonization](comprehensive_relighting_generalizable_and_consistent_monocular_human_relighting.md)
- [\[CVPR 2025\] ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models](ice_intrinsic_concept_extraction_from_a_single_image_via_diffusion_models.md)
- [\[CVPR 2026\] Learning Latent Proxies for Controllable Single-Image Relighting](../../CVPR2026/image_generation/learning_latent_proxies_for_controllable_single-image_relighting.md)
- [\[CVPR 2025\] RoomPainter: View-Integrated Diffusion for Consistent Indoor Scene Texturing](roompainter_view-integrated_diffusion_for_consistent_indoor_scene_texturing.md)

</div>

<!-- RELATED:END -->
