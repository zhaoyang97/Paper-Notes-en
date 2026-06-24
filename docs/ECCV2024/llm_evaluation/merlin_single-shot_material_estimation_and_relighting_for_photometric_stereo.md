---
title: >-
  [Paper Note] MERLiN: Single-Shot Material Estimation and Relighting for Photometric Stereo
description: >-
  [ECCV 2024][LLM Evaluation][Inverse Rendering] This paper proposes MERLiN, a single-stage attentive hourglass network, to jointly estimate spatially-varying BRDF parameters and perform physically correct relighting from a single image. It is the first to leverage relit images to drive photometric stereo methods for single-image normal estimation, bridging the gap between Shape from Shading and Photometric Stereo.
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "Inverse Rendering"
  - "Single-image Relighting"
  - "Photometric Stereo"
  - "svBRDF Estimation"
  - "Global Illumination"
date: 2026-05-08
content_hash: e9401ef9a684ee04
---

# MERLiN: Single-Shot Material Estimation and Relighting for Photometric Stereo

**Conference**: ECCV 2024  
**arXiv**: [2409.00674](https://arxiv.org/abs/2409.00674)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Inverse Rendering, Single-image Relighting, Photometric Stereo, svBRDF Estimation, Global Illumination

## TL;DR

This paper proposes MERLiN, a single-stage attentive hourglass network, to jointly estimate spatially-varying BRDF parameters and perform physically correct relighting from a single image. It is the first to leverage relit images to drive photometric stereo methods for single-image normal estimation, bridging the gap between Shape from Shading and Photometric Stereo.

## Background & Motivation

Photometric Stereo is a classical method to infer pixel-wise normal vectors by analyzing the appearance of objects under various lighting conditions, widely used in quality control, industrial inspection, medical imaging, and other fields. However, its core challenge lies in the **complex data acquisition**, which requires carefully designed controlled illumination environments and precise calibration, making it difficult to exhaustively acquire all lighting configurations in practice.

This study is driven by **three key questions**:

1. *Can deep learning be used to generate images under different lighting conditions?* — Image relighting has made progress, and CNN-based methods can achieve feed-forward relighting from a single image.
2. *Can synthesized images guarantee physical correctness?* — Perceptually photorealistic relit images may be physically incorrect (exhibiting deviations in shape and material parameters).
3. *How to verify physical correctness?* — Photometric stereo itself can serve as a validation tool: physically incorrect images will lead to erroneous normal estimations.

Key Insight: **Physically correct relighting requires a deep integration of material estimation and global illumination modeling**, rather than simple image translation.

## Method

### Overall Architecture

MERLiN is an hourglass network featuring a shared encoder and dual decoders:

```
Input Image → Shared Encoder → Material Decoder → (A, N, D, R)
                              ↘ Relighting Decoder → Relit Image
                                                    ↘ Global Illumination Network → Indirect Illumination Residual
```

| Module | Input | Output | Characteristics |
|------|------|------|------|
| Shared Encoder $f_{enc}$ | Input Image × mask | Feature $Z_{enc}$ | Extracts hierarchical features |
| Material Decoder $f_{mat}$ | $Z_{enc}$ + skip connection | Albedo (A), Normal (N), Depth (D), Roughness (R) | Jointly estimates four parameters with a single decoder |
| BRDF Rendering Layer $f_{BRDF}$ | A, N, D, R + Light Direction | Direct Illumination Image $I^{(d)}$ | Physical rendering based on the microfacet model |
| Global Illumination Network $f_{gl}$ | $I^{(d)}$ + A, R, N, D | Indirect Illumination Residual $I_{gl}$ | Achieves end-to-end global illumination modeling |
| Relighting Decoder $f_{rel}$ | $Z_{enc}$ + $Z_{mat}$ + Target Light | Relit Image | Fuses material features via attention gating |

### Key Designs

**1. Attention-Gated Feature Fusion**

Direct concatenation of skip connections and decoder features yields poor results (due to redundancy and noise). An attention-gating mechanism is adopted: using coarse-scale information from the decoder as a gating signal, it adaptively filters irrelevant or noisy responses in the skip connections, while capturing both local (surface roughness, textures) and global (light intensity attenuation, specular highlights) effects.

**2. End-to-End Global Illumination Modeling**

Different from the two-stage cascaded training of Li et al., MERLiN's global illumination network is jointly trained end-to-end with the BRDF estimation network. The global illumination network predicts the combined indirect illumination (the sum of multiple bounces) instead of modeling individual bounces sequentially. Experiments demonstrate that this collaborative training outperforms stage-wise training.

When trained using only direct illumination, the network predicts brighter albedos and flattened normals—which aligns with physical intuition: the absence of indirect illumination compensation leads to overcompensation in the albedo.

**3. Dual-path Relighting**

- **Rel-$f_{BRDF}$**: Direct physical re-rendering using the estimated BRDF parameters + indirect illumination added by the global illumination network. This better captures specular highlights.
- **Rel-$f_{rel}$**: An independent CNN-based relighting decoder with bi-directional skip connections to the material decoder. Joint training mutually enhances the accuracy of material estimation.

### Loss & Training

The total loss is a weighted sum of six L2 loss terms:

$$\mathcal{L} = \lambda_a\mathcal{L}_a + \lambda_n\mathcal{L}_n + \lambda_d\mathcal{L}_d + \lambda_r\mathcal{L}_r + \lambda_{rec}\mathcal{L}_{rec} + \lambda_{rel}\mathcal{L}_{rel}$$

where $\lambda_a = \lambda_r = \lambda_d = \lambda_{rec} = \lambda_{rel} = 1.0$, and $\lambda_n = 2.0$ (doubling the normal weight). An additional gradient L2 loss is applied to the roughness map to prevent oversmoothing.

Training is conducted on an NVIDIA RTX 5000 GPU with a batch size of 64, using the Adam optimizer. The initial learning rate is $1\times10^{-4}$ (encoder) / $2\times10^{-4}$ (decoder) and is halved every 5 epochs, for a total of 25 epochs. Target relit images are rendered on-the-fly during training via $f_{BRDF}$ (with random light directions in the upper hemisphere).

## Key Experimental Results

### Main Results

**Quantitative Comparison of svBRDF Estimation and Relighting** (Test set MSE $\times 10^{-2}$)

| Method | Albedo↓ | Roughness↓ | Normal↓ | Depth↓ | Relighting(SSIM)↑ |
|------|---------|------------|---------|--------|-------------------|
| Li et al. [22] | 4.868 | 19.431 | 3.822 | 1.505 | 0.884 |
| Sang et al. [34] | 3.856 | 12.781 | 3.459 | 1.471 | 0.872 |
| **MERLiN (Ours)** | **3.787** | **8.267** | **3.311** | **0.975** | **0.894** |

MERLiN achieves significant improvements across all svBRDF parameters, with a particularly notable gain in roughness estimation (8.267 vs 12.781), and yields the best relighting SSIM.

### Ablation Study

**Impact of Network Design Choices**

| Design | Albedo↓ | Roughness↓ | Normal↓ | Rel-frel↑ | Rel-fBRDF↑ |
|------|---------|------------|---------|-----------|------------|
| No Feature Sharing + No Attention + No GI | 6.154 | 18.071 | 4.681 | 0.697 | 0.719 |
| + Attention Gating | 5.519 | 15.277 | 3.975 | 0.701 | 0.757 |
| + Global Illumination | 5.614 | 14.485 | 3.887 | 0.746 | 0.789 |
| + Feature Sharing + Attention | 4.162 | 9.681 | 3.406 | 0.798 | 0.859 |
| **Full Model** | **3.787** | **8.267** | **3.311** | **0.819** | **0.894** |

### Key Findings

1. **Critical Role of Global Illumination**: Training solely on direct illumination images generalizes extremely poorly to real-world images, as pure direct illumination rarely exists in real scenes.
2. **Mutual Benefit of Joint Training**: Jointly training the relighting decoder and the material decoder is mutually beneficial, resulting in more accurate material estimation and more physically correct relighting.
3. **Single vs. Quad Decoders**: A single material decoder achieves performance close to four independent decoders but offers significant advantages in parameter size and speed (with only a slight drop in albedo).
4. **Photometric Stereo Validation**: Feeding 32 images relit by MERLiN into Fast-NFPS yields a mean angular error for normal estimation of 15.80°, which outperforms Sang et al. (16.43°) and Li et al. (16.21°), though it is still higher than using ground-truth images (14.11°).

## Highlights & Insights

- **Bridging Role**: It is the first to bridge Shape from Shading (single image) and Photometric Stereo (multi-light) via relighting, opening up a new direction of "single-image photometric stereo."
- **Physical Validation Loop**: Utilizing photometric stereo to validate the physical correctness of relighting, instead of relying solely on perceptual metrics, provides an elegant closed-loop validation idea.
- **Single-stage Outperforms Cascaded**: MERLiN demonstrates that in inverse rendering, an end-to-end single-stage design can outperform multi-stage cascaded designs, simplifying the pipeline while boosting performance.
- **A Caveat on "Perceptually Correct but Physically Incorrect"**: Two sets of visually similar relit images can yield vastly different reconstructed normals, reminding the community that relighting cannot be evaluated purely through perceptual metrics.

## Limitations & Future Work

1. **Inherent Ill-posedness of Single-Image Input**: Single-image inverse rendering is highly under-constrained and still exhibits significant ambiguities.
2. **Trained Only on Synthetic Data**: Quantitative evaluations are limited to synthetic datasets since real-world data lacks ground truth.
3. **Limitations of Residual Global Illumination**: It approximates GI only in image space, failing to handle inter-reflections from surfaces invisible to the camera.
4. **Near-field Point Light Assumption**: Training data is dominated by co-located near-field point light sources, leading to limited generalization to other lighting types.
5. **Performance Gap in Photometric Stereo**: The normal estimation error from relit images is still about 1.7° higher than that from real images, indicating room for improvement in physical accuracy.

## Related Work & Insights

- **Core Differences from Li et al. and Sang et al.**: (1) A single-stage architecture instead of a cascaded one; (2) End-to-end global illumination training instead of stage-wise training.
- **Complementarity with NeRF-based Methods**: NeRF-based methods can achieve high-quality relighting but require multiple images and scene-specific optimization, whereas MERLiN performs feed-forward single-image inference.
- **Insights**: Jointly training correlated tasks (material estimation and relighting) can establish a virtuous cycle; a similar strategy could be applied to multi-task scenarios involving depth, normals, segmentation, etc.

## Rating

| Dimension | Score (1-5) | Evaluation |
|------|-----------|------|
| Novelty | 4 | The concept of single-image photometric stereo is novel, and the physical validation closed-loop is elegant. |
| Technical Depth | 4 | The network design is well-thought-out, and the integration of global illumination is clever. |
| Experimental Thoroughness | 4 | Detailed ablation studies; photometric stereo validation is a unique contribution. |
| Writing Quality | 4 | The narrative structure driven by three key questions is clean and compelling. |
| Value | 3.5 | Holds potential applications for industrial inspection and cultural heritage preservation, but practical deployment requires more validation on real-world data. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Neural Multi-View Self-Calibrated Photometric Stereo without Photometric Stereo Cues](../../ICCV2025/llm_evaluation/neural_multi-view_self-calibrated_photometric_stereo_without_photometric_stereo_.md)
- [\[ECCV 2024\] Instance-dependent Noisy-label Learning with Graphical Model Based Noise-rate Estimation](instance-dependent_noisy-label_learning_with_graphical_model_based_noise-rate_es.md)
- [\[ECCV 2024\] Spherical Linear Interpolation and Text-Anchoring for Zero-shot Composed Image Retrieval](spherical_linear_interpolation_and_text-anchoring_for_zero-shot_composed_image_r.md)
- [\[ECCV 2024\] Deep Cost Ray Fusion for Sparse Depth Video Completion](deep_cost_ray_fusion_for_sparse_depth_video_completion.md)
- [\[ECCV 2024\] Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence](learn_from_the_learnt_source-free_active_domain_adaptation_via_contrastive_sampl.md)

</div>

<!-- RELATED:END -->
