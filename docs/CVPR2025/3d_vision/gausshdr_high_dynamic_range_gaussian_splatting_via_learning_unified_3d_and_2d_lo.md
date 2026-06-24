---
title: >-
  [Paper Note] GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping
description: >-
  [CVPR 2025][3D Vision][HDR Novel View Synthesis] This paper proposes GaussHDR, which improves HDR Gaussian Splatting by unifying 3D and 2D local tone mapping. By designing a residual local tone mapper and an uncertainty-adaptive modulation mechanism, it simultaneously enhances HDR reconstruction stability and LDR fitting quality, significantly outperforming existing methods on both synthetic and real-world scenes.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "HDR Novel View Synthesis"
  - "Gaussian Splatting"
  - "Tone Mapping"
  - "Uncertainty Learning"
  - "Local Tone Mapping"
date: 2026-05-08
content_hash: f8821d4a7750fc29
---

# GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping

**Conference**: CVPR 2025  
**arXiv**: [2503.10143](https://arxiv.org/abs/2503.10143)  
**Code**: [https://liujf1226.github.io/GaussHDR](https://liujf1226.github.io/GaussHDR) (Project Page)  
**Area**: 3D Vision  
**Keywords**: HDR Novel View Synthesis, Gaussian Splatting, Tone Mapping, Uncertainty Learning, Local Tone Mapping

## TL;DR
This paper proposes GaussHDR, which improves HDR Gaussian Splatting by unifying 3D and 2D local tone mapping. By designing a residual local tone mapper and an uncertainty-adaptive modulation mechanism, it simultaneously enhances HDR reconstruction stability and LDR fitting quality, significantly outperforming existing methods on both synthetic and real-world scenes.

## Background & Motivation

1. **Background**: HDR Novel View Synthesis (NVS) reconstructs HDR scenes using multi-view LDR images with different exposures. Mainstream methods extend color representation from LDR to HDR and then use a tone mapper to model the Camera Response Function (CRF) to map HDR irradiance to LDR.

2. **Limitations of Prior Work**: There exists a dilemma between two training paradigms: (a) **3D tone mapping** (mapping per-Gaussian before rendering): LDR fitting is good, but HDR reconstruction is unstable because HDR rendering is decoupled from LDR supervision, potentially leading to inconsistent HDR and LDR distributions along the ray; (b) **2D tone mapping** (rendering the HDR image before mapping): HDR reconstruction is stable, but LDR fitting degrades because ray accumulation in the HDR range $[0, +\infty)$ is less robust than that in the LDR range $[0, 1]$.

3. **Key Challenge**: 3D and 2D tone mapping have complementary strengths and weaknesses. Meanwhile, existing methods use global tone mappers, applying the same mapping characteristics to the entire scene and ignoring fine-grained differences across different spatial locations.

4. **Goal**: (i) How to combine the advantages of both 3D and 2D tone mapping? (ii) How to achieve localized, adaptive tone mapping? (iii) How to adaptively balance the weight of the two in different scenes?

5. **Key Insight**: Introduce context features for each Gaussian as an extra input to the tone mapper to achieve localization, and adaptively balance 3D and 2D results through uncertainty learning.

6. **Core Idea**: Utilize a shared residual local tone mapper to perform both 3D and 2D local tone mapping simultaneously, and adaptively fuse the dual-path LDR results at the loss level via uncertainty learning.

## Method

### Overall Architecture
Based on 3DGS, each Gaussian additionally stores a context feature $f \in \mathbb{R}^d$ in addition to the HDR irradiance. During rendering, an HDR image $E$ and a context feature map $F$ are simultaneously outputted. Through a shared residual local tone mapper, 3D tone mapping (per-Gaussian) and 2D tone mapping (per-pixel) are performed separately to obtain dual-path LDR results $I_{3d}^*$ and $I_{2d}^*$. An uncertainty predictor is employed to adaptively fuse the dual-path results at the loss level.

### Key Designs

1. **Residual Local Tone Mapper**:

    - **Function**: Add local adaptive adjustments on top of global tone mapping.
    - **Mechanism**: Decompose local tone mapping into a global mapping and a residual term: $c^*=g(\ln(et))+\Delta g([\ln(et), f])$, where $g$ is the global tone mapping MLP, $\Delta g$ is the residual MLP, and $f$ is the context feature. The same tone mapper $g^*$ is used for both 3D local mapping (taking per-Gaussian feature $f_i$ as input) and 2D local mapping (taking rendered pixel feature $F$ as input). Only the global mapping is trained in the first 6K steps, after which the residual term is enabled for joint optimization.
    - **Design Motivation**: Learning local mapping directly is difficult and resource-intensive; the residual design allows the global MLP to provide the foundational mapping, while the residual MLP only needs to capture local variations, reducing the learning difficulty. The context feature naturally achieves consistent transfer from 3D to 2D through rendering.

2. **Context Feature for Unified Local TM**:

    - **Function**: Provide spatially-aware local features for tone mapping.
    - **Mechanism**: Each Gaussian stores a context feature $f_i \in \mathbb{R}^d$ ($d=4$). During 3D tone mapping: $c_i^*=g^*(\ln(e_i t), f_i)$, which is then rendered to obtain $I_{3d}^*$. The context features can be rendered to pixels via alpha-blending just like colors: $F=\mathcal{R}_P(\{f_i\})$. Each pixel, after receiving its corresponding feature, undergoes 2D tone mapping: $I_{2d}^*=g^*(\ln(Et), F)$. This design is inspired by language-embedded scene representations (e.g., LangSplat)—the rendered pixel features and 3D Gaussian features reside in the same semantic space.
    - **Design Motivation**: Global mappers assume all locations share the same mapping characteristics, which does not hold in scenes with complex lighting. By introducing local characteristics through context features, and leveraging the continuity of Gaussian rendering, consistency between 3D and 2D feature spaces is guaranteed.

3. **Uncertainty-based Joint Learning**:

    - **Function**: Adaptively balance the contributions of 3D and 2D tone mapping results.
    - **Mechanism**: Train an uncertainty MLP $\rho$ to predict uncertainty maps $U_{3d}$ and $U_{2d}$ for the 3D and 2D results respectively. The joint loss is formulated as $\mathcal{L}_{gs}=(U_{2d}^2 \mathcal{L}_{3d}+U_{3d}^2 \mathcal{L}_{2d})/(U_{3d}^2+U_{2d}^2)$—the path with higher uncertainty receives a lower weight. Uncertainty is optimized separately through a DSSIM-based loss (with gradient stop), decoupling it from the main model training. During inference, uncertainty is also utilized to fuse the dual-path LDR: $I_{merge}=(U_{2d}^2 I_{3d}^*+U_{3d}^2 I_{2d}^*)/(U_{3d}^2+U_{2d}^2)$.
    - **Design Motivation**: The optimal balance between 3D and 2D tone mapping varies across different scenes. Fixed weights cannot adapt to all scenarios. Uncertainty learning allows the model to automatically identify the reliability of each mapping at each pixel, achieving pixel-level adaptation.

### Loss & Training
The total loss is $\mathcal{L}=\mathcal{L}_{gs}+\mathcal{L}_{unc}+\lambda_e \mathcal{L}_e$. $\mathcal{L}_{gs}$ is the uncertainty-weighted reconstruction loss (DSSIM + L1); $\mathcal{L}_{unc}$ is the uncertainty prediction loss; $\mathcal{L}_e$ is the unit-exposure constraint $\|\ln(g(0)) - \ln(0.73)\|_2^2$ (or specifically $\|\text{sg}(g(0))-0.73\|_2^2$ as per $\|g(0)-0.73\|_2^2$) for synthetic scenes. $\lambda_d=0.2, \lambda_u=0.5, \lambda_e=0.5$. All MLPs have only 1 hidden layer with 64 channels. The training consists of 30K iterations, with only global mapping in the first 6K. Trained on a single RTX 3090 GPU.

## Key Experimental Results

### Main Results

| Dataset | Setting | Metric | GaussHDR(3DGS) | HDR-GS(SOTA) | GaussHDR(Scaffold) | Gain |
|--------|------|------|----------------|-------------|---------------------|------|
| HDR-NeRF Real | LDR-OE PSNR↑ | dB | 35.78 | 35.47 | **36.77** | +1.3 |
| HDR-NeRF Real | LDR-NE PSNR↑ | dB | **33.33** | 31.66 | **33.92** | +2.3 |
| HDR-Plenoxels Real | LDR-OE PSNR↑ | dB | 31.51 | 31.04 | **32.85** | +1.8 |
| HDR-NeRF Synth | LDR-OE PSNR↑ | dB | 42.29 | 41.13 | **43.78** | +2.7 |
| HDR-NeRF Synth | HDR PSNR↑ | dB | 37.62 | 26.98* | **39.02** | +12 |

\*Note: HDR-GS fails dramatically in HDR reconstruction under fair comparison settings (without HDR GT supervision).

### Ablation Study

| Configuration | LDR-OE PSNR | LDR-NE PSNR | HDR PSNR | Explanation |
|------|-------------|-------------|----------|------|
| 3D Global TM | 33.94 | 31.88 | 26.11 | 3D global mapping (poor HDR) |
| 3D Local TM | 34.41 | 32.56 | 26.78 | Localization improves LDR |
| 2D Global TM | 32.45 | - | - | 2D global (poor LDR) |
| 3D+2D Global TM | 34.42 | 32.54 | 28.22 | Joint global |
| 3D+2D Local TM (w/o Uncertainty) | 35.51 | 33.48 | 34.12 | Joint local significantly improves HDR |
| Full (with Uncertainty) | **36.77** | **33.92** | **35.47** | Additional gain from uncertainty |

### Key Findings
- **Joint 3D+2D training is key to HDR reconstruction**: Jumping from 26.11 PSNR for 3D-only to 34.12 for joint, yielding an 8dB gain in HDR.
- **Local vs. global tone mapping**: On LDR-OE, rising from 33.94 $\rightarrow$ 34.41 (3D) and 34.42 $\rightarrow$ 35.51 (joint), showing that localization consistently contributes positively across all settings.
- **Necessity of uncertainty fusion**: Moving from 35.51 $\rightarrow$ 36.77 (LDR-OE), bringing an additional 1.3dB improvement.
- **Crucial residual design**: HDR visualization comparisons indicate that local mapping with the residual design preserves more HDR details, whereas the version without the residual design suffers from degraded tone mapping results.
- **Good representation compatibility**: Both 3DGS and Scaffold-GS serve effectively as backbones, with Scaffold-GS performing better.

## Highlights & Insights
- **Insight into the complementarity of 3D/2D tone mapping**: The paper precisely diagnoses the failure modes of both paradigms—in 3D mapping, the inconsistency of HDR-LDR distribution along rays causes HDR to fall into local optima, while in 2D mapping, ray accumulation over an infinite HDR range is less robust than that over a bounded LDR range. This analysis is valuable in itself.
- **Reusing context features for both 3D and 2D**: A single feature serves both mapping methods, naturally achieving cross-domain consistency through rendering. This "3D attribute $\rightarrow$ 2D map" design concept can be generalized to other tasks requiring 3D-2D consistency.
- **Uncertainty-driven pixel-level fusion**: It is more elegant than manually tuning weights, and can also be used to fuse the dual-path results during inference, which practically improves the final rendering quality.
- **Simplicity of the residual design**: The approach of using global mapping as a base coupled with residual fine-tuning is simple and effective, and is applicable to any global model that requires localization.

## Limitations & Future Work
- The context feature dimension is only 4, which may limit the expressiveness of the local mapping. Whether higher-dimensional features yield better results is worth exploring.
- The switching point of the two-stage training (global for the first 6K $\rightarrow$ local thereafter) is manually set, and different scenes may require different strategies.
- HDR reconstruction for dynamic scenes is not considered.
- The uncertainty MLP and tone mapping MLP share inputs but have decoupled gradients; whether there exists a better joint training method remains to be seen.
- The experiments are mainly validated on small-scale indoor scenes; the performance on large-scale outdoor scenes remains unknown.

## Related Work & Insights
- **vs. HDR-GS**: HDR-GS uses 3D tone mapping, yielding unstable HDR reconstruction (only 26.98 PSNR under fair settings). GaussHDR addresses this core issue through joint 3D+2D local mapping.
- **vs. HDR-Plenoxels**: Uses 2D tone mapping, which limits the LDR quality. The 3D branch of GaussHDR compensates for this limitation.
- **vs. HDR-NeRF**: NeRF-based methods are slow and similarly employ 3D global mapping. GaussHDR comprehensively surpasses them in both efficiency and quality.
- The framework concept of context features + rendering shares similarities with semantic field representations like LangSplat and LERF. It could be considered to combine semantic features for semantic-aware HDR reconstruction.

## Rating
- Novelty: ⭐⭐⭐⭐ The approach of unified 3D+2D local tone mapping is novel, and the uncertainty-based fusion design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Very thorough, with validations on multiple synthetic and real datasets, complete ablation studies, and two underlying representations.
- Writing Quality: ⭐⭐⭐⭐ The problem analysis is clear and thorough, and the methodology is presented in an orderly manner.
- Value: ⭐⭐⭐⭐ A significant advancement in HDR NVS, particularly solving the critical issue of unstable HDR reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [\[CVPR 2025\] Event Fields: Capturing Light Fields at High Speed, Resolution, and Dynamic Range](event_fields_capturing_light_fields_at_high_speed_resolution_and_dynamic_range.md)
- [\[CVPR 2025\] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting](rethinking_end-to-end_2d_to_3d_scene_segmentation_in_gaussian_splatting.md)
- [\[CVPR 2025\] IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing](irgs_inter-reflective_gaussian_splatting_with_2d_gaussian_ray_tracing.md)
- [\[CVPR 2025\] UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping](uvgs_reimagining_unstructured_3d_gaussian_splatting_using_uv_mapping.md)

</div>

<!-- RELATED:END -->
