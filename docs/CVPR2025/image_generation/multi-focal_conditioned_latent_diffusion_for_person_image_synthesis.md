---
title: >-
  [Paper Note] Multi-focal Conditioned Latent Diffusion for Person Image Synthesis
description: >-
  [CVPR 2025][Image Generation][Pose-Guided Person Image Synthesis] MCLD decouples the source person image into three focal conditions: face region, appearance texture, and overall image. It designs a Multi-Focal Condition Aggregation (MFCA) module to selectively inject different conditions at different stages of the UNet, effectively mitigating the face and texture detail degradation caused by LDM compression and achieving SOTA results on DeepFashion.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Pose-Guided Person Image Synthesis"
  - "Latent Diffusion Models"
  - "Multi-focal Conditioning"
  - "Face Identity Preservation"
  - "Appearance Texture Preservation"
date: 2026-05-08
content_hash: e4513655c018fc03
---

# Multi-focal Conditioned Latent Diffusion for Person Image Synthesis

**Conference**: CVPR 2025  
**arXiv**: [2503.15686](https://arxiv.org/abs/2503.15686)  
**Code**: [https://github.com/jqliu09/mcld](https://github.com/jqliu09/mcld)  
**Area**: Diffusion Models / Person Image Generation  
**Keywords**: Pose-Guided Person Image Synthesis, Latent Diffusion Models, Multi-focal Conditioning, Face Identity Preservation, Appearance Texture Preservation

## TL;DR
MCLD decouples the source person image into three focal conditions: face region, appearance texture, and overall image. It designs a Multi-Focal Condition Aggregation (MFCA) module to selectively inject different conditions at different stages of the UNet, effectively mitigating the face and texture detail degradation caused by LDM compression and achieving SOTA results on DeepFashion.

## Background & Motivation
1. **Background**: Pose-Guided Person Image Synthesis (PGPIS) aims to transfer a source image to a target pose while maintaining appearance and identity. GAN-based methods are limited by training instability and mode collapse, whereas LDM-based methods (such as PIDM, CFLD, and PoCoLD) achieve better generation quality and high-resolution support due to operations in the latent space.
2. **Limitations of Prior Work**: The autoencoder compression process of LDMs is lossy, particularly degrading details in high-frequency regions—facial features and clothing textures are severely lost during the encoding process. This degradation worsens during inference because the generated latent deviates from the real compressed latent (as shown in Figure 1, where adding a tiny shift $\epsilon=0.2$ significantly degrades details).
3. **Key Challenge**: LDM methods use the entire source image as a condition, failing to focus on critical sensitive regions such as the face and textures. The entanglement of pose and appearance information also makes detail reconstruction across diverse poses more difficult.
4. **Goal**: (1) How to mitigate LDM compression degradation of face and texture? (2) How to design conditional controls to make the model focus on sensitive areas? (3) How to maintain identity consistency across different poses?
5. **Key Insight**: Instead of using the whole image as the sole condition, the image is decoupled into three independent components: face identity, appearance texture, and global semantics. These are encoded into pose-invariant embeddings using different pre-trained models and then selectively injected at different stages of the UNet.
6. **Core Idea**: Through multi-focal condition decoupling and stage-selective injection, the LDM is guided to focus on different levels of information at different processing stages, thereby maximizing the preservation of face identity and appearance details under compression constraints.

## Method

### Overall Architecture
A dual-branch conditional diffusion model is constructed based on Stable Diffusion 1.5. The inputs are the source person image and the target DensePose. The first branch processes the VAE encoding of the appearance texture map via ReferenceNet to provide low-level semantic and detailed features $c_{ref}$, which are concatenated stage-by-stage with the UNet features. The second branch extracts the source image embedding $\mathcal{I}_{emb}$, the texture map embedding $\mathcal{A}_{emb}$, and the facial embedding $\mathcal{F}_{emb}$ using a CLIP image encoder, a CLIP encoder, and a face recognition model, respectively. These are fused and injected into the UNet through the Multi-Focal Condition Aggregation (MFCA) module. The target pose, processed via DensePose through a lightweight Pose Guider module, is integrated into the UNet.

### Key Designs

1. **Multi-Focal Region Extraction and Embedding**:

    - **Function**: Decouples three pose-invariant conditional representations from the source image.
    - **Mechanism**: (1) Face region $\mathcal{F}$: Cropped using a face detector, identity features are extracted via a pre-trained face recognition model (antelopev2) and projected to a uniform dimension to obtain $\mathcal{F}_{emb}$. The face recognition model is naturally robust to cross-view and cross-expression variations. (2) Appearance region $\mathcal{A}$: The source image is warped to an SMPL texture map (200×200 to 512×512) via DensePose estimation, completely decoupling appearance from pose. The CLIP encoder extracts $\mathcal{A}_{emb}$, while the VAE encoding processed by ReferenceNet yields $c_{ref}$. (3) The source image yields $\mathcal{I}_{emb}$ via CLIP.
    - **Design Motivation**: Although the face and texture regions occupy only small parts of the image, they contain the most important perceptual information. General CLIP encoders cannot accurately capture face identity features, necessitating a specialized face recognition model.

2. **Multi-Focal Condition Aggregation (MFCA) Module**:

    - **Function**: Selectively injects different focal conditions at different stages of the UNet.
    - **Mechanism**: Different conditions are injected into the UNet's encoder $\mathcal{U}_\mathcal{E}$, middle block $\mathcal{U}_\mathcal{M}$, and decoder $\mathcal{U}_\mathcal{D}$. The global semantics $\mathcal{I}_{emb}$ (high-level semantics such as clothing category) is injected during the encoding stage. Both $\mathcal{I}_{emb}$ and $\mathcal{A}_{emb}$ are injected in the intermediate block, and the fine-grained $\mathcal{A}_{emb}$ (texture details) is injected in the decoding stage. The facial embedding $\mathcal{F}_{emb}$ is continuously injected throughout all stages. This is implemented via multiple cross-attention layers: $\phi = \sum_{i\in\{s, F_{emb}\}} \lambda_i Attn(Q, K_i, V_i)$.
    - **Design Motivation**: Unlike simply concatenating all conditions (which causes blurry attention regions for each condition), information is precisely delivered based on the functional characteristics of different UNet layers: the encoding stage handles high-level semantics, while the decoding stage reconstructs details. This strategy reduces parameter size and guides the model to prioritize the most relevant information at each stage.

3. **Face-Focused Loss and DensePose Pose Guidance**:

    - **Function**: Provides additional supervision on the facial region and 3D pose guidance.
    - **Mechanism**: In addition to the standard MSE loss, a masked loss for the facial region is introduced: $\mathcal{L}_{face} = \mathbb{E}[||(\epsilon - \epsilon_\theta) \odot m||]$, where $m$ is the face segmentation mask parsed from DensePose. DensePose also establishes a bijective mapping between the UV texture map and the target image pixels, implicitly bridging the appearance alignment of the two focal areas.
    - **Design Motivation**: The face accounts for only a small proportion of the image, meaning its contribution is overlooked in the standard MSE loss. Extra supervision intensity is required.

### Loss & Training
The total loss is $\mathcal{L}_{overall} = \mathcal{L}_{mse} + \mathcal{L}_{face}$. The Adam optimizer is used, with a learning rate of 1e-5, trained on two A100 GPUs for 60k iterations and a batch size of 12/GPU. During inference, CFG is used with scale=3.5, and $\lambda_i$ in MFCA are set to 1 and 0.5 respectively.

## Key Experimental Results

### Main Results

| Resolution | Metric | MCLD | CFLD(CVPR24) | PIDM(CVPR23) | Gain |
|--------|------|------|-------------|-------------|------|
| 256×176 | FID↓ | **6.693** | 6.804 | 6.36 | vs. CFLD -0.111 |
| 256×176 | LPIPS↓ | **0.1482** | 0.1519 | 0.1678 | vs. CFLD -0.0037 |
| 256×176 | SSIM↑ | **0.7511** | 0.7378 | 0.7312 | vs. CFLD +0.0133 |
| 256×176 | PSNR↑ | **18.84** | 18.235 | - | vs. CFLD +0.605 |
| 512×352 | FID↓ | **7.079** | 7.149 | 5.837 | vs. CFLD -0.070 |
| 512×352 | SSIM↑ | **0.7557** | 0.7478 | 0.7419 | vs. CFLD +0.0079 |

Face identity preservation metrics (256×176):

| Metric | MCLD | CFLD | PIDM | Description |
|------|------|------|------|------|
| FSref↑ | **0.301** | 0.243 | 0.270 | Face similarity with source image |
| FStgt↑ | **0.413** | 0.363 | 0.394 | Face similarity with target GT |

### Ablation Study

| Config | Condition | Aggregation | FID↓ | LPIPS↓ | SSIM↑ | PSNR↑ |
|------|------|---------|------|--------|-------|-------|
| B1 | I | - | 6.427 | 0.1629 | 0.7371 | 18.18 |
| B3 | I+A+F | concat | 6.858 | 0.1536 | 0.7340 | 18.03 |
| B5 | I+A | MFCA | 6.723 | 0.1483 | 0.7499 | 18.72 |
| **Ours** | **I+A+F** | **MFCA** | **6.693** | **0.1482** | **0.7511** | **18.84** |

### Key Findings
- **MFCA vs. Simple Concatenation**: Concatenating all conditions (B3) yields a worse FID than using only the image condition (B1) (6.858 vs 6.427), indicating that simple concatenation cannot utilize multiple conditions effectively. MFCA resolves this through stage-selective injection.
- **Contribution of Facial Condition**: B5 (without face) and Ours are close in overall metrics, but show significant differences in face preservation metrics. The face accounts for only a small portion of the image and has limited impact on overall metrics, but is crucial for human perception.
- The reconstruction FID upper bound of VAE is 7.967, which means lower FID in LDM methods does not necessarily represent better overall performance. LPIPS/SSIM/PSNR are more informative.
- The model supports flexible appearance editing: independent editing of identity, pose, and clothing can be achieved simply by modifying specific conditional regions, without additional training or masking.

## Highlights & Insights
- **The stage-selective injection strategy** is highly ingenious: utilizing the natural functional division of the UNet encoder and decoder to allocate conditional information of different granularities is more efficient and effective than injecting all conditions throughout.
- **SMPL texture map as a pose-invariant appearance representation** is a key trick: the appearance and pose are completely decoupled through DensePose to SMPL mapping, allowing different pose images of the same person to share the exact same texture condition.
- Face recognition models are more suitable for cross-pose face representation than CLIP in extracting identity features, a choice that directly determines the effectiveness of face preservation.

## Limitations & Future Work
- Highly dependent on the quality of DensePose estimation; inaccurate DensePose estimation leads to incomplete texture maps (as demonstrated in ablation B4).
- Complex fabric textures (such as fine stripes or patterns) remain difficult to reconstruct accurately by all methods.
- The significance of FID in guiding LDM methods is limited (constrained by the reconstruction FID of the VAE).
- The training parameter size is relatively large (1717M); parameter-efficient condition injection methods could be explored.
- Can be extended to video person animation tasks, leveraging temporal consistency to achieve further improvements.

## Related Work & Insights
- **vs. CFLD**: CFLD emphasizes mixed-granularity attention for semantic understanding but still uses the entire image as the condition. MCLD maintains details better through focal decoupling and stage-selective injection.
- **vs. PoCoLD**: PoCoLD establishes 3D pose correspondence but fails to address the LDM compression degradation problem. MCLD directly mitigates degradation at the conditioning level.
- **vs. InstantID**: The cross-attention design of MFCA is inspired by InstantID, but is extended into a multi-focal and stage-selective strategy.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-focal condition decoupling accompanied by stage-selective injection is an insightful design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete ablation experiments validate the contribution of each component; the newly introduced face preservation metrics are highly convincing.
- Writing Quality: ⭐⭐⭐⭐ Thorough motivational analysis, with a clear definition of the VAE compression degradation problem.
- Value: ⭐⭐⭐⭐ The idea of multi-focal condition aggregation can be generalized to other LDM degradation scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation](zoomldm_latent_diffusion_model_for_multi-scale_image_generation.md)
- [\[CVPR 2025\] Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis](exploring_sparse_moe_in_gans_for_text-conditioned_image_synthesis.md)
- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[CVPR 2025\] Learning Flow Fields in Attention for Controllable Person Image Generation](learning_flow_fields_in_attention_for_controllable_person_image_generation.md)
- [\[CVPR 2025\] Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification](modeling_thousands_of_human_annotators_for_generalizable_text-to-image_person_re.md)

</div>

<!-- RELATED:END -->
