---
title: >-
  [Paper Note] BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training
description: >-
  [CVPR 2025][Image Generation][Virtual Try-On] Proposes BooW-VTON, which trains a virtual try-on diffusion model requiring no human parsing masks through high-quality pseudo-data construction, in-the-wild data augmentation, and try-on localization loss. It comprehensively outperforms existing methods across multiple benchmarks including VITON-HD, StreetVTON, and WildVTON.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Virtual Try-On"
  - "Mask-Free Inference"
  - "Pseudo Data Training"
  - "Attention Regularization"
  - "In-the-Wild Scenes"
date: 2026-05-08
content_hash: f1c0f37368adc9ce
---

# BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training

**Conference**: CVPR 2025  
**arXiv**: [2408.06047](https://arxiv.org/abs/2408.06047)  
**Code**: None (Not mentioned)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Virtual Try-On, Mask-Free Inference, Pseudo Data Training, Attention Regularization, In-the-Wild Scenes

## TL;DR
Proposes BooW-VTON, which trains a virtual try-on diffusion model requiring no human parsing masks through high-quality pseudo-data construction, in-the-wild data augmentation, and try-on localization loss. It comprehensively outperforms existing methods across multiple benchmarks including VITON-HD, StreetVTON, and WildVTON.

## Background & Motivation

**Background**: Image-based virtual try-on (VTON) aims to naturally render a target garment onto a person's image. Existing mainstream methods (e.g., IDM-VTON, StableVITON) adopt a mask-based inpainting paradigm—employing a human parser to acquire the mask of the try-on region, masking this area, and then using a diffusion model to repaint it. This yields decent results in simple e-commerce scenarios.

**Limitations of Prior Work**: Mask-based methods exhibit three fundamental issues. (1) **Loss of Spatial Information**: Masking the try-on region destroys spatial cues such as the depth, lighting, and texture of the original image. (2) **Foreground-Background Disconnection**: The mask disrupts the continuity between the foreground and background. (3) **Parser Dependency**: These methods rely on an external human parser to provide pose/parsing information, which becomes unreliable in complex in-the-wild scenarios (intricate poses, occlusions). In complex wild scenes, these deficiencies lead to prominent artifacts—such as lost accessories, altered skin textures, and scene inconsistencies.

**Key Challenge**: While mask-free try-on can fully leverage the spatial and lighting information of the original image, training such a model requires triplets of {try-on image, garment, original image}, which do not exist intrinsically. Direct distillation from a masked teacher model to a mask-free student (such as PFDM) inherits the teacher's flaws, failing to generalize to complex scenes.

**Goal**: To train a high-quality virtual try-on model that does not require mask inputs, especially maintaining garment rendering accuracy and non-try-on area fidelity in complex in-the-wild scenes.

**Key Insight**: Instead of distillation, the proposed method leverages a masked model to generate high-quality pseudo-data in simple scenarios as training signals, and then transfers this capability to complex scenarios using data augmentation and attention regularization.

**Core Idea**: To train a mask-free try-on diffusion model using high-quality pseudo try-on data generated via a refined masked model, combined with synthesized in-the-wild foreground/background augmentation and try-on localization losses.

## Method

### Overall Architecture
Based on SDXL as the foundation model, IP-Adapter and Reference Net are employed as garment encoders. The input consists of the original person image $P'$ (a pseudo-image of the tried-on state) and the garment image $G$. The latent of $P'$ is concatenated with the noisy target latent along the channel dimension and fed into the try-on U-Net. The training pipeline consists of three steps: (1) using a masked model in simple scenarios to perform two-stage inference to generate high-quality pseudo data; (2) applying in-the-wild foreground and background augmentation to the pseudo data; and (3) employing a try-on localization loss to constrain attention to the try-on region. Masks or human parsers are completely unnecessary during inference.

### Key Designs

1. **Two-Stage Refined Pseudo Data Generation**

    - **Function**: To obtain high-quality, mask-free training triplets from the outputs of a masked model.
    - **Mechanism**: IDM-VTON is utilized as the masked model. In the first stage, try-on is performed using a loose coarse mask $M_{coa}$ to obtain an intermediate result $P_{mid}$. The garment region $M_{mid}$ is extracted from $P_{mid}$ and unioned with the original garment region $M_P$ to establish a more precise mask $M$. In the second stage, inference is executed again with this precise mask, yielding high-quality pseudo data $P'$ that preserves more non-try-on content. This constructs the $\{P', G, P\}$ triplet, where $P'$ serves as the tried-on image (conditional input) and $P$ serves as the original image (supervision target).
    - **Design Motivation**: Direct generation with coarse masks leads to mask-boundary artifacts. The two-stage refinement produces near-perfect pseudo data in simple scenes, providing high-quality supervision signals for the mask-free model.

2. **In-the-Wild Data Augmentation**

    - **Function**: To extend simple e-commerce pseudo data to satisfy the demands of complex in-the-wild training.
    - **Mechanism**: Synthetic backgrounds and foregrounds are added simultaneously to both $\{P', P\}$. **Background Generation**: A transparent person image and a T2I model are used to inpaint blank regions, yielding a mixed person-background image, followed by inpainting the person region to obtain a clean background. **Foreground Generation**: GPT-4o is used to generate object prompts, and Layerdiffusion is used to generate transparent foreground images. During training, foregrounds and backgrounds are randomly selected and layered from bottom to top as B-P/P'-F, along with random translation and scaling transformations. Foregrounds are restricted to occluding only non-try-on regions (modifying the try-on mask $M^{Aug}$ to exclude foreground regions).
    - **Design Motivation**: Masked models are typically trained only on simple e-commerce datasets; hence, directly generated pseudo data lacks complex foregrounds and backgrounds. Synthetic augmentation trains the model to maintain precise try-on capabilities and overall fidelity despite complex occlusions and background interference.

3. **Try-On Localization Loss**

    - **Function**: To constrain attention layers to render garment features solely within the try-on region, preventing alterations in non-try-on areas.
    - **Mechanism**: Within the attention layers, the person latent code acts as the Query, and the garment tokens serve as the Key/Value. The attention score $A_k$ indicates the intensity of garment features propagating into the 2D person space. The try-on localization loss minimizes attention scores in non-try-on regions: $\mathcal{L}_{ar} = \frac{1}{n}\sum_{k=1}^{n} \text{mean}(A_k(1-M^{Aug}))$. This is applied to blocks 5–64 out of the 70 SDXL attention blocks (with image token length of 32×24), operating on both cross-attention and self-attention layers. Mask data is used only during training, leaving the inference process completely mask-free.
    - **Design Motivation**: Unconstrained attention routinely diffuses across the entire image, contaminating non-try-on regions (e.g., accessories, skin, and backgrounds) with garment features. Explicitly regulating attention focus achieves a "knowing where to edit" effect.

### Loss & Training
The total loss is defined as $\mathcal{L} = \mathcal{L}_{LDM} + \lambda_{ar}\mathcal{L}_{ar}$, where $\lambda_{ar}=1$. The weights are initialized based on IDM-VTON, and only the try-on U-Net is unfrozen. Training is conducted on 16 × H100 GPUs for approximately 12 hours with a batch size of 32, 12,000 steps, a learning rate of 5e-6, and the Adam optimizer. Inference utilizes 30-step DDIM.

## Key Experimental Results

### Main Results

| Method | VITON-HD LPIPS↓ | SSIM↑ | StreetVTON FID_u↓ | WildVTON FID_u↓ |
|------|----------------|-------|-------------------|-----------------|
| DCI-VTON | 0.1800 | 0.8545 | 20.95 | 35.66 |
| StableVITON | 0.1479 | 0.8519 | 23.15 | 42.32 |
| IDM-VTON | 0.1223 | 0.8547 | 23.62 | 38.77 |
| **BooW-VTON** | **0.1080** | **0.8618** | **20.50** | **32.53** |

DressCode-Upper: LPIPS **0.0615** vs IDM-VTON 0.0761

### Ablation Study

| Configuration | VITON-HD LPIPS↓ | StreetVTON FID_u↓ | WildVTON FID_u↓ |
|------|----------------|-------------------|-----------------|
| Base mask-free | 0.1206 | 28.81 | 57.52 |
| + High-Quality Pseudo Data | 0.1101 | 27.26 | 56.14 |
| + In-the-Wild Augmentation | 0.1173 | 21.70 | 35.62 |
| + Try-on Localization Loss (Full) | **0.1080** | **20.50** | **32.53** |

### Key Findings
- In-the-wild data augmentation yields the most significant contribution: WildVTON FID drops sharply from 56.14 to 35.62, and StreetVTON from 27.26 to 21.70, demonstrating that synthesizing foregrounds and backgrounds successfully teaches the model to handle complex scenes.
- Try-on localization loss further improves performance in wild scenarios: WildVTON FID decreases from 35.62 to 32.53, indicating that attention constraints prevent non-try-on areas from being incorrectly modified.
- On simple e-commerce benchmarks (VITON-HD), the model surpasses IDM-VTON in LPIPS, SSIM, and PSNR, proving that the mask-free paradigm not only maintains but actually enhances precision (by preserving intact spatial information).
- The model generalizes to anime-style try-on without prior training on anime data, demonstrating the cross-domain capabilities of the pre-trained model.

## Highlights & Insights
- **The fundamental analysis of mask-based methods' deficiencies** is highly compelling: demonstrating spatial information loss via depth map comparisons provides solid theoretical motivation for the mask-free paradigm.
- **The two-stage training strategy combining pseudo-data and augmentation** cleverly bypasses the absence of mask-free training data. Generating accurate pseudo-data in simple scenes first and then scaling to complex scenes via augmentation is a transferable strategy applicable to other image-editing tasks lacking paired data.
- **Attention regularization** serves as a lightweight yet effective mechanism to control the editing region without introducing any inference overhead (masks are used exclusively during training).

## Limitations & Future Work
- When trying on a t-shirt over an original image displaying a dress, the lack of reference information for the lower body leads to uncontrolled, random generations.
- Similarly, during lower-garment try-on, the upper body becomes uncontrolled, and the model cannot coordinate top-bottom outfit pairings.
- The quality of pseudo-data remains limited by IDM-VTON's capabilities; pseudo-data may exhibit flaws under extreme poses or severe occlusions.
- Training data is sourced solely from e-commerce datasets (VITON-HD/DressCode); wild generalization heavily relies on the quality of the augmentation.

## Related Work & Insights
- **vs IDM-VTON**: IDM-VTON is the SOTA among mask-based approaches. BooW-VTON initializes from its weights but eliminates mask dependency, outperforming it across all metrics, with a substantial advantage in wild scenarios (WildVTON FID 32.53 vs 38.77).
- **vs PFDM**: PFDM also pursues mask-free try-on but via distillation, thereby inheriting the flaws of the teacher mask-based model. BooW-VTON avoids this flaw propagation through refined pseudo-data.
- **vs TPD**: TPD employs a two-stage process to improve mask precision but remains tied to the mask-based paradigm, leaving the fundamental issue of spatial information loss unresolved.

## Rating
- Novelty: ⭐⭐⭐⭐ While the mask-free try-on concept is not entirely novel (e.g., PFDM), the combination of pseudo-data construction, augmentation, and localization loss is highly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely solid evaluation across four datasets, comprehensive ablation studies, multiple baselines, and both qualitative and quantitative analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed method descriptions, and rich figures and tables.
- Value: ⭐⭐⭐⭐ Removing the parser dependency for try-on has immediate practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] OmniVTON: Training-Free Universal Virtual Try-On](../../ICCV2025/image_generation/omnivton_training-free_universal_virtual_try-on.md)
- [\[CVPR 2026\] PG-VTON: Single-Pass Training-Free Virtual Try-On via Patch-Guided Reference Alignment](../../CVPR2026/image_generation/pg-vton_single-pass_training-free_virtual_try-on_via_patch-guided_reference_alig.md)
- [\[CVPR 2025\] Shining Yourself: High-Fidelity Ornaments Virtual Try-on with Diffusion Model](shining_yourself_high-fidelity_ornaments_virtual_try-on_with_diffusion_model.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)
- [\[ECCV 2024\] WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models](../../ECCV2024/image_generation/wildvidfit_video_virtual_try-on_in_the_wild_via_image-based_controlled_diffusion.md)

</div>

<!-- RELATED:END -->
