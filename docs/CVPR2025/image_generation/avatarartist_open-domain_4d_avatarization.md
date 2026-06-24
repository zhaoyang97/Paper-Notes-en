---
title: >-
  [Paper Note] AvatarArtist: Open-Domain 4D Avatarization
description: >-
  [CVPR 2025][Image Generation][4D Avatar Generation] AvatarArtist is proposed, which cooperatively constructs a multi-domain image-triplane dataset using GAN and diffusion models to train a DiT for generating parametric triplanes, combined with a motion-aware cross-domain renderer to achieve drivable 4D avatars from a single portrait of arbitrary styles.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "4D Avatar Generation"
  - "Parametric Triplane"
  - "GAN-Diffusion Synergy"
  - "Cross-Domain Rendering"
  - "Expression-Driven"
date: 2026-05-08
content_hash: 9ba13c7c064aab73
---

# AvatarArtist: Open-Domain 4D Avatarization

**Conference**: CVPR 2025  
**arXiv**: [2503.19906](https://arxiv.org/abs/2503.19906)  
**Code**: [https://kumapowerliu.github.io/AvatarArtist](https://kumapowerliu.github.io/AvatarArtist) (Code/Data/Models upcoming)  
**Area**: Diffusion Models / 3D Vision  
**Keywords**: 4D Avatar Generation, Parametric Triplane, GAN-Diffusion Synergy, Cross-Domain Rendering, Expression-Driven

## TL;DR
AvatarArtist is proposed, which cooperatively constructs a multi-domain image-triplane dataset using GAN and diffusion models to train a DiT for generating parametric triplanes, combined with a motion-aware cross-domain renderer to achieve drivable 4D avatars from a single portrait of arbitrary styles.

## Background & Motivation

**Background**: Single-image driven avatarization follows either 2D or 4D pipelines. 2D methods (e.g., LivePortrait) utilize diffusion model priors to handle multi-style inputs but lack 3D structural understanding, leading to geometric distortions during large-angle rotations. 4D methods (e.g., Portrait4D, InvertAvatar) guarantee geometric consistency via neural rendering like NeRF/3DGS, but are limited to the human face domain due to the scarcity of 4D training data.

**Limitations of Prior Work**: The core bottleneck of 4D methods is the lack of multi-domain 4D training data. Constructing image-to-4D representation pairs requires 4D captured data with multi-view and multi-expression variations, which is almost impossible to acquire for non-realistic domains such as cartoons, game characters, and sculptures. Existing 4D GANs (e.g., Next3D) can learn 4D representations in an unsupervised manner from 2D images, but their inherent mode collapse prevents them from covering diverse visual domains.

**Key Challenge**: Training a generic model requires large-scale, multi-domain 4D data, yet 4D data acquisition is extremely difficult; GANs can generate 4D data but only cover single domains; diffusion models can cover multiple domains but cannot directly generate 4D representations.

**Goal**: How to build a cross-domain 4D avatar generation system capable of creating drivable 4D avatars from a single portrait of arbitrary style (realistic, cartoon, game character, sculpture, etc.).

**Key Insight**: GANs and diffusion models have complementary strengths—diffusion models excel at cross-domain 2D image generation, while GANs excel at unsupervised 2D-to-4D transformation. Their synergy can address the data issue: first, use a diffusion model to generate multi-domain 2D data (maintaining consistent expressions/poses to reuse 3DMM meshes), and then employ a GAN to convert these 2D data into 4D representation pairs.

**Core Idea**: Leverage a diffusion model to expand GAN to 28 non-realistic domains to synthesize 560k image-triplane pairs, and then train a DiT + cross-domain renderer to achieve open-domain 4D avatar generation.

## Method

### Overall Architecture
The entire system consists of three stages. **Stage 1 (Data Construction)**: SDEdit + ControlNet are used to transfer FFHQ realistic face images to 28 non-realistic domains (anime, Lego, etc.) while preserving matching expressions/poses to reuse 3DMM meshes. Then, a Next3D GAN is fine-tuned for each domain, randomly sampling to generate 560k image-parametric triplane pairs. **Stage 2 (4D Generation)**: A Triplane VAE is trained to compress the triplane representation, followed by training an image-conditioned DiT to generate the latent code of the parametric triplane from a single image. **Stage 3 (Rendering)**: A ViT-based motion-aware cross-domain renderer is designed to fuse source image features and the generated triplane, driving expression changes via implicit motion embeddings, and finally outputting the image through volume rendering.

### Key Designs

1. **GAN-Diffusion Collaborative Data Construction Pipeline**

    - **Function**: Generates 560k image-parametric triplane training pairs covering 28 visual domains.
    - **Mechanism**: First, FFHQ realistic face images are perturbed with noise and then denoised using StableDiffusion (with domain-specific prompts) alongside landmark ControlNet to control expression, thereby transferring the images to target domains (e.g., anime, sculpture). Since the output images maintain the original poses and expressions, the 3DMM mesh labels extracted from the realistic domain can be directly reused. 6,000 images are generated per domain to fine-tune an independent Next3D GAN, which is then sampled for 20k image-triplane pairs each (totaling 560k pairs across 28 domains). SDEdit maintains the global structure and ControlNet preserves expression accuracy; both are indispensable.
    - **Design Motivation**: It is impossible to accurately extract 3DMM meshes directly in non-realistic domains, making direct GAN training unfeasible. By transferring from the realistic domain and reusing the meshes, this bottleneck is elegantly bypassed.

2. **Image-Conditioned Diffusion Transformer**

    - **Function**: Generates the latent representation of parametric triplanes from a single portrait.
    - **Mechanism**: A VAE is first trained to compress the triplanes from $\in \mathbb{R}^{256 \times 256 \times 4 \times 32}$ to $\mathbb{R}^{64 \times 64 \times 4 \times 8}$. Then, DiT-XL/2 (with 28 DiT blocks) is trained. The input is the flattened sequence of the noisy triplane latent. Conditional images are processed via CLIP to extract semantic embeddings (injected via cross-attention) and DINO to extract detailed tokens (concatenated with latents for self-attention). Training uses the IDDPM objective to predict noise and variance, with a 10% probability of dropping conditional images to support classifier-free guidance. Inference employs a 19-step DPMSolver.
    - **Design Motivation**: Dual-condition injection (CLIP semantics + DINO details) ensures that the generated triplanes are both semantically correct and detail-preserving. VAE compression reduces the computational overhead of the DiT.

3. **Motion-Aware Cross-Domain Renderer**

    - **Function**: Renders high-quality target expression/pose images from the generated parametric triplanes and source images while preserving identity.
    - **Mechanism**: An encoder $E_I$ extracts source image features, which are fed into a ViT. In the ViT's self-attention, the parametric triplanes generated by DiT are injected to neutralize facial expressions and normalize poses (eliminating the expression information in the source image). Then, an implicit motion embedding is injected via cross-attention to impart the target expression. The ViT output is decoded and fused with rasterized triplanes, followed by volume rendering and super-resolution to generate the output. The implicit motion embedding does not contain spatial information, avoiding identity leakage.
    - **Design Motivation**: The original CNN renderer in Next3D performs poorly in cross-domain scenarios, leading to severe identity leakage and expression mismatches. The attention mechanism in ViT is better suited for cross-domain feature fusion, and implicit motion embedding avoids artifacts caused by inaccurate meshes.

### Loss & Training
The VAE training utilizes an $\mathcal{L}_1$ loss for triplane reconstruction, and $\mathcal{L}_1$ and LPIPS losses for rendered images (adversarial losses are avoided as they cause training instability). DiT uses the IDDPM loss to predict noise and variance. Renderer training references the loss combinations of methods like Portrait4D, trained on 12 million multi-domain images.

## Key Experimental Results

### Main Results

| Method | Self-Drive LPIPS↓ | Self-Drive ID↑ | Cross-Domain AKD↓ | Cross-Domain APD↓ | Cross-Domain FID↓ | Cross-Domain CLIP↑ |
|------|-------------|----------|---------|---------|---------|----------|
| LivePortrait (2D) | 0.27 | 0.65 | 4.92 | 139.35 | 100.3 | 0.91 |
| XPortrait (2D) | 0.31 | 0.63 | 10.67 | 237.4 | 78.6 | 0.89 |
| Portrait4Dv2 (4D) | 0.29 | 0.58 | 7.13 | 63.3 | 140.5 | 0.75 |
| InvertAvatar (4D) | 0.42 | 0.32 | 20.78 | 134.9 | 194.7 | 0.64 |
| **AvatarArtist (Ours)** | **0.26** | **0.69** | **2.58** | **52.3** | 89.3 | 0.84 |

### Ablation Study

| Configuration | FID↓ | CLIP↑ | AKD↓ | APD↓ |
|------|------|-------|------|------|
| Next3D CNN Renderer | 130.72 | 0.73 | 5.89 | 42.93 |
| **Full Model (ViT Renderer)** | **68.69** | **0.86** | **2.56** | **40.89** |

### Key Findings
- In cross-domain driving tasks (non-realistic source $\rightarrow$ realistic target), the proposed method outperforms all other methods by a wide margin in motion accuracy metrics (AKD 2.58, APD 52.3), highlighting the advantages of 4D representations in pose/expression transfer.
- In self-driving tasks, the proposed method achieves the best performance in both LPIPS (0.26) and ID (0.69), demonstrating strong identity-preservation capabilities.
- Replacing the ViT renderer with a CNN causes the FID to surge from 68.69 to 130.72 and lowers the CLIP score from 0.86 to 0.73, indicating that the cross-domain renderer is a critical module.
- Both SDEdit and ControlNet are essential in data construction: removing SDEdit leads to large expression deviations, while removing ControlNet maintains pose but results in inconsistent expressions.

## Highlights & Insights
- **Complementary advantages of GAN and Diffusion models**: The design is highly elegant—diffusion models provide multi-domain capability while GANs provide unsupervised 2D-to-4D conversion, cooperatively solving the fundamental issue of nonexistent multi-domain 4D data. This collaborative strategy is transferable to other tasks requiring scarce 3D/4D data.
- **Reuse of 3DMM meshes in cross-domain transfer**: Avoiding mesh extraction difficulties in non-realistic domains by utilizing domain transfer with consistent expressions/poses serves as a highly practical engineering technique.
- **Implicit motion embedding replacing explicit mesh driving**: This avoids artifacts caused by inaccurate mesh extraction and is crucial for cross-domain generalization.

## Limitations & Future Work
- Self-driving FID (52.62) is slightly worse than LivePortrait (46.49), indicating room for improvement in 2D texture quality.
- The cross-domain CLIP ID score (0.84) is lower than that of the 2D method LivePortrait (0.91), showing a persistent gap in cross-domain identity fidelity.
- The data pipeline is computationally expensive, requiring the training of 28 independent domain-specific GANs (300k iterations each).
- The method heavily relies on FaceVerse 3DMM mesh extraction, which might fail on non-standard facial structures (e.g., profile views, occlusions).
- The renderer requires training on 12 million images, which demands significant resources.

## Related Work & Insights
- **vs LivePortrait**: LivePortrait is a 2D method that preserves identity well but suffers from geometric distortions under large rotation angles. AvatarArtist ensures geometric consistency using 4D representations at the cost of slightly lower texture quality.
- **vs Portrait4Dv2**: Both are 4D methods, but Portrait4Dv2 generalizes poorly to non-realistic domains (CLIP 0.75 vs 0.84). The core difference is that AvatarArtist achieves open-domain generalization through multi-domain data construction.
- **vs Rodin**: Rodin generates static avatars, whereas this work focuses on drivable 4D avatars, which involves higher complexity. However, Rodin's image-to-3D data construction approach inspired the design of the data pipeline in this paper.

## Rating
- Novelty: ⭐⭐⭐⭐ The GAN-diffusion collaborative data construction is a major highlight, though the individual component technologies are relatively mature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Direct quantitative and qualitative comparisons are exhaustive, and each component is verified via ablation studies, though user studies and evaluations on more diverse domains are still needed.
- Writing Quality: ⭐⭐⭐⭐ The overall workflow is clear with abundant figures, and the data pipeline is described in detail.
- Value: ⭐⭐⭐⭐ This is the first work to achieve open-domain 4D avatar generation, though the high training costs limit its practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RAG-IGBench: Innovative Evaluation for RAG-based Interleaved Generation in Open-domain Question Answering](../../NeurIPS2025/image_generation/rag-igbench_innovative_evaluation_for_rag-based_interleaved_generation_in_open-d.md)
- [\[CVPR 2025\] DNF: Unconditional 4D Generation with Dictionary-Based Neural Fields](dnf_unconditional_4d_generation_with_dictionary-based_neural_fields.md)
- [\[CVPR 2025\] OpenSDI: Spotting Diffusion-Generated Images in the Open World](opensdi_spotting_diffusion-generated_images_in_the_open_world.md)
- [\[CVPR 2025\] DoraCycle: Domain-Oriented Adaptation of Unified Generative Model in Multimodal Cycles](doracycle_domain-oriented_adaptation_of_unified_generative_model_in_multimodal_c.md)
- [\[CVPR 2025\] FoundHand: Large-Scale Domain-Specific Learning for Controllable Hand Image Generation](foundhand_large-scale_domain-specific_learning_for_controllable_hand_image_gener.md)

</div>

<!-- RELATED:END -->
