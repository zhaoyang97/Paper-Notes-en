---
title: >-
  [Paper Note] OSDFace: One-Step Diffusion Model for Face Restoration
description: >-
  [CVPR 2025][Image Generation][Face Restoration] OSDFace proposes the first one-step diffusion model specifically designed for face restoration. By extracting rich prior information from low-quality faces using a Visual Representation Embedder (VRE), and combining this with facial identity loss and GAN guidance, it generates high-fidelity, natural, and identity-consistent face images in just a single step of inference (~0.1 second), comprehensively outperforming existing SOTA…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Face Restoration"
  - "One-Step Diffusion"
  - "Vector Quantization"
  - "Visual Prior"
  - "GAN Guidance"
date: 2026-05-08
content_hash: 611077262cc62bc0
---

# OSDFace: One-Step Diffusion Model for Face Restoration

**Conference**: CVPR 2025  
**arXiv**: [2411.17163](https://arxiv.org/abs/2411.17163)  
**Code**: [https://github.com/jkwang28/OSDFace](https://github.com/jkwang28/OSDFace)  
**Area**: Image Generation  
**Keywords**: Face Restoration, One-Step Diffusion, Vector Quantization, Visual Prior, GAN Guidance

## TL;DR

OSDFace proposes the first one-step diffusion model specifically designed for face restoration. By extracting rich prior information from low-quality faces using a Visual Representation Embedder (VRE), and combining this with facial identity loss and GAN guidance, it generates high-fidelity, natural, and identity-consistent face images in just a single step of inference (~0.1 second), comprehensively outperforming existing SOTA methods.

## Background & Motivation

**Background**: Face restoration aims to restore high-quality (HQ) faces from low-quality (LQ) images corrupted by complex degradations such as blur, noise, downsampling, and JPEG compression. Current methods mainly fall into three categories: CNN/Transformer-based methods (e.g., RestoreFormer++), GAN-based methods (which suffer from unstable training and mode collapse), and diffusion-based methods (e.g., DifFace, DiffBIR, which offer good quality but slow inference).

**Limitations of Prior Work**: First, multi-step diffusion models incur high inference costs—PGDiff requires 1000 steps/85.8 seconds, and DiffBIR requires 50 steps/9.03 seconds. Second, existing methods perform poorly in terms of "harmony"; even when basic facial features (eyes, mouth) are restored, details such as hair and complex backgrounds remain unnatural. The root cause is the insufficient incorporation of facial priors: some methods (DifFace, DiffBIR) completely ignore face priors, while others (PGDiff) use priors but limit the generation capability. Third, general one-step diffusion restoration models (e.g., OSEDiff) are not specifically designed for faces; humans are extremely sensitive to facial features and can perceive even minor inconsistencies.

**Key Challenge**: The contradiction between fast inference (one-step diffusion) and high-quality face restoration (which requires rich priors); general image restoration models cannot meet the high standards of the specialized face domain.

**Goal**: To design a one-step diffusion model specifically for faces, achieving fast inference, high-fidelity restoration, and identity consistency simultaneously.

**Key Insight**: The authors observe that VQ-based prior methods (such as CodeFormer) can effectively utilize codebooks to capture facial features, but directly generating images using a codebook lacks details. Combining VQ priors with diffusion models—by using VQ to extract rich priors as conditioning for the diffusion model—can leverage the strengths of both.

**Core Idea**: Design a Visual Representation Embedder (VRE) to directly extract visual prior prompts from low-quality images (bypassing the information loss in image $\rightarrow$ tag $\rightarrow$ embedding), and combine it with facial identity loss and GAN distribution alignment to achieve high-quality, one-step face restoration.

## Method

### Overall Architecture

The training of OSDFace consists of two stages. First stage: Train the Visual Representation Embedder (VRE). Through self-reconstruction training of both HQ and LQ VQVAEs, a visual token dictionary is established, and the feature categories of the two domains are aligned. Second stage: Integrate the pre-trained VRE into Stable Diffusion. The LQ image is mapped to the latent space via the VAE encoder, the VRE extracts visual prompt embeddings, the UNet (fine-tuned only via LoRA) predicts the one-step noise, and the VAE decoder reconstructs the HQ face. The entire inference process takes only 0.1 seconds.

### Key Designs

1. **Visual Representation Embedder (VRE)**:

    - **Function**: Extracts rich visual prior prompts from low-quality faces.
    - **Mechanism**: VRE consists of two parts. **Visual Tokenizer**: LQ VAE encoder $E_L$ + VQ matching function $\mathcal{M}$, which maps LQ faces to category tokens $\mathbf{Q}_L = \mathcal{M}(E_L(I_L))$ in a learnable LQ dictionary $\mathbb{C}_L = \{c_q \in \mathbb{R}^d\}_{q=1}^N$. **VQ Embedder**: Uses tokens as indices to look up the corresponding embedding vectors in the dictionary, $z_k = \text{dict}(q)$, with a time complexity of $\mathcal{O}(1)$. Unlike image-to-tag methods, VRE directly tokenizes faces into visual embeddings, avoiding the information loss from image $\rightarrow$ tag $\rightarrow$ embedding.
    - **Design Motivation**: Attention map visualizations show that VRE focuses on both facial and non-facial features (hair, background, etc.), capturing richer information than text tags.

2. **Feature Association Training Strategy**:

    - **Function**: Aligns the VQ dictionary categories of the HQ and LQ domains.
    - **Mechanism**: Construct HQ and LQ VQ dictionaries, and train VQVAEs separately for self-reconstruction. Then, inspired by CLIP, construct a similarity matrix $M_{\text{assoc}}$ of HQ and LQ encoded features, and use a cross-entropy loss to enhance diagonal correlation, guiding the attention of the LQ encoder to align with the HQ encoder. The association loss is $\mathcal{L}_{\text{assoc}} = (\mathcal{L}_{\text{CE}}^H + \mathcal{L}_{\text{CE}}^L) / 2$.
    - **Design Motivation**: The LQ encoder may focus on meaningless categories due to severe degradation; cross-domain alignment ensures that LQ tokens correspond to meaningful HQ semantics.

3. **Facial Identity Loss and GAN Guidance**:

    - **Function**: Ensures identity consistency and overall realism of the restored face.
    - **Mechanism**: **Facial Identity Loss**: Uses a pre-trained ArcFace model to extract identity embeddings from the generated face and ground truth (GT), compute the cosine similarity loss $\mathcal{L}_{\text{ID}} = 1 - \cos(\mathcal{F}(I_H), \mathcal{F}(\hat{I}_H))$. **Perceptual Loss**: Uses edge-aware DISTS (EA-DISTS), which adds Sobel edge-processed DISTS to standard DISTS to enhance texture and edge detail restoration. **GAN Loss**: Uses a discriminator to align the generated distribution with the real distribution in the diffusion latent space, providing more flexible training signals than distillation. The total loss is $\mathcal{L}_{\text{gen}} = \lambda_{\text{dis}} \mathcal{L}_{\mathcal{G}} + \lambda_{\text{ID}} \mathcal{L}_{\text{ID}} + \lambda_{\text{per}} \mathcal{L}_{\text{EA-DISTS}} + \text{MSE}$.
    - **Design Motivation**: Humans are extremely sensitive to faces; even minor inconsistencies are easily detected. Multi-aspect losses are required to ensure harmony: identity loss governs identity consistency, GAN loss governs distribution alignment, and perceptual loss governs texture details.

### Loss & Training

First-stage VRE training loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_1 + \lambda_{\text{per}} \mathcal{L}_{\text{per}} + \lambda_{\text{dis}} \mathcal{L}_{\text{dis}} + \mathcal{L}_{\text{VQ}} + \lambda_{\text{assoc}} \mathcal{L}_{\text{assoc}}$, where $\lambda_{\text{assoc}}$ is initially 0 and later set to 1. In the second stage, the VAE encoder/decoder and VRE are frozen, UNet is fine-tuned only via LoRA, and the generator and discriminator are trained alternately. During inference, a predefined fixed timestep $T_L$ is used, and the LQ latent vector is directly used as the UNet input (not pure Gaussian noise).

## Key Experimental Results

### Main Results

| Method | Steps | LPIPS↓ | DISTS↓ | MUSIQ↑ | NIQE↓ | FID(HQ)↓ |
|------|------|--------|--------|--------|-------|---------|
| CodeFormer | - | 0.3412 | 0.2151 | 75.94 | 4.52 | 26.86 |
| DifFace (250 steps) | 250 | 0.3469 | 0.2126 | 66.75 | 4.64 | 22.24 |
| DiffBIR (50 steps) | 50 | 0.3740 | 0.2340 | 75.64 | 6.28 | 32.51 |
| OSEDiff* (1 step) | 1 | 0.3496 | 0.2200 | 69.98 | 5.33 | 37.13 |
| **OSDFace (1 step)** | 1 | **0.3365** | **0.1773** | **75.64** | **3.88** | **17.06** |

### Ablation Study

Figure 2 in the paper shows a comprehensive performance radar chart. OSDFace achieves the best performance across most of the 8 metrics: LPIPS, DISTS, MUSIQ, NIQE, Deg, LMD, FID(FFHQ), and FID(HQ), notably leading by a wide margin in DISTS (texture quality) and NIQE (naturalness).

### Key Findings

- OSDFace outperforms all multi-step diffusion and non-diffusion methods with only 0.1 seconds of inference, requiring only 2.132T MACs of computation.
- Compared to the general one-step diffusion model OSEDiff, DISTS decreases from 0.2200 to 0.1773 (-19.4%), and FID(HQ) drops from 37.13 to 17.06 (-54%), proving the necessity of face-specific designs.
- Excellent zero-shot generalization performance on real-world datasets (WIDER, WebPhoto, LFW).
- Attention map visualizations confirm that VRE can simultaneously focus on facial features and non-facial regions like backgrounds and hair.

## Highlights & Insights

- The design of VRE is highly elegant: it leverages a VQ dictionary to capture category-level priors and directly generates prompt embeddings from visual tokens, avoiding the information bottleneck of image-to-tag.
- The cross-domain feature association strategy enables the LQ dictionary to find meaningful semantic mappings even when the input is heavily degraded.
- It does not rely on distillation (no multi-step teacher model required), but instead uses GAN guidance to achieve distribution alignment, making training more flexible.
- Achieving 512×512 face restoration in 0.1 seconds holds great practical deployment value.

## Limitations & Future Work

- The current model is designed specifically for faces; generalizing to other image types would require additional adaptation.
- The VQ dictionary size is fixed, which might lack coverage when facing extremely diverse facial styles.
- One-step diffusion has lower generation diversity than multi-step diffusion. This has little impact on deterministic tasks like face restoration, but could be restrictive in scenarios requiring high diversity.
- Performance under extreme degradation (e.g., extremely low resolution or severe occlusions) remains to be further verified.

## Related Work & Insights

- Unlike CodeFormer (where VQ priors directly generate images) and PGDiff (where VQ priors act as diffusion guidance targets), OSDFace uses VQ priors as conditional prompts for the diffusion model, striking a better balance between flexibility and quality.
- The introduction of facial identity loss underscores the importance of identity preservation in face restoration tasks.
- The LQ dictionary representation in VRE may provide insights for other degraded-image understanding tasks (e.g., medical image enhancement).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Clever VRE design; the combination of VQ prior and one-step diffusion is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation with 8 metrics, comparisons against multiple baselines, and real-world zero-shot testing.
- **Writing Quality**: ⭐⭐⭐⭐ — Abundant diagrams, clear description of methods, and convincing visualization analyses.
- **Value**: ⭐⭐⭐⭐⭐ — High-quality face restoration in 0.1 seconds, offering direct industrial application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SVFR: A Unified Framework for Generalized Video Face Restoration](svfr_a_unified_framework_for_generalized_video_face_restoration.md)
- [\[CVPR 2025\] SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion](swiftedit_lightning_fast_text-guided_image_editing_via_one-step_diffusion.md)
- [\[CVPR 2025\] GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration](gendeg_diffusion-based_degradation_synthesis_for_generalizable_all-in-one_image_.md)
- [\[ICCV 2025\] Unlocking the Potential of Diffusion Priors in Blind Face Restoration](../../ICCV2025/image_generation/unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)
- [\[ICCV 2025\] MoFRR: Mixture of Diffusion Models for Face Retouching Restoration](../../ICCV2025/image_generation/mofrr_mixture_of_diffusion_models_for_face_retouching_restoration.md)

</div>

<!-- RELATED:END -->
