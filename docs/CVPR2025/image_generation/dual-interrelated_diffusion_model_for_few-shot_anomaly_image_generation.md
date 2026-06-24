---
title: >-
  [Paper Note] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation
description: >-
  [Image Generation] DualAnoDiff is proposed to simultaneously generate high-quality anomaly image-mask pairs via a dual-interrelated diffusion model (a global branch generating the entire anomaly image + an anomaly branch generating the local anomalous part). A background compensation module is introduced to maintain the consistency of the background and object shape, which significantly improves downstream performance in anomaly detection, localization, and classification.
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 9fc9a7efd879a5e7
---

# DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation

## TL;DR

DualAnoDiff is proposed to simultaneously generate high-quality anomaly image-mask pairs via a dual-interrelated diffusion model (a global branch generating the entire anomaly image + an anomaly branch generating the local anomalous part). A background compensation module is introduced to maintain the consistency of the background and object shape, which significantly improves downstream performance in anomaly detection, localization, and classification.

## Background & Motivation

Industrial anomaly detection faces a core challenge: **extreme scarcity of anomalous data**. Limitations of prior work:
- **Model-free methods** (e.g., DRAEM, Cut-Paste): Generate anomalies by randomly cropping and pasting textures, resulting in unrealistic appearances.
- **GAN-based methods** (e.g., SDGAN, DefectGAN): Require large amounts of anomaly data for training and cannot generate anomaly masks.
- **DFMGAN**: Pre-trains StyleGAN2 on normal data and then transfers it to the anomalous domain, but the generated anomalies are unrealistic and have poor mask alignment.
- **AnomalyDiffusion**: Individually learns the anomaly appearance and mask location based on textual inversion, leading to two issues: (1) unnatural integration between the anomaly and the original image; (2) masks appearing in background regions.

Key Insight: **Simultaneously generating the entire anomaly image and the corresponding anomalous part** inherently ensures high alignment between the mask and the anomaly, rather than learning them separately and then combining them.

## Method

### Overall Architecture

DualAnoDiff constructs dual diffusion branches based on Stable Diffusion: a global branch $SD$ generates the complete anomaly image $I$, and an anomaly branch $SD^*$ generates the image containing only the anomalous region $I_a = I \times M_a$. The two branches share time steps, exchange information via the Self-Attention Interaction Module (SAIM), and use nested prompts to guide their respective generation processes. A Background Compensation Module (BCM) injects background information to maintain shape and background stability.

### Key Designs

#### 1. Dual-Interrelated Diffusion + Nested Prompts

- **Function**: Simultaneously generates the anomaly image $I$ and the corresponding anomalous part $I_a$, ensuring high alignment between the mask and the anomaly.
- **Mechanism**: Freeze the SD weights and use two sets of LoRA to fine-tune the global branch and the anomaly branch, respectively. The two branches utilize nested prompts $p$: "a **x** with **y**" (global) and $p'$: "**y**" (anomaly), where x=vfx and y=sks are rare tokens. They undergo noise addition and denoising at the same time steps, sharing information via SAIM.
- **Design Motivation**: Resolve the anomaly-mask alignment problem through parallel generation instead of separate learning. Nested prompts reflect inclusion relationships, enabling the model to correctly separate object attributes and anomaly attributes. Using rare tokens (vfx, sks) allows for easier fitting compared to high-frequency words.

#### 2. Self-Attention Interaction Module (SAIM)

- **Function**: Synchronizes information between the two branches during the denoising process to maintain consistency between the generated overall image and the local anomaly.
- **Mechanism**: After each attention block, the intermediate features of the two branches, $\varphi_i(z)$ and $\varphi_i(z')$, are concatenated and reshaped to perform a shared self-attention computation, which is then split back into the respective branches with residual connections. Shared self-attention shares positional and detail information, while shared cross-attention shares semantic information.
- **Design Motivation**: Directly generating two paths of images independently leads to inconsistencies in anomaly locations and shapes. SAIM achieves implicit spatial and semantic alignment via shared attention, eliminating the need for extra alignment losses.

#### 3. Background Compensation Module (BCM)

- **Function**: Injects background information and object mask shapes to prevent object deformation and background confusion under few-shot training.
- **Mechanism**: Use U2-Net to segment the object to obtain the background $I_b = (1-M_f) \times I$, then pass the background image through the SD encoder to obtain intermediate features. In each self-attention layer, the background features are fused into the global branch via an adaptive MLP: $\varphi_i(z) = \varphi_i(z) + \gamma \cdot MLP(\varphi_i(z^b))$, where $\gamma$ is initialized to 0.1.
- **Design Motivation**: Few-shot training (~8 images) is highly prone to overfitting, manifesting as distorted object shapes, contaminated background colors, and coupling between objects and backgrounds (e.g., generating only half of a bottle, or toothbrushes with double handles). BCM utilizes background and object shapes as explicit constraints, encouraging the model to focus on generating anomalous regions.

### Loss & Training

$$\mathcal{L} = \mathbb{E}_{\mathcal{E}(I),\epsilon,t}\left[\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(p), I_b)\|_2^2\right] + \mathbb{E}_{\mathcal{E}(I_a),\epsilon^*,t}\left[\|\epsilon^* - \epsilon_\theta^*(z_t', t, \tau_\theta(p'))\|_2^2\right]$$

## Key Experimental Results

### Main Results (MVTec AD Generation Quality)

| Method | IS ↑ (Mean) | IC-LPIPS ↑ (Mean) |
|------|-----------|------------------|
| CDC | 1.65 | 0.07 |
| Crop-Paste | 1.51 | 0.14 |
| SDGAN | 1.71 | 0.13 |
| DFMGAN | 1.72 | 0.20 |
| AnomalyDiffusion | 1.79 | 0.32 |
| **DualAnoDiff (ours)** | **1.93** | **0.38** |

### Downstream Anomaly Detection Performance

| Method | Pixel-level AUROC ↑ | Pixel-level AP ↑ |
|------|--------------|------------|
| Without anomaly data | 96.8 | 64.1 |
| AnomalyDiffusion | 98.8 | 82.3 |
| **DualAnoDiff (ours)** | **99.1** | **84.5** |

### Ablation Study

- SAIM Contribution: Removing SAIM drops the IS from 1.93 to approximately 1.75, significantly decreasing the consistency between the generated images and their anomalous parts.
- BCM Contribution: Removing BCM results in severe object deformation and background confusion (e.g., incomplete bottles, toothbrushes with dual handles).
- Nested Prompts: Rare tokens vfx/sks outperform real category names/anomaly names.

### Key Findings

- **Simultaneous vs. Separate Generation**: Simultaneously generated anomalous images comprehensively outperform separate generation (AnomalyDiffusion) in terms of realism, diversity, and mask alignment.
- **Convenient Mask Acquisition**: Since the anomaly branch generates an independent anomalous part, high-precision masks can be easily obtained using SAM or U2-Net.
- **Works with Extremely Few Shots**: Training with only ~8 anomaly samples per category on average is sufficient to generate high-quality anomalous data.

## Highlights & Insights

1. **Dual-Stream Synchronous Generation Design**: The most critical innovation—resolving the mask alignment problem by simultaneously generating the whole and the part, reframing the challenging "generation + segmentation" workflow into an elegant "parallel generation + simple segmentation" approach.
2. **Implicit Alignment via SAIM**: Achieves spatial and semantic consistency between the two branches naturally through shared attention computations, eliminating the need for explicit alignment losses.
3. **BCM Resolving Few-Shot Overfitting**: Injects background information as an explicit prior, effectively mitigating various degradation phenomena under few-shot training.
4. **End-to-End Practicality**: Directly provides training data for downstream anomaly detection, localization, and classification, achieving a state-of-the-art (SOTA) AUROC of 99.1%.

## Limitations & Future Work

1. **Inference Efficiency**: The computational cost of dual-branch simultaneous denoising is approximately twice that of a single branch.
2. **Dependence on U2-Net Segmentation**: Both BCM and mask acquisition rely on the quality of the pre-trained segmentation model.
3. **Evaluation Only on MVTec AD**: Generalization has not been validated on other anomaly detection datasets (such as VisA or MPDD).
4. **Category-Specific Training**: Each category requires separate training, making zero-shot generalization to new categories unviable.
5. **Limited Anomaly Types**: The anomaly descriptions in the nested prompts are relatively simple (a single rare token), limiting the capability to model complex, multi-class anomalies.

## Related Work & Insights

- **AnomalyDiffusion**: Separately learns anomaly appearance and mask location; this work significantly improves alignment via dual-stream synchronous generation.
- **LayerDiffusion**: The inspiration for decomposing an image into multiple layers for parallel generation.
- **DreamBooth/Textual Inversion**: The technical foundation for few-shot customized generation.
- **Insight**: In generation tasks, "simultaneously generating multiple related outputs" is more effective and natural than "generating separately and then aligning".

## Rating

⭐⭐⭐⭐

Highly precise problem definition (few-shot anomaly generation + mask alignment), with a clever and effective dual-stream synchronous generation design. Downstream task performance reaches SOTA, demonstrating clear practical value. The BCM solution for handling few-shot overfitting is also a valuable reference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_for_unified_image_generation_and_understanding.md)
- [\[CVPR 2025\] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)
- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)
- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](multi-party_collaborative_attention_control_for_image_customization.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](../../ICCV2025/image_generation/emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)

</div>

<!-- RELATED:END -->
