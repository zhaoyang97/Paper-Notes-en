---
title: >-
  [Paper Note] GleSAM: Segment Any-Quality Images with Generative Latent Space Enhancement
description: >-
  [CVPR 2025][Segmentation][SAM robustness] GleSAM introduces the denoising capability of pre-trained Latent Diffusion Models (LDMs) into the latent space of SAM. It enhances the feature representation of low-quality images through single-step denoising, achieving robust segmentation for images of any quality.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "SAM robustness"
  - "low-quality image segmentation"
  - "Latent Diffusion Models"
  - "feature enhancement"
  - "degradation robustness"
date: 2026-05-08
content_hash: 1e9253ddcac81c67
---

# GleSAM: Segment Any-Quality Images with Generative Latent Space Enhancement

**Conference**: CVPR 2025  
**arXiv**: [2503.12507](https://arxiv.org/abs/2503.12507)  
**Code**: Coming soon  
**Area**: Image Segmentation  
**Keywords**: SAM robustness, low-quality image segmentation, Latent Diffusion Models, feature enhancement, degradation robustness

## TL;DR

GleSAM introduces the denoising capability of pre-trained Latent Diffusion Models (LDMs) into the latent space of SAM. It enhances the feature representation of low-quality images through single-step denoising, achieving robust segmentation for images of any quality.

## Background & Motivation

Although SAM/SAM2 performs exceptionally well on clean images, its performance drops significantly on low-quality images commonly found in real-world scenarios (e.g., noise, blur, compression artifacts). Existing methods like RobustSAM enhance degraded features through consistency learning, but still struggle under severe and composite degradations.

Core Observation: Latent space features extracted by SAM from severely degraded images contain significant noise, disrupting the original representation. The large gap between low-quality and high-quality features makes consistency learning difficult to converge.

Key Insight: Pre-trained LDMs (such as Stable Diffusion) have learned powerful representation priors and denoising capabilities on large-scale data, which can be introduced into the latent space of SAM to "restore" degraded features.

Unlike cascade methods that combine image restoration and segmentation, GleSAM enhances features **directly in the feature space**, which is more efficient and avoids information loss from image-domain restoration.

## Method

### Overall Architecture

GleSAM inserts a Generative Latent Enhancement (GLE) module between the image encoder and mask decoder of SAM. Degraded features $z_L$ extracted from a low-quality image by the SAM encoder are enhanced into $\hat{z}_H$, which is close to high-quality features, via single-step denoising in the GLE module. These enhanced features are then fed into the fine-tuned decoder to generate accurate masks. The training consists of two stages: first training the U-Net for denoising reconstruction, and then fine-tuning the decoder to align with the enhanced features.

### Key Designs

**1. Latent-space Single-step Denoising Enhancement (GLE)**

- **Function**: Restores low-quality features $z_L$ output by the SAM encoder to high-quality features through single-step diffusion denoising.
- **Mechanism**: Treats low-quality features $z_L$ as a noisy version of high-quality features $z_H$, performing single-step denoising at diffusion timestep $T$: $\hat{z}_H = \frac{z_L - \sqrt{1-\bar{\alpha}_T} \epsilon_\theta(z_L; T)}{\sqrt{\bar{\alpha}_T}}$. A pre-trained LDM's U-Net is used as the denoising backbone, fine-tuned with LoRA layers.
- **Design Motivation**: Although multi-step denoising offers better quality, its computational overhead is heavy. Single-step denoising strikes a good balance between efficiency and effectiveness. Starting from $z_L$ instead of random noise preserves the original content information.

**2. Feature Distribution Alignment (FDA)**

- **Function**: Bridges the distribution gap between SAM's encoder feature space and the LDM's VAE latent space.
- **Mechanism**: Introduces an adaptive scaling weight $\gamma$ to adjust the variance of segmentation features to match the VAE latent space distribution: $\hat{z}_H = \frac{\gamma z_L - \sqrt{1-\bar{\alpha}_T} \epsilon_\theta(\gamma z_L; T)}{\gamma \sqrt{\bar{\alpha}_T}}$, and divides by $\gamma$ after denoising to restore the original distribution.
- **Design Motivation**: Directly inputting SAM features into the LDM U-Net fails to leverage the denoising capability due to distribution mismatch. A simple scaling operation can effectively align the distributions.

**3. Channel Replication and Expansion (CRE)**

- **Function**: Resolves the channel dimension mismatch between the LDM U-Net (4 channels input/output) and SAM's latent space (256 channels).
- **Mechanism**: Replicates and concatenates the pre-trained 4-channel weights of the U-Net's input and output layers to the 256-channel dimension, freezes these weights, and adapts the segmentation features solely through LoRA layers.
- **Design Motivation**: Experimental results show that fine-tuning new input/output layers or the encoder-decoder degrades pre-trained generalization performance. Weight replication preserves pre-trained knowledge, while LoRA provides lightweight adaptation.

### Loss & Training

- U-Net training stage: MSE reconstruction loss $\mathcal{L}_{\text{Rec}} = \mathcal{L}_{\text{MSE}}(\text{GLE}(z_L), z_H)$
- Decoder training stage: Dice Loss + Focal Loss

## Key Experimental Results

### Main Results: Low-Quality Image Segmentation (ThinObject-5K & LVIS Test Sets)

| Method | ThinObject LQ-3 IoU | ThinObject LQ-1 IoU | LVIS LQ-3 IoU | LVIS LQ-1 IoU |
|------|-------------------|-------------------|--------------|--------------|
| SAM | 0.6285 | 0.7527 | 0.4041 | 0.5325 |
| RobustSAM | 0.7015 | 0.7922 | 0.4517 | 0.5262 |
| DiffBIR-SAM | 0.7055 | 0.7927 | 0.5316 | 0.6021 |
| **GleSAM** | **0.7594** | **0.8277** | **0.5535** | **0.6131** |

### Ablation Study: Effectiveness of Components

| Component | ThinObject LQ-3 IoU |
|------|-------------------|
| SAM baseline | 0.6285 |
| + GLE (w/o FDA) | 0.7201 |
| + GLE + FDA | 0.7452 |
| + GLE + FDA + CRE | **0.7594** |

### Key Findings

- The more severe the degradation (LQ-3 vs. LQ-1), the more pronounced the advantage of GleSAM relative to SAM/RobustSAM.
- Performs well on unseen degradation types (ECSSD/COCO-val unseen sets), demonstrating its generalization capability.
- The simple scaling operation of FDA brings about a 2.5% IoU improvement, highlighting the importance of distribution alignment.
- GleSAM based on SAM2 is equally effective, proving the strong generality of the framework.
- Only adds a small number of learnable parameters overall (LoRA + decoder tokens), requiring 30 hours of training on 4 GPUs.

## Highlights & Insights

1. **Transferring the denoising capability of diffusion models from image space to segmentation feature space** is a novel and effective paradigm, avoiding the information loss and computational overhead of image restoration.
2. The technical designs of **Feature Distribution Alignment** (FDA) and **Channel Replication and Expansion** (CRE) are simple and elegant, holding broad transfer value — serving as a reference for any scenario requiring the introduction of LDMs into non-image latent spaces.
3. The constructed LQSeg dataset contains multiple types and levels of degradation combinations, filling the gap in low-quality segmentation evaluation.

## Limitations & Future Work

- The enhancement capability of single-step denoising has limits and may be insufficient for extreme degradations (e.g., those where content is almost entirely lost).
- Currently, only synthetic degradations are addressed; the effectiveness on real-world degradations (e.g., rain, fog, underwater, etc.) remains to be validated.
- The U-Net increases inference overhead; although it is a single step, it still introduces latency to end-to-end segmentation.
- Future work can explore conditional enhancement (adaptively adjusting denoising strength based on the degradation type).

## Related Work & Insights

- **Relation to RobustSAM**: RobustSAM enhances robustness through distillation and consistency learning, while GleSAM enhances representations via generative denoising; their approaches are complementary.
- **Relation to works like VPD**: VPD uses diffusion models as backbones to extract features, whereas GleSAM introduces diffusion denoising into the latent space of existing segmentation models.
- **Insight**: Prior knowledge from pre-trained generative models can be used not only for generation but also for "restoring" degraded representations in discriminative models.

## Rating

⭐⭐⭐⭐

The idea of introducing diffusion models into the SAM latent space for feature enhancement is novel and effective. The technical route is comprehensive (GLE+FDA+CRE), and the experiments cover multiple degradation types and levels. The constructed LQSeg dataset is of independent value. Minor limitations include the gap between synthetic and real-world degradations, and the trade-off in inference efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Segment Any Motion in Videos](segment_any_motion_in_videos.md)
- [\[CVPR 2025\] SAP: Segment Any 4K Panorama](sap_segment_any_4k_panorama.md)
- [\[CVPR 2025\] Generative Video Propagation](generative_video_propagation.md)
- [\[CVPR 2025\] Image Quality Assessment: From Human to Machine Preference](image_quality_assessment_from_human_to_machine_preference.md)
- [\[CVPR 2025\] DefMamba: Deformable Visual State Space Model](defmamba_deformable_visual_state_space_model.md)

</div>

<!-- RELATED:END -->
