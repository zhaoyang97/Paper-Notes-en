---
title: >-
  [Paper Note] ReferEverything: Towards Segmenting Everything We Can Speak of in Videos
description: >-
  [ICCV 2025][Segmentation][Referring Video Segmentation] By leveraging the general visual-language mappings learned by video diffusion models, and by preserving the complete generative model architecture while shifting the prediction target from noise to mask latents, this work enables open-world referring segmentation of any concept expressible in natural language in videos — including non-object dynamic processes.
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Referring Video Segmentation"
  - "Video Diffusion Models"
  - "Open-World Segmentation"
  - "Dynamic Process Segmentation"
  - "Generative Pre-training"
date: 2026-05-08
content_hash: 767bedf5b6c765e2
---

# ReferEverything: Towards Segmenting Everything We Can Speak of in Videos

**Conference**: ICCV 2025
**arXiv**: [2410.23287](https://arxiv.org/abs/2410.23287)  
**Code**: [Project Page](https://refereverything.github.io/)  
**Area**: Image Segmentation
**Keywords**: Referring Video Segmentation, Video Diffusion Models, Open-World Segmentation, Dynamic Process Segmentation, Generative Pre-training

## TL;DR

By leveraging the general visual-language mappings learned by video diffusion models, and by preserving the complete generative model architecture while shifting the prediction target from noise to mask latents, this work enables open-world referring segmentation of any concept expressible in natural language in videos — including non-object dynamic processes.

## Background & Motivation

Referring Video Segmentation (RVS) aims to segment specific regions in a video based on natural language descriptions. However, existing work almost exclusively focuses on Referring Video **Object** Segmentation (RVOS), a limitation rooted in data construction — RVOS datasets are built upon object-tracking benchmarks, making them inherently object-centric and limited in scale.

The paper identifies a key insight: natural language can describe not only objects but also a wide variety of dynamic processes (e.g., smoke dissipating, glass shattering, raindrops rolling). If an event can be expressed in language, it should be spatiotemporally localizable in video. Nevertheless, existing RVOS methods generalize poorly outside their training distribution and fail to handle rare objects or non-object concepts.

Video diffusion models, pre-trained on internet-scale data, have learned powerful mappings from language descriptions to video regions. Prior work (e.g., VD-IT) has attempted to use diffusion models as feature extractors, but replacing portions of the generative architecture disrupts the aligned representations learned during pre-training, severely degrading generalization. The central motivation of this paper is: **preserving the complete generative model architecture is the key to unlocking maximal generalization capability**.

## Method

### Overall Architecture

The core idea of REM (Refer Everything with Diffusion Models) is to treat the video diffusion model not merely as a feature extractor, but to retain its full architecture (including the denoising network and VAE), switching only the prediction target from noise to mask latents. The framework takes noisy video frames and a language expression as input and produces a latent representation of the segmentation mask as output.

### Key Designs

1. **From Noise Prediction to Mask Latent Prediction**: Conventional approaches extract intermediate features $\epsilon_\theta^{(n)}$ from the diffusion model and train a new decoding head $f_{\text{dec}}$ to predict the mask. REM's key innovation is to directly reuse the complete denoising network $\epsilon_\theta$ and the frozen VAE decoder $\mathcal{D}$, changing the objective from predicting noise to predicting mask latents:

$$\hat{m} = \mathcal{D}(\epsilon_\theta(z_t, e_c, t))$$

This seemingly minor modification better preserves the general visual-language mappings learned during generative pre-training. The design motivation is to avoid replacing pre-trained components with randomly initialized layers, thereby preventing the destruction of alignment between pre-trained representations and newly learned features.

2. **Mask Encoding and Decoding Strategy**: During training, the single-channel ground-truth mask is replicated three times to form a three-channel representation, which is then mapped to the latent space via the pre-trained VAE encoder as $\mathcal{E}(m) = z^m$. During inference, the predicted latent is decoded into a three-channel mask by the frozen VAE decoder; the three-channel average is computed and binarized at a threshold of 0.5. Inference is **non-iterative** (single forward pass), with computational cost comparable to other methods.

3. **Two-Stage Training Protocol (ModelScope version) / Single-Stage Training (Wan version)**: For the ModelScope-1.4B backbone, a two-stage training procedure is adopted: Stage I fine-tunes spatial weights on Ref-COCO image-text pairs (1 epoch), and Stage II fine-tunes all weights on Ref-YTB video-text samples (40 epochs) with pseudo-video augmentation. For the Wan-14B DiT architecture, which jointly models spatiotemporal information, a single-stage training directly on merged data for 80K iterations is used. The text encoder and VAE are frozen throughout.

### Loss & Training

The training objective is an $\mathcal{L}_2$ loss that minimizes the distance between the predicted latent and the GT mask latent:

$$\min_\theta \mathbb{E}_{z^m \sim \mathcal{E}(m), t=0} \|z^m - \epsilon_\theta(z_t, e_c, t)\|_2^2$$

The timestep is fixed at $t = 0$, prioritizing as clean a latent as possible, ensuring the model learns precise mask prediction under minimal noise conditions.

## Key Experimental Results

### Main Results

**Standard RVOS Benchmarks (Ref-DAVIS & Ref-YTB)**

| Method | Pre-training | Ref-DAVIS $\mathcal{J\&F}$ | Ref-YTB $\mathcal{J\&F}$ |
|---|---|---|---|
| Referformer | ImageNet+Kinetics | 61.1 | 62.9 |
| UNINEXT | Object365 | 72.5 | 70.1 |
| VD-IT | LAION5B+WebVid | 69.4 | 66.5 |
| **REM (MS-1.4B)** | LAION5B+WebVid | **72.6** | 68.4 |
| **REM (Wan-14B)** | Internal+Public | **75.0** | **71.7** |

**Out-of-Domain Generalization (BURST & VSPW & Ref-VPS)**

| Method | VSPW $\mathcal{J}$ | BURST $\mathcal{J}$ | Ref-VPS $\mathcal{J}$ |
|---|---|---|---|
| UNINEXT | 10.1 | 30.2 | 28.7 |
| VD-IT | 12.7 | 29.0 | 37.9 |
| **REM (MS-1.4B)** | 15.2 | 37.5 | **49.0** |
| **REM (Wan-14B)** | **18.5** | **40.9** | **50.0** |

REM demonstrates substantially superior out-of-domain generalization, surpassing UNINEXT by 21.3 $\mathcal{J}$ points on Ref-VPS.

### Ablation Study

| Supervision Space | Resolution | Decoder | Ref-YTB $\mathcal{J\&F}$ | Ref-VPS $\mathcal{J}$ |
|---|---|---|---|---|
| Latent | 512×512 | VAE (frozen) | 63.5 | **40.0** |
| RGB | 256×256 | VAE (frozen) | 58.4 | 31.6 |
| RGB | 256×256 | VAE (fine-tuned) | 60.4 | 32.4 |
| RGB | 512×512 | CNN | 59.6 | 29.4 |
| RGB | 512×512 | MLP | 59.3 | 33.1 |

Supervising mask prediction in the latent space is the key to generalization; using the pre-trained VAE decoder consistently outperforms CNN/MLP decoders trained from scratch.

### Key Findings

- Retaining the complete generative model architecture (denoising network + VAE) is essential for maximizing generalization capability.
- Advances in video diffusion models (e.g., upgrading from MS-1.4B to Wan-14B) directly translate into improved segmentation performance.
- VD-IT, despite sharing the same diffusion backbone, suffers limited generalization due to partial architectural replacement.
- REM also achieves state-of-the-art performance on the MeViS motion-guided segmentation dataset (60.3 $\mathcal{J\&F}$).

## Highlights & Insights

- **Paradigm Shift**: The diffusion model is transformed from a feature extractor into an end-to-end mask predictor, preserving complete pre-trained representations through an elegant objective switch.
- **New Benchmark Ref-VPS**: The paper is the first to define the Referring Video Process Segmentation task, covering 39 dynamic process concepts (smoke, fire, shattering, etc.), filling a critical gap in RVS evaluation.
- **New Evidence of Diffusion Model Capability**: This work demonstrates that the visual-language mappings learned by video diffusion models through internet-scale pre-training are genuinely general and can be directly transferred to segmentation tasks.

## Limitations & Future Work

- **Inference Efficiency**: Diffusion models have large parameter counts (1.4B–14B), leading to high deployment costs.
- Segmentation accuracy on "stuff" categories remains limited (only 18.5 on VSPW).
- Fine-tuning is currently restricted to Ref-COCO and Ref-YTB, with limited annotation volume; additional labeled data may further improve performance.
- Multi-turn interaction and more complex language expressions have not been explored.

## Related Work & Insights

- VD-IT similarly leverages a T2V model for RVOS, but its use of intermediate features with a new decoding head results in weaker generalization than REM.
- UNINEXT unifies multiple localization tasks with strong in-domain performance, but poor out-of-domain generalization.
- Future work could extend this strategy to image diffusion models for static image segmentation, or explore its applicability to 3D perception tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Elevates diffusion models from feature extractors to end-to-end segmentors with full architecture preservation — a minimal modification with significant impact.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 6 benchmarks, including the newly introduced Ref-VPS, with detailed ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation, logical progression, and in-depth ablation discussion.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new paradigm for open-world video segmentation; the new benchmark is likely to foster community progress.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Open-World Skill Discovery from Unsegmented Demonstration Videos](open-world_skill_discovery_from_unsegmented_demonstration_videos.md)
- [\[ICCV 2025\] Learning Precise Affordances from Egocentric Videos for Robotic Manipulation](learning_precise_affordances_from_egocentric_videos_for_robotic_manipulation.md)
- [\[ICCV 2025\] 2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos](2handedafforder_learning_precise_actionable_bimanual_affordances_from_human_vide.md)
- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[NeurIPS 2025\] ConnectomeBench: Can LLMs Proofread the Connectome?](../../NeurIPS2025/segmentation/connectomebench_can_llms_proofread_the_connectome.md)

</div>

<!-- RELATED:END -->
