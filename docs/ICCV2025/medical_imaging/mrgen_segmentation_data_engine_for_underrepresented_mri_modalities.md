---
title: >-
  [Paper Note] MRGen: Segmentation Data Engine for Underrepresented MRI Modalities
description: >-
  [ICCV 2025][Medical Imaging][MRI synthesis] To address the lack of segmentation annotations for scarce MRI modalities, this work constructs a large-scale radiological image dataset MRGen-DB (~250K slices, 100+ modalities) and trains a controllable diffusion-based data engine MRGen. Using dual-condition control via text prompts and segmentation masks, MRGen generates high-quality MR images in target modalities for training segmentation models. Across 10 cross-modal segmentation experiments, the average DSC improves from 10%–27% to 43%–45%, enabling "zero-shot" segmentation for annotation-scarce modalities.
tags:
  - ICCV 2025
  - Medical Imaging
  - MRI synthesis
  - diffusion models
  - data engine
  - segmentation
  - cross-modal generation
date: 2026-05-08
content_hash: 475aeb7ebd64f3fa
---

# MRGen: Segmentation Data Engine for Underrepresented MRI Modalities

**Conference**: ICCV 2025
**arXiv**: [2412.04106](https://arxiv.org/abs/2412.04106)
**Code**: [haoningwu3639.github.io/MRGen](https://haoningwu3639.github.io/MRGen/)
**Area**: Medical Imaging
**Keywords**: MRI synthesis, diffusion models, data engine, segmentation, cross-modal generation

## TL;DR

To address the lack of segmentation annotations for scarce MRI modalities, this work constructs a large-scale radiological image dataset MRGen-DB (~250K slices, 100+ modalities) and trains a controllable diffusion-based data engine MRGen. Using dual-condition control via text prompts and segmentation masks, MRGen generates high-quality MR images in target modalities for training segmentation models. Across 10 cross-modal segmentation experiments, the average DSC improves from 10%–27% to 43%–45%, enabling "zero-shot" segmentation for annotation-scarce modalities.

## Background & Motivation

### Problem Definition

MRI is a non-invasive, radiation-free imaging modality of great clinical importance, but scans are expensive and exhibit extremely high modality diversity (T1, T2, FLAIR, DWI, ADC, etc.), with significant signal characteristic differences across modalities. This leads to two core problems:
- Certain clinically important but scarce modalities lack sufficient annotated data for training segmentation models.
- Existing segmentation models suffer severe performance degradation when generalizing across modalities.

### Limitations of Prior Work

**Data augmentation methods** (e.g., DiffTumor, DualNorm): augment only on already-annotated modalities and cannot generate training data for new modalities lacking annotations. Augmentation strategies simulate domain gaps via strong transformations, but with limited effectiveness.

**Image translation methods** (CycleGAN, UNSB): translate images from one modality to another, but typically require paired/registered data, are restricted to specific modality-pair conversions, suffer from training instability, and are prone to mode collapse.

**Existing medical generative models**: primarily focus on X-ray and CT (where data is relatively abundant), or are limited to specific MRI sub-domains (e.g., brain MRI), and cannot simultaneously support dual-condition control via both text and masks.

### Core Problem

**Key insight**: If a data engine can controllably generate MR images conditioned jointly on a text prompt (describing the target modality) and a segmentation mask (describing anatomical structures), then masks from annotated modalities can be leveraged to generate training data for unannotated target modalities. Through a two-stage training strategy—first learning modality generation, then learning mask-conditioned control—the model can generalize mask control capability to modalities unseen with mask annotations during training.

## Method

### Overall Architecture

MRGen consists of three core stages:
1. **Dataset curation**: constructing the large-scale radiological image dataset MRGen-DB.
2. **Model training**: autoencoder → text-guided pretraining → mask-conditioned fine-tuning.
3. **Synthetic data for segmentation training**: generation + SAM2 automatic filtering → downstream segmentation model training.

### Key Designs

#### 1. **MRGen-DB Dataset Construction**

- **Function**: Curate a large-scale MRI dataset with rich metadata (modality labels, attributes, regions, organ information) and partial mask annotations.
- **Mechanism**:

  Dual-track data sourcing:
  - **Radiopaedia**: 5,414 volumes, 205,039 slices, image–text pairs covering 100+ modalities.
  - **Open-source datasets** (PanSeg, MSD-Prostate, CHAOS-MRI, PROMISE12, LiQA): 766+ volumes with organ mask annotations.
  Total: ~6,384 volumes, 245,082 2D slices, 17,861 slices with mask annotations.

  Three-level automatic annotation:
  1. **Region classification**: a pretrained BiomedCLIP classifies slices into 6 anatomical regions (slices with confidence <40% are left unlabeled).
  2. **Modality attribute description**: GPT-4 maps modality labels to tissue signal intensity descriptions (e.g., "T1: fat high signal, muscle intermediate signal, water low signal").
  3. **Quality validation**: sampled manual verification (95.33% accuracy for region labels, 91.67% for attribute descriptions).

- **Design Motivation**: Visual differences across MRI modalities far exceed the semantic differences in modality names (T1 vs. T2 appear very similar to text encoders). Tissue signal descriptions are therefore introduced as "attributes" to help the generative model distinguish modalities. Region classification further refines anatomical location information.

#### 2. **Controllable Diffusion Generative Model (MRGen)**

- **Function**: Achieve dual-condition (text + mask) controlled MR image generation in latent space.
- **Mechanism**:

  **(a) Latent space encoding**: An autoencoder compresses $\mathcal{I} \in \mathbb{R}^{H \times W \times 1}$ to $\mathbf{z} \in \mathbb{R}^{h \times w \times d}$ (compression ratio 8, latent dimension $d=16$), with loss:

  $$\mathcal{L}_{VAE} = \|\mathcal{I} - \hat{\mathcal{I}}\|_2^2 + \gamma \mathcal{L}_{KL}$$

  **(b) Text-guided generation**: Standard diffusion model paradigm. Templatized text prompts encode four levels of information: modality, attribute, region, and organ. Prompts are encoded by BiomedCLIP and injected into the UNet via cross-attention:

  $$\mathbf{O}_{cross} = \mathcal{F}_{cross}(\mathbf{z}_t, \phi_{text}(\mathcal{T}))$$

  The objective is the standard denoising loss:

  $$\mathcal{L} = \mathbb{E}_{t, \epsilon} \left[ \|\epsilon - \hat{\epsilon}(\mathbf{z}_t, t, \mathcal{T})\|_2^2 \right]$$

  **(c) Mask-conditioned generation**: The mask encoder $\phi_{mask}$ is initialized by reusing the UNet encoder weights from the text-guided stage, augmented with a learnable downsampling module $\phi_{down}$. Mask features are injected as residuals into each layer of the UNet decoder:

  $$\mathbf{O}^i = \mathcal{F}^i(\mathbf{z}_t) + \phi_{mask}^i(\mathbf{z}_t, \phi_{down}(\mathcal{M}), \phi_{text}(\mathcal{T}))$$

  Fine-tuning uses both masked and unmasked data to prevent overfitting on the limited mask data:

  $$\mathcal{L}_c = \mathbb{E}_{t, \epsilon} \left[ \|\epsilon - \hat{\epsilon}_c(\mathbf{z}_t, t, \mathcal{T}, \mathcal{M})\|_2^2 \right]$$

- **Design Motivation**: The two-stage training strategy is central to the method's success. Stage one learns modality generation capability across large-scale image–text pairs; stage two learns controllable generation from limited mask data. Since stage one already covers diverse modalities, mask control capability naturally generalizes to modalities that were seen during pretraining but lacked mask annotations—this is the core mechanism enabling "zero-shot" segmentation.

#### 3. **Synthetic Data Filtering and Segmentation Training**

- **Function**: Automatically assess alignment between generated images and conditioning masks, and select high-quality samples for segmentation training.
- **Mechanism**:

  Pretrained SAM2-Large is used to verify alignment between generated images and input masks:
  1. The conditioning mask $\mathcal{M}'$ and generated image $\mathcal{I}'$ are fed into SAM2.
  2. SAM2 outputs a segmentation prediction and confidence score $s_{conf}$.
  3. IoU $s_{IoU}$ between the prediction and the conditioning mask is computed.
  4. Only samples with $s_{IoU} > 0.80$ and $s_{conf} > 0.90$ are retained.

  20 candidate images are generated per mask, and the top 2 are selected.

- **Design Motivation**: Generative models inevitably produce images misaligned with conditioning masks, especially in cross-modal generation. SAM2 filtering provides reliable quality assurance and prevents noisy annotations from degrading downstream segmentation models.

### Loss & Training

- **Autoencoder**: MSE + KL divergence; lr=5e-5, batch=256, 50K iterations.
- **Text-guided pretraining**: Denoising MSE loss; lr=1e-5, batch=256, 200K iterations; text dropped with 10% probability (classifier-free guidance).
- **Mask-conditioned fine-tuning**: Denoising MSE loss; lr=1e-5, batch=128, 40K iterations; only the mask encoder and downsampling module are trained.
- **Inference**: 50-step DDIM sampling, classifier-free guidance weight=7.0.
- **Hardware**: 8× NVIDIA A100.

## Key Experimental Results

### Main Results

**10 cross-modal segmentation experiments (DSC score, nnUNet framework)**:

| Source→Target | No synthesis ($\mathcal{D}_s$) | DualNorm | CycleGAN | UNSB | **MRGen** |
|---|---|---|---|---|---|
| CHAOS T1→T2-SPIR | 6.90 | — | 7.58 | 14.03 | **66.18** |
| CHAOS T2-SPIR→T1 | 0.80 | — | 1.38 | 6.44 | **58.10** |
| MSD T2→ADC | 5.52 | — | 40.92 | 52.99 | **57.83** |
| MSD ADC→T2 | 22.20 | — | 57.06 | 38.39 | **61.95** |
| PanSeg T1→T2 | 0.68 | — | 2.40 | 2.38 | **9.78** |
| PanSeg T2→T1 | 0.30 | — | 3.59 | 6.68 | **12.07** |
| **Average DSC** | **10.48** | 8.41 | 26.39 | 23.85 | **44.71** |

**Generation quality (FID ↓)**:

| Method | DualNorm | CycleGAN | UNSB | **MRGen** |
|---|---|---|---|---|
| Average FID | 290.37 | 178.18 | 194.78 | **82.18** |

### Ablation Study

**Component ablation (segmentation DSC, nnUNet)**:

| Configuration | T1→T2-SPIR | T2-SPIR→T1 | T2→ADC | ADC→T2 |
|---|---|---|---|---|
| nnUNet (source domain data) | 6.90 | 0.80 | 5.52 | 22.20 |
| + MRGen synthesis (no filtering, no target-domain images) | 16.53 | 15.10 | 39.90 | 18.92 |
| + AutoFilter | 22.30 | 20.27 | 42.79 | 25.34 |
| + Target-domain images (unannotated) | 30.16 | 29.01 | 49.04 | 40.89 |
| + AutoFilter + Target-domain images | **66.18** | **58.10** | **57.83** | **61.95** |

**Generative model ablation**:

| Model | PSNR↑ | SSIM↑ | FID↓ | CLIP-I↑ | CLIP-T↑ |
|---|---|---|---|---|---|
| SDM (original) | 31.32 | 0.989 | 249.24 | 0.3151 | 0.1748 |
| SDM-ft (fine-tuned) | 35.65 | 0.996 | 91.48 | 0.6698 | 0.3199 |
| MRGen-M (modality label only) | — | — | 41.82 | 0.7512 | 0.3765 |
| **MRGen (full template)** | **42.62** | **0.999** | **39.63** | **0.8457** | **0.3777** |

### Key Findings

1. **Substantial gains in cross-modal segmentation**: MRGen synthetic data improves average DSC from 10.48% to 44.71%, a more than fourfold improvement.
2. **Large margin over image translation methods**: CycleGAN and UNSB are unstable on complex modality transfers (e.g., PanSeg T1→T2 yields only 2.40 DSC); MRGen is considerably more robust.
3. **AutoFilter and target-domain images are both essential**: Each contributes significantly, and their combination far outperforms either alone.
4. **High-capacity autoencoder and medical text encoder are critical**: Latent dimension 16 (vs. SDM's 4) combined with BiomedCLIP yields large gains over generic SD fine-tuning.
5. **Value of templatized text prompts**: The full template (modality + attribute + region + organ) outperforms modality labels alone (FID 41.82→39.63, CLIP-I 0.75→0.85).

## Highlights & Insights

1. **Data engine paradigm**: Rather than directly addressing cross-domain segmentation or domain adaptation, this work solves the problem at the data source—generating high-quality target-domain training data.
2. **Two-stage strategy enables zero-shot transfer**: The first stage learns modality generation over 100+ modalities; the second stage learns controllable generation from limited annotated data. Mask control capability naturally generalizes to unannotated modalities.
3. **Standalone dataset contribution**: MRGen-DB is the first open-source large-scale dataset designed for MRI generation and carries independent academic value.
4. **Clever application of SAM2 filtering**: A general-purpose visual foundation model serves as a quality gatekeeper, preventing generated noise from propagating to downstream models.

## Limitations & Future Work

1. **Limited to abdominal MRI**: The current dataset and evaluation focus on the abdominal region, excluding other important anatomical sites such as the brain and heart.
2. **2D slice generation**: Images are generated slice-by-slice independently, lacking 3D consistency (though slices are stacked into 3D volumes for DSC evaluation).
3. **PanSeg segmentation performance remains low**: DSC ~10%, indicating that generation quality for certain organ/modality combinations still has room for improvement.
4. **High computational cost**: Training requires 8×A100, and 20 candidate images are generated per mask; efficiency must be considered for large-scale deployment.
5. **No comparison with dedicated domain adaptation methods**: Such as test-time adaptation methods like TENT or CoTTA.

## Related Work & Insights

- **Fundamental difference from CycleGAN**: CycleGAN learns mappings between specific modality pairs, requires paired data, and handles only one modality pair at a time. MRGen learns general controllable generation capability, supporting arbitrary modalities within a single model.
- **Relationship to DiffTumor/FreeTumor**: The latter performs tumor generation augmentation on already-annotated modalities, representing a "data augmentation" paradigm. MRGen follows a "data engine" paradigm, generating training sets from scratch for unannotated modalities.
- **Broader implications**: The two-stage strategy of "first learn general generation, then learn conditional control" can be extended to other modality-scarce scenarios (e.g., ultrasound, endoscopy).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The data engine concept is well-motivated and the two-stage design is effective, though the overall technical stack (LDM + ControlNet variant) is not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 10 cross-modal experiments, two segmentation frameworks (nnUNet + UMamba), dual evaluation of generation and segmentation quality, with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation figures are clear and the positioning relative to prior work is well articulated.
- **Value**: ⭐⭐⭐⭐⭐ — Directly addresses the practical bottleneck of MRI annotation scarcity; both the dataset and model are open-sourced, offering high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Domain-Adaptive Transformer for Data-Efficient Glioma Segmentation in Sub-Saharan MRI](../../NeurIPS2025/medical_imaging/domain-adaptive_transformer_for_data-efficient_glioma_segmentation_in_sub-sahara.md)
- [\[ICCV 2025\] M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast](m-net_mri_brain_tumor_sequential_segmentation_network_via_mesh-cast.md)
- [\[ICCV 2025\] Scaling Tumor Segmentation: Best Lessons from Real and Synthetic Data](scaling_tumor_segmentation_best_lessons_from_real_and_synthetic_data.md)
- [\[ICCV 2025\] UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation](ukbob_one_billion_mri_labeled_masks_for_generalizable_3d_medical_image_segmentat.md)
- [\[ICCV 2025\] TeethGenerator: A Two-Stage Framework for Paired Pre- and Post-Orthodontic 3D Dental Data Generation](teethgenerator_a_two-stage_framework_for_paired_pre-_and_post-orthodontic_3d_den.md)

</div>

<!-- RELATED:END -->
