---
title: >-
  [Paper Note] Co-synthesis of Histopathology Nuclei Image-Label Pairs using a Context-Conditioned Joint Diffusion Model
description: >-
  [ECCV 2024][Medical Imaging][Histopathology] A context-conditioned joint diffusion model is proposed to simultaneously synthesize histopathology nuclei images, semantic labels, and distance maps. Precise control over the synthesis process is achieved through point map (centroid layout) and text prompt conditions, generating high-quality instance-level labels for downstream cell nuclei segmentation and classification tasks.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Histopathology"
  - "nuclei segmentation"
  - "joint diffusion model"
  - "data augmentation"
  - "image-label co-synthesis"
date: 2026-05-08
content_hash: 067dbe7d1a8a2060
---

# Co-synthesis of Histopathology Nuclei Image-Label Pairs using a Context-Conditioned Joint Diffusion Model

**Conference**: ECCV 2024  
**arXiv**: [2407.14434](https://arxiv.org/abs/2407.14434)  
**Code**: None  
**Area**: Medical Image  
**Keywords**: Histopathology, nuclei segmentation, joint diffusion model, data augmentation, image-label co-synthesis

## TL;DR

A context-conditioned joint diffusion model is proposed to simultaneously synthesize histopathology nuclei images, semantic labels, and distance maps. Precise control over the synthesis process is achieved through point map (centroid layout) and text prompt conditions, generating high-quality instance-level labels for downstream cell nuclei segmentation and classification tasks.

## Background & Motivation

Histopathology nuclei segmentation and classification are critical tasks in digital pathology, where nuclei features such as size, shape, and density provide essential clues for disease diagnosis. Deep learning methods have made significant progress in this field, but their performance is constrained by **the scarcity of training data**—manually annotating histopathology images requires the expertise of pathologists, which is time-consuming and expensive.

**Limitations of Prior Work**:

- **Lack of Context Awareness**: Existing generative methods (such as GANs and diffusion models) often ignore the biophysical context (shape, spatial layout, tissue type) of biological tissues, resulting in a lack of spatial and structural realism in synthesized data.
- **Inability to Generate Image-Label Pairs Simultaneously**: Most methods either generate labels first and then images (two-stage methods, which are slow), or generate images based on existing pixel-level labels (lacking label diversity). No method has achieved the co-generation of images and labels in a single step within a unified model.
- **Insufficient Label Control**: Semantic-Palette can control class proportions but cannot precisely control spatial positions; Abousamra et al. can generate spatially-aware point layouts but only produce point-level labels, which cannot be used for segmentation tasks.
- **Difficulty in Instance Separation**: In conventional semantic label synthesis, adjacent nuclei tend to cluster into a single large region, making it impossible to distinguish individual instances.

**Key Challenge**: How to precisely control the spatial layout and tissue type of synthesized samples in a unified framework, while simultaneously generating high-quality images and multi-granularity (semantic + instance) labels?

**Key Insight**: By combining point map conditions (controlling spatial positions and classes of nuclei) with text conditions (controlling tissue types), a joint diffusion model is designed to simultaneously generate images, semantic labels, and distance maps. Subsequently, instance labels are generated from the distance maps and point maps using the marker-controlled watershed algorithm.

## Method

### Overall Architecture

The system consists of a joint diffusion model. The inputs are a point map representing the centroid layout of nuclei instances and a text prompt describing the tissue type. The output is a triplet $u := (i, d, l^s)$, containing the image $i$, distance map $d$, and semantic label $l^s$. Subsequently, instance labels $l^i$ are generated from $d$, $l^s$, and the point map through a post-processing step.

### Key Designs

1. **Joint Diffusion Process**: Appropriate noise distributions are selected for different modalities—images and distance maps are continuous data, utilizing **Gaussian diffusion** (Eq.1); semantic labels represent discrete data ($K$ classes), utilizing **categorical diffusion** (Eq.2). Three targets are denoised simultaneously in the reverse process:

$$p_\theta^u(u_{t-1}|u_t) = p_\theta^i(i_{t-1}|u_t) \cdot p_\theta^d(d_{t-1}|u_t) \cdot p_\theta^{l^s}(l^s_{t-1}|u_t)$$

Training uses a compound loss function: $\mathcal{L}_{total} = \lambda_i \cdot \mathcal{L}_i + \lambda_d \cdot \mathcal{L}_d + \lambda_{l^s} \cdot \mathcal{L}_{l^s}$, where $\lambda_i=9, \lambda_d=1, \lambda_{l^s}=3$.

**Design Motivation**: By jointly modeling the joint distribution of multiple modalities in a single model, consistency between images and labels is guaranteed while avoiding cumulative errors and inference latency associated with multi-stage methods.

2. **Context Conditions**: Two conditions are introduced to enhance generation quality and controllability:

    - **Point map condition $pc$**: Defines the centroid location and class of each nuclei instance, encoded via an RRDB network. Compared to pixel-level label conditions, the point map condition requires only a 1-pixel guidance signal per instance but generates highly diverse labels (the same point layout can generate different labels and images).
    - **Text condition $tc$**: Contains tissue type and nuclei class information, formatted as "high-quality histopathology [tissue type] tissue image including nuclei types of [cell types]", encoded using PLIP, a pathology-specific vision-language model.

   A classifier-free guidance mechanism is employed to adjust the predicted noise: $\tilde{\epsilon}_\theta(u_t, t, pc, tc) = \omega \epsilon_\theta(u_t, t, pc, tc) + (1-\omega) \epsilon_\theta(u_t, t, pc)$

**Design Motivation**: Point maps provide precise control over spatial and class distributions, while text provides global semantic information at the tissue structure level. The two complement each other to achieve comprehensive control over the generated content.

3. **Nuclei Instance Separation**: Utilizes the synthesized distance map $d$, semantic labels $l^s$, and the point map condition $pc$ as markers to apply the **marker-controlled watershed algorithm**, separating semantic labels into instance-level labels $l^i$. The distance map quantifies the normalized Euclidean distance ($0$-$1$) of each pixel to the nearest nuclei centroid.

**Design Motivation**: Conventional connectivity analysis and marker-free watershed algorithms are prone to under-segmentation or over-segmentation. Using point maps as markers precisely determines the seed points for each instance, significantly improving the quality of instance separation.

### Loss & Training

- The total loss is a weighted sum of three objectives, corresponding respectively to the image (MSE noise prediction loss), distance map (MSE noise prediction loss), and semantic labels (categorical diffusion loss).
- Optimized using the Adam optimizer ($\beta_1=0.9, \beta_2=0.99$), with a learning rate of $10^{-4}$ for Lizard/PanNuke and $10^{-5}$ for EndoNuke.
- Sampling steps $T=1000$, using three independent cosine schedules.
- During training, the text condition is discarded with a 10% probability to enable classifier-free guidance.

## Key Experimental Results

### Main Results

Synthesizing quality is evaluated on three multi-class histopathology nuclei segmentation datasets (Lizard, PanNuke, and EndoNuke) using three metrics: FID, IS, and FSD.

| Method | Lizard FID↓ | Lizard IS↑ | Lizard FSD↓ | PanNuke FID↓ | PanNuke IS↑ | PanNuke FSD↓ |
|------|-------------|------------|-------------|--------------|------------|--------------|
| Yu et al. | - | - | 963.36 | - | - | 1292.05 |
| SemanticPalette | 86.17 | 2.11 | 0.55 | 109.23 | 3.36 | 1.23 |
| Park et al. | 52.65 | 2.22 | 65.06 | 61.16 | 3.48 | 34.43 |
| SDM | 45.99 | 2.35 | - | 107.80 | 3.82 | - |
| **Ours** | **38.78** | **2.40** | **0.13** | **37.35** | **3.77** | **1.44** |

### Downstream Task Performance (Hover-Net Baseline)

| Dataset | Method | Dice | AJI | Acc | Description |
|--------|------|------|-----|-----|------|
| Lizard | Baseline | 0.620 | 0.383 | 0.763 | Real data only |
| Lizard | w/ SDM | 0.718 | 0.488 | 0.862 | Full pixel label condition |
| Lizard | **w/ Ours** | 0.716 | 0.484 | **0.866** | Point condition, close to SDM |
| PanNuke | Baseline | 0.782 | 0.598 | 0.668 | Real data only |
| PanNuke | **w/ Ours** | **0.824** | **0.662** | **0.736** | First in multiple metrics |
| EndoNuke | Baseline | 0.878 | 0.594 | 0.891 | Real data only |
| EndoNuke | **w/ Ours** | **0.899** | **0.645** | **0.926** | First in multiple metrics |

### Ablation Study: Comparison of Instance Separation Methods

| Method | Lizard mDice | PanNuke mDice | EndoNuke mDice |
|------|-------------|---------------|----------------|
| Connectivity-based | 0.9383 | 0.9146 | 0.5524 |
| Yu et al. (watershed) | 0.9374 | 0.9462 | 0.9268 |
| **Ours (point-guided)** | **0.9754** | **0.9980** | **0.9634** |

### Key Findings

- Using only a 1-pixel-per-instance point condition achieves an FSD of 0.13 (Lizard), which is significantly superior to full pixel label conditions.
- When scaling up the size of synthesized datasets, the proposed method consistently improves downstream task performance, whereas the SDM method saturates after 4 sets (with discrepancies in Dice and classification accuracy exceeding 10%), proving that the label diversity of point condition generation is more effective.
- Pathologist blind evaluation indicates that the realism score of synthesized images is even higher than that of real images, and the image-to-label alignment matches that of real data.

## Highlights & Insights

- **Extremely Minimalist Conditional Control**: Requires only 1 centroid pixel per nucleus to guide the generation of high-quality image-label pairs, greatly reducing the cost of conditional annotation.
- **One Model, Three Outputs**: A single joint diffusion model simultaneously generates images, distance maps, and semantic labels, avoiding the runtime and quality degradation associated with multi-stage inference.
- **Flexibility Advantage of Point Conditions**: The same point layout can produce diverse labels and images, whereas full pixel-level label conditions can only vary the image style. This translates to superior data diversity in data augmentation scenarios.

## Limitations & Future Work

- **Long Data Synthesis Time**: 1000-step sampling in diffusion models remains time-consuming; accelerated sampling methods need to be explored.
- **Point Layout Generation**: Currently relies on extracting point layouts from real data; methods to generate more realistic point layouts can be developed in the future.
- **Single Resolution**: Experiments were only conducted at a resolution of 256×256; analysis at the whole slide level may require larger resolutions.
- **Simplistic Text Conditions**: Text formats are relatively fixed; fine-grained text descriptions were not explored.

## Related Work & Insights

- **Dataset-GAN / SB-GAN**: Early image-label pair generation methods, which generate images/labels first and then generate the other component.
- **DDPM + Categorical Diffusion (Hoogeboom et al.)**: Diffusion processes designed for discrete data, which the proposed method adapts for semantic labels.
- **Park et al.**: Text-conditioned image-label co-synthesis without distance maps or instance labels.
- **PLIP**: Pathology-specific vision-language foundation model used to encode text conditions.
- **Insight**: The concept of joint diffusion can be extended to other medical imaging tasks requiring multi-modal aligned outputs (e.g., joint synthesis of CT images and segmentation labels).

## Rating

- Novelty: ⭐⭐⭐⭐ First to achieve point-conditioned + text-conditioned image-label-distance map three-way joint diffusion synthesis, with an ingenious design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, multiple baseline methods, pathologist blind evaluation, downstream task verification, and complete ablation study.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich illustrations, and well-motivated.
- Value: ⭐⭐⭐⭐ Directly practical for medical image data augmentation; the concept of point conditions can inspire other fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TopoCellGen: Generating Histopathology Cell Topology with a Diffusion Model](../../CVPR2025/medical_imaging/topocellgen_generating_histopathology_cell_topology_with_a_diffusion_model.md)
- [\[AAAI 2026\] CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis](../../AAAI2026/medical_imaging/cocolit_controlnet-conditioned_latent_image_translation_for_mri_to_amyloid_pet_s.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](../../NeurIPS2025/medical_imaging/semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [\[ECCV 2024\] RadEdit: Stress-Testing Biomedical Vision Models via Diffusion Image Editing](radedit_stress-testing_biomedical_vision_models_via_diffusion_image_editing.md)
- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](../../CVPR2025/medical_imaging/noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)

</div>

<!-- RELATED:END -->
