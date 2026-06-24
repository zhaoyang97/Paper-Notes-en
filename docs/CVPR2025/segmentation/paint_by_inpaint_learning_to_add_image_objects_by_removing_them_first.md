---
title: >-
  [Paper Note] Paint by Inpaint: Learning to Add Image Objects by Removing Them First
description: >-
  [CVPR 2025][Segmentation][Image Editing] This work proposes the "Paint by Inpaint" framework. Leveraging the key insight that "adding objects is the inverse process of removing them," they construct the PIPE dataset containing approximately 1 million high-quality image pairs through an automated inpainting pipeline. The trained diffusion model achieves state-of-the-art (SOTA) performance on object addition and general editing tasks.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Image Editing"
  - "Object Adding"
  - "Diffusion Models"
  - "Image Inpainting"
  - "Dataset Construction"
date: 2026-05-08
content_hash: d0b71db2f2aca796
---

# Paint by Inpaint: Learning to Add Image Objects by Removing Them First

**Conference**: CVPR 2025  
**arXiv**: [2404.18212](https://arxiv.org/abs/2404.18212)  
**Code**: [Project Page](https://rotsteinnoam.github.io/Paint-by-Inpaint/)  
**Area**: Image Segmentation  
**Keywords**: Image Editing, Object Adding, Diffusion Models, Image Inpainting, Dataset Construction

## TL;DR

This work proposes the "Paint by Inpaint" framework. Leveraging the key insight that "adding objects is the inverse process of removing them," they construct the PIPE dataset containing approximately 1 million high-quality image pairs through an automated inpainting pipeline. The trained diffusion model achieves state-of-the-art (SOTA) performance on object addition and general editing tasks.

## Background & Motivation

Text-instruction-based mask-free image object addition is a highly challenging task that requires an understanding of global context (such as placement, scale, and style). The primary bottleneck of existing methods lies in the **quality of training data**:

1. **InstructPix2Pix (IP2P)**'s data issues: It uses GPT-3 + Prompt-to-Prompt to synthesize datasets, but both the source and target images are synthetic, often suffering from inconsistencies. Despite applying Directional CLIP filtering, its effectiveness remains limited.
2. **MagicBrush**: A semi-synthetic dataset manually created using DALL-E2. While the quality is improved, its scale is heavily bottlenecked by manual annotation costs.
3. **Key Challenge**: Under a mask-free setting, it is virtually impossible to obtain a pair of natural images that differ solely in the edited region.

The core insight of this paper is remarkably elegant: **adding an object (Paint) is essentially the inverse process of removing an object (Inpaint)**. By utilizing existing images and masks from large-scale segmentation datasets, an inpainting model can be used to remove objects and generate "source images." By treating the original images as "target images," a large-scale, high-quality, mask-free object addition dataset can be successfully constructed.

The key advantages of this approach are: (i) the target images are **real natural images** rather than synthetic ones, and (ii) source-target consistency is **inherently guaranteed**, as modifications are strictly confined to the region of the removed object.

## Method

### Overall Architecture

The framework is divided into two stages:
1. **Dataset Construction Stage (PIPE)**: Object removal $\rightarrow$ Filtering $\rightarrow$ Instruction generation $\rightarrow$ Formulation of training triplets.
2. **Model Training Stage**: Training a denoising diffusion editing model based on the SD 1.5 architecture.

### Key Designs

1. **Source-Target Pairs Generation Pipeline**:
    - **Function**: Automatically construct high-quality object addition image pairs from segmentation datasets.
    - **Mechanism**:
        - **Data Source**: Merged COCO + Open Images + LVIS = 889,230 images, covering over 1,400 classes.
        - **Pre-removal Filtering**: Exclude masks that are too large, too small, or on the image boundaries. Filter out abnormal object views (blurred, heavily occluded) using CLIP semantic similarity. Apply morphological dilation to masks to ensure complete coverage.
        - **Object Removal**: Use an SD inpainting model with a positive prompt "a photo of a background" and a negative prompt "an object, a <class>", generating 3 candidates via 10-step denoising.
        - **Post-removal Verification**: CLIP consensus detection (low standard deviation among the 3 candidate embeddings = consistent removal) + multimodal CLIP filtering (low similarity between the edited region and the original class = successful removal) + $\alpha$-blending consistency enhancement + importance filtering.
    - **Design Motivation**: Inpainting models are not natively trained for object removal and might leave remnants or generate new objects. Thus, a multi-layer filtering pipeline is required to guarantee quality.

2. **Object Addition Instruction Generation**:
    - **Function**: Generate natural language editing instructions for each image pair.
    - **Mechanism**: Three strategies are employed—(i) Class-name instructions: "add a <class>"; (ii) VLM-LLM instructions: CogVLM describes the object $\rightarrow$ Mistral-7B converts it to an instruction via 5-shot in-context learning (ICL); (iii) Reference instructions: utilize manual annotation descriptions from the RefCOCO family.
    - **Design Motivation**: Combining multiple strategies yields 1,879,919 instructions, covering both brief and detailed editing scenarios.

3. **Diffusion Editing Model Training**:
    - **Function**: Learn to add objects to an image based on instructions.
    - **Mechanism**: Based on the SD 1.5 architecture, the model is conditioned on both the text instruction $c_T$ and the source image $c_I$. During training, $c_T$, $c_I$, or both are randomly dropped with a 5% probability to support classifier-free guidance during inference.
    - **Design Motivation**: Dual-conditioning CFG allows balancing editing fidelity and source image consistency at inference time.

### Loss & Training

- Standard diffusion denoising loss.
- Probability dropout for both text and image conditions during classifier-free guidance (5% each).
- The consistency-editing trade-off can be controlled by adjusting the image/text guidance scale.

## Key Experimental Results

### Main Results

MagicBrush object addition subset (144 edits):

| Method | L1↓ | CLIP-I↑ | DINO↑ | CMMD↓ |
|------|-----|---------|-------|-------|
| VQGAN-CLIP | .211 | .670 | .507 | .862 |
| SDEdit | .168 | .765 | .572 | .539 |
| IP2P | .100 | .860 | .766 | .363 |
| Hive | .095 | .846 | .782 | .353 |
| **Ours** | **.072** | **.900** | **.852** | **.301** |

PIPE Test Set (750 images):

| Method | L1↓ | CLIP-I↑ | DINO↑ | CMMD↓ |
|------|-----|---------|-------|-------|
| IP2P | .098 | .861 | .753 | .142 |
| Hive | .088 | .849 | .754 | .232 |
| **Ours** | **.057** | **.945** | **.903** | **.060** |

### Ablation Study

Human Evaluation (100 images, 57 evaluators, 1,833 ratings):

| Metric | IP2P | Ours |
|------|------|------|
| Edit Fidelity Preference (Overall%) | 26.4% | **73.6%** |
| Output Quality Preference (Overall%) | 28.5% | **71.5%** |
| Edit Fidelity (Per-image wins) | 28 | **72** |
| Output Quality (Per-image wins) | 31 | **69** |

### Key Findings

1. **Dataset scale and quality are decisive factors**: PIPE's ~1 million real target images comprehensively outperform IP2P's 310k synthetic pairs.
2. **Significant consistency advantage**: The L1 metric is substantially ahead of other methods, demonstrating that consistency outside the edited region is naturally and inherently guaranteed.
3. **Generalization to general editing**: Models trained on a merged dataset of PIPE and IP2P also outperform models trained solely on IP2P under general editing tasks.
4. **Overwhelming preference in human evaluation**: A 73.6% overall preference rate confirms a substantial improvement in generation quality.
5. **Fine-tuning yields further improvements**: Fine-tuning on MagicBrush further reduces the L1 metric from .072 to .067.

## Highlights & Insights

1. **The core insight of "reverse thinking"**: Formulating "addition = inverse of removal" is the most brilliant part of the paper, converting a hard problem into a simpler one via symmetry.
2. **Industrial-grade data pipeline**: Multiple filtering mechanisms (CLIP consensus, multimodal CLIP, consistency enhancement, importance filtering) ensure data quality in a large-scale automated pipeline.
3. **Two-stage VLM-LLM instruction generation**: Decoupling object description and instruction generation successfully avoids VLM hallucinations.
4. **Inherent guarantee of consistency**: Through $\alpha$-blending and mask constraints, source-target consistency is constructively guaranteed.

## Limitations & Future Work

1. **Limited by inpainting model quality**: If the inpainting model fails to remove objects completely or produces anomalies, some flawed examples might bypass the filters.
2. **Uncontrollable object placement**: The model implicitly learns position choices from data; users cannot specify the exact addition location.
3. **Category coverage restricted by segmentation datasets**: Although covering 1,400+ categories, the representation of long-tail classes remains insufficient.
4. **Reliance on the SD 1.5 architecture**: Better performance could potentially be achieved by scaling to more advanced base models.
5. Future work can explore multi-object addition, precise location control, integration with larger VLMs, etc.

## Related Work & Insights

- **vs InstructPix2Pix**: IP2P data are entirely synthetic and suffer from consistency issues; PIPE's target images are real, which inherently ensures consistency.
- **vs MagicBrush**: MagicBrush contains high-quality human-annotated data but is limited in scale (~10K); PIPE automatically generates ~1 million pairs.
- **vs Inst-Inpaint**: Inst-Inpaint also leverages segmentation + inpainting but for object removal; PIPE reverses this process to achieve object addition.
- **vs SmartBrush**: SmartBrush requires mask inputs from users; the model proposed in PIPE does not require any mask inputs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The core insight of "Paint by Inpaint" is extremely elegant, transforming a hard problem into a simple one through symmetry.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three benchmarks + human evaluation + general editing expansion + multi-metric evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed pipeline description, and rich illustrations.
- Value: ⭐⭐⭐⭐⭐ The PIPE dataset itself holds tremendous value, and the framework's methodology provides broad inspiration for the data construction domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] QMamba: On First Exploration of Vision Mamba for Image Quality Assessment](../../ICML2025/segmentation/qmamba_on_first_exploration_of_vision_mamba_for_image_quality_assessment.md)
- [\[NeurIPS 2025\] FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning](../../NeurIPS2025/segmentation/finers_fine-grained_reasoning_and_segmentation_of_small_objects_with_reinforceme.md)
- [\[CVPR 2025\] OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels](overlock_an_overview-first-look-closely-next_convnet_with_context-mixing_dynamic.md)
- [\[ICCV 2025\] Rethinking Detecting Salient and Camouflaged Objects in Unconstrained Scenes](../../ICCV2025/segmentation/rethinking_detecting_salient_and_camouflaged_objects_in_unconstrained_scenes.md)
- [\[CVPR 2025\] The Power of Context: How Multimodality Improves Image Super-Resolution](the_power_of_context_how_multimodality_improves_image_super-resolution.md)

</div>

<!-- RELATED:END -->
