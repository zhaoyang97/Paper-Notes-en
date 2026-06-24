---
title: >-
  [Paper Note] Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis
description: >-
  [CVPR 2025][Medical Imaging][Counterfactual Image Generation] This paper proposes Latent Drifting (LD), which introduces a scalar drift parameter $\delta$ into both the forward and reverse processes of diffusion models to bridge the gap between pre-trained natural image models and target medical image distributions, significantly improving medical image generation and counterfactual image synthesis across various fine-tuning strategies.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Counterfactual Image Generation"
  - "Diffusion Model Fine-Tuning"
  - "Latent Drifting"
  - "Medical Image Synthesis"
  - "Distribution Shift"
date: 2026-05-08
content_hash: c9f9da5f6a995f8c
---

# Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis

**Conference**: CVPR 2025  
**arXiv**: [2412.20651](https://arxiv.org/abs/2412.20651)  
**Code**: [https://latentdrifting.github.io/](https://latentdrifting.github.io/) (Project Page)  
**Area**: Medical Imaging / Diffusion Models  
**Keywords**: Counterfactual Image Generation, Diffusion Model Fine-Tuning, Latent Drifting, Medical Image Synthesis, Distribution Shift

## TL;DR
This paper proposes Latent Drifting (LD), which introduces a scalar drift parameter $\delta$ into both the forward and reverse processes of diffusion models to bridge the gap between pre-trained natural image models and target medical image distributions, significantly improving medical image generation and counterfactual image synthesis across various fine-tuning strategies.

## Background & Motivation

1. **Background**: Pre-trained diffusion models (such as Stable Diffusion) demonstrate outstanding performance in natural image generation, and the medical domain aspires to leverage the powerful generative capabilities of these models. Existing fine-tuning methods (Textual Inversion, DreamBooth, Custom Diffusion) allow the introduction of new concepts into the model with few-shot samples.
2. **Limitations of Prior Work**: There is a massive distribution discrepancy between medical images and natural images (e.g., the background of a brain MRI must be completely black, and bony structures must maintain their shapes). Directly fine-tuning pre-trained models struggles to adapt to this distribution shift. A small number of medical samples cannot effectively adjust the natural image distribution learned by the model. Meanwhile, training medical diffusion models from scratch is hindered by constraints such as data privacy, cost, and rare diseases.
3. **Key Challenge**: The latent space noise distribution of pre-trained models, $z_T \sim \mathcal{N}(0, I)$, is designed for natural images, whereas the optimal sampling distribution for medical images may be shifted from $\mathcal{N}(0, I)$. Fine-tuning only adjusts the model parameters $\theta$ but never alters the latent space distribution.
4. **Goal**: (1) How to efficiently adapt pre-trained diffusion models to the medical image domain; (2) how to achieve high-quality medical counterfactual image synthesis (e.g., disease addition/removal, age progression, gender conversion).
5. **Key Insight**: Treating the final latent variable $z_T$ as another conditioning factor rather than a fixed assumption, and modifying the mean of the latent space via a simple scalar drift $\delta$ to match the target distribution.
6. **Core Idea**: Adding a global drift $\delta$ to the mean at each timestep of the diffusion process to "drift" the latent space distribution from the natural image domain to the medical image domain.

## Method

### Overall Architecture
Latent Drifting is a general plug-and-play method that can be combined with any diffusion model fine-tuning scheme. Given a pre-trained Stable Diffusion model and a target medical dataset, LD modifies the distribution of both forward and reverse processes during fine-tuning, and modifies the reverse process distribution during inference. The method formulates counterfactual image generation as a min-max optimization problem, maximizing the desired outcome fidelity while maintaining similarity with the original image (counterfactual fidelity).

### Key Designs

1. **Latent Drifting Mechanism**:

    - **Function**: Modifying the latent space distribution of the diffusion process to match the target medical image domain by introducing a scalar drift $\delta$.
    - **Mechanism**: Adding a drift term in the transition kernel of the reverse process: $p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t) + \delta, \Sigma_\theta(x_t, t))$. Where $\delta$ is a signed scalar, and the forward diffusion $z_T$ is drifted accordingly. The optimal $\delta$ value is found via grid search (scanning $\delta$ from -0.2 to 0.2) to minimize the L1 distance between the generated distribution $\mathcal{D}_\theta$ and the target distribution $\mathcal{D}_{GT}$. Empirically, $\delta=0.1$ yields the best results on brain MRIs.
    - **Design Motivation**: Traditional fine-tuning assumes that updating the parameters $\theta$ is sufficient to capture the distribution shift. However, the latent space distribution $\mathcal{N}(\mu, \sigma)$ is never adjusted. Consequently, without LD, the latent space distribution after fine-tuning exhibits high variance and instability. With LD, the distribution reaches a stable point and becomes more robust to distribution shifts.

2. **Unified Framework for Counterfactual Generation**:

    - **Function**: Formulating medical counterfactual image generation uniformly as a constrained details optimization problem.
    - **Mechanism**: The objective function is defined as $L(x, x', y', \lambda) = \min_{\ell_o}[\lambda \cdot \ell_o(\hat{f}(x'), y')] + \min_{\ell_{in}}[\ell_{in}(x, x')]$. Here, $\ell_{in}$ ensures the similarity between the counterfactual image $x'$ and the original image $x$, while $\ell_o$ guarantees that the counterfactual outcome aligns with the target label $y'$. The two act as mutual constraints: $\ell_{in} \propto 1/\ell_o$. When $\lambda=0$, it degenerates to standard fine-tuning ($z=z'$); when $\lambda>0$, the drift $\delta$ of LD is introduced to enhance conditional control.
    - **Design Motivation**: Counterfactual image generation is inherently a trade-off between "altering target features" and "preserving original features". This optimization framework naturally integrates LD into conditional control.

3. **Combination with Various Fine-Tuning Schemes**:

    - **Function**: Demonstrating that LD, as a general plug-and-play component, can adapt to different fine-tuning strategies.
    - **Mechanism**: Combining LD with four fine-tuning methods respectively: (1) Textual Inversion—fine-tuning only the embedding space of the text encoder; (2) DreamBooth—fine-tuning the denoising U-Net with a class-prior preservation loss; (3) Custom Diffusion—fine-tuning only the weights of cross-attention layers in the U-Net; (4) Basic FT—fine-tuning the entire denoising U-Net. Each method simply adds the $\delta$ drift during the diffusion process. For image-to-image counterfactual generation, it is also integrated with Pix2Pix Zero and InstructPix2Pix.
    - **Design Motivation**: Modification via LD occurs at the level of the diffusion process (altering the mean) and is orthogonal to the model parameter fine-tuning method. Therefore, it can be seamlessly embedded into any fine-tuning scheme without modifying its internal mechanisms.

### Loss & Training
The foundational training loss is the standard denoising objective $\mathbb{E}_{x,c,\epsilon,t}[w_t\|\hat{x}_\theta(\alpha_t x + \sigma_t \epsilon, c) - x\|_2^2]$, on top of which LD only modifies the sampling distribution. Using the pre-trained SD-v1.4 model, $\delta$ is determined through a grid search in the range of [-0.2, 0.2], of which the L1 normalized distance is used as the evaluation metric. Text-to-image is evaluated with 200 samples, and image-to-image is evaluated using a longitudinal dataset.

## Key Experimental Results

### Main Results

| Fine-Tuning Method | LD | FID (Brain MR)↓ | KID (Brain MR)↓ | AUC (Brain MR)↑ | FID (Chest X-ray)↓ | AUC (Chest X-ray)↑ |
|---------|-----|-----------|-----------|-----------|-----------|-----------|
| SD + Basic FT | ✗ | 92.13 | 0.071 | 0.704 | 112 | 0.672 |
| SD + Basic FT | ✓ | **49.68** | **0.035** | **0.724** | **84** | **0.746** |
| Textual Inversion | ✗ | 120.63 | 0.098 | 0.600 | 171.77 | 0.600 |
| Textual Inversion | ✓ | 67.56 | 0.065 | 0.670 | 133.18 | 0.640 |
| DreamBooth | ✗ | 130.92 | 0.125 | 0.500 | 188 | 0.567 |
| DreamBooth | ✓ | 92.37 | 0.099 | 0.512 | 177 | 0.582 |
| Real + Synthetic | ✓ | - | - | **0.883** | - | **0.892** |

LD consistently brings significant improvements across all fine-tuning methods. Basic FT + LD reduces the FID on Brain MR from 92.13 to 49.68 (a 46% reduction), and the classifier AUC trained on synthesized + real data even exceeds that of purely real data (0.883 vs. 0.870).

### Ablation Study

| Configuration | FID (aging)↓ | SSIM↑ | LPIPS↓ | PSNR↑ |
|------|-------------|-------|--------|-------|
| InstructPix2Pix (Binned) + SD + Basic FT + LD | 15.39 | 0.74 | 0.13 | 32.77 |
| InstructPix2Pix (Word) + SD + Basic FT + LD | **15.25** | 0.75 | 0.13 | 32.78 |
| InstructPix2Pix (Numerical) + SD + Basic FT + LD | 15.37 | **0.76** | **0.12** | **32.83** |
| InstructPix2Pix + SD + CD + LD (Numerical) | 24.05 | 0.32 | 0.23 | 30.70 |

Controlled experiments on prompt formats show that the simple combination of "Diverse + Patient Info" performs the best (FID 51.35, KID 0.0351), and numerical age conditioning is comprehensively optimal in image-to-image tasks.

### Key Findings
- **LD is consistently effective across all fine-tuning schemes**: Whether only tuning text embeddings (Textual Inversion) or tuning the U-Net (Basic FT), LD dramatically lowers FID/KID. The best performance is achieved by Basic FT + LD.
- **Synthetic data augmentation surpasses real data**: A classifier trained on 50% LD synthesized + 50% real data achieves an AUC superior to that trained on 100% real data (Brain MR: 0.883 vs. 0.870), validating the practical utility of synthetic data.
- **Obvious visual improvements**: Adding LD turns the brain MRI background from gray noise to completely black, makes brain structures more realistic, and sharpens the boundary between white and gray matter.
- Prompts containing patient information (age, gender, diagnosis) significantly outperform generic prompts.
- The optimal value of $\delta$ is within the range of 0.05-0.1 and is relatively stable across different fine-tuning methods.

## Highlights & Insights
- **Extremely simple yet effective**: A single scalar parameter $\delta$ is sufficient to bridge the distribution gap between natural and medical images, with virtually zero implementation cost. This finding reveals that the latent space distribution is an overlooked critical degree of freedom in diffusion models.
- **Method-agnosticism**: As a plug-and-play component, it can be embedded into any fine-tuning scheme and remains effective. This improvement mechanism, being orthogonal to the model architecture, is highly elegant and can be directly applied to future emerging fine-tuning methods.
- **Unified framework for counterfactual generation**: It unifies various medical scenarios such as disease addition/removal, age progression, and gender conversion into a single counterfactual optimization framework. Understanding conditional generation from a min-max perspective offers theoretical contributions to the field.
- **Evidence for synthetic data augmentation**: It successfully demonstrates that synthetic data generated by LD can serve as a data augmentation method to improve downstream classification performance, providing a feasible path for medical AI under data scarcity.

## Limitations & Future Work
- **Determination of $\delta$**: $\delta$ is currently determined through grid search; a new target domain requires searching again. Automated estimation of $\delta$ based on the distribution discrepancy between the source and target domains could be considered.
- **Limitations of the global scalar**: $\delta$ is a global shift that is identical across all channels, whereas different spatial regions or channels might require distinct shift amounts. Spatially-adaptive or channel-adaptive LD can be explored.
- **2D slice processing**: Experiments are only conducted on 2D brain MRI slices, leaving 3D volumetric data unaddressed. Extending to 3D diffusion models requires validating the efficacy of LD in higher-dimensional spaces.
- **Difficulty in counterfactual evaluation**: There is a lack of real counterfactual ground truth (e.g., "what would this person's MRI look like if they developed Alzheimer's disease"). Evaluation mainly relies on distributional metrics like FID/KID and downstream classification AUC.
- Future work could attempt to combine LD with conditional control methods like ControlNet to achieve finer-grained medical image editing.

## Related Work & Insights
- **vs DreamBooth**: DreamBooth fine-tunes the U-Net via class-prior preservation loss to introduce new concepts, but suffers from an FID as high as 130.92 on brain MRIs. Adding LD reduces it to 92.37, indicating that merely fine-tuning parameters is insufficient; the latent space distribution also needs adjustment.
- **vs Textual Inversion**: TI is the most lightweight approach as it only tunes text embeddings, but it lacks sufficient understanding of medical images. LD reduces its FID from 120.63 to 67.56 and improves the AUC from 0.600 to 0.670.
- **vs Medical diffusion models trained from scratch (e.g., Khader et al., Pinaya et al.)**: These methods require heavy medical data for training, while LD leverages the prior knowledge of pre-trained models and only needs few-shot fine-tuning.
- The concept of LD can be generalized to other domain adaptation scenarios that exhibit distribution shifts, such as remote sensing imagery and industrial inspection.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of latent space shifting is elegant and unique; treating the latent variable as a tunable condition provides a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple fine-tuning methods, several medical datasets, and various generation tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations and rich visualizations.
- Value: ⭐⭐⭐⭐ A simple yet effective plug-and-play method with direct practical value for medical image synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)
- [\[CVPR 2025\] SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation](sealion_semantic_part-aware_latent_point_diffusion_models_for_3d_generation.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](../../NeurIPS2025/medical_imaging/semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [\[CVPR 2025\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)
- [\[CVPR 2025\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation?](are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)

</div>

<!-- RELATED:END -->
