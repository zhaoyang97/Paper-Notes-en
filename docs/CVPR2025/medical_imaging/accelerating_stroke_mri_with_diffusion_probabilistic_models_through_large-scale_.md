---
title: >-
  [Paper Note] Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning
description: >-
  [CVPR2025][Medical Imaging][Diffusion Probabilistic Models] Drawing on the foundation model paradigm, a Diffusion Probabilistic Model (DPM) is pre-trained on large-scale public brain MRI data and then fine-tuned on data from only 20 stroke patients. This workflow enables accelerated MRI reconstruction in data-constrained scenarios. A clinical reader study confirms that the image quality with 2× acceleration is non-inferior to the standard-of-care.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "Diffusion Probabilistic Models"
  - "MRI Reconstruction"
  - "Foundation Models"
  - "Stroke MRI"
  - "Fine-tuning"
date: 2026-05-08
content_hash: e768dc943b4c0079
---

# Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning

**Conference**: CVPR2025  
**arXiv**: [2603.13007](https://arxiv.org/abs/2603.13007)  
**Code**: Not publicly available  
**Area**: Medical Imaging  
**Keywords**: Diffusion Probabilistic Models, MRI Reconstruction, Foundation Models, Stroke MRI, Fine-tuning

## TL;DR

Drawing on the foundation model paradigm, a Diffusion Probabilistic Model (DPM) is pre-trained on large-scale public brain MRI data and then fine-tuned on data from only 20 stroke patients. This workflow enables accelerated MRI reconstruction in data-constrained scenarios. A clinical reader study confirms that the image quality with 2× acceleration is non-inferior to the standard-of-care.

## Background & Motivation

**Clinical demand for stroke MRI**: MRI is more sensitive than CT in detecting ischemic stroke, but long acquisition times and motion sensitivity delay treatment.

**Data scarcity bottleneck**: Fully sampled stroke MRI data is extremely scarce, making it impossible to directly train high-quality reconstruction models using traditional supervised learning methods.

**Limitations of prior work**: Self-supervised methods (e.g., k-space splitting, SPARK) do not require large datasets but fall short of supervised methods in reconstruction quality. End-to-end methods require the training and target datasets to share identical acquisition models.

**Inspiration from foundation models**: The "large-scale pre-training + few-shot fine-tuning" strategy widely used in fields such as LLMs has shown remarkable results but remains under-explored in MRI reconstruction.

**Unique advantages of DPM**: Diffusion models are decoupled from the acquisition forward model, allowing the pre-training data and target data to use different sampling patterns and coil configurations.

**Need for clinical translation**: A method is required that can produce clinically acceptable reconstruction quality with minimal target domain data (~20 cases).

## Method

### Overall Architecture

A two-stage training strategy: (1) Pre-train the DPM on approximately 4,000 multi-contrast brain MRI scans from fastMRI; (2) Fine-tune on the small target-domain dataset with a reduced learning rate.

### Key Designs

- **Score-based DPM**: Approximate the score function $\nabla_{x_t}\log p_t(x)$ using a U-Net architecture, employing EDM training loss and noise scheduling.
- **Contrast-Conditional Embedding**: Assign a one-hot vector for each MRI contrast. This vector is passed through a small fully connected network to generate embeddings, which are injected into each block of the U-Net, enabling the model to adapt to different contrasts.
- **Diffusion Posterior Sampling (DPS)**: Solve ODEs to achieve posterior sampling, combining the data consistency term $\nabla_x \|PFS\tilde{x} - y\|_2^2$ with the learned prior score.
- **Fine-tuning Strategy**: Reduce the learning rate by one order of magnitude ($1\times10^{-5}$), training for only about 2% of the pre-training duration (~650 epochs), while assigning a new one-hot vector to the target contrast.

### Loss & Training

- Both pre-training and fine-tuning use the EDM training loss (denoising score matching).
- During inference, a parameter $\zeta$ in the DPS algorithm is used to balance data consistency and the prior score.

## Key Experimental Results

### fastMRI Controlled Experiments

| Method | FLAIR Training Data Volume | Relative Performance |
|------|-----------------|---------|
| Methods 1-3 (Upper Bound) | 344 cases of FLAIR | Baseline |
| **Method 4 (Ours)** | **20 cases of FLAIR + Pre-training** | **Comparable to upper bound** |
| Method 5 (Only 20 cases) | 20 cases of FLAIR | Significantly inferior to ours |
| Method 6 (Joint Training) | 20 cases + external joint | Inferior to ours |

### Stroke MRI Clinical Reader Study

**Reader 1 (80 cases, 11 years of experience)**:

| Metric | Standard of Care | DPM Accelerated Reconstruction | Significance |
|------|---------|-------------|--------|
| White Matter Depiction | 4.8 | 4.9 | p<0.05 ✓ |
| Gray Matter Depiction | 4.8 | 4.9 | n.s. |
| Ventricle Depiction | 4.8 | 4.9 | p<0.05 ✓ |
| SNR | 4.2 | 4.7 | p<0.05 ✓ |
| Contrast | 4.7 | 4.8 | n.s. |
| Sharpness | 4.7 | 4.8 | p<0.05 ✓ |
| Artifacts | 4.1 | 4.3 | p<0.05 ✓ |
| Overall Quality | 4.5 | 4.8 | p<0.05 ✓ |

**Reader 2 (21 cases, 30 years of experience)**: An opposite trend in ratings was observed where standard of care was significantly superior to accelerated reconstruction on SNR (4.9 vs 4.4), but no significant differences were found in other metrics. Both methods reached clinically acceptable levels (average scores $\ge4.0$). Inter-reader reliability showed fair-to-slight agreement (Cohen's kappa 0.0-0.4).

### Key Findings

- Optimal fine-tuning configuration: Learning rate of $1\times10^{-5}$, with 650 epochs for FLAIR, 1250 epochs for SWI, and 252 epochs for DWI. Either excessive or insufficient fine-tuning leads to degraded performance.
- The optimal data consistency weight $\zeta$ increases as the acceleration factor increases (e.g., optimal $\zeta=2.1$ for SWI).
- Prospective undersampling experiments (healthy volunteers, $R\approx3.75$) achieve comparable image quality to retrospective undersampling results, with both outperforming the L1-wavelet baseline.
- Sequential fine-tuning outperforms joint training: Method 4 (Pre-training + Fine-tuning) is significantly better than Method 6 (Joint Training), demonstrating that phased adaptation is more effective than mixed training.

## Highlights & Insights

1. **High Data Efficiency**: Achieves comparable performance to models trained on 344 cases with only 20 target domain cases.
2. **Rigorous Clinical Validation**: An 80-case double-blind reader study scored by two neuroradiologists, providing high clinical credibility.
3. **Acquisition Decoupling**: DPM does not rely on specific sampling patterns, allowing pre-training and fine-tuning to use different acquisition protocols.
4. **One Method Covers Multiple Sequences**: A single pre-trained model can be successfully fine-tuned for FLAIR, MPRAGE, SWI, and DWI.

## Limitations & Future Work

1. Clinical evaluation is mainly based on retrospective undersampling. Prospective validation is limited to healthy volunteers and has not been tested in real emergency workflows.
2. Posterior sampling inference takes a long time. The speed per slice is much slower than traditional parallel imaging and end-to-end methods, rendering it unsuitable for real-time applications.
3. Inter-reader reliability is only fair-to-slight (Cohen's kappa 0.0-0.4), indicating high subjectivity in scoring.
4. Validation is limited to brain MRI, without expansion to other anatomical regions (e.g., cardiac, knee).
5. Optimal fine-tuning hyperparameters (learning rate, epochs) vary across different contrasts/sequences, requiring individual tuning for each new sequence.
6. The pre-training data is from a single source (fastMRI), leaving generalization across different scanners/field strengths unverified.

## Related Work & Insights

- **Self-supervised MRI Reconstruction**: Scan-specific methods like RAKI and SPARK do not require external data but offer limited reconstruction quality.
- **DPM for MRI**: Score-based diffusion (Song et al.) and DPS (Chung et al.) established diffusion-based reconstruction frameworks.
- **Foundation Model Transfer**: The proposed strategy resembles the pre-training and fine-tuning paradigm in NLP, adapted to the specific constraints of MRI reconstruction.
- **Data Augmentation and Self-Supervised Training**: Methods like Noise2Recon utilize undersampled data for training but still require substantial data volume.

## Rating

- Novelty: ⭐⭐⭐ (The idea is clear and straightforward; the core contribution lies in validation rather than methodological breakthrough)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Controlled experiments + clinical data + reader study + prospective validation, extremely comprehensive)
- Writing Quality: ⭐⭐⭐⭐ (Well-structured, clinically well-motivated)
- Value: ⭐⭐⭐⭐ (Addresses real-world clinical problems with direct translational potential)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](../../AAAI2026/medical_imaging/g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](../../AAAI2026/medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[CVPR 2025\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)
- [\[CVPR 2025\] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)
- [\[CVPR 2025\] Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](multi-resolution_pathology-language_pre-training_model_with_text-guided_visual_r.md)

</div>

<!-- RELATED:END -->
