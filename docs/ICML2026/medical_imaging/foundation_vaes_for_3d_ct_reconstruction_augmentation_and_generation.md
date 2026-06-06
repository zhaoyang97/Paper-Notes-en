---
title: >-
  [Paper Note] Foundation VAEs for 3D CT Reconstruction, Augmentation, and Generation
description: >-
  [ICML 2026][Medical Imaging][Foundation VAE] This paper demonstrates a counter-intuitive yet practical finding—Foundation VAEs pretrained on natural images/videos can serve as a unified interface to simultaneously suppor…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Foundation VAE"
  - "CT Reconstruction"
  - "CT Data Augmentation"
  - "Conditional Latent Diffusion"
  - "Zero-shot Medical Transfer"
date: 2026-05-08
content_hash: 0740f31fc7f76558
---

# Foundation VAEs for 3D CT Reconstruction, Augmentation, and Generation

**Conference**: ICML 2026  
**arXiv**: [2605.30893](https://arxiv.org/abs/2605.30893)  
**Code**: https://github.com/qic999/Foundation-VAE  
**Area**: Medical Imaging / 3D CT / Foundation Model Transfer  
**Keywords**: Foundation VAE, CT Reconstruction, CT Data Augmentation, Conditional Latent Diffusion, Zero-shot Medical Transfer

## TL;DR
This paper demonstrates a counter-intuitive yet practical finding—Foundation VAEs pretrained on natural images/videos can serve as a unified interface to simultaneously support CT reconstruction, augmentation, and generation without any medical fine-tuning. The reconstruction behaves as denoising without shifting boundaries; thus, reconstructed maps can serve as denoising augmentations (pancreatic / lung tumor NSD +3.9%), while the latent space can host CT conditional diffusion generation (FVD −3.9%, CT-CLIP +36.2%, multi-disease fidelity AUC +2.76%).

## Background & Motivation

**Background**: VAEs are the standard interface for contemporary generative models and large-voxel 3D representations—compressing high-resolution CT into a compact latent space significantly improves the efficiency of diffusion/segmentation. The mainstream approach is to train CT-specific VAEs: MedVAE uses self-training, and MAISI utilizes 37,243 CT volumes + 8 V100 GPUs for 300 epochs + multi-stage patch cropping.

**Limitations of Prior Work**: Medical VAE training is expensive, sensitive to scanner/protocol/disease distributions, and prone to overfitting. MedVAE exhibits reconstruction collapse on the MSD dataset (Lung PSNR 20.34, SSIM 0.52; Pancreas PSNR 18.78, SSIM 0.33), while MAISI incurs extreme training costs. Each new dataset requires retraining or re-tuning.

**Key Challenge**: The CT representation stage is widely regarded as necessitating a "medical-exclusive" approach—however, training exclusive VAEs is both costly and poorly generalized. Can the "medical-exclusive" assumption be bypassed?

**Goal**: To test a transfer hypothesis—whether Foundation VAEs pretrained on natural images/videos can serve as a universal interface for CT (reconstruction + augmentation + generation) without any medical fine-tuning.

**Key Insight**: It is observed that the reconstruction error of Foundation VAEs on CT is concentrated on high-frequency noise and scanner artifacts, while tissue/lesion boundaries remain almost perfectly aligned (Fig 2). This implies that the encoder-decoder acts as a "boundary-preserving denoiser" on CT data—exactly what downstream tasks like segmentation/detection require.

**Core Idea**: Treat Foundation VAEs as a unified interface for three CT tasks: (1) Reconstruction (denoising); (2) Using denoised reconstructions as additional views for segmentation training (augmentation); (3) Training conditional latent diffusion for CT generation within the same frozen latent space.

## Method

### Overall Architecture

The three tasks share a single frozen Foundation VAE (pretrained on natural images/videos, e.g., WAN2.1 / VideoVAE+ / IVVAE):

**Task 1: CT Reconstruction**—CT volume $\to$ $E$ (frozen encoder) $\to$ latent representation $z$ $\to$ $D$ (frozen decoder) $\to$ reconstructed CT volume $\hat{x}$. $\hat{x} \approx T(x)$, where $T$ is a boundary-preserving denoising operator.

**Task 2: CT Augmentation**—The segmentation model is trained simultaneously on $\{x\}$ and $\{\hat{x}\}$, treating $\hat{x}$ as a "training view with clearer boundaries"; downstream segmentation (pancreatic tumor, lung tumor) NSD (surface-based metric) is improved.

**Task 3: CT Generation**—A conditional latent diffusion model is trained within the frozen $E$ latent space; conditions include organ segmentation masks (spatial constraints) + radiology reports (semantic constraints); a lightweight 3D consistency module is added to ensure anatomical consistency across axial slices.

### Key Designs

1. **Foundation VAE as a Boundary-Preserving Denoiser**:
    - **Function**: Transforms CT reconstruction into a preprocessing step with inherent denoising effects.
    - **Mechanism**: Observations show that the reconstruction error of Foundation VAEs on CT consists of "high-frequency grains + slight streak artifacts," with almost no shift in organ/lesion boundaries (voxel-wise error map in Fig 2). Quantitatively, PSNR > 30, SSIM > 0.76 (Lung), and PSNR > 39, SSIM > 0.94 (Pancreas / LiTS / KiTS19)—significantly outperforming the collapse of MedVAE (PSNR in the 20s).
    - **Design Motivation**: The "high-level perceptual compression" learned by Foundation VAEs during large-scale natural image training happens to resemble the "boundary-robust denoising" commonly used in the medical domain; this transfer potential had not been systematically verified before.

2. **Reconstruction-based Augmentation**:
    - **Function**: Uses Foundation VAE reconstructions as additional views for segmentation training.
    - **Mechanism**: Traditional data augmentation uses geometric/photometric perturbations; Ours uses Foundation VAE reconstruction (with inherent noise suppression). The segmentation model is jointly trained on $(x, y)$ and $(\hat{x}, y)$ pairs, equivalent to letting the network learn to provide consistent segmentations for both "original + denoised" inputs.
    - **Design Motivation**: Boundary preservation + noise suppression make boundary neighborhoods clearer rather than blurred; the NSD (surface distance metric) particularly benefits (pancreatic/lung tumor +3.9%). This is a free inductive bias with almost no additional computational cost.

3. **Frozen Latent Space + Conditional Latent Diffusion + 3D Consistency Module**:
    - **Function**: Trains a CT conditional generative model within the same Foundation VAE latent space.
    - **Mechanism**: The diffusion model is trained in the $z$ space (input noisy $z_t$, predict $\epsilon$), with conditions including organ masks (spatial) + radiology reports (semantic). A 3D consistency module (lightweight cross-axial attention) is added to maintain anatomical coherence across slices.
    - **Design Motivation**: Traditional CT generation relies on training exclusive latent spaces (MedVAE / MAISI), which is computationally expensive and generalizes poorly. Using the Foundation VAE latent space reuses large-scale pretrained visual priors, so the diffusion only needs to learn the mapping from conditions to $z$. The 3D consistency module compensates for the inter-axial drift of 2D-trained VAEs in 3D space.

## Key Experimental Results

### Task 1: CT Reconstruction (Off-the-shelf Foundation VAE without medical fine-tuning)

| Model | Lung PSNR↑ | Lung SSIM↑ | Lung MSE↓ | Pancreas PSNR↑ | Pancreas SSIM↑ |
|------|----------|----------|---------|--------------|--------------|
| MedVAE (Medical Training) | 20.34 | 0.52 | 600+ | 18.78 | 0.33 |
| MAISI (Medical Training) | 34.5 | 0.89 | – | 38.2 | 0.92 |
| WAN2.1 (Natural VAE) | 30.93 | 0.76 | 77.97 | 39.18 | 0.94 |
| WAN2.2 | 30.93 | 0.76 | 77.97 | 39.06 | 0.95 |
| VideoVAE+ | 30.94 | 0.77 | 80.43 | 40.12 | 0.95 |
| **IVVAE** | **31.78** | **0.79** | **64.39** | **40.43** | **0.96** |

Zero-shot Foundation VAEs completely dominate MedVAE and approach or exceed the expensive MAISI.

### Task 2: CT Reconstruction $\to$ Segmentation Augmentation

| Training Data | Task | Dice↑ | **NSD↑** |
|--------|------|------|------|
| Real CT | Lung tumor | 60.2 | 50.7 |
| Real + IVVAE Recon | Lung tumor | 60.5 | **54.3** (+3.6) |
| Real CT | Pancreatic tumor | 51.4 | 42.5 |
| Real + IVVAE Recon | Pancreatic tumor | 51.8 | **46.7** (+4.2) |

NSD (surface-based metric) increased by an average of 3.9%, verifying the boundary-preserving denoising hypothesis; the slight increase in Dice indicates no deterioration in regional overlap.

### Task 3: CT Conditional Generation

| Method | FVD↓ | CT-CLIP↑ | Multi-disease AUC↑ (18 classes) |
|------|------|---------|-------------|
| MedVAE + diffusion | 320 | 0.61 | 67.3 |
| MAISI + diffusion | 305 | 0.71 | 71.2 |
| **Foundation VAE + diffusion** | **293** | **0.97** | **74.0** (+2.76) |

FVD −3.9% (vs MAISI), CT-CLIP +36.2%, and multi-disease fidelity +2.76 AUC.

### Key Findings
- **Foundation VAEs are not just usable, but more robust**: While MedVAE collapses on MSD, Foundation VAEs remain stable, indicating that visual representations learned from large-scale natural image pre-training are more robust to distribution shifts.
- **Reconstruction error is noise, not boundary shift**: Voxel-wise error maps concentrate on high-frequency noise, verifying the "boundary-preserving denoising" hypothesis—the foundation for both augmentation and generation.
- **3D consistency module is necessary**: Without it, anatomical structures drift between slices (qualitative cases provided in the paper).
- **Cross-dataset generalization**: Validated across four CT datasets: MSD Lung/Pancreas, LiTS, and KiTS19, showing independence from specific distributions.

## Highlights & Insights
- **Counter-intuitive discovery that "Foundation VAE is Medical VAE"**: This challenges the common assumption that medical imaging must have exclusive representations, saving massive computational and engineering costs for medical AI.
- **Structural analysis of reconstruction error (boundary vs noise)**: Qualitative and quantitative proof that errors are noise rather than boundary shifts is the key insight of this paper, worthy of reuse in future work.
- **Unified interface for three tasks**: Historically, CT reconstruction, augmentation, and generation used different VAEs/latent spaces; a unified interface allows them to share improvements (e.g., future stronger Foundation VAEs will automatically enhance all three tasks).
- **Engineering-friendly for training and inference**: All VAEs are frozen; downstream only requires training the segmentation head or diffusion model. Deployment only requires one shared backbone.

## Limitations & Future Work
- Only CT is validated; other modalities like MRI, PET, or Ultrasound—which have different noise characteristics and boundary structures—have not been tested for direct transfer.
- Only segmentation augmentation is evaluated; other downstream tasks like classification, registration, or detection remain untested.
- Foundation VAEs are still 2D/Temporal-VAEs; 3D consistency relies on an ad-hoc module. It remains unknown if a pure 3D Foundation VAE would perform better.
- No quantitative comparison between "reconstruction denoising vs classical denoising filters"; traditional methods might be better for certain tasks.
- In 18-class disease generation, the quality for certain rare diseases (e.g., specific congenital malformations) is not reported separately.

## Related Work & Insights
- **vs MedVAE / MAISI (Medical Exclusive VAEs)**: Those are expensive and generalize poorly; Foundation VAEs require no training and are more robust.
- **vs Traditional Data Augmentation**: Geometric/photometric perturbations are hand-crafted; reconstruction augmentation is learned and boundary-preserving.
- **vs CT Generation Exclusive Latent Spaces**: Foundation VAE latent spaces reuse natural image priors, reducing the learning burden for diffusion.
- **Insight**: The paradigm of whether medical AI "must be exclusively trained" on a large scale may need to be re-examined, especially as Foundation models become increasingly powerful. This mode of "frozen natural Foundation + task-level lightweight adaptation" could be extended to other high-cost medical subfields.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Using Foundation VAEs as zero-shot Medical VAEs is a genuine paradigm challenge.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 4 datasets for reconstruction + 2 tasks for segmentation augmentation + 18-disease generation + comparison of multiple backbones.
- **Writing Quality**: ⭐⭐⭐⭐ Clear parallel description of the three tasks; the "boundary-preserving denoising" insight is well-explained. The voxel error map in Fig 2 is crucial evidence.
- **Value**: ⭐⭐⭐⭐⭐ Saves global medical imaging AI communities massive training costs; if the conclusions continue to be validated, it will influence the design of numerous future projects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CT-FineBench: A Diagnostic Fidelity Benchmark for Fine-Grained Evaluation of CT Report Generation](../../ACL2026/medical_imaging/ct-finebench_a_diagnostic_fidelity_benchmark_for_fine-grained_evaluation_of_ct_r.md)
- [\[AAAI 2026\] GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation](../../AAAI2026/medical_imaging/guidegen_a_text-guided_framework_for_paired_full-torso_anatomy_and_ct_volume_gen.md)
- [\[NeurIPS 2025\] Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation](../../NeurIPS2025/medical_imaging/toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](../../ACL2026/medical_imaging/march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)
- [\[NeurIPS 2025\] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface](../../NeurIPS2025/medical_imaging/surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)

</div>

<!-- RELATED:END -->
