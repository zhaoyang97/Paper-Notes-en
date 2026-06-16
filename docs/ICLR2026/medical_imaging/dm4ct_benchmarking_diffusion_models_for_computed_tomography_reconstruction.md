---
title: >-
  [Paper Note] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction
description: >-
  [ICLR 2026][Medical Imaging][CT reconstruction] DM4CT is proposed as the first systematic benchmark for diffusion-based CT reconstruction…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "CT reconstruction"
  - "diffusion models"
  - "benchmark"
  - "inverse problems"
  - "sparse-view reconstruction"
date: 2026-05-08
content_hash: 4c5ce25da36e8820
---

# DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction

**Conference**: ICLR 2026
**arXiv**: [2602.18589](https://arxiv.org/abs/2602.18589)  
**Code**: [Available](https://github.com/DM4CT/DM4CT)  
**Area**: Medical Imaging
**Keywords**: CT reconstruction, diffusion models, benchmark, inverse problems, sparse-view reconstruction

## TL;DR

DM4CT is proposed as the first systematic benchmark for diffusion-based CT reconstruction, encompassing ten diffusion methods and seven baselines evaluated comprehensively across medical, industrial, and synchrotron radiation datasets, revealing both the strengths and limitations of diffusion models in CT reconstruction.

## Background & Motivation

CT reconstruction is a canonical inverse problem: recovering an unknown object from projection measurements. When measurements are sparse or noisy, the problem becomes ill-posed and requires prior knowledge. Prior-based approaches have evolved from classical regularization (TV) to deep learning (supervised learning, DIP) and more recently to diffusion models.

Following their success in image generation, diffusion models have been introduced to inverse problem solving. However, CT imaging poses unique challenges—**correlated noise, artifact structures, system geometry dependence, and value-range mismatch**—making direct application significantly harder than natural image generation. Yet a unified benchmark for systematically evaluating diffusion-based approaches has been lacking.

**Core Contribution**: Rather than proposing a new algorithm, this work constructs the first systematic benchmark to answer the question: "How well do diffusion models actually perform in CT?"

## Method

### Overall Architecture

DM4CT organizes diffusion methods from a Bayesian perspective: the posterior $p(\boldsymbol{x}|\boldsymbol{y}) \propto p(\boldsymbol{x})p(\boldsymbol{y}|\boldsymbol{x})$ is pursued by modifying the reverse SDE into a conditional reverse SDE, with the key challenge being how to approximate the measurement likelihood term $\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{y}|\boldsymbol{x}_t)$.

### Key Designs: Unified Taxonomy

Ten diffusion methods are categorized into five classes based on data consistency and prior knowledge strategies:

**1. Data Consistency Gradient Guidance (DC-grad)**: After each denoising step, a data fidelity gradient $\boldsymbol{g}_t = \nabla_{\boldsymbol{x}_t}\mathcal{L}(\boldsymbol{A}\hat{\boldsymbol{x}}_0 - \boldsymbol{y})$ is computed and applied with step size $\eta$ to control guidance strength. Representatives: DPS, PSLD, Reddiff, etc.

**2. Data Consistency Optimization Step (DC-step)**: A full data consistency optimization $\boldsymbol{x}_t^* = \arg\min \mathcal{L}(\boldsymbol{A}\boldsymbol{x}_t - \boldsymbol{y})$ is inserted between denoising iterations. Representatives: ReSample, DMPlug.

**3. Plug-and-Play (PnP)**: Decouples data fidelity and prior, alternating between a data consistency sub-problem and unconditional denoising steps. Representative: DMPlug.

**4. Pseudo-Inverse Guidance**: Uses pseudo-inverse reconstruction (FBP/SIRT approximation) to transfer information between measurement and image spaces. Representatives: MCG, PGDM.

**5. Variational Bayes**: Approximates the posterior with a parameterized distribution, eliminating the need to sample along diffusion trajectories. Representatives: Reddiff, HybridReg.

### Datasets and Configurations

Three dataset categories cover distinct application scenarios:

- **Medical CT**: 2016 Low Dose CT Challenge (9 training volumes + 1 test volume, 512×512)
- **Industrial CT**: LoDoInd (tubular specimens with 15 material types, 3000+500 slices)
- **Synchrotron Radiation CT** (newly acquired): Two rock samples scanned at a high-energy synchrotron facility, 768×768 high resolution

Five simulation configurations systematically test robustness: 40-view noiseless / 20-view low-noise / 80-view high-noise / 80-view noise + ring artifacts / 40-view limited-angle.

### Baseline Methods

Seven strong baselines spanning diverse methodological paradigms:

- Classical: FBP, SIRT
- Neural network priors: DIP, INR
- Gaussian splatting: R2Gaussian
- Iterative reconstruction: FISTA-SBTV, ADMM-PDTV
- Supervised learning: SwinIR

## Key Experimental Results

### Main Results

**Medical Dataset Reconstruction Performance (PSNR/SSIM, selected configurations)**

| Method | Config i (40-view, noiseless) | Config ii (20-view, low-noise) | Config iv (80-view, noise+ring) |
|------|:---:|:---:|:---:|
| FBP | 26.98/0.69 | 9.89/0.03 | 14.50/0.13 |
| SIRT | 30.40/0.80 | 26.23/0.47 | 25.86/0.40 |
| SwinIR (supervised) | 32.45/0.88 | **29.92/0.83** | **30.79/0.85** |
| DDS (best diffusion) | **31.43/0.84** | - | - |
| ReSample | 32.03/0.85 | 27.92/0.73 | 29.70/0.76 |
| INR | 33.21/0.86 | 26.15/0.76 | 29.50/0.74 |

**Synchrotron Real Data (PSNR/SSIM)**

| Method | 200 projections | 100 projections | 60 projections |
|------|:---:|:---:|:---:|
| SwinIR | **33.75/0.76** | **33.05/0.73** | **32.41/0.70** |
| Reddiff | 28.43/0.56 | 28.24/0.54 | 28.06/0.51 |
| DDS | 28.36/0.55 | 28.10/0.51 | 27.90/0.49 |
| SIRT | 28.16/0.56 | 28.06/0.54 | 27.92/0.52 |

### Ablation Study

**Prior vs. Data Consistency Trade-off**: Using DPS as an example, an excessively small step size $\eta$ leads to prior dominance (blurring), while an excessively large $\eta$ causes measurement noise dominance (collapse). The optimal $\eta$ requires careful tuning.

**Pixel-space vs. Latent-space Diffusion**:
- Latent-space (PSLD): Gradients must propagate through the VQ-VAE decoder, producing discontinuous artifacts even in noiseless conditions.
- Optimization-step methods (ReSample) can correct discontinuities but overfit to measurements under noisy conditions.

**Null-space Analysis**: DC-grad (DPS) permits more null-space content; DC-step (ReSample) enforces stricter constraints; pseudo-inverse (PGDM) lies between the two.

### Key Findings

1. **No single diffusion method is universally optimal**; performance varies substantially across datasets and configurations.
2. Diffusion models generally outperform classical/MBIR methods but typically fall short of fully supervised SwinIR.
3. Although visually plausible, fine details recovered by diffusion models may deviate from ground truth, yielding lower metrics than the smoother reconstructions of INR/SwinIR.
4. Performance on real data is consistently lower than on simulated data, exposing issues of training data quality and distribution shift.
5. Pixel-space diffusion is generally more memory- and time-efficient than latent-space diffusion.

## Highlights & Insights

- **First systematic CT diffusion benchmark**: unified codebase (diffusers), fair comparisons, open-source code and data.
- The **unified taxonomy** clearly delineates design choices and trade-offs across methods.
- The **real synchrotron radiation dataset** addresses the gap in existing evaluations that lack real acquisition data.
- **Practical insights**: the benchmark exposes real deployment challenges including value-range mismatch, limited training data, and geometric complexity.
- **Null-space analysis** offers a novel perspective for understanding different data consistency strategies.

## Limitations & Future Work

- Evaluation is limited to 2D slice reconstruction; 3D reconstruction (helical/cone-beam geometry) is not addressed.
- Flow-based methods (e.g., FlowDPS), an emerging direction, are not included.
- Clinical relevance assessment is insufficient; downstream task evaluations such as segmentation or radiologist scoring are absent.
- Diffusion model training is computationally expensive, and latent-space models incur higher total training time.
- Autoencoders pre-trained on natural images may not be well-suited to CT data.
- Generalization across devices and acquisition protocols is not tested.

## Related Work & Insights

- Combining the DC loss concept from DIP with diffusion models is a promising direction (cf. the DC loss paper).
- Hybrid approaches integrating INR with diffusion priors may simultaneously achieve structural fidelity and fine detail recovery.
- The uncertainty quantification capability of diffusion models (mean/variance over multiple samples) holds clinical value.
- For sparse-view and high-noise scenarios, learned priors offer the greatest advantage; classical methods remain sufficient in dense/low-noise settings.

## Rating

| Dimension | Score |
|------|:---:|
| Novelty | ★★★★☆ |
| Theoretical Depth | ★★★☆☆ |
| Experimental Thoroughness | ★★★★★ |
| Value | ★★★★★ |
| Writing Quality | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)
- [\[NeurIPS 2025\] Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?](../../NeurIPS2025/medical_imaging/are_pixel-wise_metrics_reliable_for_sparse-view_computed_tomography_reconstructi.md)
- [\[ICLR 2026\] Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity](improving_2d_diffusion_models_for_3d_medical_imaging_with_inter-slice_consistent.md)
- [\[ICLR 2026\] Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer](brain-it_image_reconstruction_from_fmri_via_brain-interaction_transformer.md)
- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](../../NeurIPS2025/medical_imaging/posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)

</div>

<!-- RELATED:END -->
