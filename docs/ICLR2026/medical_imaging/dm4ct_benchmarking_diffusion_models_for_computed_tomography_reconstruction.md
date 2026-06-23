---
title: >-
  [Paper Note] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction
description: >-
  [ICLR 2026][Medical Imaging][Diffusion Model] Proposes DM4CT—the first systematic benchmark for diffusion models in CT reconstruction, covering ten diffusion methods and seven baseline approaches evaluated across medical, industrial, and synchrotron datasets, revealing the strengths and limitations of diffusion models in CT.
tags:
  - ICLR 2026
  - Medical Imaging
  - Diffusion Model
  - benchmark
date: 2026-05-08
content_hash: 9e5faa3f808421a1
---
# DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction

**Conference**: ICLR 2026  
**arXiv**: [2602.18589](https://arxiv.org/abs/2602.18589)  
**Code**: [Available](https://github.com/DM4CT/DM4CT)  
**Area**: Medical Imaging  
**Keywords**: CT reconstruction, Diffusion models, benchmark, inverse problems, sparse-view reconstruction

## TL;DR

Proposes DM4CT—the first systematic benchmark for diffusion models in CT reconstruction, covering ten diffusion methods and seven baseline approaches evaluated across medical, industrial, and synchrotron datasets, revealing the strengths and limitations of diffusion models in CT.

## Background & Motivation

CT reconstruction is a typical inverse problem aimed at recovering an unknown object from projection measurements. When measurements are sparse or noisy, the problem is ill-posed and requires prior knowledge. Prior methods have evolved from classical regularization (TV) to deep learning (supervised learning, DIP), and recently to diffusion models.

While diffusion models have succeeded in image generation, their application to CT imaging faces specific challenges: **correlated noise, artifact structures, system geometry dependence, and value range mismatch**, making direct application much harder than natural image generation. Furthermore, there is a lack of a unified benchmark to systematically evaluate various diffusion approaches.

**Goal**: This work does not propose a new algorithm but builds the first systematic benchmark to answer "how diffusion models actually perform in CT."

## Method

### Overall Architecture

DM4CT itself does not introduce a new algorithm; instead, it places CT reconstruction within a Bayesian framework to horizontally evaluate various diffusion methods. In the posterior $p(\boldsymbol{x}|\boldsymbol{y}) \propto p(\boldsymbol{x})p(\boldsymbol{y}|\boldsymbol{x})$, the diffusion model provides the prior score $\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t)$, and the reverse SDE is transformed into a conditional reverse SDE. The differences between methods essentially lie in how they approximate the difficult measurement condition score $\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{y}|\boldsymbol{x}_t)$. The benchmark provides a unified perspective, controllable data/geometry configurations, and a fair implementation to obtain reproducible answers.

### Key Designs

**1. Unified Taxonomy: Comparing technical choices across ten diffusion methods**

Despite their surface-level diversity, these methods converge into five main categories based on how they approximate the measurement condition score. The most common is **Data Consistency Gradient Guidance (DC-grad)**, where after each denoising step, a data fidelity gradient $\boldsymbol{g}_t = \nabla_{\boldsymbol{x}_t}\mathcal{L}(\boldsymbol{A}\hat{\boldsymbol{x}}_0 - \boldsymbol{y})$ is computed from the current estimate $\hat{\boldsymbol{x}}_0$ to nudge the trajectory, using a step size $\eta$ to adjust guidance strength (e.g., DPS, PSLD). A more rigid approach is the **Data Consistency Optimization Step (DC-step)**, which inserts a complete minimization $\boldsymbol{x}_t^* = \arg\min \mathcal{L}(\boldsymbol{A}\boldsymbol{x}_t - \boldsymbol{y})$ between denoising iterations to fully satisfy measurement constraints (e.g., ReSample). Between these lies **Plug-and-Play (DMPlug)**, which alternates between prior and fidelity solvers; **Pseudo-Inverse Guidance (PGDM, MCG)**, which bridges measurement and image space using pseudo-inverse reconstructions (FBP/SIRT approximations); and **Variational Bayes (Reddiff)**, which approximates the posterior with a parameterized distribution rather than sampling step-by-step along a trajectory.

**2. Three Datasets and Five Simulation Configurations: Covering domain gaps and degradation spectra**

Diverse datasets are used to evaluate domain shifts: Medical CT uses the 2016 Low Dose CT Challenge (9 volumes for training, 1 for testing, 512×512); Industrial CT uses LoDoInd tubular multi-material samples (3000 training, 500 testing slices); and a newly collected **Synchrotron CT** dataset (two rock samples scanned at a high-energy synchrotron facility, 768×768 resolution), the latter filling a gap where existing evaluations rely almost exclusively on simulations. Robustness is tested across five progressive degradation configurations: 40-view noise-free, 20-view low-noise, 80-view high-noise, 80-view noise with ring artifacts, and 40-view limited-angle.

**3. Seven Strong Baselines: Comparing diffusion methods with classical, iterative, and supervised paradigms**

To determine the gain brought by diffusion, the benchmark includes: analytical FBP and algebraic iterative SIRT; neural network priors (DIP) and implicit representations (INR) for unsupervised learning; R2Gaussian for Gaussian splatting; FISTA-SBTV and ADMM-PDTV for regularized iterative reconstruction; and supervised SwinIR as an upper-bound reference. All methods are implemented under a unified `diffusers` framework, sharing the same forward operator and evaluation scripts to ensure fairness.

## Key Experimental Results

### Main Results

**Reconstruction Performance on Medical Dataset (PSNR/SSIM, selected configs)**

| Method | Config i (40v Noise-free) | Config ii (20v Low-noise) | Config iv (80v Noise+Ring) |
|------|:---:|:---:|:---:|
| FBP | 26.98/0.69 | 9.89/0.03 | 14.50/0.13 |
| SIRT | 30.40/0.80 | 26.23/0.47 | 25.86/0.40 |
| SwinIR (Supervised) | 32.45/0.88 | **29.92/0.83** | **30.79/0.85** |
| DDS (Best Diffusion) | **31.43/0.84** | - | - |
| ReSample | 32.03/0.85 | 27.92/0.73 | 29.70/0.76 |
| INR | 33.21/0.86 | 26.15/0.76 | 29.50/0.74 |

**Real Synchrotron Data (PSNR/SSIM)**

| Method | 200 Projections | 100 Projections | 60 Projections |
|------|:---:|:---:|:---:|
| SwinIR | **33.75/0.76** | **33.05/0.73** | **32.41/0.70** |
| Reddiff | 28.43/0.56 | 28.24/0.54 | 28.06/0.51 |
| DDS | 28.36/0.55 | 28.10/0.51 | 27.90/0.49 |
| SIRT | 28.16/0.56 | 28.06/0.54 | 27.92/0.52 |

### Ablation Study

**Trade-off between Prior and Data Consistency**: Using DPS as an example, if the step size $\eta$ is too small, the prior dominates (blurring); if too large, measurement noise dominates (reconstruction collapse). The optimal $\eta$ requires precise tuning.

**Pixel Space vs. Latent Space Diffusion**:
- **Latent Space (PSLD)**: Gradients must propagate through the VQ-VAE decoder, generating discontinuous artifacts even in noise-free conditions.
- Optimization steps (**ReSample**) can fix discontinuities, but may overfit measurements when noise is present.

**Null Space Analysis**: DC-grad (DPS) allows more content in the null space, whereas DC-step (ReSample) imposes stricter constraints, with pseudo-inverse methods (PGDM) falling in between.

### Key Findings

1. **No single diffusion method is globally optimal**; performance varies by dataset and configuration.
2. Diffusion models generally outperform classical/MBIR methods but usually lag behind fully supervised SwinIR.
3. Details recovered by diffusion models, while visually realistic, may deviate from the ground truth, leading to lower metrics compared to the smooth reconstructions of INR/SwinIR.
4. Performance on real data is generally lower than on simulated data, exposing issues with training data quality and distribution shift.
5. Pixel diffusion is often more memory and time-efficient than latent space diffusion for these tasks.

## Highlights & Insights

- **First Systematic CT Diffusion Benchmark**: Unified code framework (`diffusers`), fair comparison, and open-source code/data.
- **Unified Taxonomy** clearly organizes design choices and trade-offs among methods.
- **Real Synchrotron Dataset** addresses the lack of real-world data in existing evaluations.
- **Profound Practical Insights**: Reveals real deployment challenges such as value range mismatch, limited training data, and geometric complexity.
- **Null Space Analysis** provides a new perspective for understanding different data consistency strategies.

## Limitations & Future Work

- Evaluates only 2D slice reconstruction, not 3D (helical/cone-beam geometries are more challenging).
- Does not include flow-based methods (e.g., FlowDPS), which is an emerging direction.
- Lack of clinical relevance evaluation (no downstream tasks like segmentation or radiologist scoring).
- High training costs for diffusion models; latent space models involve longer total training times.
- Autoencoders pre-trained on natural images may not be suitable for CT data.
- Generalization across different devices or protocols remains untested.

## Related Work & Insights

- Combining the DC loss idea from DIP with diffusion models is a promising direction.
- Hybrid methods using INR combined with diffusion priors might achieve both structural fidelity and detail recovery.
- The uncertainty quantification capability of diffusion models (mean/variance from multiple samples) has significant clinical value.
- Learned priors show the greatest advantage in sparse-view and high-noise scenarios; classical methods are often sufficient for dense/low-noise cases.

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

- [\[NeurIPS 2025\] Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?](../../NeurIPS2025/medical_imaging/are_pixel-wise_metrics_reliable_for_sparse-view_computed_tomography_reconstructi.md)
- [\[ICLR 2026\] OpenPros: A Large-Scale Dataset for Limited View Prostate Ultrasound Computed Tomography](openpros_a_large-scale_dataset_for_limited_view_prostate_ultrasound_computed_tom.md)
- [\[ICLR 2026\] U2-BENCH: Benchmarking Large Vision-Language Models on Ultrasound Understanding](u2-bench_benchmarking_large_vision-language_models_on_ultrasound_understanding.md)
- [\[ICLR 2026\] Moving Beyond Diffusion: Hierarchy-to-Hierarchy Autoregression for fMRI-to-Image Reconstruction](moving_beyond_diffusion_hierarchy-to-hierarchy_autoregression_for_fmri-to-image_.md)
- [\[ICLR 2026\] Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)

</div>

<!-- RELATED:END -->
