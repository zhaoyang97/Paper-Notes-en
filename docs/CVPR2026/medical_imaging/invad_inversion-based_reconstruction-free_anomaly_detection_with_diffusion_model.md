---
title: >-
  [Paper Note] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models
description: >-
  [CVPR 2026][Medical Imaging][Anomaly detection] This paper proposes InvAD, which shifts diffusion-based anomaly detection from a "denoising-reconstruction in RGB space" paradigm to a "noising-inversion in latent space" paradigm. By applying DDIM inversion to directly infer the terminal latent variable and measuring deviation under the prior distribution, anomalies are detected without reconstruction. Only 3 inversion steps suffice to achieve state-of-the-art performance, with approximately 2× inference speedup.
tags:
  - CVPR 2026
  - Medical Imaging
  - Anomaly detection
  - diffusion models
  - DDIM inversion
  - reconstruction-free paradigm
  - industrial/medical defect detection
date: 2026-05-08
content_hash: 7975866f4f3ab53e
---

# InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models

**Conference**: CVPR 2026
**arXiv**: [2504.05662](https://arxiv.org/abs/2504.05662)
**Code**: [Project Page](https://invad-project.com)
**Area**: Medical Imaging
**Keywords**: Anomaly detection, diffusion models, DDIM inversion, reconstruction-free paradigm, industrial/medical defect detection

## TL;DR

This paper proposes InvAD, which shifts diffusion-based anomaly detection from a "denoising-reconstruction in RGB space" paradigm to a "noising-inversion in latent space" paradigm. By applying DDIM inversion to directly infer the terminal latent variable and measuring deviation under the prior distribution, anomalies are detected without reconstruction. Only 3 inversion steps suffice to achieve state-of-the-art performance, with approximately 2× inference speedup.

## Background & Motivation

- **State of the Field**: Diffusion-based anomaly detection (AD) methods have achieved strong results but suffer from a fundamental efficiency–accuracy trade-off.
- **Limitations of Prior Work**: (1) *Noise sensitivity*—excessively strong noise corrupts normal regions causing false positives, while insufficient noise allows anomalous regions to be reconstructed faithfully, leading to missed detections; (2) *Expensive multi-step denoising*—satisfactory reconstruction requires iterative denoising, and most methods operate at roughly 1 FPS or below (e.g., DiAD at 0.1 FPS, GLAD at 0.2 FPS).
- **Root Cause**: Since diffusion models are trained solely on normal data distributions, reconstruction is not necessary for anomaly detection. Inversion can directly map an image into the latent space: normal images map to high-density regions of the prior distribution, whereas anomalous images map to low-density regions, entirely bypassing reconstruction and eliminating the need to tune noise strength.
- **Paper Goals**: Propose a reconstruction-free inference paradigm based on DDIM inversion that is both more accurate and significantly faster than existing diffusion-based AD methods.

## Method

### Overall Architecture

Input image → backbone feature extraction $\mathbf{z} = g_\phi(\mathbf{x})$ → DDIM inversion (3 steps) yielding $\mathbf{z}_T$ → anomaly score computed from the deviation of $\mathbf{z}_T$ under the prior distribution. No decoder or reconstruction is required.

### Key Designs

1. **DDIM Inversion Noising (Core)**: The image is forward-propagated along the PF-ODE trajectory to directly infer $\mathbf{x}_T$ from $\mathbf{x}_0$, using the discrete Euler approximation:
$$\mathbf{x}_{\tau_{i+1}} = \sqrt{\alpha_{\tau_{i+1}}} f_\theta(\mathbf{x}_{\tau_i}) + \sqrt{1-\alpha_{\tau_{i+1}}} \epsilon_\theta^{(\tau_i)}(\mathbf{x}_{\tau_i}).$$
Crucially, only very few inversion steps are required ($S=3$, subset $\tau_3 = [333, 666, 999]$): even under the lower-accuracy Euler approximation, anomalous pixels are still mapped to low-density regions of the prior.
**Design Motivation**: The deterministic nature of the PF-ODE guarantees a one-to-one mapping between normal images and the prior distribution, so anomalous deviations can be directly quantified via distributional typicality.

2. **Feature-Space Diffusion Modeling**: A pretrained EfficientNet-B4 extracts features $\mathbf{z} = g_\phi(\mathbf{x}) \in \mathbb{R}^{C \times h \times w}$ as the input space for the diffusion model, rather than raw pixel space. Advantages: (a) backbone features are invariant to low-level variations such as noise and illumination; (b) lower spatial resolution yields more efficient inference. A DiT-gigant architecture is used as the diffusion backbone.

3. **Hybrid Anomaly Scoring**: From the inverted $\mathbf{z}_T$, a pixel-level anomaly map is obtained via the channel-wise Euclidean norm: $\mathbf{z}_T^{\text{normed}}[i,j] = \|\mathbf{z}_T[:,i,j]\|_2$. The image-level score is defined as $s = \max(A) - \min(A) + \sum_{u,v} A[u,v]$, where the max-min difference mitigates the inverse scoring problem—since anomalies are typically sparse and localized, the max-min contrast filters out the influence of global outliers.

### Loss & Training

- **Training**: Standard DDPM $\epsilon$-prediction loss, trained on normal data only; AdamW optimizer, 300 epochs, $T=1000$, linear noise schedule.
- **Inference**: $S=3$ inversion steps with uniform subset $\tau_3 = [333, 666, 999]$.
- **Plug-and-play design**: Only the inference stage is modified; InvAD can directly replace the inference procedure of existing diffusion-based AD methods without retraining.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (InvAD) | OmiAD (ICML'25) | DiAD (AAAI'24) | FPS |
|---------|--------|-------------|-----------------|----------------|-----|
| MVTecAD | I-AUROC | **99.0** | 98.8 | 97.2 | **88.1** vs 39.4 vs 0.1 |
| VisA | I-AUROC | **96.9** | 95.3 | 86.8 | **74.1** vs 35.3 |
| MPDD | I-AUROC | **96.5** | 93.7 | 74.6 | **120** vs 49.8 |
| BMAD (Medical) | mAD | **87.2** | — | — | **88** vs 20 |

### Ablation Study

| Configuration | MVTecAD mAD | Notes |
|---------------|-------------|-------|
| FDM only (no inversion) | 57.3 | Inversion is the essential component |
| Single-step inversion (pixel space) | 44.9 | Pixel-space diffusion + single step insufficient |
| FDM + single-step inversion | 71.0 | Feature space + single step |
| **FDM + multi-step inversion (full)** | **83.7** | Optimal configuration |

| Inversion Steps $S$ | Reconstruction (best $r$) | Inversion (Ours) |
|---------------------|--------------------------|-----------------|
| 3 | 64.9 | **99.0** |
| 5 | 75.0 | **98.9** |
| 10 | 97.9 | 98.4 |
| 50 | 98.0 | 96.0 |
| 1000 | 98.2 | 95.4 |

### Key Findings

- The inversion approach substantially outperforms reconstruction-based methods at very few steps ($S=3,5$); reconstruction methods require $S \geq 50$ to achieve comparable performance.
- Inversion requires no tuning of the perturbation timestep, whereas reconstruction methods are highly sensitive to both $r$ and $S$.
- Plug-and-play application: DiAD + InvAD yields +1.0 I-AUROC and +88 FPS; MDM + InvAD yields +6.3 I-AUROC and +60.8 FPS.
- The hybrid NLL + Diff scoring is robust across inversion step counts $S$; using NLL or Diff alone is not.
- State-of-the-art performance is also achieved across 6 datasets in the BMAD medical benchmark (mAD = 87.2), demonstrating cross-domain generalizability.

## Highlights & Insights

- **Paradigm Innovation**: The conceptual shift from "detect via denoising" to "detect via noising" is the core contribution—elegant in its simplicity and highly effective in practice.
- Inversion naturally eliminates both the noise-strength tuning problem and the computational bottleneck of multi-step reconstruction.
- The reason $S=3$ suffices to achieve SOTA is that precise reconstruction is unnecessary; only distinguishing distributional typicality between normal and anomalous samples is required.
- The plug-and-play design allows InvAD to serve as a universal inference accelerator for existing diffusion-based AD methods.
- Feature-space diffusion modeling is an important design choice that jointly improves efficiency and detection performance.

## Limitations & Future Work

- More than one function evaluation (NFE = 3) is still required; diffusion distillation could potentially compress this to a single step.
- Pixel-level localization metrics (AP, F1_max) are inferior to some reconstruction-based methods, as inversion has an inherent disadvantage in precise boundary delineation.
- The max-min contrast term in the scoring scheme is empirically motivated and lacks theoretical justification.
- DiT-gigant has a large parameter count (1,223M); an MLP backbone achieves comparable detection accuracy but inferior localization.
- Task-specific optimization of the inversion mechanism has not been explored.

## Related Work & Insights

- The deterministic sampling and PF-ODE formulation of DDIM (Song et al., 2020) provide the theoretical foundation for the inversion approach.
- Heng et al. (2024)'s use of score function norms to measure OOD typicality inspired the anomaly scoring design in this work.
- OmiAD (Feng et al., 2025) achieves 1-step diffusion via adversarial distillation but incurs higher training complexity.
- Compared to non-diffusion methods such as EfficientAD (Batzner et al., 2023), diffusion-based methods still maintain an accuracy advantage.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The "detect via noising" paradigm is a conceptually innovative contribution: concise, elegant, and highly effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four industrial and six medical datasets, with comprehensive ablations covering components, backbones, scoring strategies, step counts, and generalizability.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem analysis is clear, paradigm comparison figures are intuitive, and tables are well-organized.
- **Value**: ⭐⭐⭐⭐⭐ — Highly practical; serves as a plug-and-play accelerator for existing methods with significant implications for both industrial and medical AD.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [\[CVPR 2026\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[CVPR 2026\] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)
- [\[AAAI 2026\] WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images](../../AAAI2026/medical_imaging/wdt-md_wavelet_diffusion_transformers_for_microaneurysm_detection_in_fundus_imag.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)

<!-- RELATED:END -->
