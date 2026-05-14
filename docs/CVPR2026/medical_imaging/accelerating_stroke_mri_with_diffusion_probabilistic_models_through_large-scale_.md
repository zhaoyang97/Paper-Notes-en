---
title: >-
  [Paper Note] Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning
description: >-
  [CVPR 2026][Medical Imaging][Accelerated MRI reconstruction] Drawing inspiration from the "pre-train then fine-tune" paradigm of foundation models…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Accelerated MRI reconstruction"
  - "diffusion probabilistic models"
  - "large-scale pre-training"
  - "fine-tuning transfer"
  - "stroke imaging"
  - "Diffusion Posterior Sampling"
date: 2026-05-08
content_hash: 7a62d0b28c45b604
---

# Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning

**Conference**: CVPR 2026
**arXiv**: [2603.13007](https://arxiv.org/abs/2603.13007)
**Code**: To be confirmed
**Area**: Medical Imaging
**Keywords**: Accelerated MRI reconstruction, diffusion probabilistic models, large-scale pre-training, fine-tuning transfer, stroke imaging, Diffusion Posterior Sampling

## TL;DR

Drawing inspiration from the "pre-train then fine-tune" paradigm of foundation models, this work pre-trains a diffusion probabilistic model (DPM) at scale on ~4,000 fastMRI subjects spanning multiple contrasts, then fine-tunes on as few as 20 target-domain subjects using a low learning rate. The resulting model generalizes across contrasts and acquisition protocols for accelerated MRI reconstruction. In a clinical stroke validation, 2× accelerated images are rated non-inferior to fully-sampled images by blinded neuroradiologists.

## Background & Motivation

### 1. State of the Field
MRI is the primary imaging modality for stroke diagnosis, yet long scan times (15–30 minutes on average for acute stroke patients) pose a critical bottleneck where time equates to brain tissue. Accelerated MRI (undersampled k-space + algorithmic reconstruction) can shorten scan duration. Deep learning-based reconstruction methods—including end-to-end CNNs, unrolled networks, and score-based diffusion models—have demonstrated notable progress on public benchmarks such as fastMRI.

### 2. Limitations of Prior Work
Clinical stroke MRI data are extremely scarce and difficult to acquire: (1) acute stroke patients are often uncooperative, resulting in unstable data quality; (2) stroke protocols involve multiple contrast sequences (FLAIR, SWI, MPRAGE, DWI), further reducing per-contrast data volume; (3) existing ML methods typically require hundreds of training subjects and demand exact alignment between training and test sampling patterns and coil configurations—a condition rarely satisfied in deployment. Direct training of deep learning reconstruction models for clinical stroke is therefore infeasible.

### 3. Root Cause
Powerful deep learning reconstruction methods, especially DPMs, require large training datasets, yet clinical stroke data are inherently scarce. Moreover, substantial differences in sampling patterns and coil counts across MRI protocols make cross-protocol transfer difficult for conventional approaches.

### 4. Paper Goals
(1) Train reliable accelerated reconstruction models using limited stroke data; (2) generalize across contrasts and acquisition protocols; (3) validate diagnostic image quality under clinical standards.

### 5. Starting Point
Motivated by the success of foundation models in NLP and computer vision—large-scale pre-training on general data followed by fine-tuning on small target-domain datasets. Large-scale public datasets such as fastMRI serve as pre-training sources. The key insight is that DPMs learn an image prior distribution (score function) that is decoupled from sampling patterns and coil configurations, making them naturally suited for cross-protocol transfer.

### 6. Core Idea
(1) Pre-train a DPM on ~4,000 fastMRI subjects covering T1w, T2w, and T1-post contrasts to acquire a rich MRI image prior; (2) fine-tune on only 20 target-domain subjects using a very low learning rate ($10^{-5}$) for a short duration (650 epochs, ~2% of pre-training), adapting the prior to the target contrast; (3) at inference, apply Diffusion Posterior Sampling (DPS) with data-consistency constraints for reconstruction, entirely independent of the training-time sampling pattern.

## Method

### Overall Architecture

The pipeline consists of two stages: **pre-training** → **fine-tuning** → **DPS inference**.

**Pre-training**: 2D slices are extracted from fastMRI multi-contrast brain MRI volumes and used to train a score-based DPM, $s_\theta(\mathbf{x}_t, t)$, that learns a comprehensive image prior. Contrast conditioning is introduced to enable a single model to handle multiple MRI contrasts.

**Fine-tuning**: The pre-trained model is further trained on target-domain data (e.g., 20 FLAIR subjects), with the learning rate reduced from $10^{-4}$ to $10^{-5}$ and training steps set to ~2% of pre-training. Both full-parameter and partial-parameter fine-tuning strategies are explored.

**Inference**: Given undersampled k-space measurements $\mathbf{y}$, DPS injects data-consistency gradient updates at each reverse diffusion step to solve: $\hat{\mathbf{x}} = \arg\min_\mathbf{x} \|\mathbf{A}\mathbf{x} - \mathbf{y}\|_2^2$ s.t. $\mathbf{x} \sim p_\theta(\mathbf{x})$.

### Key Designs

#### 1. Large-Scale Multi-Contrast Pre-training

**Function**: Pre-train a DPM on the fastMRI brain MRI dataset, covering T1w (~1,332 subjects), T2w (~1,340), and T1-post (~1,352), totaling approximately 4,024 subjects.

**Mechanism**: Multi-channel k-space data for each subject are combined into single-channel magnitude images via Root Sum of Squares (RSS). Multiple 2D axial slices are extracted from each 3D volume. Slices from all contrasts are mixed during training, enabling the DPM to learn a general brain MRI prior across contrasts.

**Design Motivation**: (1) The score function $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ encodes the image prior distribution, fully decoupled from k-space sampling patterns—providing the theoretical basis for cross-protocol transfer; (2) multi-contrast mixed pre-training exposes the model to diverse tissue contrast patterns, enriching the learned prior; (3) the scale of fastMRI (~4,000 subjects) far exceeds any clinical stroke dataset, providing sufficient training signal.

#### 2. Contrast Conditioning

**Function**: Enable a single DPM to discriminate between MRI contrasts and produce contrast-specific score estimates.

**Mechanism**: Contrast type is encoded as a one-hot vector (e.g., T1w=[1,0,0], T2w=[0,1,0]), projected to a continuous embedding via a fully connected network, and injected into each layer of the U-Net score network. Contrast labels are provided during training; the target contrast is specified at inference.

**Design Motivation**: Different MRI contrasts exhibit substantially different tissue signal patterns (e.g., white matter bright in T1w, gray matter bright in T2w, CSF dark in FLAIR). A single unconditional model cannot accurately capture all contrast distributions simultaneously. Conditioning allows the model to share most parameters while switching to the appropriate contrast prior via the conditioning vector.

#### 3. Target-Domain Fine-Tuning Strategy

**Function**: Efficiently adapt the pre-trained DPM to the target domain (e.g., stroke FLAIR).

**Mechanism**: The learning rate is identified as the most critical hyperparameter. A systematic comparison of learning rates $\{10^{-4}, 5 \times 10^{-5}, 10^{-5}\}$ and epoch counts $\{50, 100, 325, 650\}$ is conducted:

- $\text{lr}=10^{-4}$ (same as pre-training): the model rapidly overfits to the small target-domain dataset, causing catastrophic forgetting of the pre-trained prior and degrading reconstruction quality below direct training
- $\text{lr}=10^{-5}$ (10× reduction): gentle updates preserve the pre-trained prior while adapting to target-domain statistics
- **Optimal configuration**: $\text{lr}=10^{-5}$, 650 epochs (~12.5K steps), approximately 2% of pre-training computation

**Design Motivation**: Consistent with empirical findings from fine-tuning foundation models in NLP and CV—lower learning rates prevent disruption of learned representations. The score function acquired during pre-training already captures rich structural priors; fine-tuning requires only minor adjustments to align the prior with the target contrast.

#### 4. DPS (Diffusion Posterior Sampling) Reconstruction

**Function**: Leverage the learned prior and the physical measurement model at inference to perform MRI reconstruction.

**Mechanism**: Standard DPM sampling generates images by iterative denoising from $\mathbf{x}_T \sim \mathcal{N}(0, I)$. DPS augments each denoising step with a data-consistency gradient update:

$$\mathbf{x}_{t-1} = \text{DPM-step}(\mathbf{x}_t, s_\theta) - \zeta_t \nabla_{\mathbf{x}_t} \|\mathbf{A}\hat{\mathbf{x}}_0(\mathbf{x}_t) - \mathbf{y}\|_2^2$$

where $\mathbf{A}$ denotes the MRI forward model (coil sensitivities × Fourier transform × sampling mask), $\hat{\mathbf{x}}_0$ is the Tweedie estimate from $\mathbf{x}_t$, and $\zeta_t$ is the step size.

**Design Motivation**: (1) DPS fully decouples the prior model (DPM) from the acquisition model ($\mathbf{A}$)—the prior is trained without any knowledge of the inference-time sampling pattern, acceleration factor, or coil configuration; (2) this decoupling allows the same pre-trained and fine-tuned model to be deployed across arbitrary acquisition protocols, which is the key enabler of clinical feasibility.

### Training Configuration

- Score network: NCSN++ U-Net
- Pre-training: $\text{lr}=10^{-4}$, Adam, ~625K steps (~1,000 epochs on fastMRI)
- Fine-tuning: $\text{lr}=10^{-5}$, 650 epochs (~12.5K steps), ~2% of pre-training compute
- DPS inference: 1,000 reverse diffusion steps, adaptive $\zeta_t$

## Key Experimental Results

### fastMRI Experiments

Evaluated on the fastMRI T1w test set, comparing different training data volumes and strategies.

**Table 1: 20-subject fine-tuning vs. direct training (4× acceleration, T1w, SSIM↑/PSNR↑)**

| Method | Training Subjects | SSIM | PSNR |
|--------|------------------|------|------|
| Zero-filled | — | 0.758 | 27.3 |
| Direct training (20 subjects) | 20 | 0.871 | 31.2 |
| Direct training (344 subjects) | 344 | 0.912 | 33.8 |
| **Pre-training + fine-tuning (20 subjects)** | **20 (+ 4,000 pre-train)** | **0.908** | **33.5** |

Key finding: Fine-tuning with only 20 subjects (SSIM 0.908) approaches direct training with 344 subjects (0.912), yielding an **effective data efficiency gain of ~17×**.

**Fine-tuning learning rate ablation**:

| Learning Rate | Epochs | SSIM | PSNR |
|---------------|--------|------|------|
| $10^{-4}$ (same as pre-training) | 650 | 0.862 | 30.8 |
| $5 \times 10^{-5}$ | 650 | 0.891 | 32.4 |
| $10^{-5}$ | 650 | **0.908** | **33.5** |
| $10^{-5}$ | 325 | 0.901 | 33.0 |
| $10^{-5}$ | 100 | 0.885 | 32.1 |

A learning rate of $10^{-5}$ substantially outperforms $10^{-4}$ (+0.046 SSIM), confirming that a low learning rate is critical for preserving the pre-trained prior.

### Clinical Stroke Validation

**Data**: 30 acute stroke patients (UT/MD Anderson Cancer Center); 25 for training and 5 for testing. Each patient underwent SWI, MPRAGE, DWI, and FLAIR sequences. Retrospective 2× undersampling was applied.

**Quantitative results (2× acceleration)**:

| Sequence | SSIM | PSNR | NMSE (×10⁻³) |
|----------|------|------|---------------|
| SWI | 0.941 | 35.7 | 1.2 |
| MPRAGE | 0.928 | 34.2 | 1.8 |
| DWI | 0.935 | 34.9 | 1.5 |
| FLAIR | 0.923 | 33.6 | 2.1 |

### Reader Study (Core Clinical Validation)

**Design**: Two experienced neuroradiologists independently and blindly evaluated image quality for 80 subjects (a mix of ground-truth and reconstructed images) using a 5-point Likert scale.

**Results**:
- Mean score for reconstructed images: 4.2/5.0
- Mean score for fully-sampled images: 4.4/5.0
- Statistical test: reconstructed images are non-inferior to fully-sampled images at non-inferiority margin δ=0.5 (p < 0.01)
- **No reconstructed image was rated as "non-diagnostic"**
- No significant difference in detection sensitivity for key stroke findings (infarcts, hemorrhage, edema)

This constitutes the first formal reader study validating DPM-based accelerated MRI in clinical stroke imaging.

## Highlights & Insights

1. **First systematic validation of the "pre-train then fine-tune" paradigm for MRI reconstruction**: Demonstrates that DPM image priors transfer across contrasts and acquisition protocols; 20-subject fine-tuning ≈ 344-subject direct training.
2. **Acquisition-agnostic nature of DPMs as a key advantage**: The score function encodes only the image prior, fully decoupled from sampling patterns and coil configurations—a property unavailable in unrolled networks, which hard-code the forward model into the network architecture.
3. **Practical guidance on learning rate selection**: The systematic ablation study provides a clear recommendation ($\text{lr}=10^{-5}$, 650 epochs), offering direct reference value for follow-up work.
4. **Clinical persuasiveness of the reader study**: Rather than solely optimizing SSIM/PSNR, the work employs blinded radiologist evaluation with a non-inferiority criterion—a necessary step toward clinical translation.

## Limitations & Future Work

1. **Only 2× acceleration validated**: Clinical demand may require 4× or 8× acceleration; performance at higher rates remains unknown.
2. **Slow DPS inference**: The 1,000-step reverse diffusion process is computationally intensive; reconstruction of a single slice may require tens of seconds, limiting real-time clinical use.
3. **RSS combining discards phase information**: Pre-training on RSS magnitude images loses multi-channel phase information, limiting reconstruction capability for certain sequences (e.g., SWI).
4. **Retrospective undersampling**: Stroke experiments rely on retrospectively simulated undersampling; a gap remains relative to prospective real undersampling.
5. **Pre-training data scale can be expanded further**: 4,000 subjects is small relative to the training scale of NLP/CV foundation models; whether scaling to tens of thousands of subjects would yield continued improvements is an open question.

## Related Work & Insights

- **Evolution of MRI reconstruction methods**: GRAPPA/SENSE (classical) → compressed sensing (sparse + TV priors) → end-to-end CNNs (fastMRI Challenge) → unrolled networks (VarNet, etc.) → score-based DPMs (this work). The trend moves from handcrafted priors toward data-driven priors, with DPMs representing the most flexible prior form.
- **Connection to foundation model methodology**: Analogous to ImageNet pre-training → downstream task fine-tuning, this work establishes a fastMRI pre-training → clinical protocol fine-tuning paradigm, providing a template for data-efficient deployment of medical imaging AI.
- **Implications for clinical translation**: The reader study design (non-inferiority testing, blinded evaluation) represents the gold standard for clinical acceptance of medical AI and merits adoption by other reconstruction method works.

## Rating

⭐⭐⭐⭐ The methodology is conceptually clear and practically valuable. The "pre-train then fine-tune" paradigm is systematically validated for MRI reconstruction, and the clinical reader study is a significant strength. Limitations include the restriction to 2× acceleration and slow inference speed, leaving a gap before true clinical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](../../AAAI2026/medical_imaging/g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)
- [\[ICLR 2026\] Fine-Tuning Diffusion Models via Intermediate Distribution Shaping](../../ICLR2026/medical_imaging/fine-tuning_diffusion_models_via_intermediate_distribution_shaping.md)
- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](../../AAAI2026/medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[CVPR 2026\] Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding](ultrasound-clip_semantic-aware_contrastive_pre-training_for_ultrasound_image-tex.md)

</div>

<!-- RELATED:END -->
