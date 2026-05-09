---
title: >-
  [Paper Note] Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning
description: >-
  [CVPR 2026][Medical Imaging][accelerated MRI] Inspired by the foundation model paradigm, this work proposes a data-efficient training strategy for diffusion probabilistic models (DPMs) in accelerated MRI reconstruction. A DPM is first pre-trained on large-scale multi-contrast brain MRI data (~4,000 subjects), then fine-tuned with as few as 20 target-domain subjects. The resulting model achieves reconstruction quality comparable to large-dataset training in clinical stroke MRI, with a clinical blind reader study confirming non-inferiority to standard-of-care at 2× acceleration.
tags:
  - CVPR 2026
  - Medical Imaging
  - accelerated MRI
  - diffusion probabilistic models
  - foundation model
  - stroke MRI
  - fine-tuning
date: 2026-05-08
content_hash: 631d2db82528420b
---

# Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning

**Conference**: CVPR 2026
**arXiv**: [2603.13007](https://arxiv.org/abs/2603.13007)
**Code**: None
**Area**: Medical Imaging / MRI Reconstruction
**Keywords**: accelerated MRI, diffusion probabilistic models, foundation model, stroke MRI, fine-tuning

## TL;DR
Inspired by the foundation model paradigm, this work proposes a data-efficient training strategy for diffusion probabilistic models (DPMs) in accelerated MRI reconstruction. A DPM is first pre-trained on large-scale multi-contrast brain MRI data (~4,000 subjects), then fine-tuned with as few as 20 target-domain subjects. The resulting model achieves reconstruction quality comparable to large-dataset training in clinical stroke MRI, with a clinical blind reader study confirming non-inferiority to standard-of-care at 2× acceleration.

## Background & Motivation
**Background**: Machine learning has achieved state-of-the-art performance in accelerated MRI reconstruction, but these methods typically require large amounts of fully-sampled, application-specific training data. Performance degrades substantially when target-domain data is scarce.

**Limitations of Prior Work**: (1) Stroke MRI represents a prototypical data-scarce scenario—while MRI is more sensitive than CT for detecting ischemic stroke, long scan times and motion sensitivity delay treatment; (2) existing self-supervised and few-data methods fall short of fully supervised baselines in reconstruction quality; (3) end-to-end reconstruction methods require the external training dataset and the target dataset to share the same acquisition model (sampling pattern, coil geometry, etc.), limiting cross-domain transferability.

**Key Challenge**: High-quality accelerated MRI reconstruction requires large amounts of target-domain fully-sampled data for training, yet clinically specific scenarios such as stroke imaging are precisely where such data are scarce.

**Goal**: To train a high-quality DPM-based accelerated MRI reconstruction model when target-domain fully-sampled data is extremely limited (20–25 subjects).

**Key Insight**: The acquisition-model-agnostic nature of DPMs is exploited—since DPMs learn an image prior distribution rather than an end-to-end mapping, pre-training data may originate from different sampling patterns and coil geometries than those used at inference.

**Core Idea**: Large-scale multi-contrast pre-training + carefully controlled fine-tuning (learning rate reduced by one order of magnitude + very few epochs) = high-quality DPM reconstruction in data-limited clinical stroke MRI.

## Method

### Overall Architecture
A two-stage training strategy is employed: (1) a DPM is pre-trained on multi-contrast brain MRI (T1, T2, T1-post) from ~4,000 subjects in the fastMRI dataset; (2) the model is fine-tuned on a small target-domain dataset (20–25 stroke patients) using a reduced learning rate ($1 \times 10^{-5}$) and a limited number of training epochs (~650, approximately 2% of pre-training). At inference, Diffusion Posterior Sampling (DPS) is used to reconstruct images from undersampled k-space data.

### Key Designs

1. **Contrast-Conditioned DPM Architecture**:

    - Function: Enables a single DPM to learn the image distributions of multiple MRI contrasts simultaneously.
    - Mechanism: Each contrast type is assigned a one-hot vector, which is mapped to an embedding vector via a small fully connected network; every block of the U-Net receives this embedding as an additional input. The model learns the score function $\nabla_{x_t} \log p_t(x)$, where $p_t(x)$ is the image distribution perturbed by Gaussian noise $\sigma_t$. The DPM and the embedding network are trained jointly.
    - Design Motivation: Heterogeneous multi-contrast data requires the model to distinguish contrast-specific image characteristics; conditioning allows the model to adapt its behavior according to the contrast type.

2. **Controlled Fine-Tuning Strategy**:

    - Function: Effectively adapts the model to the target domain with very limited data without overfitting.
    - Mechanism: The learning rate is reduced by one order of magnitude relative to pre-training ($10^{-4}$ → $10^{-5}$), and training is limited to ~650 epochs (~2% of pre-training). A new one-hot vector is assigned to the target contrast, and the embedding network weights are updated. Experiments show that insufficient fine-tuning leads to inadequate contrast adaptation, while excessive fine-tuning leads to overfitting—an optimal fine-tuning window exists.
    - Design Motivation: The central challenge in foundation model fine-tuning is balancing adaptation and forgetting; a low learning rate combined with short training time prevents catastrophic forgetting while achieving contrast adaptation.

3. **Diffusion Posterior Sampling (DPS) Reconstruction**:

    - Function: Reconstructs images from undersampled k-space measurements using the learned image prior.
    - Mechanism: The following ODE is solved: $d\mathbf{x} = [-t(\nabla_{\mathbf{x}} \|\mathbf{PFS}\tilde{\mathbf{x}}(\mathbf{x}) - \mathbf{y}\|_2^2 + D_\theta(\mathbf{x}, t))]dt$, where the first term enforces data consistency (driving the reconstruction toward agreement with measurements) and the second term represents the learned prior score. The data consistency weight $\zeta$ is a critical hyperparameter—higher acceleration factors require larger $\zeta$.
    - Design Motivation: The acquisition-model-agnostic property of DPMs is a key advantage—unlike end-to-end methods, DPMs learn the image distribution alone and can be combined with any acquisition model at inference via posterior sampling.

### Loss & Training
- Both pre-training and fine-tuning use the EDM training loss (score matching).
- Fixed data augmentation strategies and noise level schedules are applied throughout.
- The posterior sampling weight $\zeta$ scales with the acceleration factor (fewer k-space measurements → larger data consistency weight).

## Key Experimental Results

### Main Results
FastMRI FLAIR reconstruction NRMSE (pre-trained on T1/T2/T1-post; fine-tuned on 20 FLAIR subjects vs. training on 344 FLAIR subjects):

| Method | Data Size | R=4 | R=5 | R=6 | Notes |
|--------|-----------|-----|-----|-----|-------|
| Method 1 (Full dataset) | 4,125 | Best | Best | Best | Upper bound |
| Method 3 (344 FLAIR) | 344 | Near best | Near best | Near best | Upper bound |
| **Method 4 (Ours)** | **20 FLAIR** | **Comparable** | **Comparable** | **Comparable** | **Only 5.8% of data** |
| Method 5 (20 FLAIR only) | 20 | Significantly worse | Significantly worse | Significantly worse | No pre-training |
| Method 6 (Joint training) | 4,000+20 | Worse | Worse | Worse | Inferior to sequential fine-tuning |

### Ablation Study
Effect of fine-tuning hyperparameters on NRMSE:

| Learning Rate | 650 epochs | 1,250 epochs | 2,500 epochs | Notes |
|---------------|-----------|--------------|--------------|-------|
| $5 \times 10^{-4}$ | Moderate | Overfitting | Severe overfitting | LR too high |
| $1 \times 10^{-4}$ | Good | Slight overfitting | Overfitting | — |
| $5 \times 10^{-5}$ | Good | Good | Slight overfitting | — |
| $1 \times 10^{-5}$ | **Best** | Good | Slightly worse | **Optimal configuration** |

### Key Findings
- Clinical reader study (80 patients, 2 neuroradiologists): images reconstructed by the DPM from 2× accelerated data show **no significant difference** from standard-of-care in structural delineation and overall image quality.
- Reader 1 (80 patients): DPM reconstruction is **significantly superior** to standard-of-care on multiple metrics (SNR, sharpness, artifacts).
- Reader 2 (21 patients): standard-of-care is significantly superior to DPM only in SNR; no significant difference on remaining metrics.
- Sequential fine-tuning is substantially superior to joint training—when target-domain data are scarce, sequential fine-tuning is the more effective strategy.

## Highlights & Insights
- The strategy is remarkably simple yet highly effective—"large-scale pre-training + reduced-learning-rate fine-tuning" is systematically validated in MRI reconstruction for the first time.
- The clinical reader study provides strong evidence—not only is NRMSE improved, but clinical acceptability is genuinely verified.
- The acquisition-model-agnostic nature of DPMs is a decisive advantage—pre-training and target data may employ different sampling patterns and coil geometries.
- Prospective undersampling experiments further confirm the reliability of retrospective results.

## Limitations & Future Work
- Clinical evaluation relies on retrospective undersampling; prospective validation is limited to healthy volunteers.
- DPS reconstruction is substantially slower than conventional parallel imaging and end-to-end methods, constraining clinical deployment.
- Fine-tuning hyperparameters (learning rate, number of epochs) require separate tuning for each contrast type.
- Inter-reader agreement between the two radiologists is only "slight" to "fair," reflecting the inherent variability of subjective assessments.

## Related Work & Insights
- **fastMRI**: Provides the large-scale pre-training dataset.
- **DPS (Chung et al.)**: Establishes the posterior sampling framework.
- **Foundation model paradigm (Bommasani et al.)**: Motivates the pre-training and fine-tuning strategy.
- Insight: This strategy is directly transferable to other data-scarce medical image reconstruction tasks (e.g., cardiac MRI, musculoskeletal MRI).

## Rating
- Novelty: ⭐⭐⭐ The method itself (pre-training + fine-tuning) is not novel, but its systematic validation in MRI reconstruction is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Encompasses fastMRI controlled experiments + clinical stroke data + an 80-patient blind reader study + prospective experiments—exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clinically oriented writing style with rigorous experimental design.
- Value: ⭐⭐⭐⭐ Directly advances the clinical deployment of DPMs for accelerated MRI; the reader study confirms clinical feasibility.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](sd_fsmis_adapting_stable_diffusion_for_few_shot_medical_image_segmentation.md)
- [\[CVPR 2026\] LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](lemon_large_endoscopic_monocular_dataset_foundation_model_surgical.md)
- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[CVPR 2026\] Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding](ultrasound-clip_semantic-aware_contrastive_pre-training_for_ultrasound_image-tex.md)

<!-- RELATED:END -->
