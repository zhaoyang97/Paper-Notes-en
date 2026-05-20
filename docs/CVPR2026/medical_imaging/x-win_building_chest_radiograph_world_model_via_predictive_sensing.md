---
title: >-
  [Paper Note] X-WIN: Building Chest Radiograph World Model via Predictive Sensing
description: >-
  [CVPR 2026][Medical Imaging][World model] X-WIN is a chest radiograph world model that, for the first time, incorporates 3D CT spatial knowledge into CXR representation learning. By learning to predict 2D projections of…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "World model"
  - "chest radiograph representation learning"
  - "CT knowledge distillation"
  - "contrastive learning"
  - "domain adaptation"
date: 2026-05-08
content_hash: a1ed0d1cab0bcbc2
---

# X-WIN: Building Chest Radiograph World Model via Predictive Sensing

**Conference**: CVPR 2026
**arXiv**: [2511.14918](https://arxiv.org/abs/2511.14918)  
**Code**: None  
**Area**: Medical Imaging
**Keywords**: World model, chest radiograph representation learning, CT knowledge distillation, contrastive learning, domain adaptation

## TL;DR

X-WIN is a chest radiograph world model that, for the first time, incorporates 3D CT spatial knowledge into CXR representation learning. By learning to predict 2D projections of CT volumes at varying rotation angles, the model internalizes 3D anatomical structure. Combined with affinity-guided contrastive alignment and structure-preserving domain adaptation, X-WIN achieves state-of-the-art linear probing performance across 6 CXR benchmarks.

## Background & Motivation

- **Inherent limitations of CXR**: As 2D projection images, CXRs suffer from structural superimposition and cannot directly capture 3D anatomical information, limiting diagnostic capability.
- **CT vs. CXR trade-off**: CT provides 3D structural information but is costly, radiation-intensive, and less accessible, while CXR is safe and affordable but informationally limited.
- **Radiologist insight**: Radiologists viewing frontal/lateral CXRs can cognitively reconstruct a 3D thoracic model to assist in diagnosing occluded structures.
- **Limitations of existing world models**: CheXWorld only learns 2D local structure and global geometry, lacking 3D spatial awareness.
- **Mechanism**: If a model can accurately predict X-ray projections of a CT volume at arbitrary rotation angles, it has internalized meaningful 3D anatomical structure.

## Method

### Overall Architecture (JEPA Variant)

1. **Context encoder** $f_\theta$: Takes standard frontal/lateral CXRs as input.
2. **EMA encoder** $f_{\theta'}$: Takes multiple target projections; updated via exponential moving average.
3. **View predictor** $g_v$: Predicts latent representations of new projections conditioned on action.
4. **Masked predictor** $g_m$: Reconstructs masked patch tokens.

### Action Design

The action $a_i = k \cdot \Delta\phi$ defines the yaw rotation angle of the X-ray source relative to the input position, constrained to $[-90°, 90°]$, with $N=8$ projections sampled randomly. The optimal step size is $\Delta\phi = 3°$ (corresponding to 60 latent projections).

### Affinity-Guided Contrastive Alignment

Predicted representation: $z_i^{\text{patch}} = g_v(\text{Linear}(a_i) \oplus (f_\theta(u_{\text{context}}) + \text{PE}))$

Standard InfoNCE uses one-hot labels for hard alignment, but different projections of the same CT exhibit rich anatomical correspondences. An affinity matrix $A$ is introduced for soft regularization:

$$A_{ij} = \frac{\exp(\text{sim}(t_i, t_j)/\tilde{\tau})}{\sum_l \exp(\text{sim}(t_i, t_l)/\tilde{\tau})}$$

$$\mathcal{L}_{\text{align}} = \mathcal{L}_{\text{InfoNCE}} + \lambda_{\text{affinity}} \mathcal{L}_{\text{affinity}}$$

### Structure-Preserving Domain Adaptation

Simulated CXRs (generated from CT projections) exhibit a domain gap relative to real CXRs. The following strategies are adopted:

1. **Masked image modeling (MIM)**: Applied to both real and simulated CXRs to encode local and contextual features.
2. **Domain classifier** $f_c$: Learns to distinguish between real and simulated domain representations.
3. **Domain adaptation loss**:

$$\mathcal{L}_{\text{domain}} = \frac{1}{N} \sum_{i=1}^{N} \|z_i^{\text{patch}} - t_i^{\text{patch}}\|_2^2 - \frac{1}{N} \sum_{i=1}^{N} \log f_c(z_i^{\text{patch}})$$

This encourages projection predictions to preserve structural information while being statistically aligned with the real domain.

### Overall Loss

$$\mathcal{L}_{\text{overall}} = \mathcal{L}_{\text{align}} + \lambda_{\text{MIM}} \mathcal{L}_{\text{MIM}} + \lambda_{\text{domain}} \mathcal{L}_{\text{domain}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}$$

## Key Experimental Results

### Main Results: Linear Probing (AUROC)

| Model | Pretraining Data | VinDr | CheXpert | NIH-CXR | RSNA | JSRT | Avg. |
|-------|-----------------|-------|----------|---------|------|------|------|
| DINOv2 | LVD-142M | 0.795 | 0.776 | 0.711 | 0.798 | 0.559 | 0.728 |
| CheXFound | 987K CXR | 0.869 | 0.876 | 0.829 | 0.872 | 0.846 | 0.858 |
| Ark+ | 704K CXR | 0.906 | 0.876 | 0.831 | 0.893 | 0.807 | 0.863 |
| CheXWorld | 448K CXR | 0.903 | 0.871 | 0.833 | 0.824 | 0.791 | 0.844 |
| **X-WIN (ViT-L)** | **372K CXR+32K CT** | **0.925** | **0.908** | **0.843** | **0.929** | **0.857** | **0.892** |

X-WIN achieves a mean AUROC of 0.892, surpassing all CXR foundation models and vision-language models.

### Few-Shot Fine-Tuning (COVIDx, AUROC)

| Model | 4-shot | 8-shot | 16-shot | All |
|-------|--------|--------|---------|-----|
| CheXFound | 0.823 | 0.883 | 0.897 | 0.977 |
| CheXWorld | 0.843 | 0.893 | 0.902 | 0.981 |
| **X-WIN** | **0.868** | **0.924** | **0.939** | **0.993** |

### 3D CT Reconstruction

3D CT volumes are reconstructed via a VQ-GAN decoder combined with the FDK algorithm:
- 2D projection: PSNR 30.23 dB, SSIM 0.888
- 3D reconstruction: PSNR 27.87 dB, SSIM 0.789

### Ablation Study

- $\mathcal{L}_{\text{InfoNCE}}$ alone establishes a strong baseline.
- Combining $\mathcal{L}_{\text{MIM}}$ + $\mathcal{L}_{\text{InfoNCE}}$ yields substantial improvements.
- Domain adaptation improves cosine similarity from 0.845 to 0.967.
- Direct rotation outperforms stepwise rotation; yaw-only rotation outperforms full 3D Euler angle rotation.

## Highlights & Insights

1. **Breakthrough in medical imaging world models**: X-WIN is the first to incorporate 3D CT spatial knowledge into a 2D CXR world model, representing a paradigm shift from 2D to 3D reasoning.
2. **Elegant predictive sensing design**: 3D structure is internalized by predicting rotated projections rather than directly regressing 3D features.
3. **Affinity-guided contrastive learning**: Natural correlations among different projections of the same CT are leveraged for soft contrastive supervision, which is more principled than hard InfoNCE.
4. **Validated 3D reconstruction**: The model's ability to render projections from CXRs and reconstruct CT volumes confirms that genuine 3D knowledge has been acquired.

## Limitations & Future Work

- Relies on DiffDRR for simulated projection generation; the simulation-to-real domain gap remains a bottleneck.
- Only yaw rotation is utilized, leaving pitch and roll dimensions unexplored.
- 3D reconstruction results exhibit blurring with notable loss of local detail.
- Training requires 8×A100 40 GB GPUs for 100 epochs, incurring substantial computational cost.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MRI Contrast Enhancement Kinetics World Model](mri_contrast_enhancement_kinetics_world_model.md)
- [\[AAAI 2026\] PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis](../../AAAI2026/medical_imaging/pulsemind_a_multi-modal_medical_model_for_real-world_clinical_diagnosis.md)
- [\[CVPR 2026\] cryoSENSE: Compressive Sensing Enables High-throughput Microscopy with Sparse and Generative Priors on the Protein Cryo-EM Image Manifold](cryosense_compressive_sensing_enables_high-throughput_microscopy_with_sparse_and.md)
- [\[CVPR 2026\] Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)
- [\[CVPR 2026\] XSeg: A Large-scale X-ray Contraband Segmentation Benchmark for Real-World Security Screening](xseg_a_large-scale_x-ray_contraband_segmentation_benchmark_for_real-world_securi.md)

</div>

<!-- RELATED:END -->
