---
title: >-
  [Paper Note] Scaling Vision Transformers for Functional MRI with Flat Maps
description: >-
  [ICML 2026][Medical Imaging][fMRI Foundation Model] By projecting 3D fMRI volumes into 2D videos via "cortical flat maps" and feeding them into a standard spacetime MAE-ViT…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "fMRI Foundation Model"
  - "Cortical Flat Map"
  - "MAE"
  - "Brainmarks Benchmarking"
  - "Scaling Law"
date: 2026-05-08
content_hash: 36a7a9da50207381
---

# Scaling Vision Transformers for Functional MRI with Flat Maps

**Conference**: ICML 2026  
**arXiv**: [2510.13768](https://arxiv.org/abs/2510.13768)  
**Code**: https://github.com/MedARC-AI/CortexMAE & https://github.com/MedARC-AI/Brainmarks (Available)  
**Area**: Medical Imaging / Self-Supervised Learning / Neuroimaging Foundation Models  
**Keywords**: fMRI Foundation Model, Cortical Flat Map, MAE, Brainmarks Benchmarking, Scaling Law

## TL;DR
By projecting 3D fMRI volumes into 2D videos via "cortical flat maps" and feeding them into a standard spacetime MAE-ViT, the authors develop CortexMAE trained on 2.1K hours of HCP data. It significantly outperforms SOTA in cognitive state decoding, validating that flat maps represent the "Goldilocks zone" between voxels (volume) and region-averaging (parcellation). The study also releases Brainmarks, the first open-source fMRI foundation model benchmark, providing the first systematic scaling law for fMRI models and an honest null result showing that individual trait prediction still fails to outperform simple functional connectivity baselines.

## Background & Motivation

**Background**: The neuroscience community aims to use fMRI combined with large-scale models to decode brain activity (diagnosis, behavioral prediction, visual reconstruction). Existing fMRI self-supervised foundation models (BrainLM, Brain-JEPA, NeuroSTORM, SwiFT, etc.) mostly use **parcellation** representations (averaging 3D brain volumes into 100-400 brain regions to obtain 1D time series vectors) or **volume** representations (directly processing 4D spatiotemporal MRI data).

**Limitations of Prior Work**: (1) Parcellation is computationally cheap but suffers from **severe information loss**, where cm-scale brain regions are compressed into a single scalar, losing 99% of dimensions. (2) Volume preserves all information, but the sequence length is massive (over 2000 tokens after patching one fMRI volume), leading to explosive training compute and I/O overhead. (3) The fMRI foundation model field lacks **reproducible benchmarks**, with different studies using proprietary datasets, preprocessing, and evaluation settings. (4) Prior trait prediction papers often report beating baselines, but the baselines used are often weak compared to traditional methods like "simple functional connectivity (FC) + logistic regression."

**Key Challenge**: fMRI data is inherently a 4D spatiotemporal volume, while standard ViTs assume 2D inputs. One must either learn 4D data at high cost or use strong inductive biases (parcellation) that lose information. Is there an intermediate representation that **preserves all cortical signals while providing a ViT-friendly 2D input**?

**Goal**: (i) Identify the "Goldilocks" input representation for fMRI; (ii) train a set of comparative foundation models using standard ViT + MAE; (iii) establish an open-source, reproducible fMRI foundation model benchmark (Brainmarks); (iv) conduct the first systematic data/model scaling law study for fMRI self-supervision.

**Key Insight**: Neuroscience has long utilized **cortical flat maps**, which flatten the 2D manifold of the cerebral cortex (a 2-4mm thick folded sheet) onto a planar grid. This preserves full cortical BOLD signals without averaging details while generating a $224 \times 560$ 2D "image" that can be processed as a video by a spacetime ViT.

**Core Idea**: Projects 3D fMRI into 2D videos using cortical flat maps and applies off-the-shelf MAE-st training. By **changing only the patch embedding without altering the ViT architecture**, this simple yet overlooked choice achieves SOTA results, providing the first fMRI scaling law and open-source benchmark.

## Method

### Overall Architecture
The model consists of MAE-st (Feichtenhofer et al. 2022) with three interchangeable patch embedding input heads. The pipeline includes: (1) HCP-YA data preprocessing (FreeSurfer / fMRIPrep surface mapping) to obtain time series on the cortical surface mesh; (2) projecting the surface onto a planar grid via pycortex to obtain a 16-frame $\times 224 \times 560$ fMRI flat map video; (3) extracting $p_t \times 16 \times 16$ spatiotemporal patches (default $p_t = 4$) with a 0.9 mask ratio (tube masking); (4) the ViT-B encoder processes sparse observed patches while the decoder reconstructs masked patches using MSE loss (calculated only on non-background pixels); (5) after pre-training, the decoder is discarded, and the encoder output serves as features for downstream trait/state prediction via linear or attentive probes. Parcellation MAE (Schaefer-400) and volume MAE (4D patches) are trained simultaneously for rigorous comparison.

### Key Designs

1. **Cortical Flat Map Patch Embedding (Novelty)**:
    - **Function**: Converts 3D fMRI volumes into 2D image videos, allowing standard spacetime ViTs to process full cortical signals directly.
    - **Mechanism**: Utilizes surface-based workflows (FreeSurfer + fMRIPrep) to map fMRI signals from 3D voxels to cortical mesh vertices. Pycortex flat maps then flatten the mesh into a 2D grid, resampled to $224 \times 560$. Each timestep is a 2D frame. Patches of size $p_t \times 16 \times 16$ are used, where zero-valued background patches are excluded from computation and loss.
    - **Design Motivation**: Parcellation compresses signals by $\sim 100 \times$, while volume processing involves $132\text{K}$ voxels with high sparsity and redundancy. Flat maps provide a middle ground, preserving $\sim 77\text{K}$ cortical signals while using efficient 2D ViT processing. The sequence length is 364, comparable to volume (465) and parcellation (400) methods, but with better training bandwidth and throughput.

2. **Head-to-Head Comparison of Three Representations**:
    - **Function**: Places parcellation, flat, and volume representations on equal footing using the same architecture, data, and protocols.
    - **Mechanism**: All models use a ViT-B encoder, 16-frame inputs, and a 0.9 mask ratio. Differences are limited to patch embeddings: $p_t \times 1$ for parcel, $p_t \times 16 \times 16$ for flat, and $p_t \times 8 \times 8 \times 8$ for volume. Each variant is trained 8 times, and evaluated across 8 downstream datasets (4 clinical, sex, age, Task21, NSD COCO24).
    - **Design Motivation**: Previous studies rarely compared representations fairly. This provides the first multi-representation fMRI MAE family with higher credibility.

3. **Brainmarks Open-Source Benchmark Suite**:
    - **Function**: Provides a standardized benchmark for fMRI foundation models where all methods can be evaluated on uniform data.
    - **Mechanism**: Includes 6 existing models (SwiFT, BrainLM, etc.) and the CortexMAE family across 7 public datasets. It uses linear probes with 100 random splits for small-sample trait prediction and attentive probes with fixed splits for large-sample state prediction. 
    - **Design Motivation**: Addresses the reproducibility crisis in fMRI model evaluation by using a unified protocol to determine which methods are truly superior.

### Loss & Training
Pre-training uses MAE MSE loss on masked patches. Data normalization is critical: z-score per voxel/ROI time series (coordinate norm) and per-frame spatial z-score (frame norm) to remove static noise from the 1-2% BOLD fluctuation. Temporal patch size $p_t = 4$; default training for 625K steps with a batch size of 32 (512 frames). Downstream trait prediction utilizes average-pooled embeddings with logistic regression, while state prediction uses attentive probes.

## Key Experimental Results

### Main Results
Probe accuracy across 8 downstream tasks (average of 8 pre-training seeds):

| Dataset | Parcel | Flat | Volume | FC Baseline |
|---|---|---|---|---|
| ABIDE (ASD Diagnosis) | 62.0 | 61.4 | 60.4 | 59.8 |
| ADHD200 | 56.8 | 59.2 | 58.8 | 57.0 |
| ADNI (AD) | 61.6 | 62.4 | 64.3 | 58.6 |
| PPMI (PD) | 61.4 | 58.8 | 59.1 | 58.0 |
| HCP-A Age | 44.2 | 47.5 | **53.4** | 45.6 |
| HCP-A Sex | 71.2 | **87.4** | 86.3 | 81.9 |
| HCP-YA Task21 (State) | 97.5 | **98.9** | 96.2 | 82.4 |
| NSD COCO24 (Decoding) | 27.5 | **31.0** | 27.7 | 7.4 |

Summary: (1) **Flat maps dominate in dynamic state decoding** (Task21, COCO24, Sex); (2) Volume performs better in age prediction (likely due to capturing structural cues like cortical thickness); (3) Clinical diagnosis performance is stagnant across all methods, barely exceeding the FC baseline, highlighting limitations in small-sample clinical transfer.

### Ablation Study

| Configuration | Observation |
|---|---|
| Full flat map MAE | Baseline |
| No frame normalization | Global signal drift contaminates features; accuracy drops |
| No coordinate normalization | Static voxel differences dominate; state decoding fails |
| Tube masking → random masking | Temporal interpolation leaks info; task becomes trivial |
| Mask ratio 0.5 → 0.9 | High ratio forces structural representation learning; better downstream |
| Increased encoder depth | Saturates after depth 9 ($\sim 37\text{M}$ parameters) |
| Increased pretrain data | Follows power law (exponent -0.01) on HCP; saturates on OOD NSD |

### Key Findings
- **fMRI strictly follows data scaling laws, but the exponent is ten times weaker than LLMs** (-0.01 vs -0.1), suggesting diminishing marginal returns for fMRI scaling.
- Model scaling saturates at depth 9 ($\sim 37\text{M}$ parameters), which is the capacity limit for the $2\text{K}$-hour HCP-YA dataset.
- Models spontaneously learn the Default Mode Network (DMN): the first principal component of position embeddings aligns with known functional connectivity gradients.
- **Honest null result**: No fMRI foundation model significantly outperforms simple FC + linear methods on individual trait prediction.

## Highlights & Insights
- **Elegant Engineering**: Changing only the patch embedding to project 3D volumes to 2D manifolds avoids custom architectures, making it a viable strategy for future ViT-based fMRI research.
- **Goldilocks Zone**: Illustrates a classic trade-off in representation learning. Cortical flat maps exploit domain geometry (the 2D nature of the cortex) to find an optimal balance.
- **Scientific Integrity**: The release of Brainmarks and the admission of the null result in trait prediction provide much-needed honesty in a field plagued by "SOTA" claims on weak baselines.
- **Emergent DMN**: Self-supervised representations corresponding to known neurobiological structures provide interpretability.

## Limitations & Future Work
- The HCP-YA dataset is restricted to healthy young adults, leading to narrow pre-training distribution and poor OOD generalization.
- Clinical diagnostic results remain near 60%, suggesting foundation models currently struggle with small-sample clinical transfer.
- Flat map projection **discards subcortical structures** (thalamus, basal ganglia), which are vital for certain clinical tasks; volume models retain an advantage here.
- The study does not explore multi-modal fMRI (task + rest + diffusion) joint pre-training.

## Related Work & Insights
- **vs BrainLM / Brain-JEPA**: These are parcellation-based; CortexMAE Flat preserves full cortical signals and is significantly stronger in state decoding.
- **vs SwiFT / NeuroSTORM**: These are volume-based models; they are computationally expensive but retain advantages in structural tasks like age prediction.
- **vs FC Baselines**: Proves that deep fMRI models have yet to truly surpass simple FC baselines in trait prediction, serving as a wake-up call for the community.

## Rating
- Novelty: ⭐⭐⭐⭐ (Uses historical tools in a modern ViT-friendly way).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Rigorous comparison across representations, models, and datasets).
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic and highly persuasive visuals).
- Value: ⭐⭐⭐⭐⭐ (Brainmarks and scaling laws are landmark contributions).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy](../../CVPR2026/medical_imaging/muvit_multi-resolution_vision_transformers_for_learning_across_scales_in_microsc.md)
- [\[AAAI 2026\] WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images](../../AAAI2026/medical_imaging/wdt-md_wavelet_diffusion_transformers_for_microaneurysm_detection_in_fundus_imag.md)
- [\[AAAI 2026\] FunKAN: Functional Kolmogorov-Arnold Network for Medical Image Enhancement and Segmentation](../../AAAI2026/medical_imaging/funkan_functional_kolmogorov-arnold_network_for_medical_image_enhancement_and_se.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](../../CVPR2026/medical_imaging/forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)
- [\[ICCV 2025\] Scaling Tumor Segmentation: Best Lessons from Real and Synthetic Data](../../ICCV2025/medical_imaging/scaling_tumor_segmentation_best_lessons_from_real_and_synthetic_data.md)

</div>

<!-- RELATED:END -->
