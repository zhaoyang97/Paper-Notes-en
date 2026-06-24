---
title: >-
  [Paper Note] vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation
description: >-
  [CVPR 2025][Medical Imaging][Vessel segmentation] vesselFM is the first foundation model dedicated to 3D blood vessel segmentation. By integrating three heterogeneous data sources—a curated large-scale real annotated dataset, domain-randomized synthetic data, and flow matching-based generative data—it achieves state-of-the-art (SOTA) results in zero-shot, one-shot, and few-shot segmentation across four clinical imaging modalities.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Vessel segmentation"
  - "foundation model"
  - "domain randomization"
  - "flow matching"
  - "zero-shot generalization"
date: 2026-05-08
content_hash: 13ad12157ac7ec8d
---

# vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2411.17386](https://arxiv.org/abs/2411.17386)  
**Code**: [https://github.com/bwittmann/vesselFM](https://github.com/bwittmann/vesselFM)  
**Area**: Medical Image  
**Keywords**: Vessel segmentation, foundation model, domain randomization, flow matching, zero-shot generalization

## TL;DR

vesselFM is the first foundation model dedicated to 3D blood vessel segmentation. By integrating three heterogeneous data sources—a curated large-scale real annotated dataset, domain-randomized synthetic data, and flow matching-based generative data—it achieves state-of-the-art (SOTA) results in zero-shot, one-shot, and few-shot segmentation across four clinical imaging modalities.

## Background & Motivation

**Background**: 3D vessel segmentation is a crucial task in medical image analysis, applied in the diagnosis and treatment of vascular diseases such as stroke, aneurysm, and coronary artery disease. Although deep learning methods have made progress, a massive domain gap exists among imaging modalities (differing in signal-to-noise ratio, vessel patterns, scales, artifacts, and surrounding tissues), preventing existing models from generalizing to unseen imaging domains.

**Limitations of Prior Work**: (1) Labeling voxel-level segmentation masks for every new dataset is extremely time-consuming and labor-intensive; (2) General medical segmentation foundation models (such as SAM-Med3D, MedSAM-2, VISTA3D) perform poorly on vessel segmentation because vessels exhibit unique thin, elongated tubular structures and tiny scales, which differ dramatically from general organs and structures; (3) Existing vessel segmentation methods are restricted to specific modalities (e.g., only OCTA or only MRA) and perform poorly across modalities.

**Key Challenge**: The features required for vessel segmentation (tubular geometry, multi-scale branching, extremely fine structures) do not match the training objectives of general segmentation models, and the vessel images of different imaging modalities exhibit massive discrepancies (ranging from $\mu$m-scale microscopy to mm-scale CT), which cannot be covered by a single data source.

**Goal**: Build a universal 3D vessel segmentation foundation model that can be directly applied to unseen imaging domains in a zero-shot manner, while supporting efficient few-shot adaptation.

**Key Insight**: A three-pronged data strategy—(1) curating the largest real vessel dataset to cover core modalities; (2) using domain randomization to cover all possible vessel image styles; (3) employing conditional generative models to expand the distribution of real data. These three complementary sources constitute the training data.

**Core Idea**: Train a class-conditioned nnU-Net segmentation model on three heterogeneous data sources ($\mathcal{D}_{\text{real}}$ + $\mathcal{D}_{\text{drand}}$ + $\mathcal{D}_{\text{flow}}$) so that the model learns robust features across various vessel patterns and imaging styles, thereby achieving zero-shot cross-domain vessel segmentation.

## Method

### Overall Architecture

The training data of vesselFM consists of three components: (1) $\mathcal{D}_{\text{real}}$—with over 115,000 $128^3$ voxel patches across 23 classes from 17 data sources, covering multiple modalities such as MRA/CTA/CT/vEM/OCTA/light-sheet microscopy; (2) $\mathcal{D}_{\text{drand}}$—synthetic data generated via domain randomization, where the foreground is transformed from real vessel molds and the background is filled with Perlin noise textures to simulate various imaging conditions; (3) $\mathcal{D}_{\text{flow}}$—synthetic image-mask pairs sampled from a mask- and class-conditioned flow matching generative model. The segmentation model uses a class-conditioned nnU-Net.

### Key Designs

1. **Large-Scale Real Dataset Curating ($\mathcal{D}_{\text{real}}$)**:

    - **Function**: Provide diverse, high-quality real vessel image-segmentation pairs as the core training data.
    - **Mechanism**: Data is collected from 17 public sources and subdivided into 23 categories based on tissue type, imaging modality, and protocol (each category is assigned a unique class ID $c \in \{1, \ldots, 23\}$). It covers multiple anatomical regions like human/mouse brain, kidney, and liver, and multiple modalities like MRA/CTA/CT/vEM/OCTA/two-photon microscopy/light-sheet microscopy. All are uniformly preprocessed into $128^3$ patches with annotation quality scores of 6-10.
    - **Design Motivation**: The performance of foundation models highly depends on the diversity and scale of training data. Different protocols of the same modality also introduce domain gaps, which is why datasets with the same modality but different protocols are deliberately included. Four datasets (SMILE-UHURA/BvEM/OCTA/MSD8) are excluded from the training set and dedicated to evaluating zero-shot performance.

2. **Domain Randomization ($\mathcal{D}_{\text{drand}}$)**:

    - **Function**: Comprehensively cover the generic domain of 3D vessel images to enhance model robustness to unseen imaging conditions.
    - **Mechanism**: A three-step pipeline. **Foreground generation**: Based on 1137 real vessel mold voxel patches, spatial transformations (random cropping/flipping/rotation/dilation/scaling/elastic deformation/smoothing) are applied to simulate diverse vessel patterns, followed by artifact transformations (bias field/Gaussian noise/smoothing/dropout/offset/convex hull/identity) to simulate foreground artifacts. **Background generation**: Background images containing spheres, polyhedra, or no geometry are constructed and textured using Perlin noise. **Foreground-background merging**: Merged via addition or replacement, followed by intensive intensity transformations (bias field/Gaussian noise/k-space spikes/contrast adjustment/Rician noise/Gibbs noise/sharpening/histogram transformation).
    - **Design Motivation**: Although real data is of high quality, its coverage is limited. Domain randomization fills the uncovered spaces of real data by randomizing all possible visual attributes, preparing the model for any new imaging conditions.

3. **Conditional Generation via Flow Matching ($\mathcal{D}_{\text{flow}}$)**:

    - **Function**: Expand the distribution of real data and generate high-fidelity image-mask pairs.
    - **Mechanism**: Train a mask- and class-conditioned 3D flow matching model (based on the Med-DDPM architecture) that takes noise $x_0 \sim \mathcal{N}(0, I)$ + segmentation mask (channel-concatenated) + class embedding as input, mapping the noise to the real image distribution via the learned velocity field. New images can be sampled using synthetic masks from $\mathcal{D}_{\text{drand}}$ as conditions (labeled as $\tilde{c}$), or real masks from $\mathcal{D}_{\text{real}}$ can be used to generate pseudo-samples of one modality in another modality's style.
    - **Design Motivation**: Domain randomization is broad but lacks realism, while flow matching generates images almost indistinguishable from real ones (Fig. 5b), replenishing realism. Generating different modality images under different class conditions from the same mask dramatically enhances data volume and diversity.

### Loss & Training

The segmentation model utilizes the nnU-Net framework, with a loss function combining Dice and CE. The flow matching model is trained using the conditional flow matching (CFM) objective. The model receives a class embedding as an additional condition, and uses the domain-randomized class ($c=0$) or selects the closest class based on the target domain during zero-shot inference.

## Key Experimental Results

### Main Results

| Method | OCTA Dice | OCTA clDice | BvEM Dice | SMILE-UHURA Dice | MSD8 Dice |
|------|----------|-------------|----------|-----------------|----------|
| tUbeNet| 36.01 | 23.64 | 10.03 | 48.32 | 5.13 |
| VISTA3D | 13.60 | 3.72 | 0.94 | 5.05 | 23.83 |
| SAM-Med3D | 6.74 | 6.56 | 5.98 | 2.12 | 7.94 |
| MedSAM-2 | 28.56 | 15.76 | 10.92 | 3.85 | 14.53 |
| **vesselFM** | **46.94** | **67.07** | **67.49** | **74.66** | **29.69** |

### Ablation Study

| Method | OCTA Dice | BvEM Dice | SMILE Dice | MSD8 Dice |
|------|----------|----------|------------|----------|
| vesselFM (from scratch) | 65.57 | 63.85 | 37.99 | 27.13 |
| **vesselFM (pretrained)** | **72.10** | **78.27** | **76.43** | **36.88** |

### Key Findings

- **General Foundation Models Fail Miserably on Vessel Segmentation**: SAM-Med3D achieves only 2-8% Dice in zero-shot settings, and VISTA3D scores 0.94% on BvEM, indicating general models are entirely incapable of processing vessels. vesselFM outperforms others by large margins on all 4 datasets (SMILE-UHURA: 74.66% vs. 48.32% for the runner-up).
- **Complementarity of Three Data Sources**: Domain randomization provides the foundation for generalization, real data provides precision, and flow matching expands the distribution scope. Ablation studies show that removing any one of them leads to performance degradation.
- **Significant Gap Between Pretraining and Training from Scratch**: Under the one-shot scenario, pretrained vesselFM achieves 76.43% on SMILE-UHURA vs. 37.99% from scratch, proving that pretraining learns transferable vascular features.
- **Even Greater Advantage on clDice (Topology Connectivity Metric)** (OCTA: 67.07% vs. 23.64% for the runner-up), showing that vesselFM maintains the topological integrity of vessel structures.

## Highlights & Insights

- **The Three-Pronged Data Strategy is Highly Inspiring**: Real data ensures the core quality, domain randomization guarantees generalization coverage, and the generative model ensures realistic expansion. This data engineering paradigm can be transferred to any segmentation task requiring cross-domain generalization (e.g., road cracks, nerve fibers).
- **Novel Application of Flow Matching for Conditional 3D Medical Image Generation**: Utilizing the same mask to generate images of different modalities under different class conditions serves as a highly efficient data augmentation approach.
- The detailed design of artifact transformations in domain randomization (bias field, k-space spike, Rician noise, etc.) demonstrates a deep understanding of the physical properties of medical imaging.

## Limitations & Future Work

- Zero-shot performance is still limited on certain modalities (MSD8 liver CT is only 29.69%), possibly due to the low proportion of CT data in the training set.
- Currently employing nnU-Net as the backbone; future work could explore the scaling effects of larger models (such as 3D Swin Transformer).
- Parameters for domain randomization require manual design, which might miss certain rare imaging conditions.
- Test-time adaptation strategies (such as TTT) could be further explored to adapt to new domains during inference.

## Related Work & Insights

- **vs SAM-Med3D**: A general 3D medical segmentation model pretrained on 94 datasets, yet almost entirely failing on vessel segmentation (zero-shot Dice of 2-8%). The specialized design of vesselFM proves that "task-specific foundation models" are more valuable than "general foundation models" on specific tasks.
- **vs MedSAM-2**: A method that processes 3D images as video based on SAM 2, which also yields poor zero-shot performance on vessels (3.85-28.56%). The tubular topology of vessels is incompatible with SAM's prompt design.
- **vs tUbeNet**: A model pretrained on specific vessel data, showing some zero-shot capability but covering limited modalities. vesselFM comprehensively outperforms it through larger scale and more diverse training data.

## Rating

- Novelty: ⭐⭐⭐⭐ Combination strategy of three heterogeneous data sources and the application of conditional generation via flow matching are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four evaluation datasets, three learning paradigms (zero/one/few-shot), and comparisons with five baselines.
- Writing Quality: ⭐⭐⭐⭐ Detailed description of data strategies, with clear figures and tables.
- Value: ⭐⭐⭐⭐⭐ The first vessel-specific foundation model, with open-source models and code of massive value to the clinical and research communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging](vista3d_a_unified_segmentation_foundation_model_for_3d_medical_imaging.md)
- [\[CVPR 2025\] Deep Learning Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)
- [\[CVPR 2025\] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)
- [\[CVPR 2025\] Show and Segment: Universal Medical Image Segmentation via In-Context Learning](show_and_segment_universal_medical_image_segmentation_via_in-context_learning.md)
- [\[CVPR 2025\] UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC](unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)

</div>

<!-- RELATED:END -->
