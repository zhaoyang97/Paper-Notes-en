---
title: >-
  [Paper Note] CycleULM: A Unified Label-Free Deep Learning Framework for Ultrasound Localisation Microscopy
description: >-
  [CVPR 2025][Medical Imaging][ultrasound localisation microscopy] CycleULM is proposed as the first unified, label-free deep learning framework for ultrasound localization microscopy (ULM). It bridges the simulation-to-real domain gap by employing CycleGAN to learn a physics-informed bidirectional translation between contrast-enhanced ultrasound (CEUS) frames and a simplified microbubble (MB) domain. This delivers improvements in MB localization accuracy by up to 40% recall an…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "ultrasound localisation microscopy"
  - "CycleGAN"
  - "domain translation"
  - "microbubble"
  - "super-resolution"
date: 2026-05-08
content_hash: af8d7dcc15967548
---

# CycleULM: A Unified Label-Free Deep Learning Framework for Ultrasound Localisation Microscopy

**Conference**: CVPR 2025  
**arXiv**: [2603.09840](https://arxiv.org/abs/2603.09840)  
**Code**: None  
**Area**: Medical Imaging  
**Keywords**: ultrasound localisation microscopy, CycleGAN, domain translation, microbubble, super-resolution

## TL;DR
CycleULM is proposed as the first unified, label-free deep learning framework for ultrasound localization microscopy (ULM). It bridges the simulation-to-real domain gap by employing CycleGAN to learn a physics-informed bidirectional translation between contrast-enhanced ultrasound (CEUS) frames and a simplified microbubble (MB) domain. This delivers improvements in MB localization accuracy by up to 40% recall and 46% precision, while enabling real-time processing at 18.3 fps.

## Background & Motivation
**Background**: Ultrasound localization microscopy (ULM) achieves microvascular imaging beyond the acoustic diffraction limit by localizing and tracking microbubble (MB) contrast agents injected in vivo. It has been validated in various tissues, including tumors, brain, kidney, and heart, and has recently expanded to human clinical studies.

**Limitations of Prior Work**: (a) MB signals are difficult to separate from background noise and clutter; (b) overlapping microbubbles reduce localization quality; (c) long acquisition times lead to motion artifacts; and (d) processing large data volumes requires high computational costs. Although deep learning shows promise, the primary hurdle is the **lack of annotated in vivo data**—manual annotation of MB positions is unfeasible, and pseudo-labels inherit errors from traditional algorithms.

**Key Challenge**: Existing deep learning methods rely on simulated data for training (e.g., Gaussian PSF models, Field II). However, simulators cannot fully replicate real-world conditions (such as acoustic wave propagation, non-linear MB responses, and heterogeneous tissues). This causes a significant **simulation-to-real domain gap**, leading to a severe performance drop when models are transferred from simulation to real-world data.

**Goal**: (a) How to train deep learning models for ULM without annotated data? (b) How to bridge the domain gap between simulated training data and real CEUS data? (c) How to achieve real-time processing while preserving high image quality?

**Key Insight**: Rather than attempting to build more realistic simulators, this work utilizes CycleGAN to learn a bidirectional translation from the complex real CEUS domain to a simplified MB-only domain. This allows downstream localization and tracking to be performed in a controllable and interpretable simplified domain.

**Core Idea**: Learn a CEUS-to-MB-only domain translation through self-supervised cycle-consistent adversarial training, eliminating the reliance on high-fidelity simulators and annotated data.

## Method

### Overall Architecture
CycleULM comprises three modular neural networks: MB-DT (Microbubble Domain Translator, based on CycleGAN architecture), which translates CEUS frames to a simplified MB-only domain; MBL-Net (Microbubble Localization Network), which performs sub-pixel MB localization in the translated MB-only domain; and MBT-Net (Microbubble Trajectory Network), which tracks MB trajectories and estimates velocities from continuous sequences of MB-only frames. These three networks can serve as plug-in modules to replace corresponding steps in traditional ULM pipelines, or they can be combined into an end-to-end framework (CycleULM-E2E).

### Key Designs

1. **MB-DT (Microbubble Domain Translator)**:

    - **Function**: Translates physically complex CEUS frames into a simplified domain containing only MB signals, suppressing background clutter and shrinking the effective PSF.
    - **Mechanism**: Based on CycleGAN, the forward generator $G_{AB}$ maps three consecutive CEUS frames to a single MB-only frame, while the backward generator $G_{BA}$ reconstructs the CEUS frames. Training uses unpaired data, with the loss containing two cycle-consistency losses (L1), two adversarial losses (LSGAN), and a similarity loss $\mathcal{L}_{\text{sim}} = \|G_{AB}(a) - a\|_1$ to prevent the loss of MB signals.
    - **Design Motivation**: Leverages three-frame temporal context to enhance the discrimination between MB signals and clutter. Setting the synthetic PSF to be 0.5 times smaller encourages the network to generate narrower MB signals, directly boosting downstream localization accuracy.

2. **MBL-Net (Microbubble Localisation Network)**:

    - **Function**: Performs sub-pixel localization of microbubbles from MB-only images.
    - **Mechanism**: A U-Net-based multi-task head network that outputs seven maps: a probability map $p$, an intensity map $I$, two sub-pixel offset maps $\Delta x, \Delta y$, and three uncertainty maps $\sigma_I, \sigma_x, \sigma_y$. The loss consists of a BCE existence loss and a Gaussian mixture model-based negative log-likelihood localization loss (borrowed from DECODE).
    - **Design Motivation**: Trained purely on synthetic MB-only data. Since MB-DT has eliminated the domain gap, the synthetic data generalizes seamlessly to real-world data.

3. **MBT-Net (Microbubble Trajectory Network)**:

    - **Function**: Estimates trajectory probability maps alongside horizontal and vertical velocity maps from an 8-frame sequence of continuous MB-only images.
    - **Mechanism**: A U-Net + ConvLSTM architecture that embeds ConvLSTM blocks into the bottleneck and skip connections to capture spatio-temporal correlations, outputting 4$\times$ upsampled trajectory and velocity maps. The loss consists of a BCE trajectory loss and a weighted MSE velocity loss (weighted by the normalized trajectory probability to address the sparsity of velocity maps).
    - **Design Motivation**: ConvLSTM replaces fully connected LSTMs to preserve spatial structures while reducing the parameter count.

### Loss & Training
- MB-DT: $\mathcal{L}_G = \mathcal{L}_{G_{AB}} + \mathcal{L}_{G_{BA}} + 5\mathcal{L}_{\text{cyc1}} + 5\mathcal{L}_{\text{cyc2}} + \mathcal{L}_{\text{sim}}$
- All networks are trained on small patches (40×40 or 80×80) and applied to full frames during inference; patch-based training increases data diversity and allows for targeted ROI training.

## Key Experimental Results

### Main Results (ULTRA-SR Simulation Dataset, MB Localization Performance)

| Method | Recall Gain | Precision Gain | Localization Error Improvement |
|------|-----------|---------------|-------------|
| CycleULM-NCC vs NCC | +32% | +46% | -14.0 µm (31%) |
| CycleULM-Decon vs Decon | +7% | - | -4.5 µm (13%) |
| CycleULM-Loc vs NCC | +40% | +46% | -16.1 µm (36%) |
| CycleULM-Loc vs Decon | +8% | +26% | -3.4 µm (10%) |

### Image Quality Improvement

| Metric | Original CEUS | After MB-DT Translation | Gain |
|------|---------|-------------|------|
| CNR (ULTRA-SR) | 7.2 dB | 16.4 dB | +9.2 dB |
| CNR (Rabbit Kidney in vivo) | 14.4 dB | 29.7 dB | **+15.3 dB** |
| FWHM (ULTRA-SR) | 355 µm | 178 µm | 2× Resolution Gain |
| FWHM (Rabbit Kidney in vivo) | 725 µm | 294 µm | **2.5× Resolution Gain** |

### Processing Speed (Rabbit Kidney Data, 500 frames)

| Method | Processing Time (s) | Throughput (fps) | Speedup |
|------|-----------|------------|--------|
| Baseline 1 (Threshold + NCC) | 174.2 | 2.9 | 1× |
| Baseline 2 (Threshold + Decon) | 394.6 | 1.3 | - |
| CycleULM-E2E | **27.2** | **18.4** | **6.4×/14.5×** |

### Key Findings
- After MB-DT translation, even traditional localization methods (NCC/Decon) achieve significantly improved performance, demonstrating the intrinsic value of domain translation itself.
- CycleULM-E2E yields the best reconstruction quality while being the fastest (14.5× speedup), as it bypasses traditional denoising and localization steps.
- Cross-acquisition generalization: An MB-DT trained on one rabbit kidney dataset can be directly applied to another rabbit kidney dataset with a different imaging perspective without any fine-tuning.

## Highlights & Insights
- **Domain translation eliminates domain gaps**: Rather than trying to construct better physical simulators, this approach learns to map real-world data into a controllable domain. Synthesizing training data within this simplified domain is fully controllable, allowing downstream networks to generalize seamlessly. This concept can be extended to other fields where there is a large simulation-to-real gap (e.g., object detection trained on synthetic computer-rendered data).
- **Modular design aligned with traditional pipelines**: The three networks correspond to the three classical steps of denoising, localization, and tracking. They can be replaced individually or integrated end-to-end, lowering the barrier to clinical adoption.
- **Similarity loss to prevent MB omission**: A simple self-supervised L1 term $\|G_{AB}(a) - a\|_1$ constrains the translated results from shifting too far from the original frames, effectively preventing false positives and missed detections.

## Limitations & Future Work
- CycleULM-E2E outputs image-level velocity/trajectory representations rather than explicit MB trajectory objects, limiting its use for trajectory-level filtering and statistical analysis.
- PSF estimation relies on manually selecting 10 isolated MBs to fit a Gaussian profile, which lacks sufficient automation.
- Validated only on two in vivo datasets (rabbit kidney and rat brain); more human clinical data is required to confirm generalizability.
- The shallow CycleGAN architecture (2-layer downsampling) may capture insufficient contextual information in scenarios requiring large receptive fields.
- Both training and inference are performed on 2D frames, without extension to 3D volumetric ultrasound data.
- The assumption of an isotropically smaller PSF in the synthetic MB-only domain might not hold under certain imaging configurations.

## Related Work & Insights
- **vs Shin et al. (2024)**: They used GANs to generate more realistic MB signals but required manual selection of MB patches and neglected overlapping MB interactions. CycleULM is fully automated and ensures physical plausibility via cycle consistency.
- **vs DECODE (Speiser 2021)**: DECODE is a sub-pixel localization method in the single-molecule localization microscopy (SMLM) field. CycleULM's MBL-Net borrows its loss function design to transfer these concepts to the ultrasound domain.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The first unified label-free ULM deep learning framework; the approach of utilizing domain translation to eliminate simulation-to-real gaps is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Simulations combined with three in vivo datasets (rabbit kidney $\times$ 2 + rat brain), multi-method comparisons, comprehensive metric evaluations, speed analyses, and cross-dataset generalization.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Written in a Nature-sub-journal style with clear logic and high-quality figures.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses key bottlenecks in the ULM field (lack of annotations and domain gaps), with real-time performance opening up clinical execution possibilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[CVPR 2025\] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](deep_learning-based_assessment_of_the_relation_between_the_third_molar_and_mandi.md)
- [\[CVPR 2025\] VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging](vista3d_a_unified_segmentation_foundation_model_for_3d_medical_imaging.md)
- [\[CVPR 2025\] CARL: A Framework for Equivariant Image Registration](carl_a_framework_for_equivariant_image_registration.md)
- [\[CVPR 2025\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)

</div>

<!-- RELATED:END -->
