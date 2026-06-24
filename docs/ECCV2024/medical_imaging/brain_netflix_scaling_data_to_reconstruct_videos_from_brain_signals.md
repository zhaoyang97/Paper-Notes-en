---
title: >-
  [Paper Note] Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals
description: >-
  [ECCV 2024][Medical Imaging][Brain-to-video reconstruction] This paper proposes a novel method for reconstructing videos from functional magnetic resonance imaging (fMRI) signals. Through multi-dataset, multi-subject training and a three-stage pipeline utilizing pre-trained text-to-video and video-to-video models, it achieves state-of-the-art (SOTA) video reconstruction capabilities across both datasets and subjects.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Brain-to-video reconstruction"
  - "fMRI"
  - "Multi-subject learning"
  - "Diffusion models"
  - "Neural decoding"
date: 2026-05-08
content_hash: ca0b5ac73d960b93
---

# Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Others  
**Keywords**: Brain-to-video reconstruction, fMRI, Multi-subject learning, Diffusion models, Neural decoding

## TL;DR

This paper proposes a novel method for reconstructing videos from functional magnetic resonance imaging (fMRI) signals. Through multi-dataset, multi-subject training and a three-stage pipeline utilizing pre-trained text-to-video and video-to-video models, it achieves state-of-the-art (SOTA) video reconstruction capabilities across both datasets and subjects.

## Background & Motivation

**Background**: Brain-to-stimuli reconstruction is an active research direction at the intersection of neuroscience and computer vision. In recent years, numerous studies have successfully reconstructed static images resembling the visual stimuli watched by subjects from fMRI signals. However, video reconstruction is a much more challenging task as it requires capturing both spatial visual features and temporal dynamic information simultaneously.

**Limitations of Prior Work**: (1) Existing methods are typically subject-specific, requiring retraining for each new subject, which limits generalization. (2) Most methods are evaluated only on a single dataset, leaving cross-dataset performance unknown. (3) fMRI data acquisition is expensive and time-consuming, leading to limited data volume for a single subject, which constrains the upper bound of model performance. (4) Mapping from brain signals to video requires reconstructing a large number of latent and conditional vectors simultaneously, resulting in insufficient regression accuracy.

**Key Challenge**: A fundamental contradiction exists between the scarcity of fMRI data and the vast amount of data required for video reconstruction. Single-subject data is insufficient to learn a general neural-to-visual mapping, while fMRI signals vary significantly across different subjects and datasets (due to different scanners, resolutions, and brain parcellations).

**Goal**: (1) How to effectively aggregate fMRI data from multiple datasets and multiple subjects to scale up the training. (2) How to design a general pipeline to process fMRI data from different sources. (3) How to accurately regress the key vectors required by pre-trained video generation models.

**Key Insight**: The authors argue that data volume is the key bottleneck and propose to scale up the training data through joint training across multiple datasets and multiple subjects. Concurrently, they design a three-stage pipeline that decomposes brain decoding into three sub-problems: semantic alignment, vector regression, and video generation.

**Core Idea**: Scale up the training data volume through multi-subject, multi-dataset training, and accurately transform fMRI signals into inputs for a pre-trained video model via a three-stage pipeline (semantic alignment $\rightarrow$ vector regression $\rightarrow$ video generation) to reconstruct videos.

## Method

### Overall Architecture

The overall pipeline is divided into three stages. The input consists of fMRI signals (voxel activations) recorded while subjects watch videos, and the output is the reconstructed video clips (2-3 seconds).

Stage 1: fMRI alignment. It maps fMRI signals from different subjects and datasets into a unified semantic embedding space (e.g., CLIP space) to eliminate individual and dataset variability.

Stage 2: Vector regression. It regresses key latent and conditional vectors (including text embeddings, image embeddings, etc.) required by the pre-trained video generation model from the aligned semantic embeddings.

Stage 3: Video generation. The regressed vectors are fed into pre-trained text-to-video and video-to-video models to generate reconstructed videos matching the original stimuli.

### Key Designs

1. **Multi-dataset Multi-subject Alignment Strategy**:

    - **Function**: Project fMRI signals from different datasets and subjects into a unified semantic space.
    - **Mechanism**: Learn a linear projection layer for each subject to map their individual fMRI voxel space into a shared CLIP semantic space. Contrastive learning is used during training to bring the fMRI embeddings and corresponding video CLIP embeddings of the same video stimuli closer, while pushing mismatched pairs apart. By sharing subsequent neural layers, data from different subjects can mutually enhance each other.
    - **Design Motivation**: Brain structures and functional area divisions vary across subjects, and scanning parameters differ among datasets, preventing direct data combination. A per-subject linear projection layer preserves individual specificity, while the shared semantic space enables data aggregation.

2. **Key Vector Regression Network**:

    - **Function**: Accurately regress control and latent vectors for the video generation model from the semantic embeddings.
    - **Mechanism**: Design a specialized regression network to estimate key inputs such as text conditioning vectors and image conditioning vectors needed by the text-to-video model. The regression network adopts a residual MLP architecture with distinct regression heads for different vector types. During training, MSE loss combined with contrastive loss is applied to constrain regression accuracy.
    - **Design Motivation**: The input space of video generation models is high-dimensional and highly sensitive; minor regression errors can lead to significant discrepancies in the generated videos. Utilizing a specialized regression network and multiple loss constraints improves regression accuracy.

3. **Three-Stage Decoupled Training**:

    - **Function**: Decompose the complex fMRI-to-video mapping into manageable sub-tasks to improve training stability.
    - **Mechanism**: Train the three stages sequentially—first train the fMRI alignment module, then train the vector regression network, and finally conduct inference using the pre-trained video generation model. Each stage has clear objectives and loss functions to evade the instability of end-to-end training.
    - **Design Motivation**: End-to-end mapping from fMRI to video is excessively complex and spans too many semantic levels. Decoupled training ensures that the mapping at each step remains accurate.

### Loss & Training

- Stage 1 employs a CLIP-style contrastive loss for semantic alignment.
- Stage 2 utilizes MSE regression loss and perceptual loss to constrain vector regression accuracy.
- In multi-subject training, gradients are accumulated across different subjects, effectively expanding the data volume of each update.
- Pre-trained video generation models (text-to-video and video-to-video) are used; no training is conducted during inference.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| Multi-dataset Evaluation | SSIM | SOTA | - | Significant Gain |
| Multi-dataset Evaluation | Semantic Consistency | SOTA | - | Significant Gain |
| Cross-subject Evaluation | Visual Quality | Good | Severe Degradation | Significant Gain |

Qualitative and quantitative evaluations, including crowdsourced human evaluations, were performed on multiple fMRI datasets.

### Ablation Study

| Configuration | Key Metric | Description |
|------|----------|------|
| Single-subject vs. Multi-subject | SSIM Gain | Multi-subject training significantly outperforms single-subject |
| Single-dataset vs. Multi-dataset | Semantic Consistency Gain | Multi-dataset scaling is effective |
| Different Alignment Strategies | Reconstruction Quality | CLIP alignment outperforms other strategies |
| Increasing Subject Count | Performance Curve | More subjects consistently bring performance gains |

### Key Findings

- The improvement from joint multi-dataset and multi-subject training is significant, proving the effectiveness of the data scaling strategy.
- The three-stage pipeline is more stable and reliable than direct end-to-end approaches.
- Crowdsourced human evaluations validate the semantic consistency between reconstructed videos and the original stimuli.
- The authors observe that performance continuously improves as more subjects are added, hinting at the feasibility of zero-shot reconstruction.

## Highlights & Insights

- **Data Scaling Concept**: Introducing the NLP/CV philosophy of "more data, better performance" to brain signal decoding, achieving multi-source data aggregation through an elegant alignment strategy.
- **Three-stage Decoupling**: Decomposing the highly difficult fMRI-to-video mapping into three controllable steps, each supported by mature technologies.
- **Cross-subject Generalization**: Making a significant step toward zero-shot brain decoding, demonstrating the potential of utilizing existing subject data to aid new subjects.
- **Comprehensive Evaluation System**: Including quantitative metrics, qualitative visualizations, and crowdsourced human evaluations.

## Limitations & Future Work

- Temporal resolution limitations of fMRI (TR of approximately 2 seconds) make it unable to capture details of fast visual changes.
- Reconstructed videos are of short duration (2-3 seconds), rendering the model inadequate for handling long videos.
- Reliance on the quality of pre-trained video generation models, where model hallucinations may lead to inaccurate details.
- Performance of zero-shot reconstruction (on unseen subjects) still has substantial room for improvement.
- Privacy and ethical concerns: The development of brain decoding technology must be approached with caution.

## Related Work & Insights

- **Brain Signal Image Reconstruction**: Methods like Mind-Reader and Brain-Diffuser lay the foundation for static image reconstruction.
- **Video Generation Models**: Stable Video Diffusion and other models provide powerful conditional video generation capabilities.
- **Insight**: The data scaling strategy may also prove effective in other data-scarce domains (such as medical imaging).

## Rating

- Novelty: ⭐⭐⭐⭐ Novel idea of scaling training via multiple datasets and subjects, accompanied by a well-designed three-stage pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset evaluation + ablation studies + crowdsourced human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear pipeline description and comprehensive analyses.
- Value: ⭐⭐⭐⭐ Pushes the SOTA of brain-to-video reconstruction and provides an effective paradigm for data scaling strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] CardiacNet: Learning to Reconstruct Abnormalities for Cardiac Disease Assessment from Echocardiogram Videos](cardiacnet_learning_to_reconstruct_abnormalities_for_cardiac_disease_assessment_.md)
- [\[ECCV 2024\] UMBRAE: Unified Multimodal Brain Decoding](umbrae_unified_multimodal_brain_decoding.md)
- [\[ECCV 2024\] Brain-ID: Learning Contrast-agnostic Anatomical Representations for Brain Imaging](brain-id_learning_contrast-agnostic_anatomical_representations_for_brain_imaging.md)
- [\[AAAI 2026\] MindCross: Fast New Subject Adaptation with Limited Data for Cross-subject Video Reconstruction from Brain Signals](../../AAAI2026/medical_imaging/mindcross_fast_new_subject_adaptation_with_limited_data_for_cross-subject_video_.md)
- [\[NeurIPS 2025\] BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](../../NeurIPS2025/medical_imaging/brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)

</div>

<!-- RELATED:END -->
