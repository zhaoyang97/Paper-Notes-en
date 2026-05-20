---
title: >-
  [Paper Note] Beyond Brain Decoding: Visual-Semantic Reconstructions to Mental Creation Extension Based on fMRI
description: >-
  [ICCV 2025][Medical Imaging][fMRI brain decoding] This paper proposes NeuroCreat — a multimodal brain architecture that integrates the visual and textual capabilities of LLMs — extending fMRI decoding from single-task vi…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "fMRI brain decoding"
  - "visual-semantic reconstruction"
  - "mental creation"
  - "large language models"
  - "multimodal brain architecture"
date: 2026-05-08
content_hash: 3edb1b2390344523
---

# Beyond Brain Decoding: Visual-Semantic Reconstructions to Mental Creation Extension Based on fMRI

**Conference**: ICCV 2025
**arXiv**: No preprint (CVF Open Access only)  
**CVF**: [Paper Page](https://openaccess.thecvf.com/content/ICCV2025/html/Jing_Beyond_Brain_Decoding_Visual-Semantic_Reconstructions_to_Mental_Creation_Extension_Based_ICCV_2025_paper.html) / [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Jing_Beyond_Brain_Decoding_Visual-Semantic_Reconstructions_to_Mental_Creation_Extension_Based_ICCV_2025_paper.pdf)
**Code**: No public code  
**Authors**: Haodong Jing, Dongyao Jiang, Yongqiang Ma, Haibo Hua, Bo Huang, Nanning Zheng (Xi'an Jiaotong University)
**Area**: Medical Imaging / Brain Science
**Keywords**: fMRI brain decoding, visual-semantic reconstruction, mental creation, large language models, multimodal brain architecture

## TL;DR
This paper proposes NeuroCreat — a multimodal brain architecture that integrates the visual and textual capabilities of LLMs — extending fMRI decoding from single-task visual stimulus reconstruction to three levels: **image reconstruction + text captioning + mental creation**. A Prompt Variant Alignment (PVA) module is introduced to effectively bridge the gap between low-resolution fMRI signals and high-level semantic representations.

## Background & Motivation
Decoding visual information from fMRI signals is a key avenue for understanding how the brain represents the world, and represents a frontier in artificial general intelligence (AGI) research. The core limitations of existing work are:

1. **Single-objective focus**: Mainstream methods (MindEye, MindEye2, Brain-Diffuser, Unibrain, etc.) almost exclusively focus on the single task of reconstructing the perceived image from fMRI signals, with little exploration of generating textual descriptions or novel content from brain signals.
2. **Insufficient semantic utilization**: Most methods map fMRI signals to CLIP or SDXL latent spaces for image reconstruction, lacking fine-grained mining of high-level semantics — particularly limited in text generation capability.
3. **Far from "mental creation"**: The human brain not only reconstructs perceived scenes but also creates entirely new mental images from memory and experience. This "creative" capacity remains largely unexplored in existing brain decoding methods.

A preceding work from the same group, "See Through Their Minds" (arXiv:2403.06361), explored transferable neural representation learning for cross-subject fMRI decoding, laying the foundation for further exploitation of rich semantic content in fMRI signals.

## Core Problem
**How can fMRI brain decoding be extended from single-task visual stimulus reconstruction to multi-level, multimodal brain signal understanding — encompassing visual reconstruction, semantic captioning, and mental creation?**

Specific challenges include:
- How to extract fine-grained semantic information from fMRI signals given their low resolution and high noise?
- How to handle the divergence between different output modalities (image vs. text vs. creative content)?
- How to leverage the powerful multimodal capabilities of LLMs to enhance brain decoding?

## Method

### Overall Architecture
NeuroCreat is a **multiplexed neural decoding model** whose core mechanism integrates LLMs with brain decoding to produce outputs at three levels:

1. **Reconstruction**: Reconstructing the visual stimulus image from fMRI signals.
2. **Captioning**: Generating a textual description of the perceived image from fMRI signals.
3. **Creation**: Generating novel, previously unseen content from fMRI signals (embodied implementation).

The overall pipeline consists of:
- **fMRI Encoder**: Encodes raw fMRI voxel signals into compact neural feature representations.
- **Prompt Variant Alignment (PVA)**: Aligns and differentiates representations across output modalities.
- **LLM Decoder**: Leverages the visual and textual capabilities of an LLM to transform aligned neural representations into diverse modal outputs.

### Key Designs

#### 1. Prompt Variant Alignment (PVA) Module
This is the core technical innovation of NeuroCreat:
- **Design Motivation**: A substantial gap exists between the low resolution of fMRI signals and the fine-grained requirements of generation targets (detailed images / accurate text); different output modalities impose different demands on neural representations.
- **Mechanism**: Variant prompts are designed to disentangle modality-specific differences, constructing adapted alignment schemes for each output task.
- **Effect**: Effectively mitigates the impact of fMRI low resolution and inter-modality over-coupling.

#### 2. LLM Integration Strategy
- Leverages the visual and textual capabilities of an LLM (presumably from the Vicuna/LLaMA family).
- fMRI-encoded neural features serve as conditioning inputs to the LLM.
- The LLM simultaneously serves captioning and creation tasks, enabling parameter sharing and knowledge transfer.

#### 3. Multi-Task Multiplexed Design
- All three tasks share the fMRI encoder and LLM backbone.
- Task-specific behavior is achieved through different prompt variants in the PVA module.
- The "creation" task is a novel extension — a benchmark for this task is established on the NSD dataset for the first time.

### Loss & Training
The specific loss combination should be verified in the original paper. The following components are inferred:
- **Reconstruction loss**: Pixel-level and/or perceptual loss (for the image reconstruction pathway).
- **Semantic alignment loss**: CLIP-space alignment or contrastive learning loss.
- **Text generation loss**: Cross-entropy / autoregressive language modeling loss (for the captioning pathway).
- A multi-stage training strategy is likely adopted.

## Key Experimental Results

### Datasets
- **NSD (Natural Scenes Dataset)**: The most widely used high-quality fMRI visual decoding dataset; 7T fMRI with 8 subjects viewing natural images.
- **GOD (Generic Object Decoding)**: Another commonly used fMRI decoding dataset.

### Image Reconstruction Results

NeuroCreat is compared with prior reconstruction methods on both NSD and GOD. The paper states that "NeuroCreat not only achieves the optimal image..." — indicating state-of-the-art performance across multiple reconstruction metrics.

| Task | Dataset | Baselines | Conclusion |
|------|---------|-----------|------------|
| Image Reconstruction | NSD, GOD | Prior SOTA methods | Achieves optimal reconstruction performance |
| Captioning | NSD | Multiple methods | Captioning comparisons conducted against multiple methods |
| Creation | NSD | No prior work (first ever) | First creation benchmark established on NSD |

> Note: Complete quantitative results (PixCorr, SSIM, FID, CLIP-Score, etc.) should be consulted in the PDF tables.

### Ablation Study
- Validation of the PVA module's effectiveness (performance degradation across tasks upon PVA removal).
- Comparison of different prompt variant designs for each output modality.
- Ablation of the LLM integration strategy.

## Highlights & Insights
1. **Novel problem formulation**: The first work to explicitly extend fMRI brain decoding from "reconstructing the perceived" to "describing the perceived + creating the unseen," aligning with the cognitive hierarchy of perception → understanding → imagination.
2. **Elegant PVA module design**: Variant prompts disentangle cross-modal differences, gracefully addressing the challenge of serving multiple output modalities from a single encoder.
3. **Pioneering "mental creation" task**: The first creation benchmark on NSD, opening a new direction for creative applications in brain-computer interfaces.
4. **Effective LLM–brain decoding integration**: Demonstrates the potential of large language models in brain signal understanding beyond image reconstruction alone.
5. **Unified framework**: A single model jointly accomplishes reconstruction, captioning, and creation.

## Limitations & Future Work
1. **No public code**: Severely limits reproducibility and community follow-up.
2. **No arXiv preprint**: CVF Open Access only, restricting dissemination and citation.
3. **Inherent fMRI limitations**: Low temporal/spatial resolution, strong subject specificity, and costly data acquisition limit practical applicability.
4. **Definition and evaluation of "creation"**: Standardized metrics for defining and quantifying the quality of "mental creation" are lacking.
5. **Cross-subject generalization**: Whether the paper addresses cross-subject adaptation remains unclear (this was the specific focus of the group's preceding work).
6. **LLM specification**: The specific LLM used (scale/version) and its impact on results are not disclosed.
7. **Computational cost**: A brain decoding framework integrating an LLM likely demands substantial computational resources.

## Related Work & Insights

| Method | Reconstruction | Captioning | Creation | LLM | Characteristics |
|--------|:--------------:|:----------:|:--------:|:---:|-----------------|
| **NeuroCreat (Ours)** | ✅ | ✅ | ✅ | ✅ | First unified three-task brain decoding framework |
| MindEye2 | ✅ | ❌ | ❌ | ❌ | SOTA reconstruction with 1-hour data |
| Unibrain | ✅ | ✅ | ❌ | ❌ | Unified diffusion model for reconstruction + captioning |
| Brain-Streams | ✅ | ❌ | ❌ | ❌ | Multimodal-guided fMRI-to-Image |
| BrainSCUBA | ❌ | ✅ | ❌ | ✅ | Voxel-level semantic captioning |
| See Through Their Minds | ✅ | ✅ | ❌ | ❌ | Cross-subject transfer learning |

The key differentiators of NeuroCreat are: (1) "creation" as an entirely new task dimension; (2) the PVA module for unified multimodal output handling; (3) deep LLM integration rather than reliance on CLIP alone.

Key takeaways:
1. **Prompts as modality adapters**: The PVA design philosophy (using prompt variants to differentiate output modalities) is transferable to other multi-task, multimodal scenarios.
2. **LLM + signal processing**: Applying LLMs to non-textual signals (fMRI) offers insights into LLMs as general-purpose multimodal reasoning engines.
3. **Evaluation paradigm for "creation"**: Evaluating creative outputs is an open problem — semantic plausibility? Novelty? Consistency with individual experience?
4. **BCI prospects**: The ability to generate novel content from brain signals has far-reaching implications for assistive communication and creativity augmentation.
5. **Connection to visual foundation models**: Progress in fMRI decoding may reciprocally illuminate the internal representations of visual foundation models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The three-level extension of reconstruction → captioning → creation is highly visionary; the creation task is a genuine first.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comparisons across multiple methods on NSD and GOD, including captioning and the first-ever creation evaluation; lack of code limits verification.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clear and the framework design is logically coherent.
- Value: ⭐⭐⭐ — Insightful for the brain decoding field, though somewhat removed from the core research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] NEURONS: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction](neurons_emulating_the_human_visual_cortex_improves_fidelity_and_interpretability.md)
- [\[ACL 2026\] Language Reconstruction with Brain Predictive Coding from fMRI Data](../../ACL2026/medical_imaging/language_reconstruction_with_brain_predictive_coding_from_fmri_data.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](../../NeurIPS2025/medical_imaging/semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [\[ICCV 2025\] M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast](m-net_mri_brain_tumor_sequential_segmentation_network_via_mesh-cast.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](../../CVPR2026/medical_imaging/forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)

</div>

<!-- RELATED:END -->
