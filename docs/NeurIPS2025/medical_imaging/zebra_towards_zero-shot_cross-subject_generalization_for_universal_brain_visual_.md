---
title: >-
  [Paper Note] Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding
description: >-
  [NeurIPS 2025][Medical Imaging][fMRI decoding] This paper proposes Zebra, the first zero-shot brain visual decoding framework, which disentangles fMRI representations into subject-invariant and semantic-specific components via adversarial training and residual decomposition, enabling cross-subject visual reconstruction generalization without fine-tuning on new subjects.
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "fMRI decoding"
  - "zero-shot generalization"
  - "adversarial training"
  - "representation disentanglement"
  - "brain visual decoding"
date: 2026-05-08
content_hash: 113cd1c4afca27c9
---

# Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding

**Conference**: NeurIPS 2025
**arXiv**: [2510.27128](https://arxiv.org/abs/2510.27128)  
**Code**: [GitHub](https://github.com/xmed-lab/ZEBRA)  
**Area**: Other
**Keywords**: fMRI decoding, zero-shot generalization, adversarial training, representation disentanglement, brain visual decoding

## TL;DR

This paper proposes Zebra, the first zero-shot brain visual decoding framework, which disentangles fMRI representations into subject-invariant and semantic-specific components via adversarial training and residual decomposition, enabling cross-subject visual reconstruction generalization without fine-tuning on new subjects.

## Background & Motivation

fMRI-to-image reconstruction is a frontier direction in computational neuroscience and computer vision, aiming to reverse-engineer BOLD signals from the visual cortex into images. However, existing methods face a critical challenge: **inability to generalize across individuals**.

Current approaches (MindEye2, MindTuner, etc.) typically adopt a two-stage paradigm—pretraining a unified model on multi-subject data followed by subject-specific fine-tuning. This paradigm suffers from severe limitations: (1) every new patient requires AI expert intervention for fine-tuning; (2) the fine-tuning process is time-consuming (~one day), hindering real-time brain-computer interface applications; (3) no universal feature space capable of learning neural representations across human subjects exists.

**Core argument**: Despite inter-individual variability in brain activity, the human cortex encodes semantic information in a topographically organized, cross-subject consistent manner (supported by neuroscientific evidence). Therefore, zero-shot generalization can be achieved by explicitly separating subject-invariant components from semantic-specific ones.

Existing methods all fail under zero-shot settings: MindTuner's subject-specific design breaks down directly on unseen subjects; NeuroPictor, while transforming fMRI from different subjects into a unified shape, remains sensitive to subject noise and fails to learn invariant representations.

## Method

### Overall Architecture

Zebra builds upon a baseline framework (fMRI-PTE encoder + unCLIP diffusion prior + SDXL decoder), augmented with two core modules: Subject-Invariant Feature Extraction (SIFE) and Semantic-Specific Feature Extraction (SSFE). Training is performed once on training-set subjects, and inference on unseen subjects is performed directly without fine-tuning.

### Key Designs

1. **Subject-Invariant Feature Extraction (SIFE)**: Subject-invariant features are separated via residual decomposition and adversarial training. A self-attention module $\mathcal{F}_i$ extracts invariant features $\bm{E}_i = \mathcal{F}_i(\bm{E})$, with the residual yielding subject-specific features $\bm{E}_s = \bm{E} - \bm{E}_i$.

   Adversarial training ensures $\bm{E}_i$ contains no subject identity information—a subject discriminator $\mathcal{D}_{dis}$ attempts to identify the subject from $\bm{E}_i$, while the invariant extractor $\mathcal{F}_i$ is trained to prevent such identification:

$$\min_{\theta_{\mathcal{E}}, \theta_{\mathcal{F}}} \max_{\theta_{\mathcal{D}_{dis}}} \left\{ \mathcal{L}_{dis}^{\bm{E}} := -\mathbb{E}_{x,s} [s \log \mathcal{D}_{dis}(\mathcal{E}(\mathcal{F}_i(\bm{E})))] \right\}$$

   A classifier $\mathcal{D}_{cls}$ is simultaneously trained to retain subject identity in $\bm{E}_s$ (via $\mathcal{L}_{cls}^{\bm{E}}$), forming a complementary constraint.

2. **Representation Preservation Anchor**: Adversarial training may distort the original feature space. An auxiliary fMRI reconstruction task is introduced to preserve the informational integrity of the feature space:

$$\mathcal{L}_{rec} = \mathbb{E}_{(x, \hat{x})} [|\hat{x} - x|]$$

   A two-layer deconvolution network with a linear prediction head reconstructs the input signal, ensuring that $\bm{E}$ retains biological fidelity and semantic coherence under adversarial training.

3. **Semantic-Specific Feature Extraction (SSFE)**: Semantic information is further injected into $\bm{E}_i$. A vision projector maps brain features into the CLIP visual space, yielding semantic-specific features $\bm{F}_s = \mathcal{P}_s(\bm{E}_i)$ and semantic-invariant features $\bm{F}_i = \mathcal{P}_i(\bm{E}_s)$. A BiMixCo loss aligns $\bm{F}_s$ with OpenCLIP embeddings ($\mathcal{L}_{spe}^{\bm{F}}$), while a gradient reversal layer (GRL) prevents $\bm{F}_i$ from aligning with CLIP features ($\mathcal{L}_{inv}^{\bm{F}}$), forcing more semantic information to flow into $\bm{F}_s$.

### Loss & Training

The total loss integrates seven components:

$$\mathcal{L} = \mathcal{L}_{rec} + \mathcal{L}_{dis}^{\bm{E}} + \mathcal{L}_{cls}^{\bm{E}} + \mathcal{L}_{inv}^{\bm{F}} + \mathcal{L}_{spe}^{\bm{F}} + \mathcal{L}_{sem} + \lambda \mathcal{L}_{prior}$$

where $\mathcal{L}_{sem} = \mathcal{L}_{cls} + \mathcal{L}_{\text{CLIP}_v} + \mathcal{L}_{\text{CLIP}_t}$ and $\lambda=30$. Training runs for 60 epochs on 8 H800 GPUs with batch size 128, AdamW optimizer, and learning rate 1e-4. Inference uses a two-stage SDXL unCLIP decoding pipeline.

## Key Experimental Results

### Main Results (NSD dataset, average over subjects 1/2/5/7)

| Method | Training | PixCorr↑ | SSIM↑ | Alex(2)↑ | Alex(5)↑ | Incep↑ | CLIP↑ |
|--------|----------|---------|-------|----------|----------|--------|-------|
| NeuroPictor⋆ | Zero-shot | 0.057 | 0.297 | 71.4% | 74.7% | 62.5% | 66.0% |
| Our baseline | Zero-shot | 0.074 | 0.316 | 70.8% | 74.0% | 63.5% | 62.5% |
| **Zebra** | **Zero-shot** | **0.131** | **0.375** | **74.6%** | **81.2%** | **72.2%** | **71.5%** |
| MindEye2 | Few-shot (1h) | 0.195 | 0.419 | 84.2% | 90.6% | 81.2% | 79.2% |
| MindTuner | Full fine-tune | 0.322 | 0.421 | 95.8% | 98.8% | 95.6% | 93.8% |

Zebra substantially outperforms other zero-shot methods (PixCorr +0.074, Incep +9.7%), with some metrics approaching fully fine-tuned models.

### Ablation Study

| Baseline | SIFE Adv. | SIFE Anchor | SSFE Adv. | SSFE Anchor | PixCorr | Alex(5) | CLIP |
|----------|-----------|-------------|-----------|-------------|---------|---------|------|
| ✓ | | | | | 0.089 | 74.7% | 63.2% |
| ✓ | ✓ | | | | 0.129 | 77.4% | 66.8% |
| ✓ | ✓ | ✓ | | | 0.134 | 78.3% | 69.3% |
| ✓ | ✓ | ✓ | ✓ | | 0.142 | 79.6% | 70.8% |
| ✓ | ✓ | ✓ | ✓ | ✓ | **0.153** | **81.8%** | **72.3%** |

### Key Findings

- All metrics improve steadily as the number of training subjects increases from 4 to 7 (CLIP: 63.7% → 72.3%), indicating that more subject data benefits generalization.
- UMAP/t-SNE visualizations confirm that $\bm{E}_i$ is highly mixed across subjects (no subject-specific clustering), while $\bm{E}_s$ clearly clusters by subject identity.
- Zero-shot inference takes approximately 1 second per image, compared to over 12 hours required by conventional fine-tuning pipelines.
- Zebra shows larger advantages on low-level perceptual metrics; semantic accuracy remains weaker than few-shot methods.

## Highlights & Insights

- **Pioneering problem formulation**: This work is the first to define zero-shot brain visual decoding, advancing fMRI decoding from subject-specific fine-tuning towards a plug-and-play paradigm.
- **Neuroscience-driven design**: The architecture is grounded in neuroscientific evidence that the cortex encodes semantics consistently across individuals, achieving representation disentanglement via adversarial training and residual decomposition.
- **Elegant representation preservation anchor**: This design addresses the classical problem of adversarial training corrupting feature spaces, using fMRI reconstruction as an anchor to maintain informational integrity.
- From a practical standpoint, zero-shot decoding holds significant value for clinical applications such as brain-computer interfaces and neural rehabilitation.

## Limitations & Future Work

- Semantic fidelity remains inferior to few-shot methods, with degraded performance on rare object categories.
- Validation is limited to the NSD dataset with only 8 subjects, constraining the scale of evaluation.
- The work focuses solely on image reconstruction, leaving more complex modalities such as text and video unexplored.
- Additional subjects and fMRI recordings are needed to comprehensively capture real-world visual experiences.

## Related Work & Insights

Compared to methods requiring fine-tuning such as MindEye2 and MindTuner, Zebra requires no test-subject data whatsoever. Compared to NeuroPictor's unified brain encoding, Zebra effectively removes subject noise through explicit disentanglement. Key insight: in biomedical scenarios with large inter-individual variability, adversarial disentanglement may serve as a general strategy for achieving zero-shot generalization.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First zero-shot brain visual decoding work; pioneering in both problem formulation and methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive quantitative, qualitative, ablation, and visualization analyses, though dataset and subject scale are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, methodology is presented intuitively, and experiments are well organized.
- Value: ⭐⭐⭐⭐⭐ Significant practical implications for brain-computer interfaces and clinical neuroscience.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [\[ICLR 2026\] A Cognitive Process-Inspired Architecture for Subject-Agnostic Brain Visual Decoding](../../ICLR2026/medical_imaging/a_cognitive_process-inspired_architecture_for_subject-agnostic_brain_visual_deco.md)
- [\[NeurIPS 2025\] EEGReXferNet: A Lightweight Gen-AI Framework for EEG Subspace Reconstruction via Cross-Subject Transfer Learning and Channel-Aware Embedding](eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)
- [\[NeurIPS 2025\] FAPEX: Fractional Amplitude-Phase Expressor for Robust Cross-Subject Seizure Prediction](fapex_fractional_amplitude-phase_expressor_for_robust_cross-subject_seizure_pred.md)
- [\[ICCV 2025\] AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images](../../ICCV2025/medical_imaging/aczerots_active_learning_for_zeroshot_tissue_segmentation_in.md)

</div>

<!-- RELATED:END -->
