---
title: >-
  [Paper Note] UMBRAE: Unified Multimodal Brain Decoding
description: >-
  [ECCV 2024][Medical Imaging][Brain signal decoding] This paper proposes UMBRAE, which aligns fMRI signals with image features using a universal brain encoder and feeds them into a frozen MLLM to achieve multimodal brain decoding (description, grounding, retrieval, visual reconstruction). It innovatively introduces a cross-subject training strategy, enabling a single model to serve multiple subjects and outperform single-subject models.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Brain signal decoding"
  - "fMRI"
  - "Cross-subject training"
  - "Multimodal LLM"
  - "Brain-visual alignment"
date: 2026-05-08
content_hash: 939e31f7d7bf3001
---

# UMBRAE: Unified Multimodal Brain Decoding

**Conference**: ECCV 2024  
**arXiv**: [2404.07202](https://arxiv.org/abs/2404.07202)  
**Code**: [https://weihaox.github.io/UMBRAE](https://weihaox.github.io/UMBRAE)  
**Area**: Multimodal VLM  
**Keywords**: Brain signal decoding, fMRI, Cross-subject training, Multimodal LLM, Brain-visual alignment

## TL;DR

This paper proposes UMBRAE, which aligns fMRI signals with image features using a universal brain encoder and feeds them into a frozen MLLM to achieve multimodal brain decoding (description, grounding, retrieval, visual reconstruction). It innovatively introduces a cross-subject training strategy, enabling a single model to serve multiple subjects and outperform single-subject models.

## Background & Motivation

**Background**: Brain signal decoding research has made progress in recent years, decoding fMRI signals into images (such as MindEye), videos, or text, but remains limited to single-modality outputs.

**Limitations of Prior Work**: (a) Single-modality decoding yields lossy representations—text lacks spatial location information, while image reconstruction is an underdetermined problem and lacks explicit representation of scene structures; (b) each subject requires a separately trained model due to structural and functional differences in activation patterns across different brains.

**Key Challenge**: Brain signals contain rich multimodal information (semantic concepts + spatial locations + object relations), but existing methods can only decode into a single modality. Subject-specific training cannot leverage the synergy of multi-subject data.

**Goal**: (a) Achieve unified decoding from brain signals to multimodal representations; (b) train a cross-subject universal model.

**Key Insight**: Align brain signals with intermediate features of a pre-trained image encoder, and then leverage the multitasking capability of MLLMs to achieve decoding at different levels of granularity.

**Core Idea**: Once brain signals are aligned to the image feature space, the multimodal understanding capabilities of the MLLM can be directly reused.

## Method

### Overall Architecture

UMBRAE consists of three components: (1) A brain encoder (subject-specific tokenizer + universal perceive encoder) that maps fMRI signals to fixed-length brain tokens; (2) a multimodal alignment module that aligns brain tokens with CLIP image features; (3) a frozen MLLM (e.g., Shikra/LLaVA) that executes different tasks through a prompt interface.

### Key Designs

**1. Brain Encoder Architecture**
- **Function**: Encodes variable-length fMRI signals from different subjects into a unified, fixed-length token sequence.
- **Mechanism**: A lightweight tokenizer per subject + a shared perceive encoder (Transformer cross-attention); each subject has learnable subject tokens (5x1024).
- **Design Motivation**: Structural brain differences among subjects are processed by dedicated tokenizers, while universal cognitive patterns are captured by the shared encoder.

**2. Cross-Subject Training Strategy**
- **Function**: Jointly trains data from multiple subjects within a single model.
- **Mechanism**: Within each batch, 50% of the data comes from the same subject, while the remaining is uniformly sampled from other subjects.
- **Design Motivation**: Key discovery—the cross-subject model actually outperforms single-subject models, indicating the presence of transferable brain activity patterns.

**3. Alignment with Intermediate Image Features**
- **Function**: Aligns brain features with the second-to-last layer features of CLIP ViT-L/14 (16x16x1024).
- **Mechanism**: Simple MSE reconstruction loss $\mathcal{L}_{\text{rec}} = \mathbb{E}[\|V(v) - B(b)\|^2]$.
- **Design Motivation**: Intermediate features preserve both semantic and spatial information, which can be directly fed into the MLLM adapter.

**4. Brain Prompting Interface**
- **Function**: Uses different prompt templates for different tasks—"Describe this image" for captioning, "Locate <expr>" for grounding.
- **Mechanism**: Brain features replace image features in the prompt embeddings.
- **Design Motivation**: The instruction-following capability of MLLMs naturally supports multi-task switching.

**5. Weakly Supervised Adaptation to New Subjects**
- **Function**: Rapidly adapts to new subject data with a small amount of training samples.
- **Mechanism**: Freezes the perceive encoder and only trains the new subject's tokenizer.
- **Design Motivation**: Cross-subject training has already learned universal patterns; new subjects only need to learn to "translate" their format.

### Loss & Training

- Training Loss: MSE reconstruction loss aligning brain features with image features.
- Optimizer: AdamW, $\beta_1=0.9$, $\beta_2=0.95$, weight decay = 0.01.
- Learning Rate: One-cycle scheduler, initial learning rate of 3e-4.
- Training Scale: Single A100 GPU, 240 epochs, batch size 256, approximately 12 hours.
- Data: NSD dataset, 24,980 training samples and 982 testing samples for each of the 4 subjects.

## Key Experimental Results

### Main Results

| Method | BLEU1 | METEOR | CIDEr | SPICE | CLIP-S |
|------|-------|--------|-------|-------|--------|
| SDRecon | 36.21 | 10.03 | 13.83 | 5.02 | 61.07 |
| OneLLM | 47.04 | 13.55 | 22.99 | 6.26 | 54.80 |
| BrainCap | 55.96 | 16.68 | 41.30 | 9.06 | 64.31 |
| UMBRAE-S1 | 57.63 | Best | Best | Best | 65.00+ |
| **UMBRAE** | **Best** | **Best** | **Best** | **Best** | **Best** |

### Ablation Study

| Setting | Performance |
|-----|------|
| Single-subject training (UMBRAE-S1) | Baseline |
| Cross-subject training (UMBRAE) | Outperforms single-subject, without increasing training time |
| Weakly supervised adaptation (10% data) | Still achieves reasonable performance |
| 7B vs 13B LLM | 13B yields further improvements |

### Key Findings

1. **Cross-subject outperforms single-subject**: Joint training leverages shared neural patterns across subjects, enhancing generalization.
2. UMBRAE is the first to achieve direct brain-signal grounding (brain grounding), approaching the baseline of utilizing ground-truth images while being over 10 times faster.
3. Simple MSE alignment to intermediate image features successfully recovers semantic and spatial information without requiring contrastive learning or a diffusion prior.
4. Weakly supervised adaptation can scale to new subjects with only a minimal amount of data.

## Highlights & Insights

- **Counter-intuitive conclusion on cross-subject training**: While neuroscience posits significant individual brain differences, experiments prove that shared training performs better, hinting at a deeper universal pattern in speech and human cognition.
- **Extreme simplicity of the alignment target**: Aligning brain features with intermediate image features purely via MSE is sufficient to unlock the full potential of the MLLM.
- **BrainHub benchmark**: The first comprehensive brain understanding evaluation benchmark, extending NSD to support captioning, grounding, and retrieval multi-tasks.
- **Model-agnostic design**: The proposed method can be combined with any image encoder, LLM, and MLLM.

## Limitations & Future Work

1. fMRI equipment is expensive and non-portable, limiting practical applications.
2. The NSD dataset scale is limited (~25K per subject); whether larger-scale data can yield further improvements remains to be explored.
3. Current grounding precision is limited by the spatial resolution of fMRI.
4. The study only utilizes 4 subjects, and scalability to larger cohorts remains to be validated.
5. The quality of visual reconstruction depends on the downstream generative model.

## Related Work & Insights

- **MindEye/BrainDiffuser**: Single-modality visual reconstruction; UMBRAE extends this to multimodal decoding.
- **OneLLM**: A unified multimodal encoder that includes brain signals, but requires massive data and computational resources.
- **Shikra/LLaVA**: Serve as the MLLM base; UMBRAE demonstrates that brain signals can "disguise" themselves as image features to be understood by the MLLM.
- **Insight**: Brain signals are essentially an "encoding" of natural images; a good decoder only needs to learn the mapping to an existing representation space.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (First unified multimodal brain decoding + cross-subject training)
- **Technical Depth**: ⭐⭐⭐⭐ (Reasonable architectural design with theoretically motivated cross-subject sampling strategies)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Comprehensive evaluation across multiple tasks + BrainHub benchmark + thorough ablation studies)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear motivation and intuitive methodology representation)
- **Value**: ⭐⭐⭐⭐ (Significant value at the intersection of brain-computer interfaces and multimodal learning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] NeuroFlow: Toward Unified Visual Encoding and Decoding from Neural Activity](../../CVPR2026/medical_imaging/neuroflow_toward_unified_visual_encoding_and_decoding_from_neural_activity.md)
- [\[ECCV 2024\] Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals](brain_netflix_scaling_data_to_reconstruct_videos_from_brain_signals.md)
- [\[ICLR 2026\] Reducing Semantic Mismatch in Brain-to-Text Decoding Through Personalized Multimodal Masking](../../ICLR2026/medical_imaging/reducing_semantic_mismatch_in_brain-to-text_decoding_through_personalized_multim.md)
- [\[ICLR 2026\] Unified Brain Surface and Volume Registration](../../ICLR2026/medical_imaging/unified_brain_surface_and_volume_registration.md)
- [\[ECCV 2024\] Brain-ID: Learning Contrast-agnostic Anatomical Representations for Brain Imaging](brain-id_learning_contrast-agnostic_anatomical_representations_for_brain_imaging.md)

</div>

<!-- RELATED:END -->
