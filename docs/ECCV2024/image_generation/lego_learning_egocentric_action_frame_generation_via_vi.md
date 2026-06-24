---
title: >-
  [Paper Note] LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning
description: >-
  [ECCV 2024][Image Generation][Egocentric Perspective] This paper proposes LEGO, a model that enhances the action description capability of VLLMs through visual instruction tuning and injects the image/text embeddings of VLLMs as additional conditions into a diffusion model, enabling the generation of action execution frames from an egocentric perspective.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Egocentric Perspective"
  - "Action Frame Generation"
  - "Visual Instruction Tuning"
  - "Diffusion Model"
  - "VLLM"
date: 2026-05-08
content_hash: 762ba22682e9e65d
---

# LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning

**Conference**: ECCV 2024  
**arXiv**: [2312.03849](https://arxiv.org/abs/2312.03849)  
**Code**: [https://bolinlai.github.io/Lego_EgoActGen/](https://bolinlai.github.io/Lego_EgoActGen/)  
**Area**: Multimodal VLM  
**Keywords**: Egocentric Perspective, Action Frame Generation, Visual Instruction Tuning, Diffusion Model, VLLM

## TL;DR

This paper proposes LEGO, a model that enhances the action description capability of VLLMs through visual instruction tuning and injects the image/text embeddings of VLLMs as additional conditions into a diffusion model, enabling the generation of action execution frames from an egocentric perspective.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Core Problem**: How to synthesize images showing the action execution process (action frames) from an egocentric perspective to efficiently convey skills.

### Background

**Background**: **Current Dilemma**:

### Key Challenge

**Key Challenge**: Labels in existing egocentric action datasets (such as Ego4D and Epic-Kitchens) are merely "verb + noun", lacking detailed descriptions of action execution.

### Core Idea

**Core Idea**: Existing diffusion models are pre-trained primarily on third-person images, resulting in a domain gap with egocentric images.

### Supplementary Note

**Supplementary Note**: Pure text instructions are not intuitive enough; the human brain processes images much faster than text.

### Supplementary Note

**Supplementary Note**: **Motivation**: After wearing a camera, a user inputs a current scene image and an action query, and the model directly generates a target image showing the action execution, providing visual guidance.

## Method

### Overall Architecture

LEGO consists of two core phases:
1. **Prompt Enhancement Phase**: Train the VLLM using visual instruction tuning to generate rich action descriptions.
2. **Action Frame Generation Phase**: Inject VLLM embeddings into a Latent Diffusion Model (LDM) to generate egocentric action frames.

### Key Designs

**1. Data Organization and Visual Instruction Tuning**
- Utilizing GPT-3.5 for in-context learning based on action labels and object bounding boxes to generate detailed action descriptions.
- Perform visual instruction tuning on the VLLM (based on LLaVA): freeze the CLIP image encoder, and fine-tune the projection layer and LLM.
- The fine-tuned VLLM does not require bounding box inputs and can generate rich action descriptions at scale.

**2. Injecting VLLM Embeddings into LDM**
- The CLIP text encoder extracts the text representation $\psi(\mathcal{R})$ of the action description.
- The VLLM image embedding $\mathcal{H}_i$ is projected into the LDM feature space via a linear layer.
- The VLLM text embedding $\mathcal{H}_t$ is processed through a projection layer + self-attention layer.
- The three are concatenated to form the complete condition $\mathcal{C} = [\psi(\mathcal{R}), \sigma(\mathcal{H}_i), \pi(\mu(\mathcal{H}_t))]$.
- This is injected into multiple layers of the UNet via a cross-attention mechanism.

### Loss & Training

- VLLM phase: Cross-entropy loss, trained for 3 epochs.
- LDM phase: L2 regression loss (predicted noise vs. ground truth noise), trained for 20K iterations.
- Classifier-free guidance is adopted.
- Data preprocessing: Filter frames based on aesthetic scores, and filter out frame pairs that are too similar or too different based on similarity.

## Key Experimental Results

### Main Results

| Method | EgoVLP | EgoVLP+ | CLIP | FID↓ | PSNR | LPIPS↓ |
|------|--------|---------|------|------|------|--------|
| ProxEdit | 44.51 | 72.68 | 68.17 | 33.01 | 11.88 | 40.90 |
| SDEdit | 50.07 | 72.90 | 73.35 | 33.35 | 11.81 | 41.60 |
| IP2P | 62.19 | 78.84 | 78.75 | 24.73 | 12.16 | 37.16 |
| **LEGO** | **65.65** | **80.44** | **80.61** | **23.83** | **12.29** | **36.43** |

(Results on the Ego4D dataset)

### Ablation Study

| Condition Setup | User Study | EgoVLP | EgoVLP+ | CLIP |
|---------|-----------|--------|---------|------|
| Action Labels | 5.33 | 62.19 | 78.84 | 78.75 |
| Descriptions | 13.00 | 62.91 | 79.09 | 79.18 |
| Desc.+Img Embed. | 26.00 | 65.35 | 80.13 | 80.57 |
| Desc.+Txt Embed. | 21.33 | 63.29 | 79.40 | 79.21 |
| Desc.+Joint Embed. | **34.34** | **65.65** | **80.44** | **80.61** |

### Key Findings

- VLLM image embeddings yield a larger performance improvement than text embeddings, as they contain high-level semantic information not captured by autoencoders or text.
- Fine-tuned VLLM embeddings outperform the non-fine-tuned version (EgoVLP improves by 1.08%), demonstrating that visual instruction tuning is crucial for bridging the domain gap.
- In the user study, LEGO's win rate outperforms the strongest baseline by 44% (Ego4D) and 38.34% (Epic-Kitchens).

## Highlights & Insights

1. **Pioneers the problem of egocentric action frame generation**, upgrading skill transfer from textual guidance to visual guidance.
2. **An innovative coupled VLLM-LDM architecture**: It not only uses VLLM to generate better textual descriptions but also utilizes its internal embeddings as diffusion conditions.
3. Visual instruction tuning improves prompt alignment from $27\% \to 87\%$, significantly reducing hallucinations.
4. The model exhibits strong generalization capability: given the same input frame and different action queries, it can generate distinct and reasonable action frames.

## Limitations & Future Work

- The generation resolution is only $256 \times 256$, which limits practical applications.
- It depends on the action annotation structures of existing datasets, making it difficult to generalize to open domains.
- It does not consider temporal consistency in videos, generating only single frames.
- BLIP-based text evaluation metrics are still affected by the domain gap.

## Related Work & Insights

- InstructPix2Pix inspired the instruction-based image editing paradigm, but lacks egocentric domain adaptation.
- GILL explored learning image embeddings from VLLMs for generation, but did not perform visual instruction tuning.
- Potential inspiration: Extend the coupled VLLM + diffusion model paradigm to scenarios such as robotic manipulation instruction visualization and AR-assisted teaching.

## Rating

- Novelty: ⭐⭐⭐⭐ (New problem definition + innovative VLLM embedding injection design)
- Technical Depth: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Dual datasets, user study, and comprehensive ablation studies)
- Writing Quality: ⭐⭐⭐⭐
- Overall Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition](idempotent_unsupervised_representation_learning_for_skeleton-based_action_recogn.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] Lego: Learning to Disentangle and Invert Personalized Concepts Beyond Object Appearance in Text-to-Image Diffusion Models](lego_learning_to_disentangle_and_invert_personalized_concepts_beyond_object_appe.md)
- [\[ECCV 2024\] Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality](learning_trimodal_relation_for_audio-visual_question_answering_with_missing_moda.md)
- [\[ECCV 2024\] Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion](mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)

</div>

<!-- RELATED:END -->
