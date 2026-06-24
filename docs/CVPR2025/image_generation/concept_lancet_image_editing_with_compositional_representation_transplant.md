---
title: >-
  [Paper Note] Concept Lancet: Image Editing with Compositional Representation Transplant
description: >-
  [CVPR 2025][Image Generation][Image Editing] Proposes Concept Lancet (CoLan), a zero-shot plug-and-play image editing framework. By sparsely decomposing the latent representation of the source image into a linear combination of visual concept vectors and then performing customized concept transplantation according to the editing task (replacement/addition/removal), CoLan resolves the challenge of editing intensity calibration.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Editing"
  - "Concept Transplant"
  - "Sparse Decomposition"
  - "Zero-shot Plug-and-Play"
  - "Editing Intensity Calibration"
date: 2026-05-08
content_hash: e88eddd95dfda9f7
---

# Concept Lancet: Image Editing with Compositional Representation Transplant

**Conference**: CVPR 2025  
**arXiv**: [2504.02828](https://arxiv.org/abs/2504.02828)  
**Code**: [https://peterljq.github.io/project/colan](https://peterljq.github.io/project/colan) (Project Page + CoLan-150K Dataset)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Image Editing, Concept Transplant, Sparse Decomposition, Zero-shot Plug-and-Play, Editing Intensity Calibration

## TL;DR
Proposes Concept Lancet (CoLan), a zero-shot plug-and-play image editing framework. By sparsely decomposing the latent representation of the source image into a linear combination of visual concept vectors and then performing customized concept transplantation according to the editing task (replacement/addition/removal), CoLan resolves the challenge of editing intensity calibration.

## Background & Motivation

### Background

**Background**: Diffusion-model-based image editing (e.g., P2P-Zero, InfEdit) typically achieves concept modification through vector addition and subtraction in the text embedding or score space. For example, to replace "cat $\rightarrow$ dog", the "cat" embedding is subtracted and the "dog" embedding is added.

**Limitations of Prior Work**: Simple vector addition/subtraction (VecAdd) faces severe editing intensity calibration problems: (1) subtracting too much disrupts the structure of the source image; (2) subtracting too little leaves residues of the source concept; (3) different concepts require different intensities but cannot be automatically determined. This makes it difficult to balance editing effectiveness and consistency.

**Key Challenge**: Vector addition/subtraction assumes that concepts are independent in the embedding space, whereas embeddings of different concepts are highly entangled in reality. It is necessary to first accurately "locate" the components belonging to the target concept in the source representation, and then perform precise replacement.

**Goal**: To achieve precise concept replacement/addition/deletion while preserving editing consistency (without disrupting non-edited regions).

**Key Insight**: Decomposing the source image representation into a sparse linear combination of vectors from a concept dictionary (via Elastic Net optimization), and then manipulating only the coefficients of the target concepts. A VLM (GPT-4V) is utilized to automatically select a relevant subset of concepts to reduce computational overhead.

**Core Idea**: Sparsely decomposing the source representation into a linear combination of a concept dictionary via Elastic Net, and then precisely replacing, adding, or deleting the coefficients of target concepts to achieve controllable editing.

## Method

### Overall Architecture
The CoLan-150K concept dataset (5,078 concepts, 152,971 stimulus images) is constructed. During inference: (1) A VLM parses relevant concepts from the source/target prompts and selects a dictionary subset; (2) Elastic Net is used to decompose the source text embedding (or score) into a sparse combination of concept vectors; (3) The concept coefficients are manipulated based on the editing type—replacement (swapping coefficients), addition (increasing coefficients), or deletion (zeroing out coefficients). This is a plug-and-play approach compatible with various inversion methods and backbones.

### Key Designs

1. **Sparse Concept Decomposition**

    - **Function**: Accurately locating components in the source representation that belong to each concept.
    - **Mechanism**: Given a source text embedding $e_s$ and a concept dictionary $V = \{v_1, ..., v_n\}$, the objective is to solve $e_s \approx \sum_i \alpha_i v_i$, where $\alpha$ is optimized via Elastic Net (L1 + L2 regularization ensuring sparsity and stability). A similar operation is performed in the score space. After decomposition, the coefficient $\alpha_i$ of each concept accurately quantifies its contribution to the source representation.
    - **Design Motivation**: VecAdd assumes a subtraction of the entire "cat" embedding, which also contains components shared with other concepts. Sparse decomposition subtracts only the portion that genuinely belongs to "cat".

2. **Customized Concept Transplantation**

    - **Function**: Precisely manipulating concept coefficients based on the type of editing task.
    - **Mechanism**: **Replacement**: Locate the coefficient of the source concept $\alpha_{src}$ and the target concept $\alpha_{tgt}$ (obtained by decomposing the target prompt), and swap the coefficients. **Addition**: Add the target concept's coefficient to the current decomposition. **Deletion**: Set the target concept's coefficient to zero. Reconstruct the new embedding/score after manipulation for denoising.
    - **Design Motivation**: Different editing primitives require different operational logics; thus, three basic edits are supported within a unified framework.

3. **CoLan-150K Concept Dictionary**

    - **Function**: Providing rich visual concept coverage to support the accuracy of sparse decomposition.
    - **Mechanism**: 5,078 visual concepts (colors, textures, objects, styles, etc.), with approximately 30 stimulus images per concept. Concept vectors are obtained through text embeddings or score statistics from the diffusion process. A VLM automatically selects a subset of concepts related to the edit (~100 concepts) to avoid full-dictionary optimization.
    - **Design Motivation**: The accuracy of linear decomposition depends on the coverage and diversity of the dictionary.

### Loss & Training
Training-free—Elastic Net optimization is executed at inference time. VLM API calls add some inference overhead.

## Key Experimental Results

### Main Results

| Method | StruDist↓ (×10⁻³) | PSNR↑ | Description |
|------|-------------------|-------|------|
| VecAdd + P2P-Zero | 53.04 / 25.54 | 17.65 / 21.59 | Severe structural disruption |
| **CoLan + P2P-Zero** | **15.91 / 6.61** | **23.08 / 26.08** | Significant consistency improvement |
| VecAdd + InfEdit | Higher | Lower | — |
| **CoLan + InfEdit** | **13.97 / 6.20** | **23.42 / 28.46** | Best among all methods |

### Key Findings
- CoLan consistently improves editing consistency (StruDist reduced by 3-4×) and editing precision across all backbone networks and inversion methods.
- Its plug-and-play nature allows it to directly enhance existing editing methods without retraining.
- Operating in the score space is generally superior to operating in the text embedding space.

## Highlights & Insights
- The paradigm of **using sparse decomposition instead of vector addition/subtraction** solves the fundamental issue of editing intensity calibration—concept contributions are precisely quantified instead of heuristically tuning the intensity.
- The **plug-and-play design** enables the method to be widely combined with other techniques, increasing its practical value.

## Limitations & Future Work
- Requires a VLM (GPT-4V) to parse concepts, which increases inference cost.
- Elastic Net optimization increases the latency of each edit.
- Assumes that concepts are linearly composable in the latent space; thus, complex non-linear relationships may not be handled effectively.
- The coverage of the concept dictionary determines the quality of the decomposition.

## Rating
- Novelty: ⭐⭐⭐⭐ Elegant design of the sparse decomposition and concept transplantation framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple backbones, multiple inversion methods, and three editing types.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problems.
- Value: ⭐⭐⭐⭐ Resolves practical pain points in image editing, with plug-and-play capability enhancing its utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] InsightEdit: Towards Better Instruction Following for Image Editing](insightedit_towards_better_instruction_following_for_image_editing.md)
- [\[CVPR 2025\] MagicQuill: An Intelligent Interactive Image Editing System](magicquill_an_intelligent_interactive_image_editing_system.md)
- [\[CVPR 2026\] Intrinsic Concept Extraction Based on Compositional Interpretability](../../CVPR2026/image_generation/intrinsic_concept_extraction_based_on_compositional_interpretability.md)
- [\[CVPR 2025\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)
- [\[CVPR 2025\] Multi-Group Proportional Representation for Text-to-Image Models](multi-group_proportional_representations_for_text-to-image_models.md)

</div>

<!-- RELATED:END -->
