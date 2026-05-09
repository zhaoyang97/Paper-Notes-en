---
title: >-
  [Paper Note] LaVCa: LLM-assisted Visual Cortex Captioning
description: >-
  [ICLR 2026][3D Vision][visual cortex] This paper proposes LaVCa, a method that leverages LLMs to generate natural language captions for individual voxels in the human visual cortex. Through a four-step pipeline—encoding model construction, optimal image selection, MLLM-based captioning, and LLM-driven keyword extraction with sentence composition—LaVCa reveals voxel-level visual selectivity more accurately and with greater semantic diversity than the prior method BrainSCUBA.
tags:
  - ICLR 2026
  - 3D Vision
  - visual cortex
  - voxel selectivity
  - LLM
  - fMRI encoding model
  - brain activity prediction
date: 2026-05-08
content_hash: e2bd04b3aade15fc
---

# LaVCa: LLM-assisted Visual Cortex Captioning

**Conference**: ICLR 2026
**arXiv**: [2502.13606](https://arxiv.org/abs/2502.13606)
**Code**: [https://github.com/suyamat/LaVCa](https://github.com/suyamat/LaVCa)
**Area**: 3D Vision / Neuroscience
**Keywords**: visual cortex, voxel selectivity, LLM, fMRI encoding model, brain activity prediction

## TL;DR

This paper proposes LaVCa, a method that leverages LLMs to generate natural language captions for individual voxels in the human visual cortex. Through a four-step pipeline—encoding model construction, optimal image selection, MLLM-based captioning, and LLM-driven keyword extraction with sentence composition—LaVCa reveals voxel-level visual selectivity more accurately and with greater semantic diversity than the prior method BrainSCUBA.

## Background & Motivation

**Background**: fMRI encoding models are the standard tool for studying visual representations in the brain. Early approaches relied on hand-crafted features or one-hot semantic labels, offering interpretability at the cost of granularity; modern methods leverage DNN features (e.g., CLIP) to substantially improve prediction accuracy, but DNNs are black boxes that offer little insight into why individual voxels are activated.

**Limitations of Prior Work**: Existing data-driven captioning methods such as BrainSCUBA directly apply image captioning models (ClipCap) to generate voxel captions, resulting in limited vocabulary and semantic diversity. SASC concatenates short n-gram phrases but suffers from insufficient expressiveness. Both approaches lack the semantic richness needed to precisely characterize voxel selectivity.

**Key Challenge**: The fundamental tension lies in maintaining interpretability (concise captions) without losing the rich information present in the optimal image set.

**Goal**: To generate precise, concise, and semantically rich natural language descriptions for each voxel—descriptions that can accurately predict brain activity while revealing both inter-voxel and intra-voxel diversity.

**Key Insight**: The pipeline is decoupled into four interpretable steps, separating image selection from captioning and exploiting the open-vocabulary capacity of LLMs for keyword extraction and sentence composition.

**Core Idea**: An LLM first extracts common keywords from the optimal image set of a voxel and then composes them into a caption, achieving high accuracy and high semantic diversity in voxel-level visual cortex description.

## Method

### Overall Architecture

LaVCa four-step pipeline:
- **Input**: fMRI brain activity data recorded while subjects viewed images from the NSD dataset
- **Step 1**: Construct an encoding model for each voxel using CLIP-Vision embeddings and ridge regression: $\mathbf{y}_i = \mathbf{W}\mathbf{x}_i + \bm{\varepsilon}_i$
- **Step 2**: Compute predicted responses of the encoding model over 1.7 million external OpenImages, selecting the top-N optimal images
- **Step 3**: Generate a description for each optimal image using an MLLM (MiniCPM-V)
- **Step 4**: Extract keywords from the descriptions using an LLM (GPT-4o) → filter via CLIP-Text cosine similarity → compose the final caption using MeaCap Sentence Composer
- **Output**: One natural language caption per voxel

### Key Designs

1. **Encoding Model Construction (Step 1)**:

    - Function: Build a linear predictive model mapping images to brain activity for each voxel
    - Mechanism: Extract L2-normalized CLIP-Vision projection embeddings and fit encoding weights $\mathbf{W} \in \mathbb{R}^{v \times d}$ via ridge regression
    - Design Motivation: Linear models are interpretable; CLIP features are aligned in the visual-language space, facilitating subsequent text-based evaluation

2. **Optimal Image Set Retrieval (Step 2)**:

    - Function: Identify the set of images that most strongly activates a given voxel
    - Mechanism: Compute the inner product between encoding weights and CLIP embeddings of 1.7 million external images; select top-N
    - Design Motivation: Using external large-scale data (outside the training set) mitigates overfitting; $N$ is tunable; OpenImages-v6 provides broad coverage

3. **LLM Keyword Extraction and Sentence Composition (Step 4)**:

    - Function: Distill common keywords from captions of multiple optimal images and synthesize a voxel caption
    - Mechanism: GPT-4o in-context learning extracts keywords → CLIP-Text cosine similarity between each keyword and the encoding weight is computed and filtered via a softmax threshold → MeaCap Sentence Composer combines keywords into a sentence, substituting encoding weights for the original image features
    - Design Motivation: More concise and interpretable than direct caption concatenation; covers a broader vocabulary than the end-to-end BrainSCUBA approach; keyword filtering ensures relevance

### Evaluation Protocol

- **Sentence-level prediction**: Voxel activity is predicted as the cosine similarity between the caption (encoded via Sentence-BERT) and captions of NSD images; accuracy is measured by Spearman correlation
- **Image-level prediction**: Images are generated from captions using FLUX.1-schnell → CLIP-Vision embeddings are compared against NSD images, eliminating the confounding effect of language

## Key Experimental Results

### Main Results

Sentence-level brain activity prediction accuracy (top-5000 voxels, mean ± std across 4 subjects):

| Method | #Keywords | Sentence Composer | subj01 | subj02 | subj05 | subj07 |
|--------|-----------|-------------------|--------|--------|--------|--------|
| Shuffled | - | - | 0.007±0.199 | 0.058±0.223 | 0.068±0.243 | 0.009±0.175 |
| BrainSCUBA | - | - | 0.207±0.062 | 0.251±0.071 | 0.264±0.084 | 0.182±0.065 |
| LaVCa | 1 | ✗ | 0.205±0.068 | 0.250±0.075 | 0.272±0.086 | 0.186±0.072 |
| **LaVCa** | **5** | **✓** | **0.246±0.066** | **0.287±0.075** | **0.306±0.084** | **0.218±0.073** |

Image-level prediction accuracy similarly shows that LaVCa (5 keywords + SC) outperforms BrainSCUBA across all subjects (e.g., subj01: 0.213 vs. 0.188).

### Ablation Study

| Configuration | Vocabulary Size (inter-voxel) | Semantic Variance | PCA 90% Dimensions |
|---------------|-------------------------------|-------------------|--------------------|
| BrainSCUBA | 3,193 | 0.0588 | 127 |
| Top-1 MLLM caption | 13,959 | 0.0638 | 210 |
| **LaVCa** | **16,922** | **0.0642** | **219** |

Intra-ROI shuffle test (validating inter-voxel diversity):

| ROI | Original | Shuffled | Ratio |
|-----|----------|----------|-------|
| OFA (face region) | 0.095 | 0.028 | 3.3× |
| PPA (scene region) | 0.213 | 0.151 | 1.4× |
| EBA (body region) | 0.157 | 0.018 | 8.7× |

### Key Findings

- The combination of 5 keywords and Sentence Composer significantly outperforms BrainSCUBA and single-keyword variants across all subjects
- LaVCa's vocabulary is 5.3× larger than BrainSCUBA's (16,922 vs. 3,193), with higher semantic diversity
- Even within ROIs traditionally considered selective for a single category (e.g., faces in OFA, scenes in PPA), LaVCa reveals rich multi-concept encoding—individual voxels can simultaneously encode multiple distinct concepts
- Cross-subject analysis confirms that intra-ROI diversity is reproducible

## Highlights & Insights

- **Elegant decoupled design**: Decomposing voxel captioning into four steps—image selection, captioning, keyword extraction, and sentence composition—makes each component independently replaceable (any VLM/LLM), yielding greater flexibility and interpretability than end-to-end approaches. This modular paradigm is transferable to other tasks requiring interpretable feature descriptions from data
- **Encoding weights as image feature surrogates**: Using encoding weights in place of image features to guide MeaCap Sentence Composer cleverly connects neuroscience signals directly into an NLP pipeline
- **Revealing diversity within canonical ROIs**: Prior work assumed OFA encodes only "faces," yet LaVCa finds voxels encoding "tongue," "smile," "animals," and other concepts—a conceptual advance enabled by methodological improvement

## Limitations & Future Work

- **Dependence on CLIP feature space**: Both the encoding model and keyword filtering rely on CLIP; concepts poorly represented in CLIP (e.g., subtle textures, abstract notions) may limit LaVCa's expressiveness
- **Linear encoding assumption**: Ridge regression assumes a linear relationship between voxel responses and CLIP features, which may be an oversimplification for higher-level visual areas
- **LLM hallucination risk**: GPT-4o may introduce hallucinated keywords during extraction; CLIP-based filtering mitigates but does not eliminate this risk
- **Restricted to visual cortex**: The method has not yet been extended to other brain regions (e.g., auditory or language areas); similar approaches could be explored for studying language encoding

## Related Work & Insights

- **vs. BrainSCUBA**: BrainSCUBA applies ClipCap end-to-end to generate voxel captions, with vocabulary constrained by ClipCap's training data. LaVCa's decoupled design exploits the open vocabulary of LLMs, yielding a 5× vocabulary increase and superior accuracy
- **vs. SASC**: SASC concatenates short n-gram phrases, retaining minimal information. LaVCa preserves richer semantic content through multi-image keyword extraction
- **vs. brain decoding work**: Brain decoding asks "what did the subject see?", while encoding models ask "what does this voxel represent?"—these are complementary research directions

## Rating

- Novelty: ⭐⭐⭐⭐ The method is an ingenious combination of existing components, but the decoupled design and integration of LLMs into neuroscience represent a genuinely new direction
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Sentence-level and image-level evaluation, diversity analysis, intra-ROI shuffle tests, and cross-subject validation are all conducted—highly comprehensive
- Writing Quality: ⭐⭐⭐⭐⭐ Logically clear with well-designed figures and precisely described methodology
- Value: ⭐⭐⭐⭐ Offers an important contribution to understanding visual cortex representations, though the scope is relatively narrow (neuroscience-focused)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Meta-Learning an In-Context Transformer Model of Human Higher Visual Cortex](../../NeurIPS2025/3d_vision/meta-learning_an_in-context_transformer_model_of_human_higher_visual_cortex.md)
- [\[CVPR 2026\] 3DrawAgent: Teaching LLM to Draw in 3D with Early Contrastive Experience](../../CVPR2026/3d_vision/3drawagent_teaching_llm_to_draw_in_3d_with_early_contrastive_experience.md)
- [\[ICLR 2026\] Quantized Visual Geometry Grounded Transformer](quantized_visual_geometry_grounded_transformer.md)
- [\[CVPR 2026\] ArtLLM: Generating Articulated Assets via 3D LLM](../../CVPR2026/3d_vision/artllm_generating_articulated_assets_via_3d_llm.md)
- [\[AAAI 2026\] NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling](../../AAAI2026/3d_vision/nurbgen_high-fidelity_text-to-cad_generation_through_llm-driven_nurbs_modeling.md)

<!-- RELATED:END -->
