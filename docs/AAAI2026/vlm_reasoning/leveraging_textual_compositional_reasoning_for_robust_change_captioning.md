---
title: >-
  [Paper Note] Leveraging Textual Compositional Reasoning for Robust Change Captioning
description: >-
  [AAAI 2026][Multimodal VLM][Change Captioning] This paper proposes CORTEX, a framework that introduces VLM-generated compositional reasoning text as explicit cues, combined with an Image-Text Dual Alignment (ITDA) module…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Change Captioning"
  - "Compositional Reasoning"
  - "Vision-Language Models"
  - "Image-Text Alignment"
  - "Scene Change Description"
date: 2026-05-08
content_hash: 3a68e1d43cf06422
---

# Leveraging Textual Compositional Reasoning for Robust Change Captioning

**Conference**: AAAI 2026
**arXiv**: [2511.22903](https://arxiv.org/abs/2511.22903)  
**Code**: [https://github.com/VisualAIKHU/CORTEX](https://github.com/VisualAIKHU/CORTEX)  
**Area**: Multimodal VLM
**Keywords**: Change Captioning, Compositional Reasoning, Vision-Language Models, Image-Text Alignment, Scene Change Description

## TL;DR

This paper proposes CORTEX, a framework that introduces VLM-generated compositional reasoning text as explicit cues, combined with an Image-Text Dual Alignment (ITDA) module, to enhance purely visual change captioning methods in understanding structured semantics such as object relationships and spatial configurations.

## Background & Motivation

Change captioning aims to generate natural language descriptions explaining the differences between two images, with important applications in surveillance and medical imaging. The core limitations of existing methods are:

**Limitations of purely visual methods**: Methods such as SCORER, SMART, and DIRL rely solely on visual features to capture changes. While effective at detecting low-level appearance differences, they are incapable of **compositional reasoning**—i.e., understanding structured semantics such as object relationships and spatial configurations. Such information is not directly encoded in images but is implicitly embedded.

**Typical failure cases**: As shown in Figure 1, existing methods frequently misidentify spatial relations (e.g., "on the left") or incorrectly recognize reference objects (e.g., "a small brown cylinder"), due to the absence of explicit structured information representations.

**Complementary advantage of text**: Unlike visual information, text can explicitly describe structured semantics embedded in images in a clear and interpretable form, providing strong signals for high-level reasoning.

Based on these observations, the authors argue that augmenting existing purely visual methods with explicit textual compositional reasoning cues can better capture relational and contextual meaning in scene changes.

## Method

### Overall Architecture

CORTEX (COmpositional Reasoning-aware TEXt-guided) is a plug-and-play framework consisting of three modules:
- **Image-level Change Detector**: Directly reuses an existing method (e.g., DIRL/SCORER/SMART) to extract visual features from the "before" image $I_{bef}$ and "after" image $I_{aft}$ and encode low-level change cues $f_{icd}$
- **Reasoning-aware Text Extraction module (RTE)**: Leverages a VLM to extract structured sentences containing compositional reasoning information from each image
- **Image-Text Dual Alignment module (ITDA)**: Unifies visual and textual features via intra-scene (static) and cross-scene (dynamic) alignment

### Key Designs

#### 1. **Reasoning-aware Text Extraction Module (RTE)**

**Core Idea**: A frozen VLM (e.g., InternVL2-8B) is used to generate compositional reasoning sentences for each image, extracting semantic information that is difficult for traditional visual features to capture.

- **Carefully designed prompts**: Rather than generating generic descriptions, the VLM is guided to extract compositional reasoning cues containing detailed attributes (color, shape, size) and spatial relationships
- **Dynamic sentence count**: The number of extracted sentences is dynamically determined based on the object density and complexity of each scene; $N$ sentences $T_{bef}$ are generated for the "before" image and $M$ sentences $T_{aft}$ for the "after" image
- **Text encoding**: The generated sentences are embedded into sentence-level features $t_{bef}, t_{aft} \in \mathbb{R}^c$ via a BERT encoder, and concatenated to produce the RTE feature $f_{rte}$

**Design Motivation**: Although existing change detectors effectively capture appearance differences, they lack fine-grained contextual reasoning based on relative attributes and spatial context.

#### 2. **Image-Text Dual Alignment Module (ITDA)**

**Core Idea**: Textual features extracted by RTE and visual features reside in different latent spaces. ITDA unifies the complementary strengths of both modalities through intra-scene and cross-scene alignment strategies.

**Static Alignment**: Enhances intra-scene compositional understanding
- Aligns visual features from the same scene with the corresponding compositional reasoning sentence features via cross-attention
- For the "before" scene: $f_{bef}^{s(t \to i)} = \frac{1}{N}\sum_{n=1}^{N} \text{Attn}(t_{bef}^n, f_{bef}, f_{bef})$
- Self-attended visual features $f_{bef}^{s(i \to i)}$ are also computed for semantic consistency constraints
- Static alignment loss: $\mathcal{L}_{sa} = \frac{1}{2}(\|f_{bef}^{s(t \to i)} - f_{bef}^{s(i \to i)}\|_2^2 + \|f_{aft}^{s(t \to i)} - f_{aft}^{s(i \to i)}\|_2^2)$

**Dynamic Alignment**: Captures cross-scene changes
- Cross-attends the visual features of one scene with the text features of the other
- For the "before" scene: $f_{bef}^{d(t \to i)} = \frac{1}{M}\sum_{m=1}^{M} \text{Attn}(t_{aft}^m, f_{bef}, f_{bef})$
- Cross-scene visual attention features are similarly computed for constraints
- Dynamic alignment loss: $\mathcal{L}_{da} = \frac{1}{2}(\|f_{bef}^{d(t \to i)} - f_{bef}^{d(i \to i)}\|_2^2 + \|f_{aft}^{d(t \to i)} - f_{aft}^{d(i \to i)}\|_2^2)$

All static and dynamic enhanced features are finally concatenated into $f_{itda}$.

#### 3. **Transformer Decoder for Caption Generation**

The model integrates the outputs of all modules ($f_{icd}$, $f_{rte}$, $f_{itda}$) and generates change captions via a Transformer decoder.

### Loss & Training

Total loss function: $\mathcal{L}_{total} = \mathcal{L}_{cap} + \lambda \mathcal{L}_{align}$

- $\mathcal{L}_{cap}$: Standard captioning loss
- $\mathcal{L}_{align} = \mathcal{L}_{sa} + \mathcal{L}_{da}$: Alignment loss
- $\lambda$ is adjusted per baseline: $10^{-3}$ for SCORER, $10^{-4}$ for SMART/DIRL
- Trained on a single RTX 4090

## Key Experimental Results

### Main Results

Overall performance on the CLEVR-Change dataset:

| Method | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|--------|--------|--------|---------|-------|-------|
| SCORER (ICCV'23) | 56.3 | 41.2 | 74.5 | 126.8 | 33.3 |
| **CORTEX (SCORER)** | **57.0** | **42.7** | **75.9** | **128.8** | **33.9** |
| SMART (TPAMI'24) | 56.1 | 40.8 | 74.2 | 127.0 | 33.4 |
| **CORTEX (SMART)** | **56.5** | **42.1** | **75.7** | **130.2** | **34.0** |
| DIRL (ECCV'24) | 55.5 | 40.8 | 73.4 | 125.3 | 33.4 |
| **CORTEX (DIRL)** | **57.4** | **43.0** | **76.2** | **130.7** | **34.2** |

Performance on the Spot-the-Diff real-world dataset:

| Method | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|--------|--------|--------|---------|-------|-------|
| DIRL (ECCV'24) | 10.3 | 13.8 | 32.8 | 40.9 | 19.9 |
| **CORTEX (DIRL)** | **11.6** | **13.9** | **33.4** | **49.5** | **21.4** |

### Ablation Study

| Configuration | BLEU-4 | METEOR | CIDEr | Notes |
|---------------|--------|--------|-------|-------|
| Baseline (DIRL) | 55.5 | 40.8 | 125.3 | Visual-only baseline |
| + RTE | 55.8 | 41.6 | 128.5 | Text extraction module only |
| + RTE + ITDA | **57.4** | **43.0** | **130.7** | Full model |

Alignment loss ablation (CLEVR-Change):

| $\mathcal{L}_{sa}$ | $\mathcal{L}_{da}$ | BLEU-4 | CIDEr |
|---|---|--------|-------|
| ✗ | ✗ | 56.6 | 127.9 |
| ✓ | ✗ | 56.3 | 128.4 |
| ✗ | ✓ | 56.6 | 128.9 |
| ✓ | ✓ | **57.4** | **130.7** |

### Key Findings

1. **Plug-and-play effectiveness**: CORTEX adapts to three different baseline methods and consistently yields improvements across all
2. **Compositional reasoning prompts outperform generic descriptions**: Specially designed compositional reasoning prompts outperform generic description prompts (CIDEr 130.7 vs. 129.5)
3. **Complementarity of dual alignment**: Static and dynamic alignment each contribute independently, and joint use achieves the best results
4. **VLM robustness**: The method remains effective when replacing the VLM (LLaVA or InternVL2), demonstrating that it does not rely on a specific VLM
5. **Auxiliary context outperforms direct prediction**: Using the VLM to generate auxiliary text for individual images substantially outperforms directly prompting the VLM to compare image pairs

## Highlights & Insights

- **Plug-and-play design**: The RTE and ITDA modules can be seamlessly integrated into any existing image-level change detector without modifying the original model architecture
- **Text as a structured reasoning carrier**: The approach cleverly leverages VLMs to transform implicit compositional semantics in images into explicit textual representations, compensating for the limitations of visual features in structured reasoning
- **Dual alignment design rationale**: Static alignment enhances intra-scene understanding ("what is in this scene"), while dynamic alignment emphasizes cross-scene differences ("what differs between the two scenes")—a well-motivated design choice

## Limitations & Future Work

- The RTE module depends on a frozen VLM for inference, introducing additional preprocessing time (though the authors provide offline-extracted text data)
- Improvements on the real-world dataset (Spot-the-Diff) are more limited compared to the synthetic dataset (CLEVR-Change), possibly because real-world scene changes are more complex and diverse
- Prompt design significantly affects results, yet the paper does not provide sufficiently systematic exploration of prompt engineering

## Related Work & Insights

- **DIRL (ECCV'24)**, **SCORER (ICCV'23)**, and **SMART (TPAMI'24)** serve as baseline methods, representing the state of the art in purely visual change captioning
- **InternVL2** is used as the VLM for extracting textual cues, demonstrating the potential of large VLMs in assisting downstream visual tasks
- Broader implication: In other vision tasks requiring structured reasoning (e.g., VQA, scene graph generation), VLM-generated explicit textual reasoning cues could similarly serve as auxiliary inputs

## Rating

- Novelty: ⭐⭐⭐⭐ — Augmenting change captioning with textual compositional reasoning is a novel and well-motivated idea
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets, three baselines, detailed ablations, and comparisons across multiple VLMs
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-articulated motivation
- Value: ⭐⭐⭐⭐ — The plug-and-play design is highly practical and the code is publicly available

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DisCode: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning](discode_distribution-aware_score_decoder_for_robust_automatic_evaluation_of_imag.md)
- [\[ICML 2026\] Self-Captioning Multimodal Interaction Tuning: Amplifying Exploitable Redundancies for Robust Vision Language Models](../../ICML2026/multimodal_vlm/self-captioning_multimodal_interaction_tuning_amplifying_exploitable_redundancie.md)
- [\[AAAI 2026\] BiPrompt: Bilateral Prompt Optimization for Visual and Textual Debiasing in Vision-Language Models](biprompt_bilateral_prompt_optimization_for_visual_and_textual_debiasing_in_visio.md)
- [\[AAAI 2026\] Difference Vector Equalization for Robust Fine-tuning of Vision-Language Models](difference_vector_equalization_for_robust_fine-tuning_of_vis.md)
- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)

</div>

<!-- RELATED:END -->
