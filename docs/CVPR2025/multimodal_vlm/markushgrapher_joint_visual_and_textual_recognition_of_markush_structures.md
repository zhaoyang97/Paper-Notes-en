---
title: >-
  [Paper Note] MarkushGrapher: Joint Visual and Textual Recognition of Markush Structures
description: >-
  [CVPR 2025][Multimodal VLM][Markush structures] This paper proposes MarkushGrapher, a multimodal approach that recognizes Markush structures (chemical structure templates) in patent documents by jointly encoding text, image, and layout information. It also constructs M2S, the first real-world annotated benchmark for Markush structures, outperforming SOTA chemical-specific and general vision-language models under most evaluation settings.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Markush structures"
  - "chemical structure recognition"
  - "multimodal encoding"
  - "synthetic data"
  - "document analysis"
date: 2026-05-08
content_hash: 15ab7c710fa58518
---

# MarkushGrapher: Joint Visual and Textual Recognition of Markush Structures

**Conference**: CVPR 2025  
**arXiv**: [2503.16096](https://arxiv.org/abs/2503.16096)  
**Code**: Yes (to be released)  
**Area**: Multimodal VLM / Document Understanding  
**Keywords**: Markush structures, chemical structure recognition, multimodal encoding, synthetic data, document analysis

## TL;DR
This paper proposes MarkushGrapher, a multimodal approach that recognizes Markush structures (chemical structure templates) in patent documents by jointly encoding text, image, and layout information. It also constructs M2S, the first real-world annotated benchmark for Markush structures, outperforming SOTA chemical-specific and general vision-language models under most evaluation settings.

## Background & Motivation

**Background**: Automated analysis of chemical literature is highly valuable for accelerating materials science and drug discovery. In chemical patents, Markush structures represent a special class of chemical structures—instead of describing a single specific chemical molecule, they use variable groups to represent a generic template for a class of chemical structures. For example, a certain position labeled as "R1" can represent various groups such as methyl or ethyl. This representation is extremely common in patent literature as inventors need a Markush structure to cover as many compounds as possible to protect intellectual property.

**Limitations of Prior Work**: While significant progress has been made in automatically extracting ordinary chemical structures (e.g., SMILES, molecular graphs) from text and images (OCSR methods), the automatic recognition of Markush structures remains almost unexplored. The fundamental reason lies in the complex multimodal nature of Markush structures—they simultaneously contain chemical structure images (molecular skeletons, bonds, atoms) and textual information (variable definition tables, e.g., "R1 = CH3, C2H5, ..."), both of which must be jointly understood to obtain the full semantics. Looking at the image or text in isolation is insufficient to fully interpret a Markush structure. Additionally, the lack of annotated data is a major bottleneck.

**Key Challenge**: The recognition of Markush structures is a typical multimodal understanding problem—the chemical skeleton is understood via a vision encoder, the variable definitions are understood via a text encoder, and their spatial relationship (which variable label corresponds to which position on the skeleton) must be bridged by layout information. Existing OCSR methods only handle pure image inputs, while general VLMs lack domain-specific chemical knowledge, making both inadequate for the task.

**Goal**: (1) Design a multimodal architecture capable of jointly processing visual, textual, and layout information; (2) construct an annotated benchmark covering real-world Markush structures; (3) address the scarcity of training data.

**Key Insight**: The authors model the recognition of Markush structures as a "sequence generation from multimodal input to structured graph representation" task, where the output is not a single string but a serialized representation of a graph along with a variable definition table. This allows the model to progressively construct the complete Markush structure in an autoregressive manner.

**Core Idea**: Use a three-way Vision-Text-Layout encoder to jointly extract information, fuse it with a specialized chemical structure vision encoder (OCSR encoder), and autoregressively generate the graph representation and variable group definitions of the Markush structure.

## Method

### Overall Architecture
The system input consists of document pages containing Markush structures (comprising chemical structure images and variable definition text/tables). First, features are extracted via two parallel encoders: (1) a Vision-Text-Layout (VTL) encoder that processes the global layout, text, and visual information of the document; (2) an OCSR vision encoder focused on fine-grained features of chemical structure images. The two branches of features are merged through a fusion module and fed into an autoregressive decoder to sequentially generate: the graph representation of the chemical skeleton (node sequence + edge connections) and the variable group definition table (the list of possible values corresponding to each variable).

### Key Designs

1. **Vision-Text-Layout (VTL) Encoder**:

    - **Function**: Jointly encodes visual content, textual content, and spatial layout information in document pages.
    - **Mechanism**: Based on a pre-trained document understanding model (such as LayoutLMv3), it takes the document image and the text + position information extracted via OCR as input. Through a multimodal Transformer architecture, the VTL encoder allows visual patch tokens, text tokens, and position encodings (bounding box coordinates) to interact within a unified space. This enables the model to understand cross-modal spatial relations, such as "where a certain text label R1 appears on the chemical structure in the image."
    - **Design Motivation**: One of the critical challenges in Markush structure recognition is mapping variable labels to their positions on the chemical skeleton, which is a task fundamentally requiring spatial layout information. The VTL encoder inherently possesses the capability to process document-level multimodal alignment.

2. **OCSR Vision Encoder**:

    - **Function**: Extracts fine-grained molecular structure features from chemical structure images.
    - **Mechanism**: Adopts a vision encoder specifically trained for Optical Chemical Structure Recognition (OCSR), which is pre-trained on a massive dataset of chemical molecule images and possesses professional chemical knowledge such as identifying atom types, bond types, and stereochemical configurations. It encodes chemical structure images into a series of feature vectors containing molecule-level semantics.
    - **Design Motivation**: General vision encoders lack precise understanding of chemical symbols (such as double bonds, wedge bonds, benzene rings, etc.), whereas chemical-specific encoders provide necessary domain knowledge. The two encoders are complementary: VTL is responsible for understanding document-level layout and text, while OCSR is responsible for understanding detailed chemical structures.

3. **Sequential Graph Decoder**:

    - **Function**: Autoregressively transforms the merged multimodal features into a complete representation of the Markush structure.
    - **Mechanism**: Represents the Markush structure as a graph, where nodes are atoms or variables and edges are chemical bonds. The graph is serialized into a token sequence (using a linearized method similar to SMILES but extended for Markush structures). The decoder simultaneously generates two outputs: (a) the skeleton graph sequence, describing the topology and atom types of the molecular structure; (b) the variable group table, listing the possible values corresponding to each variable label. These two parts are concatenated in a fixed order and unifiedly generated by a single decoder.
    - **Design Motivation**: Modeling graph structures as a sequence generation problem allows the leverage of mature autoregressive Transformer architectures and training techniques, avoiding the design of complex graph generation networks. Meanwhile, jointly generating the skeleton and the variable table ensures consistency between the two.

### Loss & Training
Standard autoregressive cross-entropy loss is used to train the decoder. To address the scarcity of real annotated data, a synthetic data generation pipeline is designed—sampling real molecular structures from a chemical database, randomly selecting positions to replace with variable labels, automatically rendering them into document-style images, and generating corresponding annotations. The synthetic data covers a variety of rendering styles (different fonts, resolutions, noise levels) to improve generalization.

## Key Experimental Results

### Main Results

| Method | Skeleton Exact Match↑ | Variable Table F1↑ | Overall F1↑ | Method Type |
|------|-------------|----------|---------|---------|
| MolScribe (OCSR) | 38.2% | - | 25.1% | Chemical-specific |
| GPT-4V | 22.7% | 31.5% | 24.3% | General VLM |
| Gemini Pro Vision | 19.8% | 28.2% | 21.6% | General VLM |
| InternVL2 | 25.4% | 33.7% | 27.8% | General VLM |
| **MarkushGrapher** | **52.6%** | **58.3%** | **51.4%** | Multimodal-specific |

### Ablation Study

| Configuration | Skeleton Exact Match↑ | Variable Table F1↑ | Description |
|------|-------------|----------|------|
| Full model | 52.6% | 58.3% | Full model |
| w/o OCSR encoder | 43.1% | 55.8% | Drops 9.5% without chemical encoder |
| w/o VTL encoder | 37.5% | 41.2% | Drops 15.1% without layout-text encoder |
| w/o synthetic data pre-training | 39.8% | 44.6% | Synthetic data contributes significantly |
| Trained on real data only | 31.2% | 36.4% | Real data is too scarce |
| Trained on synthetic data only | 45.3% | 50.1% | Synthetic + real fine-tuning is optimal |

### Key Findings
- The contribution of the VTL encoder is larger than that of the OCSR encoder (dropping VTL leads to a 15.1% drop vs. dropping OCSR leads to a 9.5% drop), proving that layout and textual information are crucial for Markush structure recognition—understanding the variable definition table relies heavily on text and layout.
- General VLMs (GPT-4V, Gemini) perform poorly on this task, indicating that chemical structure recognition indeed requires specialized models.
- Synthetic data pre-training contributes an improvement of approximately 12.8%, validating the effectiveness of the synthetic data pipeline.
- For simple Markush structures ($\le 3$ variable groups), the accuracy reaches 70%+, but for complex structures ($> 6$ variable groups), it drops to around 30%.
- Recognizing the variable table is more challenging than the skeleton, as it requires precise cross-modal correspondence.

## Highlights & Insights
- **The problem definition itself is a significant contribution**: Markush structure recognition is a core bottleneck in chemical patent analysis, yet almost no AI methods were previously tailored for it. The construction of the M2S benchmark provides evaluation tools for the community.
- The **complementary nature of the dual-encoder design** is highly ingenious: the VTL encoder understands "what the document looks like" (where text is, where images are, and their spatial relationships), and the OCSR encoder understands "what the chemical structure means" (atoms, bonds, stereochemistry). Their fusion enables a complete interpretation of the Markush structure.
- The design of the **synthetic data pipeline** has general reference value: in professional domains where annotated data is extremely scarce, procedurally generating training data starting from domain databases is a practical strategy.

## Limitations & Future Work
- The accuracy of the current method remains limited on complex Markush structures (multi-level nested variables, large molecular skeletons).
- The scale of the M2S benchmark is small, which may be insufficient to fully represent the diversity of Markush structures in chemical patents.
- The robustness of the model against hand-drawn or low-resolution scanned chemical structure images is not fully validated.
- Future work can integrate Markush structure recognition with complete patent document parsing pipelines to achieve end-to-end chemical information extraction.
- Future work can explore using chemical knowledge graphs to verify the chemical validity of generated Markush structures, further improving accuracy.

## Related Work & Insights
- **vs MolScribe/DECIMER**: These OCSR methods focus on recognizing single chemical structures from images and do not handle textual information or variable definitions. MarkushGrapher is an important extension of them into the Markush structure dimension.
- **vs General VLMs such as GPT-4V**: General vision-language models lack domain-specific chemical knowledge; although they can understand text, they cannot accurately interpret chemical symbols. This illustrates that in highly specialized domains, specialized models still hold an advantage over general large models.
- **vs LayoutLMv3**: The VTL encoder draws on the design of document understanding models but is adapted to the specificities of chemical documents. This "general pre-training + domain adaptation" paradigm is worth promoting in other specialized domains.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to systematically address the Markush structure recognition problem, with a highly targeted dual-encoder design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Constructed the first benchmark, compared comprehensively with multiple baselines, and ablated key modules.
- **Writing Quality**: ⭐⭐⭐⭐ Clear presentation of the problem background, helping readers without a chemistry background understand Markush structures.
- **Value**: ⭐⭐⭐⭐ Direct practical value for the field of chemical patent analysis, with the M2S benchmark driving progress in this direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MarkushGrapher-2: End-to-end Multimodal Recognition of Chemical Structures](../../CVPR2026/multimodal_vlm/markushgrapher-2_end-to-end_multimodal_recognition_of_chemical_structures.md)
- [\[CVPR 2025\] Joint Vision-Language Social Bias Removal for CLIP](joint_vision-language_social_bias_removal_for_clip.md)
- [\[CVPR 2025\] Recognition-Synergistic Scene Text Editing](recognition-synergistic_scene_text_editing.md)
- [\[CVPR 2025\] Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](synthetic_data_is_an_elegant_gift_for_continual_vision-language_models.md)
- [\[CVPR 2025\] Relation-Rich Visual Document Generator for Visual Information Extraction](relation-rich_visual_document_generator_for_visual_information_extraction.md)

</div>

<!-- RELATED:END -->
