---
title: >-
  [Paper Note] What's in the Image? A Deep-Dive into the Vision of Vision Language Models
description: >-
  [CVPR 2025][Multimodal VLM][VLM interpretability] This paper systematically analyzes the visual information processing mechanism of VLMs (InternVL2-76B and LLaVA-1.5-7B) through Attention Knockout experiments, revealing three key findings: (1) query text tokens act as global image describers that compress high-level visual information, (2) the middle layers (~25%) dominate the cross-modal information transfer while early and late layers contribute minimally…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "VLM interpretability"
  - "attention mechanism"
  - "visual information compression"
  - "middle-layer analysis"
  - "Image Re-prompting"
date: 2026-05-08
content_hash: af84c76ff809d287
---

# What's in the Image? A Deep-Dive into the Vision of Vision Language Models

**Conference**: CVPR 2025  
**arXiv**: [2411.17491](https://arxiv.org/abs/2411.17491)  
**Code**: [https://vision-of-vlm.github.io/](https://vision-of-vlm.github.io/) (with project page)  
**Area**: Multimodal VLM  
**Keywords**: VLM interpretability, attention mechanism, visual information compression, middle-layer analysis, Image Re-prompting

## TL;DR
This paper systematically analyzes the visual information processing mechanism of VLMs (InternVL2-76B and LLaVA-1.5-7B) through Attention Knockout experiments, revealing three key findings: (1) query text tokens act as global image describers that compress high-level visual information, (2) the middle layers (~25%) dominate the cross-modal information transfer while early and late layers contribute minimally, and (3) fine-grained object details are extracted from image tokens through spatial localization. Based on these findings, an Image Re-prompting application is proposed, which maintains 96% of VQA performance using only 5% of the image tokens.

## Background & Motivation
VLMs have demonstrated impressive capabilities on tasks such as image captioning and VQA. However, their internal visual information processing mechanism remains a black box—it is largely unknown how the model extracts information from image tokens, how cross-modal information flows, and what roles different layers play. Understanding these mechanisms is crucial for improving model transparency, enhancing inference efficiency, and guiding future VLM designs. Existing interpretability works either focus on LLMs (excluding vision) or small models (<10B), with very few conducting deep analyses on SOTA VLMs at the 76B parameter scale. **Core Problem**: How is visual information encoded, transferred, and utilized when VLMs generate text?

## Method

### Overall Architecture
The analysis framework is based on Attention Knockout: during VLM inference, information flow between specific token types is blocked by modifying the attention mask, and changes in the output are observed. A customized LLM-as-a-judge evaluation protocol is integrated to quantify these changes. The analysis focuses on three token types: image tokens $\mathbf{T}_{img}$, query text tokens $\mathbf{T}_{txt}$ (e.g., "describe the image"), and generated tokens $\mathbf{T}_{gen}$.

### Key Designs
1. **Attention Knockout Experimental Framework**:
    - **Function**: Revealing the roles of various token types and layers by blocking information flow.
    - **Mechanism**: Defining the knockout mask $\mathbf{M}_{ko}[p,q; P_{src}, P_{tgt}]$, which is set to $-\infty$ if $q \in P_{src}$ and $p \in P_{tgt}$. Three core configurations are used: (a) $\text{KO}_{img \to gen}$—blocking the direct influence of image tokens on generated tokens, so visual information can only transfer indirectly through query tokens; (b) $\text{KO}_{img \to txt}$—blocking the influence of image tokens on query tokens; (c) $\text{KO}_{img \to txt+gen}$—completely blocking the influence of image tokens on other tokens. By applying knockouts starting from different layers $l$, the contributions of individual layers can be analyzed.
    - **Design Motivation**: Because VLMs use causal attention masks, the representations of image tokens are fixed after the first decoding step; thus, blocking information flow in different directions can cleanly isolate the contributions of different information pathways.

2. **LLM-as-a-Judge Evaluation Protocol**:
    - **Function**: Automatically quantifying the impact of knockouts on output quality.
    - **Mechanism**: Instructing the LLM to extract object lists $O_{orig}$ and $O_{ko}$ from the original and modified VLM descriptions respectively, and calculating TP (objects mentioned in both), FN (present in raw but missing in knockout), and FP (hallucinated objects newly appearing in knockout) to obtain Precision/Recall/F1 scores. Chain-of-thought and three in-context examples are used to enhance reliability.
    - **Design Motivation**: Comparing semantic differences between two free-form text snippets is challenging due to large variations in wording and style, and annotations in datasets like COCO are often incomplete. Therefore, the language understanding capability of LLMs is leveraged for automatic evaluation. User studies indicate that the LLM's judgments achieve a 95% agreement rate with humans.

3. **Spatial Localization Analysis**:
    - **Function**: Revealing how VLMs extract fine-grained information of specific objects from image tokens.
    - **Mechanism**: Identifying objects lost under $\text{KO}_{img \to gen}$ (i.e., details that must be directly retrieved from image tokens) and visualizing the attention heatmaps of generated tokens on image tokens. SAM is used to generate pseudo-ground-truth segmentation masks of the objects to detect whether attention peaks fall within the object regions (within a 40-pixel tolerance, equivalent to approximately 1 token distance). A Localization Accuracy metric is defined.
    - **Design Motivation**: The $\text{KO}_{img \to gen}$ experiments demonstrate that high-level information is compressed into query tokens, but fine-grained details are lost. Through what mechanism are these details retrieved? The answer is spatial localization—attention in the middle layers precisely points to the target objects.

### Loss & Training
This work is analytical and involves no training. For the Image Re-prompting application, the top-K% image tokens with the highest middle-layer attention weights during "describe the image" generation are extracted as the compressed context. Subsequent questions directly utilize this compressed context instead of the complete image.

## Key Experimental Results

### Quantitative Metrics of Key Findings

| Knockout Configuration | F1 Score | Implication |
|--------------|---------|------|
| $\text{KO}_{img \to gen}$ (all layers) | 0.40 | Some objects can be recognized relying solely on compressed information in query tokens. |
| $\text{KO}_{img \to txt}$ (all layers) | 0.00 | Completely fails after blocking query tokens from obtaining visual information! |
| $\text{KO}_{img \to gen}^{l \notin [20,40]}$ (only middle layers accessed) | 0.75-0.81 | Middle layers 20-40 preserve almost full performance. |
| KO after layer 40 | ≈0.80 | Attention after layer 40 plays almost no role. |

### Image Re-prompting (MME benchmark)

| Method | ACC | ACC+ | Token Count |
|------|-----|------|---------|
| Naive (InternVL2 full) | 84.83 | 70.60 | 1695 |
| Describe-to-LLM | 73.21 | 56.14 | 172 |
| Query + K=5% | **81.46** | **64.52** | 201 |
| Query + K=2% | 77.16 | 55.94 | 151 |
| Query only (no image tokens) | 61.03 | 28.45 | 60 |

### Spatial Localization Accuracy

| Layer Range | Localization Accuracy |
|--------|---------------------|
| Layers 0-10 | ~30% |
| Layers 10-20 | ~45% |
| Layers 20-30 | ~65% |
| Layers 30-40 | **~73%** |
| Layers 40-50 | ~60% |

### Key Findings
- **Query tokens act as global image describers**: Although query tokens constitute less than 5% of total tokens, they capture more than 60% of the attention in most layers. Relying solely on query tokens ($\text{KO}_{img \to gen}$) can still produce descriptive responses.
- **Middle layers dominate visual information transfer**: Allowing only layers 20-40 to access image tokens (25% of the total 80 layers) yields an F1 score of 0.75-0.81, showing almost no degradation compared to the full model.
- **Fine-grained information is retrieved via spatial localization**: Attention peaks in the middle layers precisely correspond to the target object's location in the image, with localization accuracy reaching 73% in the middle layers.
- Using only 5% of the high-attention image tokens preserves near-complete performance (F1 score saturates rapidly).

## Highlights & Insights
- **Discovery of the dual-pathway visual processing mechanism** is highly insightful: (1) global information is compressed into query tokens through the middle layers; (2) fine-grained information is extracted from image tokens via spatially localized attention in the middle layers.
- **The LLM-as-a-judge evaluation protocol** serves as a useful general-purpose tool that addresses the difficulty of free-form text comparison, achieving a 95% agreement rate with human evaluation.
- **Image Re-prompting**, as a practical application derived from the analysis, is highly valuable—it significantly reduces token consumption in multi-turn QA (yielding a 12x compression).
- This is the first work to perform interpretability analysis on a VLM of 76B parameter scale.
- Query tokens account for less than 5% of the total tokens but bear over 60% of the attention; this finding implies an inherent compression tendency in VLMs.
- An F1 score of 0 in the $\text{KO}_{img \to txt}$ experiment proves that for VLMs, query tokens serve not only as instruction carriers but also as indispensable hubs for visual information transit.
- The long-tailed distribution of attention implies that only a small fraction of image tokens receive significant attention, providing strong theoretical support for token pruning.

## Limitations & Future Work
- The analysis only covers a single query ("describe the image"); different query types (e.g., reasoning, counting) may exhibit different information flow patterns.
- Only the attention modules are analyzed, neglecting the role of FFNs in information storage and transformation.
- Image Re-prompting requires performing a full forward pass of "describe" beforehand to extract the compressed context, which does not constitute entirely "free" acceleration.
- The localization accuracy analysis utilizes pseudo-ground-truth annotations generated by SAM, which inherently contain noise.
- The definition of the middle layers (20-40) is specific to InternVL2-76B; it may vary across different architectures or scales.

## Related Work & Insights
- **vs. Logit Lens methods**: Logit Lens reveals what information is encoded in each layer, whereas this work demonstrates how information flows across modalities, offering a complementary perspective.
- **vs. LLM interpretability works (e.g., Geva et al.)**: Prior works analyze FFNs/attention in unimodal LLMs, while this work is the first to systematically analyze the cross-modal information flow in VLMs.
- **vs. token pruning methods**: Existing token compression techniques are typically based on importance scores, whereas this work provides theoretical support from an interpretability perspective, showing that most image tokens are indeed redundant.
- **vs. Basu et al. (2024)**: They study information storage locations, while this work focuses more on how information flows across modalities.

## Supplementary Notes
- Experiments carry out attention analysis on 80 random images from the COCO dataset, with 231 objects used for localization accuracy evaluation.
- InternVL2-76B consists of 80 Transformer layers, with the middle layers defined as layers 20-40.
- Results for LLaVA-1.5-7B are presented in the appendix, exhibiting consistent trends.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to systematically reveal the dual-pathway processing mechanism of visual information in large-scale VLMs; the findings are highly surprising.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Involves various knockout configurations, LLM evaluation paired with human validation, and spatial localization analysis, though on a relatively small data scale (80 images).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Very clear logical flow, progressively deeper experiments, and excellent visualizations.
- **Value**: ⭐⭐⭐⭐⭐ Highly insightful for understanding internal VLM mechanisms and guiding efficient VLM design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Identifying and Mitigating Position Bias of Multi-image Vision-Language Models](identifying_and_mitigating_position_bias_of_multi-image_vision-language_models.md)
- [\[CVPR 2025\] FastVLM: Efficient Vision Encoding for Vision Language Models](fastvlm_efficient_vision_encoding_for_vision_language_models.md)
- [\[ACL 2026\] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](../../ACL2026/multimodal_vlm/what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)
- [\[CVPR 2025\] Cropper: Vision-Language Model for Image Cropping through In-Context Learning](cropper_vision-language_model_for_image_cropping_through_in-context_learning.md)
- [\[CVPR 2025\] Words or Vision: Do Vision-Language Models Have Blind Faith in Text?](words_or_vision_do_vision-language_models_have_blind_faith_in_text.md)

</div>

<!-- RELATED:END -->
