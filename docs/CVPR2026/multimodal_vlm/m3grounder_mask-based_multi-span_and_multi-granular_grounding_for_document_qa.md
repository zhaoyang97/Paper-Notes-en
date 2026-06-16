---
title: >-
  [Paper Note] M3Grounder: Mask-Based Multi-Span and Multi-Granular Grounding for Document QA
description: >-
  [CVPR 2026][Multimodal VLM][Paper Note] M3Grounder transforms "answer localization" in Document QA from coarse bounding boxes to pixel-level segmentation. While the VLM generates answers, it emits `[GROUND]` tokens. Each token drives a promptable segmentation module via three MLP heads (phrase, line, and block levels) to produce nested multi-granular evidenc
tags:
  - CVPR 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 037fb8059a65087b
---
# M3Grounder: Mask-Based Multi-Span and Multi-Granular Grounding for Document QA

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Venna_M3Grounder_Mask-Based_Multi-Span_and_Multi_Granular_Grounding_for_Document_QA_CVPR_2026_paper.html)  
**Code**: To be confirmed (Paper states code / data / models will be open-sourced)  
**Area**: Multimodal VLM  
**Keywords**: Document QA, pixel-level grounding, segmentation, multi-granularity, data engine

## TL;DR
M3Grounder transforms "answer localization" in Document QA from coarse bounding boxes to pixel-level segmentation. While the VLM generates answers, it emits `[GROUND]` tokens. Each token drives a promptable segmentation module via three MLP heads (phrase, line, and block levels) to produce nested multi-granular evidence masks, achieving SOTA results across four benchmarks.

## Background & Motivation

**Background**: Document Visual Question Answering (DocVQA) requires Vision-Language Models (VLMs) to simultaneously understand text and layout. The mainstream approach treats it as a pure text generation task, outputting only answer strings. A few methods supporting "answer localization" (grounding), such as DOGR, Qwen3-VL, and InternVL3.5, intersperse bounding box coordinates within the textual answers.

**Limitations of Prior Work**: Pure text generation fails to indicate where in the page the answer was derived from, which is critical in high-traceability scenarios like medical, legal, and financial domains. Conversely, methods outputting bounding boxes provide only coarse rectangular localizations; rectangles cannot fit curved text (circular layouts, tilted headers) and often include significant irrelevant background noise, leading to localization ambiguity.

**Key Challenge**: Coupling "language modeling" and "spatial localization" within the same autoregressive sequence (mixing coordinates and text) forces the model to manage both semantic correctness and geometric precision simultaneously, often compromising both. Furthermore, rectangular representations cannot accurately capture the irregular geometric shapes of text truly present in documents.

**Goal**: (1) Replace box prediction with pixel-level segmentation for grounding to match real text shapes; (2) Support multiple dispersed evidence regions (multi-span) for a single answer; (3) Perform multi-granular localization following the natural document hierarchy of "phrase ⊂ line ⊂ block"; (4) Construct a large-scale dataset with pixel mask annotations.

**Key Insight**: The authors observe that documents possess a natural spatial hierarchy—words form lines, and lines form blocks. Each granularity corresponds to a different reasoning scope (extractive questions like "Name:" only need phrase-level grounding, while summarization requires block-level). Thus, grounding is decoupled into specialized segmentation heads, allowing the VLM to focus on semantic accuracy and span boundaries.

**Core Idea**: During autoregressive answer generation, the VLM appends a `[GROUND]` token after each evidence span. The hidden state of this token prompts a segmentation module to output nested masks at phrase, line, and block levels. This "segmentation instead of boxes, decoupled via specialized tokens" approach solves fine-grained grounding.

## Method

### Overall Architecture
Given a document image $x$ and a question $q$, M3Grounder autoregressively generates an answer in the format `... <e> yₖ </e>[GROUND] ...`. Answer spans are marked by `<e>...</e>`, and the subsequent `[GROUND]` token triggers multi-granular mask generation for that span. The model is a hybrid of a "VLM backbone + promptable segmentation module"—the VLM reads images and determines which text segments require grounding, while the segmentation module executes the pixel-level localization.

The key lies in **decoupling**: unlike previous methods that insert coordinates into text sequences, M3Grounder separates language modeling from spatial prediction. The final hidden state $\tilde h_k$ of the `[GROUND]` token is mapped by three granularity-specific MLP heads into phrase, line, and block prompts $h_k^{(p)}, h_k^{(l)}, h_k^{(b)}$. Meanwhile, the segmentation module (based on SAM) extracts dense image features $z=F_{enc}(x)$ once, which are reused across all spans and granularities. Finally, a mask decoder $F_{dec}$ uses the prompts to decode masks $\hat M_k^{(i)}$ for each level.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Document Image x + Question q"] --> B["VLM Autoregressive Generation:<br/>Answer + Decoupled [GROUND] tokens"]
    B -->|Hidden state of each [GROUND]| C["Multi-Granular Hierarchical Grounding:<br/>Three MLP Heads + Nested Inclusion Constraints"]
    A -->|Dense Image Feature Reuse| D["Promptable Segmentation Module:<br/>SAM Mask Decoder"]
    C --> D
    D --> E["Bleed Suppression Constraint:<br/>Masks stay within text pixels"]
    E --> F["Phrase/Line/Block Evidence Masks<br/>for each span"]
    G["GroundingDocQA Data Engine:<br/>200K Docs / 2M QA pairs"] -.Training Supervision.-> B
```

### Key Designs

**1. Decoupled VLM-Segmentation Grounding: Separating "Writing Answers" from "Drawing Masks"**

To address the issue where coordinate insertion drags down both semantics and geometry, M3Grounder introduces a special `[GROUND]` token. The VLM emits this token immediately after generating an answer span (`<e>yₖ</e>`), establishing a one-to-one mapping. The language model handles text and span boundaries, while geometric localization is delegated to the segmentation head. The final hidden state $\tilde h_k$ of the `[GROUND]` token is projected into segmentation prompts, bypassing the need for the VLM to generate numerical coordinates. This division of labor allows the VLM to focus on semantics while the segmentation head focuses on geometric perception. In experiments, this provides a significant advantage over box-based methods on curved or tilted text (CS subset), as masks can conform to actual text contours.

**2. Multi-Granular Hierarchical Grounding: Three MLP Heads + Nested Inclusion Constraints**

To meet the requirement that different questions need different localization scales and that granularities should be spatially consistent, each `[GROUND]` hidden state is projected by three independent MLP heads into phrase, line, and block prompts. To ensure spatial consistency, a hierarchical inclusion loss $L_{hier}$ is introduced to force finer masks to be contained within coarser ones ($p \subset l \subset b$). This loss penalizes pixels that violate inclusion relationships:

$$L_{hier}=\sum_{k=1}^{K}\sum_{(i,j)\in\{(p,l),(l,b)\}}\frac{\sum_{m\in\Omega}\hat M_k^{(i)}(m)\,[1-\hat M_k^{(j)}(m)]}{\sum_{m\in\Omega}\hat M_k^{(i)}(m)+\epsilon}$$

This calculates the proportion of finer mask pixels $\hat M_k^{(i)}$ falling outside the coarser mask $\hat M_k^{(j)}$. This constraint not only ensures spatial consistency but also improves overall grounding accuracy and answer quality (Block-level F1g reached 82.5 / 87.5). Replacing SAM's default shared MLP with three granularity-specific heads is a critical design choice.

**3. Bleed Suppression Loss $L_{bleed}$: Keeping Masks Within Text Pixels**

To prevent segmentation masks from spilling over into non-text background areas, the authors add a bleed suppression loss. Let $M_{ref}$ be the union of all text regions and $\Omega$ be all pixel coordinates:

$$L_{bleed}=\sum_{i\in\mathcal G}\sum_{k=1}^{K}\frac{\sum_{j\in\Omega}\hat M_k^{(i)}(j)\,[1-M_{ref}(j)]}{\sum_{j\in\Omega}\hat M_k^{(i)}(j)+\epsilon}$$

The numerator represents the part of the predicted mask outside text regions (background), while the denominator is the total mask area—essentially penalizing the "proportion drawn on background." The total loss is $L_{total}=\lambda_{lm}L_{lm}+\lambda_{seg}L_{seg}+\lambda_{bleed}L_{bleed}+\lambda_{hier}L_{hier}$, where $L_{seg}$ is a combination of Dice and BCE with decreasing weights for phrase/line/block levels.

**4. GroundingDocQA Data Engine: Creating Multi-Span and Multi-Granular Supervision**

To fill the gap where existing document grounding data only contains boxes and lacks hierarchical clues, the authors constructed GroundingDocQA with 200,000 documents and 2 million QA pairs using three complementary pipelines. ① **Layout-aware Documents**: Uses the REPLICA engine to convert documents into high-fidelity HTML (Fid-HTML) that preserves spatial structure. LLMs generate QA pairs with associated HTML element IDs, which are mapped to bounding boxes and then to pixel masks for phrase, line, and block levels. ② **Curved Text Documents**: Employs curved text segmentation to get pixel-level masks, used as "highlight" prompts for a VLM to generate QA pairs ensuring precise alignment. ③ **Charts**: Executes plotting scripts to render charts, intercepting drawing library functions to record exact element bounding boxes, thus avoiding detection errors.

### Loss & Training
The model is optimized end-to-end for text generation and mask prediction. $L_{lm}$ is standard cross-entropy; $L_{seg}$ uses Dice + BCE with decreasing coefficients for phrase/line/block (BCE: 2.0 / 1.0 / 0.5, Dice: 1.0 / 0.5 / 0.25). Weights are fixed at $\lambda_{lm}=1$, $\lambda_{seg}=2$, $\lambda_{bleed}=\lambda_{hier}=0.5$. The optimizer is AdamW with a learning rate of $2\times10^{-6}$, 3% warmup, and cosine decay. Two backbone variants were used: M3Grounder-I (InternVL3.5-8B) and M3Grounder-Q (Qwen3-VL-8B), both paired with SAM. Training utilized 64 H100 GPUs.

## Key Experimental Results

### Main Results
Comparison on four grounding benchmarks using F1g (IoU>0.5) for grounding and G-Eval for Answer Quality (AQ). SS/MS denote Single/Multi-span F1g; CS denotes the Curved/Slanted subset.

| Model | Scale | BD-Test F1g | DOGR-Bench F1g | MMDoc IoU | GroundingDocQA F1g | GroundingDocQA CS |
|------|------|------|------|------|------|------|
| Gemini-2.5-Pro (Comm.) | – | 70.0 | 59.3 | 49.4 | 43.4 | 32.1 |
| InternVL3.5 (Zero-shot) | 8B | 41.5 | 12.5 | 15.5 | 7.6 | 6.3 |
| Qwen3-VL (Zero-shot) | 8B | 44.5 | 27.6 | 28.7 | 12.8 | 11.6 |
| DOGR (Grounding-spec.) | 8B | – | 66.4 | – | – | – |
| Qwen3-VL Finetuned | 8B | 62.3 | 35.8 | 43.4 | 60.6 | 38.3 |
| **M3Grounder-I** | 8B | 77.2 | 69.6 | 65.5 | 71.3 | 81.7 |
| **M3Grounder-Q** | 8B | **81.4** | **73.3** | **68.2** | **79.0** | **85.3** |

Both variants achieved open-source SOTA across all benchmarks. Most notably, on the Curved/Slanted (CS) subset, M3Grounder-Q reached 85.3, far exceeding Gemini-2.5-Pro (32.1) and finetuned Qwen3-VL (38.3), proving that segmentation fits irregular geometries better than boxes.

Multi-granular results (GroundingDocQA-Bench, F1g):

| Model | Phrase | Line | Block |
|------|------|------|------|
| M3Grounder-I | 71.3 | 74.9 | 82.5 |
| M3Grounder-Q | 79.0 | 81.37 | 87.5 |

### Ablation Study
Tested on hierarchy, loss terms, and finetuning strategies.

| Configuration | GR-B Phrase | GR-B Block | Description |
|------|------|------|------|
| Full M3Grounder-Q | 79.0 | 87.5 | Full model |
| w/o Hierarchy (Phrase only) | 71.3 | – | Removing multi-granularity drops F1g from 79.0 to 71.3 |
| Shared MLP + SAM Default | 63.6 | 77.6 | Replacing 3 heads with SAM single head causes total performance drop |
| $L_{lm}+L_{seg}$ (Base) | 74.0 | 84.4 | Baseline without auxiliary losses |
| $+\,L_{hier}$ | 78.2 | 84.7 | Adding hierarchical loss shows stable improvement |
| $+\,L_{bleed}$ | 77.3 | 86.5 | Adding bleed suppression provides stable gains |
| LoRA (PEFT) | 61.5 | 75.7 | PEFT is significantly inferior to full finetuning |

### Key Findings
- **Multi-granular supervision is the biggest contributor**: Removing the hierarchy drops performance from 79.0 to 71.3.
- **Dedicated MLP heads are essential**: Using SAM's default single head leads to performance degradation across benchmarks.
- **Auxiliary losses are effective**: $L_{hier}$ improves grounding stability via nested inclusion, while $L_{bleed}$ prevents evidence from overflowing into adjacent text.
- **Fine-grained grounding requires full finetuning**: LoRA significantly underperforms, particularly at the phrase level, suggesting pixel-level document localization requires updating all parameters.

## Highlights & Insights
- **Decoupling language and localization via `[GROUND]` tokens**: Instead of embedding coordinates in text, using a dedicated token as a "pointer" to a segmentation head allows the VLM to focus on semantics while the segmentation module handles geometry. This paradigm is transferable to any task requiring "text generation + region localization."
- **Hierarchical loss as a spatial prior**: $L_{hier}$ was intended for spatial consistency but was found to improve grounding and answer accuracy by acting as a structural regularizer.
- **Intercepting drawing functions for charts**: Capturing chart element boxes during the rendering process instead of using detectors ensures zero detection error, providing a robust data engineering trick.

## Limitations & Future Work
- Mask-based grounding provides higher spatial precision but incurs higher computational overhead compared to box-based methods.
- Currently operates at a single-page level; hierarchical grounding for multi-page or long-context documents remains an open challenge.
- The method relies heavily on external components in the data engine (REPLICA, curved text pipelines, library interception), which limits annotation quality to the performance of these tools.

## Related Work & Insights
- **vs DOGR**: DOGR predicts boxes jointly with text, which is better than general VLMs but limited to single-granularity rectangular regions. M3Grounder uses masks and three granularities, outperforming DOGR by 6.9 points on DOGR-Bench.
- **vs General VLMs (Qwen3-VL / InternVL3.5)**: These models support box output but aren't optimized for grounding. Even when finetuned on GroundingDocQA, they are outperformed by M3Grounder’s specialized segmentation head.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

- [DOGR: Document Grounding with Large Vision-Language Models](https://arxiv.org/abs/2410.xxxx)
- [Qwen2-VL: To See the World More Clearly](https://arxiv.org/abs/2410.xxxx)

</div>
<!-- RELATED:END -->

## Related Papers

- [\[CVPR 2026\] β-CLIP: Text-Conditioned Contrastive Learning for Multi-Granular Vision-Language Alignment](b-clip_text-conditioned_contrastive_learning_for_multi-granular_vision-language_.md)
- [\[CVPR 2025\] MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding](../../CVPR2025/multimodal_vlm/marten_visual_question_answering_with_mask_generation_for_multi-modal_document_u.md)
- [\[CVPR 2026\] GroundingME: Exposing the Visual Grounding Gap in MLLMs through Multi-Dimensional Evaluation](groundingme_exposing_the_visual_grounding_gap_in_mllms_through_multi-dimensional.md)
- [\[CVPR 2026\] VinQA: Visual Elements Interleaved Long-form Answer Generation for Real-World Multimodal Document QA](vinqa_visual_elements_interleaved_long-form_answer_generation_for_real-world_mul.md)
- [\[CVPR 2026\] Hierarchical Attacks for Multi-Modal Multi-Agent Reasoning](hierarchical_attacks_for_multi-modal_multi-agent_reasoning.md)

</div>

<!-- RELATED:END -->
