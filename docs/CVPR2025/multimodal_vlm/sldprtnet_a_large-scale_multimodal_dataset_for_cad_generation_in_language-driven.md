---
title: >-
  [Paper Note] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design
description: >-
  [CVPR 2025][Multimodal VLM][CAD dataset] This paper constructs SldprtNet, a large-scale multimodal CAD dataset containing over 240k industrial parts. Each sample aligns four modalities: 3D models, multi-view images, parametric modeling scripts, and natural language descriptions. An encoder/decoder tool supporting 13 CAD operations is developed to achieve lossless bidirectional conversion. Experiments demonstrate that multimodal input significantly outperforms text-only input.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "CAD dataset"
  - "Multimodal"
  - "Text-to-CAD"
  - "Parametric modeling"
  - "SolidWorks"
date: 2026-05-08
content_hash: d89e6f819810693e
---

# SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design

**Conference**: CVPR 2025  
**arXiv**: [2603.13098](https://arxiv.org/abs/2603.13098)  
**Code**: Unreleased  
**Area**: Others  
**Keywords**: CAD dataset, Multimodal, Text-to-CAD, Parametric modeling, SolidWorks

## TL;DR

This paper constructs SldprtNet, a large-scale multimodal CAD dataset containing over 240k industrial parts. Each sample aligns four modalities: 3D models, multi-view images, parametric modeling scripts, and natural language descriptions. An encoder/decoder tool supporting 13 CAD operations is developed to achieve lossless bidirectional conversion. Experiments demonstrate that multimodal input significantly outperforms text-only input.

## Background & Motivation

**Background**: CAD (Computer-Aided Design) is crucial in mechanical design and manufacturing. Existing 3D model datasets like ShapeNet and ModelNet are primarily stored in mesh or point cloud formats, retaining only the final surface geometry while losing the modeling history and parametric information. A few parametric datasets such as ABC and Fusion 360 Gallery maintain geometric precision, but lack semantic-level text annotations.

**Limitations of Prior Work**: Text-to-CAD modeling faces three core problems: (1) Small data scale—CAD datasets require manual creation by professionals, making them far smaller than image/text datasets; (2) Missing modalities—existing datasets typically cover only a single modality (geometry, sequence, or text), failing to support cross-modal learning; (3) Restricted operation types—DeepCAD and Text2CAD only support sketch and extrusion, severely limiting the types of covered parts.

**Key Challenge**: Modern multimodal models (such as CLIP, Flamingo, BLIP-2) have proven that cross-modal aligned learning is essential for generalization and transfer, but the CAD domain lacks a genuinely aligned, large-scale multimodal dataset to sustain these approaches.

**Goal**: To build a large-scale CAD dataset that supports multimodal data, bidirectional conversion, semantic annotations, and is editable and human-readable.

**Key Insight**: The authors use the SolidWorks API to develop encoder/decoder tools that losslessly convert native .sldprt files into structured text, while rendering multi-view images and utilizing a multimodal LLM to generate natural language descriptions, achieving full alignment across the four modalities.

**Core Idea**: To build a closed-loop toolchain with the SolidWorks API to extract aligned 3D models, images, parametric scripts, and natural language descriptions from over 240k industrial parts, thereby constructing a multimodal dataset for Text-to-CAD.

## Method

### Overall Architecture

The pipeline for constructing SldprtNet consists of four stages: (1) collecting approximately 680k .sldprt files from three platforms (GrabCAD, McMaster-Carr, and FreeCAD); (2) filtering to retain over 240k high-quality models containing at least one of the 13 feature types; (3) extracting four modal data sources via automated tools; (4) utilizing a multimodal LLM to generate and manually verify natural language descriptions.

### Key Designs

1. **Encoder (CAD $\to$ Text)**:

    - **Function**: Losslessly convert .sldprt files into structured parametric text.
    - **Mechanism**: Automatically traverse the Feature Tree, extracting feature types, names, and parent-child relationships in chronological modeling order. It then calls corresponding modules for each feature to extract detailed parameters (dimensions, constraints, sketch entities, etc.), generating a human- and machine-readable text format.
    - **Design Motivation**: Support 13 CAD operations (including extrude, chamfer, fillet, linear pattern, mirror pattern, etc.), which far exceeds the 2 operations of DeepCAD, greatly expanding the complexity and diversity of the coverable parts.

2. **Decoder (Text $\to$ CAD)**:

    - **Function**: Reconstruct complete 3D part models from parametric text.
    - **Mechanism**: Create a blank .sldprt document first, parse the Feature Tree, and sequentially call the SolidWorks API according to the feature order and hierarchical relationships to construct the corresponding features, ensuring geometric and topological consistency.
    - **Design Motivation**: Forming a closed-loop system with the encoder to support "model $\to$ text $\to$ model" round-trip translation, which can be applied to structural validation, data augmentation, and synthetic data generation.

3. **Multimodal Alignment Generation**:

    - **Function**: Generate aligned multi-view synthetic images and natural language descriptions for each 3D model.
    - **Mechanism**: Render 6 orthographic views (front/back/left/right/top/bottom) + 1 isometric view, merging them into a single image to reduce token usage; feed the synthetic image + parametric text into Qwen2.5-VL-7B to generate descriptions, which are then manually verified.
    - **Design Motivation**: Leverage 12 A100 GPUs for a total of 368 GPU-hours to complete description generation for over 240k samples. The seven-view composite image guarantees both geometric completeness and inference efficiency.

### Loss & Training

Baseline experiments utilize standard language model fine-tuning strategies to fine-tune Qwen2.5-7B (text-only) and Qwen2.5-7B-VL (image + text) on a subset of 50k samples, evaluating with metrics such as Exact Match Score, BLEU Score, and Command-Level F1.

## Key Experimental Results

### Main Results

| Metric | Qwen2.5-7B (Text-Only) | Qwen2.5-7B-VL (Img+Text) | Gain |
|------|---------------------|----------------------|------|
| Exact Match Score | 0.0058 | 0.0099 | +70.7% |
| BLEU Score | 97.18 | 97.93 | +0.77% |
| Command-Level F1 | 0.3247 | 0.3670 | +13.0% |
| Partial Match Rate | 0.5554 | 0.6162 | +10.9% |
| Tolerance Accuracy | 0.5016 | 0.4630 | -7.7% |

### Dataset Statistics

| Complexity Level | Number of Features | Number of Samples | Ratio |
|-----------|---------|--------|------|
| Level 1 (Simple) | 1-5 | 93,188 | ~38.4% |
| Level 2 (Moderate) | 6-10 | 78,926 | ~32.5% |
| Level 3 (Advanced) | 11-100 | 69,259 | ~28.5% |
| Level 4 (Expert) | 100+ | 1,234 | ~0.5% |

### Key Findings
- Multimodal input (image + text) comprehensively outperforms text-only input on three structural alignment metrics: Exact Match, F1, and Partial Match, validating the importance of visual information for CAD semantic understanding.
- The text-only model performs slightly better in Tolerance Accuracy, potentially due to overfitting to numerical parameters rather than structural semantics.
- 2D Sketch is the most frequently used feature type, with Chamfer and Fillet also being common, reflecting the industrial orientation of the dataset.

## Highlights & Insights
- The **closed-loop encoder-decoder design** is highly elegant, supporting lossless round-trips of model $\to$ text $\to$ model, which can be applied to verify the correctness of the generated outputs automatically, as well as facilitate data augmentation and scaling.
- The **seven-view into a single composite image** strategy reduces the input token length while maintaining complete geometric information, making it suitable for multimodal model inference.
- The parametric representation supporting 13 CAD operations is a major upgrade over DeepCAD (which supports only 2), enabling the dataset to cover the complexity of real-world industrial parts.

## Limitations & Future Work
- The paper is published at ICRA instead of typical top CAD/Vision conferences, and the evaluation of the dataset's actual CAD generation performance remains preliminary (only conducting a simple baseline comparison).
- The encoder/decoder relies on the API of commercial SolidWorks software, making it difficult to reproduce and extend to open-source CAD platforms.
- The subset used in the baseline experiments is only 50k samples (about 20% of the total), failing to demonstrate full-scale training performance.
- Although the natural language descriptions have been human-verified, the coverage rate and quality of manual validation are questionable given the scale of over 240k samples.

## Related Work & Insights
- **vs DeepCAD**: DeepCAD pioneered the paradigm of treating CAD modeling as a sequence generation problem but supports only 2 operations (sketch + extrude) and lacks natural language input. SldprtNet comprehensively upgrades both the operation types (13 types) and modal richness.
- **vs Text2CAD**: Although Text2CAD introduces text descriptions, they are synthesized from modeling sequences rather than visual information, which is prone to semantic misalignment with the actual geometry. SldprtNet utilizes a multimodal LLM to generate descriptions from both images and parameters simultaneously, yielding higher alignment quality.
- **vs ABC/Fusion 360**: ABC is large in scale (over 1M B-Rep) but lacks sequences and text annotations; Fusion 360 contains modeling history but is small-scale with no text annotations. SldprtNet is the most comprehensive regarding modality completeness.

## Rating
- Novelty: ⭐⭐⭐ Primarily a dataset construction work, with limited methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐ Only simple baseline comparisons are provided, lacking validation on additional downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, and the data processing workflow is detailed.
- Value: ⭐⭐⭐⭐ Fills the gap in multimodal CAD datasets, providing crucial support for Text-to-CAD research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SketchAgent: Language-Driven Sequential Sketch Generation](sketchagent_language-driven_sequential_sketch_generation.md)
- [\[CVPR 2026\] CADFS: A Big CAD Program Dataset and Framework for Computer-Aided Design with Large Language Models](../../CVPR2026/multimodal_vlm/cadfs_a_big_cad_program_dataset_and_framework_for_computer-aided_design_with_lar.md)
- [\[CVPR 2025\] CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation](comm_a_coherent_interleaved_image-text_dataset_for_multimodal_understanding_and_.md)
- [\[CVPR 2025\] Active Data Curation Effectively Distills Large-Scale Multimodal Models](active_data_curation_effectively_distills_large-scale_multimodal_models.md)
- [\[CVPR 2025\] Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents](scalable_video-to-dataset_generation_for_cross-platform_mobile_agents.md)

</div>

<!-- RELATED:END -->
