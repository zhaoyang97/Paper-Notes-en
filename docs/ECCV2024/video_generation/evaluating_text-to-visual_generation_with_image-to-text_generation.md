---
title: >-
  [Paper Note] Evaluating Text-to-Visual Generation with Image-to-Text Generation
description: >-
  [ECCV 2024][Video Generation][Text-to-Visual Generation Evaluation] The authors propose VQAScore, which uses Visual Question Answering (VQA) models instead of CLIP to evaluate text-to-visual generation quality. It significantly outperforms CLIPScore on complex compositional prompts and releases the GenAI-Bench benchmark.
tags:
  - "ECCV 2024"
  - "Video Generation"
  - "Text-to-Visual Generation Evaluation"
  - "VQAScore"
  - "Image-Text Alignment"
  - "Compositional Prompting"
  - "GenAI-Bench"
date: 2026-05-08
content_hash: 197cb182ab52a7e2
---

# Evaluating Text-to-Visual Generation with Image-to-Text Generation

**Conference**: ECCV 2024  
**arXiv**: [2404.01291](https://arxiv.org/abs/2404.01291)  
**Code**: Yes (Open-source data, models, and code)  
**Area**: Video Generation  
**Keywords**: Text-to-Visual Generation Evaluation, VQAScore, Image-Text Alignment, Compositional Prompting, GenAI-Bench

## TL;DR

The authors propose VQAScore, which uses Visual Question Answering (VQA) models instead of CLIP to evaluate text-to-visual generation quality. It significantly outperforms CLIPScore on complex compositional prompts and releases the GenAI-Bench benchmark.

## Background & Motivation

Although text-to-image/video generation models (e.g., Stable Diffusion, DALL-E 3) have progressed rapidly, reliably evaluating the generation quality remains an unresolved key problem. The most widely used evaluation metric, CLIPScore, has a fundamental flaw: CLIP's text encoder is essentially a "bag of words" that cannot distinguish prompts with different semantic structures but identical vocabularies. For instance, "a horse eating grass" and "grass eating a horse" would yield similar CLIPScores, which is clearly unreasonable.

The core problems are: (1) CLIPScore evaluates inaccurately for complex prompts involving compositional relations (such as spatial relations, attribute binding, action relations, etc.); (2) Existing improvement schemes (e.g., using larger CLIP models or introducing extra parsers) either offer limited gains or are overly complex; (3) There is a lack of high-quality evaluation benchmarks tailored for compositional generation.

This paper proposes a counter-intuitive yet highly effective solution: using image-to-text VQA models to evaluate text-to-image generation quality. The Core Idea is to reformulate the evaluation problem into a simple Visual Question Answering task—"Does this figure show '{text}'?"—and calculate the probability of the VQA model answering "Yes" as the alignment score.

## Method

### Overall Architecture

The pipeline of the VQAScore evaluation framework is highly straightforward: (1) Given a generated image and a text prompt; (2) Embed the text prompt into a template question "Does this figure show '{text}'?"; (3) Use a VQA model to compute the probability of answering "Yes"; (4) This probability serves as the VQAScore alignment score.

### Key Designs

1. **VQAScore Evaluation Metric**:
    - Function: Accurately measure the semantic alignment between generated images and text prompts.
    - Mechanism: Utilize the vision-language reasoning capability of VQA models to reformulate alignment evaluation as a binary question-answering task. The VQA model determines semantic consistency by jointly processing images and text, which avoids compositional understanding deficiencies caused by the independent encoding of images and text in CLIP.
    - Design Motivation: VQA models naturally possess compositional reasoning capabilities (e.g., understanding "who is doing what to whom"), which is precisely what CLIPScore lacks.

2. **CLIP-FlanT5 Model**:
    - Function: Further improve the performance of VQAScore.
    - Mechanism: Train a bidirectional image-question encoder. Unlike standard VQA models, CLIP-FlanT5 allows image embeddings to depend on question content (and vice versa), enabling deeper cross-modal interaction. It uses FlanT5 as the language model backbone combined with a CLIP vision encoder.
    - Design Motivation: Standard unidirectional encoding ignores the guiding effect of the question content on image understanding. A bidirectional encoder can capture finer-grained image-text interactions.

3. **GenAI-Bench Benchmark**:
    - Function: Provide a more challenging benchmark for evaluating compositional text-to-visual generation.
    - Mechanism: It contains 1,600 compositional text prompts across dimensions such as scene parsing, object recognition, attribute binding, relation reasoning, and high-level logical reasoning. It collects over 15,000 human ratings covering mainstream generative models like Stable Diffusion, DALL-E 3, and Gen2.
    - Design Motivation: Text prompts in existing evaluation benchmarks are too simple to adequately test the compositional understanding capabilities of generative models.

### Loss & Training

CLIP-FlanT5 is trained on large-scale image-text pairs using standard VQA training objectives. Key strategies include: employing a bidirectional attention mechanism instead of unidirectional attention, and training solely on image data while demonstrating generalization to video and 3D model evaluations.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (VQAScore) | CLIPScore | Gain |
|--------|------|------|----------|------|
| 8 Image-Text Alignment Benchmarks | Kendall τ | SOTA | Second-best | Average +15-25% |
| Winoground | Accuracy | Significantly leading | ~50% (Random) | +20-30% |
| GenAI-Bench | Human Correlation | Best | Low | Significant improvement |
| Video Alignment | Kendall τ | Applicable | Not applicable | Cross-modal generalization |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Different VQA Models | Performance Variance | Larger VQA models perform better |
| Different Question Templates | Robust | VQAScore is robust to the choice of templates |
| CLIP-FlanT5 vs GPT-4V | Leading or on par | Open-source models outperform proprietary models |
| Image vs Video vs 3D | All effective | Training solely on images generalizes to other modalities |

### Key Findings

- VQAScore achieves SOTA on all 8 image-text alignment benchmarks, despite its extreme simplicity.
- The open-source CLIP-FlanT5 even outperforms baseline methods utilizing GPT-4V.
- VQAScore generalizes to video and 3D model evaluation, demonstrating strong cross-modal capabilities.
- GenAI-Bench reveals significant deficiencies in the compositional understanding of current generative models.

## Highlights & Insights

- The core idea is exceptionally simple and efficient—a direct VQA question outperforms complex evaluation methods by a wide margin.
- It reveals an important methodological insight: text-to-image generation can conversely be evaluated using image-to-text models.
- As an open-source alternative, CLIP-FlanT5 outperforms GPT-4V, lowering evaluation costs.
- The paper has accumulated 411 citations, indicating the broad influence of this work.

## Limitations & Future Work

- VQAScore still depends on the quality of VQA models and may fail in scenarios that are challenging for VQA models.
- Question templates like "Does this figure show..." may lack flexibility for certain types of prompts.
- GenAI-Bench primarily focuses on English prompts, without evaluating multilingual scenarios.
- The capability of VQAScore in fine-grained aesthetic quality assessment (such as composition and color) has not been explored.

## Related Work & Insights

- **CLIPScore**: The classic evaluation method by Hessel et al., widely used but suffering from compositional flaws.
- **TIFA**: An evaluation approach that uses an LLM to generate questions and then assesses them via VQA; it is far more complex than VQAScore.
- **DSG**: It evaluates via dependency parsing and scene graphs, requiring an extra parsing step.
- Insight: Sometimes the simplest methods are the most effective; evaluation and generation can establish connections through inverse models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The idea of using VQA to evaluate T2I generation is extremely simple yet highly effective, having a massive impact.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering 8 benchmarks, multi-modal generalization, human evaluation, and a new benchmark.
- Writing Quality: ⭐⭐⭐⭐ Highly logical with compelling motivation.
- Value: ⭐⭐⭐⭐⭐ Over 411 citations demonstrate its broad practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](../../ICCV2025/video_generation/stiv_scalable_text_and_image_conditioned_video_generation.md)
- [\[ECCV 2024\] MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model](mofa-video_controllable_image_animation_via_generative_motion_field_adaptions_in.md)
- [\[ICCV 2025\] TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation](../../ICCV2025/video_generation/tip-i2v_a_million-scale_real_text_and_image_prompt_dataset_for_image-to-video_ge.md)
- [\[ECCV 2024\] Exploring Pre-trained Text-to-Video Diffusion Models for Referring Video Object Segmentation](exploring_pre-trained_text-to-video_diffusion_models_for_referring_video_object_.md)
- [\[CVPR 2026\] Reasoning Diffusion for Unpaired Test Time Out-of-distribution Text-Image to Video Generation](../../CVPR2026/video_generation/reasoning_diffusion_for_unpaired_test_time_out-of-distribution_text-image_to_vid.md)

</div>

<!-- RELATED:END -->
