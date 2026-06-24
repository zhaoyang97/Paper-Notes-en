---
title: >-
  [Paper Note] RAG-IGBench: Innovative Evaluation for RAG-based Interleaved Generation in Open-domain Question Answering
description: >-
  [NeurIPS 2025][Image Generation][interleaved image-text generation] This paper introduces RAG-IGBench, a benchmark specifically designed to evaluate the quality of interleaved image-text content generated via retrieval-augmented generation. It proposes novel automatic evaluation metrics spanning three dimensions—text quality, image quality, and image-text consistency—and demonstrates strong correlation with human evaluation.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "interleaved image-text generation"
  - "retrieval-augmented generation"
  - "multimodal evaluation"
  - "open-domain question answering"
  - "benchmark"
date: 2026-05-08
content_hash: 25eb847815b679c1
---

# RAG-IGBench: Innovative Evaluation for RAG-based Interleaved Generation in Open-domain Question Answering

**Conference**: NeurIPS 2025
**arXiv**: [2512.05119](https://arxiv.org/abs/2512.05119)  
**Code**: [https://github.com/USTC-StarTeam/RAG-IGBench](https://github.com/USTC-StarTeam/RAG-IGBench)  
**Area**: Information Retrieval
**Keywords**: interleaved image-text generation, retrieval-augmented generation, multimodal evaluation, open-domain question answering, benchmark

## TL;DR
This paper introduces RAG-IGBench, a benchmark specifically designed to evaluate the quality of interleaved image-text content generated via retrieval-augmented generation. It proposes novel automatic evaluation metrics spanning three dimensions—text quality, image quality, and image-text consistency—and demonstrates strong correlation with human evaluation.

## Background & Motivation

**Background**: Interleaved image-text generation requires models to jointly produce text and images, which is a core requirement for practical applications such as content creation and visual storytelling.

**Limitations of Prior Work**:
- End-to-end generation methods (e.g., Chameleon) unify text and image processing but exhibit limited capacity for following complex instructions.
- Existing evaluation frameworks either assess only unimodal metrics (e.g., FID for images alone) or rely on MLLM-based scoring (GPT-4o-based), which introduces model bias and instability.
- High-quality open-domain datasets for interleaved image-text generation are lacking.

**Key Challenge**: How to comprehensively evaluate the quality of interleaved image-text content without relying on model-induced bias.

**Key Insight**: Adopting a RAG framework in which an MLLM selects images from retrieved documents and embeds them into text, rather than generating images from scratch.

## Method

### Overall Architecture
The RAG-IG framework: given a user query → retrieve relevant documents and images → MLLM generates a Markdown-formatted response with image placeholders → placeholders are replaced with actual images to produce the final multimodal response. Evaluation is conducted across three dimensions: text quality, image quality, and image-text consistency.

### Key Designs

1. **Dataset Construction (Three-Stage Pipeline)**:

    - Function: Construct a high-quality interleaved image-text QA dataset.
    - Mechanism: Stage 1 uses an MLLM to generate raw QA pairs; Stage 2 employs expert annotators to refine image selection and arrangement; Stage 3 filters out low-quality samples based on quality criteria.
    - Design Motivation: Sourcing recent public content from social platforms ensures diversity and timeliness; manual annotation guarantees ground truth quality.

2. **Image Quality Evaluation (Edit Distance + Kendall Score)**:

    - Function: Evaluate the match between the model's selected image sequence and the ground truth.
    - Mechanism: Edit Distance measures selection accuracy (the number of insertions/deletions/substitutions required), normalized as $1 - dp(m,n)/\max(m,n)$; Kendall Score measures ordering correctness by computing the proportion of concordant pairs among correctly selected images.
    - Design Motivation: Traditional FID/IS metrics assess generated image quality, but in the RAG setting images are selected rather than generated, necessitating metrics for selection accuracy and ordering correctness.

3. **Image-Text Consistency Evaluation (CLIP Score + Alignment Score)**:

    - Function: Evaluate the semantic alignment between images and their surrounding text.
    - Mechanism: CLIP Score directly computes the cosine similarity between image and text embeddings; Alignment Score compares the contextual text surrounding the same image in the generated answer and the ground truth.
    - Design Motivation: CLIP Score captures direct semantic alignment but lacks contextual understanding; Alignment Score compensates for this limitation.

## Key Experimental Results

### Main Results — Performance of Mainstream MLLMs on RAG-IGBench

| Model | Rouge-1↑ | Edit Dist↑ | Kendall↑ | Align Score↑ | Mean↑ |
|------|---------|-----------|---------|-------------|-------|
| GPT-4o | 0.374 | 0.471 | 0.532 | 0.495 | 0.468 |
| Claude-3.5 | 0.350 | 0.439 | 0.490 | 0.481 | 0.440 |
| Qwen2VL-72B | 0.319 | 0.390 | 0.451 | 0.438 | 0.400 |
| InternVL2-40B | 0.281 | 0.328 | 0.368 | 0.402 | 0.345 |

### Ablation Study — Correlation Between Evaluation Metrics and Human Judgments

| Metric | Pearson Correlation | Spearman Correlation |
|------|-----------------|------------------|
| Rouge-1 | 0.72 | 0.68 |
| Edit Distance | 0.81 | 0.78 |
| Kendall Score | 0.75 | 0.71 |
| CLIP Score | 0.65 | 0.62 |
| Alignment Score | 0.74 | 0.70 |

### Key Findings
- GPT-4o leads across all dimensions, yet a significant gap relative to human performance remains.
- Image selection (Edit Distance) constitutes the largest bottleneck; models generally underperform in terms of both image quantity and selection accuracy.
- Fine-tuned models exhibit performance improvements across multiple benchmarks, validating the high quality of the dataset.
- The gap between open-source and closed-source models is substantial, particularly in image-text consistency.

## Highlights & Insights
- The **RAG-IG paradigm** is more practical than end-to-end image generation—image selection is more controllable and yields more stable quality than image generation.
- The **combination of Edit Distance and Kendall Score** elegantly disentangles the two dimensions of "which images were correctly selected" and "whether the ordering of images is correct."
- The design rationale behind **Alignment Score** is noteworthy: the same image should appear in semantically similar contexts across different answers.

## Limitations & Future Work
- The data sources are limited to social platforms, biasing the domain toward everyday lifestyle topics and underrepresenting professional or academic scenarios.
- Although the evaluation metrics exhibit strong correlation with human judgments, the CLIP Score correlation remains relatively low (0.65), indicating room for further improvement.
- The benchmark evaluates image selection but not image comprehension—a model may select the correct image without being able to explain its content.
- The impact of layout and typesetting on user experience is not considered.

## Related Work & Insights
- **vs. INTERLEAVEDBENCH**: Relies on GPT-4o-based scoring, introducing model bias; RAG-IGBench employs rule-based metrics, yielding more objective evaluation.
- **vs. MMIE**: Depends on fine-tuned VLMs for evaluation, leading to inconsistency issues; the proposed metrics are more stable and reproducible.
- **vs. MEGA-Bench**: Focuses on multimodal understanding capabilities and does not address interleaved generation.

## Rating
- Novelty: ⭐⭐⭐⭐ RAG-IG paradigm + innovative evaluation metrics
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers mainstream open-source and closed-source models with human evaluation validation
- Writing Quality: ⭐⭐⭐⭐ Dataset construction pipeline is detailed and metric definitions are clear
- Value: ⭐⭐⭐⭐ Fills a gap in the evaluation of interleaved image-text generation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] AvatarArtist: Open-Domain 4D Avatarization](../../CVPR2025/image_generation/avatarartist_open-domain_4d_avatarization.md)
- [\[ECCV 2024\] Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality](../../ECCV2024/image_generation/learning_trimodal_relation_for_audio-visual_question_answering_with_missing_moda.md)
- [\[NeurIPS 2025\] OVERT: A Benchmark for Over-Refusal Evaluation on Text-to-Image Models](overt_a_benchmark_for_over-refusal_evaluation_on_text-to-image_models.md)
- [\[NeurIPS 2025\] Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)
- [\[NeurIPS 2025\] RLZero: Direct Policy Inference from Language Without In-Domain Supervision](rlzero_direct_policy_inference_from_language_without_in-domain_supervision.md)

</div>

<!-- RELATED:END -->
