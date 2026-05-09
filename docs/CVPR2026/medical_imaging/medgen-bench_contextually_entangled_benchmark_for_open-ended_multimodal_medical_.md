---
title: >-
  [Paper Note] MedGEN-Bench: Contextually Entangled Benchmark for Open-Ended Multimodal Medical Generation
description: >-
  [CVPR 2026][Medical Imaging][Multimodal medical generation] This paper introduces MedGEN-Bench, the first comprehensive benchmark for open-ended multimodal medical generation, comprising 6,422 expert-verified image-text pairs spanning 6 imaging modalities and 16 clinical tasks, accompanied by a three-tier evaluation framework. The benchmark reveals that compositional pipelines outperform unified models in cross-modal consistency.
tags:
  - CVPR 2026
  - Medical Imaging
  - Multimodal medical generation
  - benchmark
  - VLM evaluation
  - image-text entanglement
  - open-ended generation
date: 2026-05-08
content_hash: 96321d1323618d7a
---

# MedGEN-Bench: Contextually Entangled Benchmark for Open-Ended Multimodal Medical Generation

**Conference**: CVPR 2026
**arXiv**: [2511.13135](https://arxiv.org/abs/2511.13135)
**Code**: To be released (open-source promised in paper)
**Area**: Medical Imaging
**Keywords**: Multimodal medical generation, benchmark, VLM evaluation, image-text entanglement, open-ended generation

## TL;DR

This paper introduces MedGEN-Bench, the first comprehensive benchmark for open-ended multimodal medical generation, comprising 6,422 expert-verified image-text pairs spanning 6 imaging modalities and 16 clinical tasks, accompanied by a three-tier evaluation framework. The benchmark reveals that compositional pipelines outperform unified models in cross-modal consistency.

## Background & Motivation

Existing medical vision benchmarks (VQA-RAD, SLAKE, PMC-VQA, etc.) suffer from three fundamental limitations: (1) **Query-image decoupling**—questions are generic templates lacking deep association with image content, reducing VQA to simple classification; (2) **Closed-form shortcuts**—multiple-choice formats allow models to rank answers without engaging in complex clinical reasoning; (3) **Text-only outputs**—the indispensable image generation capabilities in clinical practice (e.g., lesion localization, region editing) are entirely overlooked. These issues are severely misaligned with real-world clinical workflows. This paper aims to construct a comprehensive benchmark that jointly evaluates text-based diagnostic generation and clinically relevant image synthesis.

## Method

### Overall Architecture

MedGEN-Bench is constructed through a four-stage pipeline: (1) **Preprocessing**—a two-stage filtering process (coarse metadata filtering + GPT-4o semantic verification) selects task-relevant medical images; (2) **Image pair synthesis**—rule-based transformations (classical image processing) and generative transformations (diffusion models, etc.) produce input-output image pairs; (3) **Text pair synthesis**—Qwen3-VL extracts semantic information, followed by GPT-4o contextual augmentation to generate instruction-answer pairs; (4) **Post-processing**—automated VLM review combined with expert manual verification.

The final benchmark contains 6,422 expert-verified image-text pairs (11,744 high-quality images), covering six modalities—CT, MRI, ultrasound, X-ray, pathology, and clinical photography—organized into three task formats: VQA, image editing, and contextual multimodal generation.

### Key Designs

1. **Cross-Modal Entanglement**: Instructions are deliberately designed to include detailed, image-specific visual cues, compelling models to ground textual semantics in pixel-level evidence. This stands in sharp contrast to conventional generic template queries, demanding deep cross-modal reasoning rather than shallow pattern matching.

2. **Contextual Augmentation**: Qwen3-VL first extracts structured semantics $\boldsymbol{\mathcal{M}}$ from image pairs and populates task templates to produce raw instruction pairs $\boldsymbol{\mathcal{I}}_{\text{raw}}$. GPT-4o then applies a refinement function $\boldsymbol{\psi}$, incorporating input/output images, metadata, and raw instructions to generate the final instruction-answer pairs. Refinement includes synonym substitution, syntactic restructuring, and domain terminology injection, yielding linguistic diversity while preserving semantic accuracy. Ablation experiments show this augmentation improves the average text-image semantic similarity by 36.3%.

3. **Three-Tier Evaluation Framework**: (a) **Pixel tier**—SSIM, PSNR, and LPIPS assess structural and perceptual similarity; (b) **Text tier**—PubMedBERT-based BERTScore evaluates semantic similarity; (c) **Holistic tier**—a VLM-as-a-Judge paradigm (Analyze-then-Judge, scored 1–10) evaluates five dimensions: consistency, visual-text alignment, content accuracy, relevance, and modality consistency, operating in both reference-based and reference-free modes.

### Loss & Training

This is a benchmark paper and does not involve model training. During evaluation, results across metrics are binarized using predefined thresholds, and accuracy (the proportion of passing samples) is reported. The benchmark's quality assurance process includes:
- **Automated review**: GPT-4o assesses consistency between generated samples and ground truth
- **Expert review**: Medical experts evaluate samples across three dimensions—question validity, answer accuracy, and multimodal relevance
- **Image annotation**: Unobtrusive text labels are added to input/output images to assist VLM-based review

## Key Experimental Results

### Main Results

| Task / Model | Holistic w. GT | Holistic w/o GT | Text (BERTScore) | Notes |
|---|---|---|---|---|
| **Multimodal Generation** | | | | |
| Qwen3-VL & Imagen-4.0-fast | 30.11 | **75.32** | 51.14 | Best compositional pipeline |
| Gemini-2.5-flash-image (unified) | 23.58 | 49.78 | 46.86 | High image quality, weak text |
| Ming-UniVision (unified) | 8.54 | 11.48 | 24.93 | Severe cross-modal misalignment |
| **Image Editing** | | | | |
| Qwen3-VL & Gpt-image-1-mini | **72.59** | **87.62** | — | Best on editing tasks |
| Gemini-2.5-flash (unified) | 71.28 | 84.22 | — | Best among unified models |
| **VQA** | | | | |
| Qwen3-VL | **53.10** | **98.27** | 29.83 | General VLM leads overall |
| HuaTuoGPT-Vision (medical-specific) | 36.03 | 75.82 | **53.67** | Strong text but weak holistic |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Raw template instructions | Avg. similarity 0.273 | Baseline |
| Contextually augmented instructions | Avg. similarity 0.372 | +36.3%, Pass Rate 86.9% |
| Peak distribution shift | 0.25 → 0.40 | Significant rightward shift in image-text semantic alignment |

### Key Findings

- **Compositional pipelines > unified models**: Compositional frameworks substantially outperform unified models in cross-modal consistency through task decomposition and modular collaboration.
- **Local metrics mask systemic failures**: Ming-UniVision achieves high PSNR/LPIPS but very low holistic scores, demonstrating that pixel-level quality does not equate to clinical correctness.
- **Limitations of medical-specific models**: HuaTuoGPT-Vision exhibits strong text capability (BERTScore 53.67) but lags behind general-purpose models in holistic evaluation, exposing cross-modal misalignment.
- **Contextual augmentation is critical**: Query-image entanglement directly improves generation quality, validating the core design philosophy of this benchmark.

## Highlights & Insights

- **Paradigm shift**: The paper is the first to extend medical AI evaluation from understanding-centric to a dual focus on understanding and generation, better reflecting real clinical workflows.
- **Three-tier evaluation architecture**: The combination of pixel-level, semantic-level, and holistic assessment reveals true model capabilities more effectively than any single metric.
- **Revealing finding**: The "cross-modal misalignment" phenomenon in unified models—high pixel fidelity but poor semantic consistency—carries important implications for future model design.

## Limitations & Future Work

- Evaluation relies on GPT-4o as a judge, which may introduce its own biases (the circular problem of VLMs evaluating VLMs).
- Image generation data undergoes generative model transformations, potentially introducing unnatural artifacts.
- The scale of 6,422 pairs covering 6 modalities and 16 tasks remains limited, averaging approximately 230 pairs per sub-task.
- 3D volumetric imaging (e.g., full CT/MRI sequences) is not included; the benchmark is restricted to 2D slices.
- Expert verification has limited scalability, making large-scale continuous updates difficult.
- All benchmark data originate from public datasets, potentially introducing distribution shifts relative to real clinical data.
- The benchmark does not evaluate model performance in longitudinal follow-up scenarios (e.g., comparing two successive examinations).

## Related Work & Insights

- CheXGenBench and MedEBench attempt to incorporate generative tasks but are limited to specific modalities (X-ray); this paper is the first to provide full modality coverage.
- DrVD-Bench focuses on reasoning consistency, and SMMILE focuses on few-shot learning—both remain at the level of understanding.
- SMMILE's multimodal ICL and the multimodal generation in this paper represent two distinct development directions.
- Insight: The next frontier for medical multimodal AI lies not only in better image understanding but also in generating clinically meaningful images and reports.
- The finding that compositional pipelines outperform unified models raises questions about the development trajectory of large unified models such as Gemini and GPT.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic benchmark for medical multimodal generation, filling an important gap
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluates 10 compositional + 3 unified + 5 VLM configurations with broad coverage
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated; pilot study is convincing
- Value: ⭐⭐⭐⭐⭐ Serves as a foundational benchmark for medical multimodal generation; evaluation framework is reusable

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis](../../ICLR2026/medical_imaging/survhte-bench_a_benchmark_for_heterogeneous_treatment_effect_estimation_in_survi.md)
- [\[CVPR 2026\] LUMINA: A Multi-Vendor Mammography Benchmark with Energy Harmonization Protocol](lumina_a_multi-vendor_mammography_benchmark_with_energy_harmonization_protocol.md)
- [\[CVPR 2026\] RDFace: A Benchmark Dataset for Rare Disease Facial Image Analysis under Extreme Data Scarcity and Phenotype-Aware Synthetic Generation](rdface_a_benchmark_dataset_for_rare_disease_facial_image_analysis_under_extreme_.md)
- [\[NeurIPS 2025\] SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning](../../NeurIPS2025/medical_imaging/smmile_an_expert-driven_benchmark_for_multimodal_medical_in-context_learning.md)
- [\[AAAI 2026\] CountVid: Open-World Object Counting in Videos](../../AAAI2026/medical_imaging/open-world_object_counting_in_videos.md)

<!-- RELATED:END -->
