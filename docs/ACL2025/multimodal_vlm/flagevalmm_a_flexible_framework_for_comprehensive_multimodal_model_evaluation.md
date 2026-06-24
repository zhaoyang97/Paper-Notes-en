---
title: >-
  [Paper Note] FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation
description: >-
  [ACL 2025][Multimodal VLM][Multimodal Evaluation] This paper proposes FlagEvalMM, an open-source multimodal model evaluation framework. By leveraging an architectural design that decouples model inference from the evaluation process, it uniformly supports the evaluation of various multimodal tasks, including vision-language understanding (VQA), text-to-image/video generation, and image-text retrieval.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multimodal Evaluation"
  - "Vision-Language Models"
  - "Text-to-Image Evaluation"
  - "Evaluation Framework"
  - "Decoupled Architecture"
date: 2026-05-08
content_hash: eee49c18ed4c41a4
---

# FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation

**Conference**: ACL 2025  
**arXiv**: [2506.09081](https://arxiv.org/abs/2506.09081)  
**Code**: [https://github.com/flageval-baai/FlagEvalMM](https://github.com/flageval-baai/FlagEvalMM)  
**Area**: Multimodal / VLM Evaluation  
**Keywords**: Multimodal Evaluation, Vision-Language Models, Text-to-Image Evaluation, Evaluation Framework, Decoupled Architecture

## TL;DR

This paper proposes FlagEvalMM, an open-source multimodal model evaluation framework. By leveraging an architectural design that decouples model inference from the evaluation process, it uniformly supports the evaluation of various multimodal tasks, including vision-language understanding (VQA), text-to-image/video generation, and image-text retrieval.

## Background & Motivation

With the rapid advancement of multimodal models, there is a pressing need for a unified framework capable of comprehensively, efficiently, and conveniently evaluating diverse multimodal capabilities. However, existing solutions exhibit significant limitations:

**Incomplete Task Coverage**: VLMEvalKit and Lmms-Eval primarily target VLM understanding tasks, while VBench focuses on video generation evaluation. Currently, no single framework covers both understanding and generation tasks simultaneously.

**Coupling of Inference and Evaluation**: Existing frameworks execute model inference and evaluation within the same environment, leading to environment conflicts (e.g., dependency mismatches between model inference and LLM-as-the-Judge evaluation) and low resource utilization efficiency.

**Poor Extensibility**: VLMEvalKit requires intrusive code modifications to add new benchmarks. VHELM is based on HELM, which has a complex architecture and relies heavily on API calls. Lmms-Eval only supports Transformers and the vLLM inference framework.

FlagEvalMM is motivated to address these issues through a decoupled architecture and modular design, providing a one-stop multimodal evaluation experience.

## Method

### Overall Architecture

FlagEvalMM consists of two main components: the **Evaluation Server** and the **Model Runner**, which communicate via a lightweight HTTP RESTful protocol. This decoupled design completely separates the inference environment from the evaluation environment.

### Key Designs

1. **Evaluation Server**: Each evaluation task is the minimal execution unit, comprising three core components: the Processor (data preprocessing that standardizes datasets from different sources into a uniform format), the Config (configuration parameters such as evaluation metrics and prompt templates), and the Evaluator (evaluating model outputs and generating performance metrics). The design is highly extensible, allowing users to register custom Datasets and Evaluators.

2. **Model Runner**: This component includes the Model Adapter and the Backend. The Model Adapter serves as a bridge between the evaluation server and inference engines, with built-in adapters for the OpenAI REST API, Gemini, Anthropic, and others. The Backend is the actual inference engine, with official support for high-performance backends such as vLLM, SGLang, and LMDeploy. It implements an SQLite-based caching mechanism that computes hash values of input data (text, images, parameters) as unique keys to prevent redundant inference.

3. **Communication Protocol**: A six-step lifecycle is established: get_tasks (retrieving available tasks) → task_info (querying task information) → get_meta (obtaining metadata) → get_data(i) (fetching evaluation items) → inference → submit(result) (submitting results). Each step supports distributed and parallelized evaluation.

4. **Evaluation Acceleration**: Utilizes inference acceleration tools like vLLM and SGLang, coupled with asynchronous data loading (e.g., data prefetching) to significantly reduce waiting times.

### Evaluation Capacity Coverage

- **Multimodal Understanding**: Covers five capability dimensions: general knowledge, mathematics, chart understanding, visual perception, and text recognition (OCR). The testbed includes public datasets like MMMU, MMMU-Pro, MathVision, MathVerse, Blink, and CharXiv, supplemented by self-constructed subjective evaluation suites and OCR evaluation datasets. Both Chinese and English are supported.
- **Multimodal Generation**: Evaluates four dimensions: prompt consistency, realism, aesthetic quality, and safety. Automatic metrics include VQAScore, Q-Align, and VideoScore, combined with human evaluation (averaging scores from three annotators).

## Key Experimental Results

### VLM Understanding Evaluation

| Model | Average Rank | General Knowledge | Math | Charts | Visual Perception | Text Recognition |
|------|----------|---------|------|------|----------|---------|
| Gemini-2.0-pro | 2.1 | 64.00 | 52.18 | 67.06 | 62.73 | 78.22 |
| Qwen2.5-VL-72B | 4.6 | 61.30 | 35.45 | 67.00 | 60.90 | 77.63 |
| Claude-3.7-Sonnet | 6.9 | 58.98 | 49.31 | 71.19 | 66.55 | 67.69 |
| GPT-4o-2024-11 | 8.1 | 58.39 | 30.82 | 65.50 | 62.02 | 70.31 |
| InternVL2.5-78B | 6.9 | 61.31 | 37.80 | 60.14 | 62.97 | 70.87 |

### Text-to-Image Evaluation

| Model | Weighted Score | Consistency | Realism | Aesthetics | Safety | VQAScore |
|------|--------|--------|--------|------|--------|----------|
| Hunyuan-Image | 73.00 | 67.93 | 66.67 | 78.50 | 100.0 | 73.76 |
| DALL-E 3 | 70.12 | 70.24 | 57.51 | 68.38 | 98.21 | 81.82 |
| FLUX.1 schnell | 68.39 | 61.95 | 64.34 | 73.18 | 99.11 | 77.95 |
| Midjourney v6.1 | 65.91 | 67.56 | 46.95 | 64.58 | 98.21 | 77.63 |

### Ablation & Analysis

| Analysis Dimension | Key Findings |
|---------|---------|
| Open-source vs. Closed-source VLMs | The Qwen2.5 series outperforms several early closed-source models, narrowing the gap. |
| Cross-lingual Performance | Mistral-3.1 and Claude-3.7 perform significantly worse in Chinese evaluation compared to English. |
| Automatic vs. Human Evaluation | In the consistency dimension, the Pearson correlation between VQAScore and human evaluation is only 0.76. |
| Aesthetic Evaluation | The correlation between OneAlign-Aesthetic and human evaluation is only 0.59. |

### Key Findings

- **Significant Progress in Open-Source Models**: Qwen2.5-VL-72B outperforms GPT-4o and Claude-3.5-Sonnet across multiple capabilities.
- **Cross-Lingual Generalization Remains a Challenge**: Some models perform far better in English than in Chinese, indicating a lack of cultural adaptability.
- **VLMs remain unstable on classic computer vision (CV) tasks**, such as spatial reasoning, counting, and occlusion.
- **Closed-source T2I models generally outperform open-source counterparts**, though a clear gap remains between automatic evaluation metrics and human judgment.
- **The decoupled architecture** effectively resolves environment conflicts and supports flexible resource allocation.

## Highlights & Insights

- The architectural design decoupling inference from evaluation serves as the core innovation, addressing pain points in actual engineering workflows of model evaluation.
- Concurrently supporting both understanding and generation tasks fills a critical gap in existing evaluation frameworks.
- Provides a comprehensive performance comparison of state-of-the-art VLM and T2I models, offering highly valuable reference data.
- The SQLite-based caching mechanism effectively avoids redundant inference, demonstrating strong practical utility.
- Integrated into the FlagEval platform and HuggingFace Spaces for immediate and out-of-the-box usage.

## Limitations & Future Work

- The coverage of evaluation methodologies remains limited, with multi-turn dialogues, interactive games, and advanced reasoning capabilities yet to be integrated.
- A substantial gap exists between automatic and human evaluations in generation tasks, still necessitating reliance on human annotation.
- Specific construction details of the self-built datasets are not fully disclosed, which limits reproducibility.
- The text-to-video evaluation is small in scale, involving only 148 prompts.
- Quantitative analysis regarding the efficiency of the evaluation framework itself (e.g., communication overhead) is not provided.

## Related Work & Insights

- Compared to Lmms-Eval, the primary advantage of FlagEvalMM lies in its support for generation task evaluation and more flexible backend options.
- Compared to VLMEvalKit, its plug-and-play design avoids intrusive code modifications.
- The revealed performance gaps between automatic metrics and human evaluations highlight the necessity of designing better automatic evaluation metrics.
- The design paradigm of a decoupled architecture can be generalized to other large language model evaluation scenarios.

## Rating

- **Novelty**: ⭐⭐⭐ — The decoupled architecture offers engineering innovation, but the core focus is system design rather than methodological novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — A large-scale evaluation covering over 50 VLMs and 30+ generative models.
- **Writing Quality**: ⭐⭐⭐ — The architectural description is clear, but certain experimental details (e.g., self-built datasets, exact pipeline) lack depth.
- **Value**: ⭐⭐⭐⭐ — Highly practical as an open-source evaluation tool, with benchmarking data offering valuable reference points.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] UPME: An Unsupervised Peer Review Framework for Multimodal Large Language Model Evaluation](../../CVPR2025/multimodal_vlm/upme_an_unsupervised_peer_review_framework_for_multimodal_large_language_model_e.md)
- [\[CVPR 2026\] SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence](../../CVPR2026/multimodal_vlm/spatialscore_towards_comprehensive_evaluation_for_spatial_intelligence.md)
- [\[CVPR 2026\] 4DWorldBench: A Comprehensive Evaluation Framework for 3D/4D World Generation Models](../../CVPR2026/multimodal_vlm/4dworldbench_a_comprehensive_evaluation_framework_for_3d4d_world_generation_mode.md)
- [\[ACL 2025\] MMSciBench: Benchmarking Language Models on Chinese Multimodal Scientific Problems](mmscibench_benchmarking_language_models_on_chinese_multimodal_scientific_problem.md)
- [\[CVPR 2025\] Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment](../../CVPR2025/multimodal_vlm/harnessing_frozen_unimodal_encoders_for_flexible_multimodal_alignment.md)

</div>

<!-- RELATED:END -->
