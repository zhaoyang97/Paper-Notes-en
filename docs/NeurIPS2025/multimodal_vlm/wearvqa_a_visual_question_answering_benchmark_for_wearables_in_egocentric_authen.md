---
title: >-
  [Paper Note] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios
description: >-
  [NeurIPS 2025][Multimodal VLM][VQA] This paper introduces WearVQA, the first VQA benchmark specifically designed for wearable device (smart glasses) scenarios. It comprises 2,520 egocentric image–question–answer triplets…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "VQA"
  - "wearable devices"
  - "egocentric view"
  - "image quality degradation"
  - "smart glasses"
  - "benchmark"
date: 2026-05-08
content_hash: 4ecd5a8c40f1f2bc
---

# WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios

**Conference**: NeurIPS 2025
**arXiv**: [2511.22154](https://arxiv.org/abs/2511.22154)  
**Code**: [HuggingFace Dataset](https://huggingface.co/datasets/tonyliao-meta/WearVQA)  
**Area**: Multimodal VLM / Wearable Devices
**Keywords**: VQA, wearable devices, egocentric view, image quality degradation, smart glasses, benchmark
**Authors**: Eun Chang, Zhuangqun Huang, Yiwei Liao et al. (Meta Reality Labs)

## TL;DR

This paper introduces WearVQA, the first VQA benchmark specifically designed for wearable device (smart glasses) scenarios. It comprises 2,520 egocentric image–question–answer triplets, systematically covering 7 visual domains, 10 cognitive task types, and 6 categories of wearable-specific image quality degradation. An accompanying LLM-as-a-judge evaluation framework achieves 96% accuracy, and the benchmark reveals that current SOTA multimodal models attain only 24–52% accuracy in this setting.

## Background & Motivation

**Background**: Multimodal large language models (MM-LLMs) have advanced rapidly on VQA tasks, yet existing benchmarks (VQAv2, OK-VQA, MMMU, etc.) are predominantly built on high-quality third-person images, placing evaluation conditions far from the real-world deployment environment of wearable devices. Wearable AI assistants (e.g., Meta Ray-Ban smart glasses) are becoming mainstream, but no dedicated evaluation framework exists to measure model capability in this context.

**Limitations of Prior Work**: Egocentric images captured by wearable devices inherently suffer from multiple quality degradations—occlusion, low illumination, motion blur, unzoomed wide-angle framing, truncation, and rotation—with 54% of images exhibiting at least one such issue. Existing benchmarks use high-quality images and thus fail to reflect these challenges. While EgoVQA and VizWiz address egocentric or low-quality images to some extent, the former focuses on video and the latter covers only narrow question types; neither provides systematic multi-dimensional coverage.

**Key Challenge**: Wearable AI assistants must perform VQA spanning recognition, reasoning, counting, and spatial understanding on low-quality egocentric images, yet no benchmark simultaneously covers egocentric perspective, diverse quality degradations, and complex reasoning demands, leaving model development without targeted guidance.

**Goal**: To construct a comprehensive VQA benchmark for wearable device scenarios that systematically evaluates SOTA multimodal models on egocentric, low-quality images paired with complex cognitive tasks, and to identify critical performance bottlenecks.

**Key Insight**: Drawing on Meta Reality Labs' smart glasses products, the authors collect images with RayBan Meta smart glasses and design the benchmark dataset along three axes—domain × task type × image quality—to ensure statistical significance, accompanied by a reliable automated evaluation framework.

**Core Idea**: By constructing the first VQA benchmark that simultaneously covers egocentric low-quality images, multiple domains, and diverse task types, the work exposes systematic shortcomings of current multimodal models in wearable scenarios.

## Method

### Overall Architecture

WearVQA consists of three components: (1) **Dataset**—2,520 image–question–answer triplets captured with RayBan Meta smart glasses; (2) **Evaluation Framework**—a GPT-4o-based LLM-as-a-judge system that scores responses along five dimensions: factual correctness, relevance, completeness, egocentric perspective, and conciseness; (3) **Multi-dimensional Analysis System**—systematic model performance analysis across domain, task type, image quality, and image resolution. The dataset is randomly split into 1,500 public test samples and 1,000 private test samples.

### Key Designs

1. **Three-Dimensional Systematic Data Construction (Domain × Task × Quality)**

    - **Function**: Ensures comprehensive coverage of wearable VQA challenges, with each slice containing ≥220 samples to guarantee statistical significance (error <5.7% at 90% confidence).
    - **Mechanism**: 7 domains (animals/pets, food/beverages, gardening/plants, hobbies/activities, landmarks/travel, shopping/products, text/documents) × 10 cognitive task types (recognition, activity recognition, how-to/usage, counting, spatial reasoning, reasoning, next-state prediction, math, etc.) × 6 quality degradation categories (blur, truncation, low light, unzoomed, occlusion, rotation); 60% of questions are reasoning-type, 54% of images have quality issues, and 42% involve hand-held/pointing.
    - **Design Motivation**: Unlike existing benchmarks that focus on a single dimension, the three-dimensional cross design precisely locates where models fail—in which scenario, task type, and quality condition—providing actionable directions for model improvement.

2. **Forward-looking Difficulty Calibration (Forward-looking Filtering)**

    - **Function**: Filters out questions that all current SOTA models answer correctly, ensuring the benchmark remains challenging over time.
    - **Mechanism**: Multiple SOTA models are first used to pre-evaluate all candidate questions; samples answered correctly by all models are removed, retaining only discriminative, difficult questions. All retained questions must satisfy four constraints: image-grounded, no external knowledge required, short answer, and unambiguous.
    - **Design Motivation**: Traditional benchmarks rapidly saturate as models improve; forward-looking filtering keeps WearVQA effective as an evaluation tool in a rapidly evolving model landscape.

3. **Five-Dimensional LLM-as-a-Judge Evaluation Framework**

    - **Function**: Provides scalable and reliable automated evaluation supporting open-ended answer judgment.
    - **Mechanism**: GPT-4o serves as the judge, scoring responses across five dimensions—factual correctness (no hallucination), relevance, completeness, egocentric perspective (whether the response aligns with the wearer's viewpoint), and conciseness. Human annotation validation yields 96% accuracy, 98.2% error-identification precision, 95.5% recall, and 96.8% F1.
    - **Design Motivation**: Open-ended answers in wearable VQA cannot be reliably evaluated by simple string matching and require semantic-level judgment. The inclusion of "egocentric perspective" as a unique evaluation dimension ensures responses conform to the natural interaction style of wearable devices.

### Loss & Training

WearVQA is a purely evaluative benchmark and does not involve model training. The evaluation metric is QA Accuracy, decomposable at fine granularity across domain, task type, and image quality dimensions.

## Key Experimental Results

### Main Results

Overall performance of 10 SOTA MM-LLMs on WearVQA:

| Model | Parameters | Type | Overall Acc (%) | High-Quality Images (%) | Low-Quality Images (%) | Quality Drop |
|-------|-----------|------|----------------|------------------------|----------------------|-------------|
| GPT-4o | ~200B | Closed | **51.5** | 58.5 | 45.5 | -13.0 |
| Gemini-1.5-Pro | ~200B | Closed | 45.4 | 50.4 | 41.0 | -9.4 |
| Qwen2.5-VL-72B | 72B | Open | 45.1 | 51.1 | 39.9 | -11.2 |
| Llama-4-Maverick | 17B×128E | Open | 42.4 | 45.9 | 39.3 | -6.6 |
| Llama-4-Scout | 17B×16E | Open | 41.5 | 45.6 | 38.0 | -7.6 |
| Llama-3.2v-90B | 90B | Open | 37.7 | 41.6 | 33.8 | -7.8 |
| Claude-3.7-Sonnet | ~175B | Closed | 33.9 | 42.6 | 26.4 | **-16.2** |
| Pixtral-12B | 12B | Open | 25.8 | 29.5 | 22.7 | -6.8 |
| Molmo-72B | 72B | Open | 25.7 | 29.7 | 22.2 | -7.5 |
| Phi-4-mini | 3.8B | Open | 23.9 | 26.9 | 21.3 | -5.6 |

### Ablation Study

Performance of the top-4 models under different image quality degradation types:

| Quality Issue | GPT-4o (%) | Gemini-1.5 (%) | Llama-4-Mav (%) | Qwen2.5-VL (%) |
|--------------|-----------|--------------|----------------|---------------|
| Low Light | **58.5** | 49.5 | 43.9 | 47.2 |
| Truncation | 50.2 | 43.0 | 46.0 | 44.2 |
| Blur | 46.3 | 39.0 | 39.0 | 41.6 |
| Rotation | 43.1 | 43.4 | 38.8 | 39.4 |
| Occlusion | 41.8 | 35.2 | 36.6 | 34.1 |
| **Unzoomed** | **30.9** | **34.7** | **29.6** | **33.9** |

Performance of the top-4 models across different cognitive task types:

| Task Type | GPT-4o (%) | Gemini-1.5 (%) | Llama-4-Mav (%) | Qwen2.5-VL (%) |
|----------|-----------|--------------|----------------|---------------|
| How-to/Usage | **69.4** | 61.9 | 51.7 | 55.8 |
| Reasoning (Image) | 59.0 | 48.4 | 44.6 | 49.8 |
| Recognition (Image) | 57.2 | 51.4 | 44.8 | 49.5 |
| Reasoning (Text) | 57.7 | 51.4 | 43.6 | 51.4 |
| Activity Recognition | 53.1 | 48.1 | 36.9 | 40.4 |
| Recognition (Text) | 51.3 | 47.4 | 43.5 | 51.7 |
| Math | 30.5 | 28.0 | **40.0** | 34.5 |
| Counting | 38.5 | 35.2 | 37.5 | 33.9 |
| Spatial Reasoning | 41.4 | 32.1 | 38.6 | 36.4 |

### Key Findings

1. **Overall accuracy is extremely low**: The best-performing model, GPT-4o, achieves only 51.5%, indicating that wearable VQA is far from solved and that the benchmark retains long-term value.
2. **"Unzoomed" is the most critical quality bottleneck**: All models perform worst on unzoomed images (GPT-4o: 30.9%), as small subjects—especially small text—are difficult to discern.
3. **Low light has a surprisingly small impact**: Models perform unexpectedly well under low-light conditions (GPT-4o: 58.5%), presumably because concentrated illumination makes the subject more salient while distractors are hidden in shadow.
4. **Counting, math, and spatial reasoning are the hardest task types**: Counting (38.5%), math (30.5%), and spatial reasoning (41.4%) accuracy fall well below that of recognition and how-to tasks.
5. **Claude-3.7 is most sensitive to image quality**: Its performance drops by 16.2%, whereas Llama-4-Maverick and Phi-4 are most robust (5–7% drop).
6. **Resolution effects show no consistent pattern**: High vs. low resolution affects different models in different directions, likely depending on each model's training data resolution distribution.
7. **The best open-source model (Qwen2.5-VL) matches the second-best closed-source model (Gemini-1.5-Pro)**: Both achieve approximately 45%.

## Highlights & Insights

- **Product-driven benchmark design**: Originating from Meta Reality Labs' smart glasses product line, images are captured with actual RayBan Meta devices and questions are grounded in real wearable use cases (navigation, shopping, DIY repair, etc.), making the benchmark more deployment-relevant than academically constructed alternatives.
- **Forward-looking difficulty filtering**: Pre-removing questions answered correctly by all models keeps the benchmark challenging as models rapidly improve—a design philosophy worth adopting in other benchmarks.
- **Wearable-adapted LLM-as-a-judge**: Adding an "egocentric perspective" dimension to standard evaluation criteria requires responses to be phrased from the wearer's viewpoint (e.g., "the word is Strombolli" rather than "the image shows..."), which is crucial for natural wearable interaction.
- **"Unzoomed" as the primary bottleneck**: This finding has direct implications for wearable device design—integrating optical zoom at the hardware level or super-resolution at the software level may be more effective than improving the model alone.
- **Three-dimensional cross-analysis**: Systematic analysis across domain × task × quality precisely identifies model weaknesses and provides a roadmap for targeted optimization.

## Limitations & Future Work

1. **Excludes questions requiring external knowledge**: Real wearable scenarios frequently demand RAG capabilities (e.g., "what food pairs with this wine?"), but such questions are deliberately excluded, leaving MM-RAG evaluation unaddressed.
2. **English only**: Wearable devices serve a global user base requiring multilingual VQA capability; the monolingual benchmark limits representativeness.
3. **Static images rather than video**: Wearable devices naturally produce video streams; static images cannot capture temporal dynamics such as motion and illumination changes.
4. **Limited dataset scale (2,520)**: Compared to VQAv2 (1.4M+), the scale is modest; while forward-looking filtering improves discriminability, certain slices (e.g., Next-state Prediction with only 65 samples) lack sufficient statistical significance.
5. **Evaluation depends on GPT-4o**: The LLM judge is itself among the models being evaluated, introducing potential bias; API call costs are also non-trivial.
6. **No multi-turn dialogue**: Wearable interactions typically involve follow-up questions (e.g., "What does that sign say? → How far away is it?"), and the single-turn setting deviates from real usage patterns.

## Related Work & Insights

- **VQAv2 / OK-VQA**: Classic VQA benchmarks, but images are high-quality and third-person; WearVQA introduces wearable-specific challenges on top of this foundation.
- **VizWiz**: Low-quality image VQA from visually impaired users' phone captures, partially addressing quality degradation but lacking systematic task categorization and reasoning-type questions.
- **MMMU / MMBench / MMVet**: Comprehensive multimodal evaluations focusing on expert reasoning, fine-grained capabilities, and integrated abilities, respectively, but all based on high-quality third-person images.
- **EgoVQA**: Egocentric video VQA focused on action understanding, but with limited question types and applicable only to video.
- **Insights**: The "three-dimensional cross + forward-looking filtering" benchmark design paradigm is generalizable to other vertical domains (e.g., in-vehicle VQA, medical wearables). The finding that "unzoomed" is the core bottleneck suggests that **hardware–software co-optimization** (optical zoom + super-resolution + model fine-tuning) may be more effective than improving the model in isolation.

## Rating

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Novelty | ⭐⭐⭐⭐ | First systematic wearable VQA benchmark; the three-dimensional cross design and forward-looking filtering represent meaningful methodological contributions, though the core work is dataset construction rather than algorithmic innovation. |
| Technical Depth | ⭐⭐⭐ | Dataset design and evaluation framework are solid, with thorough statistical significance analysis; however, as a benchmark paper, it lacks new model or method contributions. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Covers 10 SOTA models with systematic analysis across four dimensions (domain, task, quality, resolution), yielding rich findings. |
| Practical Value | ⭐⭐⭐⭐⭐ | Directly motivated by Meta Reality Labs product needs; both the dataset and evaluation framework are open-sourced, providing immediate guidance for wearable AI development. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering](are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)
- [\[ICCV 2025\] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering](../../ICCV2025/multimodal_vlm/reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)
- [\[ICLR 2026\] Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering](../../ICLR2026/multimodal_vlm/meta-adaptive_prompt_distillation_for_few-shot_visual_question_answering.md)

</div>

<!-- RELATED:END -->
