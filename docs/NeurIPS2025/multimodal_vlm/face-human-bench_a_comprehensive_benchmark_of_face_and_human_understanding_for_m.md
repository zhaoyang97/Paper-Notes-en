---
title: >-
  [Paper Note] Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants
description: >-
  [NeurIPS 2025][Multimodal VLM][Multimodal Large Language Models] This paper proposes Face-Human-Bench, the first systematic benchmark to evaluate the face and human understanding capabilities of Multimodal Large Language Models (MLLMs). It features a three-level capability taxonomy (2 L1 $\times$ 10 L2 $\times$ 18 L3), contains 1,800 questions each in the development and test sets, supports both Chinese and English, evaluates 25 mainstream MLLMs…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Multimodal Large Language Models"
  - "Benchmark Evaluation"
  - "Face Understanding"
  - "Human Understanding"
  - "MLLM"
  - "Ability Taxonomy"
date: 2026-05-08
content_hash: 51cceab6eb4e5225
---

# Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants

**Conference**: NeurIPS 2025  
**arXiv**: [2501.01243](https://arxiv.org/abs/2501.01243)  
**Code**: [Project Page](https://face-human-bench.github.io/)  
**Area**: Multi-modal Benchmarks / Face & Human Understanding  
**Keywords**: Multimodal Large Language Models, Benchmark Evaluation, Face Understanding, Human Understanding, MLLM, Ability Taxonomy

## TL;DR
This paper proposes Face-Human-Bench, the first systematic benchmark to evaluate the face and human understanding capabilities of Multimodal Large Language Models (MLLMs). It features a three-level capability taxonomy (2 L1 $\times$ 10 L2 $\times$ 18 L3), contains 1,800 questions each in the development and test sets, supports both Chinese and English, evaluates 25 mainstream MLLMs, and reveals a substantial performance gap between general MLLMs and specialized domain-expert models.

## Background & Motivation

**Background**: Faces and human bodies are core elements of social interaction, widely appearing in daily photos and videos. MLLM assistants capable of deeply understanding face and human information would significantly enhance response quality and broaden application scope.

**Limitations of Prior Work**: (a) Existing multimodal benchmarks (such as MME, SEED-Bench, and MMBench) only cover a limited set of face and human capabilities (e.g., celebrity recognition, action recognition), leaving many critical capabilities unevaluated. (b) Although there are numerous domain-specific datasets in the face and human field (e.g., CelebA, LFW), they have not been systematically integrated into a unified benchmark for MLLM evaluation.

**Key Challenge**: While MLLMs perform exceptionally well on general visual tasks, how do they compare against domain-specific expert models in specialized tasks such as deepfake detection, crowd counting, and cross-pose face recognition? A systematic metric is currently lacking.

**Goal**: To build the first comprehensive benchmark specifically designed to evaluate face and human understanding in MLLMs.

## Method

### Three-Level Capability Taxonomy

**Level-1 (L1)**: Two dimensions
- Target dimension: Face understanding vs. Human understanding  
- Cognitive process: Perception vs. Reasoning

**Level-2 (L2)**: 10 capabilities
- Face: Facial attribute recognition, age estimation, facial expression recognition, face attack detection, face recognition
- Human: Human attribute recognition, action recognition, spatial relationship understanding, social relationship understanding, pedestrian re-identification

**Level-3 (L3)**: 18 fine-grained capabilities
- Expression $\rightarrow$ Basic / Compound expressions
- Attack detection $\rightarrow$ Deepfake detection / Liveness detection
- Face recognition $\rightarrow$ Basic / Cross-pose / Cross-age / Similar faces / Occlusion
- Spatial relationship $\rightarrow$ Relative position / Counting
- Social relationship $\rightarrow$ Social relationship recognition / Identity reasoning

### Data Generation Pipeline

1. **Source Data**: Collect images and annotations from 16 open-source datasets.
2. **Image Processing $p_{image}$**: Cropping, stitching, bounding box annotation, or keeping the original image.
3. **Text Processing $p_{text}$**: Convert labels into 1 correct option + $n-1$ distractors (with distractors generated using ChatGPT).
4. **Quality Control**: Manual review to ensure options are unambiguous and there is a unique correct answer.
5. **Question Format**: Multiple-choice questions $(V_i, Q_i, O_i, A_i)$ with 2-4 options.

### Evaluation Design
- Weighted accuracy, where each L2 capability is weighted equally.
- Option shuffling to prevent models from biasing towards specific letters.
- Constraint instructions to ensure models output only the option letter.
- Regex-based extraction supplemented by ChatGPT fallback for answer extraction.

### New Metric: RPSS
Relative Position Sensitivity Score (RPSS) measures the fluctuation of model performance when target objects are located at different positions within the image.

## Key Experimental Results

### Overall Performance of 25 MLLMs

| Model | Params | Face | Human | Perception | Reasoning | Total Score |
|------|--------|------|------|------|------|------|
| GPT-4o | - | 72.5 | 73.6 | ~68 | ~70 | ~70 |
| InternVL-Chat-v1.2-Plus | - | 67.0 | 70.8 | ~66 | ~68 | ~67 |
| LLaVA-NeXT-34B | - | 71.0 | 72.0 | ~65 | ~66 | ~66 |
| Gemini-1.5-Pro | - | 60.0 | 85.6 | ~60 | ~62 | ~61 |
| MiniGPT-4-7B | - | 25.0 | 29.0 | ~24 | ~34 | ~28 |
| Random | - | 25-50 | 25-50 | 29.2 | 37.5 | 32.5 |

### Key Findings

**Q1 Performance Analysis**:
- The best open-source model, InternVL-Chat-v1.2-Plus, outperforms the best closed-source model, GPT-4o, in the zero-shot setting.
- Facial attribute recognition: InternLM-XComposer2-VL-7B achieves 92.0 (highest).
- Overall face recognition: Gemini-1.5-Pro leads significantly at 85.6.
- Position sensitivity RPSS: InternLM-XComposer2-VL-7B proves to be the most stable.
- CoT prompting significantly improves GPT-4o but shows no effect on open-source models.

**Q2 Expert Models vs. MLLMs**:

| Task | Best MLLM | Expert Model | Gap |
|------|-----------|----------|------|
| Deepfake Detection | ~64% | >90% | Significant |
| Crowd Counting | ~35% | >80% | Significant |
| Face Recognition (Hard Scenes) | ~55% | >95% | Significant |

### Capability Correlation
- Statistically significant positive correlations exist among certain L2 and L3 capabilities.
- Facial attribute recognition positively correlates with expression recognition.
- Spatial relationship understanding positively correlates with social relationship understanding.

## Highlights & Insights
- **First Systematic Benchmark**: The three-level taxonomy covering 18 fine-grained capabilities fills the gap in MLLM evaluation for face and human understanding.
- **Bilingual Support**: Native support for Chinese and English facilitates cross-lingual evaluation.
- **Rich Practical Insights**: (a) Open-source models can surpass closed-source ones; (b) position sensitivity is an important yet neglected evaluation dimension; (c) CoT is effective for closed-source models but ineffective for open-source models.
- **Clear Capability Gap**: In deepfake detection, crowd counting, and hard face recognition tasks, MLLMs fall far behind specialized expert models, pointing out clear directions for future improvements.

## Limitations & Future Work
- Only static images are evaluated, without covering face and human understanding in videos.
- The multiple-choice format limits the assessment of open-ended understanding capabilities.
- Sources are public datasets, introducing potential risks of data contamination (where model training data might include parts of the test set).
- Uneven sample distribution across L3 categories, leading to limited statistical significance for certain capabilities.
- Latest models (such as GPT-4.1, Qwen2.5-VL, InternVL3) have not been evaluated.
- Social relationship annotations carry a high degree of subjectivity.

## Related Work & Insights
- **vs. MMBench/SEED-Bench**: General benchmarks contain only a few face and human-related questions; Face-Human-Bench is focused and comprehensive.
- **vs. CelebA/LFW**: Domain-specific datasets evaluate expert models; this work adapts them into formats suitable for evaluating MLLMs.
- **vs. FaceCaption-15M**: A face captioning dataset, whereas this work functions as an evaluation benchmark.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic benchmark evaluating MLLM capabilities in face and human understanding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive multidimensional analysis evaluating 25 models across 18 distinct capabilities.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with a logically sound classification system.
- Value: ⭐⭐⭐⭐ Establishes a clear roadmap for the development of MLLMs in face and human understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector](../../CVPR2025/multimodal_vlm/rethinking_vision-language_model_in_face_forensics_multi-modal_interpretable_for.md)
- [\[NeurIPS 2025\] DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding](danmakutppbench_a_multimodal_benchmark_for_temporal_point_pr.md)
- [\[NeurIPS 2025\] CAPability: A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Thoroughness](capability_a_comprehensive_visual_caption_benchmark_for_eval.md)
- [\[ICLR 2026\] UrbanFeel: A Comprehensive Benchmark for Temporal and Perceptual Understanding of City Scenes through Human Perspective](../../ICLR2026/multimodal_vlm/urbanfeela_comprehensive_benchmark_for_temporal_and_perceptual_understanding_of_.md)
- [\[ICLR 2026\] Can Vision-Language Models Answer Face to Face Questions in the Real-World?](../../ICLR2026/multimodal_vlm/can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)

</div>

<!-- RELATED:END -->
