---
title: >-
  [Paper Note] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models
description: >-
  [AAAI 2026][Multimodal VLM][MLLM safety] This paper proposes SDEval, the first safety dynamic evaluation framework for MLLMs. By applying text dynamics (6 strategies), image dynamics (2 categories), and cross-modal dynamics (4 strategies), SDEval generates variant samples of controllable complexity from existing safety benchmarks. On MLLMGuard and VLSBench, it reduces the safety rate of InternVL-3-78B by nearly 10%, effectively mitigating data leakage and exposing model safety vulnerabilities.
tags:
  - AAAI 2026
  - Multimodal VLM
  - MLLM safety
  - dynamic evaluation
  - data leakage
  - jailbreak attacks
  - safety benchmark
date: 2026-05-08
content_hash: 84fdc8592e3b8220
---

# SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2508.06142](https://arxiv.org/abs/2508.06142)
**Code**: [SDEval](https://github.com/hq-King/SDEval)
**Area**: Multimodal VLM
**Keywords**: MLLM safety, dynamic evaluation, data leakage, jailbreak attacks, safety benchmark

## TL;DR

This paper proposes SDEval, the first safety dynamic evaluation framework for MLLMs. By applying text dynamics (6 strategies), image dynamics (2 categories), and cross-modal dynamics (4 strategies), SDEval generates variant samples of controllable complexity from existing safety benchmarks. On MLLMGuard and VLSBench, it reduces the safety rate of InternVL-3-78B by nearly 10%, effectively mitigating data leakage and exposing model safety vulnerabilities.

## Background & Motivation

**Background**: MLLMs have achieved remarkable progress in multimodal understanding, yet they also pose risks of generating harmful content. The community has established multiple safety evaluation benchmarks (MLLMGuard, VLSBench, MMSafetyBench, etc.) to assess models' resistance to harmful outputs.

**Limitations of Prior Work**:
(1) **Severe data leakage** — most safety benchmarks are aggregated from open-source datasets and are likely included in MLLM training corpora, leading to inflated evaluation results;
(2) **Fixed complexity of static datasets** — manually constructed benchmarks cannot keep pace with the rapid advancement of MLLMs, making it difficult to precisely assess model performance ceilings;
(3) **Continuously evolving attack methods** — novel jailbreak attacks emerge persistently, and fixed benchmarks cannot timely cover emerging risks.

**Key Challenge**: Existing dynamic evaluation methods (e.g., DyVal) target capability assessment only and are ill-suited for open-ended safety evaluation scenarios, while also neglecting the capability–safety trade-off.

**Key Insight**: Design a general and flexible safety dynamic evaluation framework that can generate unlimited variant samples with adjustable complexity and reduced data contamination from any original benchmark.

## Method

### Overall Architecture

SDEval takes original safety benchmark samples $P=(T, I)$ as input and generates new text-image pairs $P'=(T', I')$ via a dynamic strategy set $\mathfrak{D}$. The framework operates along three dimensions: text dynamics, image dynamics, and cross-modal dynamics. Generated samples are verified by a validator agent for semantic consistency, and a scorer then judges the harmfulness of model responses.

### Key Designs

1. **Text Dynamic Strategies (6 types)**

    - **Function**: Drawing from human strategies for bypassing content moderation, these strategies modify text without altering semantics to increase the difficulty of safety recognition.
    - **Core strategies**:
        - Word substitution (synonym/contextually approximate word replacement of ≤5 words)
        - Sentence paraphrasing (preserving core concepts while varying sentence structure)
        - Description insertion (adding relevant/irrelevant descriptions to distract model attention)
        - Spelling error injection (repeated letters, special characters, and similar readable distortions)
        - Multilingual mixing (multilingual reconstruction using Chinese, English, Russian, French, Japanese, and Korean)
        - Chain-of-thought injection (appending "answer step by step" instructions)
    - **Design Motivation**: Simulate real-world user behavior of circumventing safety filters through linguistic variation.

2. **Image Dynamic Strategies (2 categories)**

    - **Function**: Modify images through basic augmentation and generative operations to reduce data leakage and test models' visual safety recognition capability.
    - **Basic augmentation**: Spatial transformations (random padding 10%–20% + flipping) and color transformations (color inversion + salt-and-pepper noise).
    - **Generation and manipulation**: Caption-guided regeneration using SD3.5-Large; ICEdit-based object insertion, text insertion, and style transfer.
    - **Quality assurance**: GPT-4o verifies semantic consistency between generated and original images.
    - **Design Motivation**: Generatively produced images exhibit large visual divergence from original samples, effectively reducing data leakage rates.

3. **Cross-Modal Dynamic Strategies (4 types)**

    - **Function**: Explore the impact of text–image interaction on safety.
    - **Text-to-Image**: Injects text dynamic variants into image generation (sample text perturbation → generate caption → SD generates new image).
    - **Image-to-Text**: Injects image dynamic variants into text (sample image perturbation → GPT-4o generates safety-relevant caption → prepend to original text).
    - **FigStep jailbreak**: Converts text prompts into typographic images as direct input (bypasses text-based safety alignment).
    - **HADES jailbreak**: Transfers unsafe keywords from text into images.

### Evaluation Protocol

On MLLMGuard, two metrics are used: ASD (Attack Success Degree, ↓) and PAR (Perfect Answer Rate, ↑). On VLSBench, the safety rate SR (proportion of safe refusals + safe warnings, ↑) is used.

## Key Experimental Results

### Main Results — MLLMGuard Dynamic Evaluation

| Model | ASD↓ (Dynamic) | ASD (Original) | PAR↑ (Dynamic) | PAR (Original) |
|-------|----------------|----------------|----------------|----------------|
| GPT-4o | 32.78 | 29.22 | 24.71 | 40.38 |
| Claude-4-Sonnet | 25.42 | 23.49 | 51.89 | 56.37 |
| InternVL-3-78B | 39.34 | 30.04 | 21.40 | 39.04 |
| Qwen-VL-2.5-7B | 40.17 | 29.46 | 33.96 | 44.04 |

### Ablation Study — Effects of Individual Dynamic Strategies (InternVL-Chat-V1.5)

| Strategy | ASD↓ | ΔASD | PAR↑ | ΔPAR |
|----------|------|------|------|------|
| Original | 32.21 | - | 40.19 | - |
| Word Substitution | 38.71 | +6.30 | 26.94 | -13.25 |
| FigStep | 41.96 | +9.55 | 17.08 | -23.11 |
| Object Insertion | 39.41 | +7.00 | 26.45 | -13.74 |
| Text-to-Image | 35.10 | +2.69 | 24.36 | -15.83 |

### Capability Evaluation Impact (SDEval's Effect on Capability Benchmarks)

| Model | MMVet (Original→Dynamic) | MMBench (Original→Dynamic) |
|-------|--------------------------|---------------------------|
| GPT-4o | 68.8→67.5 (−1.3) | 83.4→81.8 (−1.6) |
| Qwen2.5VL-7B | 67.1→63.9 (−3.2) | 83.5→79.3 (−4.2) |

### Key Findings

- FigStep is the most effective single strategy, raising ASD by ~10% and lowering PAR by over 23%, indicating that the visual embedding space is not aligned with the LLM's safety mechanisms.
- All MLLMs exhibit significant safety degradation under dynamic evaluation, suggesting that models largely "memorize" safe answers rather than genuinely understanding unsafe factors.
- Safety performance shows no clear positive correlation with model scale — larger models may follow harmful instructions more readily due to better instruction comprehension.
- SDEval has minimal impact on capability benchmarks (1–4 point drop), revealing that safety is more fragile than capability.

## Highlights & Insights

1. **General and flexible framework design**: The three-dimensional dynamic strategies can be combined and applied to any safety benchmark, co-evolving alongside the benchmarks themselves.
2. **Effective mitigation of data leakage**: Generatively produced image and text variants have minimal overlap with training corpora.
3. **Reveals the safety–capability imbalance**: Safety evaluations fluctuate far more than capability evaluations under dynamic perturbation, suggesting insufficient depth of model safety alignment.
4. **Surprising effectiveness of FigStep**: Typographic attacks directly bypass text-based safety guards, exposing a fundamental flaw in vision–language safety alignment.

## Limitations & Future Work

1. Dynamic strategies rely on external models such as GPT-4o and SD3.5, incurring high costs and introducing additional biases.
2. Semantic consistency verification depends on GPT-4o judgments, which may result in missed detections.
3. The interaction effects of different strategy combinations are not thoroughly analyzed.
4. Evaluation covers only two safety benchmarks; generalizability to broader scenarios (e.g., toxicity detection, bias detection benchmarks) remains unverified.

## Related Work & Insights

- The dynamic evaluation paradigm can be extended to other AI safety domains (e.g., LLM alignment evaluation, code security evaluation).
- The effectiveness of FigStep and HADES attacks highlights that vision–language safety alignment is a severely overlooked security blind spot.
- The observed safety–capability imbalance provides empirical support for the AI 45° law.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐: First safety dynamic evaluation framework with a systematic three-dimensional strategy design.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Covers 17 MLLMs, 2 safety benchmarks, and 2 capability benchmarks, with detailed ablation studies.
- **Writing Quality** ⭐⭐⭐⭐: Clear motivation and an intuitive architectural diagram.
- **Value** ⭐⭐⭐⭐: Provides a sustainable and evolving methodology for MLLM safety evaluation with practical guidance for the community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](../../CVPR2026/multimodal_vlm/evolving_contextual_safety_in_multi-modal_large_language_models_via_inference-ti.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)
- [\[NeurIPS 2025\] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](../../NeurIPS2025/multimodal_vlm/video-safetybench_a_benchmark_for_safety_evaluation_of_video_lvlms.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/multimodal_vlm/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)
- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)

<!-- RELATED:END -->
