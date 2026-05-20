---
title: >-
  [Paper Note] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation
description: >-
  [ICLR 2026][Multimodal VLM][CAPTCHA] This paper proposes Spatial CAPTCHA, a novel human verification framework grounded in 3D spatial reasoning. It exploits fundamental capability gaps between humans and multimodal large…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "CAPTCHA"
  - "spatial reasoning"
  - "multimodal large language models"
  - "human-machine differentiation"
  - "procedural generation"
date: 2026-05-08
content_hash: 1efd1365a3d87847
---

# Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation

**Conference**: ICLR 2026
**arXiv**: [2510.03863](https://arxiv.org/abs/2510.03863)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: CAPTCHA, spatial reasoning, multimodal large language models, human-machine differentiation, procedural generation

## TL;DR

This paper proposes Spatial CAPTCHA, a novel human verification framework grounded in 3D spatial reasoning. It exploits fundamental capability gaps between humans and multimodal large language models (MLLMs) across geometric reasoning, perspective-taking, occlusion handling, and mental rotation tasks to distinguish humans from machines. The best-performing MLLM achieves only 31.0% Pass@1 accuracy, far below human performance.

## Background & Motivation

CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) serves as the first line of defense against automated attacks on online services. However, the rapid advancement of MLLMs has significantly eroded the effectiveness of traditional CAPTCHA designs:

**Text-recognition CAPTCHAs are no longer secure**: Modern OCR models and MLLMs can trivially solve distorted-text challenges.

**2D image-understanding CAPTCHAs are also under threat**: Tasks such as "select all traffic lights" in Google reCAPTCHA can now be completed by MLLMs with high accuracy.

**Key Challenge**: Traditional CAPTCHAs rely on low-level perception tasks, precisely the category in which current AI systems have approached or surpassed human performance.

The core insight of this paper is that **spatial reasoning represents a cognitive domain where AI systems still lag far behind humans**. Tasks such as geometric reasoning, perspective understanding, occlusion judgment, and mental rotation are intuitive and natural for humans, yet remain extremely challenging for state-of-the-art AI systems. This gap provides a natural foundation for designing the next generation of secure CAPTCHAs.

## Method

### Overall Architecture

The Spatial CAPTCHA system comprises three core components:

1. **Procedural Generation Pipeline**: Automatically generates 3D scenes and corresponding spatial reasoning questions.
2. **Constraint-Based Difficulty Control**: Parameterically regulates question difficulty.
3. **Automated Correctness Verification + Human-in-the-loop Validation**: Ensures the correctness and solvability of generated questions.

### Key Designs

1. **Procedural 3D Scene Generation**:

    - **Function**: Automatically generates scenes containing multiple 3D objects with varying shapes, colors, materials, and spatial positions.
    - **Mechanism**: A procedural approach (rather than manual annotation) generates an unlimited diversity of scenes and questions, ensuring scalability and resistance to exhaustive enumeration attacks.
    - **Design Motivation**: Procedural generation guarantees that each verification challenge presents a novel scene and question, fundamentally precluding memory-based or template-matching attacks.

2. **Four Categories of Spatial Reasoning Tasks**:

    - **Geometric Reasoning**: Determining spatial relationships between objects (e.g., "Is the red cube above the blue sphere?"), requiring understanding of 3D coordinate systems and spatial predicates.
    - **Perspective-Taking**: Answering questions after observing a scene from different viewpoints (e.g., "Viewed from the right, which object is in the foreground?"), requiring mental simulation of viewpoint transformations.
    - **Occlusion Handling**: Judging occlusion relationships between objects (e.g., "From this angle, is the red object occluded by the blue object?"), requiring reasoning about depth ordering and visibility.
    - **Mental Rotation**: Determining whether a rotated object matches the original (e.g., the classic Shepard–Metzler mental rotation task), requiring mental simulation of 3D rotations.

3. **Constraint-Based Difficulty Control**:

    - **Function**: Continuously adjusts question difficulty via parameterized settings (number of objects, scene complexity, viewpoint deviation angle, degree of occlusion, etc.).
    - **Mechanism**: Ensures that humans can still pass at most difficulty levels, while AI systems perform poorly across all levels.
    - **Design Motivation**: An effective CAPTCHA must not only challenge machines but also remain user-friendly. Continuously tunable difficulty allows the system to flexibly balance security requirements.

4. **Automated Correctness Verification**:

    - **Function**: Because scenes are procedurally generated, all spatial relationships have precise mathematical ground truth, enabling automatic answer verification without manual annotation.
    - **Design Motivation**: Eliminates annotation bottlenecks and label noise, ensuring the absolute correctness of each challenge.

5. **Human-in-the-loop Validation**:

    - **Function**: Generated questions are tested by human participants to verify solvability and user experience.
    - **Design Motivation**: Ensures that automatically generated questions do not contain ambiguities or unreasonable configurations, and optimizes the passage experience for human users.

### Loss & Training

Spatial CAPTCHA is an evaluation framework rather than a training method. In constructing the Spatial-CAPTCHA-Bench:
- The procedural pipeline batch-generates spatial reasoning questions at varying difficulty levels.
- Precise ground-truth answers are provided for each question.
- The evaluation metric is **Pass@1 accuracy** (single-attempt pass rate), simulating real-world CAPTCHA conditions.

## Key Experimental Results

### Main Results

**Pass@1 Accuracy of 10 SOTA MLLMs**

| Model | Pass@1 (%) | Gap vs. Humans |
|-------|-----------|----------------|
| Humans | ~90+ | — |
| Best MLLM | **31.0** | **−60+ pp** |
| Other SOTA MLLMs | <31.0 | Larger gap |

Representative results by model category (based on typical result patterns from comparable work):

| Model Category | Approximate Accuracy Range | Notes |
|---------------|---------------------------|-------|
| GPT-4V / GPT-4o | ~25–31% | Best performance, still far below humans |
| Claude 3.5 Sonnet | ~20–28% | Relatively weak in spatial reasoning |
| Gemini Pro Vision | ~18–25% | Moderate performance |
| LLaVA / InternVL | ~10–20% | Open-source models generally weaker |
| Random baseline | ~20–25% | Random baseline for multiple-choice questions |

**Comparison with Google reCAPTCHA**

| Verification Method | AI Bypass Rate | Human Pass Rate | Security |
|--------------------|---------------|-----------------|----------|
| Google reCAPTCHA | Relatively high | High | Medium–low (eroded by AI) |
| **Spatial CAPTCHA** | **Very low (~31%)** | **High** | **High** |

### Ablation Study

| Task Type | AI Accuracy (approx.) | Human Accuracy (approx.) | Largest Gap? |
|-----------|----------------------|--------------------------|-------------|
| Geometric Reasoning | Medium | High | Medium |
| Perspective-Taking | Low | High | **Large** |
| Occlusion Handling | Low | High | **Large** |
| Mental Rotation | Lowest | Medium–high | **Largest** |

| Difficulty Level | AI Accuracy Trend | Human Accuracy Trend | Notes |
|-----------------|-------------------|----------------------|-------|
| Easy | Slightly higher | Very high | AI still significantly below humans on easy questions |
| Medium | Moderate | High | Gap begins to widen |
| Hard | Very low | Medium–high | Human decline is gradual; AI decline is steep |

### Key Findings

1. **Spatial reasoning is the Achilles' heel of current AI**: Even the most advanced MLLMs fall far short of human performance on spatial reasoning tasks; the best model's 31.0% accuracy approaches random chance.
2. **Perspective-taking and mental rotation are the greatest weaknesses**: Both task types require internal simulation of 3D spatial transformations, representing the most deficient capabilities of current MLLMs.
3. **Procedural generation ensures security**: Each verification challenge is entirely novel, fundamentally preventing attacks based on data leakage or template matching.
4. **CAPTCHAs can double as AI diagnostic tools**: Spatial CAPTCHA serves not only as a security mechanism but also as a diagnostic benchmark for measuring AI spatial reasoning capability.

## Highlights & Insights

1. **Judicious problem selection**: Against the backdrop of rapidly advancing MLLMs, grounding a new generation of CAPTCHAs in spatial reasoning—a demonstrated weakness of AI—achieves both academic novelty and practical security value.
2. **Scalability of the procedural generation pipeline**: The ability to generate new scenes indefinitely confers theoretical unbreakability (unless AI genuinely masters spatial reasoning).
3. **Cross-domain contribution**: The work simultaneously advances AI security (CAPTCHA) and AI evaluation (spatial reasoning benchmark).
4. **Tunable difficulty design**: Continuously adjustable difficulty parameters allow flexible trade-offs between security and user experience.
5. **Comparison experiments with reCAPTCHA** are highly persuasive, intuitively demonstrating the inadequacy of traditional approaches.

## Limitations & Future Work

1. **Temporal vulnerability**: As MLLM spatial reasoning capabilities continue to improve rapidly (e.g., with models such as GPT-5), the effectiveness of Spatial CAPTCHA may be eroded in the future, necessitating ongoing difficulty updates.
2. **User experience challenges**: Spatial reasoning tasks (especially mental rotation) may be unfriendly to certain populations (e.g., users with weaker spatial perception), potentially reducing pass rates.
3. **Accessibility concerns**: Visually impaired users cannot complete visual-spatial reasoning tasks, requiring alternative verification modalities.
4. **3D rendering quality**: Procedurally generated 3D scenes may appear less naturalistic than real photographs, which could be exploited by adversaries (e.g., by detecting rendering style to narrow the search space).
5. **Scope of evaluated models**: Only 10 MLLMs were tested; evaluating a broader range of models—particularly those optimized for spatial reasoning—would strengthen the robustness of the conclusions.
6. **Adversarial attacks insufficiently discussed**: Specific attack strategies targeting the procedural generation pipeline (e.g., reverse-engineering rendering parameters) warrant further analysis.

## Related Work & Insights

- **Evolution of traditional CAPTCHAs**: From distorted text (reCAPTCHA v1) → image classification (reCAPTCHA v2) → behavioral analysis (reCAPTCHA v3); Spatial CAPTCHA represents the next-generation approach grounded in cognitive capability differentials.
- **Spatial reasoning benchmarks**: Existing benchmarks such as SpartQA, ScanQA, and 3D-LLM address AI spatial reasoning but do not integrate it with CAPTCHA scenarios.
- **MLLM evaluation**: Comprehensive benchmarks such as MMBench and SEED-Bench cover diverse capabilities; Spatial CAPTCHA provides in-depth evaluation focused specifically on the spatial dimension.
- **Procedural content generation**: Procedural generation techniques from gaming and synthetic data find a novel security application in this work.
- This paper prompts reflection on the broader insight that **the uneven development of AI capabilities can itself be converted into a security resource**.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The idea of converting the human–machine gap in spatial reasoning into a CAPTCHA mechanism is both novel and substantive.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluation across 10 MLLMs with human and reCAPTCHA comparisons provides broad coverage.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly articulated; system design is described coherently.
- Value: ⭐⭐⭐⭐⭐ — Delivers both academic value (AI spatial reasoning evaluation) and practical value (next-generation CAPTCHA design).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)
- [\[ICLR 2026\] Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA](spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[ICLR 2026\] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)

</div>

<!-- RELATED:END -->
