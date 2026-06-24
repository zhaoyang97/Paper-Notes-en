---
title: >-
  [Paper Note] BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-Language Models
description: >-
  [ECCV 2024][Hallucination Detection][Vision-Language Models] BEAF proposes a "before-after comparison" hallucination evaluation paradigm: by observing changes in VLM responses after removing objects through image editing and introducing four change-aware metrics (TU/IG/SB/ID), it reveals hallucination behaviors that cannot be detected by traditional text-axis evaluations.
tags:
  - "ECCV 2024"
  - "Hallucination Detection"
  - "Vision-Language Models"
  - "Hallucination Evaluation"
  - "Scene Manipulation"
  - "Image Editing"
  - "Benchmark"
date: 2026-05-08
content_hash: 5d39764ff0e833c3
---

# BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-Language Models

**Conference**: ECCV 2024  
**arXiv**: [2407.13442](https://arxiv.org/abs/2407.13442)  
**Code**: [https://beafbench.github.io/](https://beafbench.github.io/)  
**Area**: Hallucination Detection  
**Keywords**: Vision-Language Models, Hallucination Evaluation, Scene Manipulation, Image Editing, Benchmark

## TL;DR

BEAF proposes a "before-after comparison" hallucination evaluation paradigm: by observing changes in VLM responses after removing objects through image editing and introducing four change-aware metrics (TU/IG/SB/ID), it reveals hallucination behaviors that cannot be detected by traditional text-axis evaluations.

## Background & Motivation

1. **Background**: VLMs (such as LLaVA, InstructBLIP, etc.) exhibit strong multimodal reasoning capabilities but are prone to hallucinations—where outputs do not reflect the ground-truth content of the input images. POPE is a mainstream benchmark for hallucination evaluation, employing a discriminative question-answering format.

2. **Limitations of Prior Work**: (1) Existing benchmarks (POPE, CIEM, AMBER) only manipulate the text axis (constructing different questions) without manipulating the visual axis, failing to determine whether the VLM truly "sees" the object or merely responds based on language biases. (2) Certain objects frequently co-occur (e.g., tables and chairs); question-answering evaluations alone cannot distinguish whether the VLM truly understands or simply exploits co-occurrence biases.

3. **Key Challenge**: VLMs receive multimodal inputs (image + text), but current evaluations only consider variations along the text axis. If an object is removed and the VLM still answers "Yes", it indicates the model fails to inspect the image, a failure mode that traditional accuracy metrics cannot capture.

4. **Goal**: To design an evaluation framework that simultaneously considers variations across both visual and textual axes, conducting a fine-grained analysis of hallucinations by observing the VLM's awareness of scene changes.

5. **Key Insight**: Core assumption—if an apple is removed from an image and the model is asked "Is there an apple?", a model with true scene understanding should shift its answer from "Yes" to "No". Tracking answer changes allows distinguishing "true understanding" from various hallucination patterns.

6. **Core Idea**: Manipulate the visual scene through image editing, observe the before-and-after changes in VLM responses, and introduce change-aware metrics for fine-grained hallucination evaluation.

## Method

### Overall Architecture

The BEAF benchmark consists of two components: (1) Dataset Construction—selecting 500 original images from MS-COCO, and removing objects via SAM+LaMa to generate 1,727 manipulated images, paired with 26K image-question pairs; (2) Evaluation Metrics—designing 4 change-aware metrics (TU, IG, SB, ID) to analyze the hallucination behaviors of VLMs from different dimensions.

### Key Designs

1. **Three-Stage Image Manipulation Pipeline**:
    - **Function**: Precisely remove objects from original images to generate high-quality manipulated images.
    - **Mechanism**: Stage 1 utilizes SAM to extract masks and the LaMa inpainting model to automatically remove target objects; Stage 2 involves human filtering of low-quality results (e.g., residual shadows, inpainting failures); Stage 3 performs fine-grained manual refinement to eliminate clues like ghost shadows, artifacts, and fragmented objects.
    - **Design Motivation**: If residual clues of object removal (such as shadows) exist in the manipulated image, the VLM might infer the prior existence of the object, complicating the evaluation. The three-stage design ensures that manipulated images closely resemble natural images.

2. **Four Change-Aware Evaluation Metrics**:
    - **True Understanding (TU)**: Measures whether the model truly understands the scene—answering correctly both before and after removal. $$TU = \frac{|Filter(True, True, True)|}{|Filter(*, *, True)|} \times 100$$
    - **IGnorance (IG)**: Measures the model's lack of cognition—answering incorrectly both before and after removal. A high IG indicates the model remains completely unaware of the object.
    - **StuBbornness (SB)**: Measures the model's stubbornness—giving the same answer even after removal. It is divided into SBp (stubbornly answering Yes) and SBn (stubbornly answering No), where $$SB = 100 - TU - IG$$.
    - **InDecision (ID)**: Measures change in replies to non-relevant objects—the answer changes when it should not, indicating that the response is random.
    - **Design Motivation**: Traditional accuracy cannot distinguish between "answering correctly without comprehension" and "genuine understanding". For instance, if a model answers "Yes" both before and after removal, traditional evaluation deems the pre-removal answer correct. However, BEAF reveals through the SBp metric that this represents stubbornness rather than understanding.

3. **Dual-Axis Analysis Framework**:
    - **Function**: Perform a comprehensive analysis by combining the visual axis (scene changes) and the textual axis (question changes).
    - **Mechanism**: For each triplet of (original image, manipulated image, question), record the model's response on both images. In conjunction with whether the question is relevant to the removed object (flag R), the four metrics are computed.
    - **Design Motivation**: Evaluating along the text axis alone may overestimate model capability—some "correct" answers are actually based on co-occurrence biases rather than visual understanding.

### Loss & Training

Pure evaluation work, no training process involved.

## Key Experimental Results

### Main Results

| Model | Params | TU↑ | IG↓ | SB↓ | ID↓ | F1↑ |
|------|--------|-----|-----|-----|-----|-----|
| LLaVA | 13B | 56.4 | 8.3 | 35.3 | 11.2 | 67.1 |
| InstructBLIP | 13B | 42.1 | 3.5 | 54.4 | 8.7 | 56.8 |
| Shikra | 7B | 58.2 | 7.1 | 34.7 | 10.5 | 68.9 |
| mPLUG-Owl | 7B | 45.3 | 12.4 | 42.3 | 14.1 | 56.0 |

### Ablation Study

| Model | POPE Accuracy | BEAF TU | Difference Analysis |
|------|--------------|---------|----------|
| InstructBLIP | ~85% | 42.1% | POPE overestimates comprehension ability. |
| LLaVA | ~83% | 56.4% | The narrower gap suggests LLaVA relies more heavily on visual signals. |

### Key Findings

- **High SBp in InstructBLIP exposes critical issues**: The model tends to answer "Yes" regardless of how the scene changes, a preference undetected by traditional accuracy.
- **Position-aware training in Shikra helps reduce hallucinations**: It achieves the highest TU, likely because the position-aware strategy assists in judging object existence.
- **"Correct answers" in traditional evaluations can be hallucinations**: BEAF reveals that many answers previously considered "non-hallucinations" are actually SB (stubbornly repeating the same answer).
- **Object co-occurrence relationships affect hallucination patterns**: After removing an object, the ID values of other objects that frequently co-occur with it increase significantly.

## Highlights & Insights

- **Novelty in Evaluation Paradigm**: Shifting from "static QA" to "dynamic change awareness" represents a significant paradigm shift in VLM evaluation. Observing behavioral changes by manipulating visual inputs resembles control group designs in psychology.
- **Four-Metric System Precisely Characterizes Hallucination Types**: TU, IG, SB, and ID correspond to "True Understanding, Ignorance, Stubbornness, and InDecision" respectively, providing a finer-grained diagnostic tool than overall accuracy.
- **Reveals the "Ostriches Strategy" of VLMs**: Facing uncertainty, many VLMs do not guess randomly but stubbornly repeat a favored answer (typically "Yes"), which is clearly reflected in the SBp metric.

## Limitations & Future Work

- Small dataset scale (500 original images), which may not cover all scene categories.
- Although image editing quality was human-inspected, imperfect inpainting traces might still remain.
- Currently only evaluates discriminative QA (Yes/No), leaving generative description hallucinations uncovered.
- Only 4 VLMs were evaluated; expansion to newer models like GPT-4V is required.
- Object removal may alter the global semantic meaning of the scene (e.g., removing a person majorly impacts scene semantics), which affects the ID measurement of non-relevant objects.

## Related Work & Insights

- **vs POPE**: POPE only manipulates the textual axis (constructing positive/negative sample questions), whereas BEAF manipulates the visual axis as well, offering a more comprehensive evaluation.
- **vs AMBER**: AMBER adds generative evaluation but remains confined to text-axis manipulation, marking BEAF's visual manipulation dimension as a unique contribution.
- This "before-after comparison" evaluation concept can be transferred to video understanding hallucination evaluation (object changes across sequential frames) and 3D scene understanding evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Visual-axis manipulation + change-aware metrics, marking a breakthrough in the evaluation paradigm
- Experimental Thoroughness: ⭐⭐⭐⭐ Rigorously designed metrics, with in-depth comparative analysis across multiple models
- Writing Quality: ⭐⭐⭐⭐ Clear explanations of concepts and intuitive examples
- Value: ⭐⭐⭐⭐⭐ Significantly advances the field of VLM hallucination evaluation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](../../ACL2026/hallucination/benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[ICML 2025\] Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models](../../ICML2025/hallucination/look_twice_before_you_answer_memory-space_visual_retracing_for_hallucination_mit.md)
- [\[CVPR 2026\] SVHalluc: Benchmarking Speech-Vision Hallucination in Audio-Visual Large Language Models](../../CVPR2026/hallucination/svhalluc_benchmarking_speech-vision_hallucination_in_audio-visual_large_language.md)
- [\[ACL 2026\] Mechanisms of Prompt-Induced Hallucination in Vision–Language Models](../../ACL2026/hallucination/mechanisms_of_prompt-induced_hallucination_in_vision-language_models.md)
- [\[ACL 2025\] FIHA: Autonomous Fine-grained Hallucination Evaluation in Vision-Language Models with Davidson Scene Graphs](../../ACL2025/hallucination/fiha_autonomous_hallucination_evaluation_in_vision-language_models_with_davidson.md)

</div>

<!-- RELATED:END -->
