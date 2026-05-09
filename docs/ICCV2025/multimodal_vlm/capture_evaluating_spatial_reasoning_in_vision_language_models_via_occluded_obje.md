---
title: >-
  [Paper Note] CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting
description: >-
  [ICCV 2025][Multimodal VLM][VLM evaluation] This paper introduces CAPTURe, a benchmark that evaluates spatial reasoning and world model construction in VLMs by requiring amodal counting of regularly arranged objects under occlusion. Results show that even the strongest model, GPT-4o, achieves a 14.75% counting error under occlusion, while humans perform nearly perfectly.
tags:
  - ICCV 2025
  - Multimodal VLM
  - VLM evaluation
  - spatial reasoning
  - amodal completion
  - occlusion
  - counting benchmark
date: 2026-05-08
content_hash: ec7a1bcec3c28b4e
---

# CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting

**Conference**: ICCV 2025
**arXiv**: [2504.15485](https://arxiv.org/abs/2504.15485)
**Code**: [https://github.com/atinpothiraj/CAPTURe](https://github.com/atinpothiraj/CAPTURe)
**Area**: Multimodal VLM
**Keywords**: VLM evaluation, spatial reasoning, amodal completion, occlusion, counting benchmark

## TL;DR
This paper introduces CAPTURe, a benchmark that evaluates spatial reasoning and world model construction in VLMs by requiring amodal counting of regularly arranged objects under occlusion. Results show that even the strongest model, GPT-4o, achieves a 14.75% counting error under occlusion, while humans perform nearly perfectly.

## Background & Motivation
- **Background**: VLMs have achieved remarkable progress across various visual reasoning tasks, yet whether they can understand occluded scenes and infer invisible objects as humans do remains an open question.
- **Limitations of Prior Work**: (1) Existing VLM evaluations overlook occlusion reasoning; (2) amodal completion is typically assessed via pixel-level prediction, which is incompatible with text-output VLMs; (3) objective and quantifiable metrics for occlusion reasoning are lacking.
- **Key Challenge**: The human visual system effortlessly infers and counts objects behind occluders, but whether VLMs possess analogous world model capabilities is unknown.
- **Goal**: Design an objective and quantifiable benchmark for evaluating VLM occlusion reasoning.
- **Key Insight**: Leverage regular spatial arrangements (e.g., grids, circles) so that counting under occlusion yields a uniquely determined answer, enabling counting accuracy as the evaluation metric.
- **Core Idea**: Pattern + Occlusion + Counting = measurable world model evaluation, simultaneously probing VLMs along three dimensions: pattern recognition, spatial reasoning, and counting.

## Method

### Overall Architecture
CAPTURe is an evaluation benchmark rather than a methodological contribution. It comprises two subsets: CAPTURe$^{\text{real}}$ (924 real images across 92 object categories) and CAPTURe$^{\text{synthetic}}$ (1,250 synthetic images with controlled variables). Each image contains regularly arranged objects, with a portion occluded by a black rectangle; VLMs must infer the hidden objects and report the total count.

### Key Designs
1. **CAPTURe$^{\text{real}}$ Dataset**:

    - **Function**: Provides amodal counting evaluation in real-world scenes.
    - **Mechanism**: Images with regularly arranged objects are filtered from the FSC-147 dataset via GPT-4o pre-screening followed by manual verification, yielding 924 images. Black occlusion blocks are applied manually. Both occluded and non-occluded versions are retained for comparison.
    - **Design Motivation**: Evaluate VLMs in natural scenes across 92 object types, with an average of 61.45 objects per image and 13.97 occluded objects.

2. **CAPTURe$^{\text{synthetic}}$ Dataset**:

    - **Function**: Provides fully controlled, diagnostic evaluation.
    - **Mechanism**: Synthetic images of simple shapes (dots, squares) arranged in various patterns are generated, with systematic variation in object count (5–15), arrangement shape (rectangle/circle/triangle), position (5 types), and color (5 types).
    - **Design Motivation**: Eliminate confounding factors such as background clutter and texture variation to precisely identify failure modes in VLMs.

3. **Auxiliary Information Experiments (Oracle & Prediction)**:

    - **Function**: Diagnose the sources of VLM errors by providing additional information.
    - **Mechanism**: (1) *All Object Coordinate Oracle*: provides coordinates of all objects, requiring only textual counting; (2) *Visible Object Coordinate Oracle*: provides coordinates of visible objects, still requiring inference of occluded ones; (3) *Inpainting Pipeline*: occluded regions are restored using FLUX.1-Fill before being passed to the VLM.
    - **Design Motivation**: Decouple "visual counting ability" from "world model / occlusion reasoning ability" to identify the root cause of errors.

### Evaluation Metric
- Primary metric: sMAPE (Symmetric Mean Absolute Percentage Error), ranging from 0–100%; lower is better.
- $\text{sMAPE} = 100 \cdot \frac{1}{n}\sum_{i=1}^{n}\frac{|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}$
- Responses that fail to produce an answer are assigned the maximum error of 100%.

## Key Experimental Results

### Main Results

| Model | CAPTURe$^{\text{real}}$ No Occ. | CAPTURe$^{\text{real}}$ Occ. | Δ | CAPTURe$^{\text{syn}}$ No Occ. | CAPTURe$^{\text{syn}}$ Occ. | Δ |
|------|------|------|-----|------|------|-----|
| GPT-4o | 13.34% | 14.75% | +1.41 | 5.90% | 9.71% | +3.81 |
| InternVL2 | 26.17% | 32.90% | +6.73 | 16.44% | 17.57% | +1.13 |
| Molmo | 25.90% | 32.49% | +6.59 | 8.40% | 17.73% | +9.33 |
| Qwen2VL | 18.96% | 29.33% | +10.37 | 6.63% | 11.74% | +5.11 |
| 6 VLM Avg. | 21.95% | 27.59% | +5.64 | 11.89% | 15.64% | +3.75 |
| **Human** | - | **3.79%** | - | - | **0.92%** | - |

### Ablation Study (Effect of Auxiliary Information on CAPTURe$^{\text{real}}$ Occluded Set)

| Model | Orig. Occ. | +All Coords | +Visible Coords | +Inpainted |
|------|---------|----------|----------|----------|
| GPT-4o | 14.75% | 2.93% (−11.82) | 9.20% (−5.55) | 15.89% (+1.14) |
| InternVL2 | 32.90% | 17.48% (−15.42) | 25.13% (−7.77) | 31.12% (−1.78) |
| Qwen2VL | 29.33% | 9.62% (−19.71) | 17.70% (−11.63) | 22.64% (−6.69) |
| 3 VLM Avg. | 25.66% | 10.01% (−15.65) | 17.34% (−8.32) | 23.22% (−2.44) |

### Key Findings
- All VLMs exhibit substantial counting errors under both occluded and non-occluded conditions, with occlusion consistently degrading performance.
- Humans achieve very low error under occlusion (3.79% / 0.92%), outperforming VLMs by a factor of 7–14.
- Providing all object coordinates yields a large reduction in error (avg. −15.65%), indicating that visual counting itself is a major bottleneck.
- Image inpainting offers limited benefit (avg. −2.44%), suggesting that diffusion models do not serve as perfect world models either.
- Models recognize arrangement patterns reasonably well (accuracy >80%), but accuracy drops by approximately 11% under occlusion.
- Error increases with the number of occluded objects, while total object count has comparatively little effect.
- CountGD (a detection-based model) substantially outperforms VLMs in the non-occluded setting but cannot handle occlusion.

## Highlights & Insights
- The benchmark design is elegant: combining pattern + occlusion + counting converts world model construction into an objective, quantifiable metric.
- The experimental analysis is thorough: oracle experiments precisely disentangle errors arising from "visual counting" versus "occlusion reasoning."
- A fundamental limitation of VLMs is identified: counting objects in images is challenging even in the absence of occlusion.
- A hybrid VLM+CountGD system demonstrates that feeding specialized detector outputs to a VLM can improve performance.

## Limitations & Future Work
- Only 4–6 VLMs are evaluated; more recent models (e.g., GPT-4.5, Gemini) are not covered.
- Objects in CAPTURe$^{\text{real}}$ are predominantly drawn from FSC-147, limiting data diversity.
- Answer extraction relies on Llama 3.1 8B; although verified to be 100% accurate, this adds pipeline complexity.
- Only regularly arranged objects are considered; irregular arrangements are not addressed.
- Occlusion blocks are restricted to rectangular shapes; irregular occluders are not studied.
- The benchmark focuses on diagnosing capability rather than proposing improvements.

## Related Work & Insights
- **FSC-147**: A dense counting dataset that serves as the image source for CAPTURe$^{\text{real}}$.
- **CountGD**: A state-of-the-art detection-based counting method used as a reference baseline.
- **SpartQA**: A spatial reasoning VQA benchmark, though it only tests relationships among visible objects.
- **FLUX.1-Fill**: A diffusion-based inpainting model used to provide "predicted world model" assistance.
- **Insight**: VLM evaluation should devote greater attention to the "invisible" aspects of scenes — occlusion reasoning and commonsense inference — rather than focusing solely on the processing of visible information.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First work to use amodal counting as a probe for VLM spatial reasoning and world model capabilities; a distinctly original perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic evaluation with multi-VLM comparisons, human baselines, oracle conditions, inpainting pipeline, and factor analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definition is clear; experimental analysis proceeds in a well-structured, progressive manner with rich figures and tables.
- **Value**: ⭐⭐⭐⭐ Exposes fundamental deficiencies in VLMs regarding visual counting and occlusion reasoning, providing clear directions for future improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [\[ICCV 2025\] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat.md)
- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](../../CVPR2026/multimodal_vlm/spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)

</div>

<!-- RELATED:END -->
