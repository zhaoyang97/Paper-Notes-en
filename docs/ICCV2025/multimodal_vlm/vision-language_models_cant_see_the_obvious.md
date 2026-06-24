---
title: >-
  [Paper Note] Vision-Language Models Can't See the Obvious
description: >-
  [ICCV 2025][Multimodal VLM][Visual Saliency] Introduces SalBench, a benchmark designed to evaluate Large Vision-Language Models (LVLMs) on detecting visual saliency features that are obvious to humans (e.g., differences in color, orientation, size). The findings reveal a critical gap between LVLMs and human visual attention, where even the state-of-the-art GPT-4o achieves only 47.6% accuracy on the detection task.
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Visual Saliency"
  - "LVLM Evaluation"
  - "Attention Mechanism"
  - "Low-level Feature Perception"
  - "Benchmark"
date: 2026-05-08
content_hash: 00fd18061c3a8836
---

# Vision-Language Models Can't See the Obvious

**Conference**: ICCV 2025  
**arXiv**: [2507.04741](https://arxiv.org/abs/2507.04741)  
**Code**: [SalBench](https://salbench.github.io)  
**Area**: Multimodal VLM  
**Keywords**: Visual Saliency, LVLM Evaluation, Attention Mechanism, Low-level Feature Perception, Benchmark

## TL;DR

Introduces SalBench, a benchmark designed to evaluate Large Vision-Language Models (LVLMs) on detecting visual saliency features that are obvious to humans (e.g., differences in color, orientation, size). The findings reveal a critical gap between LVLMs and human visual attention, where even the state-of-the-art GPT-4o achieves only 47.6% accuracy on the detection task.

## Background & Motivation

While current LVLMs excel at high-level semantic understanding tasks (e.g., VQA, MMMU), a key issue has been overlooked:

**Moravec's Paradox**: AI systems perform exceptionally well on high-level reasoning but may struggle with low-level perception tasks that are effortless for humans. For instance, identifying an obviously large circle among a set of small circles, or finding an oddly colored object among a row of objects of the same color.

**Limitations of Prior Work**: Benchmarks like MMBench, MMMU, and MathVista test high-level, complex tasks, but there is no benchmark to systematically evaluate the low-level visual perception capabilities of LVLMs (basic visual features such as color, orientation, and size).

**Alignment with Human Visual Attention Mechanisms**: Human visual search relies on Feature Integration Theory (FIT), where the brain processes regions that differ significantly along a certain feature dimension in parallel. Do LVLMs possess this capability?

The authors aim to quantify the gap in low-level visual perception between LVLMs and humans through a simple yet meticulously designed benchmark, thereby pointing out directions for improvement.

## Method

### Overall Architecture

SalBench is constructed based on the P3 (synthetic images) and O3 (natural images) datasets, containing scenes where a single salient target stands out among numerous distractors in the stimulus image. Three tasks are defined on these images to evaluate the perception capabilities of LVLMs.

### Key Designs

1. **Odd-One-Out Detection**

   Given an image containing multiple similar objects and one distinct, salient object, the model is required to predict along which dimension(s) the target differs from a predefined list of feature categories. Synthetic images are restricted to 3 categories (color, shape, orientation), while natural images are expanded to 7 categories (orientation, color, focus, shape, size, location, pattern).

   **Design Motivation**: To directly evaluate the model's ability to "see" salient targets. Experiments show that direct inquiries often lead to incorrect answers, indicating that the model genuinely lacks this basic perceptual ability.

2. **Referring Odd-One-Out**

   Building upon the detection task, this task additionally provides the bounding box coordinates of the target as a text prompt (e.g., `"(x_min, y_min, x_max, y_max)"`). The model must determine which features of the object inside that region differ from the remaining objects.

   **Design Motivation**: To eliminate the difficulty of "localization" and purely test the model's capability to recognize feature differences once the target location is known. Even when location information is provided, the models still perform poorly.

3. **Visual Referring Odd-One-Out**

   This task modifies the reference by visually marking the target object (with a red bounding box) instead of providing textual coordinates. The model must focus on the highlighted region using visual attention and identify the differences.

   **Design Motivation**: To test the model's ability to integrate highlighted visual information, which closely mimics the natural human interaction of identifying "marked objects."

### Loss & Training

SalBench itself is an evaluation benchmark rather than a training method. However, the authors conducted additional **training experiments** to verify whether performance could be improved through training:

- Generated 1 million synthetic saliency image-text pairs for the alignment phase.
- Generated 1 million instruction tuning data points.
- Combined with 2 million natural image data points from the Cambrian dataset.
- Utilized the LLaVA training pipeline to test 4 model variants (LLama3.1-8B/Qwen2-7B $\times$ CLIP/SigLip).

**Key Findings**: Even when trained on in-domain saliency data, the performance remains extremely low (16-19% in detection), suggesting that current architectures and training paradigms are inherently unsuited for capturing saliency information.

## Key Experimental Results

### Main Results

**SalBench Zero-shot F1 Scores (Detection/Referring/Visual Referring, Natural/Synthetic Images)**:

| Model | Detection NAT | Detection SYN | Referring NAT | Visual Ref. SYN |
|------|--------------|--------------|--------------|----------------|
| GPT-4o | 47.6 | 89.2 | 47.3 | 73.5 |
| Claude-sonnet | 48.2 | 86.7 | 51.1 | 87.7 |
| Qwen2-VL-72B | 41.6 | 88.8 | 44.6 | 74.7 |
| Qwen2-VL-7B | 32.5 | 55.7 | 32.5 | 57.4 |
| LLaVA 1.6-7B | 24.5 | 16.3 | 21.4 | 16.6 |
| InternVL-2-8B | 20.0 | 58.7 | 23.0 | 23.0 |

The performance of all models on natural images is far lower than on synthetic images (a gap of 30-40%).

### Ablation Study

**Accuracy on Synthetic Images by Difficulty Level (Qwen2-VL-72B / GPT-4o)**:

| Category | Difficulty | Qwen2-VL-72B Detection | GPT-4o Detection |
|------|------|----------------------|----------------|
| Orientation | Easy | 98.6 | 96.2 |
| Orientation | Hard | 95.7 | 98.6 |
| Size | Easy | 94.2 | 93.3 |
| Size | Hard | 46.0 | 36.8 |
| Color | Easy | 100.0 | 99.8 |
| Color | Hard | 60.1 | 66.1 |

**Key Findings**: Orientation recognition is relatively robust across difficulty levels, but performance on size and color drops sharply at high difficulty levels (where differences are subtle).

**Visual Backbone Retrieval Test**:

| Vision Encoder | SYN Top-1 | NAT Top-1 |
|-----------|-----------|-----------|
| SigLip-so400m | 55.3 | 87.9 |
| CLIP-ViT-Large-Patch14 | 41.2 | 78.6 |
| Random | 24.6 | 53.2 |

Feature representations of vision encoders themselves are not discriminative enough for saliency information.

### Key Findings

1. **Model scale matters**: For Qwen2-VL, from 1.5B $\rightarrow$ 7B $\rightarrow$ 72B, F1 scores improve significantly (from 23.8 $\rightarrow$ 54.9 $\rightarrow$ 89.9 on synthetic detection), but even the largest model scores only ~44% on natural images.
2. **Few-shot provides no consistent improvement**: Increasing the number of shots does not always improve performance and sometimes degrades it (GPT-4o decreases from 47.6% in 0-shot to 38.9% in 3-shot).
3. **Preference for color**: All models recognize the color category much better than other features, because color is directly provided by RGB image information, whereas size and shape require higher-level encoding.
4. **Impact of distractor count**: As the number of distractors increases (from $<7$ to $>25$), the average F1 drops from 44.5% to 37.4%.
5. **Training does not solve the issue**: Even after training on in-domain data, performance remains very low, implying that the issue might stem from the architectural level of the vision encoder.

## Highlights & Insights

- **Strong empirical evidence of Moravec's Paradox**: The first work to systematically demonstrate the failure of LVLMs in "simple" perceptual tasks.
- **Clever task design**: Three progressive tasks (no prompt $\rightarrow$ text location $\rightarrow$ visual marker) gradually reduce localization difficulty while still exposing the perceptual deficiencies of the models.
- **Deep root-cause analysis**: Instead of merely testing the models as black-boxes, the paper separately tests the LLM's FIT knowledge (GPT-4o knows the FIT theory at 97.5%) and the visual encoder's retrieval capability, locating the root cause in the visual representation side.
- **Practical implications**: For application scenarios requiring low-level visual judgment (e.g., industrial quality inspection, medical image anomaly detection), this finding serves as an important warning.

## Limitations & Future Work

- SalBench mainly focuses on the "odd-one-out" type of saliency tasks, without covering other types of low-level perception (e.g., texture, depth).
- Synthetic images are arranged in a $7 \times 7$ grid, which may deviate from the complexity of real-world scenes.
- Multi-label classification evaluation metrics (exact match and F1) might underestimate partially correct recognitions.
- Training experiments only utilized the LLaVA framework, leaving other architectures possibly more suitable for low-level feature learning unexplored.
- No concrete mitigation strategy or new vision encoder design was proposed.

## Related Work & Insights

This paper is related to vision-centric benchmarks like MMVP (CLIP-blind pairs), RealWorldQA, and CV-Bench, but uniquely focuses on well-defined lower-level saliency concepts from neuroscience. Inspiring directions: (1) It may be necessary to introduce multi-scale features or saliency priors into vision encoders; (2) Feature Integration Theory can guide the design of better vision backbones.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first benchmark to systematically evaluate the low-level perception of LVLMs, yielding highly significant findings.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated 15+ models across three tasks in various settings (zero/few-shot), backbone analyses, and training experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Clear and thorough analysis with rich visualizations and tables.
- **Value**: ⭐⭐⭐⭐⭐ Uncovers a fundamental blind spot of LVLMs, offering crucial guidance for future model design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](../../ACL2025/multimodal_vlm/can_vision_language_models_understand_mimed_actions.md)
- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](../../ACL2025/multimodal_vlm/cant_see_the_forest_for_the.md)
- [\[ECCV 2024\] BLINK: Multimodal Large Language Models Can See but Not Perceive](../../ECCV2024/multimodal_vlm/blink_multimodal_large_language_models_can_see_but_not_perceive.md)
- [\[ACL 2026\] "I See What You Did There": Can Large Vision-Language Models Understand Multimodal Puns?](../../ACL2026/multimodal_vlm/i_see_what_you_did_there_can_large_vision-language_models_understand_multimodal_.md)
- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](../../ACL2025/multimodal_vlm/negvqa_can_vision_language_models_understand_negation.md)

</div>

<!-- RELATED:END -->
