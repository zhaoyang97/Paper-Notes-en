---
title: >-
  [Paper Note] BLINK: Multimodal Large Language Models Can See but Not Perceive
description: >-
  [ECCV 2024][Multimodal VLM][Multimodal Evaluation Benchmark] Introduces BLINK—a multimodal evaluation benchmark containing 14 classic computer vision perception tasks (3,807 multiple-choice questions) that humans can solve "in a blink" (95.7% accuracy), but the strongest GPT-4V achieves only 51.26% (only 13.17% above random guessing), revealing a severe deficiency of current MLLMs in core visual perception capabilities.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Multimodal Evaluation Benchmark"
  - "Visual Perception"
  - "Classic CV Tasks"
  - "Visual Prompting"
  - "Perception vs. Recognition"
date: 2026-05-08
content_hash: 7d14b37cc0b4c164
---

# BLINK: Multimodal Large Language Models Can See but Not Perceive

**Conference**: ECCV 2024  
**arXiv**: [2404.12390](https://arxiv.org/abs/2404.12390)  
**Code**: [GitHub](https://github.com/zeyofu/BLINK_Benchmark)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Evaluation Benchmark, Visual Perception, Classic CV Tasks, Visual Prompting, Perception vs. Recognition

## TL;DR

Introduces BLINK—a multimodal evaluation benchmark containing 14 classic computer vision perception tasks (3,807 multiple-choice questions) that humans can solve "in a blink" (95.7% accuracy), but the strongest GPT-4V achieves only 51.26% (only 13.17% above random guessing), revealing a severe deficiency of current MLLMs in core visual perception capabilities.

## Background & Motivation

**Background**: Multimodal LLMs have made remarkable progress in high-level tasks such as VQA, image description, and visual reasoning. Existing evaluation benchmarks like MMBench and MMMU primarily focus on recognition-based visual question answering capabilities.

**Limitations of Prior Work**: Existing benchmarks mainly test "seeing" (recognition) rather than "perceiving" (deep visual understanding). The authors find that questions in many existing benchmarks can be resolved using text-only LLMs after converting images into dense descriptions (dense captions), indicating that these benchmarks do not truly test visual perception capabilities.

**Key Challenge**: There are numerous core perception tasks in classic computer vision (e.g., depth estimation, visual correspondence, multi-view reasoning) that humans can intuitively complete in an instant, yet these capabilities are difficult to "mediate" through natural language. Have these perception capabilities already "emerged" in MLLMs?

**Goal**: Systematically evaluate MLLM performance on core visual perception tasks to reveal the massive gap between MLLMs and human visual perception.

**Key Insight**: Reformat 14 classic CV tasks into a multiple-choice format, paired with single/multi-image inputs and visual prompts (such as circles, bounding boxes), to construct a benchmark that humans solve easily but MLLMs find highly challenging.

**Core Idea**: "Seeing" does not equal "perceiving"—MLLMs possess the capability to recognize objects but lack deep visual perception capabilities, such as depth understanding, spatial correspondence, and jigsaw puzzle reasoning.

## Method

### Overall Architecture

14 classic CV tasks → Elaborate multiple-choice formatting (single/multi-image + visual prompting) → 3,807 test questions → Zero-shot evaluation of various MLLMs → Comparison against human/expert models/random baselines

### Key Designs

1. **14 Perception Task System**:

    - Function: Covers perception tasks that are core to classic CV but overlooked by existing MLLM benchmarks.
    - Core Task List:
        - **Relative Depth**: Estimating which of two marked points is closer/further.
        - **Jigsaw Puzzle**: Restoring pieces to their correct positions.
        - **Multi-view Reasoning**: Judging spatial relationships across multiple viewpoints.
        - **Visual Correspondence**: Finding corresponding points across different images.
        - **Semantic Correspondence**: Semantic correspondence across object instances.
        - **Functional Correspondence**: Correspondence of functionally similar parts.
        - **Forensics Detection**: Identifying whether an image has been manipulated.
        - **Visual IQ Test**: Visual pattern reasoning.
        - **Visual Similarity**: Judging which reference is more similar to the target.
        - **Relative Reflectance**: Judging the reflectance properties of material surfaces.
        - **Object Localization**: Normalizing target object locations in a scene.
        - **Counting**: Counting the number of specific targets.
        - **Art Style**: Identifying the artistic style/genre of artworks.
        - **Spatial Relation**: Understanding the spatial layout among objects.
    - Design Motivation: These tasks require genuine visual perception capabilities that cannot be substituted by textual descriptions.

2. **Visual Prompting Design**:

    - Function: Overlays visual markers (circles, bounding boxes, masks, etc.) on images to indicate the target location.
    - Mechanism: Unlike text-only questions, BLINK heavily utilizes visual prompts to refer to specific locations or regions, forcing the model to understand the question through visual inspection.
    - Design Motivation: Visual prompting is one of the key features distinguishing BLINK from other benchmarks. Experiments show that the color and size of visual prompts significantly affect MLLM performance (red outperforms gray; a 10px circle is optimal), indicating that MLLMs have limited capability in interpreting visual prompts.

3. **Dense Caption Comparative Experiment**:

    - Function: Validates whether localizing images into detailed text descriptions combined with a text-only LLM can solve BLINK.
    - Mechanism: Generates task-agnostic detailed descriptions for each image using GPT-4V, and then uses a text-only LLM to answer.
    - Key Conclusion: Caption+LLM performs reasonably well on MMBench and MMMU but fails severely on BLINK. This proves that BLINK indeed requires visual perception capabilities beyond text descriptions.
    - Design Motivation: Indirectly proves the irreplaceability of BLINK—it tests perception capabilities that language cannot mediate.

### Loss & Training

BLINK is an evaluation benchmark and does not involve model training. All evaluations are conducted in a zero-shot setting using the standardized prompts provided with the dataset.

## Key Experimental Results

### Main Results (Validation Set Accuracy %)

| Model | Overall Mean | Depth Estimation | Jigsaw | Multi-view | Visual Corr. | Forensics | IQ Test | Visual Sim. | Reflectance | Obj. Local. |
|------|---------|---------|------|--------|---------|---------|--------|---------|--------|---------|
| Human | **95.67** | 96.70 | 93.75 | 99.19 | 99.00 | 95.30 | 80.77 | 96.07 | 98.25 | 98.00 |
| GPT-4o | 60.04 | 72.59 | 49.17 | 74.19 | 55.33 | 82.91 | 40.77 | 53.96 | 69.23 | 59.84 |
| GPT-4V | 51.14 | 78.52 | 60.83 | 59.68 | 70.00 | 79.49 | 26.15 | 28.78 | 72.73 | 54.92 |
| Gemini Pro | 45.16 | 52.59 | 52.50 | 40.32 | 57.33 | 50.43 | 24.62 | 26.62 | 74.83 | 53.28 |
| LLaVA-v1.6-34B | 46.80 | 48.89 | 66.67 | 67.74 | 54.67 | 43.59 | 20.77 | 23.74 | 74.83 | 59.02 |
| Random Guess | 38.09 | 50 | 25 | 50 | 50 | 50 | 25 | 25 | 50 | 50 |

### Expert Models vs. MLLMs

| Configuration | Key Conclusion | Explanation |
|------|---------|------|
| Expert CV Models vs. GPT-4V | Expert models outperform by 18%-57% | Indicates that professional visual capabilities are learnable, and MLLMs have immense room for improvement |
| Caption+LLM vs. Direct MLLM (BLINK) | Caption+LLM performs poorly | BLINK requires visual perception beyond text |
| Caption+LLM vs. Direct MLLM (MMBench) | Caption+LLM performs well | Indicates that MMBench information can be captured by text |
| Red Circles vs. Gray Circles | Red is generally better | The design of visual prompts has a significant impact |
| 10px vs. Other Circle Sizes | 10px is optimal on average | The optimal size varies by task |

### Key Findings

- **Astonishing Human-AI Gap**: Human 95.7% vs GPT-4V 51.3%, a gap of nearly 45 percentage points. GPT-4V is only 13% above random guessing.
- **MLLMs Perform Worse than Random Guessing on Certain Tasks**: On tasks such as jigsaw puzzle, semantic correspondence, multi-view reasoning, object localization, and relative reflectance, some MLLMs perform even below the random baseline.
- **7B/13B Open-Source Models Offer Performance Similar to Random Guessing**: Yielding a mean of 35-42%, showing no significant difference from random guessing (38.09%).
- **"Seeing" is Not "Perceiving"**: MLLMs can recognize what objects are in an image but cannot comprehend deeper visual properties such as depth, correspondence, and spatial layout.
- **Expert Models Far Outperform General MLLMs**: Outperforming MLLMs by 18-57% on the same tasks, indicating that these perception capabilities are learnable in principle but not covered by current MLLM training paradigms.
- **Textual Descriptions Cannot Substitute Visual Perception**: The Caption+LLM method fails on BLINK but succeeds on MMBench/MMMU, proving that BLINK indeed evaluates a different level of capability.

## Highlights & Insights

- **Extremely Precise Positioning of "See but Not Perceive"**: Clearly distinguishes between recognition (recognition) and perception (perception), revealing the true bottleneck of MLLMs.
- **14 Tasks Sourced from Classic CV Domains**: Brings core problems studied in traditional CV for decades into MLLM evaluation, bridging traditional CV and the large model era.
- **Dense Caption Experiment as a Killer Argument**: Indirectly proves that almost all prior benchmarks can be "cheated" using textual descriptions, whereas BLINK cannot.
- **Inspiring Study on Visual Prompting**: The impact of prompt attributes such as color and size exposes the fragility of MLLM visual understanding.
- **Comparison with Expert Models Points the Way**: Indicates that perception capabilities are learnable, with the core bottleneck being data and training strategies.

## Limitations & Future Work

- Uses only a multiple-choice format for evaluation, which may not fully reflect open-ended perceptual reasoning capabilities.
- Although the 14 tasks cover a wide scope, the amount of data per task is limited (3,807 questions in total).
- Lacks analysis on training data coverage—the poor performance of MLLMs might simply stem from the lack of such samples in the training data.
- Methods to incorporate these perception tasks into the training pipeline of MLLMs to enhance their capabilities can be explored.
- The optimal design of visual prompts requires further automated search.
- Does not cover more complex spatio-temporal perception tasks, such as video perception and 3D scene understanding.

## Related Work & Insights

- **vs MMBench/MMMU**: Most information in these benchmarks can be captured by text descriptions, essentially testing text combined with shallow visual understanding; BLINK evaluates pure visual perception that cannot be mediated through language.
- **vs MathVerse**: MathVerse reveals that MLLMs fail to "understand" mathematical diagrams; BLINK more broadly demonstrates that MLLMs fail to "perceive" fundamental visual attributes (depth, correspondence, spatial relations, etc.).
- **vs Traditional CV Evaluation**: Traditional CV evaluations target the performance of expert models on single tasks; BLINK reformats multiple tasks uniformly to evaluate general-purpose MLLMs, establishing a unique cross-evaluation perspective.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The perspective of introducing classic CV perception tasks into MLLM evaluation is highly novel, and the concept of "See but Not Perceive" is profound.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 20+ models and 14 tasks, but the data volume per task is relatively small, and the expert model comparison only covers a subset of tasks.
- Writing Quality: ⭐⭐⭐⭐ The arguments are clear and compelling, and the benchmark design is fully articulated, though some analyses could be deeper.
- Value: ⭐⭐⭐⭐⭐ Identifies a core blind spot of MLLMs, offering important guidance for the future direction of MLLM development, and has pushed the community to focus on perception capability training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VisionZip: Longer is Better but Not Necessary in Vision Language Models](../../CVPR2025/multimodal_vlm/visionzip_longer_is_better_but_not_necessary_in_vision_language_models.md)
- [\[ICCV 2025\] Vision-Language Models Can't See the Obvious](../../ICCV2025/multimodal_vlm/vision-language_models_cant_see_the_obvious.md)
- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_visionlanguage_models.md)
- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](../../CVPR2026/multimodal_vlm/aif_adaptive_information_flow_vlm.md)
- [\[ACL 2026\] "I See What You Did There": Can Large Vision-Language Models Understand Multimodal Puns?](../../ACL2026/multimodal_vlm/i_see_what_you_did_there_can_large_vision-language_models_understand_multimodal_.md)

</div>

<!-- RELATED:END -->
