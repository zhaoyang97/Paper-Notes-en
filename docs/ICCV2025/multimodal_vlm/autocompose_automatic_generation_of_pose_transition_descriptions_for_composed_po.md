---
title: >-
  [Paper Note] AutoComPose: Automatic Generation of Pose Transition Descriptions for Composed Pose Retrieval Using Multimodal LLMs
description: >-
  [ICCV 2025][Multimodal VLM][Composed Pose Retrieval] This paper proposes AutoComPose, the first framework leveraging multimodal large language models (MLLMs) to automatically generate human pose transition descriptions. Through body-part-level description generation, diversification augmentation, and a cyclic consistency loss, AutoComPose achieves superior composed pose retrieval performance while eliminating the need for costly manual annotation.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Composed Pose Retrieval
  - MLLM
  - Pose Transition
  - Cyclic Consistency
  - Data Annotation
date: 2026-05-08
content_hash: 5221a9c2d8859b6a
---

# AutoComPose: Automatic Generation of Pose Transition Descriptions for Composed Pose Retrieval Using Multimodal LLMs

**Conference**: ICCV 2025
**arXiv**: [2503.22884](https://arxiv.org/abs/2503.22884)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: Composed Pose Retrieval, MLLM, Pose Transition, Cyclic Consistency, Data Annotation

## TL;DR
This paper proposes AutoComPose, the first framework leveraging multimodal large language models (MLLMs) to automatically generate human pose transition descriptions. Through body-part-level description generation, diversification augmentation, and a cyclic consistency loss, AutoComPose achieves superior composed pose retrieval performance while eliminating the need for costly manual annotation.

## Background & Motivation
Composed Pose Retrieval (CPR) enables users to search for a target pose by specifying a reference pose together with a transition description, representing a specialization of Composed Image Retrieval (CIR) for the human pose domain. The central bottleneck in CPR is **annotation difficulty**:

**High cost of manual annotation**: Pose transitions involve fine-grained motion descriptions across multiple joints; annotators may overlook subtle changes, use inconsistent wording, or introduce subjective language.

**Limitations of rule-based generation**: Methods such as PoseFix rely on predefined aggregation rules and templated sentences, constrained by fixed "paircode" descriptors (based on absolute 3D keypoint positions), resulting in limited expressiveness and generalization.

**Data scarcity**: Unlike CIR datasets such as FashionIQ and CIRR, CPR lacks large-scale annotated data, and pose variations are continuous with no texture cues.

**Mechanism**: The paper exploits MLLMs' pose understanding capabilities to automatically generate expressive, structured, and diverse pose transition descriptions, while employing cyclic consistency constraints to mitigate potential MLLM errors.

## Method

### Overall Architecture
AutoComPose consists of two main stages: (1) automatic pose transition description generation via a three-stage pipeline; and (2) retrieval model training with cyclic consistency constraints.

### Key Designs

1. **Stage 1: Body-Part-Level Description Generation**

    - **Function**: An MLLM analyzes and compares changes in individual body parts between two poses.
    - **Mechanism**: The human body is decomposed into key anatomical landmarks including the head, neck, shoulders, arms, elbows, wrists, hands, torso, hips, legs, knees, ankles, and feet. The MLLM generates concise motion descriptions for each part that undergoes change.
    - **Design Motivation**: Directly generating holistic descriptions tends to overlook subtle but critical joint motions (e.g., wrist rotation, knee flexion); part-level analysis ensures fine-grained coverage.

2. **Stage 2: Integration and Diversification**

    - **Function**: Consolidates body-part-level descriptions into natural, fluent complete sentences and generates multiple paraphrase variants.
    - **Mechanism**: The MLLM is prompted to synthesize structured part-level descriptions into coherent narratives, with encouragement to use analogical expressions for improved intuitiveness. By default, three paraphrase versions are generated per pose pair.
    - **Design Motivation**: Real user queries are natural language sentences rather than structured lists; multiple paraphrases cover linguistic variability.

3. **Stage 3: Swap and Mirror Augmentation**

    - **Function**: Automatically generates additional transition descriptions by swapping (temporal reversal) and mirroring (left–right flipping) input image pairs.
    - **Mechanism**: Three transformations—swap, mirror, and their combination—are applied to each pose pair, with the MLLM generating corresponding transition descriptions for each. By default, each pose pair ultimately yields $3 \times 4 = 12$ descriptions.
    - **Design Motivation**: Directly applying data augmentation to images in CPR requires simultaneous modification of descriptions, which traditional methods cannot achieve without additional annotation. AutoComPose automatically generates consistent descriptions.

4. **Cyclic Consistency Constraint**

    - **Function**: Constructs cyclic constraints from forward and backward transition descriptions during training to mitigate MLLM generation errors.
    - **Mechanism**: Based on the assumption that if a composed feature (reference image + forward description) is correct, it should be mappable back to the reference image feature via the backward description. The total training loss is: $L_{total} = \omega \cdot L_{bbc} + (1-\omega) \cdot L_{cycle}$, where $L_{bbc}$ is the standard batch classification loss, and $L_{cycle}$ constrains the composed feature to match the reference image after applying the reverse transition. $\omega = 0.5$.
    - **Design Motivation**: MLLMs may produce erroneous descriptions (misidentifying body parts or hallucinating non-existent motions); cyclic constraints provide a self-verification mechanism without requiring explicit error detection or correction.

### Training Details
Four CLIP-based backbones are employed (RN50, RN101, ViT-B/32, ViT-B/16), with a two-step training procedure: the text encoder is first fine-tuned for 50 epochs, followed by training of the Combiner module for 100 epochs with frozen encoders. GPT-4o is used as the MLLM.

## Key Experimental Results

### Main Results (CLIP-RN50 + Combiner)

| Dataset | Description Source | R@1 | R@5 | R@10 | R@50 |
|--------|---------|-----|-----|------|------|
| FIXMYPOSE | Manual Annotation | 3.14 | 13.14 | 20.98 | 44.12 |
| FIXMYPOSE | **AutoComPose** | **9.41** | **31.76** | **43.92** | **75.49** |
| PoseFixCPR | Manual Annotation | 67.93 | 82.32 | 86.36 | 94.11 |
| PoseFixCPR | **AutoComPose** | **81.40** | **92.68** | **94.95** | **98.15** |
| PoseFixCPR | Rule-based | 73.15 | 86.62 | 90.07 | 96.46 |
| PoseFixCPR | **AutoComPose** | **81.40** | **92.68** | **94.95** | **98.15** |

### Ablation Study (CLIP-RN50)

| Configuration | FIXMYPOSE R@1 | FIXMYPOSE R@50 | PoseFixCPR R@1 | Notes |
|------|-------------|--------------|---------------|------|
| AutoComPose (Full) | 8.24 | 63.53 | 61.36 | All components |
| (−) Cyclic | 5.88 | 55.10 | 56.48 | Remove cyclic loss |
| (−) SW & MI | 2.35 | 34.51 | 48.15 | Remove swap & mirror |
| 1 paraphrase | 5.88 | 54.31 | — | No diversification |
| 3 paraphrases (default) | 8.24 | 63.53 | — | Moderate diversification |
| 5 paraphrases | 9.02 | 64.71 | — | More diversification |

### Key Findings
- AutoComPose consistently outperforms manual annotation across **all configurations and datasets**, never falling behind.
- Swap and mirror augmentation contributes the most (removing it drops R@50 on FIXMYPOSE from 63.53 to 34.51, nearly halved).
- The cyclic consistency loss provides stable gains (+2–5 percentage points) at zero inference cost.
- Using the smaller GPT-4o mini for description generation yields slightly lower performance but still substantially outperforms manual annotation.
- The cyclic training strategy also benefits rule-based descriptions (PoseFixCPR improves from 42.00 to 55.05).

## Highlights & Insights
- This work is the first to demonstrate that automatically generated pose transition descriptions can **comprehensively surpass** manual annotation, fundamentally challenging the conventional assumption that finer annotation yields better results.
- The body-part-level intermediate representation is a key innovation—decomposing complex whole-body motion into tractable sub-problems.
- The cyclic consistency constraint cleverly exploits backward descriptions obtained automatically from swapped image pairs, requiring no additional annotation cost.
- Two new benchmarks (AIST-CPR and PoseFixCPR) address the gap in CPR evaluation standards.

## Limitations & Future Work
- Dependency on GPT-4o API calls incurs costs (though far lower than manual annotation).
- MLLMs occasionally produce responses that deviate from the prompting guidelines (approximately 2.5%); these are currently discarded without correction.
- The large gallery size and low pose diversity in FIXMYPOSE lead to high retrieval ambiguity, resulting in generally low R@1 scores.
- End-to-end training of MLLMs to directly optimize description quality remains unexplored.
- The AIST-CPR dataset lacks manual annotations for the training set, precluding direct comparison with human annotation.

## Related Work & Insights
- **vs. PoseFix rule-based method**: PoseFix relies on 3D keypoints and templated sentences, limiting expressiveness. AutoComPose's free-text generation yields improvements of 11+ percentage points in R@1.
- **vs. Manual annotation**: Manual annotation is not only costly but also inconsistent in quality (missing details, subjective language); AutoComPose outperforms it across all evaluated scenarios.
- **vs. General CIR**: CPR demands higher description precision (continuous joint motion vs. discrete attribute changes); AutoComPose's part-level analysis specifically addresses this challenge.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First application of MLLMs to automatic CPR annotation; clean and effective framework design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three datasets, four backbones, multiple comparison baselines, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, well-structured stages, and sufficient illustration of figures and tables.
- **Value**: ⭐⭐⭐⭐ Provides a scalable annotation solution and new benchmarks for CPR research, pioneering a new direction in automatic annotation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models](capellm_support-free_category-agnostic_pose_estimation_with_multimodal_large_lan.md)
- [\[ICCV 2025\] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation](multimodal_llms_as_customized_reward_models_for_text-to-image_generation.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[ICCV 2025\] Taming the Untamed: Graph-Based Knowledge Retrieval and Reasoning for MLLMs to Conquer the Unknown](taming_the_untamed_graph-based_knowledge_retrieval_and_reasoning_for_mllms_to_co.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)

</div>

<!-- RELATED:END -->
