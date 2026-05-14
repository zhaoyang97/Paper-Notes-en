---
title: >-
  [Paper Note] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving
description: >-
  [ICCV 2025][Multimodal VLM][vision-language model evaluation] This paper proposes VLADBench, a fine-grained vision-language model evaluation benchmark for autonomous driving scenarios, covering 5 major domains…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "vision-language model evaluation"
  - "autonomous driving benchmark"
  - "fine-grained assessment"
  - "visual question answering"
  - "driving scene understanding"
date: 2026-05-08
content_hash: c76152942e58615b
---

# Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving

**Conference**: ICCV 2025
**arXiv**: [2503.21505](https://arxiv.org/abs/2503.21505)
**Code**: None
**Area**: Multimodal VLM / Autonomous Driving
**Keywords**: vision-language model evaluation, autonomous driving benchmark, fine-grained assessment, visual question answering, driving scene understanding

## TL;DR

This paper proposes VLADBench, a fine-grained vision-language model evaluation benchmark for autonomous driving scenarios, covering 5 major domains, 11 second-level dimensions, and 29 third-level tasks. Using a closed-ended QA format, it progressively assesses VLM capabilities from static knowledge to dynamic reasoning, and trains small-scale domain-specific (DS) models on 1.4M domain-specific QA data to validate cognitive interactions across domains.

## Background & Motivation

**Background**: As large vision-language models (VLMs) achieve breakthroughs in general visual understanding, researchers have begun applying them to autonomous driving (AD), aiming to leverage VLMs' perception and reasoning capabilities to improve driving scene understanding and decision-making.

**Limitations of Prior Work**: Existing AD VLM evaluation benchmarks primarily assess model interpretability through open-ended visual question answering. However, this evaluation paradigm is too coarse-grained—it fails to distinguish model performance across different driving sub-tasks, such as traffic sign recognition, pedestrian intention prediction, and ego-vehicle decision planning. Furthermore, open-ended answer evaluation suffers from poor consistency and difficulty in automated scoring, undermining the reliability and comparability of results.

**Key Challenge**: Complex driving scenarios require models to possess hierarchical cognitive capabilities ranging from basic element recognition to high-level reasoning, yet existing benchmarks conflate these abilities, making it impossible to precisely identify a model's capability bottlenecks.

**Goal**: To construct a hierarchical, fine-grained closed-ended evaluation benchmark that systematically assesses VLM performance across autonomous driving sub-tasks and explores the synergistic relationships among different cognitive domains.

**Key Insight**: The authors observe that driving cognition can be decomposed into five progressively deeper domains—from static traffic knowledge understanding to dynamic online reasoning and decision-making—a hierarchical structure that naturally lends itself to fine-grained evaluation.

**Core Idea**: Design VLADBench, a closed-ended QA benchmark covering 5 major domains and 29 third-level tasks, with an evaluation chain that progressively transitions from static foundational knowledge to dynamic decision reasoning, comprehensively profiling VLM capabilities in autonomous driving.

## Method

### Overall Architecture

VLADBench adopts a hierarchical evaluation framework: given driving scene images and closed-ended QA pairs as input, the framework produces fine-grained capability scores across multiple dimensions. The framework consists of two components: (1) benchmark dataset construction—5 domains, 11 second-level dimensions, and 29 third-level tasks; and (2) domain-specific model training—small-scale VLMs trained on 1.4M QA data to validate cognitive synergy across domains.

### Key Designs

1. **Hierarchical Design of Five Evaluation Domains**:

    - Function: Systematically evaluate VLM driving cognition from foundational to advanced levels.
    - Mechanism: Driving cognition is decomposed into five progressive domains: (1) Traffic Knowledge Understanding—assessing mastery of static knowledge such as traffic rules and sign meanings; (2) General Element Recognition—assessing perception of basic elements such as vehicles, pedestrians, and lane markings; (3) Traffic Graph Generation—assessing understanding of scene topological relationships; (4) Target Attribute Comprehension—assessing fine-grained recognition of traffic participant attributes and states; (5) Ego Decision-Making and Planning—assessing reasoning and decision-making capabilities in dynamic scenes.
    - Design Motivation: The progressive design from static to dynamic and from perception to reasoning enables precise identification of weaknesses along the cognitive chain.

2. **Closed-Ended QA Evaluation Format**:

    - Function: Provide a standardized, quantifiable, and automatically scorable evaluation modality.
    - Mechanism: Unlike open-ended QA, all questions are designed as multiple-choice or true/false items with distractor options. The 29 third-level tasks correspond to different question types and difficulty levels, ensuring comprehensiveness and consistency.
    - Design Motivation: The closed-ended format eliminates the subjective scoring issues inherent in open-ended responses, enabling large-scale automated evaluation while more precisely probing model understanding rather than language generation ability.

3. **Domain-Specific (DS) Model Training and Cognitive Interaction Validation**:

    - Function: Validate the cognitive synergy among the five evaluation domains.
    - Mechanism: 1.4M domain-specific QA data are collected from public sources; DS models based on small-scale VLMs are trained separately on each domain dataset; performance differences between single-domain and cross-domain training are analyzed to reveal positive transfer relationships among cognitive capabilities.
    - Design Motivation: Beyond evaluating existing models, the study empirically validates the hypothesis that "improvement in foundational capabilities promotes higher-level reasoning," providing guidance for future AD VLM training strategies.

### Loss & Training

DS models are fine-tuned using a standard vision-language model training strategy with cross-entropy loss, trained both separately and jointly on domain-specific datasets to investigate cross-domain transfer effects.

## Key Experimental Results

### Main Results

| Model | Traffic Knowledge | Element Recognition | Topology Generation | Attribute Comprehension | Decision Planning | Overall |
|-------|------------------|--------------------|--------------------|------------------------|-------------------|---------|
| GPT-4V | Strong | Moderate | Weak | Moderate | Weak | Above Average |
| InternVL2 | Moderate | Strong | Moderate | Moderate | Weak | Moderate |
| Qwen-VL | Moderate | Moderate | Weak | Moderate | Weak | Moderate |
| DS-Single | Strong | Strong | Moderate | Moderate | Moderate | Above Average |
| DS-Joint | Strong | Strong | Strong | Strong | Above Moderate | Strong |

### Ablation Study

| Training Configuration | Advanced Reasoning Accuracy | Notes |
|------------------------|----------------------------|-------|
| Decision domain only | Baseline | Single-domain training |
| + Traffic knowledge | Significant improvement | Foundational knowledge promotes reasoning |
| + Element recognition | Notable improvement | Perceptual capability transfer |
| Full joint training | Optimal | Domain synergy effect pronounced |

### Key Findings

- General-purpose VLMs (including GPT-4V) perform poorly on tasks requiring driving domain expertise, with notable deficiencies in traffic topology understanding and ego decision-making and planning.
- Domain-specific training substantially improves performance on individual sub-tasks, and cross-domain joint training outperforms single-domain training, confirming positive transfer among cognitive domains.
- Improvements in foundational cognitive capabilities (traffic knowledge, element recognition) effectively promote higher-level reasoning (decision planning), validating the rationality of hierarchical cognitive modeling.

## Highlights & Insights

- The **hierarchical evaluation design** is the most prominent contribution of this work—decomposing the ambiguous notion of "autonomous driving understanding" into a three-level 5→11→29 taxonomy. This design paradigm is transferable to any domain requiring fine-grained capability assessment (e.g., medical diagnosis, industrial inspection).
- The **choice of closed-ended QA is elegant**—it ensures objectivity and automation while probing deep model understanding through carefully designed distractors, rather than testing superficial language generation ability.
- The **cognitive interaction experiments** yield an important training strategy insight: in AD VLM training, first establishing foundational perceptual capabilities before progressively introducing higher-level reasoning tasks may be more effective than end-to-end training.

## Limitations & Future Work

- The benchmark currently relies primarily on **static image**-based QA and lacks evaluation of **video sequence** temporal understanding, whereas real-world driving requires reasoning over dynamic changes across consecutive frames.
- The **scene diversity** covered by the benchmark may be limited—long-tail scenarios such as extreme weather, nighttime driving, and rare traffic events are insufficiently represented.
- The **quality and annotation consistency** of the 1.4M QA data are not analyzed in detail; large-scale automatically generated QA may contain noise.
- Future work could extend evaluation to **3D scene understanding** and **multi-sensor fusion** to more comprehensively reflect the requirements of autonomous driving systems.

## Related Work & Insights

- **vs. DriveLM**: DriveLM focuses on conversational driving understanding using open-ended QA; VLADBench employs closed-ended QA at a finer evaluation granularity, making it more suitable for large-scale standardized assessment.
- **vs. NuScenes-QA**: NuScenes-QA focuses on 3D annotation-based scene understanding; VLADBench places greater emphasis on hierarchical visual cognition assessment from 2D images.
- **vs. DriveVLMs**: DriveVLMs focuses on VLM integration in end-to-end driving; VLADBench is more focused on capability evaluation and diagnosis—the two are complementary.
- This work provides a methodological paradigm for systematically evaluating domain-specific VLM capabilities, which can inspire the construction of fine-grained evaluation benchmarks in other vertical domains.

## Rating

- Novelty: ⭐⭐⭐⭐ The hierarchical fine-grained evaluation framework is conceptually novel, though technical innovation is relatively limited for a benchmark contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple general-purpose and domain-specific VLMs are evaluated, and the cognitive interaction experiment design is well-conceived, though some experimental details lack transparency.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, hierarchical relationships are accurately conveyed, and figures and tables are intuitively designed.
- Value: ⭐⭐⭐⭐ Provides an important evaluation tool for AD VLM research; the cognitive interaction findings have practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)
- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [\[CVPR 2026\] Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving](../../CVPR2026/multimodal_vlm/prune2drive_a_plug-and-play_framework_for_accelerating_vision-language_models_in.md)
- [\[NeurIPS 2025\] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)
- [\[CVPR 2026\] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](../../CVPR2026/multimodal_vlm/vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)

</div>

<!-- RELATED:END -->
