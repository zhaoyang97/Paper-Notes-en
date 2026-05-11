---
title: >-
  [Paper Note] MA-Bench: Towards Fine-grained Micro-Action Understanding
description: >-
  [CVPR 2026][Multimodal VLM][micro-action understanding] This paper proposes MA-Bench, a micro-action understanding benchmark comprising 1,000 videos and 12,000 structured QA pairs. It introduces a three-tier "Perception–Comprehension–Reasoning" evaluation architecture to systematically assess fine-grained micro-action understanding across 23 MLLMs, and constructs a 20.5K training corpus, MA-Bench-Train, to support model fine-tuning and improvement.
tags:
  - CVPR 2026
  - Multimodal VLM
  - micro-action understanding
  - fine-grained action recognition
  - multimodal large model evaluation
  - affective analysis
  - video question answering
date: 2026-05-08
content_hash: 626cdb8c411ad370
---

# MA-Bench: Towards Fine-grained Micro-Action Understanding

**Conference**: CVPR 2026
**arXiv**: [2603.26586](https://arxiv.org/abs/2603.26586)
**Code**: [https://MA-Bench.github.io](https://MA-Bench.github.io)
**Area**: Video Understanding / Multimodal VLM
**Keywords**: micro-action understanding, fine-grained action recognition, multimodal large model evaluation, affective analysis, video question answering

## TL;DR

This paper proposes MA-Bench, a micro-action understanding benchmark comprising 1,000 videos and 12,000 structured QA pairs. It introduces a three-tier "Perception–Comprehension–Reasoning" evaluation architecture to systematically assess fine-grained micro-action understanding across 23 MLLMs, and constructs a 20.5K training corpus, MA-Bench-Train, to support model fine-tuning and improvement.

## Background & Motivation

1. **Background**: Micro-actions are spontaneous subtle body movements triggered by emotional changes and are critical for interpersonal interaction and affective state analysis. Existing micro-action datasets such as iMiGUE, SMG, and MA-52 primarily serve traditional classification models.
2. **Limitations of Prior Work**: Although MLLMs have advanced rapidly in video understanding, micro-action understanding remains entirely unexplored — no dedicated evaluation benchmark exists. Existing video understanding benchmarks (e.g., MVBench, Video-MME) focus on everyday activities and long videos, without addressing fine-grained micro-actions.
3. **Key Challenge**: Micro-actions are extremely subtle (average duration of only 2.12 seconds, involving localized movements of fingers, head, etc.), and whether current MLLMs can capture such fine-grained motion is completely unknown.
4. **Goal**: (1) Construct a benchmark specifically designed to evaluate MLLM micro-action understanding; (2) Design a multi-tier evaluation framework spanning from perception to reasoning; (3) Provide training data to support model improvement.
5. **Key Insight**: Building upon the Micro-Action-52 dataset, the paper utilizes optical flow and skeleton information to construct motion descriptors, which are subsequently fed into an MLLM to generate structured annotations.
6. **Core Idea**: Introduce the first micro-action understanding benchmark targeting MLLMs, exposing significant deficiencies in current models' ability to capture motion granularity and body-part dynamics.

## Method

### Overall Architecture

The MA-Bench construction pipeline consists of three stages: (1) **Micro-Motion Tracker**: extracts per-body-part motion descriptors (motion vectors + coordinates) from video; (2) **Structured Micro-Action Annotation Generation**: feeds motion descriptors along with prompts into an MLLM to produce structured micro-action descriptions; (3) **Benchmark Generation**: generates three-tier Perception–Comprehension–Reasoning QA pairs from the descriptions. The final output is an evaluation set of 1,000 videos with 12,000 QA pairs and a training set of 20.5K videos.

### Key Designs

1. **Motion Descriptor Extraction (Micro-Motion Tracker)**:

    - Function: Extract precise motion information for each body part.
    - Mechanism: Fuses optical flow information with skeletal keypoint coordinates to construct motion vectors for each body region in the video (head, upper limbs, lower limbs, torso, etc.). Optical flow captures pixel-level motion magnitude and direction, while the skeleton provides structured spatial localization.
    - Design Motivation: Micro-actions involve subtle movements across multiple body parts, requiring finer part-level motion information than global motion descriptors can provide. Relying solely on MLLMs to observe raw video tends to miss such fine details.

2. **Three-Tier Evaluation Architecture (Perception–Comprehension–Reasoning)**:

    - Function: Progressively evaluate MLLM micro-action understanding from simple to complex tasks.
    - Mechanism:
        - **Perception Tier** (CMAR/FMAR): Coarse- and fine-grained action recognition, addressing "what is being done."
        - **Comprehension Tier** (SAD/MAS/MMAD/PPR): Spatial-temporal reasoning and inter-part dynamics, addressing "how it is done," using a YES/NO response format.
        - **Reasoning Tier** (MADU/MARE): Generates detailed motion descriptions and reasoning chains, addressing "why this judgment is made."
    - Design Motivation: Simple action classification alone cannot fully assess the depth of MLLM micro-action understanding; evaluation must progressively escalate from basic perception to semantic reasoning.

3. **MA-Bench-Train Training Corpus**:

    - Function: Provide large-scale micro-action understanding fine-tuning data.
    - Mechanism: Extracts 20.5K videos from 166 participants in the MA-52 dataset, paired with structured micro-action descriptions. A cross-subject design (no participant overlap between training and evaluation sets) is maintained to ensure evaluation fairness.
    - Design Motivation: Beyond identifying the problem, a solution must be provided — fine-tuning experiments validate the effectiveness of training data for improving micro-action understanding.

### Loss & Training

Closed-ended tasks (CMAR/FMAR/relational comprehension) are evaluated using accuracy. Open-ended tasks (MADU/MARE) are scored via VLM-as-a-judge (1–5 scale) across three levels: L1 (description quality), L2 (motion detail), and L3 (reasoning coherence). Fine-tuning is performed on Qwen3-VL-8B using standard instruction tuning on MA-Bench-Train.

## Key Experimental Results

### Main Results

| Model | CMAR | FMAR | SAD | MMAD | MAS | PPR | AVG |
|-------|------|------|-----|------|-----|-----|-----|
| Random | 14.7 | 20.0 | 50.0 | 50.0 | 50.0 | 50.0 | 39.05 |
| GPT-4o | 20.50 | 30.70 | 51.30 | 62.35 | 49.25 | 55.10 | 44.87 |
| Gemini-2.5-Flash | 43.00 | 31.40 | 56.55 | 60.50 | 55.50 | 57.25 | 50.70 |
| InternVideo2-Chat-8B | 22.90 | 28.10 | 57.60 | 58.95 | 55.80 | 49.00 | 45.39 |

### Ablation Study

| Configuration | Key Finding | Notes |
|---------------|-------------|-------|
| Qwen3-VL-8B (baseline) | Baseline level | Limited micro-action understanding capability |
| Qwen3-VL-8B + MA-Bench-Train | Notable gains on MARE/MADU | Structured annotation fine-tuning is effective |
| Closed-ended vs. Open-ended | Closed-ended tasks generally near random | MLLMs struggle to distinguish fine-grained action categories |
| Proprietary vs. Open-source | Gemini-2.5-Flash best (50.70%) | Proprietary models show clear advantage at the perception tier |

### Key Findings

- **MLLMs perform near random on micro-action recognition**: GPT-4o achieves only 20.50% on the CMAR task (7-class coarse classification; random baseline: 14.7%), demonstrating that current models are nearly incapable of distinguishing body-part-level motion.
- **Comprehension tier outperforms perception tier**: Models perform relatively better on YES/NO relational comprehension tasks (e.g., SAD) than on classification tasks, suggesting that models possess some local judgment capability but lack holistic classification ability.
- **Open-ended task scores are extremely low**: On the MARE reasoning task, most models score below 1/5 on L3, indicating an inability to generate coherent micro-action reasoning chains.
- **Gemini-2.5-Flash unexpectedly leads**: It achieves 43.00% on CMAR, substantially outperforming GPT-4o (20.50%), possibly benefiting from stronger temporal modeling capabilities.

## Highlights & Insights

- The **motion-descriptor-driven annotation strategy** is particularly well-conceived: rather than having MLLMs directly annotate raw video (which tends to miss subtle details), the approach first extracts precise motion information via optical flow and skeleton estimation, then feeds this structured data into an MLLM to generate natural language descriptions. This "precise detection first, then verbalization" pipeline ensures annotation quality.
- The **three-tier progressive evaluation design** is transferable to other fine-grained video understanding tasks (e.g., micro-expressions, sign language), offering a generalizable evaluation paradigm.
- The **cross-subject design** maintaining participant non-overlap between training and test sets follows standard practice in behavioral analysis research and ensures the generalizability of evaluation.

## Limitations & Future Work

- All MA-Bench videos are drawn from psychological interview settings, limiting scene diversity (predominantly seated postures) and excluding micro-actions in standing, walking, or other scenarios.
- Although 12,000 QA pairs constitute a substantial dataset, the long-tailed distribution across 52 action categories may result in insufficient evaluation of minority classes.
- Open-ended task evaluation relies on VLM-as-a-judge, and the evaluator itself may not accurately assess micro-action descriptions.
- **Future directions**: (1) Extend to diverse scenes (e.g., social interaction, classroom, interview settings); (2) Incorporate audio modalities to assist micro-action understanding; (3) Design dedicated micro-action guidance modules embedded within VLM architectures.

## Related Work & Insights

- **vs. MotionBench**: MotionBench targets general fine-grained motion understanding (5,385 videos), while MA-Bench focuses specifically on the micro-action domain (1,000 videos + 12K QA pairs) — narrower in scope but with higher data quality and more structured annotations.
- **vs. FAVOR-Bench**: FAVOR-Bench emphasizes the level of detail in action descriptions, whereas MA-Bench additionally incorporates reasoning and relational comprehension tiers, yielding a richer set of evaluation dimensions.
- **vs. Micro-Action-52**: MA-52 is a traditional classification dataset; MA-Bench elevates it into an MLLM evaluation benchmark, representing a paradigm shift within the same domain.

## Rating

- Novelty: ⭐⭐⭐⭐ First micro-action understanding benchmark targeting MLLMs; the three-tier evaluation design is innovative, though the overall construction methodology is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage across 23 models, but deeper analyses of frame sampling strategies and temporal modeling are lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-defined task formulations, and well-designed figures and tables.
- Value: ⭐⭐⭐⭐ Exposes critical capability gaps of MLLMs in fine-grained micro-action understanding, with practical implications for affective computing and human–computer interaction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Zina: Multimodal Fine-grained Hallucination Detection and Editing](zina_multimodal_fine-grained_hallucination_detection_and_editing.md)
- [\[CVPR 2026\] ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps](reasonmap_towards_fine-grained_visual_reasoning_from_transit_maps.md)
- [\[CVPR 2026\] Concept-wise Attention for Fine-grained Concept Bottleneck Models](coat_cbm_concept_wise_attention.md)
- [\[CVPR 2026\] FINER: MLLMs Hallucinate under Fine-grained Negative Queries](finer_mllms_hallucinate_under_fine-grained_negative_queries.md)
- [\[CVPR 2026\] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)

</div>

<!-- RELATED:END -->
